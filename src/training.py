"""End-to-end training orchestration (S6 split, S7 models, S3 plots).

``train_all`` is the function behind ``python -m src.cli train``. It:

1. loads the latest raw scrapes, validates and merges them (source weights),
2. filters to usable iPhone rows and writes the processed training table,
3. makes an **out-of-time** chronological split (the only honest split for
   price data — a random split would leak the future into the past),
4. fits the single ``FeaturePipeline`` on train only,
5. trains Ridge (baseline), XGBoost (main, optional grid search), LightGBM
   (benchmark) and the quantile interval pair,
6. evaluates everything on the held-out test set, renders plots, and
7. serializes the artifacts the inference detector loads.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data import latest_parquet, load_parquet, merge_sources, save_parquet, validate_listings
from .features.parser import is_accessory_listing, parse_listing
from .features.pipeline import FeaturePipeline
from .logging_setup import get_logger
from .models import (
    LightGBMBenchmark,
    QuantileIntervalModel,
    RidgeBaseline,
    XGBoostRegressorModel,
    interval_coverage,
    regression_metrics,
)
from .reporting import viz

log = get_logger(__name__)

PRICE_MIN, PRICE_MAX = 50.0, 2500.0


# --- data preparation ----------------------------------------------------
def prepare_training_frame(cfg: Config) -> pd.DataFrame:
    """Load → validate → merge → filter to usable iPhone rows; persist."""
    frames: dict[str, pd.DataFrame] = {}
    for source, prefix in (
        ("wallapop", "wallapop_iphones"),
        ("backmarket", "backmarket_reference"),
        ("ebay", "ebay_sold"),
        ("ebay_archive", "ebay_archive"),
        ("ecommerce_us", "ecommerce_us"),
    ):
        path = latest_parquet(cfg.paths.raw_dir, prefix)
        if path is None:
            log.info("no raw data for source '%s' — skipping", source)
            continue
        df = validate_listings(load_parquet(path), source=source)
        frames[source] = df

    if not frames:
        raise FileNotFoundError(
            f"No raw scrapes found in {cfg.paths.raw_dir}. Run `scrape` first."
        )

    # Optional restriction of the training corpus to specific sources.
    training_sources = cfg.raw.get("training_sources")
    if training_sources:
        frames = {s: f for s, f in frames.items() if s in training_sources}
        if not frames:
            raise FileNotFoundError(
                f"no configured training sources found in {cfg.paths.raw_dir}; "
                f"expected one of {training_sources}"
            )
        log.info("training on sources: %s", list(frames))

    merged = merge_sources(frames, cfg.source_weights)

    # filter to identifiable iPhones in a sane price band (drop cases/accessories)
    specs = pd.DataFrame(
        [parse_listing(t, d) for t, d in zip(merged["title"], merged["description"])],
        index=merged.index,
    )
    accessory = merged["title"].fillna("").map(is_accessory_listing)
    keep = (
        specs["is_iphone"].fillna(False)
        & specs["model_family"].notna()
        & merged["price"].between(PRICE_MIN, PRICE_MAX)
        & ~accessory
    )
    usable = merged.loc[keep].reset_index(drop=True)
    log.info(
        "training frame: %d usable rows of %d merged (dropped %d accessories/parts)",
        len(usable), len(merged), int(accessory.sum()),
    )

    out_path = cfg.paths.processed_dir / "training.parquet"
    save_parquet(usable, out_path)
    return usable


def out_of_time_split(
    df: pd.DataFrame, val_frac: float, test_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split by ``created_at`` (oldest→train, newest→test)."""
    d = df.copy()
    d["_t"] = pd.to_numeric(d["created_at"], errors="coerce")
    d["_t"] = d["_t"].fillna(d["_t"].median())
    d = d.sort_values("_t").reset_index(drop=True)
    n = len(d)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    train = d.iloc[:n_train].drop(columns="_t")
    val = d.iloc[n_train:n_train + n_val].drop(columns="_t")
    test = d.iloc[n_train + n_val:].drop(columns="_t")
    log.info("out-of-time split: train=%d val=%d test=%d", len(train), len(val), len(test))
    return train, val, test


# --- training ------------------------------------------------------------
def train_all(cfg: Config, *, tune: bool = False, make_plots: bool = True) -> dict:
    """Train every model, evaluate, save artifacts, return the report dict."""
    cfg.paths.ensure()
    df = prepare_training_frame(cfg)
    if len(df) < 100:
        raise ValueError(f"only {len(df)} usable rows — need more data to train")

    train, val, test = out_of_time_split(df, cfg.model.val_fraction, cfg.model.test_fraction)

    # --- features (fit on train only) ------------------------------------
    pipe = FeaturePipeline(tfidf_max_features=cfg.features.tfidf_max_features)
    X_train = pipe.fit_transform(train)
    X_val, X_test = pipe.transform(val), pipe.transform(test)
    y_train = train["price"].to_numpy(float)
    y_val, y_test = val["price"].to_numpy(float), test["price"].to_numpy(float)
    w_train = train.get("sample_weight", pd.Series(1.0, index=train.index)).to_numpy(float)

    report: dict = {
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "n_rows": int(len(df)),
        "n_features": int(X_train.shape[1]),
        "split": {"train": len(train), "val": len(val), "test": len(test)},
        "price_band": [PRICE_MIN, PRICE_MAX],
        "models": {},
    }

    # --- baseline --------------------------------------------------------
    ridge = RidgeBaseline().fit(X_train, y_train, sample_weight=w_train)
    report["models"]["ridge_baseline"] = {
        "val": regression_metrics(y_val, ridge.predict(X_val)),
        "test": regression_metrics(y_test, ridge.predict(X_test)),
    }
    log.info("Ridge test: %s", report["models"]["ridge_baseline"]["test"])

    # --- main XGBoost ----------------------------------------------------
    xgb = XGBoostRegressorModel(params=cfg.model.xgboost, random_state=cfg.model.random_state)
    if tune:
        xgb.tune(X_train, y_train, cfg.model.grid, sample_weight=w_train, cv=3)
        xgb.fit(X_train, y_train, sample_weight=w_train)
    else:
        xgb.fit(X_train, y_train, sample_weight=w_train)
    report["models"]["xgboost"] = {
        "val": regression_metrics(y_val, xgb.predict(X_val)),
        "test": regression_metrics(y_test, xgb.predict(X_test)),
        "best_params": xgb.best_params_,
    }
    log.info("XGBoost test: %s", report["models"]["xgboost"]["test"])

    # --- LightGBM benchmark ---------------------------------------------
    try:
        lgbm = LightGBMBenchmark(random_state=cfg.model.random_state)
        lgbm.fit(X_train, y_train, sample_weight=w_train)
        report["models"]["lightgbm"] = {
            "val": regression_metrics(y_val, lgbm.predict(X_val)),
            "test": regression_metrics(y_test, lgbm.predict(X_test)),
        }
        log.info("LightGBM test: %s", report["models"]["lightgbm"]["test"])
    except Exception as exc:  # noqa: BLE001
        log.warning("LightGBM benchmark skipped: %s", exc)

    # --- quantile intervals ---------------------------------------------
    q = QuantileIntervalModel(
        low_alpha=cfg.model.quantile["low_alpha"],
        high_alpha=cfg.model.quantile["high_alpha"],
        random_state=cfg.model.random_state,
    )
    q.fit(X_train, y_train, sample_weight=w_train)
    # Conformalize on the validation split so test coverage matches the nominal
    # 80% (un-calibrated quantile regressors typically under-cover).
    q.calibrate(X_val, y_val)
    lo, hi = q.predict_interval(X_test)
    cov = interval_coverage(y_test, lo, hi)
    report["quantile"] = {
        "nominal": cfg.model.quantile["high_alpha"] - cfg.model.quantile["low_alpha"],
        "empirical_coverage": round(cov, 4),
        "mean_interval_width": float(np.mean(hi - lo)),
    }
    log.info("quantile coverage on test: %.3f", cov)

    # --- feature importance ---------------------------------------------
    fi = xgb.feature_importance(pipe.feature_names_, top=20)
    report["top_features"] = fi.to_dict(orient="records")

    # --- persist artifacts ----------------------------------------------
    a = cfg.paths.artifacts_dir
    pipe.save(a / "feature_pipeline.pkl")
    xgb.save(a / "model.pkl")
    q_path = a / "model_quantile.pkl"
    import pickle
    with open(q_path, "wb") as fh:
        pickle.dump(q, fh)
    with open(a / "training_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=str)
    log.info("artifacts written to %s", a)

    # --- plots -----------------------------------------------------------
    if make_plots:
        rd = cfg.paths.reports_dir
        y_pred_test = xgb.predict(X_test)
        viz.predicted_vs_actual(y_test, y_pred_test, rd / "predicted_vs_actual.png")
        viz.residual_hist(y_test, y_pred_test, rd / "residuals.png")
        viz.feature_importance(fi, rd / "feature_importance.png")
        viz.calibration_by_decile(y_test, y_pred_test, rd / "calibration.png")
        # EDA plots from the full usable frame
        fam = pd.DataFrame(
            [parse_listing(t, d) for t, d in zip(df["title"], df["description"])]
        )["model_family"]
        eda = df.assign(model_family=fam)
        viz.price_by_model(eda, "model_family", "price", rd / "price_by_model.png")
        viz.deals_by_city(df, "query_city", rd / "listings_by_city.png")
        log.info("plots written to %s", rd)

    return report
