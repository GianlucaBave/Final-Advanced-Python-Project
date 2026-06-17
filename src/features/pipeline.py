"""FeaturePipeline — the single fit/transform path shared by train & inference.

This is the most important class in the project for production correctness.
*One* object is fitted at training time and serialized to
``artifacts/feature_pipeline.pkl``; at inference the very same object is loaded
and used. Because parsing, imputation, encoding and the TF-IDF vocabulary are
all captured inside it, the feature matrix at serve time is guaranteed to have
the same columns, in the same order, with the same imputed values as at train
time. That eliminates train/serve skew — the most common production-ML bug.

Pipeline stages (``fit`` learns state, ``transform`` applies it):

1. parse title+description → structured specs (regex/fuzzy, stateless)
2. engineer features (age, description length, seller activity, battery flag…)
3. impute missing numerics (battery/storage/year by model-family median)
4. encode categoricals (one-hot model_family + colour) and TF-IDF description
5. assemble a dense numeric DataFrame with a fixed column order
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ..logging_setup import get_logger
from .encoder import city_tier
from .parser import parse_listing

log = get_logger(__name__)

# Columns that always exist in the assembled matrix (besides one-hot/tfidf).
_NUMERIC_BASE = [
    "model_year", "storage_gb", "battery_pct", "condition_label",
    "n_photos", "description_length", "days_since_posted",
    "city_tier", "seller_n_listings",
]
_BINARY_BASE = [
    "has_box", "has_warranty", "has_accessories", "has_battery_info",
    "shipping_available", "is_refurbished", "has_warranty_flag",
]


class FeaturePipeline:
    """Stateful, picklable feature transformer for iPhone listings."""

    def __init__(self, tfidf_max_features: int = 50):
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_: TfidfVectorizer | None = None
        self.model_family_cats_: list[str] = []
        self.color_cats_: list[str] = []
        self.battery_median_by_family_: dict[str, float] = {}
        self.storage_median_by_family_: dict[str, float] = {}
        self.global_medians_: dict[str, float] = {}
        self.feature_names_: list[str] = []
        self.fitted_ = False

    # --- stage 1+2: raw frame → engineered (pre-encoding) frame -----------
    @staticmethod
    def _parse_frame(df: pd.DataFrame) -> pd.DataFrame:
        """Run the spec parser over every row → a structured DataFrame."""
        specs = [
            parse_listing(t, d)
            for t, d in zip(df.get("title", ""), df.get("description", ""))
        ]
        return pd.DataFrame(specs, index=df.index)

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the engineered (still un-encoded) feature frame."""
        specs = self._parse_frame(df)
        out = pd.DataFrame(index=df.index)

        # parsed specs
        out["model_family"] = specs["model_family"].fillna("unknown")
        out["color"] = specs["color"].fillna("unknown")
        out["model_year"] = specs["model_year"]
        out["storage_gb"] = specs["storage_gb"]
        out["battery_pct"] = specs["battery_pct"]
        out["condition_label"] = specs["condition_label"]
        out["has_box"] = specs["has_box"].astype(int)
        out["has_warranty"] = specs["has_warranty"].astype(int)
        out["has_accessories"] = specs["has_accessories"].astype(int)
        out["has_battery_info"] = specs["battery_pct"].notna().astype(int)

        # listing-derived signals
        desc = df.get("description", pd.Series("", index=df.index)).fillna("")
        out["description_length"] = desc.astype(str).str.len()
        out["n_photos"] = pd.to_numeric(df.get("n_photos", 0), errors="coerce").fillna(0)

        # temporal: age in days from epoch-ms created_at (S6 time-series feature)
        out["days_since_posted"] = self._days_since_posted(df.get("created_at"))

        # geography
        city_src = df.get("query_city", df.get("city"))
        out["city_tier"] = (city_src if city_src is not None else "unknown").map(city_tier) \
            if isinstance(city_src, pd.Series) else 3

        # seller activity within the batch
        if "seller_id" in df.columns:
            counts = df["seller_id"].map(df["seller_id"].value_counts())
            out["seller_n_listings"] = counts.fillna(1).astype(float)
        else:
            out["seller_n_listings"] = 1.0

        # boolean flags carried straight from the scrape
        for col in ("shipping_available", "is_refurbished", "has_warranty_flag"):
            out[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0).astype(int)

        # keep description text for TF-IDF stage
        out["_desc_text"] = desc.astype(str)
        return out

    @staticmethod
    def _days_since_posted(created: pd.Series | None) -> pd.Series:
        if created is None:
            return pd.Series(np.nan)
        c = pd.to_numeric(created, errors="coerce")
        now_ms = pd.Timestamp.now().timestamp() * 1000.0
        days = (now_ms - c) / 86_400_000.0
        # Clamp to [0, 10 years]: anything outside is bad/missing timestamp data,
        # and unbounded values would destabilise the linear baseline.
        return days.clip(lower=0, upper=3650)

    # --- stage 3: imputation ---------------------------------------------
    def _fit_imputers(self, eng: pd.DataFrame) -> None:
        self.battery_median_by_family_ = (
            eng.groupby("model_family")["battery_pct"].median().dropna().to_dict()
        )
        self.storage_median_by_family_ = (
            eng.groupby("model_family")["storage_gb"].median().dropna().to_dict()
        )
        self.global_medians_ = {
            "battery_pct": float(eng["battery_pct"].median(skipna=True) or 85.0),
            "storage_gb": float(eng["storage_gb"].median(skipna=True) or 128.0),
            "model_year": float(eng["model_year"].median(skipna=True) or 2021.0),
            "condition_label": 2.0,  # "used" prior
            "days_since_posted": float(eng["days_since_posted"].median(skipna=True) or 30.0),
        }

    def _apply_imputers(self, eng: pd.DataFrame) -> pd.DataFrame:
        eng = eng.copy()
        # family-specific medians for battery & storage, then global fallback
        for col, table in (
            ("battery_pct", self.battery_median_by_family_),
            ("storage_gb", self.storage_median_by_family_),
        ):
            fam_fill = eng["model_family"].map(table)
            eng[col] = eng[col].fillna(fam_fill).fillna(self.global_medians_[col])
        for col in ("model_year", "condition_label", "days_since_posted"):
            eng[col] = eng[col].fillna(self.global_medians_[col])
        return eng

    # --- stage 4+5: encode + assemble ------------------------------------
    def _assemble(self, eng: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        # one-hot for low-cardinality categoricals
        if fit:
            self.model_family_cats_ = sorted(eng["model_family"].unique().tolist())
            self.color_cats_ = sorted(eng["color"].unique().tolist())
        fam = pd.get_dummies(eng["model_family"], prefix="fam")
        col = pd.get_dummies(eng["color"], prefix="color")

        # TF-IDF on description
        if fit:
            self.tfidf_ = TfidfVectorizer(
                max_features=self.tfidf_max_features,
                strip_accents="unicode",
                lowercase=True,
                token_pattern=r"(?u)\b[a-zA-ZáéíóúüñÁÉÍÓÚÑ]{3,}\b",
            )
            tfidf_mat = self.tfidf_.fit_transform(eng["_desc_text"])
        else:
            tfidf_mat = self.tfidf_.transform(eng["_desc_text"])
        tfidf_df = pd.DataFrame(
            tfidf_mat.toarray(),
            columns=[f"tfidf_{t}" for t in self.tfidf_.get_feature_names_out()],
            index=eng.index,
        )

        numeric = eng[_NUMERIC_BASE + _BINARY_BASE].astype(float)
        X = pd.concat([numeric, fam, col, tfidf_df], axis=1)

        if fit:
            self.feature_names_ = X.columns.tolist()
        else:
            # enforce identical columns/order as training (train/serve parity)
            X = X.reindex(columns=self.feature_names_, fill_value=0.0)
        return X.astype(float)

    # --- public API -------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        eng = self._engineer(df)
        self._fit_imputers(eng)
        eng = self._apply_imputers(eng)
        X = self._assemble(eng, fit=True)
        self.fitted_ = True
        log.info("FeaturePipeline fitted: %d features, %d rows", X.shape[1], X.shape[0])
        return X

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("FeaturePipeline.transform called before fit")
        eng = self._engineer(df)
        eng = self._apply_imputers(eng)
        return self._assemble(eng, fit=False)

    def parsed_specs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Expose the raw parsed specs (used by EDA + the report layer)."""
        return self._parse_frame(df)

    # --- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        log.info("saved FeaturePipeline → %s", path)
        return path

    @staticmethod
    def load(path: str | Path) -> "FeaturePipeline":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, FeaturePipeline):
            raise TypeError(f"{path} is not a FeaturePipeline")
        return obj
