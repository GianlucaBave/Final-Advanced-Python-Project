"""Typed configuration loader.

``settings.yaml`` is parsed once and frozen into nested, immutable dataclasses
so the rest of the codebase gets attribute access and IDE autocompletion
(``cfg.scoring.margin_weight``) instead of stringly-typed dict lookups.

Usage::

    from src.config import load_config
    cfg = load_config()                       # default config/settings.yaml
    cfg = load_config("config/settings.yaml") # explicit path
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DataPaths:
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    reports_dir: Path

    def ensure(self) -> None:
        """Create every directory if it does not yet exist."""
        for p in (
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.artifacts_dir,
            self.reports_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ScoringThresholds:
    strong_buy: float
    buy: float
    hold: float


@dataclass(frozen=True)
class ScoringConfig:
    margin_weight: float
    confidence_weight: float
    risk_weight: float
    wallapop_fee_pct: float
    wallapop_fee_flat: float
    confidence_interval_full_width: float
    thresholds: ScoringThresholds


@dataclass(frozen=True)
class ModelConfig:
    test_fraction: float
    val_fraction: float
    random_state: int
    xgboost: dict[str, Any]
    grid: dict[str, Any]
    quantile: dict[str, Any]


@dataclass(frozen=True)
class FeatureConfig:
    tfidf_max_features: int
    target_encoding_min_count: int


@dataclass(frozen=True)
class Config:
    """Root configuration object. Frozen → safe to share across modules."""

    paths: DataPaths
    scraping: dict[str, Any]
    source_weights: dict[str, float]
    model: ModelConfig
    features: FeatureConfig
    scoring: ScoringConfig
    logging: dict[str, str]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    # convenience accessors -------------------------------------------------
    @property
    def cities(self) -> dict[str, dict]:
        return self.scraping["cities"]

    @property
    def models(self) -> list[str]:
        return self.scraping["models"]


def _resolve(path_str: str) -> Path:
    """Resolve a possibly-relative path against the project root."""
    p = Path(path_str)
    return p if p.is_absolute() else (_PROJECT_ROOT / p)


@lru_cache(maxsize=4)
def load_config(path: str | Path | None = None) -> Config:
    """Load and cache the configuration. Repeated calls return the same object."""
    cfg_path = Path(path) if path else _DEFAULT_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    d = raw["data"]
    paths = DataPaths(
        raw_dir=_resolve(d["raw_dir"]),
        interim_dir=_resolve(d["interim_dir"]),
        processed_dir=_resolve(d["processed_dir"]),
        artifacts_dir=_resolve(d["artifacts_dir"]),
        reports_dir=_resolve(d["reports_dir"]),
    )

    s = raw["scoring"]
    scoring = ScoringConfig(
        margin_weight=s["margin_weight"],
        confidence_weight=s["confidence_weight"],
        risk_weight=s["risk_weight"],
        wallapop_fee_pct=s["wallapop_fee_pct"],
        wallapop_fee_flat=s["wallapop_fee_flat"],
        confidence_interval_full_width=s["confidence_interval_full_width"],
        thresholds=ScoringThresholds(**s["thresholds"]),
    )

    m = raw["model"]
    model = ModelConfig(
        test_fraction=m["test_fraction"],
        val_fraction=m["val_fraction"],
        random_state=m["random_state"],
        xgboost=m["xgboost"],
        grid=m["grid"],
        quantile=m["quantile"],
    )

    f = raw["features"]
    features = FeatureConfig(
        tfidf_max_features=f["tfidf_max_features"],
        target_encoding_min_count=f["target_encoding_min_count"],
    )

    return Config(
        paths=paths,
        scraping=raw["scraping"],
        source_weights=raw["source_weights"],
        model=model,
        features=features,
        scoring=scoring,
        logging=raw["logging"],
        raw=raw,
    )
