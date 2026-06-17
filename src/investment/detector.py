"""DealDetector — the inference orchestrator.

Loads the three serialized artifacts (feature pipeline, main regressor, quantile
pair), and turns a batch of raw listings into a ranked list of scored
:class:`~src.schema.Deal` objects. This is the object the CLI ``scan`` command
and the live-report notebook both drive.

Train/serve parity is guaranteed because the *same* ``FeaturePipeline`` instance
that was fitted during training is loaded here — no feature is recomputed by
hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..features.pipeline import FeaturePipeline
from ..logging_setup import get_logger
from ..models.base import BaseModel
from ..models.quantile import QuantileIntervalModel
from ..schema import Deal
from .scorer import InvestmentScorer

log = get_logger(__name__)


class DealDetector:
    """End-to-end inference: raw listings → ranked, scored deals."""

    def __init__(
        self,
        pipeline: FeaturePipeline,
        model: BaseModel,
        quantile: QuantileIntervalModel,
        scorer: InvestmentScorer,
    ):
        self.pipeline = pipeline
        self.model = model
        self.quantile = quantile
        self.scorer = scorer

    # --- construction from artifacts -------------------------------------
    @classmethod
    def from_artifacts(cls, cfg: Config) -> "DealDetector":
        a = cfg.paths.artifacts_dir
        pipeline = FeaturePipeline.load(a / "feature_pipeline.pkl")
        model = BaseModel.load(a / "model.pkl")
        quantile = _load_pickle(a / "model_quantile.pkl")
        scorer = InvestmentScorer(cfg.scoring)
        log.info("DealDetector loaded artifacts from %s", a)
        return cls(pipeline, model, quantile, scorer)

    # --- inference --------------------------------------------------------
    def detect(self, listings: pd.DataFrame) -> list[Deal]:
        """Score every listing and return Deals sorted by investment score desc."""
        if listings.empty:
            return []

        X = self.pipeline.transform(listings)
        specs = self.pipeline.parsed_specs(listings)
        predicted = self.model.predict(X)
        low, high = self.quantile.predict_interval(X)

        deals: list[Deal] = []
        for i, (_, row) in enumerate(listings.reset_index(drop=True).iterrows()):
            battery = specs.iloc[i]["battery_pct"]
            deal = self.scorer.score(
                listing_id=str(row.get("listing_id")),
                title=str(row.get("title", "")),
                asking_price=float(row.get("price", 0.0)),
                predicted_price=float(predicted[i]),
                price_low=float(low[i]),
                price_high=float(high[i]),
                n_photos=int(row.get("n_photos", 0) or 0),
                battery_pct=None if pd.isna(battery) else float(battery),
                seller_id=row.get("seller_id"),
                description=str(row.get("description", "")),
                city=row.get("city") or row.get("query_city"),
                url=row.get("url"),
            )
            deals.append(deal)

        deals.sort(key=lambda d: d.investment_score, reverse=True)
        log.info("scored %d listings → %d deals", len(listings), len(deals))
        return deals

    # --- reporting --------------------------------------------------------
    def to_report(self, deals: list[Deal], *, top: int = 20,
                  meta: dict | None = None) -> dict:
        return {
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "meta": meta or {},
            "n_scored": len(deals),
            "deals": [d.to_dict() for d in deals[:top]],
        }

    def save_report(self, deals: list[Deal], path: str | Path, *,
                    top: int = 20, meta: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_report(deals, top=top, meta=meta), fh,
                      ensure_ascii=False, indent=2)
        log.info("saved deal report → %s", path)
        return path


def _load_pickle(path: str | Path):
    import pickle
    with open(path, "rb") as fh:
        return pickle.load(fh)
