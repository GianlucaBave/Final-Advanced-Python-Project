"""Composite investment scorer — the user-facing recommendation engine.

Combines three normalized components into a single ``[0, 1]`` score:

    investment_score = w_margin   * sigmoid(margin * 5)
                     + w_conf     * confidence
                     + w_risk     * (1 - risk)

* **margin** = (predicted_resale − asking − fees) / asking, squashed by a
  sigmoid so huge or negative margins saturate rather than dominate.
* **confidence** = how tight the model's 80% interval is, relative to the
  prediction (wide interval → low confidence).
* **risk** = the heuristic scam/low-info penalty.

The weights come from ``settings.yaml`` so the product behaviour is configurable
without code changes.
"""

from __future__ import annotations

import math

from ..config import ScoringConfig
from ..schema import Deal
from .risk import assess_risk


def _sigmoid(x: float) -> float:
    # numerically stable
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class InvestmentScorer:
    """Turns a (listing, prediction, interval) triple into a scored :class:`Deal`."""

    def __init__(self, cfg: ScoringConfig):
        self.cfg = cfg

    # --- components -------------------------------------------------------
    def expected_margin(self, asking: float, predicted: float) -> float:
        fees = self.cfg.wallapop_fee_pct * predicted + self.cfg.wallapop_fee_flat
        profit = predicted - asking - fees
        return profit / asking if asking > 0 else 0.0

    def confidence(self, predicted: float, low: float, high: float) -> float:
        if predicted <= 0:
            return 0.0
        width_pct = (high - low) / predicted
        full = self.cfg.confidence_interval_full_width
        return max(0.0, min(1.0, 1.0 - width_pct / full))

    def decide(self, score: float) -> str:
        t = self.cfg.thresholds
        if score >= t.strong_buy:
            return "STRONG BUY"
        if score >= t.buy:
            return "BUY"
        if score >= t.hold:
            return "HOLD"
        return "SKIP"

    # --- orchestration ----------------------------------------------------
    def score(
        self,
        *,
        listing_id: str,
        title: str,
        asking_price: float,
        predicted_price: float,
        price_low: float,
        price_high: float,
        n_photos: int,
        battery_pct: float | None,
        seller_id: str | None,
        description: str,
        city: str | None = None,
        url: str | None = None,
    ) -> Deal:
        margin = self.expected_margin(asking_price, predicted_price)
        conf = self.confidence(predicted_price, price_low, price_high)
        risk, flags = assess_risk(
            price=asking_price,
            predicted_price=predicted_price,
            n_photos=n_photos,
            battery_pct=battery_pct,
            seller_id=seller_id,
            description=description,
            title=title,
        )

        score = (
            self.cfg.margin_weight * _sigmoid(margin * 5.0)
            + self.cfg.confidence_weight * conf
            + self.cfg.risk_weight * (1.0 - risk)
        )
        score = max(0.0, min(1.0, score))

        return Deal(
            listing_id=listing_id,
            title=title,
            asking_price=round(asking_price, 2),
            predicted_fair_price=round(predicted_price, 2),
            price_low=round(price_low, 2),
            price_high=round(price_high, 2),
            expected_margin=round(margin, 4),
            confidence=round(conf, 4),
            risk_score=round(risk, 4),
            investment_score=round(score, 4),
            decision=self.decide(score),
            risk_flags=flags,
            city=city,
            url=url,
        )
