"""Tests for the investment scorer and risk heuristics (S9.5).

Covers the edge cases that matter for trust: a great deal scores high, an
obvious scam is penalised, an overpriced listing is rejected, and a broken
phone is flagged regardless of its cheap price.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.investment.risk import assess_risk
from src.investment.scorer import InvestmentScorer


def _scorer() -> InvestmentScorer:
    return InvestmentScorer(load_config().scoring)


def test_perfect_deal_scores_high():
    s = _scorer()
    deal = s.score(
        listing_id="1", title="iPhone 13 Pro 256GB como nuevo",
        asking_price=400, predicted_price=600,
        price_low=585, price_high=615,           # tight interval → high confidence
        n_photos=6, battery_pct=95, seller_id="s1",
        description="como nuevo, con caja y factura",
    )
    assert deal.expected_margin > 0.2
    assert deal.confidence > 0.6
    assert deal.investment_score >= 0.7
    assert deal.decision in ("STRONG BUY", "BUY")
    assert deal.risk_flags == []


def test_overpriced_listing_is_skipped():
    s = _scorer()
    deal = s.score(
        listing_id="2", title="iPhone 13 Pro 256GB",
        asking_price=800, predicted_price=550,   # asking far above fair value
        price_low=520, price_high=580,
        n_photos=5, battery_pct=90, seller_id="s2",
        description="buen estado",
    )
    assert deal.expected_margin < 0
    assert deal.decision in ("HOLD", "SKIP")


def test_too_good_to_be_true_flagged_as_scam():
    s = _scorer()
    deal = s.score(
        listing_id="3", title="iPhone 15 Pro Max 256GB",
        asking_price=150, predicted_price=1100,  # < 30% of fair → scam heuristic
        price_low=900, price_high=1300,
        n_photos=1, battery_pct=None, seller_id=None,
        description="urge vender, pago por transferencia bancaria",
    )
    assert deal.risk_score >= 0.5
    assert "price too low (possible scam)" in deal.risk_flags
    assert deal.decision in ("HOLD", "SKIP", "BUY")  # risk drags the score down
    assert deal.investment_score < 0.75             # never STRONG BUY


def test_damaged_phone_flagged():
    risk, flags = assess_risk(
        price=80, predicted_price=300, n_photos=3, battery_pct=88,
        seller_id="s9", description="placa dañada, no enciende",
        title="iPhone 13 Pro para piezas",
    )
    assert risk >= 0.4
    assert "damaged / for parts" in flags


def test_all_zero_inputs_do_not_crash():
    s = _scorer()
    deal = s.score(
        listing_id="0", title="", asking_price=0, predicted_price=0,
        price_low=0, price_high=0, n_photos=0, battery_pct=None,
        seller_id=None, description="",
    )
    assert 0.0 <= deal.investment_score <= 1.0


def test_decision_thresholds_monotonic():
    s = _scorer()
    assert s.decide(0.9) == "STRONG BUY"
    assert s.decide(0.6) == "BUY"
    assert s.decide(0.45) == "HOLD"
    assert s.decide(0.2) == "SKIP"
