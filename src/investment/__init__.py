"""Investment layer: risk heuristics, composite scorer, detector orchestrator."""

from .detector import DealDetector
from .risk import assess_risk
from .scorer import InvestmentScorer

__all__ = ["assess_risk", "InvestmentScorer", "DealDetector"]
