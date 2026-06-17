"""Regression + interval evaluation metrics (S7 / S8).

Centralised so the training script, evaluate command and notebooks all report
the same numbers the same way.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error (%). Guards against divide-by-zero."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = y_true > 1e-6
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": mape(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def interval_coverage(y_true, low, high) -> float:
    """Fraction of actuals falling inside [low, high] — should match nominal 80%."""
    y_true = np.asarray(y_true, float)
    low = np.asarray(low, float)
    high = np.asarray(high, float)
    return float(np.mean((y_true >= low) & (y_true <= high)))
