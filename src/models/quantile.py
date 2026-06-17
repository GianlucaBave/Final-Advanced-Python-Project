"""Quantile regression for prediction intervals (S7).

Point predictions alone can't express *uncertainty*, and uncertainty is half
the investment score. We train two XGBoost models with the native pinball
(quantile) loss — one at the 10th percentile, one at the 90th — giving an 80%
prediction interval ``[p10, p90]`` per listing. A wide interval means the model
is unsure (rare configuration, sparse data) and the confidence term is pulled
down accordingly.

Requires XGBoost ≥ 2.0 (``objective='reg:quantileerror'``). If unavailable, the
class falls back to a residual-based heuristic interval so the pipeline never
hard-fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ..logging_setup import get_logger

log = get_logger(__name__)


class QuantileIntervalModel:
    """Pair of quantile regressors producing an [low, high] interval."""

    def __init__(self, low_alpha: float = 0.1, high_alpha: float = 0.9,
                 params: dict | None = None, log_target: bool = True,
                 random_state: int = 42):
        self.low_alpha = low_alpha
        self.high_alpha = high_alpha
        self.log_target = log_target
        self.random_state = random_state
        self.params = params or dict(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
        )
        self.low_model_: XGBRegressor | None = None
        self.high_model_: XGBRegressor | None = None
        self._residual_band_: float = 0.25  # fallback half-width in log space
        self._native = True
        # CQR widening offset (log space), learned by :meth:`calibrate`.
        self._cqr_offset_: float = 0.0

    def _encode_y(self, y):
        y = np.asarray(y, float)
        return np.log1p(y) if self.log_target else y

    def _decode_y(self, y):
        y = np.asarray(y, float)
        return np.expm1(y) if self.log_target else y

    def _make(self, alpha: float) -> XGBRegressor:
        return XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=alpha,
            tree_method="hist", random_state=self.random_state, n_jobs=-1,
            **self.params,
        )

    def fit(self, X: pd.DataFrame, y, sample_weight=None) -> "QuantileIntervalModel":
        y_enc = self._encode_y(y)
        try:
            self.low_model_ = self._make(self.low_alpha)
            self.high_model_ = self._make(self.high_alpha)
            self.low_model_.fit(X, y_enc, sample_weight=sample_weight)
            self.high_model_.fit(X, y_enc, sample_weight=sample_weight)
            self._native = True
            log.info("quantile pair trained (native pinball loss, a=%.2f/%.2f)",
                     self.low_alpha, self.high_alpha)
        except Exception as exc:  # XGBoost < 2.0 or objective unsupported
            log.warning("native quantile objective unavailable (%s) — "
                        "falling back to residual band", exc)
            self._native = False
            mean = XGBRegressor(tree_method="hist", random_state=self.random_state,
                                n_jobs=-1, **self.params)
            mean.fit(X, y_enc, sample_weight=sample_weight)
            self.low_model_ = mean  # reuse mean model; band applied at predict
            self.high_model_ = mean
            resid = y_enc - mean.predict(X)
            self._residual_band_ = float(np.std(resid) * 1.2816)  # ~80% band
        return self

    def _predict_log_interval(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Raw quantile predictions in *log* space, before decoding/widening."""
        if self._native:
            lo = self.low_model_.predict(X)
            hi = self.high_model_.predict(X)
        else:
            mid = self.low_model_.predict(X)
            lo = mid - self._residual_band_
            hi = mid + self._residual_band_
        return lo, hi

    def calibrate(self, X_cal: pd.DataFrame, y_cal) -> float:
        """Conformalize the interval on held-out data to hit nominal coverage.

        Implements Conformalized Quantile Regression (Romano et al., 2019):
        the raw quantile interval is widened by the (1-α) empirical quantile of
        the conformity scores ``E_i = max(lo_i - y_i, y_i - hi_i)``. This gives
        finite-sample coverage guarantees and fixes the typical under-coverage
        of un-calibrated quantile regressors. Returns the learned offset.
        """
        if self.low_model_ is None:
            raise RuntimeError("calibrate called before fit")
        lo, hi = self._predict_log_interval(X_cal)
        y_log = self._encode_y(y_cal)
        scores = np.maximum(lo - y_log, y_log - hi)
        target = self.high_alpha - self.low_alpha          # e.g. 0.8
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * target) / n)
        self._cqr_offset_ = float(max(0.0, np.quantile(scores, level)))
        log.info("CQR calibration: offset=%.4f (log space), target coverage=%.0f%%",
                 self._cqr_offset_, target * 100)
        return self._cqr_offset_

    def predict_interval(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return calibrated ``(p10, p90)`` arrays in raw price space."""
        if self.low_model_ is None:
            raise RuntimeError("QuantileIntervalModel.predict_interval before fit")
        lo, hi = self._predict_log_interval(X)
        lo = lo - self._cqr_offset_           # widen in log space (CQR)
        hi = hi + self._cqr_offset_
        lo, hi = self._decode_y(lo), self._decode_y(hi)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        return np.clip(lo, 0, None), np.clip(hi, 0, None)
