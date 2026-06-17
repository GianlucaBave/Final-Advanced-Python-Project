"""Abstract regression model (S1 — OOP).

A thin common interface so the training script and the inference detector can
treat Ridge, XGBoost and LightGBM interchangeably (Liskov substitution): every
model exposes ``fit``, ``predict``, ``save`` and ``load``. Concrete models hold
their own estimator and (optionally) log-transform the target.
"""

from __future__ import annotations

import abc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_setup import get_logger


class BaseModel(abc.ABC):
    """Common interface for every price-regression model."""

    name: str = "base"

    def __init__(self, log_target: bool = True):
        # Prices are right-skewed and strictly positive → modelling log1p(price)
        # stabilises variance and makes the loss closer to relative (MAPE-like).
        self.log_target = log_target
        self.log = get_logger(f"model.{self.name}")
        self._fitted = False

    # subclasses implement the estimator-specific fit/predict
    @abc.abstractmethod
    def _fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight=None) -> None: ...

    @abc.abstractmethod
    def _predict(self, X: pd.DataFrame) -> np.ndarray: ...

    # --- target transform helpers ----------------------------------------
    def _encode_y(self, y) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        return np.log1p(y) if self.log_target else y

    def _decode_y(self, y) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if not self.log_target:
            return y
        # Clip in log space before expm1 so a pathological prediction can never
        # overflow to inf (exp(12) ≈ €160k is already far beyond any phone).
        return np.expm1(np.clip(y, 0.0, 12.0))

    # --- public API -------------------------------------------------------
    def fit(self, X: pd.DataFrame, y, sample_weight=None) -> "BaseModel":
        self._fit(X, self._encode_y(y), sample_weight=sample_weight)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: predict called before fit")
        return np.clip(self._decode_y(self._predict(X)), 0, None)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        self.log.info("saved %s → %s", self.name, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "BaseModel":
        with open(path, "rb") as fh:
            return pickle.load(fh)
