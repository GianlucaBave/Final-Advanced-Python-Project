"""Ridge baseline (S7).

A regularized linear model is the sanity floor: it is fast, interpretable, and
sets the bar the gradient-boosted model must clearly beat to justify its
complexity. Features are standardized first (linear models are scale-sensitive,
unlike trees).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import BaseModel


class RidgeBaseline(BaseModel):
    name = "ridge_baseline"

    def __init__(self, alpha: float = 10.0, log_target: bool = True):
        super().__init__(log_target=log_target)
        self.alpha = alpha
        self.estimator = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True)),
                ("ridge", Ridge(alpha=alpha, random_state=0)),
            ]
        )

    def _fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight=None) -> None:
        self.estimator.fit(X.values, y, ridge__sample_weight=sample_weight)

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(X.values)
