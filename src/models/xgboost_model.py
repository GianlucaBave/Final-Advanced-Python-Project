"""XGBoost main model + LightGBM benchmark (S7).

:class:`XGBoostRegressorModel` is the production estimator. It supports an
optional :meth:`tune` step using :class:`GridSearchCV` so hyperparameters are
chosen on validation MAE rather than guessed. :class:`LightGBMBenchmark` is a
drop-in comparison model to show the result is not XGBoost-specific.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from .base import BaseModel


class XGBoostRegressorModel(BaseModel):
    name = "xgboost"

    def __init__(self, params: dict | None = None, log_target: bool = True,
                 random_state: int = 42):
        super().__init__(log_target=log_target)
        self.random_state = random_state
        default = dict(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, min_child_weight=3,
            reg_alpha=0.1, objective="reg:squarederror",
            tree_method="hist", random_state=random_state, n_jobs=-1,
        )
        if params:
            default.update(params)
        self.params = default
        self.estimator = XGBRegressor(**default)
        self.best_params_: dict | None = None

    def _fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight=None) -> None:
        self.estimator.fit(X, y, sample_weight=sample_weight)

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(X)

    def tune(self, X: pd.DataFrame, y, grid: dict, sample_weight=None,
             cv=3) -> dict:
        """Grid-search hyperparameters on (negative) MAE. Refits the best model.

        ``y`` is supplied in raw price space; we tune in the model's target
        space (log) for consistency with ``fit``. The CV folds are
        :class:`TimeSeriesSplit` (the data arrives already sorted oldest→newest),
        so tuning honours the same out-of-time discipline as final evaluation —
        a plain shuffled K-fold would leak future prices into the past and pick
        hyperparameters that look good in CV but generalise worse on real future
        listings.
        """
        y_enc = self._encode_y(y)
        base = XGBRegressor(
            objective="reg:squarederror", tree_method="hist",
            random_state=self.random_state, n_jobs=-1,
        )
        splitter = TimeSeriesSplit(n_splits=cv if isinstance(cv, int) else 3)
        search = GridSearchCV(
            base, grid, scoring="neg_mean_absolute_error",
            cv=splitter, n_jobs=-1, refit=True, verbose=0,
        )
        fit_params = {"sample_weight": sample_weight} if sample_weight is not None else {}
        search.fit(X, y_enc, **fit_params)
        self.best_params_ = search.best_params_
        self.params.update(search.best_params_)
        self.estimator = search.best_estimator_
        self._fitted = True
        self.log.info("grid search best params: %s (CV MAE=%.4f log-space)",
                      search.best_params_, -search.best_score_)
        return search.best_params_

    def feature_importance(self, feature_names: list[str], top: int = 20) -> pd.DataFrame:
        imp = self.estimator.feature_importances_
        df = pd.DataFrame({"feature": feature_names, "importance": imp})
        return df.sort_values("importance", ascending=False).head(top).reset_index(drop=True)


class LightGBMBenchmark(BaseModel):
    name = "lightgbm"

    def __init__(self, log_target: bool = True, random_state: int = 42):
        super().__init__(log_target=log_target)
        from lightgbm import LGBMRegressor

        self.estimator = LGBMRegressor(
            n_estimators=500, max_depth=-1, num_leaves=31,
            learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            random_state=random_state, n_jobs=-1, verbose=-1,
        )

    def _fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight=None) -> None:
        self.estimator.fit(X, y, sample_weight=sample_weight)

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(X)
