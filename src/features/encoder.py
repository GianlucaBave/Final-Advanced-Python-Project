"""Encoders for categorical and high-cardinality features.

* :func:`city_tier` — maps a city name to an ordinal market tier.
* :class:`TargetEncoder` — smoothed mean-target encoding for high-cardinality
  categoricals (e.g. seller id, exact colour). Fitted on training data only and
  stored inside the :class:`~src.features.pipeline.FeaturePipeline` so inference
  reuses the exact same mapping — no train/serve skew.

Low-cardinality categoricals (model_family, colour) are one-hot encoded directly
in the pipeline; XGBoost needs no scaling, so numerics pass through untouched.
"""

from __future__ import annotations

import pandas as pd

# Tier 1 = deepest markets (most liquidity), tier 3 = thin markets.
_CITY_TIER = {
    "madrid": 1, "barcelona": 1,
    "valencia": 2, "sevilla": 2, "zaragoza": 2, "malaga": 2,
}


def city_tier(city: str | None) -> int:
    """Return the ordinal market tier for a city (3 = unknown/other)."""
    if not city:
        return 3
    return _CITY_TIER.get(str(city).strip().lower(), 3)


class TargetEncoder:
    """Smoothed mean-target encoder for a single categorical column.

    encoding(category) = (n_c * mean_c + m * global_mean) / (n_c + m)

    where ``m`` is the smoothing strength: rare categories are pulled toward the
    global mean, frequent ones toward their own mean. Unknown categories at
    inference time fall back to the global mean.
    """

    def __init__(self, smoothing: float = 20.0):
        self.smoothing = smoothing
        self.mapping_: dict = {}
        self.global_mean_: float = 0.0
        self.fitted_ = False

    def fit(self, values: pd.Series, target: pd.Series) -> "TargetEncoder":
        df = pd.DataFrame({"v": values.astype("object"), "y": target.astype(float)})
        self.global_mean_ = float(df["y"].mean())
        agg = df.groupby("v")["y"].agg(["mean", "count"])
        m = self.smoothing
        agg["enc"] = (agg["count"] * agg["mean"] + m * self.global_mean_) / (agg["count"] + m)
        self.mapping_ = agg["enc"].to_dict()
        self.fitted_ = True
        return self

    def transform(self, values: pd.Series) -> pd.Series:
        if not self.fitted_:
            raise RuntimeError("TargetEncoder.transform called before fit")
        return values.astype("object").map(self.mapping_).fillna(self.global_mean_).astype(float)

    def fit_transform(self, values: pd.Series, target: pd.Series) -> pd.Series:
        return self.fit(values, target).transform(values)
