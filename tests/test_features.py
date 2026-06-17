"""Tests for the FeaturePipeline — train/inference parity (S9.5).

The single most important production property: a model fitted on training data
must see the *exact same* feature columns, in the same order, when scoring new
listings. These tests assert that parity and that imputation removes all NaNs.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.pipeline import FeaturePipeline
from src.schema import RAW_COLUMNS


def _sample_frame(n=40) -> pd.DataFrame:
    models = ["iPhone 13 Pro 256GB", "iPhone 11 64GB", "iPhone 14 Plus 128GB",
              "iPhone 12 mini", "iPhone 15 Pro Max 256GB titanio"]
    rows = []
    for i in range(n):
        rows.append({
            "listing_id": f"id{i}",
            "source": "wallapop",
            "title": models[i % len(models)],
            "description": f"Batería {80 + i % 20}% como nuevo con caja",
            "price": 200 + (i % 10) * 50,
            "currency": "EUR",
            "city": "Madrid",
            "n_photos": i % 5,
            "seller_id": f"s{i % 7}",
            "created_at": 1781600000000 + i * 1_000_000,
            "shipping_available": bool(i % 2),
            "is_refurbished": False,
            "has_warranty_flag": bool(i % 3),
            "query_city": "madrid",
        })
    df = pd.DataFrame(rows)
    # ensure all RAW_COLUMNS exist
    for c in RAW_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df


def test_fit_transform_shapes():
    pipe = FeaturePipeline(tfidf_max_features=20)
    X = pipe.fit_transform(_sample_frame())
    assert X.shape[0] == 40
    assert X.shape[1] == len(pipe.feature_names_)
    assert not X.isna().any().any(), "pipeline must impute all NaNs"


def test_train_inference_column_parity():
    pipe = FeaturePipeline(tfidf_max_features=20)
    pipe.fit_transform(_sample_frame())
    # an inference batch with a brand-new colour and unseen model
    infer = _sample_frame(5)
    infer.loc[0, "title"] = "iPhone 99 Ultra 256GB plata"  # unseen model
    X_inf = pipe.transform(infer)
    assert list(X_inf.columns) == pipe.feature_names_
    assert not X_inf.isna().any().any()


def test_transform_before_fit_raises():
    pipe = FeaturePipeline()
    with pytest.raises(RuntimeError):
        pipe.transform(_sample_frame(3))


def test_save_load_roundtrip(tmp_path):
    pipe = FeaturePipeline(tfidf_max_features=15)
    X = pipe.fit_transform(_sample_frame())
    path = pipe.save(tmp_path / "fp.pkl")
    loaded = FeaturePipeline.load(path)
    X2 = loaded.transform(_sample_frame())
    assert list(X.columns) == list(X2.columns)
    assert np.allclose(X.values, X2.values)
