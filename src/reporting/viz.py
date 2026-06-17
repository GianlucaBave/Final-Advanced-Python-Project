"""Visualization helpers (S3 — data visualization).

Every function takes data + an output path, renders a single focused figure with
matplotlib/seaborn, saves it as PNG and returns the path. Keeping plotting out
of the training/inference logic means the notebooks and the CLI can reuse the
exact same figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend — safe in scripts and CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def predicted_vs_actual(y_true, y_pred, path, title="Predicted vs actual price") -> Path:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.35, s=18, edgecolor="none")
    lim = [0, max(np.max(y_true), np.max(y_pred)) * 1.05]
    ax.plot(lim, lim, "r--", lw=2, label="ideal y = x")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual price (€)"); ax.set_ylabel("Predicted price (€)")
    ax.set_title(title); ax.legend()
    return _save(fig, path)


def residual_hist(y_true, y_pred, path, title="Residuals (actual − predicted)") -> Path:
    resid = np.asarray(y_true, float) - np.asarray(y_pred, float)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(resid, bins=40, kde=True, ax=ax, color="#4C72B0")
    ax.axvline(0, color="red", ls="--", lw=2)
    ax.set_xlabel("Residual (€)"); ax.set_title(title)
    return _save(fig, path)


def feature_importance(importance_df: pd.DataFrame, path,
                       title="Top feature importances (XGBoost)") -> Path:
    fig, ax = plt.subplots(figsize=(9, 7))
    d = importance_df.iloc[::-1]
    ax.barh(d["feature"], d["importance"], color="#55A868")
    ax.set_xlabel("Gain importance"); ax.set_title(title)
    return _save(fig, path)


def score_distribution(scores, decisions, path,
                       title="Investment score distribution") -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    df = pd.DataFrame({"score": scores, "decision": decisions})
    palette = {"STRONG BUY": "#2ca02c", "BUY": "#98df8a",
               "HOLD": "#c7c7c7", "SKIP": "#d62728"}
    sns.histplot(data=df, x="score", hue="decision", bins=30, multiple="stack",
                 palette=palette, ax=ax)
    ax.set_xlabel("Investment score"); ax.set_title(title)
    return _save(fig, path)


def price_by_model(df: pd.DataFrame, model_col, price_col, path,
                   title="iPhone price by model") -> Path:
    order = df.groupby(model_col)[price_col].median().sort_values().index
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(data=df, x=model_col, y=price_col, order=order, ax=ax,
                showfliers=False, hue=model_col, legend=False, palette="viridis")
    ax.tick_params(axis="x", rotation=60)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    ax.set_xlabel(""); ax.set_ylabel("Price (€)"); ax.set_title(title)
    return _save(fig, path)


def deals_by_city(df: pd.DataFrame, city_col, path,
                  title="Listings by city") -> Path:
    counts = df[city_col].fillna("unknown").value_counts().head(12)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=counts.values, y=counts.index, ax=ax,
                hue=counts.index, legend=False, palette="mako")
    ax.set_xlabel("count"); ax.set_title(title)
    return _save(fig, path)


def calibration_by_decile(y_true, y_pred, path,
                          title="Calibration: mean predicted vs actual by decile") -> Path:
    df = pd.DataFrame({"y": np.asarray(y_true, float), "p": np.asarray(y_pred, float)})
    df["decile"] = pd.qcut(df["p"], 10, labels=False, duplicates="drop")
    agg = df.groupby("decile").agg(pred=("p", "mean"), actual=("y", "mean"))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(agg["pred"], agg["actual"], "o-", lw=2)
    lim = [0, max(agg["pred"].max(), agg["actual"].max()) * 1.05]
    ax.plot(lim, lim, "r--", lw=2, label="ideal")
    ax.set_xlabel("Mean predicted (€)"); ax.set_ylabel("Mean actual (€)")
    ax.set_title(title); ax.legend()
    return _save(fig, path)
