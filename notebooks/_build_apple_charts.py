"""Generate Apple/glass-styled (dark, transparent-bg) charts for the deck."""
from __future__ import annotations
import sys, warnings; sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
from pathlib import Path
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RPT = ROOT / "artifacts" / "reports"
RPT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "text.color": "white", "axes.labelcolor": "#EBEBF5",
    "xtick.color": "#C7C7CC", "ytick.color": "#C7C7CC",
    "axes.edgecolor": "#48484A", "axes.linewidth": 0.8,
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
    "font.family": ["SF Pro Display", "Helvetica Neue", "Arial"],
    "font.size": 15, "grid.color": "#3A3A3C", "grid.alpha": 0.5,
})
BLUE, GREEN, AMBER, RED, PURPLE = "#0A84FF", "#30D158", "#FF9F0A", "#FF453A", "#BF5AF2"


def save(fig, name):
    p = RPT / name
    fig.savefig(p, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig); print("wrote", p.name); return p


# --- 1. model comparison (R2) ---
rep = json.load(open(ROOT / "artifacts/training_report.json"))
m = rep["models"]
names, r2s, maes = [], [], []
for k, lbl in [("ridge_baseline", "Ridge\nbaseline"), ("xgboost", "XGBoost"), ("lightgbm", "LightGBM")]:
    if k in m:
        names.append(lbl); r2s.append(m[k]["test"]["r2"]); maes.append(m[k]["test"]["mae"])
fig, ax = plt.subplots(figsize=(8, 5))
cols = ["#5E5CE6", BLUE, GREEN]
bars = ax.bar(names, r2s, color=cols[:len(names)], width=0.6, zorder=3)
for b, r, mae in zip(bars, r2s, maes):
    ax.text(b.get_x()+b.get_width()/2, r+0.012, f"R² {r:.3f}", ha="center", fontsize=15, fontweight="bold")
    ax.text(b.get_x()+b.get_width()/2, r/2, f"MAE\n€{mae:.0f}", ha="center", va="center", color="white", fontsize=13)
ax.set_ylim(0, max(r2s)*1.18); ax.set_ylabel("R²  (held-out test)")
ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
save(fig, "slide_model_comparison.png")

# --- 2. data sources donut ---
src = {"Wallapop · ES live": 8262, "eBay sold · US": 957, "eBay archive · US '23": 2165, "US e-commerce '26": 2361}
fig, ax = plt.subplots(figsize=(7, 7))
w, _ = ax.pie(src.values(), colors=[BLUE, GREEN, PURPLE, AMBER], startangle=90,
              wedgeprops=dict(width=0.42, edgecolor="#0B0B0F", linewidth=3))
ax.text(0, 0.12, f"{sum(src.values()):,}", ha="center", fontsize=34, fontweight="bold")
ax.text(0, -0.16, "listings", ha="center", fontsize=16, color="#C7C7CC")
ax.legend(w, [f"{k}  ·  {v:,}" for k, v in src.items()], loc="center",
          bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=13, labelcolor="white")
save(fig, "slide_data_sources.png")

# --- 3 & 4. predicted-vs-actual + feature importance from artifacts ---
try:
    from src.config import load_config
    from src.training import prepare_training_frame, out_of_time_split
    from src.features.pipeline import FeaturePipeline
    from src.models.base import BaseModel
    cfg = load_config(); df = prepare_training_frame(cfg)
    tr, va, te = out_of_time_split(df, cfg.model.val_fraction, cfg.model.test_fraction)
    pipe = FeaturePipeline.load(cfg.paths.artifacts_dir / "feature_pipeline.pkl")
    mdl = BaseModel.load(cfg.paths.artifacts_dir / "model.pkl")
    Xte = pipe.transform(te); yte = te.price.values; yp = mdl.predict(Xte)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(yte, yp, alpha=0.30, s=22, color=BLUE, edgecolor="none", zorder=3)
    lim = [0, np.percentile(np.concatenate([yte, yp]), 99)*1.05]
    ax.plot(lim, lim, "--", lw=2, color=GREEN, label="perfect prediction")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual price (€)"); ax.set_ylabel("Predicted price (€)")
    ax.legend(frameon=False, labelcolor="white"); ax.grid(alpha=0.3)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    save(fig, "slide_pred_vs_actual.png")
except Exception as e:
    print("pred-vs-actual skipped:", e)

# feature importance from report
fi = pd.DataFrame(rep["top_features"]).head(10).iloc[::-1]
nice = (fi["feature"].str.replace("fam_", "").str.replace("src_", "source: ")
        .str.replace("tfidf_", "word: ").str.replace("_", " "))
fig, ax = plt.subplots(figsize=(8.5, 6))
ax.barh(nice, fi["importance"], color=GREEN, zorder=3)
ax.set_xlabel("Gain importance"); ax.grid(axis="x", alpha=0.3); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
save(fig, "slide_feature_importance.png")
print("charts done")
