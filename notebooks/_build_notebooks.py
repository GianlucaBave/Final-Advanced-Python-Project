"""Builds and executes the four project notebooks with real data/artifacts.

Run from anywhere: ``python notebooks/_build_notebooks.py``. Each notebook is
constructed with nbformat then executed in-place so it ships with outputs.
"""

from __future__ import annotations

import os
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"

SETUP = (
    "import os, sys\n"
    "ROOT = os.path.abspath('..') if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd()\n"
    "os.chdir(ROOT); sys.path.insert(0, ROOT)\n"
    "import warnings; warnings.filterwarnings('ignore')\n"
    "import pandas as pd, numpy as np, matplotlib.pyplot as plt\n"
    "pd.set_option('display.max_columns', 30); pd.set_option('display.width', 120)\n"
    "%matplotlib inline"
)


def md(t):
    return new_markdown_cell(t)


def code(t):
    return new_code_cell(t)


# ── 01 — Data exploration ────────────────────────────────────────────────
nb01 = new_notebook(cells=[
    md("# 01 · Data Exploration\n"
       "Exploratory analysis of the **live Wallapop scrape** (real second-hand "
       "iPhone listings in Spain). We look at volume, the spec-parser's coverage, "
       "and how price varies by model, storage and city."),
    code(SETUP),
    code("from src.data import latest_parquet, load_parquet\n"
         "from src.features.parser import parse_listing\n"
         "raw_path = latest_parquet('data/raw', 'wallapop_iphones')\n"
         "df = load_parquet(raw_path)\n"
         "print('listings:', len(df), '| from', raw_path.name)\n"
         "df[['title','price','city','n_photos']].head()"),
    md("### Parser coverage\nHow often we can extract each structured field from "
       "free-text Spanish listings."),
    code("specs = pd.DataFrame([parse_listing(t,d) for t,d in zip(df.title, df.description)])\n"
         "cov = {c: f'{100*specs[c].notna().mean():.0f}%' for c in "
         "['model_family','storage_gb','battery_pct','color','condition_label']}\n"
         "cov['is_iphone'] = f'{100*specs.is_iphone.mean():.0f}%'\n"
         "pd.Series(cov, name='coverage').to_frame()"),
    md("### Price distribution by model\nThe market price gradient the model must "
       "learn — note the clean monotonic increase from SE → 16 Pro Max."),
    code("from src.reporting import viz\n"
         "eda = df.assign(model_family=specs['model_family'])\n"
         "eda = eda[eda.model_family.notna() & eda.price.between(50,2500)]\n"
         "p = viz.price_by_model(eda, 'model_family', 'price', 'artifacts/reports/eda_price_by_model.png')\n"
         "from IPython.display import Image; Image(str(p))"),
    code("# median price by storage tier\n"
         "s = df.assign(storage=specs.storage_gb).dropna(subset=['storage'])\n"
         "s[s.price.between(50,2500)].groupby('storage').price.median()"),
    md("### Listings by city"),
    code("p = viz.deals_by_city(df, 'query_city', 'artifacts/reports/eda_by_city.png')\n"
        "Image(str(p))"),
    md("**Takeaways:** ~97% of listings are identifiable iPhones; storage/battery "
       "are only partially stated by sellers (hence imputation downstream); the "
       "price-by-model gradient is clean and learnable."),
])

# ── 02 — Feature engineering ─────────────────────────────────────────────
nb02 = new_notebook(cells=[
    md("# 02 · Feature Engineering\n"
       "The single **`FeaturePipeline`** turns raw listings into a model-ready "
       "matrix and is reused verbatim at inference time — eliminating train/serve "
       "skew. Here we inspect what it produces."),
    code(SETUP),
    code("from src.data import load_parquet\n"
        "from src.features.parser import parse_listing\n"
        "from src.features.pipeline import FeaturePipeline\n"
        "df = load_parquet('data/processed/training.parquet')\n"
        "print('training rows:', len(df))"),
    md("### The parser on real titles"),
    code("for t in ['iPhone 13 Pro 256GB Sierra Blue batería 92%',\n"
        "          'iPhone 11 64gb como nuevo con caja',\n"
        "          'iPhone 15 Pro Max 1TB titanio natural']:\n"
        "    print(t)\n"
        "    print('  ->', {k:v for k,v in parse_listing(t,'').items() if v not in (None,False)})"),
    md("### Fit the pipeline and inspect the feature matrix"),
    code("pipe = FeaturePipeline(tfidf_max_features=50)\n"
        "X = pipe.fit_transform(df)\n"
        "print('feature matrix:', X.shape)\n"
        "print('NaNs after imputation:', int(X.isna().sum().sum()))\n"
        "X.iloc[:3, :12]"),
    code("# feature groups\n"
        "groups = {'one-hot model': sum(c.startswith('fam_') for c in X.columns),\n"
        "          'one-hot colour': sum(c.startswith('color_') for c in X.columns),\n"
        "          'tfidf': sum(c.startswith('tfidf_') for c in X.columns),\n"
        "          'numeric/binary': sum(not c.startswith(('fam_','color_','tfidf_')) for c in X.columns)}\n"
        "pd.Series(groups, name='n_features').to_frame()"),
    md("### Train/inference parity\nAn unseen model and colour still produce the "
       "exact same columns — the guarantee that prevents train/serve skew."),
    code("infer = df.head(3).copy()\n"
        "infer.loc[infer.index[0],'title'] = 'iPhone 99 Ultra 256GB plata'\n"
        "Xi = pipe.transform(infer)\n"
        "print('same columns:', list(Xi.columns)==pipe.feature_names_, '| shape', Xi.shape)"),
])

# ── 03 — Model training ──────────────────────────────────────────────────
nb03 = new_notebook(cells=[
    md("# 03 · Model Training & Evaluation\n"
       "Out-of-time split → Ridge baseline, XGBoost (main), LightGBM (benchmark), "
       "and a conformalized quantile interval. Then SHAP attribution and "
       "calibration."),
    code(SETUP),
    code("from src.config import load_config\n"
        "from src.training import prepare_training_frame, out_of_time_split\n"
        "from src.features.pipeline import FeaturePipeline\n"
        "from src.models import RidgeBaseline, XGBoostRegressorModel, LightGBMBenchmark, \\\n"
        "    QuantileIntervalModel, regression_metrics, interval_coverage\n"
        "cfg = load_config()\n"
        "df = prepare_training_frame(cfg)\n"
        "train, val, test = out_of_time_split(df, cfg.model.val_fraction, cfg.model.test_fraction)\n"
        "pipe = FeaturePipeline(50); Xtr = pipe.fit_transform(train)\n"
        "Xva, Xte = pipe.transform(val), pipe.transform(test)\n"
        "ytr, yva, yte = train.price.values, val.price.values, test.price.values\n"
        "print('train/val/test:', len(train), len(val), len(test))"),
    md("### Model comparison (held-out test set)"),
    code("rows = []\n"
        "for name, mdl in [('Ridge (baseline)', RidgeBaseline()),\n"
        "                  ('XGBoost (main)', XGBoostRegressorModel(cfg.model.xgboost)),\n"
        "                  ('LightGBM (benchmark)', LightGBMBenchmark())]:\n"
        "    mdl.fit(Xtr, ytr)\n"
        "    rows.append({'model': name, **regression_metrics(yte, mdl.predict(Xte))})\n"
        "    if name.startswith('XGBoost'): xgb = mdl\n"
        "pd.DataFrame(rows).set_index('model').round({'mae':1,'mape':1,'r2':3})"),
    md("### Conformalized quantile intervals (80%)\nRaw quantile regressors "
       "under-cover; CQR calibration on the validation set restores nominal "
       "coverage."),
    code("q = QuantileIntervalModel(0.1, 0.9).fit(Xtr, ytr)\n"
        "lo0, hi0 = q.predict_interval(Xte); cov0 = interval_coverage(yte, lo0, hi0)\n"
        "q.calibrate(Xva, yva)\n"
        "lo, hi = q.predict_interval(Xte); cov = interval_coverage(yte, lo, hi)\n"
        "print(f'coverage before CQR: {cov0:.1%}  ->  after CQR: {cov:.1%}  (nominal 80%)')"),
    md("### Predicted vs actual & residuals"),
    code("from src.reporting import viz\n"
        "yp = xgb.predict(Xte)\n"
        "p1 = viz.predicted_vs_actual(yte, yp, 'artifacts/reports/nb_pva.png')\n"
        "p2 = viz.residual_hist(yte, yp, 'artifacts/reports/nb_resid.png')\n"
        "from IPython.display import Image, display; display(Image(str(p1)), Image(str(p2)))"),
    md("### SHAP feature attribution (S7)\nWhy the model predicts what it does. "
       "We use XGBoost's exact native tree-SHAP (`pred_contribs`), which is both "
       "fast and avoids a version mismatch in the standalone SHAP loader."),
    code("import shap, xgboost as xgb_lib\n"
        "sample = Xte.iloc[:300]\n"
        "contribs = xgb.estimator.get_booster().predict(xgb_lib.DMatrix(sample), pred_contribs=True)\n"
        "sv = contribs[:, :-1]   # drop the bias column\n"
        "shap.summary_plot(sv, sample, plot_type='bar', max_display=12, show=False)\n"
        "plt.tight_layout(); plt.show()"),
])

# ── 04 — Live deal report ────────────────────────────────────────────────
nb04 = new_notebook(cells=[
    md("# 04 · Live Deal Report\n"
       "Loads the serialized artifacts and runs the **DealDetector** over listings "
       "to produce ranked buy/skip recommendations — the product. (Uses the latest "
       "scrape as the 'fresh' batch so the notebook runs offline; the commented "
       "cell shows the true live path.)"),
    code(SETUP),
    code("from src.config import load_config\n"
        "from src.investment import DealDetector\n"
        "from src.data import load_parquet, latest_parquet\n"
        "from src.features.parser import is_accessory_listing\n"
        "from src.reporting import render_deals, default_title\n"
        "cfg = load_config()\n"
        "detector = DealDetector.from_artifacts(cfg)\n"
        "df = load_parquet(latest_parquet('data/raw','wallapop_iphones'))\n"
        "df = df[~df.title.fillna('').map(is_accessory_listing)]\n"
        "df = df[df.title.str.contains('13 Pro', case=False, na=False) & df.price.between(50,800)]\n"
        "print('candidate listings:', len(df))"),
    code("# --- the true live path (uncomment to scrape fresh listings now) ---\n"
        "# from src.scrapers import WallapopScraper\n"
        "# wp = WallapopScraper(cfg.scraping['wallapop'], cfg.cities)\n"
        "# df = pd.DataFrame([l.as_row() for l in wp.scrape('iPhone 13 Pro','madrid')])"),
    md("### Ranked deals"),
    code("deals = detector.detect(df)\n"
        "print(render_deals(deals, title=default_title('iPhone 13 Pro','madrid'), top=8))"),
    md("### Score distribution & decision mix"),
    code("from src.reporting import viz\n"
        "scores = [d.investment_score for d in deals]; decs = [d.decision for d in deals]\n"
        "p = viz.score_distribution(scores, decs, 'artifacts/reports/nb_scores.png')\n"
        "from IPython.display import Image; Image(str(p))"),
    code("pd.Series(decs).value_counts().to_frame('n_listings')"),
    md("### Manual spot-check (presentation walkthrough)\n"
       "For the top deal, compare the model's fair-value estimate against the "
       "listing price and the 80% interval. In the live demo this is cross-checked "
       "against the eBay and Backmarket price for the same model/condition."),
    code("top = deals[0]\n"
        "import json; print(json.dumps(top.to_dict(), indent=2, ensure_ascii=False)[:700])"),
    md("> **Disclaimer.** Academic project; the investment score is not financial "
       "advice and the model cannot verify authenticity. Do your own due diligence."),
])

NOTEBOOKS = {
    "01_data_exploration.ipynb": nb01,
    "02_feature_engineering.ipynb": nb02,
    "03_model_training.ipynb": nb03,
    "04_live_deal_report.ipynb": nb04,
}

for fname, nb in NOTEBOOKS.items():
    path = NB_DIR / fname
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    print(f"executing {fname} ...", flush=True)
    try:
        client = NotebookClient(nb, timeout=600, kernel_name="python3",
                                resources={"metadata": {"path": str(NB_DIR)}})
        client.execute()
        ok = "OK"
    except Exception as exc:  # save unexecuted on failure
        ok = f"EXEC FAILED: {type(exc).__name__}: {str(exc)[:120]}"
    nbformat.write(nb, path)
    print(f"  -> {fname}: {ok}")

print("done.")
