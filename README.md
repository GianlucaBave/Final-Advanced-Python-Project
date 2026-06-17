# iPhone Deal-Finder

> Predicts the **fair market value** of any second-hand iPhone listing on
> Wallapop, computes an **investment score**, and outputs a **buy / hold / skip**
> decision with the reasoning behind it — on **live data**, in seconds.

Advanced Python final project · ESADE. Built end-to-end on **13,745 real iPhone
price records** from **4 sources** — 8,262 scraped live from Wallapop Spain, plus
eBay and two US resale datasets.

```
$ python -m src.cli scan --model "iPhone 13 Pro" --city madrid --max-price 700

╔══════════════════════════════════════════════════════════════════╗
║   TOP DEALS — iPhone 13 Pro — Madrid — 2026-06-17                 ║
╠══════════════════════════════════════════════════════════════════╣
║  #1 🟢 €299 — iPhone 13 Pro |TIENDA |GARANTIA                     ║
║      Fair: €409 [300–403]                                         ║
║      Margin: +31.0%   Conf: MEDIUM   Score: 0.76  → STRONG BUY    ║
║      Risk flags: none                                             ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## What it does

A user (or a scheduled script) asks *"among all iPhones for sale on Wallapop in
Madrid, which are the best deals to buy and resell?"* and gets a ranked answer:

1. **Scrape** live listings → 2. **parse** specs from free Spanish text →
3. **predict** fair value with a gradient-boosted model →
4. **quantify uncertainty** with calibrated prediction intervals →
5. **score** each listing (margin × confidence × low-risk) →
6. **rank** and explain the decision.

The **investment score** ∈ [0, 1] combines:

| Component | Weight | Definition |
|---|---|---|
| Expected margin | 0.50 | `(predicted_resale − price − fees) / price`, sigmoid-squashed |
| Prediction confidence | 0.30 | tightness of the 80% prediction interval |
| Risk penalty | 0.20 | scam / counterfeit / low-information heuristics |

---

## Results (held-out, out-of-time test set)

| Model | MAE | MAPE | R² |
|---|---|---|---|
| Ridge (baseline) | €79.2 | 24.3% | 0.581 |
| **XGBoost (production)** | **€77.9** | **23.5%** | **0.593** |
| LightGBM (benchmark) | €72.4 | 22.4% | 0.642 |

- Trained on the **combined 4-source dataset** (10,166 usable rows, 105 features).
- **Quantile interval coverage ≈ 69%** (nominal 80%) — CQR calibrates on the
  validation split; the multi-market test set is more heterogeneous.
- Dominant price drivers — `model_year`, model family, condition — match real
  market economics.

Plots are regenerated on every `train` run into `artifacts/reports/`
(predicted-vs-actual, residuals, feature importance, calibration, price-by-model,
listings-by-city).

---

## Data strategy (and an honest note on sources)

| Source | Rows | Market / era | Status |
|---|---|---|---|
| **Wallapop** | 8,262 | ES · Jun 2026 | ✅ **live scrape** — JSON backend, no auth |
| **eBay** | 957 | US · 2026 | ✅ live scrape (curl_cffi TLS impersonation + cookie warming) |
| **eBay archive-5** | 2,165 | US · Nov 2023 | ✅ CSV ingested (`src/data/external.py`) |
| **ecommerce USA** | 2,361 | US · 2025–26 | ✅ CSV ingested |
| **TOTAL** | **13,745** | | merged with quality-weighted `sample_weight` |

> Wallapop (Spain) is the deployment target market and dominates training; the US
> sources add cross-market and historical breadth. Backmarket (JSON-LD) is
> implemented but bot-blocked from cloud IPs. All prices normalized to EUR.

The official eBay *Marketplace Insights* API is a **gated, manually-approved**
product — not the "free 10-minute" path it's often assumed to be. So instead of an
API we scrape eBay's public search HTML directly. eBay and Backmarket front those
pages with Akamai/PerimeterX bot protection that fingerprints the TLS handshake;
we defeat it with `curl_cffi` (Chrome/Safari JA3 impersonation) + a cookie-warming
first request. It works from a normal network but is rate-limited from cloud IPs.

The system ships as a **multi-source model** with quality-weighted
`sample_weight`, and the merge is data-driven: drop an `ebay_*.parquet` /
`backmarket_reference_*.parquet` into `data/raw/` and `train` picks it up
automatically.

---

## Install & run

```bash
pip install -r requirements.txt          # Python 3.11+

python -m src.cli scrape                  # live Wallapop sweep → data/raw/*.parquet
python -m src.cli scrape --sources ebay,backmarket   # try the anchor sources
python -m src.cli train                   # train all models → artifacts/
python -m src.cli train --tune            # + time-aware GridSearchCV
python -m src.cli scan --model "iPhone 13 Pro" --city madrid --max-price 700
python -m src.cli evaluate                # print the latest training report

pytest -q                                 # 36 tests
```

Notebooks in `notebooks/` (executed, with outputs):
`01_data_exploration` · `02_feature_engineering` · `03_model_training` ·
`04_live_deal_report`. Regenerate with `python notebooks/_build_notebooks.py`.

---

## Architecture

```
scrapers/ ─▶ data/ ─▶ features/ ─▶ models/ ─▶ investment/ ─▶ reporting/
 Wallapop    validate  FeaturePipeline  Ridge      scorer       CLI table
 eBay        dedup     (one fit object  XGBoost    risk         matplotlib
 Backmarket  parquet    train≡serve)    quantile   detector     JSON report
```

The **single `FeaturePipeline`** object is fitted once and serialized; inference
loads the same object, so the feature matrix is guaranteed identical at train and
serve time — no train/serve skew (the most common production-ML bug).

```
src/
├── config.py          # typed dataclass over settings.yaml
├── logging_setup.py    # console + per-run file logging
├── exceptions.py       # custom hierarchy
├── schema.py           # Listing / Deal containers (__slots__)
├── scrapers/           # BaseScraper + Wallapop/eBay/Backmarket
├── data/               # loader (parquet), validator, merger
├── features/           # parser, encoder, pipeline
├── models/             # base, baseline, xgboost, quantile, metrics
├── investment/         # risk, scorer, detector
├── reporting/          # viz, cli_table
├── training.py         # end-to-end training orchestration
└── cli.py              # argparse entry point
```

---

## Performance (S2)

Measured on the real 8,262-row dataset (`timeit`, 20 runs):

| Operation | Result |
|---|---|
| Parquet read | **7.8 ms** (1.67 MB) |
| CSV read | 45.5 ms (5.01 MB) |
| **Parquet vs CSV** | **5.9× faster, 3.0× smaller** |
| Deal scoring throughput | **~1,670 listings/sec** |

Parquet for all I/O, vectorized pandas, and a single batched feature-transform
keep a full-city scan well under the 60-second product target.

---

## Course module coverage

| Session | Topic | Where |
|---|---|---|
| S1 | OOP | `BaseScraper`, `BaseModel`, `FeaturePipeline`, `DealDetector`; `__slots__` on `Listing`/`Deal` |
| S2 | Performance | parquet I/O, vectorized pandas, `timeit` benchmarks above |
| S3 | Visualization | `reporting/viz.py` — 7 figure types |
| S4 | Parallel & Exceptions | custom exception hierarchy, per-page try/except, `UserWarning` on soft data issues, backoff |
| S5 | Web Scraping | Wallapop JSON, eBay sold HTML, Backmarket JSON-LD, TLS impersonation |
| S6 | Time Series | out-of-time split, `days_since_posted`, time-aware `TimeSeriesSplit` tuning |
| S7 | Supervised ML | Ridge / XGBoost / LightGBM, GridSearchCV, quantile regression + CQR, native SHAP |

NN / DL / GA / SNA are documented as realistic extensions (see plan §14).

---

## Testing

`pytest` — 36 tests across the parser (30+ real titles), the FeaturePipeline
(train/serve column parity, save/load round-trip), and the investment scorer
(perfect deal, overpriced, scam, damaged, all-zero edge cases). The tests caught
two real parser bugs during development (unit-less storage, single-digit TB).

---

## Disclaimer

Academic project. The investment score is **not financial advice**; the model
cannot verify authenticity or reliably distinguish counterfeits. Perform your own
due diligence before purchasing any second-hand product.
