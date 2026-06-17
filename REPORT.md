# iPhone Deal-Finder — Technical Report

**Course:** Advanced Python · ESADE — Final Project
**System:** an end-to-end machine-learning pipeline that collects second-hand
iPhone listings, predicts each phone's fair market value, quantifies the
uncertainty of that prediction, and converts it into an actionable
**buy / hold / skip** recommendation.

This report documents the full build: what was done at every stage, **why** it
was done that way, and **how** it was implemented. It is meant to be read on its
own by someone who has never seen the code.

---

## 1. Objective and approach

The goal is to answer a single practical question at scale:

> *"Among the iPhones currently for sale, which ones are priced below what they
> are actually worth — and which of those are safe to buy?"*

To answer it, the system needs three capabilities, and the project is organised
around them:

1. **Know the market** — collect a large, current dataset of real iPhone prices.
2. **Estimate fair value** — train a regression model that predicts the typical
   market price of a phone from its specifications, and express how confident it
   is in each prediction.
3. **Decide** — combine predicted upside, confidence, and risk into one score
   and a clear recommendation.

The work is therefore a supervised **regression** problem (predict a continuous
price) wrapped in an **uncertainty-quantification** layer (prediction intervals)
and a **decision** layer (the investment score). The remainder of the report
follows the data as it flows through the pipeline:

```
collection → cleaning/validation → feature engineering → modelling
          → evaluation → uncertainty calibration → scoring → reporting
```

A guiding design principle throughout is **train/serve parity**: every
transformation applied while training the model is applied identically at
prediction time, because the two paths share one serialized object. This
eliminates the single most common production-ML bug, where the model is fed
subtly different features in production than it saw in training.

---

## 2. Data collection

### 2.1 Why multiple sources

A model is only as good as the price signal it learns from. A single snapshot of
one marketplace would be narrow and fragile, so the system pulls from **four
independent sources** and merges them into one training corpus. Breadth across
marketplaces, regions and time makes the learned notion of "fair value" more
robust and lets the model see a wider range of models, storage tiers and
conditions than any one source contains.

| Source | Records | Region / era | How collected |
|---|---|---|---|
| Wallapop | 8,262 | Spain · current | Live scrape of the site's JSON search backend |
| eBay (sold) | 957 | US · recent | Live scrape of completed-listing HTML |
| eBay archive | 2,165 | US · 2023 | Ingested from exported CSV |
| US e-commerce | 2,361 | US · 2026 | Ingested from a structured CSV |
| **Total** | **13,745** | | merged into one corpus |

### 2.2 How the live scraping works

The primary source, **Wallapop**, is collected live. The site is a JavaScript
application that internally calls a JSON API; the scraper calls that same
endpoint directly, which returns clean structured data and avoids the fragility
of parsing rendered HTML. Engineering choices that make the scraper robust:

- **A realistic browser identity** (User-Agent and the headers the site's own
  front-end sends) so requests are accepted.
- **Polite rate-limiting** (one request per second) and **exponential backoff**
  on transient failures (HTTP 429/5xx), so the scraper neither hammers the site
  nor gives up on a temporary hiccup.
- **Geographic sweep**: the search is location-anchored, so the scraper iterates
  over 20 iPhone models × 6 Spanish cities to maximise coverage and variety.
- **Failure isolation**: each page is fetched inside its own `try/except`; one
  bad page is logged and skipped rather than aborting the whole run, and a
  warning is raised if more than 10% of queries come back empty.

The **eBay** scraper faces aggressive bot protection that fingerprints the
network handshake and blocks automated clients. It is defeated with a browser-
TLS-impersonation transport plus a "session-warming" step (loading the homepage
first to obtain the cookies a real browser would have). Two further data files
were **ingested from CSV** through a dedicated, documented loader rather than
scraped, which is the normal way to bring an external dataset into a pipeline.

All collected records are normalised into **one common schema** (`Listing`) and
written to **Parquet** files, timestamped by collection date for
reproducibility. Parquet was chosen over CSV because it is columnar, typed and
compressed — in benchmarks it loads ~6× faster and uses ~3× less disk.

---

## 3. Data cleaning and validation

Raw marketplace data is noisy: typos, mis-priced items, non-phone listings,
mixed currencies. Cleaning happens in two layers.

### 3.1 Schema and range validation

Every collected batch passes through a validator whose philosophy is
**"hard-fail only on structural problems, soft-drop everything else."** It:

- **raises** only if required columns are missing (continuing would be
  meaningless);
- **drops with a logged warning** any row that violates a sanity rule — a
  non-numeric or out-of-range price (≤ €0 or > €5,000), a too-short title, a
  missing identifier, or a non-EUR currency;
- raises a soft warning if an unusually large fraction (>25%) of a batch is
  dropped, which is an early signal that a source's format has changed.

This design means a single malformed listing can never abort a 13,000-row run,
while genuine data-quality problems still surface in the logs.

### 3.2 Currency normalisation

The non-Spanish sources are priced in USD; all prices are converted to **EUR**
with a configured exchange rate so that every record lives on one comparable
scale before modelling.

### 3.3 Removing non-phones and bad rows

The biggest source of noise is listings that are not actually a sellable phone —
cases, screens, chargers, spare parts, and "wanted"/"swap" posts. These are
filtered with a **title-based detector** that recognises accessory and
spare-part vocabulary (in both Spanish and English) and part-noun-led titles.
This step alone removed ~2,000 rows. A final filter keeps only rows where:

- a real iPhone model could be identified from the text,
- the price sits in a plausible band for a phone (€50–€2,500), and
- the listing is not an accessory/part.

Of 13,745 merged records, **10,166 survived cleaning** and became the modelling
corpus. Within-source **deduplication** on the listing identifier removes
re-scrapes of the same item.

### 3.4 Source weighting

When the four sources are merged, each row carries a **`sample_weight`**
proportional to how relevant it is to the deployment market. Because the product
ultimately values Spanish listings, the Spanish source dominates the weighting
while the others contribute supporting signal. Weights are normalised to a mean
of 1 so they change the *balance* of the data without changing the effective
learning rate.

---

## 4. Feature engineering

A model cannot read "iPhone 13 Pro 256GB Sierra Blue batería 92% como nuevo"; it
needs numbers. Turning free Spanish (and English) text into a clean numeric
matrix is the heart of the project. All of this is done by a single
`FeaturePipeline` class so that training and inference are guaranteed identical.

### 4.1 Parsing specifications from text

A deterministic parser extracts structured fields from each title and
description using a layered strategy chosen to match each field's nature:

- **Regex** for high-precision fields: model family (iPhone 11–16, Pro / Pro Max
  / Plus / mini variants), storage (64 GB … 1 TB, including unit-less mentions
  like "…Pro 256"), and battery health (several Spanish/English phrasings).
- **Fuzzy matching** for colour, which sellers spell many ways (e.g. "azul
  sierra", "sierra blue", "grafito"), mapped to a small set of canonical colours.
- **Keyword scoring** for condition, turning phrases like "para piezas",
  "usado", "buen estado", "como nuevo", "precintado" into an ordinal 0–4 scale.

On the real data the parser identifies the model in ~97% of listings, which is
the foundation everything else builds on. Because the parser is pure and
deterministic, it is the most heavily unit-tested component (its tests caught two
real bugs during development).

### 4.2 Engineered and contextual features

Beyond the parsed specs, the pipeline derives features that carry price signal:

- **`model_year`** — looked up from the model family (a dominant price driver).
- **`days_since_posted`** — listing age from the posting timestamp, clamped to a
  sane range; captures time-on-market effects.
- **Description signals** — description length and a **TF-IDF** vectorisation of
  the top 50 description tokens, which lets wording ("tienda", "garantía",
  "libre") inform the price.
- **Seller and listing signals** — number of photos, seller activity, shipping
  availability, refurbished/warranty flags.
- **Geography** — a city "tier" ordinal.
- **Market / source indicators** — a binary "is this the US market?" feature and
  a one-hot of the source. These are crucial: when several marketplaces are
  merged, they let one model learn a separate price *level* per market instead
  of averaging incompatible price scales together. They are known at prediction
  time (a live scan is always the deployment market), so they are not leaky.

### 4.3 Encoding and imputation

- **Categoricals** (model family, colour, source) are one-hot encoded; tree
  models need no scaling, so numeric features pass through untouched.
- **Missing values are imputed inside the pipeline**: missing battery and storage
  are filled with the median *for that model family* (a far better guess than a
  global median), with a global fallback; a `has_battery_info` flag lets the
  model know when a value was imputed rather than observed.

The fitted encoders, vocabularies, medians and final column order are all stored
**inside** the serialized pipeline object. At inference an unseen model or colour
still produces the exact same columns in the same order — the train/serve parity
guarantee. The final matrix has ~105 features.

---

## 5. Modelling

### 5.1 The honest train/test split

Price data is a time series: prices drift, new models launch, old ones
depreciate. A random train/test split would let the model "see the future"
(train on tomorrow's prices, test on yesterday's) and report optimistic,
dishonest accuracy. Instead the data is sorted chronologically and split
**out-of-time**: the oldest 70% trains, the next 15% validates, and the most
recent 15% is the held-out test set. This mirrors how the model is actually used
— trained on the past, asked about the present.

### 5.2 Models, from simple to strong

Three models are trained on the identical feature matrix so they can be compared
fairly:

1. **Ridge regression (baseline).** A regularised linear model that sets the
   sanity floor and provides interpretability. Any complex model must clearly
   beat it to justify itself. Features are standardised first (linear models are
   scale-sensitive).
2. **XGBoost (main).** A gradient-boosted tree ensemble — the workhorse for
   tabular data, able to capture non-linear interactions (e.g. how condition
   matters more for newer models). It supports an optional hyperparameter search.
3. **LightGBM (benchmark).** A second gradient-boosting implementation, included
   to confirm the result is not specific to one library.

All three predict **log-price** rather than raw price. Prices are right-skewed
and strictly positive; modelling on the log scale stabilises variance and makes
the error behave more like a *percentage* error, which is what matters when phones
range from €60 to €1,500. Predictions are exponentiated back at the end.

### 5.3 Hyperparameter tuning

The main model can be tuned with a grid search. Critically, the cross-validation
folds are **time-aware** (`TimeSeriesSplit`), not random — tuning honours the
same no-future-leakage discipline as the final evaluation. (A genuine finding
during the project: time-aware tuning did not beat well-chosen defaults on the
out-of-time test, so the defaults were kept — documented rather than hidden.)

### 5.4 Uncertainty: prediction intervals

A single predicted number is not enough to recommend a purchase; the system must
know *how sure* it is. Two additional gradient-boosted models are trained with
the **quantile (pinball) loss** to predict the 10th and 90th percentiles, giving
an 80% prediction interval around each estimate. A wide interval (rare
configuration, sparse data) signals low confidence.

Raw quantile models are known to **under-cover** (their nominal-80% intervals
contain the truth less than 80% of the time). To fix this honestly, the intervals
are **calibrated with Conformalized Quantile Regression (CQR)**: on the
validation set the system measures how much the intervals need to widen to reach
the target coverage, and applies that correction. This is a principled,
finite-sample method that turns optimistic intervals into trustworthy ones.

---

## 6. Evaluation

All metrics are computed on the **held-out, out-of-time test set** the model
never saw during training or tuning.

| Model | MAE | MAPE | R² |
|---|---|---|---|
| Ridge (baseline) | €79.2 | 24.3% | 0.581 |
| XGBoost | €77.9 | 23.5% | 0.593 |
| **LightGBM (best)** | **€72.4** | **22.4%** | **0.642** |

- **MAE** (mean absolute error in euros) and **MAPE** (percentage error) measure
  typical prediction error; **R²** measures how much of the price variation the
  model explains.
- The gradient-boosted models clearly beat the linear baseline, confirming that
  the non-linear structure is real and worth modelling.
- The dominant learned price drivers — `model_year`, model family, and condition
  — match real market economics, which is a strong sanity check that the model
  learned signal rather than noise.

Diagnostic plots are regenerated on every training run: predicted-vs-actual
(points cluster along the ideal line), a residual histogram (errors centred on
zero), a feature-importance chart, and a calibration plot. Feature attribution
is computed with tree-SHAP for interpretability.

These error levels are honest for this problem: market prices for an identical
phone genuinely vary (negotiable vs firm vs urgent sellers), so there is an
irreducible floor of noise no model can remove.

---

## 7. From prediction to decision: the investment score

The model's fair-value estimate and interval feed a transparent scoring layer
that produces the user-facing recommendation. Three components are combined:

- **Expected margin** — the predicted resale value minus the asking price minus
  fees, as a fraction of the asking price. Passed through a sigmoid so enormous
  or negative margins saturate instead of dominating.
- **Confidence** — derived from the width of the calibrated 80% interval
  relative to the prediction; tight interval → high confidence.
- **Risk penalty** — a set of heuristics that flag listings which look unsafe
  regardless of price: a price implausibly far below fair value (a classic scam
  signal), missing battery information, too few photos, damaged / "for-parts"
  wording, and other suspicious patterns.

These combine into a single score in [0, 1]:

```
investment_score = 0.5 · sigmoid(margin · 5) + 0.3 · confidence + 0.2 · (1 − risk)
```

which maps to a decision: **≥ 0.75 STRONG BUY · 0.55–0.75 BUY · 0.35–0.55 HOLD ·
< 0.35 SKIP**. The weights live in configuration so the product's behaviour can be
tuned without touching code. The result is not a black box: every recommendation
comes with the fair-value estimate, the expected margin, the confidence band and
the specific risk flags that produced it.

---

## 8. Engineering quality

The project is built as a real, maintainable software package, not a single
notebook:

- **Object-oriented design** — abstract base classes for scrapers and models let
  new sources/algorithms drop in without changing callers; lightweight
  `__slots__` data containers keep records explicit and memory-efficient.
- **One pipeline for train and serve** — the central correctness guarantee
  described throughout.
- **Configuration as code** — all paths, parameters and thresholds live in a
  single YAML file loaded into a typed object, so experiments change one file.
- **Structured logging** — every module logs to the console and to a per-run log
  file, making runs auditable.
- **A custom exception hierarchy** — errors are raised at the right boundary so
  failures are localised and catchable at the right granularity.
- **Testing** — 36 automated tests cover the parser (dozens of real titles), the
  feature pipeline (including train/serve column parity and save/load
  round-trips), and the scorer's edge cases (great deal, overpriced, scam,
  damaged, all-zero inputs).
- **Performance** — Parquet I/O, vectorised pandas, and a single batched
  feature-transform let the system score ~1,670 listings/second, keeping a full
  scan well under a minute.
- **Reproducible artefacts** — training serialises the model, the quantile pair,
  the feature pipeline and a JSON metrics report; inference loads exactly these.

---

## 9. Challenges and how they were solved

| Challenge | How it was diagnosed | Fix |
|---|---|---|
| Prediction intervals were too narrow (under-covered) | measured empirical coverage vs nominal 80% | Conformalized Quantile Regression calibration |
| Target sites blocked automated requests (HTTP 403) | inspected responses; saw bot-protection stubs | browser-TLS impersonation + session warming |
| Public sold-search results are capped | per-query counts plateaued | narrower per-model/storage queries to widen coverage |
| Merging multiple markets *lowered* accuracy | error rose after adding cross-market data | added market & source indicator features and reweighted toward the deployment market so one model serves multiple price levels |
| The interpretability library crashed on the model version | traced the serialization error | used the model's native tree-SHAP implementation |
| Accessories and spare parts polluted the data | manual inspection of cheap "phones" | title-based junk filter + a damaged-device risk flag |

Each fix is a deliberate, documented decision rather than a workaround, and
several (the calibration, the market features, the time-aware split) materially
improved the system's honesty or accuracy.

---

## 10. Conclusion and future work

The project delivers a working, end-to-end machine-learning system: it collects
real iPhone price data at scale, engineers a rich feature set from messy text
through a single train/serve-consistent pipeline, trains and compares
regression models with an honest out-of-time protocol, calibrates its own
uncertainty, and turns predictions into transparent, explainable buy/skip
recommendations — fast enough to run interactively.

Natural extensions, in rough priority order:

1. **Photo-based authenticity detection** — a convolutional model on listing
   images to flag probable counterfeits, the main risk the current heuristics
   cannot fully catch.
2. **Time-to-sell prediction** — adding a liquidity dimension (how quickly a deal
   will likely sell) to the score.
3. **Longitudinal data collection** — scheduled repeated scrapes to build a true
   panel dataset and observe price movements over time.
4. **Productionisation** — a small web service for on-demand scoring, continuous
   integration on the test suite, and automated retraining.

---

*Academic project. The investment score is decision support, not financial
advice; the model cannot verify authenticity. Users should perform their own due
diligence before any purchase.*
