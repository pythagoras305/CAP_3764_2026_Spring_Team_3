# Sentiment-Driven Trading Signal
## AAPL Next-Day Price Direction Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?style=flat-square)
![License](https://img.shields.io/badge/Course-CAP%203764-lightgrey?style=flat-square)
![Status](https://img.shields.io/badge/Status-Submission%202-green?style=flat-square)

**FIU Advanced Data Science — CAP 3764 | Team 3**

</div>

---

## The Business Case

> *A trading strategy is only as good as the edge it creates over doing nothing.*

This project builds a machine learning pipeline that uses **daily news sentiment about Apple Inc.** to predict whether AAPL stock will close higher or lower the following trading day. When deployed as a simple long-only signal — buy on predicted UP days, stay out on predicted DOWN days — the strategy generated a **24.4% cumulative return** over the Sep–Dec 2024 test period, compared to **10.3% for passive buy-and-hold**.

That 14-point outperformance over a single quarter demonstrates a compoundable edge. Applied across multiple quarters, a consistent sentiment signal could meaningfully separate active returns from benchmark performance — the core value proposition of systematic trading.

**This is a prediction system, not a causal claim.** The models identify statistical associations between sentiment patterns and next-day price direction. No assertion is made that news sentiment causes price movement.

---

## Results at a Glance

```
────────────────────────────────────────────────────────────
  BACKTEST  |  Sep – Dec 2024  |  83 Trading Days
────────────────────────────────────────────────────────────
  Strategy Return (Sentiment Signal)     +24.4%
  Benchmark Return (Buy and Hold)        +10.3%
  Outperformance                         +14.1 pp
  Days Traded                            67 / 83
────────────────────────────────────────────────────────────
```

```
────────────────────────────────────────────────────────────
  MODEL COMPARISON  |  Test Period: Sep – Dec 2024
────────────────────────────────────────────────────────────
  Model                  Accuracy   ROC-AUC   Overfit Gap
  ─────────────────────────────────────────────────────────
  XGBoost (tuned)         56.6%      0.576      0.119  (!)
  Logistic Regression     57.8%      0.510     -0.032  (ok)
  Naive Baseline          59.0%      0.500        —
────────────────────────────────────────────────────────────
```

Logistic Regression is the preferred production model — its near-zero overfitting gap indicates it learned generalizable patterns rather than memorizing training data. XGBoost shows mild overfitting but achieves the stronger AUC and drove the backtest results.

---

## Team

| Member | Role | Contributions |
|--------|------|---------------|
| Hector | Analysis & Strategy | EDA, modeling, Optuna tuning, backtesting |
| Yana   | Data Engineering    | Data pipeline, validation, processed exports |
| Kevin  | Documentation       | Repo structure, environment, README, presentation |

---

## Repository Structure

```
sentiment-trading/
│
├── data/
│   ├── raw/                          # Source CSVs — excluded from version control
│   └── processed/
│       ├── train_dataset.csv         # Jan 2023 – Aug 2024
│       └── test_dataset.csv          # Sep 2024 – Dec 2024
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   └── 02_modeling.ipynb             # Training, evaluation, backtest
│
├── src/
│   ├── __init__.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py            # Feature engineering functions
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py                # Evaluation suite and backtest
│
├── environment.yml                   # Conda environment (sentiment-trading-env)
├── .gitignore
└── README.md
```

All reusable logic lives in `src/`. Notebooks import from `src/` and contain analysis and visualizations only — no functions are defined inside notebooks.

---

## Dataset

| Property | Detail |
|----------|--------|
| Ticker | AAPL (Apple Inc.) |
| Source | Financial news sentiment API + Yahoo Finance |
| Date Range | January 2, 2023 – December 30, 2024 |
| Granularity | One row per trading day |
| Total Rows | 492 |
| Training Set | ~390 rows — Jan 2023 through Aug 2024 |
| Test Set | 83 rows — Sep through Dec 2024 |
| Target Variable | `1` if next-day close > today's close, else `0` |
| Class Balance | 56% UP / 44% DOWN |

### Feature Set

| Feature | Type | Description |
|---------|------|-------------|
| `avg_sentiment` | Sentiment | Mean daily sentiment score across all articles |
| `rolling_3d_sentiment` | Sentiment | 3-day rolling average sentiment |
| `rolling_7d_sentiment` | Sentiment | 7-day rolling average sentiment |
| `daily_return` | Price | Previous day percentage return |
| `price_momentum_5d` | Price | 5-day price momentum |
| `high_low_range` | Price | (High − Low) / Close — intraday volatility |
| `ma_5` | Price | 5-day moving average |
| `ma_10` | Price | 10-day moving average |
| `volume` | Market | Daily share volume |
| `day_of_week` | Temporal | 0 = Monday through 4 = Friday |

---

## Setup and Installation

### Prerequisites

- [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Quickstart

```bash
# Clone the repository
git clone https://github.com/<your-org>/sentiment-trading.git
cd sentiment-trading

# Create and activate the conda environment
conda env create -f environment.yml
conda activate sentiment-trading-env

# Register as a Jupyter kernel
python -m ipykernel install --user --name sentiment-trading-env

# Launch Jupyter
jupyter lab
```

### Environment

All dependencies are defined in `environment.yml` under the environment name `sentiment-trading-env`:

```
Python 3.11   pandas   numpy   scikit-learn   xgboost
optuna        shap     matplotlib   seaborn    jupyter
```

---

## How to Run

Execute notebooks in order from the repository root:

```
Step 1 — notebooks/01_eda.ipynb
         Exploratory analysis: sentiment distributions, correlation
         heatmaps, rolling averages, target balance, feature insights

Step 2 — notebooks/02_modeling.ipynb
         XGBoost + Logistic Regression training, Optuna tuning,
         evaluation metrics, ROC curves, feature importance, backtest
```

The processed CSVs in `data/processed/` are version-controlled and load directly. Raw source files are excluded from the repository via `.gitignore`.

---

## Methodology

### Validation Strategy

A strict temporal split prevents data leakage and simulates real-world deployment:

```
Training   |  Jan 2023 ──────────────── Aug 2024  |  ~390 rows
           |                                       |
Test       |                    Sep 2024 ── Dec 2024  |  83 rows
```

The model never observes test data during training. This mirrors the actual constraint a practitioner faces — a model built on historical data used to make forward-looking predictions.

### Modeling Approach

Two models were evaluated:

**XGBoost** — Gradient-boosted ensemble classifier. Hyperparameters were tuned using Optuna across 50 trials with 5-fold stratified cross-validation optimizing ROC-AUC. Class imbalance was addressed via `scale_pos_weight`.

**Logistic Regression** — Interpretable linear baseline with L2 regularization. Included to assess whether a simpler model could match or exceed the ensemble on a small dataset. It did — with a superior generalization gap.

### Backtest Logic

The backtest simulates a long-only strategy: enter a position on any day the model predicts UP, hold for one day, exit at close. Days predicted as DOWN are skipped — capital sits idle. Cumulative returns are compared against passive buy-and-hold over the same period. Transaction costs and slippage are not modeled.

---

## Limitations

**Dataset size** — 492 total rows is small for financial ML. Thin sentiment coverage (avg. 2.3 articles/day, 75 days with no coverage) introduces noise into daily sentiment aggregates.

**Test window** — The 83-day test period coincides with a strong AAPL bull run. Strategy returns may not replicate in a bearish or high-volatility regime.

**Sentiment methodology** — Scores were pre-computed by a third-party API. The underlying scoring approach is opaque and may not generalize to other data sources.

**No transaction costs** — Real-world execution involves bid-ask spreads, commissions, and slippage. Reported backtest returns are gross of these costs.

**Statistical association only** — Models learn correlations, not causes. Sentiment predicting direction does not imply sentiment drives direction.

---

## Reproducing Results

```python
# Standard import pattern used across all notebooks
from src.features.engineering import load_and_merge, temporal_split, get_feature_matrix
from src.evaluation.metrics   import evaluate_classifier, plot_roc_curve, backtest_simulation

# Load and split
df                       = load_and_merge('data/raw/final_dataset.csv')
train_df, test_df        = temporal_split(df, split_date='2024-09-01')
X_train, y_train, FEATS  = get_feature_matrix(train_df)
X_test,  y_test,  _      = get_feature_matrix(test_df)
```

---

<div align="center">

Florida International University — Department of Computer Science
Advanced Data Science CAP 3764 — Spring 2026

</div>

