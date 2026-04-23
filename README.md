# CAP_3764 Spring 2026 - Team 3 Project

Financial news sentiment analysis for AAPL next-day direction prediction.

## Project Overview

This project combines financial news sentiment and market price data to model whether Apple (AAPL) will close up or down the next trading day. The repository includes data preparation notebooks, model development, reusable feature/evaluation utilities, and a deployable FastAPI + Streamlit app.

## Repository Structure

```text
CAP_3764_2026_Spring_Team_3/
├── data/
│   ├── raw/
│   │   ├── articles.csv
│   │   ├── prices.csv
│   │   └── final_dataset.csv
│   └── processed/
│       ├── train_dataset.csv
│       └── test_dataset.csv
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_sentiment_analysis.ipynb
│   ├── 03_analysis.ipynb
│   ├── 04_models.ipynb
│   └── figures/
├── src/
│   ├── features/
│   │   └── engineering.py
│   └── evaluation/
│       └── metrics.py
├── deployment/
│   ├── fast_api.py
│   ├── streamlit.py
│   └── models/
│       ├── logistic_regression.pkl
│       ├── xgboost.pkl
│       └── metadata.json
├── sentiment-nlp-env.yml
├── LICENSE
└── README.md
```

## Notebook Order

Run notebooks in this order:

1. `notebooks/01_data_collection.ipynb`
2. `notebooks/02_sentiment_analysis.ipynb`
3. `notebooks/03_analysis.ipynb`
4. `notebooks/04_models.ipynb`

## Environment Setup

```bash
git clone <repository-url>
cd CAP_3764_2026_Spring_Team_3
conda env create -f sentiment-nlp-env.yml
conda activate cap_3764_env
jupyter notebook
```

## Data Files

- `data/raw/articles.csv`: collected financial news articles.
- `data/raw/prices.csv`: historical AAPL OHLCV market data.
- `data/raw/final_dataset.csv`: merged modeling dataset used by training/inference scripts.
- `data/processed/train_dataset.csv`: training split.
- `data/processed/test_dataset.csv`: test split.

## Deployment (FastAPI + Streamlit)

From `deployment/`, train model artifacts (one-time):

```bash
python fast_api.py --train --data ../data/raw/final_dataset.csv
```

Run API:

```bash
uvicorn fast_api:app --host 0.0.0.0 --port 8000 --reload
```

Run Streamlit (separate terminal):

```bash
streamlit run streamlit.py --server.port 8501
```

If API URL differs from local default:

```bash
export API_URL="http://<your-api-host>:8000"
streamlit run streamlit.py
```

## Key API Endpoints

- `GET /`
- `GET /health`
- `GET /features`
- `POST /predict`
- `POST /predict_batch`
- `GET /feature_importance`
- `GET /price_context`

## Contributors

CAP 3764 Spring 2026 Team 3

## License

This project is licensed under the terms in `LICENSE`.
