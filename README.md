# CAP_3764 Spring 2026 — Team 3 Project

## Project Title

Financial News Sentiment Analysis for Market Trend Insights

---

## Project Overview

This project focuses on analyzing financial news articles and stock price data to determine how sentiment in news media may influence or correlate with market trends. By combining natural language processing (NLP) techniques with historical stock price data, we aim to extract meaningful sentiment signals from financial news and evaluate their potential predictive value in financial markets.

The project pipeline includes:

- Collecting financial news data
- Performing sentiment analysis on articles
- Merging sentiment results with historical stock price data
- Conducting exploratory data analysis (EDA)
- Visualizing relationships between sentiment and price movement

---

## Project Goals

- Perform sentiment analysis on financial news articles
- Generate sentiment scores for each article
- Analyze correlations between news sentiment and stock price movement
- Explore potential predictive insights from sentiment trends
- Visualize sentiment impact on financial performance

---

## Repository Structure

```text
CAP_3764_2026_Spring_Team_3
│
├── data
│   ├── raw
│   │   ├── articles.csv
│   │   ├── prices.csv
│   │   └── final_dataset.csv
│   └── processed
│       ├── train_dataset.csv
│       └── test_dataset.csv
│
├── notebooks
│   ├── 01_data_collection.ipynb
│   ├── 02_sentiment_analysis.ipynb
│   ├── 03_analysis.ipynb
│   ├── 04_models.ipynb
│   └── figures/
│
├── src
│   ├── features
│   │   └── engineering.py
│   └── evaluation
│       └── metrics.py
│
├── deployment
│   ├── fast_api.py
│   ├── streamlit.py
│   └── models
│       ├── logistic_regression.pkl
│       ├── xgboost.pkl
│       └── metadata.json
│
├── sentiment-nlp-env.yml
├── LICENSE
└── README.md
```

---

## Dataset Description

### 1. `data/raw/articles.csv`

Contains collected financial news articles used for sentiment analysis. Typical fields may include:

- Article title
- Publication date
- Source
- Article content

### 2. `data/raw/prices.csv`

Contains historical stock price data. Typical fields may include:

- Date
- Open price
- Close price
- High
- Low
- Volume

### 3. `data/raw/final_dataset.csv`

Final merged dataset used for feature engineering, model training, and app inference.

### 4. `data/processed/train_dataset.csv` and `data/processed/test_dataset.csv`

Processed temporal splits used for training and evaluation workflows.

---

## Environment Setup

Follow the steps below to set up the development environment.

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd CAP_3764_2026_Spring_Team_3
```

### Step 2: Create Conda Environment

```bash
conda env create -f sentiment-nlp-env.yml
```

### Step 3: Activate Environment

```bash
conda activate cap_3764_env
```

### Step 4: Launch Jupyter Notebook

```bash
jupyter notebook
```

---

## Notebooks Description (Run in Order)

### 1) `01_data_collection.ipynb`

Responsible for collecting and preparing raw financial news data.

### 2) `02_sentiment_analysis.ipynb`

Applies sentiment analysis techniques to financial news articles and generates sentiment scores.

### 3) `03_analysis.ipynb`

Explores the relationship between sentiment and stock price movement using visualizations and statistical analysis.

### 4) `04_models.ipynb`

Builds and evaluates predictive models for AAPL next-day direction (UP/DOWN).

---

## Deployment

Use this section to run and deploy the FastAPI backend and Streamlit frontend in `deployment/`.

### 1) Prerequisites

Activate the project environment:

```bash
conda activate cap_3764_env
```

If needed, install deployment dependencies:

```bash
pip install fastapi uvicorn streamlit requests plotly scikit-learn xgboost joblib pandas numpy
```

### 2) Train and Save Models (One-Time)

From the `deployment/` folder, train and save model artifacts to `deployment/models/`:

```bash
cd deployment
python fast_api.py --train --data ../data/raw/final_dataset.csv
```

### 3) Run FastAPI Locally

Start the API server from `deployment/`:

```bash
uvicorn fast_api:app --host 0.0.0.0 --port 8000 --reload
```

Useful endpoints:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

### 4) Run Streamlit Locally

In a second terminal (still from `deployment/`), run:

```bash
streamlit run streamlit.py --server.port 8501
```

Then open:

- `http://127.0.0.1:8501`

If your API runs on a different URL, set:

```bash
export API_URL="http://<your-api-host>:8000"
streamlit run streamlit.py
```

### 5) Deploy FastAPI (Production)

Recommended: deploy FastAPI as a web service (for example, Render, Railway, or an Ubuntu VM with Docker/systemd).

Start command:

```bash
uvicorn fast_api:app --host 0.0.0.0 --port $PORT
```

- Ensure the `deployment/models/` directory is included on the server.
- Enable CORS rules in `fast_api.py` for your Streamlit domain if hosting frontend and API separately.

### 6) Deploy Streamlit (Production)

Recommended: deploy Streamlit on Streamlit Community Cloud or as a separate web service.

Start command:

```bash
streamlit run streamlit.py --server.port $PORT --server.address 0.0.0.0
```

- Set environment variable `API_URL` to the public FastAPI URL.
- Verify that Streamlit can call FastAPI endpoints over HTTPS.

### 7) Quick Deployment Checklist

- FastAPI service is reachable and `/health` returns success.
- Streamlit app loads and can fetch API metadata.
- `API_URL` points to deployed FastAPI URL.
- Trained models and `metadata.json` are present in `deployment/models/`.

---

## Project Workflow

1. Collect financial news data
2. Perform preprocessing and cleaning
3. Apply sentiment analysis techniques
4. Store sentiment-labeled results
5. Merge sentiment data with stock prices
6. Conduct exploratory data analysis
7. Train and evaluate predictive models
8. Visualize trends and model outputs

---

## License

This project is licensed under the terms specified in the `LICENSE` file included in this repository.

---

## Contributors

CAP 3764 Spring 2026 Team 3

---

## Future Improvements

- Implement advanced NLP models for improved sentiment accuracy
- Integrate real-time news feeds
- Explore predictive modeling for price forecasting
- Extend analysis to additional financial instruments
