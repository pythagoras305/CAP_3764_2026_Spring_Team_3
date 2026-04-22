"""
ENDPOINTS
---------
GET  /                → Service status + model metadata (JSON for clients)
GET  /health          → Lightweight liveness check
GET  /features        → List of feature names the model expects
POST /predict         → Single prediction from a JSON payload
POST /predict_batch   → Batch predictions from an uploaded CSV file
GET  /feature_importance → Global feature importance for the selected model
GET  /price_context      → Recent close prices leading up to an as-of session date
"""

import io
import os
import sys
import json
import argparse
from typing import List, Literal

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


#Paths for Model Files
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
LR_PATH   = os.path.join(MODEL_DIR, "logistic_regression.pkl")
XGB_PATH  = os.path.join(MODEL_DIR, "xgboost.pkl")
META_PATH = os.path.join(MODEL_DIR, "metadata.json")

FEATURES: List[str] = [
    "avg_sentiment",
    "rolling_3d_sentiment",
    "rolling_7d_sentiment",
    "daily_return",
    "price_momentum_5d",
    "high_low_range",
    "ma_5",
    "ma_10",
    "volume",
    "day_of_week",
    "lagged_target",
    "sentiment_std_5d",
    "sentiment_vs_trend",
]

DATASET_CANDIDATES = [
    os.path.join(HERE, "..", "data", "raw", "final_dataset.csv"),
    os.path.join(HERE, "data", "raw", "final_dataset.csv"),
    os.path.join(HERE, "..", "data", "final_dataset.csv"),
    os.path.join(HERE, "final_dataset.csv"),
]


def _first_existing_dataset_path() -> str | None:

    for path in DATASET_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


#Copied from src/features/engineering.py to prevent breaking when runnning app
def engineer_features(final_dataset_path: str) -> pd.DataFrame:
    # Load the raw dataset and compute the 13 model features.
    df = pd.read_csv(final_dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["rolling_3d_sentiment"] = df["avg_sentiment"].rolling(3).mean()
    df["rolling_7d_sentiment"] = df["avg_sentiment"].rolling(7).mean()
    df["price_momentum_5d"]    = df["close"].pct_change(5)
    df["high_low_range"]       = (df["high"] - df["low"]) / df["close"]
    df["day_of_week"]          = df["date"].dt.dayofweek
    df["lagged_target"]        = df["target"].shift(1)
    df["sentiment_std_5d"]     = df["avg_sentiment"].rolling(5).std()
    df["sentiment_vs_trend"]   = df["avg_sentiment"] - df["rolling_7d_sentiment"]

    return df.dropna().reset_index(drop=True)


def temporal_split(df: pd.DataFrame, split_date: str = "2024-09-01"):
    split = pd.Timestamp(split_date)
    return df[df["date"] < split].copy(), df[df["date"] >= split].copy()


# In case and keeps fast-api and stresmlit self sufficient 
def train_and_save(data_path: str) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Loading data: {data_path}")
    df = engineer_features(data_path)
    train_df, test_df = temporal_split(df)

    X_train, y_train = train_df[FEATURES], train_df["target"]
    X_test,  y_test  = test_df[FEATURES],  test_df["target"]
    print(f"  train: {len(X_train)} rows   test: {len(X_test)} rows")

    print("Training Logistic Regression ...")
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
    ])
    lr_pipe.fit(X_train, y_train)
    lr_acc = float(lr_pipe.score(X_test, y_test))
    print(f"  LR test accuracy: {lr_acc:.4f}")

    print("Training XGBoost ...")
    xgb = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", early_stopping_rounds=20, random_state=42,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_acc = float(xgb.score(X_test, y_test))
    print(f"  XGB test accuracy: {xgb_acc:.4f}")

    joblib.dump(lr_pipe, LR_PATH)
    joblib.dump(xgb, XGB_PATH)
    with open(META_PATH, "w") as f:
        json.dump({
            "features": FEATURES,
            "target_meaning": {"0": "DOWN", "1": "UP"},
            "split_date": "2024-09-01",
            "models": {
                "logistic_regression": {"test_accuracy": lr_acc},
                "xgboost":             {"test_accuracy": xgb_acc},
            },
        }, f, indent=2)
    print(f"\nSaved:\n  {LR_PATH}\n  {XGB_PATH}\n  {META_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AAPL Direction Predictor")
    parser.add_argument("--train", action="store_true",
                        help="Train both models and save to models/*.pkl")
    parser.add_argument("--data", default="../data/raw/final_dataset.csv",
                        help="Path to final_dataset.csv")
    args = parser.parse_args()
    if args.train:
        train_and_save(args.data)
        sys.exit(0)
    print("Nothing to do. Run with --train to train, or start the server with:")
    print("    uvicorn fast_api:app --reload")
    sys.exit(0)



# Load models at import time
if not all(os.path.exists(p) for p in [LR_PATH, XGB_PATH, META_PATH]):
    raise RuntimeError(
        f"Model artifacts not found in {MODEL_DIR}. "
        "Run: python fast_api.py --train --data path/to/final_dataset.csv"
    )

print("Loading models ...")
MODELS = {
    "logistic_regression": joblib.load(LR_PATH),
    "xgboost":              joblib.load(XGB_PATH),
}
with open(META_PATH) as f:
    META = json.load(f)
print(f"  Models loaded: {list(MODELS.keys())}")


def _patch_legacy_logreg_model(model_obj) -> None:

    lr = None
    if hasattr(model_obj, "named_steps") and "lr" in model_obj.named_steps:
        lr = model_obj.named_steps["lr"]
    elif model_obj.__class__.__name__ == "LogisticRegression":
        lr = model_obj
    if lr is None:
        return
    if not hasattr(lr, "multi_class"):
        lr.multi_class = "auto"
_patch_legacy_logreg_model(MODELS.get("logistic_regression"))

# ===========================================================================
# Pydantic schemas
# ===========================================================================
class PredictRequest(BaseModel):
    """Single-row feature payload."""
    avg_sentiment:        float = Field(..., description="Mean daily news sentiment")
    rolling_3d_sentiment: float
    rolling_7d_sentiment: float
    daily_return:         float
    price_momentum_5d:    float
    high_low_range:       float
    ma_5:                 float
    ma_10:                float
    volume:               float
    day_of_week:          int = Field(..., ge=0, le=6)
    lagged_target:        int = Field(..., ge=0, le=1)
    sentiment_std_5d:     float
    sentiment_vs_trend:   float

    model_config = {
        "json_schema_extra": {
            "example": {
                "avg_sentiment": 0.1234,
                "rolling_3d_sentiment": 0.1050,
                "rolling_7d_sentiment": 0.0820,
                "daily_return": 0.0075,
                "price_momentum_5d": 0.0125,
                "high_low_range": 0.0180,
                "ma_5": 226.45,
                "ma_10": 224.10,
                "volume": 50_000_000,
                "day_of_week": 2,
                "lagged_target": 1,
                "sentiment_std_5d": 0.0412,
                "sentiment_vs_trend": 0.0414,
            }
        }
    }


class PredictResponse(BaseModel):
    model: str
    prediction: int
    label: Literal["DOWN", "UP"]
    probability_up: float
    probability_down: float
    confidence: float


class BatchRow(BaseModel):
    row_index: int
    prediction: int
    label: Literal["DOWN", "UP"]
    probability_up: float
    probability_down: float


class BatchResponse(BaseModel):
    model: str
    n_rows: int
    up_count: int
    down_count: int
    mean_probability_up: float
    predictions: List[BatchRow]


#FASTAPI
app = FastAPI(
    title="AAPL Direction Predictor",
    description=(
        "Next-day price direction (UP/DOWN) for AAPL using news sentiment "
        "and technical features. Models: Logistic Regression and XGBoost."
    ),
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def _get_model(name: str):
    if name not in MODELS:
        raise HTTPException(
            400, f"Unknown model '{name}'. Available: {list(MODELS.keys())}"
        )
    return MODELS[name]


def _row_to_float_matrix(payload: PredictRequest) -> np.ndarray:
    d = payload.model_dump()
    return np.array([[float(d[f]) for f in FEATURES]], dtype=np.float64)


def _prepare_single_input(payload: PredictRequest, model: str):
    if model == "logistic_regression":
        return _row_to_float_matrix(payload)
    return pd.DataFrame([payload.model_dump()])[FEATURES]


def _prepare_batch_input(df: pd.DataFrame, model: str):
    if model == "logistic_regression":
        return np.asarray(df[FEATURES].values, dtype=np.float64)
    return df[FEATURES]


def _predict_up_probs(mdl, X) -> np.ndarray:
    return mdl.predict_proba(X)[:, 1]


def _normalize_importance(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    s = float(vals.sum())
    if s > 0:
        vals = vals / s
    return vals


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "AAPL Direction Predictor",
        "available_models": list(MODELS.keys()),
        "features": FEATURES,
        "test_accuracy": {
            n: META["models"][n]["test_accuracy"] for n in MODELS
        },
        "docs": "/docs",
        "health": "/health",
        "feature_importance": "/feature_importance",
        "price_context": "/price_context",
    }


@app.get("/health")
def health():
    #Checks for Health and Connectivity.
    return {
        "status": "ok",
        "models_loaded": True,
        "available_models": list(MODELS.keys()),
    }


@app.get("/features")
def features():
    return {"features": FEATURES, "target_meaning": META["target_meaning"]}


@app.get("/feature_importance")
def feature_importance(
    model: str = Query("xgboost", description="logistic_regression | xgboost"),
):
    #Return one positive importance score per feature forfeature importance bar charts.
    mdl = _get_model(model)
    if model == "logistic_regression":
        lr = mdl.named_steps["lr"]
        imp = _normalize_importance(np.abs(lr.coef_[0]))
        method = "normalized absolute coefficients (scaled features)"
    else:
        imp = _normalize_importance(mdl.feature_importances_)
        method = "XGBoost feature_importances_ (normalized)"

    return {
        "model": model,
        "method": method,
        "importance": {f: float(v) for f, v in zip(FEATURES, imp)},
    }


# Pre-compute a sample batch from the training CSV so Streamlit can show a built-in dataset without the user uploading a file.
_SAMPLE_BATCH: pd.DataFrame | None = None
# Full engineered history for price context charts
_ENGINEERED_FULL: pd.DataFrame | None = None


def _load_engineered_full() -> pd.DataFrame | None:
    global _ENGINEERED_FULL
    if _ENGINEERED_FULL is not None:
        return _ENGINEERED_FULL
    path = _first_existing_dataset_path()
    if path is not None:
        try:
            _ENGINEERED_FULL = engineer_features(path)
            print(f"  Full engineered dataset loaded ({len(_ENGINEERED_FULL)} rows)")
            return _ENGINEERED_FULL
        except Exception as exc:
            print(f"  (failed to load full dataset {path}: {exc})")
    print("  (no final_dataset.csv for full history — /price_context will 404)")
    return None


def _load_sample_batch() -> pd.DataFrame | None:
    global _SAMPLE_BATCH
    if _SAMPLE_BATCH is not None:
        return _SAMPLE_BATCH
    path = _first_existing_dataset_path()
    if path is not None:
        try:
            df = engineer_features(path)
            _, test_df = temporal_split(df)
            _SAMPLE_BATCH = test_df.reset_index(drop=True)
            print(f"  Sample batch loaded from {path} ({len(_SAMPLE_BATCH)} rows)")
            return _SAMPLE_BATCH
        except Exception as exc:
            print(f"  (failed to load {path}: {exc})")
    print("  (no final_dataset.csv found — /sample_test_data will return 404)")
    return None


# Try to load at startup
_load_sample_batch()


@app.get("/sample_test_data")
def sample_test_data(limit: int = Query(20, ge=1, le=83)):
    
    #Return rows from the pre-loaded test dataset
    batch = _load_sample_batch()
    if batch is None:
        raise HTTPException(
            404,
            "No sample dataset available on the server. "
            "Place final_dataset.csv in ../data/raw/ relative to fast_api.py.",
        )
    rows = batch.head(limit).copy()
    # Convert date to string 
    if "date" in rows.columns:
        rows["date"] = rows["date"].astype(str)
    return {
        "n_rows": len(rows),
        "columns": list(rows.columns),
        "rows": rows.to_dict(orient="records"),
    }


@app.get("/price_context")
def price_context(
    end_date: str | None = Query(
        None,
        description="ISO date (YYYY-MM-DD). Features are as-of this session; "
        "prediction is for the *next* session. Omit to use the last date in the dataset.",
    ),
    days: int = Query(20, ge=5, le=20, description="Trading rows to return ending at end_date (max 20)"),
):
    # Return recent date rows from the dataset so the  UI can plot price leading up to the scored session.
    df = _load_engineered_full()
    if df is None or len(df) == 0:
        raise HTTPException(
            404,
            "No dataset available for price history. Place final_dataset.csv next to the API.",
        )
    dmax = pd.Timestamp(df["date"].max())
    if end_date:
        end = pd.Timestamp(end_date)
    else:
        end = dmax
    dmin = pd.Timestamp(df["date"].min())
    if end < dmin or end > dmax:
        raise HTTPException(
            400,
            f"end_date must be between {dmin.date()} and {dmax.date()} (got {end.date()}).",
        )
    sub = df.loc[df["date"] <= end].sort_values("date").tail(int(days)).copy()
    if len(sub) == 0:
        raise HTTPException(400, "No rows on or before the requested end_date.")

    end_used_dt = sub["date"].iloc[-1]
    sub["date"] = sub["date"].astype(str)
    rows_out = sub[["date", "close"]].to_dict(orient="records")

    realized_next_label: str | None = None
    realized_next_code: int | None = None
    rw = df.loc[df["date"] == end_used_dt]
    if len(rw) > 0 and "target" in df.columns:
        tv = rw["target"].iloc[0]
        if pd.notna(tv):
            realized_next_code = int(tv)
            realized_next_label = "UP" if realized_next_code == 1 else "DOWN"

    return {
        "rows": rows_out,
        "end_date_used": str(pd.Timestamp(end_used_dt).date()),
        "end_close": float(sub["close"].iloc[-1]),
        "n_rows": len(rows_out),
        "min_date_available": str(dmin.date()),
        "max_date_available": str(dmax.date()),
        "realized_next_label": realized_next_label,
        "realized_next_code": realized_next_code,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(
    payload: PredictRequest,
    model: str = Query("xgboost", description="logistic_regression | xgboost"),
):
    mdl = _get_model(model)
    X = _prepare_single_input(payload, model)
    try:
        probs = _predict_up_probs(mdl, X)
    except Exception as exc:
        import traceback
        traceback.print_exc() 
        raise HTTPException(
            500,
            f"Prediction failed: {type(exc).__name__}: {exc}. "
            "Check the server log for the full traceback. "
            "If you see a scikit-learn version warning, retrain with: "
            "python fast_api.py --train --data path/to/final_dataset.csv",
        )

    p_up = float(probs[0])
    pred = int(p_up >= 0.5)
    return PredictResponse(
        model=model,
        prediction=pred,
        label="UP" if pred == 1 else "DOWN",
        probability_up=round(p_up, 4),
        probability_down=round(1 - p_up, 4),
        confidence=round(max(p_up, 1 - p_up), 4),
    )


@app.post("/predict_batch", response_model=BatchResponse)
async def predict_batch(
    file: UploadFile = File(..., description="CSV with the 13 feature columns"),
    model: str = Query("xgboost", description="logistic_regression | xgboost"),
):
    mdl = _get_model(model)

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(400, f"Could not parse CSV: {exc}")

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise HTTPException(400, f"CSV is missing required columns: {missing}")

    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    if len(df) == 0:
        raise HTTPException(400, "After dropping NaN rows, 0 rows remain.")

    Xb = _prepare_batch_input(df, model)
    probs = _predict_up_probs(mdl, Xb)
    preds = (probs >= 0.5).astype(int)

    rows = [
        BatchRow(
            row_index=i,
            prediction=int(preds[i]),
            label="UP" if preds[i] == 1 else "DOWN",
            probability_up=round(float(probs[i]), 4),
            probability_down=round(1.0 - float(probs[i]), 4),
        )
        for i in range(len(df))
    ]

    return BatchResponse(
        model=model,
        n_rows=len(df),
        up_count=int((preds == 1).sum()),
        down_count=int((preds == 0).sum()),
        mean_probability_up=round(float(probs.mean()), 4),
        predictions=rows,
    )
