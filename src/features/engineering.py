
"""
Feature engineering for AAPL sentiment → direction prediction.
All functions are pure (no side effects). Notebooks import from here.
"""
import pandas as pd
import numpy as np


def load_and_merge(final_dataset_path: str) -> pd.DataFrame:
    #Loads final_dataset.csv and engineers all features, returns DF sorted and cleaned

    df = pd.read_csv(final_dataset_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    df['rolling_3d_sentiment'] = df['avg_sentiment'].rolling(3).mean()
    df['rolling_7d_sentiment'] = df['avg_sentiment'].rolling(7).mean()

    df['price_momentum_5d'] = df['close'].pct_change(5)

    df['high_low_range'] = (df['high'] - df['low']) / df['close']

    df['day_of_week'] = df['date'].dt.dayofweek

    df['lagged_target']        = df['target'].shift(1)
    df['sentiment_std_5d']     = df['avg_sentiment'].rolling(5).std()
    df['sentiment_vs_trend']   = df['avg_sentiment'] - df['rolling_7d_sentiment']

    # drops empty values
    df = df.dropna().reset_index(drop=True)

    return df


def temporal_split(df: pd.DataFrame, split_date: str = '2024-09-01'):
    #Splits DF into train and test with two dates(before_ and after_ split_date
    
    split = pd.Timestamp(split_date)
    train = df[df['date'] < split].copy()
    test  = df[df['date'] >= split].copy()

    assert train['date'].max() < test['date'].min(), \
        "Data leakage detected: train and test dates overlap"

    print(f"Train: {train['date'].min().date()} → {train['date'].max().date()} ({len(train)} rows)")
    print(f"Test:  {test['date'].min().date()} → {test['date'].max().date()} ({len(test)} rows)")
    print(f"Train UP%: {train['target'].mean()*100:.1f}%  |  Test UP%: {test['target'].mean()*100:.1f}%")

    return train, test


def get_feature_matrix(df: pd.DataFrame):
    #Returns features and targets from the DF.
    
    FEATURES = [
        'avg_sentiment',
        'rolling_3d_sentiment',
        'rolling_7d_sentiment',
        'daily_return',
        'price_momentum_5d',
        'high_low_range',
        'ma_5',
        'ma_10',
        'volume',
        'day_of_week',
        'lagged_target',
        'sentiment_std_5d',
        'sentiment_vs_trend',
    ]
    TARGET = 'target'

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in DataFrame: {missing}")

    X = df[FEATURES]
    y = df[TARGET]
    return X, y, FEATURES


def export_splits(train_df: pd.DataFrame, test_df: pd.DataFrame,
                  train_path: str = 'data/processed/train_dataset.csv',
                  test_path:  str = 'data/processed/test_dataset.csv'):
    
    # saving the DF
    import os
    os.makedirs('data/processed', exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,   index=False)
    print(f"Saved: {train_path} ({len(train_df)} rows)")
    print(f"Saved: {test_path} ({len(test_df)} rows)")