import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv


NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"
TICKERS = ["AAPL", "NVDA"]

QUERIES = {
    "AAPL": '(AAPL OR Apple) AND (stock OR shares OR earnings OR iPhone OR "Apple Inc")',
    "NVDA": '(NVDA OR Nvidia OR "NVIDIA") AND (stock OR shares OR earnings OR GPU OR AI)',
}

DAYS_BACK = 21  # change to 14 for 2 weeks, 21 for 3 weeks


def get_date_range(days_back: int):
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days_back)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def fetch_articles(api_key: str, ticker: str, start: str, end: str, max_results: int = 100):
    """NewsAPI free tier allows up to 100 results per query."""
    params = {
        "q": QUERIES[ticker],
        "from": start,
        "to": end,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": max_results,  # <= 100
        "page": 1,
    }
    headers = {"X-Api-Key": api_key}

    r = requests.get(NEWSAPI_ENDPOINT, params=params, headers=headers, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"{ticker} NewsAPI error {r.status_code}: {r.text}")

    rows = []
    for a in r.json().get("articles", []):
        published_at = a.get("publishedAt")
        if not published_at:
            continue

        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        rows.append(
            {
                "ticker": ticker,
                "date": dt.strftime("%Y-%m-%d"),
                "headline": (a.get("title") or "").strip(),
                "content": (a.get("content") or a.get("description") or "").strip(),
                "source": ((a.get("source") or {}).get("name") or "").strip(),
                "url": (a.get("url") or "").strip(),  # helper for dedup
            }
        )
    return rows


def clean_articles(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["ticker", "date"])
    df = df[df["ticker"].astype(str).str.len() > 0]

    # dedupe
    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"], keep="first")
    df = df.drop_duplicates(subset=["ticker", "date", "headline"], keep="first")

    return df[["ticker", "date", "headline", "content", "source"]].copy()


def fetch_prices(start: str, end: str) -> pd.DataFrame:
    # yfinance end can be exclusive; add 1 day buffer
    end_plus = (datetime.fromisoformat(end) + timedelta(days=1)).strftime("%Y-%m-%d")

    data = yf.download(
        tickers=" ".join(TICKERS),
        start=start,
        end=end_plus,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
    )

    rows = []
    for t in TICKERS:
        sub = data[t].reset_index()
        sub["ticker"] = t
        rows.append(sub)

    prices = pd.concat(rows, ignore_index=True)

    prices = prices.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    prices = prices.dropna(subset=["ticker", "date"])
    prices = prices.drop_duplicates(subset=["ticker", "date"], keep="first")

    return prices[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]].copy()


def main():
    load_dotenv()
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise RuntimeError("NEWSAPI_KEY not found. Add it to a .env file or environment variable.")

    start, end = get_date_range(DAYS_BACK)
    print(f"Date range: {start} to {end}")

    # --- Articles ---
    all_rows = []
    for t in TICKERS:
        print(f"Fetching articles for {t}...")
        rows = fetch_articles(api_key, t, start, end, max_results=100)
        print(f"  Raw pulled: {len(rows)}")
        all_rows.extend(rows)
        time.sleep(1)

    articles = clean_articles(pd.DataFrame(all_rows))
    print("Counts after cleaning:")
    print(articles["ticker"].value_counts())

    articles.to_csv("articles.csv", index=False, encoding="utf-8")
    print("Saved articles.csv")

    # --- Prices ---
    prices = fetch_prices(start, end)
    print("Price rows per ticker:")
    print(prices["ticker"].value_counts())

    prices.to_csv("prices.csv", index=False, encoding="utf-8")
    print("Saved prices.csv")


if __name__ == "__main__":
    main()