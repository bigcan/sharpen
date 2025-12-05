"""Script to ingest Phase 6 data (15 years of S&P 500 subset)."""

import pandas as pd
from pathlib import Path
from finrl_pro.data.yahoo_loader import YahooLoader

# Top 20 S&P 500 constituents (approximate) + SPY
TICKERS = [
    "SPY", "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", 
    "TSLA", "BRK-B", "JPM", "JNJ", "V", "XOM", "UNH", "PG",
    "MA", "HD", "LLY", "CVX", "MRK", "ABBV"
]

START_DATE = "2010-01-01"
END_DATE = "2025-01-01"
OUTPUT_PATH = Path("data/sp500_full_2010_2025.parquet")

def main():
    print(f"Fetching data for {len(TICKERS)} tickers from {START_DATE} to {END_DATE}...")
    loader = YahooLoader()
    df = loader.fetch(tickers=TICKERS, start=START_DATE, end=END_DATE)
    
    print(f"Downloaded {len(df)} rows.")
    print(df.head())
    print(df.tail())
    
    OUTPUT_PATH.parent.mkdir(exist_ok=True, parents=True)
    df.to_parquet(OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
