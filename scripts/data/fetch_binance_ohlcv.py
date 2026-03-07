"""
Fetch Binance OHLCV Data
Fetch 6 months of 1-minute BTCUSDT Perpetual Futures data.
"""
import ccxt
import pandas as pd
import argparse
from datetime import datetime, timedelta
import time
from pathlib import Path

def fetch_binance_ohlcv(symbol: str, days: int = None, start_date: str = None, end_date: str = None, output_file: str = None, dry_run: bool = False):
    """
    Fetches OHLCV data from Binance Futures.
    """
    print(f"Initializing Binance client for {symbol}...")
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })

    timeframe = '1m'

    # Determine start timestamp
    if start_date:
        since_dt = datetime.strptime(start_date, "%Y-%m-%d")
        since_ts = int(since_dt.timestamp() * 1000)
    else:
        since_dt = datetime.now() - timedelta(days=days or 180)
        since_ts = int(since_dt.timestamp() * 1000)

    # Determine end timestamp
    end_ts = int(datetime.now().timestamp() * 1000)
    if end_date:
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

    print(f"Fetching data from {datetime.fromtimestamp(since_ts/1000)} to {datetime.fromtimestamp(end_ts/1000)}...")

    if dry_run:
        print("Dry run: fetching 1 batch only.")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since_ts, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        print(f"fetched {len(df)} rows.")
        return

    all_ohlcv = []

    # Simple pagination loop
    current_since = since_ts
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, current_since, limit=1000)
            if not ohlcv:
                break

            # Filter if beyond end_date
            if ohlcv[0][0] > end_ts:
                break

            all_ohlcv.extend([x for x in ohlcv if x[0] <= end_ts])
            current_since = ohlcv[-1][0] + 60000 # +1 minute

            # Print progress
            last_date = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
            print(f"Fetched up to {last_date}...")

            if ohlcv[-1][0] >= end_ts:
                break

            time.sleep(exchange.rateLimit / 1000)

        except Exception as e:
            print(f"Error fetching: {e}")
            break

    if not all_ohlcv:
        print("No data fetched!")
        return

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Save
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)
    print(f"Saved {len(df)} rows to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair")
    parser.add_argument("--days", type=int, help="Days of history (from now)")
    parser.add_argument("--start-date", help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", help="End date YYYY-MM-DD")
    parser.add_argument("--output", default="c:/data/raw/binance_ohlcv_6m.parquet", help="Output file")
    parser.add_argument("--dry-run", action="store_true", help="Run quick test")
    args = parser.parse_args()

    fetch_binance_ohlcv(args.symbol, args.days, args.start_date, args.end_date, args.output, args.dry_run)
