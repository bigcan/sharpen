"""
Fetch CME futures data from Databento for oracle gate comparison.

Phase 1: OHLCV-1m for CL and GC (cheap, fast) → oracle gate
Phase 2: MBP-10 for winners → full pipeline (if oracle passes)

Usage:
    python scripts/fetch_cme_data.py --phase 1    # OHLCV only (~$1.22)
    python scripts/fetch_cme_data.py --phase 2    # MBP-10 full book (~$69)
"""
import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / '.env')

import databento as db
import pandas as pd

API_KEY = os.environ.get('DATABENTO_API_KEY')
if not API_KEY:
    print("ERROR: DATABENTO_API_KEY not set in .env")
    sys.exit(1)

DATASET = "GLBX.MDP3"
DATA_DIR = Path(__file__).parent.parent / "data"

# Full year 2025 to match BTC data
START = "2025-01-01"
END = "2025-12-31"

# Assets to fetch
ASSETS = {
    "CL": {"name": "Crude Oil", "symbol": "CL.c.0", "tick_size": 0.01, "multiplier": 1000,
            "notional_approx": 70000, "commission_per_side": 2.50},
    "GC": {"name": "Gold", "symbol": "GC.c.0", "tick_size": 0.10, "multiplier": 100,
            "notional_approx": 200000, "commission_per_side": 2.50},
}


def fetch_ohlcv(client, symbol_key: str, info: dict) -> pd.DataFrame:
    """Fetch 1-minute OHLCV bars."""
    print(f"\n  Fetching OHLCV-1m for {info['name']} ({info['symbol']})...")

    data = client.timeseries.get_range(
        dataset=DATASET,
        symbols=[info['symbol']],
        stype_in="continuous",
        schema="ohlcv-1m",
        start=START,
        end=END,
    )

    df = data.to_df()
    print(f"  Received {len(df):,} rows")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Columns: {list(df.columns)}")

    return df


def fetch_mbp1(client, symbol_key: str, info: dict) -> pd.DataFrame:
    """Fetch top-of-book (MBP-1) for spread calculation."""
    print(f"\n  Fetching MBP-1 for {info['name']} ({info['symbol']})...")

    data = client.timeseries.get_range(
        dataset=DATASET,
        symbols=[info['symbol']],
        stype_in="continuous",
        schema="mbp-1",
        start=START,
        end=END,
    )

    df = data.to_df()
    print(f"  Received {len(df):,} rows (tick-level)")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Columns: {list(df.columns)}")

    return df


def resample_mbp1_to_1min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample tick-level MBP-1 to 1-minute snapshots (last value per minute)."""
    # Take last snapshot per minute
    resampled = df.resample('1min').last().dropna(subset=['close'])
    print(f"  Resampled to {len(resampled):,} 1-min bars")
    return resampled


def save_ohlcv_parquet(df: pd.DataFrame, symbol_key: str, info: dict):
    """Save OHLCV data in a clean format."""
    out_dir = DATA_DIR / "cme"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Standardize column names
    out = pd.DataFrame()
    out['timestamp'] = df.index
    out['open'] = df['open'].values / 1e9  # Databento prices in fixed-point (1e-9)
    out['high'] = df['high'].values / 1e9
    out['low'] = df['low'].values / 1e9
    out['close'] = df['close'].values / 1e9
    out['volume'] = df['volume'].values

    # Add mid_price (use close as proxy for OHLCV-only data)
    out['mid_price'] = out['close']

    # Add asset metadata
    out['symbol'] = symbol_key
    out['tick_size'] = info['tick_size']
    out['multiplier'] = info['multiplier']

    out_path = out_dir / f"{symbol_key.lower()}_2025_ohlcv_1min.parquet"
    out.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path} ({len(out):,} rows, {out_path.stat().st_size / 1e6:.1f} MB)")

    return out


def phase1(client):
    """Phase 1: Fetch OHLCV-1m for all assets."""
    print("=" * 60)
    print("PHASE 1: Fetching OHLCV-1m data (~$1.22 total)")
    print("=" * 60)

    results = {}
    for sym, info in ASSETS.items():
        try:
            df = fetch_ohlcv(client, sym, info)
            out = save_ohlcv_parquet(df, sym, info)
            results[sym] = out

            # Print summary stats
            print(f"\n  {info['name']} Summary:")
            print(f"    Bars: {len(out):,}")
            print(f"    Price range: ${out['close'].min():.2f} - ${out['close'].max():.2f}")
            print(f"    Avg volume/bar: {out['volume'].mean():.0f}")
            print(f"    Trading days: {out['timestamp'].dt.date.nunique()}")

        except Exception as e:
            print(f"  ERROR fetching {sym}: {e}")
            import traceback
            traceback.print_exc()

    return results


def phase2(client, symbols=None):
    """Phase 2: Fetch MBP-10 full book data for specified symbols."""
    print("=" * 60)
    print("PHASE 2: Fetching MBP-10 data (full order book)")
    print("=" * 60)

    if symbols is None:
        symbols = list(ASSETS.keys())

    for sym in symbols:
        if sym not in ASSETS:
            print(f"  Unknown symbol: {sym}")
            continue

        info = ASSETS[sym]
        print(f"\n  Fetching MBP-10 for {info['name']} ({info['symbol']})...")
        print(f"  NOTE: This may take a while for large datasets...")

        try:
            # Stream to file to manage memory
            out_dir = DATA_DIR / "cme" / "raw"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{sym.lower()}_2025_mbp10.dbn.zst"

            data = client.timeseries.get_range(
                dataset=DATASET,
                symbols=[info['symbol']],
                stype_in="continuous",
                schema="mbp-10",
                start=START,
                end=END,
            )

            # Save raw DBN file
            data.to_file(str(out_path))
            print(f"  Saved raw: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

        except Exception as e:
            print(f"  ERROR fetching MBP-10 for {sym}: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Fetch CME futures data from Databento")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1,
                        help="Phase 1: OHLCV only. Phase 2: MBP-10 full book.")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Symbols to fetch (default: all)")
    args = parser.parse_args()

    client = db.Historical(API_KEY)

    if args.phase == 1:
        phase1(client)
    elif args.phase == 2:
        phase2(client, args.symbols)


if __name__ == "__main__":
    main()
