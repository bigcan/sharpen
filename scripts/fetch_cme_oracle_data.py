"""
Fetch CME OHLCV-1m data for oracle gate comparison.
CL (Crude Oil), GC (Gold active month), ES (E-mini S&P).

Total cost: ~$3.14
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import databento as db
import pandas as pd
import numpy as np

API_KEY = os.environ.get('DATABENTO_API_KEY')
if not API_KEY:
    print("ERROR: DATABENTO_API_KEY not set")
    sys.exit(1)

client = db.Historical(API_KEY)

DATASET = "GLBX.MDP3"
DATA_DIR = Path(__file__).parent.parent / "data" / "cme"
DATA_DIR.mkdir(parents=True, exist_ok=True)

START = "2025-01-01"
END = "2025-12-31"

ASSETS = {
    "CL": {
        "name": "Crude Oil",
        "symbol": "CL.c.0",
        "tick_size": 0.01,       # $0.01 per barrel
        "multiplier": 1000,       # 1000 barrels per contract
        "point_value": 10.0,      # $10 per tick ($0.01 * 1000)
        "commission_per_side": 2.50,  # ~$2.50 per side retail
        "typical_spread_ticks": 1,    # Usually 1 tick spread
    },
    "GC": {
        "name": "Gold (active)",
        "symbol": "GC.c.2",      # Active month, NOT front month
        "tick_size": 0.10,        # $0.10 per troy oz
        "multiplier": 100,        # 100 troy oz per contract
        "point_value": 10.0,      # $10 per tick ($0.10 * 100)
        "commission_per_side": 2.50,
        "typical_spread_ticks": 1,
    },
    "ES": {
        "name": "E-mini S&P 500",
        "symbol": "ES.c.0",
        "tick_size": 0.25,        # $0.25 per index point
        "multiplier": 50,         # $50 per point
        "point_value": 12.50,     # $12.50 per tick ($0.25 * 50)
        "commission_per_side": 2.50,
        "typical_spread_ticks": 1,
    },
}


def fetch_and_save(sym_key: str, info: dict):
    """Fetch OHLCV-1m and save as parquet."""
    print(f"\n{'='*50}")
    print(f"Fetching {info['name']} ({info['symbol']})")
    print(f"{'='*50}")

    data = client.timeseries.get_range(
        dataset=DATASET,
        symbols=[info['symbol']],
        stype_in="continuous",
        schema="ohlcv-1m",
        start=START,
        end=END,
    )

    df = data.to_df()
    print(f"  Raw rows: {len(df):,}")

    # Build clean dataframe
    out = pd.DataFrame()
    out['timestamp'] = df.index.tz_localize(None)  # Remove timezone for consistency
    out['open'] = df['open'].values
    out['high'] = df['high'].values
    out['low'] = df['low'].values
    out['close'] = df['close'].values
    out['volume'] = df['volume'].values.astype(np.float64)

    # Compute mid price and spread estimate
    out['mid_price'] = (out['open'] + out['close']) / 2

    # Metadata columns
    out['symbol'] = sym_key

    # Remove any rows with zero volume or NaN prices
    before = len(out)
    out = out.dropna(subset=['close'])
    out = out[out['close'] > 0]
    after = len(out)
    if before != after:
        print(f"  Dropped {before - after} invalid rows")

    # Save
    out_path = DATA_DIR / f"{sym_key.lower()}_2025_ohlcv_1min.parquet"
    out.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path} ({len(out):,} rows, {out_path.stat().st_size / 1e6:.1f} MB)")

    # Summary
    print(f"\n  Summary for {info['name']}:")
    print(f"    Date range: {out['timestamp'].min()} to {out['timestamp'].max()}")
    print(f"    Trading days: {out['timestamp'].dt.date.nunique()}")
    print(f"    Total bars: {len(out):,}")
    print(f"    Avg bars/day: {len(out) / out['timestamp'].dt.date.nunique():.0f}")
    print(f"    Price range: ${out['close'].min():.2f} - ${out['close'].max():.2f}")
    print(f"    Avg price: ${out['close'].mean():.2f}")
    print(f"    Avg volume/bar: {out['volume'].mean():.0f}")

    # Fee analysis
    avg_price = out['close'].mean()
    notional = avg_price * info['multiplier']
    spread_cost_bps = (info['typical_spread_ticks'] * info['tick_size'] / avg_price) * 10000
    commission_bps = (info['commission_per_side'] / notional) * 10000
    total_one_way_bps = spread_cost_bps + commission_bps

    print(f"\n  Fee Analysis:")
    print(f"    Notional per contract: ${notional:,.0f}")
    print(f"    Spread (1 tick): {spread_cost_bps:.2f} bps")
    print(f"    Commission: {commission_bps:.2f} bps")
    print(f"    Total one-way: {total_one_way_bps:.2f} bps")
    print(f"    Total round-trip: {total_one_way_bps * 2:.2f} bps")
    print(f"    vs BTC Binance (5.0 bps/side): {5.0 / total_one_way_bps:.1f}x cheaper")
    print(f"    vs BTC Hyperliquid (2.5 bps/side): {2.5 / total_one_way_bps:.1f}x cheaper")

    return out


if __name__ == "__main__":
    for sym, info in ASSETS.items():
        try:
            fetch_and_save(sym, info)
        except Exception as e:
            print(f"  ERROR fetching {sym}: {e}")
            import traceback
            traceback.print_exc()
