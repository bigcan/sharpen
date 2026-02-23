"""Check gold liquidity - maybe we need the active month, not front month."""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import databento as db

API_KEY = os.environ['DATABENTO_API_KEY']
client = db.Historical(API_KEY)

# GC contract months in 2025: G(Feb), J(Apr), M(Jun), Q(Aug), V(Oct), Z(Dec)
# The "active" gold contract is usually 2-3 months ahead
print("=== Checking different GC continuous months ===")
for n in range(4):
    try:
        cost = client.metadata.get_cost(
            dataset="GLBX.MDP3",
            symbols=[f"GC.c.{n}"],
            stype_in="continuous",
            schema="ohlcv-1m",
            start="2025-06-01",
            end="2025-06-08",
        )
        data = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=[f"GC.c.{n}"],
            stype_in="continuous",
            schema="ohlcv-1m",
            start="2025-06-02",
            end="2025-06-03",
        )
        df = data.to_df()
        print(f"  GC.c.{n}: {len(df)} bars/day, avg vol={df['volume'].mean():.0f}, "
              f"price ~${df['close'].mean():.0f}, symbol={df['symbol'].unique()}")
    except Exception as e:
        print(f"  GC.c.{n}: ERROR - {e}")

# Try specific contract months directly
print("\n=== Specific GC contracts for Jun 2025 ===")
for contract in ["GCQ5", "GCM5", "GCZ5", "GCQ25", "GCM25"]:
    try:
        data = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=[contract],
            schema="ohlcv-1m",
            start="2025-06-02",
            end="2025-06-03",
        )
        df = data.to_df()
        print(f"  {contract}: {len(df)} bars/day, avg vol={df['volume'].mean():.0f}, "
              f"price ~${df['close'].mean():.0f}")
    except Exception as e:
        print(f"  {contract}: ERROR - {str(e)[:80]}")

# Also check ES (E-mini S&P 500)
print("\n=== ES (E-mini S&P) ===")
try:
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=["ES.c.0"],
        stype_in="continuous",
        schema="ohlcv-1m",
        start="2025-06-02",
        end="2025-06-03",
    )
    df = data.to_df()
    print(f"  ES.c.0: {len(df)} bars/day, avg vol={df['volume'].mean():.0f}, "
          f"price ~${df['close'].mean():.0f}")
except Exception as e:
    print(f"  ES.c.0: ERROR - {e}")

# Check ES cost for full year
print("\n=== ES cost check ===")
for schema in ["ohlcv-1m", "mbp-10"]:
    try:
        cost = client.metadata.get_cost(
            dataset="GLBX.MDP3",
            symbols=["ES.c.0"],
            stype_in="continuous",
            schema=schema,
            start="2025-01-01",
            end="2025-12-31",
        )
        print(f"  ES.c.0 {schema}: ${cost:.2f}/year")
    except Exception as e:
        print(f"  ES.c.0 {schema}: ERROR - {e}")

# Check micro crude (MCL) and micro gold (MGC) — might be more liquid
print("\n=== Micro contracts (MCL, MGC) ===")
for sym in ["MCL.c.0", "MGC.c.0"]:
    try:
        data = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=[sym],
            stype_in="continuous",
            schema="ohlcv-1m",
            start="2025-06-02",
            end="2025-06-03",
        )
        df = data.to_df()
        print(f"  {sym}: {len(df)} bars/day, avg vol={df['volume'].mean():.0f}")
    except Exception as e:
        print(f"  {sym}: ERROR - {str(e)[:80]}")
