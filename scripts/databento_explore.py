"""
Databento data exploration - check availability and cost for CL and GC futures.
"""
import os
import sys

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

import databento as db  # noqa: E402

API_KEY = os.environ.get('DATABENTO_API_KEY')
if not API_KEY:
    print("ERROR: DATABENTO_API_KEY not set")
    sys.exit(1)

client = db.Historical(API_KEY)

# CME Group futures are on GLBX.MDP3
DATASET = "GLBX.MDP3"

# Check available symbols for CL and GC
print("=" * 60)
print("DATABENTO EXPLORATION - CL (Crude Oil) & GC (Gold)")
print("=" * 60)

# We want continuous front-month contracts
# Databento uses instrument IDs or symbology
# CL.c.0 = continuous front month crude
# GC.c.0 = continuous front month gold

# First, let's check cost for different schemas
schemas_to_check = ["mbp-1", "mbp-10", "ohlcv-1m", "ohlcv-1s"]
symbols_to_check = {
    "CL.FUT": "Crude Oil (all contracts)",
    "GC.FUT": "Gold (all contracts)",
}

# Check cost for a 1-month sample first (January 2025)
print("\n--- Cost Estimates (January 2025 only, 1 month sample) ---")
for schema in ["mbp-1", "mbp-10", "ohlcv-1m"]:
    for symbol, name in symbols_to_check.items():
        try:
            cost = client.metadata.get_cost(
                dataset=DATASET,
                symbols=[symbol],
                schema=schema,
                start="2025-01-01",
                end="2025-02-01",
            )
            print(f"  {name:40s} | {schema:10s} | ${cost:.2f}/month")
        except Exception as e:
            print(f"  {name:40s} | {schema:10s} | ERROR: {e}")

# Check cost for full year
print("\n--- Cost Estimates (Full Year 2025, Jan-Dec) ---")
for schema in ["mbp-1", "mbp-10", "ohlcv-1m"]:
    for symbol, name in symbols_to_check.items():
        try:
            cost = client.metadata.get_cost(
                dataset=DATASET,
                symbols=[symbol],
                schema=schema,
                start="2025-01-01",
                end="2025-12-31",
            )
            print(f"  {name:40s} | {schema:10s} | ${cost:.2f}/year")
        except Exception as e:
            print(f"  {name:40s} | {schema:10s} | ERROR: {e}")

# Also check with continuous contract symbology
print("\n--- Continuous Front Month Cost (Full Year 2025) ---")
for schema in ["mbp-1", "mbp-10", "ohlcv-1m"]:
    for sym_root in ["CL", "GC"]:
        try:
            cost = client.metadata.get_cost(
                dataset=DATASET,
                symbols=[f"{sym_root}.c.0"],
                stype_in="continuous",
                schema=schema,
                start="2025-01-01",
                end="2025-12-31",
            )
            print(f"  {sym_root}.c.0 (continuous front) | {schema:10s} | ${cost:.2f}/year")
        except Exception as e:
            print(f"  {sym_root}.c.0 (continuous front) | {schema:10s} | ERROR: {e}")

# Check NQ too as a reference
print("\n--- NQ (Nasdaq) for reference ---")
for schema in ["mbp-10", "ohlcv-1m"]:
    try:
        cost = client.metadata.get_cost(
            dataset=DATASET,
            symbols=["NQ.c.0"],
            stype_in="continuous",
            schema=schema,
            start="2025-01-01",
            end="2025-12-31",
        )
        print(f"  NQ.c.0 (continuous front) | {schema:10s} | ${cost:.2f}/year")
    except Exception as e:
        print(f"  NQ.c.0 (continuous front) | {schema:10s} | ERROR: {e}")

print("\n--- Done ---")
