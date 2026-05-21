"""
Fetch EUR/USD OHLCV from OANDA v20 Practice REST API.
=====================================================

Sibling to `fetch_xauusd_oanda.py`; only diffs are the instrument code,
default output name, and default date range. Same auth, same pagination,
same chunk semantics.

S544-cont-5 (2026-05-21): created to source training data for the SG-1
EURUSD HPO redirect (path (b) under `project_sg1_eurusd_retrain_queued_s477`),
after AlphaSeek v3 KILL freed gpuhub-1 capacity.

Auth:
  set -a; source .env; set +a            # OANDA_TOKEN lives in .env, not shell

Output: data/oanda/eurusd_m1_m.parquet (schema: timestamp UTC, OHLCV)

Usage:
  python scripts/fetch_eurusd_oanda.py
  python scripts/fetch_eurusd_oanda.py --start 2024-01-01 --end 2026-05-21
  python scripts/fetch_eurusd_oanda.py --granularity M5 --price B

After fetch, pipe through cleaner (DATA-CLEAN invariant, creates .bak):
  python scripts/clean_ohlcv.py --input data/oanda/eurusd_m1_m.parquet
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api-fxpractice.oanda.com/v3"
INSTRUMENT = "EUR_USD"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "oanda"

GRANULARITY_SECONDS = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D": 86400,
}


def fetch_chunk(token: str, instrument: str, granularity: str,
                from_ts: pd.Timestamp, price: str, count: int = 5000) -> list:
    url = f"{BASE_URL}/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {token}", "Accept-Datetime-Format": "RFC3339"}
    params = {
        "granularity": granularity,
        "price": price,
        "from": from_ts.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
        "count": count,
    }
    r = requests.get(url, headers=headers, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"OANDA {r.status_code}: {r.text[:300]}")
    return r.json().get("candles", [])


def candles_to_rows(candles: list, price: str) -> list:
    key = {"M": "mid", "B": "bid", "A": "ask"}[price]
    rows = []
    for c in candles:
        if not c.get("complete"):
            continue
        p = c[key]
        rows.append({
            "timestamp": pd.Timestamp(c["time"]).tz_convert("UTC"),
            "open": float(p["o"]),
            "high": float(p["h"]),
            "low": float(p["l"]),
            "close": float(p["c"]),
            "volume": int(c.get("volume", 0)),
        })
    return rows


def fetch_range(token: str, start: pd.Timestamp, end: pd.Timestamp,
                granularity: str, price: str) -> pd.DataFrame:
    step_s = GRANULARITY_SECONDS[granularity]
    chunk_span = pd.Timedelta(seconds=step_s * 4900)

    all_rows: list = []
    cursor = start
    req_count = 0
    while cursor < end:
        candles = fetch_chunk(token, INSTRUMENT, granularity, cursor, price, count=5000)
        req_count += 1
        if not candles:
            cursor = cursor + chunk_span
            continue
        rows = candles_to_rows(candles, price)
        all_rows.extend(rows)
        last_ts = pd.Timestamp(candles[-1]["time"]).tz_convert("UTC")
        next_cursor = last_ts + pd.Timedelta(seconds=step_s)
        if next_cursor <= cursor:
            next_cursor = cursor + chunk_span
        cursor = next_cursor
        print(f"[{req_count:4d}] fetched {len(rows):4d} complete bars, "
              f"cursor -> {cursor.isoformat()}", flush=True)
        time.sleep(0.12)

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
    df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-06-01", help="ISO date, UTC (default: 2024-06-01 for ~2yr coverage)")
    ap.add_argument("--end", default=None, help="ISO date, UTC (default: now)")
    ap.add_argument("--granularity", default="M1", choices=list(GRANULARITY_SECONDS))
    ap.add_argument("--price", default="M", choices=["M", "B", "A"])
    ap.add_argument("--output", default=None, help="Output parquet path")
    args = ap.parse_args()

    token = os.environ.get("OANDA_TOKEN")
    if not token:
        print("ERROR: OANDA_TOKEN env var not set.", file=sys.stderr)
        print("  In this repo it lives in .env -- source it first:", file=sys.stderr)
        print("    set -a; source .env; set +a", file=sys.stderr)
        print("  Or get a fresh practice token: https://www.oanda.com/demo-account/tpa/personal_token", file=sys.stderr)
        sys.exit(2)

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp.now(tz="UTC")

    print(f"Fetching {INSTRUMENT} {args.granularity} price={args.price} "
          f"{start.date()} -> {end.date()}", flush=True)

    df = fetch_range(token, start, end, args.granularity, args.price)
    if df.empty:
        print("No bars returned.", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.output) if args.output else (
        DATA_DIR / f"eurusd_{args.granularity.lower()}_{args.price.lower()}.parquet")
    df.to_parquet(out, index=False)

    span_d = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 86400
    print(f"\nWrote {len(df):,} bars to {out}")
    print(f"  range: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]} ({span_d:.1f} days)")
    print(f"  next: python scripts/clean_ohlcv.py --input {out}")


if __name__ == "__main__":
    main()
