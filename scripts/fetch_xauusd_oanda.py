"""
Fetch XAU/USD OHLCV-15m from OANDA v20 Practice REST API.
==========================================================

Free replacement for yfinance (60-day cap) — needed for true Q1 2026 OOS
validation of the gmgp1-xauusd paper strategy.

OANDA practice account + personal access token is free forever. Endpoint
returns up to 5000 candles per request; paginate via the `to` cursor.

Auth:
  export OANDA_TOKEN=<practice personal access token>
  # Account ID not required for instrument candles endpoint.

Granularities: M1, M5, M15, M30, H1, H4, D. We default M15.
Price components: M (mid) | B (bid) | A (ask). Default M.

Output: data/oanda/xauusd_15m.parquet (schema: timestamp UTC, OHLCV)

Usage:
  python scripts/fetch_xauusd_oanda.py
  python scripts/fetch_xauusd_oanda.py --start 2020-01-01 --end 2026-04-16
  python scripts/fetch_xauusd_oanda.py --granularity M5 --price B

After fetch, pipe through cleaner (DATA-CLEAN invariant, creates .bak):
  python scripts/clean_ohlcv.py --input data/oanda/xauusd_15m.parquet
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api-fxpractice.oanda.com/v3"
INSTRUMENT = "XAU_USD"
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
    chunk_span = pd.Timedelta(seconds=step_s * 4900)  # stay under 5000 cap

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
        # Advance one bar past last returned to avoid duplicate
        next_cursor = last_ts + pd.Timedelta(seconds=step_s)
        if next_cursor <= cursor:
            # No forward progress — OANDA returned only pre-cursor candles (weekend gap etc.)
            next_cursor = cursor + chunk_span
        cursor = next_cursor
        print(f"[{req_count:4d}] fetched {len(rows):4d} complete bars, "
              f"cursor -> {cursor.isoformat()}", flush=True)
        time.sleep(0.12)  # ~8 req/s, well under OANDA's limit

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
    df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01", help="ISO date, UTC")
    ap.add_argument("--end", default=None, help="ISO date, UTC (default: now)")
    ap.add_argument("--granularity", default="M15", choices=list(GRANULARITY_SECONDS))
    ap.add_argument("--price", default="M", choices=["M", "B", "A"])
    ap.add_argument("--output", default=None, help="Output parquet path")
    args = ap.parse_args()

    token = os.environ.get("OANDA_TOKEN")
    if not token:
        print("ERROR: OANDA_TOKEN env var not set.", file=sys.stderr)
        print("  Get one: https://www.oanda.com/demo-account/tpa/personal_token", file=sys.stderr)
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
        DATA_DIR / f"xauusd_{args.granularity.lower()}_{args.price.lower()}.parquet")
    df.to_parquet(out, index=False)

    span_d = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 86400
    print(f"\nWrote {len(df):,} bars to {out}")
    print(f"  range: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]} ({span_d:.1f} days)")
    print(f"  next: python scripts/clean_ohlcv.py --input {out}")


if __name__ == "__main__":
    main()
