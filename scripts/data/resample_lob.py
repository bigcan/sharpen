"""Resample a 1s LOB parquet into Ns bars.

Input schema (from build_btcusdt_lob_1s.py):
    timestamp, timestamp_ms, segment_id, open, high, low, close, volume,
    bbo_bid_qty, bbo_ask_qty, spread_mean

Aggregation within each (segment_id, floor(ts / bar_seconds)) bucket:
    open        : first
    high        : max
    low         : min
    close       : last
    volume      : sum
    bbo_*_qty   : mean
    spread_mean : mean
    timestamp   : bucket start

Bars that fall across a segment break are dropped (segment_id must be unique
within a bucket). Buckets with fewer than `min_snapshots` are also dropped.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="source 1s parquet")
    ap.add_argument("--out", type=Path, required=True, help="destination Ns parquet")
    ap.add_argument("--bar_seconds", type=int, required=True)
    ap.add_argument("--min_snapshots", type=int, default=2)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[load] {args.src}")
    df = pd.read_parquet(args.src).sort_values("timestamp_ms").reset_index(drop=True)
    print(f"[load] {len(df):,} rows in {time.time()-t0:.1f}s")

    bar_ms = args.bar_seconds * 1000
    df["bucket"] = (df["timestamp_ms"].astype("int64") // bar_ms).astype("int64")

    # Drop buckets that span segment breaks
    seg_uniq = df.groupby("bucket")["segment_id"].nunique()
    good = seg_uniq[seg_uniq == 1].index
    df = df[df["bucket"].isin(good)]

    g = df.groupby("bucket", sort=True)
    agg = pd.DataFrame({
        "timestamp_ms": (g["bucket"].first() * bar_ms).astype("int64"),
        "segment_id": g["segment_id"].first().astype("int32"),
        "open": g["open"].first().astype("float64"),
        "high": g["high"].max().astype("float64"),
        "low": g["low"].min().astype("float64"),
        "close": g["close"].last().astype("float64"),
        "volume": g["volume"].sum().astype("float64"),
        "bbo_bid_qty": g["bbo_bid_qty"].mean().astype("float64"),
        "bbo_ask_qty": g["bbo_ask_qty"].mean().astype("float64"),
        "spread_mean": g["spread_mean"].mean().astype("float32"),
        "n_snaps": g.size().astype("int32"),
    })
    before = len(agg)
    agg = agg[agg["n_snaps"] >= args.min_snapshots].drop(columns=["n_snaps"])
    print(f"[resample] {before:,} -> {len(agg):,} bars (drop < {args.min_snapshots} snaps)")

    agg.insert(0, "timestamp", pd.to_datetime(agg["timestamp_ms"], unit="ms", utc=True))
    agg = agg.reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(args.out, index=False, compression="snappy")
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"[write] {args.out} ({size_mb:.1f} MB, {len(agg):,} bars)")

    print("\n=== AUDIT ===")
    print(f"rows            : {len(agg):,}")
    print(f"date range      : {agg['timestamp'].iloc[0]} -> {agg['timestamp'].iloc[-1]}")
    print(f"segments        : {agg['segment_id'].nunique():,}")
    print(f"spread median   : {np.median(agg['spread_mean']):.4f}")
    print(f"spread p99      : {np.percentile(agg['spread_mean'], 99):.4f}")
    print(f"mid median      : {np.median((agg['open']+agg['close'])/2):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
