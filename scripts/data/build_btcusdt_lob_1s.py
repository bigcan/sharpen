"""Build BTCUSDT 1s LOB parquet from secure_lob_collection.db.

Each snapshot is one bar. Adjacent snapshots form the bar's OHLC:
    open_i  = mid_i
    close_i = mid_{i+1}
    high_i  = max(mid_i, mid_{i+1})
    low_i   = min(mid_i, mid_{i+1})

Volume is unused by the A-S baseline (PriceCrossFillModel ignores volume), so
it is set to 0. Spread / BBO qty come directly from the snapshot. Schema is a
subset of the 10s parquet: (timestamp, timestamp_ms, segment_id, open, high,
low, close, volume, bbo_bid_qty, bbo_ask_qty, spread_mean). That is enough to
drive scripts/baselines/avellaneda_stoikov_mm.py unchanged.

Segment IDs are assigned by detecting gaps > 5 seconds between consecutive
snapshots (collection dropouts).
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "secure_lob_collection.db"
OUT_DIR = ROOT / "data" / "processed"
SYMBOL = "BTCUSDT"
GAP_SECONDS = 5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--symbol", default=SYMBOL)
    args = ap.parse_args()

    if args.out is None:
        args.out = OUT_DIR / f"{args.symbol.lower()}_lob_1s.parquet"

    t0 = time.time()
    print(f"[load] reading {args.symbol} snapshots from {args.db}")
    conn = sqlite3.connect(str(args.db))
    q = """
        SELECT timestamp_ms, datetime, best_bid_price, best_bid_qty,
               best_ask_price, best_ask_qty
        FROM lob_snapshots
        WHERE symbol = ?
        ORDER BY timestamp_ms ASC
    """
    df = pd.read_sql_query(q, conn, params=(args.symbol,))
    conn.close()
    print(f"[load] {len(df):,} rows in {time.time()-t0:.1f}s")

    # Clean: drop non-positive prices / qty, dedupe timestamps (keep last)
    before = len(df)
    df = df[
        (df["best_bid_price"] > 0) & (df["best_ask_price"] > 0)
        & (df["best_bid_qty"] > 0) & (df["best_ask_qty"] > 0)
        & (df["best_ask_price"] > df["best_bid_price"])
    ]
    df = df.drop_duplicates(subset=["timestamp_ms"], keep="last")
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    print(f"[clean] dropped {before - len(df):,} invalid rows -> {len(df):,} remaining")

    # Mid + spread
    df["mid"] = (df["best_bid_price"] + df["best_ask_price"]) / 2.0
    df["spread_mean"] = (df["best_ask_price"] - df["best_bid_price"]).astype("float32")

    # Segment IDs on gaps > GAP_SECONDS
    dt_s = df["timestamp_ms"].diff().fillna(0) / 1000.0
    seg_break = (dt_s > GAP_SECONDS).astype(np.int32)
    df["segment_id"] = seg_break.cumsum().astype("int32")
    n_segs = int(df["segment_id"].nunique())
    print(f"[segments] {n_segs} segments (gap > {GAP_SECONDS}s)")

    # OHLC from adjacent snapshots. Bar i bridges snapshot i -> i+1.
    mid = df["mid"].values
    n = len(df)
    open_ = mid.copy()
    close = np.roll(mid, -1)
    close[-1] = mid[-1]
    high = np.maximum(open_, close)
    low = np.minimum(open_, close)

    # If the next snapshot is in a new segment, the bar is degenerate: set
    # close = open, high = low = open. Prevents cross-segment leakage.
    seg = df["segment_id"].values
    seg_next = np.roll(seg, -1)
    seg_next[-1] = seg[-1]
    degenerate = seg_next != seg
    close[degenerate] = open_[degenerate]
    high[degenerate] = open_[degenerate]
    low[degenerate] = open_[degenerate]

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True),
        "timestamp_ms": df["timestamp_ms"].astype("int64"),
        "segment_id": df["segment_id"].astype("int32"),
        "open": open_.astype("float64"),
        "high": high.astype("float64"),
        "low": low.astype("float64"),
        "close": close.astype("float64"),
        "volume": np.zeros(n, dtype="float64"),
        "bbo_bid_qty": df["best_bid_qty"].astype("float64").values,
        "bbo_ask_qty": df["best_ask_qty"].astype("float64").values,
        "spread_mean": df["spread_mean"].values,
    })

    # Drop final row of last segment (no valid forward close)
    out = out.iloc[:-1].reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False, compression="snappy")
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"[write] {args.out} ({size_mb:.1f} MB, {len(out):,} bars)")

    # Audit summary
    print("\n=== AUDIT ===")
    print(f"rows            : {len(out):,}")
    print(f"date range      : {out['timestamp'].iloc[0]} -> {out['timestamp'].iloc[-1]}")
    print(f"segments        : {out['segment_id'].nunique():,}")
    print(f"spread median   : {np.median(out['spread_mean']):.4f} USD")
    print(f"spread p99      : {np.percentile(out['spread_mean'], 99):.4f} USD")
    print(f"mid median      : {np.median((out['open']+out['close'])/2):.2f}")
    ts = out["timestamp_ms"].values
    dt_s = np.diff(ts) / 1000.0
    # exclude segment breaks from cadence stats
    same_seg = np.diff(out["segment_id"].values) == 0
    dt_intra = dt_s[same_seg]
    print(f"intra-seg dt    : median={np.median(dt_intra):.2f}s  p95={np.percentile(dt_intra,95):.2f}s  max={dt_intra.max():.2f}s")
    print(f"degenerate bars : {int(degenerate.sum()):,} (last bar of each segment)")

    # Split coverage (match 10s config)
    splits = {
        "train": ("2025-10-24", "2025-11-03"),
        "val":   ("2026-02-07", "2026-02-24"),
        "test":  ("2026-03-15", "2026-03-21"),
    }
    print("\n=== SPLIT COVERAGE ===")
    for name, (a, b) in splits.items():
        start = pd.Timestamp(a, tz="UTC")
        end = pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1)
        sub = out[(out["timestamp"] >= start) & (out["timestamp"] < end)]
        if len(sub) == 0:
            print(f"{name:>5}: 0 bars — NO COVERAGE")
            continue
        print(f"{name:>5}: {len(sub):>9,} bars  {sub['timestamp'].iloc[0]} -> {sub['timestamp'].iloc[-1]}  segs={sub['segment_id'].nunique()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
