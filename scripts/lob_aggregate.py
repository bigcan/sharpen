"""
LOB 1s → 10s Aggregation with Microstructure Features.

Reads cleaned 1s LOB parquet (from lob_etl.py), aggregates to 10-second bars
with OHLCV (from mid-price) and LOB microstructure features. Respects
segment boundaries — never aggregates across data gaps.

Output columns:
  - timestamp, open, high, low, close, volume (mid-price OHLCV)
  - segment_id: data continuity segment
  - bbo_imbalance: mean BBO quantity imbalance over bar
  - depth_imbalance_5: mean top-5 level imbalance
  - depth_ratio_5: mean bid_depth_5 / (bid_depth_5 + ask_depth_5)
  - bbo_bid_qty, bbo_ask_qty: mean BBO quantities
  - spread_mean, spread_max: spread statistics
  - n_bbo_changes: count of BBO price changes (activity proxy)
  - bid_qty_delta, ask_qty_delta: BBO qty at end - start (queue change)
  - microprice_offset: mean (microprice - mid) / mid * 10000 (bps)

Usage:
    python scripts/lob_aggregate.py --symbol BTCUSDT --bar-size 10
    python scripts/lob_aggregate.py --symbol BTCUSDT --bar-size 5
"""
import argparse
import logging
import os
import time

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOB_DIR = os.path.join(PROJECT_ROOT, "data", "lob_parquet")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def aggregate_lob(
    input_path: str,
    output_path: str,
    bar_size_s: int = 10,
    min_ticks_per_bar: int = 3,
) -> str:
    """Aggregate 1s LOB ticks to N-second bars with microstructure features."""
    t0 = time.time()
    logger.info(f"Loading {input_path}...")
    df = pd.read_parquet(input_path)
    logger.info(f"  Loaded {len(df):,} rows, {df['segment_id'].nunique()} segments")

    # Pre-compute columns needed for aggregation
    # Mid price (already exists)
    if "mid_price" not in df.columns:
        df["mid_price"] = (df["best_bid_price"] + df["best_ask_price"]) / 2

    # Microprice
    total_qty = df["best_bid_qty"] + df["best_ask_qty"]
    df["microprice"] = np.where(
        total_qty > 0,
        (df["best_bid_price"] * df["best_ask_qty"]
         + df["best_ask_price"] * df["best_bid_qty"]) / total_qty,
        df["mid_price"],
    )
    df["microprice_offset_bps"] = (df["microprice"] - df["mid_price"]) / df["mid_price"] * 10000

    # BBO imbalance (already exists but recompute for safety)
    df["bbo_imbalance"] = np.where(
        total_qty > 0,
        (df["best_bid_qty"] - df["best_ask_qty"]) / total_qty,
        0.0,
    )

    # Top-5 level depth
    bid_depth_5 = sum(df[f"bid_q_{i}"].fillna(0) for i in range(5))
    ask_depth_5 = sum(df[f"ask_q_{i}"].fillna(0) for i in range(5))
    total_depth_5 = bid_depth_5 + ask_depth_5
    df["depth_imbalance_5"] = np.where(
        total_depth_5 > 0, (bid_depth_5 - ask_depth_5) / total_depth_5, 0.0,
    )
    df["depth_ratio_5"] = np.where(
        total_depth_5 > 0, bid_depth_5 / total_depth_5, 0.5,
    )

    # BBO change detection — per-segment diff to avoid cross-segment contamination (AUD-05)
    df["bid_changed"] = (
        df.groupby("segment_id")["best_bid_price"].diff().abs().fillna(0) > 1e-8
    ).astype(np.int32)
    df["ask_changed"] = (
        df.groupby("segment_id")["best_ask_price"].diff().abs().fillna(0) > 1e-8
    ).astype(np.int32)
    df["bbo_change"] = ((df["bid_changed"] + df["ask_changed"]) > 0).astype(np.int32)

    # Assign bar groups: floor(timestamp_ms / bar_size_ms) within each segment
    bar_size_ms = bar_size_s * 1000
    df["bar_group"] = df["timestamp_ms"] // bar_size_ms

    # Aggregate per (segment_id, bar_group) — never crosses segments
    logger.info(f"  Aggregating to {bar_size_s}s bars...")

    agg = df.groupby(["segment_id", "bar_group"]).agg(
        # OHLCV from mid-price
        timestamp_ms=("timestamp_ms", "first"),
        open=("mid_price", "first"),
        high=("mid_price", "max"),
        low=("mid_price", "min"),
        close=("mid_price", "last"),
        # Volume proxy: number of ticks (count) and total BBO qty
        n_ticks=("mid_price", "count"),
        volume=("bbo_change", "sum"),  # BBO changes as activity/volume proxy
        # LOB microstructure features
        bbo_imbalance=("bbo_imbalance", "mean"),
        depth_imbalance_5=("depth_imbalance_5", "mean"),
        depth_ratio_5=("depth_ratio_5", "mean"),
        bbo_bid_qty=("best_bid_qty", "mean"),
        bbo_ask_qty=("best_ask_qty", "mean"),
        spread_mean=("spread", "mean"),
        spread_max=("spread", "max"),
        microprice_offset=("microprice_offset_bps", "mean"),
        # For qty delta: first and last
        bid_qty_first=("best_bid_qty", "first"),
        bid_qty_last=("best_bid_qty", "last"),
        ask_qty_first=("best_ask_qty", "first"),
        ask_qty_last=("best_ask_qty", "last"),
    ).reset_index()

    # Drop bars with too few ticks (partial bars at segment boundaries)
    n_before = len(agg)
    agg = agg[agg["n_ticks"] >= min_ticks_per_bar].reset_index(drop=True)
    logger.info(f"  Dropped {n_before - len(agg)} bars with <{min_ticks_per_bar} ticks")

    # Compute qty deltas
    agg["bid_qty_delta"] = agg["bid_qty_last"] - agg["bid_qty_first"]
    agg["ask_qty_delta"] = agg["ask_qty_last"] - agg["ask_qty_first"]

    # Convert timestamp
    agg["timestamp"] = pd.to_datetime(agg["timestamp_ms"], unit="ms", utc=True)

    # Select and order final columns
    out_cols = [
        "timestamp", "timestamp_ms", "segment_id",
        "open", "high", "low", "close", "volume",
        "bbo_imbalance", "depth_imbalance_5", "depth_ratio_5",
        "bbo_bid_qty", "bbo_ask_qty",
        "spread_mean", "spread_max",
        "n_bbo_changes",
        "bid_qty_delta", "ask_qty_delta",
        "microprice_offset",
    ]
    # Rename volume → n_bbo_changes for clarity, and add a proper volume col
    agg = agg.rename(columns={"volume": "n_bbo_changes"})
    # Volume = total BBO quantity transacted (bid+ask mean × ticks as proxy)
    agg["volume"] = (agg["bbo_bid_qty"] + agg["bbo_ask_qty"]) * agg["n_ticks"]

    out_cols = [
        "timestamp", "timestamp_ms", "segment_id",
        "open", "high", "low", "close", "volume",
        "bbo_imbalance", "depth_imbalance_5", "depth_ratio_5",
        "bbo_bid_qty", "bbo_ask_qty",
        "spread_mean", "spread_max",
        "n_bbo_changes",
        "bid_qty_delta", "ask_qty_delta",
        "microprice_offset",
    ]
    agg = agg[out_cols].copy()

    # Sort
    agg = agg.sort_values(["segment_id", "timestamp_ms"]).reset_index(drop=True)

    # Downcast for space
    float32_cols = [
        "bbo_imbalance", "depth_imbalance_5", "depth_ratio_5",
        "spread_mean", "spread_max", "microprice_offset",
        "bid_qty_delta", "ask_qty_delta",
    ]
    for col in float32_cols:
        agg[col] = agg[col].astype(np.float32)
    agg["n_bbo_changes"] = agg["n_bbo_changes"].astype(np.int16)
    agg["segment_id"] = agg["segment_id"].astype(np.int16)

    # Export
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    agg.to_parquet(output_path, index=False, engine="pyarrow")

    elapsed = time.time() - t0
    file_mb = os.path.getsize(output_path) / 1e6

    # Summary per segment
    seg_stats = agg.groupby("segment_id").agg(
        n_bars=("close", "count"),
        first_ts=("timestamp", "min"),
        last_ts=("timestamp", "max"),
    )

    logger.info(f"{'='*60}")
    logger.info("AGGREGATION COMPLETE")
    logger.info(f"  Input:  {len(df):>12,} ticks (1s)")
    logger.info(f"  Output: {len(agg):>12,} bars ({bar_size_s}s)")
    logger.info(f"  Ratio:  {len(df)/len(agg):.1f}:1")
    logger.info(f"  Segments: {agg['segment_id'].nunique()}")
    logger.info(f"  Columns: {len(agg.columns)}")
    logger.info(f"  File: {file_mb:.1f} MB")
    logger.info(f"  Time: {elapsed:.1f}s")
    logger.info(f"  Output: {output_path}")
    logger.info("\nSegment details:")
    for seg_id, row in seg_stats.iterrows():
        logger.info(
            f"  Seg {seg_id:2d}: {str(row['first_ts'])[:19]} → {str(row['last_ts'])[:19]} "
            f"| {row['n_bars']:>8,} bars",
        )
    logger.info(f"{'='*60}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="LOB 1s → Ns aggregation")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--bar-size", type=int, default=10, help="Bar size in seconds")
    parser.add_argument("--input", default=None, help="Override input parquet path")
    parser.add_argument("--output", default=None, help="Override output parquet path")
    parser.add_argument("--min-ticks", type=int, default=3, help="Min ticks per bar")
    args = parser.parse_args()

    sym = args.symbol.upper()
    input_path = args.input or os.path.join(LOB_DIR, f"{sym.lower()}_lob_1s.parquet")
    output_path = args.output or os.path.join(
        OUTPUT_DIR, f"{sym.lower()}_lob_{args.bar_size}s.parquet",
    )

    aggregate_lob(input_path, output_path, bar_size_s=args.bar_size, min_ticks_per_bar=args.min_ticks)


if __name__ == "__main__":
    main()
