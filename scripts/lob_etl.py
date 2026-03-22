"""
LOB SQLite → Parquet ETL with Data Cleaning.

Reads raw LOB snapshots from secure_lob_collection.db, cleans,
and exports per-symbol Parquet files with exploded LOB levels.

Cleaning steps:
  1. Drop corrupt price rows (cross-symbol data leaks)
  2. Drop micro-segments (<min_segment_hours)
  3. Dedup sub-second timestamps (keep last per rounded second)
  4. Normalize LOB depth to N levels (pad short books with NaN)
  5. Explode JSON arrays → columnar (bid_p_0..N, bid_q_0..N, etc.)
  6. Tag each row with segment_id for episode boundary awareness
  7. Validate: no crossed books, monotonic LOB, spread sanity

Usage:
    python scripts/lob_etl.py --symbol BTCUSDT
    python scripts/lob_etl.py --symbol BTCUSDT --max-depth 10
    python scripts/lob_etl.py --all
"""
import argparse
import json
import logging
import os
import time
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "secure_lob_collection.db")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "lob_parquet")

# --- Price sanity bounds (used to detect cross-symbol leaks) ---
PRICE_BOUNDS = {
    "BTCUSDT": (10_000, 200_000),
    "ETHUSDT": (500, 20_000),
    "BNBUSDT": (100, 2_000),
    "SOLUSDT": (10, 1_000),
    "XRPUSDT": (0.1, 50),
}

# All crypto symbols in the DB
ALL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]


def load_symbol_from_db(
    db_path: str, symbol: str, chunk_size: int = 500_000
) -> pd.DataFrame:
    """Stream-load a symbol from SQLite in chunks to manage memory."""
    import sqlite3

    logger.info(f"Loading {symbol} from {db_path}...")
    t0 = time.time()

    conn = sqlite3.connect(db_path)

    # Scalar columns we always need
    cols = [
        "timestamp_ms", "best_bid_price", "best_bid_qty",
        "best_ask_price", "best_ask_qty", "spread",
        "bid_prices", "bid_quantities", "ask_prices", "ask_quantities",
        "bid_levels_count", "ask_levels_count", "data_quality_score",
        "collection_batch_id",
    ]
    col_str = ", ".join(cols)
    query = f"SELECT {col_str} FROM lob_snapshots WHERE symbol=? ORDER BY timestamp_ms"

    chunks = []
    for chunk in pd.read_sql_query(query, conn, params=(symbol,), chunksize=chunk_size):
        chunks.append(chunk)
        logger.info(f"  Loaded {sum(len(c) for c in chunks):,} rows...")

    conn.close()

    if not chunks:
        logger.error(f"No data found for {symbol}")
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    elapsed = time.time() - t0
    logger.info(f"  Loaded {len(df):,} rows in {elapsed:.1f}s")
    return df


def clean_prices(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Drop rows with prices outside expected bounds (cross-symbol leaks)."""
    lo, hi = PRICE_BOUNDS.get(symbol, (0, 1e9))
    n_before = len(df)

    mask = (
        (df["best_bid_price"] >= lo) & (df["best_bid_price"] <= hi) &
        (df["best_ask_price"] >= lo) & (df["best_ask_price"] <= hi)
    )
    df = df[mask].reset_index(drop=True)

    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning(f"  Dropped {n_dropped} rows with prices outside [{lo}, {hi}]")
    else:
        logger.info(f"  Price bounds check: all {n_before:,} rows OK")
    return df


def clean_crossed_books(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where bid >= ask (crossed/locked book)."""
    n_before = len(df)
    df = df[df["best_bid_price"] < df["best_ask_price"]].reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning(f"  Dropped {n_dropped} crossed/locked book rows")
    return df


def dedup_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Dedup rows with same timestamp_ms, keeping last (most recent snapshot)."""
    n_before = len(df)
    df = df.drop_duplicates(subset=["timestamp_ms"], keep="last").reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.info(f"  Deduped {n_dropped} rows with duplicate timestamps")
    return df


def assign_segments(
    df: pd.DataFrame, gap_threshold_s: float = 60.0
) -> pd.DataFrame:
    """Assign segment_id based on timestamp gaps > threshold."""
    deltas_ms = df["timestamp_ms"].diff()
    is_gap = deltas_ms > (gap_threshold_s * 1000)
    df["segment_id"] = is_gap.cumsum().astype(np.int32)
    n_segments = df["segment_id"].nunique()
    logger.info(f"  Found {n_segments} segments (gap threshold={gap_threshold_s}s)")
    return df


def filter_short_segments(
    df: pd.DataFrame, min_hours: float = 1.0
) -> pd.DataFrame:
    """Drop segments shorter than min_hours."""
    n_before = len(df)
    seg_stats = df.groupby("segment_id")["timestamp_ms"].agg(["min", "max", "count"])
    seg_stats["duration_h"] = (seg_stats["max"] - seg_stats["min"]) / 3_600_000
    keep_segs = seg_stats[seg_stats["duration_h"] >= min_hours].index
    df = df[df["segment_id"].isin(keep_segs)].reset_index(drop=True)

    n_dropped = n_before - len(df)
    n_segs_dropped = seg_stats.shape[0] - len(keep_segs)
    logger.info(
        f"  Dropped {n_segs_dropped} segments (<{min_hours}h) = {n_dropped:,} rows. "
        f"Kept {len(keep_segs)} segments, {len(df):,} rows."
    )

    # Re-assign sequential segment IDs
    old_ids = sorted(df["segment_id"].unique())
    id_map = {old: new for new, old in enumerate(old_ids)}
    df["segment_id"] = df["segment_id"].map(id_map).astype(np.int32)

    return df


def explode_lob_levels(
    df: pd.DataFrame, max_depth: int = 20
) -> pd.DataFrame:
    """Parse JSON LOB arrays and explode into columnar format.

    Creates columns: bid_p_0..bid_p_{max_depth-1}, bid_q_0..{}, ask_p_0..{}, ask_q_0..{}
    Pads shorter books with NaN.
    """
    logger.info(f"  Exploding LOB JSON arrays to {max_depth} levels...")
    t0 = time.time()

    n = len(df)

    # Pre-allocate arrays
    bid_p = np.full((n, max_depth), np.nan, dtype=np.float64)
    bid_q = np.full((n, max_depth), np.nan, dtype=np.float64)
    ask_p = np.full((n, max_depth), np.nan, dtype=np.float64)
    ask_q = np.full((n, max_depth), np.nan, dtype=np.float64)

    # Vectorized JSON parse — pandas str operations are slow, use list comp
    def parse_json_col(series):
        """Parse JSON strings to lists. Handles both str and already-parsed."""
        results = []
        for val in series:
            if val is None:
                results.append([])
            elif isinstance(val, str):
                try:
                    results.append(json.loads(val))
                except (json.JSONDecodeError, ValueError):
                    results.append([])
            elif isinstance(val, list):
                results.append(val)
            else:
                results.append([])
        return results

    bp_lists = parse_json_col(df["bid_prices"].values)
    bq_lists = parse_json_col(df["bid_quantities"].values)
    ap_lists = parse_json_col(df["ask_prices"].values)
    aq_lists = parse_json_col(df["ask_quantities"].values)

    for i in range(n):
        bp = bp_lists[i]
        bq = bq_lists[i]
        ap = ap_lists[i]
        aq = aq_lists[i]

        d = min(len(bp), max_depth)
        if d > 0:
            bid_p[i, :d] = bp[:d]
            bid_q[i, :d] = bq[:d]
        d = min(len(ap), max_depth)
        if d > 0:
            ask_p[i, :d] = ap[:d]
            ask_q[i, :d] = aq[:d]

    # Build column DataFrames
    bp_cols = {f"bid_p_{j}": bid_p[:, j] for j in range(max_depth)}
    bq_cols = {f"bid_q_{j}": bid_q[:, j] for j in range(max_depth)}
    ap_cols = {f"ask_p_{j}": ask_p[:, j] for j in range(max_depth)}
    aq_cols = {f"ask_q_{j}": ask_q[:, j] for j in range(max_depth)}

    lob_df = pd.DataFrame({**bp_cols, **bq_cols, **ap_cols, **aq_cols})

    # Drop original JSON columns and join exploded
    df = df.drop(columns=["bid_prices", "bid_quantities", "ask_prices", "ask_quantities"])
    df = pd.concat([df.reset_index(drop=True), lob_df], axis=1)

    elapsed = time.time() - t0
    logger.info(f"  LOB explosion done in {elapsed:.1f}s — {max_depth * 4} new columns")
    return df


def validate_lob_monotonicity(df: pd.DataFrame, max_depth: int = 20) -> int:
    """Check that bid prices are descending and ask prices are ascending."""
    violations = 0

    # Bid: level 0 should be highest
    for j in range(min(max_depth - 1, 9)):  # Check first 10 levels
        col_hi = f"bid_p_{j}"
        col_lo = f"bid_p_{j+1}"
        if col_hi in df.columns and col_lo in df.columns:
            mask = df[col_lo].notna() & df[col_hi].notna()
            bad = (df.loc[mask, col_lo] > df.loc[mask, col_hi]).sum()
            violations += bad

    # Ask: level 0 should be lowest
    for j in range(min(max_depth - 1, 9)):
        col_lo = f"ask_p_{j}"
        col_hi = f"ask_p_{j+1}"
        if col_lo in df.columns and col_hi in df.columns:
            mask = df[col_lo].notna() & df[col_hi].notna()
            bad = (df.loc[mask, col_hi] < df.loc[mask, col_lo]).sum()
            violations += bad

    return violations


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add commonly useful derived columns from BBO."""
    df["mid_price"] = (df["best_bid_price"] + df["best_ask_price"]) / 2
    df["spread_bps"] = (df["spread"] / df["mid_price"]) * 10_000
    df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)

    # Imbalance at BBO
    total_qty = df["best_bid_qty"] + df["best_ask_qty"]
    df["bbo_imbalance"] = np.where(
        total_qty > 0,
        (df["best_bid_qty"] - df["best_ask_qty"]) / total_qty,
        0.0,
    )
    return df


def export_symbol(
    symbol: str,
    db_path: str = DB_PATH,
    output_dir: str = OUTPUT_DIR,
    max_depth: int = 20,
    min_segment_hours: float = 1.0,
    gap_threshold_s: float = 60.0,
) -> Optional[str]:
    """Full ETL pipeline for one symbol."""
    logger.info(f"{'='*60}")
    logger.info(f"ETL START: {symbol}")
    logger.info(f"{'='*60}")
    t_total = time.time()

    # 1. Load
    df = load_symbol_from_db(db_path, symbol)
    if df.empty:
        return None
    n_raw = len(df)

    # 2. Clean prices
    df = clean_prices(df, symbol)

    # 3. Clean crossed books
    df = clean_crossed_books(df)

    # 4. Dedup timestamps
    df = dedup_timestamps(df)

    # 5. Assign segments
    df = assign_segments(df, gap_threshold_s=gap_threshold_s)

    # 6. Filter short segments
    df = filter_short_segments(df, min_hours=min_segment_hours)
    if df.empty:
        logger.error(f"No usable segments for {symbol}")
        return None

    # 7. Explode LOB levels
    df = explode_lob_levels(df, max_depth=max_depth)

    # 8. Derived features
    df = add_derived_features(df)

    # 9. Validate
    logger.info("  Validating...")
    mono_violations = validate_lob_monotonicity(df, max_depth=max_depth)
    if mono_violations > 0:
        logger.warning(f"  LOB monotonicity violations: {mono_violations}")
    else:
        logger.info("  LOB monotonicity: OK")

    spread_outliers = (df["spread_bps"] > 100).sum()
    if spread_outliers > 0:
        logger.warning(f"  Spread > 100bps: {spread_outliers} rows")

    # 10. Drop intermediate columns, set dtypes
    drop_cols = ["bid_levels_count", "ask_levels_count", "collection_batch_id"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Downcast floats where safe
    for col in ["data_quality_score", "spread_bps", "bbo_imbalance"]:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)

    # 11. Export
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{symbol.lower()}_lob_1s.parquet")
    df.to_parquet(out_path, index=False, engine="pyarrow")

    file_size_mb = os.path.getsize(out_path) / 1e6
    elapsed = time.time() - t_total

    # Summary
    n_segments = df["segment_id"].nunique()
    total_hours = (
        df.groupby("segment_id")["timestamp_ms"]
        .apply(lambda x: (x.max() - x.min()) / 3_600_000)
        .sum()
    )

    logger.info(f"{'='*60}")
    logger.info(f"ETL COMPLETE: {symbol}")
    logger.info(f"  Raw rows:      {n_raw:>12,}")
    logger.info(f"  Clean rows:    {len(df):>12,}")
    logger.info(f"  Dropped:       {n_raw - len(df):>12,} ({(n_raw - len(df))/n_raw*100:.1f}%)")
    logger.info(f"  Segments:      {n_segments:>12}")
    logger.info(f"  Total hours:   {total_hours:>12.1f}")
    logger.info(f"  LOB depth:     {max_depth:>12} levels")
    logger.info(f"  Columns:       {len(df.columns):>12}")
    logger.info(f"  File size:     {file_size_mb:>12.1f} MB")
    logger.info(f"  Time:          {elapsed:>12.1f}s")
    logger.info(f"  Output:        {out_path}")
    logger.info(f"{'='*60}")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="LOB SQLite → Parquet ETL with cleaning"
    )
    parser.add_argument(
        "--symbol", type=str, default="BTCUSDT",
        help="Symbol to export (default: BTCUSDT)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Export all crypto symbols",
    )
    parser.add_argument(
        "--db", type=str, default=DB_PATH,
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
        help="Output directory for Parquet files",
    )
    parser.add_argument(
        "--max-depth", type=int, default=20,
        help="Max LOB depth to export (default: 20)",
    )
    parser.add_argument(
        "--min-segment-hours", type=float, default=1.0,
        help="Minimum segment duration in hours (default: 1.0)",
    )
    parser.add_argument(
        "--gap-threshold", type=float, default=60.0,
        help="Gap threshold in seconds for segment detection (default: 60)",
    )

    args = parser.parse_args()

    symbols = ALL_SYMBOLS if args.all else [args.symbol.upper()]

    for sym in symbols:
        export_symbol(
            symbol=sym,
            db_path=args.db,
            output_dir=args.output_dir,
            max_depth=args.max_depth,
            min_segment_hours=args.min_segment_hours,
            gap_threshold_s=args.gap_threshold,
        )


if __name__ == "__main__":
    main()
