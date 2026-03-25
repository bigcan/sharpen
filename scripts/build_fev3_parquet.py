#!/usr/bin/env python
"""
EXP-F1: Build Feature Engineering v3 Parquet
=============================================

Preprocessing script that computes cross-timeframe (1-min → 5-min) aggregate
features and merges them into an existing 5-min parquet file.

New columns added (5 cross-TF features):
  obi_burst        — max(|total_obi_1min|) per 5-min window  [0,1]
  obi_trend        — linear regression slope of OBI over 5 one-min bars
  dofi_burst       — max(|total_dofi_1min|) per 5-min window
  microprice_range — (max-min)(microprice_basis_1min) per window
  spread_max       — max(spread_bps_1min) per window

The 5 acceleration features (obi_accel, dofi_accel, microprice_accel,
spread_velocity, depth_drain) are computed inline by process_micro()
in feature_engineering.py — they don't need pre-computation here.

Usage:
  python scripts/build_fev3_parquet.py \
    --input_1min data/processed/btc_2025_full_year.parquet \
    --input_5min data/processed/btc_2025_full_year_5min.parquet \
    --output data/processed/btc_2025_full_year_5min_fev3.parquet
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())


def compute_1min_raw_features(df_1min: pd.DataFrame) -> pd.DataFrame:
    """Compute raw (un-normalized) LOB features at 1-min resolution.

    Uses the same formulas as feature_engineering.process_micro() but
    WITHOUT normalization — we only need raw values for aggregation.
    """
    bp1 = df_1min['bid_price_1'].values.astype(np.float64)
    ap1 = df_1min['ask_price_1'].values.astype(np.float64)
    mid = (bp1 + ap1) / 2.0
    mid_safe = np.where(mid > 0, mid, 1e-9)

    # ── total_obi: 5-level book imbalance [-1, 1] ──
    total_bid_vol = np.zeros(len(df_1min), dtype=np.float64)
    total_ask_vol = np.zeros(len(df_1min), dtype=np.float64)
    for i in range(1, 6):
        bcol = f'bid_vol_{i}'
        acol = f'ask_vol_{i}'
        if bcol in df_1min.columns and acol in df_1min.columns:
            total_bid_vol += df_1min[bcol].values.astype(np.float64)
            total_ask_vol += df_1min[acol].values.astype(np.float64)
    total_vol_sum = total_bid_vol + total_ask_vol
    obi = np.clip(
        (total_bid_vol - total_ask_vol) / (total_vol_sum + 1e-8),
        -1.0, 1.0
    )

    # ── total_dofi: sum of DOFI across 5 levels ──
    total_dofi = np.zeros(len(df_1min), dtype=np.float64)
    for i in range(1, 6):
        bp = df_1min[f'bid_price_{i}'].values.astype(np.float64)
        bv = df_1min[f'bid_vol_{i}'].values.astype(np.float64)
        ap = df_1min[f'ask_price_{i}'].values.astype(np.float64)
        av = df_1min[f'ask_vol_{i}'].values.astype(np.float64)

        bp_prev = np.roll(bp, 1)
        bp_prev[0] = bp[0]
        bv_prev = np.roll(bv, 1)
        bv_prev[0] = bv[0]
        ap_prev = np.roll(ap, 1)
        ap_prev[0] = ap[0]
        av_prev = np.roll(av, 1)
        av_prev[0] = av[0]

        w_b = np.where(bp > bp_prev, bv,
                np.where(bp < bp_prev, -bv_prev, bv - bv_prev))
        w_a = np.where(ap < ap_prev, av,
                np.where(ap > ap_prev, -av_prev, av - av_prev))
        total_dofi += (w_b - w_a)

    # ── microprice_basis (bps) ──
    bv1 = df_1min['bid_vol_1'].values.astype(np.float64)
    av1 = df_1min['ask_vol_1'].values.astype(np.float64)
    vol_sum = bv1 + av1
    microprice = (bp1 * av1 + ap1 * bv1) / (vol_sum + 1e-8)
    mp_safe = np.where(microprice > 0, microprice, mid_safe)
    microprice_basis = np.log(mp_safe / mid_safe) * 10000.0

    # ── spread_bps ──
    spread_bps = ((ap1 - bp1) / mid_safe) * 10000.0

    result = pd.DataFrame({
        'timestamp': df_1min['timestamp'].values,
        'total_obi_1min': obi,
        'total_dofi_1min': total_dofi,
        'microprice_basis_1min': microprice_basis,
        'spread_bps_1min': spread_bps,
    })
    return result


def aggregate_to_5min(df_1min_features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-min features to 5-min windows.

    Timestamp alignment: floor('5min') groups 1-min bars into 5-min windows.
    """
    df = df_1min_features.copy()
    df['window_5min'] = df['timestamp'].dt.floor('5min')

    agg = df.groupby('window_5min').agg(
        # obi_burst: max absolute OBI within window [0, 1]
        obi_burst=('total_obi_1min', lambda x: np.max(np.abs(x))),
        # obi_trend: linear regression slope of OBI over bars in window
        obi_trend=('total_obi_1min', lambda x: _polyfit_slope(x.values)),
        # dofi_burst: max absolute DOFI within window
        dofi_burst=('total_dofi_1min', lambda x: np.max(np.abs(x))),
        # microprice_range: max - min of microprice_basis within window
        microprice_range=('microprice_basis_1min', lambda x: x.max() - x.min()),
        # spread_max: max spread within window
        spread_max=('spread_bps_1min', 'max'),
    ).reset_index()

    agg.rename(columns={'window_5min': 'timestamp'}, inplace=True)
    return agg


def _polyfit_slope(values: np.ndarray) -> float:
    """Compute linear regression slope over an array of values.

    For short arrays (< 2 points), returns 0.0.
    Uses np.polyfit degree 1 — robust and fast for small N.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    try:
        coeffs = np.polyfit(x, values.astype(np.float64), 1)
        return float(coeffs[0])  # slope
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def merge_and_save(df_5min: pd.DataFrame, agg_5min: pd.DataFrame,
                   output_path: str) -> pd.DataFrame:
    """Merge aggregated cross-TF features onto existing 5-min parquet."""
    # Ensure timestamps are compatible
    df_5min = df_5min.copy()
    df_5min['timestamp'] = pd.to_datetime(df_5min['timestamp'])
    agg_5min['timestamp'] = pd.to_datetime(agg_5min['timestamp'])

    # Left merge: keep all 5-min rows, add cross-TF columns where available
    merged = df_5min.merge(agg_5min, on='timestamp', how='left')

    # Fill any missing cross-TF values with 0.0 (e.g., if 1-min data gaps)
    cross_tf_cols = ['obi_burst', 'obi_trend', 'dofi_burst',
                     'microprice_range', 'spread_max']
    for col in cross_tf_cols:
        merged[col] = merged[col].fillna(0.0).astype(np.float32)

    # Validation
    n_orig = len(df_5min)
    n_merged = len(merged)
    assert n_merged == n_orig, (
        f"Row count mismatch after merge: {n_orig} -> {n_merged}"
    )

    # Check for unexpected NaN in cross-TF columns
    for col in cross_tf_cols:
        nan_count = merged[col].isna().sum()
        assert nan_count == 0, f"NaN found in {col}: {nan_count} rows"

    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    merged.to_parquet(output_path, index=False, engine='pyarrow')
    print(f"  Saved: {output_path}")
    print(f"  Shape: {merged.shape}")
    print(f"  Columns: {len(merged.columns)}")

    return merged


def validate_output(merged: pd.DataFrame, df_5min_orig: pd.DataFrame):
    """Post-merge validation checks."""
    cross_tf_cols = ['obi_burst', 'obi_trend', 'dofi_burst',
                     'microprice_range', 'spread_max']

    print("\n  Validation:")
    print(f"    Row count: {len(merged)} (original: {len(df_5min_orig)}) "
          f"{'OK' if len(merged) == len(df_5min_orig) else 'FAIL'}")

    # Timestamp alignment
    ts_match = (merged['timestamp'].values == df_5min_orig['timestamp'].values).all()
    print(f"    Timestamps match: {'OK' if ts_match else 'FAIL'}")

    # Value ranges
    for col in cross_tf_cols:
        vals = merged[col]
        print(f"    {col:>20s}: min={vals.min():>10.4f}, "
              f"max={vals.max():>10.4f}, "
              f"mean={vals.mean():>10.4f}, "
              f"zeros={int((vals == 0).sum()):>6d}/{len(vals)}")

    # obi_burst should be in [0, 1]
    obi_burst = merged['obi_burst']
    assert obi_burst.min() >= 0 and obi_burst.max() <= 1.0 + 1e-6, (
        f"obi_burst out of [0,1]: [{obi_burst.min()}, {obi_burst.max()}]"
    )
    print("    obi_burst range [0,1]: OK")

    print("  All validation checks passed.")


def main():
    parser = argparse.ArgumentParser(
        description="EXP-F1: Build Feature Engineering v3 Parquet"
    )
    parser.add_argument("--input_1min", type=str, required=True,
                        help="Path to 1-min LOB parquet")
    parser.add_argument("--input_5min", type=str, required=True,
                        help="Path to existing 5-min parquet (v2)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for fev3 parquet")
    args = parser.parse_args()

    print("=" * 70)
    print("  EXP-F1: Build Feature Engineering v3 Parquet")
    print("=" * 70)

    # ── 1. Load 1-min data ──
    print(f"\n[1/5] Loading 1-min data: {args.input_1min}")
    df_1min = pd.read_parquet(args.input_1min)
    df_1min['timestamp'] = pd.to_datetime(df_1min['timestamp'])
    print(f"  Rows: {len(df_1min)}, Columns: {len(df_1min.columns)}")

    # Validate required columns
    required = ['timestamp', 'bid_price_1', 'ask_price_1',
                 'bid_vol_1', 'ask_vol_1']
    missing = [c for c in required if c not in df_1min.columns]
    if missing:
        raise ValueError(f"Missing columns in 1-min data: {missing}")

    # ── 2. Compute raw 1-min LOB features ──
    print("\n[2/5] Computing raw 1-min LOB features...")
    df_1min_feat = compute_1min_raw_features(df_1min)
    print(f"  Computed {len(df_1min_feat)} rows of 1-min features")

    # ── 3. Aggregate to 5-min windows ──
    print("\n[3/5] Aggregating to 5-min windows...")
    agg_5min = aggregate_to_5min(df_1min_feat)
    print(f"  Aggregated to {len(agg_5min)} 5-min windows")

    # ── 4. Load 5-min data and merge ──
    print(f"\n[4/5] Loading 5-min data: {args.input_5min}")
    df_5min = pd.read_parquet(args.input_5min)
    df_5min['timestamp'] = pd.to_datetime(df_5min['timestamp'])
    print(f"  Rows: {len(df_5min)}, Columns: {len(df_5min.columns)}")

    print("\n  Merging cross-TF features...")
    merged = merge_and_save(df_5min, agg_5min, args.output)

    # ── 5. Validate ──
    print("\n[5/5] Validating output...")
    validate_output(merged, df_5min)

    print(f"\n{'=' * 70}")
    print(f"  fev3 parquet built successfully: {args.output}")
    print(f"  Original 5-min columns: {len(df_5min.columns)}")
    print(f"  fev3 columns: {len(merged.columns)} (+5 cross-TF)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
