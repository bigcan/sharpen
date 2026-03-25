#!/usr/bin/env python
"""
EXP-E2a: Oracle Bar Profiling
==============================

Tags every 5-min bar as oracle-long / oracle-short / oracle-hold based on
1-step forward return vs taker fee threshold. Computes per-group feature
statistics and discriminability (Cohen's d, KS test) to identify which
features naturally separate profitable bars.

This answers: "What do features look like when alpha is present?"

Usage:
  python scripts/oracle_bar_profile.py --config configs/postaudit_hpo.yaml
  python scripts/oracle_bar_profile.py --config configs/postaudit_hpo.yaml --fee_bps 5
"""
import argparse
import os
import sys
import yaml
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())

from finrl_pro_ds.data.parquet_handler import ParquetDataHandler
from finrl_pro_ds.data.feature_engineering import MICRO_FEATURE_COLS, MACRO_FEATURE_COLS
from scipy import stats


def load_split(config, start_date, end_date, norm_cutoff_date=None):
    """Load a single split via ParquetDataHandler, return features + mid_price."""
    data_config = config.get("data", {})
    file_path = data_config.get("file_path")
    ticker = data_config.get("ticker", "BTCUSDT")

    handler = ParquetDataHandler(
        file_path=file_path,
        ticker=ticker,
        feature_config=config.get("features", {}),
        start_date=start_date,
        end_date=end_date,
        norm_cutoff_date=norm_cutoff_date,
    )

    feature_cols = list(MICRO_FEATURE_COLS) + list(MACRO_FEATURE_COLS)
    missing = [c for c in feature_cols if c not in handler._data_arrays]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = np.column_stack([handler._data_arrays[c] for c in feature_cols])
    mid = handler._data_arrays['mid_price'].copy()
    timestamps = handler._data_arrays['timestamp'].copy()

    print(f"  Loaded {len(mid)} rows ({start_date} -> {end_date}), X shape: {X.shape}")
    handler.close()
    return X, mid, timestamps, feature_cols


def classify_oracle_bars(mid, fee_bps=5.0):
    """Classify each bar by 1-step forward return vs fee threshold.

    Args:
        mid: mid_price array
        fee_bps: one-way taker fee in bps (default 5 = 10bps RT)

    Returns:
        labels: array of 'long', 'short', 'hold' strings
        returns_bps: 1-step forward returns in bps
    """
    n = len(mid)
    returns_bps = np.zeros(n, dtype=np.float64)

    # 1-step forward return in bps
    mid_safe = np.where(mid > 0, mid, 1e-9)
    returns_bps[:-1] = ((mid[1:] - mid[:-1]) / mid_safe[:-1]) * 10000.0
    returns_bps[-1] = 0.0  # Last bar has no forward return

    # Classify: net of round-trip fee
    threshold = fee_bps  # One-way threshold (entry fee)
    labels = np.full(n, 'hold', dtype='U5')
    labels[returns_bps > threshold] = 'long'
    labels[returns_bps < -threshold] = 'short'

    return labels, returns_bps


def compute_group_stats(X, labels, feature_cols, returns_bps):
    """Compute per-group statistics for each feature.

    Returns:
        stats_df: DataFrame with feature, group means/stds, Cohen's d, KS p-value
    """
    long_mask = labels == 'long'
    short_mask = labels == 'short'
    hold_mask = labels == 'hold'

    n_long = long_mask.sum()
    n_short = short_mask.sum()
    n_hold = hold_mask.sum()
    n_total = len(labels)

    print("\n  Oracle distribution:")
    print(f"    Long:  {n_long:>6d} ({100*n_long/n_total:.1f}%)")
    print(f"    Short: {n_short:>6d} ({100*n_short/n_total:.1f}%)")
    print(f"    Hold:  {n_hold:>6d} ({100*n_hold/n_total:.1f}%)")
    print(f"    Total: {n_total:>6d}")

    print("\n  Return stats (bps):")
    print(f"    Long  mean: {returns_bps[long_mask].mean():>+7.2f}, "
          f"median: {np.median(returns_bps[long_mask]):>+7.2f}")
    print(f"    Short mean: {returns_bps[short_mask].mean():>+7.2f}, "
          f"median: {np.median(returns_bps[short_mask]):>+7.2f}")
    print(f"    Hold  mean: {returns_bps[hold_mask].mean():>+7.2f}, "
          f"median: {np.median(returns_bps[hold_mask]):>+7.2f}")

    rows = []
    for i, col in enumerate(feature_cols):
        feat_long = X[long_mask, i]
        feat_short = X[short_mask, i]
        feat_hold = X[hold_mask, i]
        feat_all = X[:, i]

        # Group means
        mean_long = feat_long.mean()
        mean_short = feat_short.mean()
        mean_hold = feat_hold.mean()
        mean_all = feat_all.mean()

        # Group stds
        std_long = feat_long.std()
        std_short = feat_short.std()

        # Cohen's d: long vs short effect size
        pooled_std = np.sqrt((std_long**2 + std_short**2) / 2.0)
        cohens_d = (mean_long - mean_short) / (pooled_std + 1e-10)

        # Directional signal: mean_long vs mean_short (positive = feature higher for longs)
        signal_direction = "long>short" if mean_long > mean_short else "short>long"

        # KS test: long vs short distributions
        ks_stat, ks_pval = stats.ks_2samp(feat_long, feat_short)

        # Mean absolute deviation from hold (how much does the feature shift
        # when there's a trade opportunity)
        mad_long = abs(mean_long - mean_hold)
        mad_short = abs(mean_short - mean_hold)
        trade_signal = (mad_long + mad_short) / 2.0

        rows.append({
            'feature': col,
            'mean_long': mean_long,
            'mean_short': mean_short,
            'mean_hold': mean_hold,
            'mean_all': mean_all,
            'std_long': std_long,
            'std_short': std_short,
            'cohens_d': cohens_d,
            'abs_cohens_d': abs(cohens_d),
            'ks_stat': ks_stat,
            'ks_pval': ks_pval,
            'signal_direction': signal_direction,
            'trade_signal': trade_signal,
        })

    stats_df = pd.DataFrame(rows)
    return stats_df


def compute_temporal_stats(X, labels, feature_cols, lookback=5):
    """Compute how feature values BEFORE a profitable bar differ from average.

    For each feature, look at the mean value at t-1, t-2, ... t-lookback
    before long/short bars vs before hold bars. This tests whether features
    have predictive lead time.
    """
    long_mask = labels == 'long'
    short_mask = labels == 'short'

    rows = []
    for i, col in enumerate(feature_cols):
        feat = X[:, i]
        for lag in range(1, lookback + 1):
            # Get feature values `lag` steps before each bar type
            valid_long = long_mask.copy()
            valid_short = short_mask.copy()
            valid_long[:lag] = False
            valid_short[:lag] = False

            if valid_long.sum() < 100 or valid_short.sum() < 100:
                continue

            # Feature value at t-lag for bars that become long/short at t
            long_indices = np.where(valid_long)[0] - lag
            short_indices = np.where(valid_short)[0] - lag

            feat_before_long = feat[long_indices]
            feat_before_short = feat[short_indices]

            cohens_d = ((feat_before_long.mean() - feat_before_short.mean()) /
                        (np.sqrt((feat_before_long.std()**2 + feat_before_short.std()**2) / 2.0) + 1e-10))

            rows.append({
                'feature': col,
                'lag': lag,
                'mean_before_long': feat_before_long.mean(),
                'mean_before_short': feat_before_short.mean(),
                'cohens_d': cohens_d,
                'abs_cohens_d': abs(cohens_d),
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def save_report(output_dir, stats_df, temporal_df, split_results, fee_bps):
    """Save profiling report."""
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, "oracle_bar_profile.txt")
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("  EXP-E2a: ORACLE BAR PROFILING\n")
        f.write(f"  Fee threshold: {fee_bps} bps one-way ({2*fee_bps} bps RT)\n")
        f.write("=" * 80 + "\n")

        for split_name, result in split_results.items():
            f.write(f"\n{'=' * 80}\n")
            f.write(f"  SPLIT: {split_name.upper()}\n")
            f.write(f"{'=' * 80}\n")

            sdf = result['stats_df']
            n_long = result['n_long']
            n_short = result['n_short']
            n_hold = result['n_hold']
            n_total = n_long + n_short + n_hold

            f.write("\n  Oracle distribution:\n")
            f.write(f"    Long:  {n_long:>6d} ({100*n_long/n_total:.1f}%)\n")
            f.write(f"    Short: {n_short:>6d} ({100*n_short/n_total:.1f}%)\n")
            f.write(f"    Hold:  {n_hold:>6d} ({100*n_hold/n_total:.1f}%)\n")

            # Top features by |Cohen's d| (long vs short discriminability)
            f.write("\n  TOP 20 FEATURES BY |Cohen's d| (long vs short separability):\n")
            f.write(f"  {'#':>3} {'Feature':<25} {'|d|':>6} {'d':>7} {'Direction':<12} "
                    f"{'Mean(L)':>8} {'Mean(S)':>8} {'Mean(H)':>8} {'KS-stat':>8} {'KS-p':>10}\n")
            f.write(f"  {'-' * 110}\n")

            sdf_sorted = sdf.sort_values('abs_cohens_d', ascending=False)
            for rank, (_, row) in enumerate(sdf_sorted.head(20).iterrows(), 1):
                sig = "***" if row['ks_pval'] < 0.001 else "**" if row['ks_pval'] < 0.01 else "*" if row['ks_pval'] < 0.05 else ""
                f.write(f"  {rank:>3} {row['feature']:<25} {row['abs_cohens_d']:>6.4f} "
                        f"{row['cohens_d']:>+7.4f} {row['signal_direction']:<12} "
                        f"{row['mean_long']:>8.4f} {row['mean_short']:>8.4f} "
                        f"{row['mean_hold']:>8.4f} {row['ks_stat']:>8.4f} "
                        f"{row['ks_pval']:>10.2e} {sig}\n")

            # Bottom features (least discriminating)
            f.write("\n  BOTTOM 10 FEATURES (least discriminating):\n")
            for rank, (_, row) in enumerate(sdf_sorted.tail(10).iterrows(), 1):
                f.write(f"  {rank:>3} {row['feature']:<25} |d|={row['abs_cohens_d']:.4f}\n")

        # Temporal analysis (across splits)
        if temporal_df is not None and len(temporal_df) > 0:
            f.write(f"\n{'=' * 80}\n")
            f.write("  TEMPORAL LEAD ANALYSIS (train split)\n")
            f.write("  Feature values at t-lag before long/short bars\n")
            f.write(f"{'=' * 80}\n\n")

            # Show top features by |Cohen's d| at each lag
            for lag in sorted(temporal_df['lag'].unique()):
                lag_df = temporal_df[temporal_df['lag'] == lag]
                lag_df_sorted = lag_df.sort_values('abs_cohens_d', ascending=False)

                f.write(f"  --- Lag t-{lag} ---\n")
                f.write(f"  {'Feature':<25} {'|d|':>6} {'d':>7} "
                        f"{'Mean(bef.L)':>11} {'Mean(bef.S)':>11}\n")
                f.write(f"  {'-' * 65}\n")

                for _, row in lag_df_sorted.head(10).iterrows():
                    f.write(f"  {row['feature']:<25} {row['abs_cohens_d']:>6.4f} "
                            f"{row['cohens_d']:>+7.4f} "
                            f"{row['mean_before_long']:>11.4f} "
                            f"{row['mean_before_short']:>11.4f}\n")
                f.write("\n")

            # Decay analysis: which features maintain signal across lags?
            f.write("  SIGNAL PERSISTENCE (|d| decay across lags):\n")
            line = f"  {'Feature':<25}"
            for lag in sorted(temporal_df['lag'].unique()):
                line += f" {'t-'+str(lag):>6}"
            line += f" {'Decay':>8}\n"
            f.write(line)
            f.write(f"  {'-' * (25 + 7 * len(temporal_df['lag'].unique()) + 8)}\n")

            # Get top 15 features by t-1 |d|
            t1_df = temporal_df[temporal_df['lag'] == 1].sort_values('abs_cohens_d', ascending=False)
            for _, t1_row in t1_df.head(15).iterrows():
                feat = t1_row['feature']
                f.write(f"  {feat:<25}")
                d_values = []
                for lag in sorted(temporal_df['lag'].unique()):
                    lag_row = temporal_df[(temporal_df['feature'] == feat) & (temporal_df['lag'] == lag)]
                    if len(lag_row) > 0:
                        d_val = lag_row.iloc[0]['abs_cohens_d']
                        d_values.append(d_val)
                        f.write(f" {d_val:>6.4f}")
                    else:
                        f.write(f" {'--':>6}")
                if len(d_values) >= 2:
                    decay = (d_values[-1] / d_values[0]) if d_values[0] > 0 else 0
                    f.write(f" {decay:>7.1%}")
                f.write("\n")

    print(f"  Report saved to {report_path}")

    # Save raw stats as CSV
    stats_path = os.path.join(output_dir, "oracle_feature_stats.csv")
    all_stats = []
    for split_name, result in split_results.items():
        sdf = result['stats_df'].copy()
        sdf['split'] = split_name
        all_stats.append(sdf)
    pd.concat(all_stats).to_csv(stats_path, index=False)
    print(f"  Stats CSV saved to {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="EXP-E2a: Oracle Bar Profiling")
    parser.add_argument("--config", type=str, default="configs/postaudit_hpo.yaml")
    parser.add_argument("--output_dir", type=str, default="results/oracle_profile")
    parser.add_argument("--fee_bps", type=float, default=5.0,
                        help="One-way taker fee in bps (default: 5 = 10bps RT)")
    args = parser.parse_args()

    print("=" * 70)
    print("  EXP-E2a: Oracle Bar Profiling")
    print(f"  Fee threshold: {args.fee_bps} bps (1-way), {2*args.fee_bps} bps (RT)")
    print("=" * 70)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_config = config.get("data", {})

    # Load all splits
    splits_def = {
        "train": {
            "start_date": data_config.get("train_start_date"),
            "end_date": data_config.get("train_end_date"),
            "norm_cutoff": None,
        },
        "val": {
            "start_date": data_config.get("val_start_date"),
            "end_date": data_config.get("val_end_date"),
            "norm_cutoff": data_config.get("val_start_date"),
        },
        "test": {
            "start_date": data_config.get("test_start_date"),
            "end_date": data_config.get("test_end_date"),
            "norm_cutoff": data_config.get("test_start_date"),
        },
    }

    split_results = {}
    temporal_df = None

    for split_name, sdef in splits_def.items():
        print(f"\n{'=' * 50}")
        print(f"  Processing {split_name} split...")
        print(f"{'=' * 50}")

        X, mid, timestamps, feature_cols = load_split(
            config, sdef["start_date"], sdef["end_date"], sdef["norm_cutoff"]
        )

        # Classify bars
        labels, returns_bps = classify_oracle_bars(mid, fee_bps=args.fee_bps)

        # Compute feature stats per group
        stats_df = compute_group_stats(X, labels, feature_cols, returns_bps)

        split_results[split_name] = {
            'stats_df': stats_df,
            'n_long': (labels == 'long').sum(),
            'n_short': (labels == 'short').sum(),
            'n_hold': (labels == 'hold').sum(),
        }

        # Temporal analysis only on train (largest split)
        if split_name == "train":
            print("\n  Computing temporal lead analysis (t-1 to t-5)...")
            temporal_df = compute_temporal_stats(X, labels, feature_cols, lookback=5)
            if len(temporal_df) > 0:
                top_t1 = temporal_df[temporal_df['lag'] == 1].sort_values('abs_cohens_d', ascending=False)
                print("  Top 5 features by |d| at t-1:")
                for _, row in top_t1.head(5).iterrows():
                    print(f"    {row['feature']:<25} |d|={row['abs_cohens_d']:.4f}")

    # Save report
    print(f"\n  Saving report to {args.output_dir}/...")
    save_report(args.output_dir, stats_df, temporal_df, split_results, args.fee_bps)

    # Summary
    print(f"\n{'=' * 70}")
    print("  E2a SUMMARY")
    print(f"{'=' * 70}")

    for split_name, result in split_results.items():
        sdf = result['stats_df']
        top5 = sdf.sort_values('abs_cohens_d', ascending=False).head(5)
        print(f"\n  {split_name.upper()} — Top 5 discriminating features:")
        for _, row in top5.iterrows():
            print(f"    {row['feature']:<25} |d|={row['abs_cohens_d']:.4f}  "
                  f"({row['signal_direction']})")

    # Cross-split stability check
    train_top = set(split_results['train']['stats_df']
                    .sort_values('abs_cohens_d', ascending=False)
                    .head(10)['feature'])
    val_top = set(split_results['val']['stats_df']
                  .sort_values('abs_cohens_d', ascending=False)
                  .head(10)['feature'])
    overlap = train_top & val_top
    print(f"\n  Cross-split stability: {len(overlap)}/10 top features overlap "
          f"between train and val")
    print(f"  Stable features: {sorted(overlap)}")

    print(f"\n  Artifacts: {args.output_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
