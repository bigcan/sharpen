#!/usr/bin/env python
"""
EXP-F1: Feature Engineering v3 Validation
==========================================

Unified validation combining oracle profiling + RF comparison for fev3 features.

1. Oracle bar profiling: Cohen's d at t=0 and t-1 for all 40 features
2. RF training: 40-feature RF on train split, compare H1 val AUC to v2 baseline (0.5942)
3. Feature importance ranking

Decision gate: H1 val AUC > 0.61 → proceed to RL experiments with fev3 features.

Usage:
  python scripts/validate_fev3.py --config configs/phase_f1_fev3_validation.yaml
  python scripts/validate_fev3.py --config configs/phase_f1_fev3_validation.yaml --v2_config configs/phase_b5_bdq_5min_hpo.yaml
"""
import argparse
import os
import sys
import time
import yaml
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

from finrl_pro_ds.data.parquet_handler import ParquetDataHandler
from finrl_pro_ds.data.feature_engineering import MICRO_FEATURE_COLS, MACRO_FEATURE_COLS


def load_split(config, start_date, end_date, norm_cutoff_date=None):
    """Load a single split via ParquetDataHandler, return features + mid_price."""
    data_config = config.get("data", {})
    handler = ParquetDataHandler(
        file_path=data_config.get("file_path"),
        ticker=data_config.get("ticker", "BTCUSDT"),
        feature_config=config.get("features", {}),
        start_date=start_date,
        end_date=end_date,
        norm_cutoff_date=norm_cutoff_date,
    )

    feature_cols = list(MICRO_FEATURE_COLS) + list(MACRO_FEATURE_COLS)
    available = [c for c in feature_cols if c in handler._data_arrays]
    missing = [c for c in feature_cols if c not in handler._data_arrays]
    if missing:
        print(f"  [WARN] Missing columns (defaulting to 0): {missing}")

    X = np.column_stack([
        handler._data_arrays[c] if c in handler._data_arrays
        else np.zeros(handler._len, dtype=np.float32)
        for c in feature_cols
    ])
    mid = handler._data_arrays['mid_price'].copy()
    timestamps = handler._data_arrays['timestamp'].copy()

    print(f"  Loaded {len(mid)} rows ({start_date} -> {end_date}), "
          f"X shape: {X.shape}, features: {len(available)}/{len(feature_cols)}")
    handler.close()
    return X, mid, timestamps, feature_cols


def classify_oracle_bars(mid, fee_bps=5.0):
    """Classify bars as oracle-long / oracle-short / oracle-hold."""
    n = len(mid)
    returns_bps = np.zeros(n, dtype=np.float64)
    mid_safe = np.where(mid > 0, mid, 1e-9)
    returns_bps[:-1] = ((mid[1:] - mid[:-1]) / mid_safe[:-1]) * 10000.0

    labels = np.full(n, 'hold', dtype='U5')
    labels[returns_bps > fee_bps] = 'long'
    labels[returns_bps < -fee_bps] = 'short'
    return labels, returns_bps


def compute_oracle_stats(X, labels, feature_cols, lag=0):
    """Compute Cohen's d for each feature at given lag."""
    long_mask = labels == 'long'
    short_mask = labels == 'short'

    if lag > 0:
        long_mask = long_mask.copy()
        short_mask = short_mask.copy()
        long_mask[:lag] = False
        short_mask[:lag] = False

    rows = []
    for i, col in enumerate(feature_cols):
        if lag > 0:
            long_idx = np.where(long_mask)[0] - lag
            short_idx = np.where(short_mask)[0] - lag
            feat_long = X[long_idx, i]
            feat_short = X[short_idx, i]
        else:
            feat_long = X[long_mask, i]
            feat_short = X[short_mask, i]

        if len(feat_long) < 50 or len(feat_short) < 50:
            continue

        mean_l = feat_long.mean()
        mean_s = feat_short.mean()
        std_l = feat_long.std()
        std_s = feat_short.std()
        pooled = np.sqrt((std_l**2 + std_s**2) / 2.0)
        d = (mean_l - mean_s) / (pooled + 1e-10)

        rows.append({
            'feature': col,
            'lag': lag,
            'mean_long': mean_l,
            'mean_short': mean_s,
            'cohens_d': d,
            'abs_d': abs(d),
        })

    return pd.DataFrame(rows)


def construct_target(mid, horizon):
    """Binary target: 1 if price goes up in `horizon` steps."""
    n = len(mid)
    y = np.zeros(n, dtype=np.int32)
    valid = np.ones(n, dtype=bool)
    if horizon >= n:
        valid[:] = False
        return y, valid
    y[:-horizon] = (mid[horizon:] > mid[:-horizon]).astype(np.int32)
    valid[-horizon:] = False
    return y, valid


def train_rf(X_train, y_train, valid_train, X_val, y_val, valid_val,
             X_test, y_test, valid_test, label=""):
    """Train RF and evaluate on all splits."""
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=100,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
    )

    X_fit = X_train[valid_train]
    y_fit = y_train[valid_train]
    print(f"  [{label}] Training on {len(y_fit)} samples, "
          f"features={X_fit.shape[1]}")

    t0 = time.time()
    rf.fit(X_fit, y_fit)
    elapsed = time.time() - t0
    print(f"  [{label}] Trained in {elapsed:.1f}s")

    results = {}
    for split_name, X_s, y_s, mask_s in [
        ("train", X_train, y_train, valid_train),
        ("val", X_val, y_val, valid_val),
        ("test", X_test, y_test, valid_test),
    ]:
        probs = rf.predict_proba(X_s[mask_s])[:, 1]
        auc = roc_auc_score(y_s[mask_s], probs)
        acc = accuracy_score(y_s[mask_s], (probs >= 0.5).astype(int))
        results[split_name] = {"auc": auc, "accuracy": acc, "n": mask_s.sum()}
        print(f"  [{label}] {split_name}: AUC={auc:.4f}, Acc={acc:.4f}")

    return rf, results


def main():
    parser = argparse.ArgumentParser(
        description="EXP-F1: Feature Engineering v3 Validation"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="fev3 config (points to *_fev3.parquet)")
    parser.add_argument("--v2_config", type=str, default=None,
                        help="Optional v2 config for side-by-side comparison")
    parser.add_argument("--output_dir", type=str,
                        default="results/fev3_validation")
    parser.add_argument("--fee_bps", type=float, default=5.0)
    parser.add_argument("--horizons", nargs="*", type=int, default=[1],
                        help="Prediction horizons (default: [1])")
    parser.add_argument("--v2_baseline_auc", type=float, default=0.5942,
                        help="v2 H1 val AUC baseline for comparison")
    parser.add_argument("--gate_threshold", type=float, default=0.61,
                        help="AUC threshold for decision gate")
    args = parser.parse_args()

    print("=" * 70)
    print("  EXP-F1: Feature Engineering v3 Validation")
    print(f"  v2 baseline H1 val AUC: {args.v2_baseline_auc:.4f}")
    print(f"  Decision gate: val AUC > {args.gate_threshold:.2f}")
    print("=" * 70)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_config = config.get("data", {})
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

    # ── 1. Load fev3 data ──
    print("\n[1/4] Loading fev3 data...")
    data = {}
    for split_name, sdef in splits_def.items():
        print(f"\n  {split_name}:")
        X, mid, ts, feat_cols = load_split(
            config, sdef["start_date"], sdef["end_date"], sdef["norm_cutoff"]
        )
        data[split_name] = {"X": X, "mid": mid, "timestamps": ts}

    n_features = len(feat_cols)
    n_micro = len(MICRO_FEATURE_COLS)
    n_macro = len(MACRO_FEATURE_COLS)
    print(f"\n  Total features: {n_features} "
          f"(micro={n_micro}, macro={n_macro})")

    # ── 2. Oracle profiling ──
    print(f"\n[2/4] Oracle bar profiling (fee={args.fee_bps} bps)...")
    train_X = data["train"]["X"]
    train_mid = data["train"]["mid"]
    labels, returns_bps = classify_oracle_bars(train_mid, args.fee_bps)

    n_long = (labels == 'long').sum()
    n_short = (labels == 'short').sum()
    n_hold = (labels == 'hold').sum()
    n_total = len(labels)
    print(f"  Oracle distribution: long={n_long} ({100*n_long/n_total:.1f}%), "
          f"short={n_short} ({100*n_short/n_total:.1f}%), "
          f"hold={n_hold} ({100*n_hold/n_total:.1f}%)")

    # Cohen's d at t=0 and t-1
    stats_t0 = compute_oracle_stats(train_X, labels, feat_cols, lag=0)
    stats_t1 = compute_oracle_stats(train_X, labels, feat_cols, lag=1)

    print("\n  Top 10 features by |Cohen's d| at t=0:")
    for _, row in stats_t0.sort_values('abs_d', ascending=False).head(10).iterrows():
        print(f"    {row['feature']:<25} |d|={row['abs_d']:.4f}")

    print("\n  Top 10 features by |Cohen's d| at t-1:")
    for _, row in stats_t1.sort_values('abs_d', ascending=False).head(10).iterrows():
        print(f"    {row['feature']:<25} |d|={row['abs_d']:.4f}")

    # Highlight new fev3 features
    fev3_features = [
        'obi_burst', 'obi_trend', 'dofi_burst', 'microprice_range',
        'spread_max', 'obi_accel', 'dofi_accel', 'microprice_accel',
        'spread_velocity', 'depth_drain',
    ]
    print("\n  fev3 feature discriminability:")
    for feat in fev3_features:
        t0_row = stats_t0[stats_t0['feature'] == feat]
        t1_row = stats_t1[stats_t1['feature'] == feat]
        d0 = t0_row.iloc[0]['abs_d'] if len(t0_row) > 0 else 0.0
        d1 = t1_row.iloc[0]['abs_d'] if len(t1_row) > 0 else 0.0
        decay = d1 / d0 if d0 > 0 else 0.0
        tag = "NEW" if feat in fev3_features else ""
        print(f"    {feat:<25} t=0 |d|={d0:.4f}  t-1 |d|={d1:.4f}  "
              f"decay={decay:.1%}  {tag}")

    # ── 3. RF evaluation ──
    print("\n[3/4] RF evaluation...")

    all_results = {}
    for horizon in args.horizons:
        h_label = f"H{horizon}"
        print(f"\n  --- Horizon: {h_label} ---")

        targets = {}
        for split_name in ["train", "val", "test"]:
            y, valid = construct_target(data[split_name]["mid"], horizon)
            targets[split_name] = {"y": y, "valid": valid}

        # Train fev3 RF (all 40 micro + 15 macro = 55 features)
        rf, results = train_rf(
            data["train"]["X"], targets["train"]["y"], targets["train"]["valid"],
            data["val"]["X"], targets["val"]["y"], targets["val"]["valid"],
            data["test"]["X"], targets["test"]["y"], targets["test"]["valid"],
            label=f"{h_label}_fev3",
        )

        # Feature importance
        importances = rf.feature_importances_
        imp_order = np.argsort(importances)[::-1]
        print("\n  Top 15 features by importance:")
        for rank, idx in enumerate(imp_order[:15], 1):
            tag = " [fev3]" if feat_cols[idx] in fev3_features else ""
            print(f"    {rank:>2}. {feat_cols[idx]:<25} {importances[idx]:.4f}{tag}")

        # Importance share: v2 vs fev3
        v2_imp = sum(importances[i] for i, c in enumerate(feat_cols)
                     if c not in fev3_features)
        fev3_imp = sum(importances[i] for i, c in enumerate(feat_cols)
                       if c in fev3_features)
        print(f"\n  Importance share: v2={v2_imp:.3f} ({100*v2_imp:.1f}%), "
              f"fev3={fev3_imp:.3f} ({100*fev3_imp:.1f}%)")

        all_results[h_label] = {
            'results': results,
            'importances': importances,
        }

    # ── 4. Decision gate ──
    print("\n[4/4] Decision gate...")
    os.makedirs(args.output_dir, exist_ok=True)

    h1_val_auc = all_results.get('H1', {}).get('results', {}).get('val', {}).get('auc', 0.0)
    delta = h1_val_auc - args.v2_baseline_auc
    gate_pass = h1_val_auc > args.gate_threshold

    # Save report
    report_path = os.path.join(args.output_dir, "fev3_validation.txt")
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("  EXP-F1: FEATURE ENGINEERING v3 VALIDATION\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"  v2 baseline H1 val AUC: {args.v2_baseline_auc:.4f}\n")
        f.write(f"  fev3 H1 val AUC:        {h1_val_auc:.4f} ({delta:+.4f})\n")
        f.write(f"  Decision gate ({args.gate_threshold:.2f}): "
                f"{'PASS' if gate_pass else 'FAIL'}\n\n")

        # Oracle profiling
        f.write("  ORACLE BAR PROFILING (train split)\n")
        f.write("  " + "-" * 76 + "\n")
        f.write(f"  {'Feature':<25} {'t=0 |d|':>8} {'t-1 |d|':>8} {'Decay':>8} {'Type':>6}\n")
        f.write("  " + "-" * 76 + "\n")

        combined = stats_t0.merge(stats_t1, on='feature', suffixes=('_t0', '_t1'))
        combined = combined.sort_values('abs_d_t0', ascending=False)
        for _, row in combined.iterrows():
            decay = row['abs_d_t1'] / row['abs_d_t0'] if row['abs_d_t0'] > 0 else 0
            tag = "fev3" if row['feature'] in fev3_features else "v2"
            f.write(f"  {row['feature']:<25} {row['abs_d_t0']:>8.4f} "
                    f"{row['abs_d_t1']:>8.4f} {decay:>7.1%} {tag:>6}\n")

        # RF results
        for h_label, r in all_results.items():
            f.write(f"\n  RF RESULTS — {h_label}\n")
            f.write("  " + "-" * 60 + "\n")
            for split in ["train", "val", "test"]:
                sr = r['results'][split]
                f.write(f"  {split:>6}: AUC={sr['auc']:.4f}, "
                        f"Acc={sr['accuracy']:.4f}, N={sr['n']}\n")

    print(f"  Report saved: {report_path}")

    # Save oracle stats CSV
    stats_path = os.path.join(args.output_dir, "oracle_stats_fev3.csv")
    combined.to_csv(stats_path, index=False)
    print(f"  Oracle stats: {stats_path}")

    # Summary
    print(f"\n{'=' * 70}")
    print("  EXP-F1 VALIDATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  v2 baseline H1 val AUC: {args.v2_baseline_auc:.4f}")
    print(f"  fev3 H1 val AUC:        {h1_val_auc:.4f} ({delta:+.4f})")
    print(f"  Decision gate ({args.gate_threshold:.2f}): "
          f"{'PASS — proceed to RL experiments' if gate_pass else 'FAIL — insufficient signal improvement'}")
    print(f"\n  Artifacts: {args.output_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
