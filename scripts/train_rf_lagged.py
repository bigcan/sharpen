#!/usr/bin/env python
"""
EXP-E2b: Lagged Feature RF — Temporal Pattern Detection
========================================================

Tests whether adding t-1 through t-4 lagged features improves RF prediction
compared to baseline E1. This directly tests the RL agent's theoretical
advantage: if lagged features improve AUC, then the TCN/LSTM's temporal
processing should capture similar patterns.

Feature construction:
  - Base: 45 features (30 micro + 15 macro)
  - Lagged: each feature at t-1, t-2, t-3, t-4
  - Total: 45 * 5 = 225 features
  - Also tests: delta features (t - t-1) as alternative temporal encoding

Usage:
  python scripts/train_rf_lagged.py --config configs/postaudit_hpo.yaml
  python scripts/train_rf_lagged.py --config configs/postaudit_hpo.yaml --max_lag 4 --horizons 1 30
"""
import argparse
import os
import sys
import time
import yaml
import numpy as np

sys.path.append(os.getcwd())

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

from finrl_pro_ds.data.parquet_handler import ParquetDataHandler
from finrl_pro_ds.data.feature_engineering import MICRO_FEATURE_COLS, MACRO_FEATURE_COLS


def load_split(config, start_date, end_date, norm_cutoff_date=None):
    """Load a single split via ParquetDataHandler."""
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
    missing = [c for c in feature_cols if c not in handler._data_arrays]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = np.column_stack([handler._data_arrays[c] for c in feature_cols])
    mid = handler._data_arrays['mid_price'].copy()
    timestamps = handler._data_arrays['timestamp'].copy()

    print(f"  Loaded {len(mid)} rows ({start_date} -> {end_date}), X shape: {X.shape}")
    handler.close()
    return X, mid, timestamps, feature_cols


def add_lagged_features(X, feature_cols, max_lag=4, include_deltas=True):
    """Add lagged and delta features to feature matrix.

    Args:
        X: (N, F) feature matrix
        feature_cols: list of F feature names
        max_lag: number of lags to add (1..max_lag)
        include_deltas: also add delta (t - t-1) features

    Returns:
        X_aug: augmented feature matrix
        aug_cols: augmented column names
        valid_mask: bool mask (first max_lag rows are invalid)
    """
    N, F = X.shape
    aug_arrays = [X]  # Start with original features
    aug_cols = list(feature_cols)  # t=0 names

    # Add lagged features
    for lag in range(1, max_lag + 1):
        X_lag = np.zeros_like(X)
        X_lag[lag:] = X[:-lag]
        # Fill first `lag` rows with the first valid value (avoid NaN artifacts)
        X_lag[:lag] = X[lag]
        aug_arrays.append(X_lag)
        aug_cols.extend([f"{col}_t{lag}" for col in feature_cols])

    # Add delta features (t - t-1)
    if include_deltas:
        X_delta = np.zeros_like(X)
        X_delta[1:] = X[1:] - X[:-1]
        X_delta[0] = 0.0
        aug_arrays.append(X_delta)
        aug_cols.extend([f"{col}_delta" for col in feature_cols])

    X_aug = np.column_stack(aug_arrays)

    # First max_lag rows are invalid (lagged features are backfilled, not real)
    valid_mask = np.ones(N, dtype=bool)
    valid_mask[:max_lag] = False

    return X_aug, aug_cols, valid_mask


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


def train_and_evaluate(X_train, y_train, valid_train,
                       X_val, y_val, valid_val,
                       X_test, y_test, valid_test,
                       label="baseline"):
    """Train RF and evaluate on all splits."""
    # Combine valid masks
    mask_train = valid_train
    mask_val = valid_val
    mask_test = valid_test

    X_fit = X_train[mask_train]
    y_fit = y_train[mask_train]

    n_pos = y_fit.sum()
    print(f"\n  [{label}] Training on {len(y_fit)} samples "
          f"(pos={n_pos}, features={X_fit.shape[1]})")

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=100,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
    )

    t0 = time.time()
    rf.fit(X_fit, y_fit)
    elapsed = time.time() - t0
    print(f"  [{label}] Trained in {elapsed:.1f}s")

    results = {}
    for split_name, X_s, y_s, mask_s in [
        ("train", X_train, y_train, mask_train),
        ("val", X_val, y_val, mask_val),
        ("test", X_test, y_test, mask_test),
    ]:
        probs = rf.predict_proba(X_s[mask_s])[:, 1]
        auc = roc_auc_score(y_s[mask_s], probs)
        acc = accuracy_score(y_s[mask_s], (probs >= 0.5).astype(int))
        results[split_name] = {"auc": auc, "accuracy": acc, "n": mask_s.sum()}
        print(f"  [{label}] {split_name}: AUC={auc:.4f}, Acc={acc:.4f}, N={mask_s.sum()}")

    return rf, results


def main():
    parser = argparse.ArgumentParser(description="EXP-E2b: Lagged Feature RF")
    parser.add_argument("--config", type=str, default="configs/postaudit_hpo.yaml")
    parser.add_argument("--output_dir", type=str, default="results/rf_lagged")
    parser.add_argument("--max_lag", type=int, default=4,
                        help="Number of lags to add (default: 4)")
    parser.add_argument("--horizons", nargs="*", type=int, default=[1, 30],
                        help="Prediction horizons (default: 1 30)")
    parser.add_argument("--no_deltas", action="store_true",
                        help="Skip delta (t - t-1) features")
    args = parser.parse_args()

    print("=" * 70)
    print("  EXP-E2b: Lagged Feature RF — Temporal Pattern Detection")
    print(f"  Max lag: {args.max_lag}, Horizons: {args.horizons}")
    print(f"  Deltas: {'yes' if not args.no_deltas else 'no'}")
    print("=" * 70)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_config = config.get("data", {})

    # Load splits
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

    print("\n[1/4] Loading data...")
    data = {}
    for split_name, sdef in splits_def.items():
        X, mid, ts, feat_cols = load_split(
            config, sdef["start_date"], sdef["end_date"], sdef["norm_cutoff"]
        )
        data[split_name] = {"X": X, "mid": mid, "timestamps": ts}

    feature_cols = feat_cols  # Same across splits

    # Construct augmented features
    print(f"\n[2/4] Constructing lagged features (max_lag={args.max_lag})...")

    aug_data = {}
    for split_name in ["train", "val", "test"]:
        X_aug, aug_cols, lag_valid = add_lagged_features(
            data[split_name]["X"], feature_cols,
            max_lag=args.max_lag,
            include_deltas=not args.no_deltas,
        )
        aug_data[split_name] = {"X_aug": X_aug, "lag_valid": lag_valid}

    n_base = len(feature_cols)
    n_aug = len(aug_cols)
    n_lagged = args.max_lag * n_base
    n_delta = n_base if not args.no_deltas else 0
    print(f"  Base features: {n_base}")
    print(f"  Lagged features: {n_lagged} ({args.max_lag} lags x {n_base})")
    print(f"  Delta features: {n_delta}")
    print(f"  Total augmented: {n_aug}")

    # Run experiments for each horizon
    print("\n[3/4] Training models...")

    all_results = {}

    for horizon in args.horizons:
        h_label = f"H{horizon}"
        print(f"\n{'=' * 50}")
        print(f"  Horizon: {h_label}")
        print(f"{'=' * 50}")

        # Construct targets
        targets = {}
        for split_name in ["train", "val", "test"]:
            y, target_valid = construct_target(data[split_name]["mid"], horizon)
            targets[split_name] = {"y": y, "valid": target_valid}

        # ── Experiment A: Baseline (no lags) ──
        valid_train = targets["train"]["valid"]
        valid_val = targets["val"]["valid"]
        valid_test = targets["test"]["valid"]

        print("\n  --- A: Baseline (45 features) ---")
        _, baseline_results = train_and_evaluate(
            data["train"]["X"], targets["train"]["y"], valid_train,
            data["val"]["X"], targets["val"]["y"], valid_val,
            data["test"]["X"], targets["test"]["y"], valid_test,
            label=f"{h_label}_baseline",
        )

        # ── Experiment B: Lagged features only (no deltas) ──
        print(f"\n  --- B: Lagged only ({n_base + n_lagged} features) ---")
        # Create lag-only version
        lag_only_data = {}
        for sn in ["train", "val", "test"]:
            X_lag_only, _, _ = add_lagged_features(
                data[sn]["X"], feature_cols,
                max_lag=args.max_lag, include_deltas=False,
            )
            lag_only_data[sn] = X_lag_only

        # Combine lag_valid with target_valid
        lag_valid_train = aug_data["train"]["lag_valid"] & valid_train
        lag_valid_val = aug_data["val"]["lag_valid"] & valid_val
        lag_valid_test = aug_data["test"]["lag_valid"] & valid_test

        _, lagged_results = train_and_evaluate(
            lag_only_data["train"], targets["train"]["y"], lag_valid_train,
            lag_only_data["val"], targets["val"]["y"], lag_valid_val,
            lag_only_data["test"], targets["test"]["y"], lag_valid_test,
            label=f"{h_label}_lagged",
        )

        # ── Experiment C: Full augmented (lags + deltas) ──
        if not args.no_deltas:
            print(f"\n  --- C: Lagged + Deltas ({n_aug} features) ---")
            _, full_results = train_and_evaluate(
                aug_data["train"]["X_aug"], targets["train"]["y"], lag_valid_train,
                aug_data["val"]["X_aug"], targets["val"]["y"], lag_valid_val,
                aug_data["test"]["X_aug"], targets["test"]["y"], lag_valid_test,
                label=f"{h_label}_full",
            )
        else:
            full_results = lagged_results  # Same if no deltas

        # ── Experiment D: Deltas only (no lags) ──
        if not args.no_deltas:
            print(f"\n  --- D: Deltas only ({n_base + n_delta} features) ---")
            delta_data = {}
            for sn in ["train", "val", "test"]:
                X_base = data[sn]["X"]
                X_delta = np.zeros_like(X_base)
                X_delta[1:] = X_base[1:] - X_base[:-1]
                X_delta[0] = 0.0
                delta_data[sn] = np.column_stack([X_base, X_delta])

            delta_valid_train = valid_train.copy()
            delta_valid_train[0] = False
            delta_valid_val = valid_val.copy()
            delta_valid_val[0] = False
            delta_valid_test = valid_test.copy()
            delta_valid_test[0] = False

            _, delta_results = train_and_evaluate(
                delta_data["train"], targets["train"]["y"], delta_valid_train,
                delta_data["val"], targets["val"]["y"], delta_valid_val,
                delta_data["test"], targets["test"]["y"], delta_valid_test,
                label=f"{h_label}_deltas",
            )
        else:
            delta_results = None

        # ── Experiment E: Feature importance on augmented model ──
        # Re-train full model and extract importances
        print("\n  --- Feature importance analysis ---")
        rf_full = RandomForestClassifier(
            n_estimators=500, max_depth=12, min_samples_leaf=100,
            max_features='sqrt', class_weight='balanced',
            n_jobs=-1, random_state=42,
        )
        rf_full.fit(
            aug_data["train"]["X_aug"][lag_valid_train],
            targets["train"]["y"][lag_valid_train],
        )
        importances = rf_full.feature_importances_

        # Aggregate importance by type
        base_imp = importances[:n_base].sum()
        lag_imp = importances[n_base:n_base + n_lagged].sum()
        delta_imp = importances[n_base + n_lagged:].sum() if not args.no_deltas else 0

        print(f"  Importance share: base={base_imp:.3f}, lagged={lag_imp:.3f}, "
              f"delta={delta_imp:.3f}")

        # Top 10 augmented features
        imp_order = np.argsort(importances)[::-1]
        print("  Top 10 augmented features:")
        for rank, idx in enumerate(imp_order[:10], 1):
            print(f"    {rank:>2}. {aug_cols[idx]:<30} {importances[idx]:.4f}")

        all_results[h_label] = {
            'baseline': baseline_results,
            'lagged': lagged_results,
            'full': full_results,
            'delta': delta_results,
            'importances': importances,
            'aug_cols': aug_cols,
            'base_imp': base_imp,
            'lag_imp': lag_imp,
            'delta_imp': delta_imp,
        }

    # Save report
    print(f"\n[4/4] Saving report to {args.output_dir}/...")
    os.makedirs(args.output_dir, exist_ok=True)

    report_path = os.path.join(args.output_dir, "rf_lagged_evaluation.txt")
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("  EXP-E2b: LAGGED FEATURE RF — TEMPORAL PATTERN DETECTION\n")
        f.write(f"  Max lag: {args.max_lag}, Horizons: {args.horizons}\n")
        f.write(f"  Features: {n_base} base + {n_lagged} lagged + {n_delta} delta = {n_aug}\n")
        f.write("=" * 80 + "\n")

        for horizon in args.horizons:
            h_label = f"H{horizon}"
            r = all_results[h_label]

            f.write(f"\n{'=' * 80}\n")
            f.write(f"  HORIZON: {h_label}\n")
            f.write(f"{'=' * 80}\n")

            f.write(f"\n  {'Experiment':<25} {'Train AUC':>10} {'Val AUC':>10} "
                    f"{'Test AUC':>10} {'Val Delta':>10}\n")
            f.write(f"  {'-' * 70}\n")

            baseline_val_auc = r['baseline']['val']['auc']

            for exp_name, exp_key in [
                (f"A: Baseline ({n_base}f)", 'baseline'),
                (f"B: Lagged ({n_base+n_lagged}f)", 'lagged'),
                (f"C: Full ({n_aug}f)", 'full'),
                (f"D: Deltas ({n_base+n_delta}f)", 'delta'),
            ]:
                if r.get(exp_key) is None:
                    continue
                exp_r = r[exp_key]
                delta = exp_r['val']['auc'] - baseline_val_auc
                delta_str = f"{delta:>+10.4f}" if exp_key != 'baseline' else f"{'(base)':>10}"
                f.write(f"  {exp_name:<25} {exp_r['train']['auc']:>10.4f} "
                        f"{exp_r['val']['auc']:>10.4f} {exp_r['test']['auc']:>10.4f} "
                        f"{delta_str}\n")

            f.write("\n  Feature importance share:\n")
            f.write(f"    Base (t=0):  {r['base_imp']:.3f} ({100*r['base_imp']:.1f}%)\n")
            f.write(f"    Lagged:      {r['lag_imp']:.3f} ({100*r['lag_imp']:.1f}%)\n")
            f.write(f"    Delta:       {r['delta_imp']:.3f} ({100*r['delta_imp']:.1f}%)\n")

            f.write("\n  Top 15 augmented features by importance:\n")
            imp = r['importances']
            cols = r['aug_cols']
            imp_order = np.argsort(imp)[::-1]
            for rank, idx in enumerate(imp_order[:15], 1):
                tag = ""
                if "_t" in cols[idx]:
                    tag = " [lagged]"
                elif "_delta" in cols[idx]:
                    tag = " [delta]"
                f.write(f"    {rank:>2}. {cols[idx]:<30} {imp[idx]:.4f}{tag}\n")

    print(f"  Report saved to {report_path}")

    # Summary
    print(f"\n{'=' * 70}")
    print("  E2b SUMMARY — TEMPORAL PATTERN VALUE")
    print(f"{'=' * 70}")

    for horizon in args.horizons:
        h_label = f"H{horizon}"
        r = all_results[h_label]
        base_auc = r['baseline']['val']['auc']
        lag_auc = r['lagged']['val']['auc']
        full_auc = r['full']['val']['auc']
        delta_auc = r['delta']['val']['auc'] if r['delta'] else 0

        delta_lag = lag_auc - base_auc
        delta_full = full_auc - base_auc

        print(f"\n  {h_label}:")
        print(f"    Baseline val AUC:  {base_auc:.4f}")
        print(f"    + Lags val AUC:    {lag_auc:.4f}  ({delta_lag:+.4f})")
        print(f"    + Full val AUC:    {full_auc:.4f}  ({delta_full:+.4f})")
        if r['delta']:
            print(f"    + Deltas val AUC:  {delta_auc:.4f}  ({delta_auc - base_auc:+.4f})")
        print(f"    Importance: base={100*r['base_imp']:.0f}% lag={100*r['lag_imp']:.0f}% "
              f"delta={100*r['delta_imp']:.0f}%")

        if delta_full > 0.01:
            print(f"    --> Temporal patterns ADD signal (+{delta_full:.4f} AUC)")
            print("        RL agent's TCN/LSTM SHOULD capture this advantage")
        elif delta_full > 0.003:
            print(f"    --> Weak temporal signal (+{delta_full:.4f} AUC)")
            print("        RL agent has marginal advantage from sequences")
        else:
            print(f"    --> No temporal signal ({delta_full:+.4f} AUC)")
            print("        RL agent's TCN/LSTM adds NO value over snapshot features")

    print(f"\n  Artifacts: {args.output_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
