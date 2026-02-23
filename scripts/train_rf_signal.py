#!/usr/bin/env python
"""
Phase E-0: Train Random Forest Signal Generator + E-1 Threshold Backtests
=========================================================================

Trains an RF classifier on normalized LOB features to predict directional
price movement. Two horizons: H30 (primary) and H1 (comparison).

LEAK-1 Compliant:
  - RF trained ONLY on train split (Jan-Apr 2025)
  - Val/test get predictions from the train-fitted model
  - Each split loaded via separate ParquetDataHandler with proper norm_cutoff

Output artifacts:
  results/rf_signal/
    rf_model_h30.joblib          -- trained RF model
    rf_probs.npz                 -- {timestamps, probs_h1, probs_h30, split_labels}
    rf_feature_importance.csv    -- top features by importance
    rf_evaluation.txt            -- AUC, accuracy, calibration, E-1 results

Usage:
  python scripts/train_rf_signal.py --config configs/postaudit_hpo.yaml
  python scripts/train_rf_signal.py --config configs/postaudit_hpo.yaml --no_e1
"""
import argparse
import os
import sys
import time
import yaml
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "scripts"))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve

from finrl_pro_ds.data.parquet_handler import ParquetDataHandler
from finrl_pro_ds.data.feature_engineering import MICRO_FEATURE_COLS, MACRO_FEATURE_COLS


# ============================================================================
# DATA LOADING — LEAK-1 compliant: separate handler per split
# ============================================================================

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

    # Validate all feature columns exist
    missing = [c for c in feature_cols if c not in handler._data_arrays]
    if missing:
        raise ValueError(f"Missing feature columns in handler: {missing}")

    X = np.column_stack([handler._data_arrays[c] for c in feature_cols])
    mid = handler._data_arrays['mid_price'].copy()
    timestamps = handler._data_arrays['timestamp'].copy()

    n_rows = len(mid)
    print(f"  Loaded {n_rows} rows ({start_date} -> {end_date}), "
          f"X shape: {X.shape}, mid range: [{mid.min():.2f}, {mid.max():.2f}]")

    handler.close()
    return X, mid, timestamps, feature_cols


def construct_target(mid, horizon):
    """Construct binary target: 1 if mid_price goes UP in `horizon` steps.

    Returns:
        y: int32 array (0=down/flat, 1=up)
        valid: bool mask (False for last `horizon` rows with no target)
    """
    n = len(mid)
    y = np.zeros(n, dtype=np.int32)
    valid = np.ones(n, dtype=bool)

    if horizon >= n:
        valid[:] = False
        return y, valid

    # Strictly causal: compare current price vs future price
    y[:-horizon] = (mid[horizon:] > mid[:-horizon]).astype(np.int32)
    valid[-horizon:] = False

    return y, valid


# ============================================================================
# RF TRAINING & EVALUATION
# ============================================================================

def train_rf(X_train, y_train, valid_train, horizon_label="H30"):
    """Train Random Forest classifier on train split only."""
    X_fit = X_train[valid_train]
    y_fit = y_train[valid_train]

    n_pos = y_fit.sum()
    n_neg = len(y_fit) - n_pos
    print(f"\n  [{horizon_label}] Training RF on {len(y_fit)} samples "
          f"(pos={n_pos}, neg={n_neg}, ratio={n_pos/len(y_fit):.3f})")

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=100,       # Prevent overfitting on 1-min noise
        max_features='sqrt',
        class_weight='balanced',    # Handle any class imbalance
        n_jobs=-1,
        random_state=42,
    )

    t0 = time.time()
    rf.fit(X_fit, y_fit)
    elapsed = time.time() - t0
    print(f"  [{horizon_label}] RF trained in {elapsed:.1f}s "
          f"({rf.n_estimators} trees, max_depth={rf.max_depth})")

    return rf


def evaluate_rf(rf, X, y, valid, split_name, horizon_label):
    """Evaluate RF predictions: AUC, accuracy, Brier score, calibration."""
    X_eval = X[valid]
    y_eval = y[valid]

    probs = rf.predict_proba(X_eval)[:, 1]  # P(up)
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_eval, probs)
    acc = accuracy_score(y_eval, preds)
    brier = brier_score_loss(y_eval, probs)

    # Calibration curve (10 bins)
    try:
        frac_pos, mean_pred = calibration_curve(y_eval, probs, n_bins=10, strategy='uniform')
        cal_error = np.mean(np.abs(frac_pos - mean_pred))
    except Exception:
        cal_error = float('nan')

    metrics = {
        "split": split_name,
        "horizon": horizon_label,
        "auc": auc,
        "accuracy": acc,
        "brier_score": brier,
        "calibration_error": cal_error,
        "n_samples": len(y_eval),
        "class_balance": float(y_eval.mean()),
    }

    print(f"  [{horizon_label}] {split_name}: AUC={auc:.4f}, Acc={acc:.4f}, "
          f"Brier={brier:.4f}, CalErr={cal_error:.4f}, "
          f"N={len(y_eval)}, P(up)={y_eval.mean():.3f}")

    return metrics, probs


def get_full_probs(rf, X):
    """Get P(up) for all rows (including invalid target rows)."""
    return rf.predict_proba(X)[:, 1]


# ============================================================================
# E-1: THRESHOLD BACKTESTS
# ============================================================================

def run_threshold_backtest(config, probs, mid, timestamps, split_name,
                           start_date, end_date, norm_cutoff, threshold,
                           horizon, maker_fee, execution_mode="maker"):
    """Run a simple threshold strategy through the DeepScalper env.

    Strategy:
      - Buy when P(up) > threshold
      - Sell when P(up) < (1 - threshold)
      - Hold otherwise
      - Close position after `horizon` steps (timeout exit)

    Args:
        execution_mode: "maker" (MAKER_BUY/SELL, next-bar fill) or
                        "taker" (TAKER_BUY/SELL, immediate fill)
    """
    from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv

    # Import run_backtest from baselines
    from run_baselines import run_backtest, make_env

    # make_env forces discrete_dims=6: TakerBuy=0, MakerBuy=1, Hold=2,
    # Cancel=3, MakerSell=4, TakerSell=5
    if execution_mode == "taker":
        BUY = 0   # TAKER_BUY — immediate fill
        SELL = 5  # TAKER_SELL — immediate fill
    else:
        BUY = 1   # MAKER_BUY — next-bar fill
        SELL = 4  # MAKER_SELL — next-bar fill
    HOLD = 2

    env = make_env(config, start_date, end_date, norm_cutoff)

    # The env has its own data length — align probs to env steps
    # Strategy: at each step, look up the RF probability for that step
    state = {"entry_step": -1}

    def rf_threshold_policy(obs, info, step, env):
        current_step = env.current_step
        pos = info.get("position", 0)
        if isinstance(pos, np.ndarray):
            pos = float(pos.flat[0])
        has_position = abs(pos) > 1e-6

        # Timeout exit: close after horizon steps
        if has_position and state["entry_step"] >= 0:
            if current_step - state["entry_step"] >= horizon:
                state["entry_step"] = -1
                return SELL if pos > 0 else BUY
            return HOLD

        # Reset stale entry
        if not has_position and state["entry_step"] >= 0:
            if current_step - state["entry_step"] > 2:
                state["entry_step"] = -1

        if state["entry_step"] >= 0:
            return HOLD  # Waiting for fill

        # Get RF probability for current step
        if current_step >= len(probs):
            return HOLD

        p = probs[current_step]
        if p > threshold:
            state["entry_step"] = current_step
            return BUY
        elif p < (1.0 - threshold):
            state["entry_step"] = current_step
            return SELL
        return HOLD

    exec_tag = "taker" if execution_mode == "taker" else "maker"
    name = f"E1_{exec_tag}_RF_T{threshold:.2f}_H{horizon}_{split_name}"
    metrics = run_backtest(env, rf_threshold_policy, name)
    metrics["split"] = split_name
    metrics["threshold"] = threshold
    metrics["horizon"] = horizon
    env.close()

    return metrics


def run_e1_threshold_sweep(config, data_splits, rf_model, horizon=30,
                           execution_mode="maker"):
    """Sweep thresholds on val and test splits."""
    thresholds = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64]
    maker_fee = config.get("env", {}).get("maker_fee", 0.0002)
    data_config = config.get("data", {})
    results = []

    for split_name, split_data in data_splits.items():
        if split_name == "train":
            continue  # Only backtest on val/test

        X, mid, timestamps = split_data["X"], split_data["mid"], split_data["timestamps"]
        start_date = split_data["start_date"]
        end_date = split_data["end_date"]
        norm_cutoff = split_data["norm_cutoff"]

        # Get RF probs for this split
        probs = get_full_probs(rf_model, X)

        for thresh in thresholds:
            try:
                metrics = run_threshold_backtest(
                    config, probs, mid, timestamps, split_name,
                    start_date, end_date, norm_cutoff,
                    thresh, horizon, maker_fee,
                    execution_mode=execution_mode,
                )
                results.append(metrics)
                print(f"  E1 T={thresh:.2f} {split_name}: "
                      f"PF={metrics['profit_factor']:.3f}, "
                      f"Sharpe={metrics['sharpe']:.2f}, "
                      f"Trades={metrics['trade_count']}")
            except Exception as e:
                print(f"  E1 T={thresh:.2f} {split_name}: FAILED — {e}")

    return results


# ============================================================================
# ARTIFACT SAVING
# ============================================================================

def save_artifacts(output_dir, rf_models, feature_cols, feature_importances,
                   all_probs, eval_metrics, e1_results):
    """Save all artifacts to results/rf_signal/."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save RF models
    import joblib
    for h_label, rf in rf_models.items():
        if rf is not None:
            joblib.dump(rf, os.path.join(output_dir, f"rf_model_{h_label.lower()}.joblib"))
    print(f"  Saved RF models to {output_dir}/")

    # 2. Save probabilities
    np.savez(
        os.path.join(output_dir, "rf_probs.npz"),
        **all_probs,
    )
    print(f"  Saved rf_probs.npz ({len(all_probs)} arrays)")

    # 3. Save feature importances
    # Use first available horizon as primary sort key
    primary_key = sorted(feature_importances.keys())[0]
    imp_df = pd.DataFrame({"feature": feature_cols})
    for h_key, importances in sorted(feature_importances.items()):
        imp_df[f"importance_{h_key}"] = importances
    imp_df = imp_df.sort_values(f"importance_{primary_key}", ascending=False)
    imp_df.to_csv(os.path.join(output_dir, "rf_feature_importance.csv"), index=False)
    print(f"  Saved feature importances (top 5: {imp_df['feature'].head().tolist()})")

    # 4. Save evaluation report
    report_path = os.path.join(output_dir, "rf_evaluation.txt")
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("  PHASE E-0: RF SIGNAL EVALUATION\n")
        f.write("=" * 70 + "\n\n")

        for m in eval_metrics:
            f.write(f"  [{m['horizon']}] {m['split']}: "
                    f"AUC={m['auc']:.4f}, Acc={m['accuracy']:.4f}, "
                    f"Brier={m['brier_score']:.4f}, CalErr={m['calibration_error']:.4f}, "
                    f"N={m['n_samples']}, P(up)={m['class_balance']:.3f}\n")

        f.write(f"\n{'=' * 70}\n")
        imp_col = f"importance_{primary_key}"
        f.write(f"  TOP 10 FEATURES BY IMPORTANCE ({primary_key.upper()})\n")
        f.write(f"{'=' * 70}\n\n")
        for _, row in imp_df.head(10).iterrows():
            f.write(f"  {row['feature']:<25} {row[imp_col]:.4f}\n")

        if e1_results:
            f.write(f"\n{'=' * 70}\n")
            f.write("  E-1: THRESHOLD BACKTEST RESULTS\n")
            f.write(f"{'=' * 70}\n\n")
            f.write(f"  {'Threshold':>10} {'Split':<6} {'Return':>10} {'Sharpe':>8} "
                    f"{'Trades':>7} {'PF':>7} {'MaxDD':>8}\n")
            f.write(f"  {'-' * 60}\n")
            for r in e1_results:
                f.write(f"  {r.get('threshold', 0):>10.2f} {r.get('split', '?'):<6} "
                        f"{r['total_return'] * 100:>9.2f}% {r['sharpe']:>8.2f} "
                        f"{r['trade_count']:>7d} {r['profit_factor']:>7.3f} "
                        f"{r['max_drawdown'] * 100:>7.2f}%\n")

    print(f"  Saved evaluation report to {report_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase E-0: RF Signal Generator")
    parser.add_argument("--config", type=str, default="configs/postaudit_hpo.yaml")
    parser.add_argument("--output_dir", type=str, default="results/rf_signal")
    parser.add_argument("--no_e1", action="store_true", help="Skip E-1 threshold backtests")
    parser.add_argument("--horizons", nargs="*", type=int, default=[1, 30],
                        help="Prediction horizons (default: 1 30)")
    parser.add_argument("--execution", type=str, default="maker",
                        choices=["maker", "taker"],
                        help="E-1 execution mode: maker (next-bar fill) or taker (immediate fill)")
    parser.add_argument("--e1_horizons", nargs="*", type=int, default=None,
                        help="Horizons for E-1 backtest (default: largest trained horizon)")
    args = parser.parse_args()

    print("=" * 70)
    print("  PHASE E-0: Random Forest Signal Generator")
    print("=" * 70)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_config = config.get("data", {})

    # ── 1. Load data: 3 separate handlers (LEAK-1 compliant) ──
    print("\n[1/5] Loading data (LEAK-1: separate handlers per split)...")

    splits_def = {
        "train": {
            "start_date": data_config.get("train_start_date"),
            "end_date": data_config.get("train_end_date"),
            "norm_cutoff": None,  # Train-only stats, no cutoff needed
        },
        "val": {
            "start_date": data_config.get("val_start_date"),
            "end_date": data_config.get("val_end_date"),
            "norm_cutoff": data_config.get("val_start_date"),  # Reset stats at val boundary
        },
        "test": {
            "start_date": data_config.get("test_start_date"),
            "end_date": data_config.get("test_end_date"),
            "norm_cutoff": data_config.get("test_start_date"),  # Reset stats at test boundary
        },
    }

    data_splits = {}
    for split_name, sdef in splits_def.items():
        print(f"\n  Loading {split_name} split...")
        X, mid, timestamps, feature_cols = load_split(
            config,
            sdef["start_date"],
            sdef["end_date"],
            sdef["norm_cutoff"],
        )
        data_splits[split_name] = {
            "X": X, "mid": mid, "timestamps": timestamps,
            "start_date": sdef["start_date"],
            "end_date": sdef["end_date"],
            "norm_cutoff": sdef["norm_cutoff"],
        }

    # ── 2. Construct targets ──
    print("\n[2/5] Constructing targets...")

    targets = {}
    for horizon in args.horizons:
        h_label = f"H{horizon}"
        targets[h_label] = {}
        for split_name, sd in data_splits.items():
            y, valid = construct_target(sd["mid"], horizon)
            targets[h_label][split_name] = {"y": y, "valid": valid}
            n_valid = valid.sum()
            n_pos = y[valid].sum()
            print(f"  {h_label} {split_name}: {n_valid} valid samples, "
                  f"P(up)={n_pos / n_valid:.3f}" if n_valid > 0 else f"  {h_label} {split_name}: 0 valid")

    # ── 3. Train RF on train split ──
    print("\n[3/5] Training Random Forest classifiers...")

    rf_models = {}
    feature_importances = {}
    for horizon in args.horizons:
        h_label = f"H{horizon}"
        train_data = data_splits["train"]
        train_target = targets[h_label]["train"]
        rf = train_rf(
            train_data["X"], train_target["y"], train_target["valid"],
            horizon_label=h_label,
        )
        rf_models[h_label] = rf
        feature_importances[h_label.lower()] = rf.feature_importances_

    # ── 4. Evaluate on all splits ──
    print("\n[4/5] Evaluating RF predictions...")

    eval_metrics = []
    all_probs = {}

    for horizon in args.horizons:
        h_label = f"H{horizon}"
        rf = rf_models[h_label]

        for split_name, sd in data_splits.items():
            target_data = targets[h_label][split_name]

            # Evaluate on valid rows
            metrics, probs_valid = evaluate_rf(
                rf, sd["X"], target_data["y"], target_data["valid"],
                split_name, h_label,
            )
            eval_metrics.append(metrics)

            # Full probs for artifact saving
            full_probs = get_full_probs(rf, sd["X"])
            all_probs[f"probs_{h_label.lower()}_{split_name}"] = full_probs
            all_probs[f"timestamps_{split_name}"] = sd["timestamps"]

    # Add split labels to archive
    all_probs["feature_cols"] = np.array(feature_cols)

    # ── 5. E-1 Threshold backtests ──
    e1_results = []
    if not args.no_e1:
        # Determine which horizons to backtest
        if args.e1_horizons:
            e1_horizons = args.e1_horizons
        else:
            e1_horizons = [max(args.horizons)]  # Default: largest trained horizon

        exec_label = args.execution
        print(f"\n[5/5] Running E-1 threshold backtests ({exec_label} execution)...")

        for e1_h in e1_horizons:
            h_label = f"H{e1_h}"
            if h_label in rf_models:
                print(f"\n  --- {h_label} ({exec_label}) ---")
                results = run_e1_threshold_sweep(
                    config, data_splits, rf_models[h_label],
                    horizon=e1_h, execution_mode=args.execution,
                )
                e1_results.extend(results)
            else:
                print(f"  Skipping E-1 {h_label}: model not trained "
                      f"(available: {list(rf_models.keys())})")
    else:
        print("\n[5/5] E-1 threshold backtests skipped (--no_e1)")

    # ── Save artifacts ──
    print(f"\nSaving artifacts to {args.output_dir}/...")

    save_artifacts(
        args.output_dir,
        rf_models,
        feature_cols, feature_importances,
        all_probs, eval_metrics, e1_results,
    )

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("  PHASE E-0 SUMMARY")
    print(f"{'=' * 70}")

    for m in eval_metrics:
        if m["split"] == "val":
            print(f"  {m['horizon']} val AUC: {m['auc']:.4f}  "
                  f"(accuracy={m['accuracy']:.4f}, N={m['n_samples']})")

    if e1_results:
        e1_h_set = sorted(set(r.get("horizon", 30) for r in e1_results))
        exec_label = args.execution
        print(f"\n  E-1 THRESHOLD SWEEP ({exec_label} execution, H={e1_h_set}):")
        print(f"  {'H':>3} {'Thresh':>7} {'Val PF':>8} {'Val Trades':>10} {'Test PF':>9} {'Test Trades':>11}")
        print(f"  {'-' * 55}")

        best_val_pf = 0
        best_thresh = None
        best_horizon = None
        for h in e1_h_set:
            h_results = [r for r in e1_results if r.get("horizon") == h]
            thresholds_seen = sorted(set(r.get("threshold", 0) for r in h_results))
            for t in thresholds_seen:
                val_r = [r for r in h_results if r.get("threshold") == t and r.get("split") == "val"]
                test_r = [r for r in h_results if r.get("threshold") == t and r.get("split") == "test"]
                val_pf = val_r[0]["profit_factor"] if val_r else 0
                val_trades = val_r[0]["trade_count"] if val_r else 0
                test_pf = test_r[0]["profit_factor"] if test_r else 0
                test_trades = test_r[0]["trade_count"] if test_r else 0
                marker = " <-- BEST" if val_pf > best_val_pf and val_trades >= 50 else ""
                if val_pf > best_val_pf and val_trades >= 50:
                    best_val_pf = val_pf
                    best_thresh = t
                    best_horizon = h
                print(f"  {h:>3} {t:>7.2f} {val_pf:>8.3f} {val_trades:>10d} "
                      f"{test_pf:>9.3f} {test_trades:>11d}{marker}")

        if best_thresh is not None:
            print(f"\n  Best: H{best_horizon} T={best_thresh:.2f} (val PF={best_val_pf:.3f})")
            best_test = [r for r in e1_results
                         if r.get("threshold") == best_thresh
                         and r.get("horizon") == best_horizon
                         and r.get("split") == "test"]
            if best_test:
                test_pf = best_test[0]["profit_factor"]
                if test_pf >= 1.2:
                    print(f"  VERDICT: Test PF={test_pf:.3f} >= 1.2 — tradeable alpha exists!")
                elif test_pf >= 1.0:
                    print(f"  VERDICT: Test PF={test_pf:.3f} — marginal, may improve with RL+RF hybrid (E-2)")
                else:
                    print(f"  VERDICT: Test PF={test_pf:.3f} < 1.0 — RF threshold alone insufficient")
        else:
            print("\n  No threshold produced >= 50 trades on val")

    print(f"\n  Artifacts saved to: {args.output_dir}/")
    print("  Next: Phase E-2 (RL + RF injection) if E-1 shows promise")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
