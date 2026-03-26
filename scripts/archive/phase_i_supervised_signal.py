#!/usr/bin/env python
"""
Phase I: Supervised Gradient Boosting Signal Pipeline
=====================================================
XGBoost + LightGBM REGRESSION on RAW features (no EMA-Z normalization).
Target: continuous forward return in bps. Backtest: threshold-based taker
execution through DeepScalper env for honest fee accounting.

Context:
  14 RL experiments failed (0 profitable). Expert diagnosed: 5-min taker-only
  is a Contextual Bandit, not MDP. DQN/BDQ can't resolve the 0.3 bps signal
  in 50-100 bps noise. Pivoting to supervised regression with threshold trading.
  Gold CME is primary (oracle PF 2.15, 0.35 bps fees).

Assets:
  Gold Level-1: 13 computable micro + 15 macro = 28 features (5 cross-TF excluded)
  BTC Level-5:  45 v2 features (reused from Phase E)

Usage:
    # I1: Gold CME (primary)
    python scripts/phase_i_supervised_signal.py --asset gold

    # I2: BTC Hyperliquid (secondary)
    python scripts/phase_i_supervised_signal.py --asset btc

    # Quick sanity check (regression metrics only, ~30s)
    python scripts/phase_i_supervised_signal.py --asset gold --horizons 1 --thresholds 1.0 2.0 --no_backtest

    # Custom config override
    python scripts/phase_i_supervised_signal.py --asset gold --config configs/my_config.yaml
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import yaml

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "scripts"))
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.metrics import r2_score, mean_absolute_error  # noqa: E402

# ── Action constants (Discrete(6)) ──
TAKER_BUY = 0
HOLD = 2
TAKER_SELL = 5

# ── Default configs per asset ──
ASSET_DEFAULTS = {
    "gold": {
        "config": "configs/phase_g5b_bdq_gc_5min_front.yaml",
    },
    "btc": {
        "config": "configs/phase_f1_fev3_validation.yaml",
    },
}


# ============================================================================
# RAW FEATURE LOADING — GOLD (Level-1 LOB)
# ============================================================================

def load_raw_features_gold(file_path, start_date, end_date):
    """Load Gold parquet and compute Level-1 features WITHOUT EMA-Z normalization.

    Gold has Level-1 LOB only -> 13 computable micro features + 15 macro = 28 total.
    Cross-TF features (obi_burst etc.) excluded — require 1-min pre-computation.

    Returns:
        X: (N, 28) raw feature matrix
        mid: (N,) mid-price array
        feature_cols: list of 28 column names
    """
    df = pd.read_parquet(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    mask = (df["timestamp"] >= pd.Timestamp(start_date)) & \
           (df["timestamp"] <= pd.Timestamp(end_date))
    df = df[mask].reset_index(drop=True)
    n = len(df)
    print(f"  Loaded {n} rows ({start_date} -> {end_date})")

    # ── Mid price ──
    bp1 = df["bid_price_1"].values.astype(np.float64)
    ap1 = df["ask_price_1"].values.astype(np.float64)
    mid = ((bp1 + ap1) / 2.0).astype(np.float32)
    mid_safe = np.where(mid > 0, mid, 1e-9)

    # ── Level-1 Micro features (raw, 13 dims) ──
    bv1 = df["bid_vol_1"].values.astype(np.float64)
    av1 = df["ask_vol_1"].values.astype(np.float64)
    vol_sum = bv1 + av1

    # microprice_basis (bps)
    microprice = (bp1 * av1 + ap1 * bv1) / (vol_sum + 1e-8)
    mp_safe = np.where(microprice > 0, microprice, mid_safe)
    microprice_basis = np.log(mp_safe / mid_safe) * 10000.0

    # DOFI level 1
    bp_prev = np.roll(bp1, 1)
    bp_prev[0] = bp1[0]
    bv_prev = np.roll(bv1, 1)
    bv_prev[0] = bv1[0]
    ap_prev = np.roll(ap1, 1)
    ap_prev[0] = ap1[0]
    av_prev = np.roll(av1, 1)
    av_prev[0] = av1[0]

    w_b = np.where(bp1 > bp_prev, bv1,
            np.where(bp1 < bp_prev, -bv_prev, bv1 - bv_prev))
    w_a = np.where(ap1 < ap_prev, av1,
            np.where(ap1 > ap_prev, -av_prev, av1 - av_prev))
    dofi_1 = w_b - w_a
    dofi_int_1 = np.cumsum(dofi_1)

    # OBI level 1 (= total_obi for single level)
    obi_1 = np.clip((bv1 - av1) / (bv1 + av1 + 1e-8), -1, 1)

    # Distance from mid (bps)
    dist_bid_1 = (mid_safe - bp1) / mid_safe * 10000.0
    dist_ask_1 = (ap1 - mid_safe) / mid_safe * 10000.0

    # Spread (bps)
    spread_bps = ((ap1 - bp1) / mid_safe) * 10000.0

    # DOFI velocity (1st derivative)
    dofi_vel = np.diff(dofi_1, prepend=0.0)

    # Acceleration features (2nd derivatives)
    obi_accel = np.diff(np.diff(obi_1, prepend=obi_1[0]), prepend=0.0)
    dofi_accel = np.diff(np.diff(dofi_1, prepend=dofi_1[0]), prepend=0.0)
    microprice_accel = np.diff(
        np.diff(microprice_basis, prepend=microprice_basis[0]), prepend=0.0)
    spread_velocity = np.diff(spread_bps, prepend=spread_bps[0])

    # ── Macro features (raw, 15 dims) ──
    log_mid = np.log(mid_safe)
    logret_1 = np.diff(log_mid, prepend=log_mid[0]) * 10000.0
    logret_3 = np.zeros_like(logret_1)
    logret_3[3:] = (log_mid[3:] - log_mid[:-3]) * 10000.0
    logret_5 = np.zeros_like(logret_1)
    logret_5[5:] = (log_mid[5:] - log_mid[:-5]) * 10000.0
    logret_15 = np.zeros_like(logret_1)
    logret_15[15:] = (log_mid[15:] - log_mid[:-15]) * 10000.0

    # Parkinson volatility (20-bar)
    if "high" in df.columns and "low" in df.columns:
        h_vals = df["high"].values.astype(np.float64)
        l_vals = df["low"].values.astype(np.float64)
        hl = np.log(h_vals / (l_vals + 1e-9))
        parkinson = pd.Series(hl**2).rolling(20, min_periods=1).mean().values
        parkinson = np.sqrt(parkinson / (4 * np.log(2)))
    else:
        parkinson = np.abs(logret_1) * 0.01

    # Vol regime ratio (short/long)
    vol_15 = pd.Series(np.abs(logret_1)).rolling(15, min_periods=1).std().values
    vol_60 = pd.Series(np.abs(logret_1)).rolling(60, min_periods=1).std().values
    vol_regime = np.clip(vol_15 / (vol_60 + 1e-8) - 1.0, -1.0, 1.0)

    # CVD proxy
    sign_ret = np.sign(logret_1)
    vol = df["volume"].values.astype(np.float64) if "volume" in df.columns \
        else (bv1 + av1)
    signed_vol = sign_ret * vol
    cvd_5 = pd.Series(signed_vol).rolling(5, min_periods=1).sum().values
    cvd_15 = pd.Series(signed_vol).rolling(15, min_periods=1).sum().values

    # Relative volume
    vol_20 = pd.Series(vol).rolling(20, min_periods=1).mean().values
    rvol = vol / (vol_20 + 1e-8) - 1.0

    # RSI (14-bar, centered to [-1, 1])
    gains = np.maximum(logret_1, 0)
    losses = np.maximum(-logret_1, 0)
    avg_gain = pd.Series(gains).ewm(span=14, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = (2 * rs / (1 + rs) - 1.0)

    # Bollinger %B (20-bar, centered to [-1, 1])
    mid_series = pd.Series(mid.astype(np.float64))
    sma20 = mid_series.rolling(20, min_periods=1).mean().values
    std20 = mid_series.rolling(20, min_periods=1).std().values + 1e-8
    bbpct = np.clip((mid - sma20) / (2 * std20), -1, 1)

    # CME time features: day-of-week + session
    ts = pd.to_datetime(df["timestamp"])
    dow = ts.dt.dayofweek.values
    dow_sin = np.sin(2 * np.pi * dow / 5.0)
    dow_cos = np.cos(2 * np.pi * dow / 5.0)
    minutes = ts.dt.hour * 60 + ts.dt.minute
    session_sin = np.sin(2 * np.pi * minutes / 1440)
    session_cos = np.cos(2 * np.pi * minutes / 1440)

    # ── Assemble ──
    feature_arrays = {
        # Micro (13)
        "microprice_basis": microprice_basis,
        "dofi_1": dofi_1,
        "dofi_int_1": dofi_int_1,
        "obi_1": obi_1,
        "dist_bid_1": dist_bid_1,
        "dist_ask_1": dist_ask_1,
        "spread_bps": spread_bps,
        "dofi_velocity": dofi_vel,
        "obi_accel": obi_accel,
        "dofi_accel": dofi_accel,
        "microprice_accel": microprice_accel,
        "spread_velocity": spread_velocity,
        "total_obi": obi_1,
        # Macro (15)
        "logret_1": logret_1,
        "logret_3": logret_3,
        "logret_5": logret_5,
        "logret_15": logret_15,
        "parkinson_vol": parkinson,
        "vol_regime_ratio": vol_regime,
        "cvd_proxy_5": cvd_5,
        "cvd_proxy_15": cvd_15,
        "rvol": rvol,
        "rsi_14": rsi,
        "bbpct_20": bbpct,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "session_sin": session_sin,
        "session_cos": session_cos,
    }

    feature_cols = list(feature_arrays.keys())
    X = np.column_stack([feature_arrays[c] for c in feature_cols]).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, mid, feature_cols


# ============================================================================
# RAW FEATURE LOADING — BTC (Level-5 LOB, reuse from Phase E)
# ============================================================================

def load_raw_features_btc(file_path, start_date, end_date):
    """Load BTC raw features — delegates to phase_e_rf_raw_hyperliquid.load_raw_features().

    Returns:
        X: (N, 45) raw feature matrix (v2 columns, no fev3)
        mid: (N,) mid-price array
        feature_cols: list of 45 column names
    """
    from phase_e_rf_raw_hyperliquid import load_raw_features
    return load_raw_features(file_path, start_date, end_date)


# ============================================================================
# REGRESSION TARGET
# ============================================================================

def construct_regression_target(mid, horizon=1):
    """Continuous forward return in basis points.

    y[t] = (mid[t+H] - mid[t]) / mid[t] * 10000

    Returns:
        y: (N,) return in bps
        valid: (N,) boolean mask (False for last H rows)
    """
    n = len(mid)
    mid_safe = np.where(mid > 0, mid, 1e-9).astype(np.float64)
    y = np.zeros(n, dtype=np.float64)
    valid = np.ones(n, dtype=bool)

    if horizon >= n:
        valid[:] = False
        return y, valid

    y[:-horizon] = ((mid_safe[horizon:] - mid_safe[:-horizon])
                     / mid_safe[:-horizon]) * 10000.0
    valid[-horizon:] = False

    return y, valid


# ============================================================================
# TRAIN REGRESSORS
# ============================================================================

def train_regressors(X_train, y_train, X_val, y_val, horizon,
                     feature_names=None):
    """Train XGBoost + LightGBM regressors with early stopping.

    Conservative hyperparameters (anti-overfit):
      XGB:  n_est=1000, depth=6, lr=0.03, min_child_weight=100, early=50
      LGBM: n_est=1000, depth=6, lr=0.03, min_child_samples=100, early=50
      Both: subsample=0.8, colsample=0.8, reg_alpha=0.1, reg_lambda=1.0

    Returns dict of {model_name: fitted_model}
    """
    import xgboost as xgb
    import lightgbm as lgb

    # Wrap numpy arrays as DataFrames so models get feature names
    if feature_names is not None:
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_val_df = pd.DataFrame(X_val, columns=feature_names)
    else:
        X_train_df = X_train
        X_val_df = X_val

    models = {}

    # ── XGBoost ──
    print(f"  [XGB H{horizon}] Training on {len(X_train)} samples...")
    t0 = time.time()
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.03,
        min_child_weight=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
        early_stopping_rounds=50,
    )
    xgb_model.fit(
        X_train_df, y_train,
        eval_set=[(X_val_df, y_val)],
        verbose=False,
    )
    elapsed = time.time() - t0
    best_iter = getattr(xgb_model, "best_iteration", xgb_model.n_estimators)
    print(f"  [XGB H{horizon}] Trained in {elapsed:.1f}s "
          f"(best_iteration={best_iter})")
    models["xgb"] = xgb_model

    # ── LightGBM ──
    print(f"  [LGBM H{horizon}] Training on {len(X_train)} samples...")
    t0 = time.time()
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.03,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(
        X_train_df, y_train,
        eval_set=[(X_val_df, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    elapsed = time.time() - t0
    best_iter = getattr(lgb_model, "best_iteration_", lgb_model.n_estimators)
    print(f"  [LGBM H{horizon}] Trained in {elapsed:.1f}s "
          f"(best_iteration={best_iter})")
    models["lgbm"] = lgb_model

    return models


# ============================================================================
# EVALUATE REGRESSION
# ============================================================================

def evaluate_regression(model, X, y, valid, split_name, model_name,
                        feature_names=None):
    """Evaluate regression: R^2, MAE, directional accuracy, IC (correlation).

    Returns:
        metrics: dict
        full_preds: (N,) predictions for ALL rows (for backtest alignment)
    """
    if feature_names is not None:
        X_df = pd.DataFrame(X, columns=feature_names)
        X_eval = X_df[valid]
    else:
        X_eval = X[valid]
    y_eval = y[valid]
    preds_valid = model.predict(X_eval)

    r2 = r2_score(y_eval, preds_valid)
    mae = mean_absolute_error(y_eval, preds_valid)

    # Directional accuracy (exclude near-zero moves)
    nonzero = np.abs(y_eval) > 0.01
    if nonzero.sum() > 100:
        dir_acc = np.mean(
            np.sign(preds_valid[nonzero]) == np.sign(y_eval[nonzero]))
    else:
        dir_acc = 0.5

    # Information Coefficient (rank correlation)
    if np.std(preds_valid) > 1e-9 and np.std(y_eval) > 1e-9:
        corr = np.corrcoef(preds_valid, y_eval)[0, 1]
    else:
        corr = 0.0

    print(f"  [{model_name}] {split_name}: R2={r2:.6f}  MAE={mae:.3f}bps  "
          f"DirAcc={dir_acc:.4f}  IC={corr:.4f}  N={len(y_eval)}")

    # Full predictions for env alignment
    if feature_names is not None:
        full_preds = model.predict(X_df)
    else:
        full_preds = model.predict(X)

    return {
        "model": model_name,
        "split": split_name,
        "r2": r2,
        "mae": mae,
        "directional_accuracy": dir_acc,
        "ic": corr,
        "n_samples": len(y_eval),
        "pred_mean": float(np.mean(preds_valid)),
        "pred_std": float(np.std(preds_valid)),
        "target_mean": float(np.mean(y_eval)),
        "target_std": float(np.std(y_eval)),
    }, full_preds


# ============================================================================
# THRESHOLD BACKTEST (through DeepScalper env)
# ============================================================================

def run_regression_backtest(config, predictions, split_name, start_date,
                            end_date, norm_cutoff, threshold, horizon,
                            model_label):
    """Run regression threshold strategy through the env.

    Policy:
      - Buy (TAKER_BUY) when predicted_bps > +threshold
      - Sell (TAKER_SELL) when predicted_bps < -threshold
      - Hold otherwise
      - Close position after `horizon` steps (timeout exit)
    """
    from run_baselines import make_env, run_backtest

    env = make_env(config, start_date, end_date, norm_cutoff)

    state = {"entry_step": -1}

    def regression_policy(obs, info, step, env_ref):
        current_step = env_ref.current_step
        pos = info.get("position", 0)
        if isinstance(pos, np.ndarray):
            pos = float(pos.flat[0])
        has_position = abs(pos) > 1e-6

        # Timeout exit: close after horizon steps
        if has_position and state["entry_step"] >= 0:
            if current_step - state["entry_step"] >= horizon:
                state["entry_step"] = -1
                return TAKER_SELL if pos > 0 else TAKER_BUY
            return HOLD

        # Reset stale entry tracker (fill didn't happen)
        if not has_position and state["entry_step"] >= 0:
            if current_step - state["entry_step"] > 2:
                state["entry_step"] = -1

        if state["entry_step"] >= 0:
            return HOLD  # Waiting for fill

        # Regression signal
        if current_step >= len(predictions):
            return HOLD

        pred = predictions[current_step]
        if pred > threshold:
            state["entry_step"] = current_step
            return TAKER_BUY
        elif pred < -threshold:
            state["entry_step"] = current_step
            return TAKER_SELL
        return HOLD

    name = f"{model_label}_T{threshold:.2f}_H{horizon}_{split_name}"
    metrics = run_backtest(env, regression_policy, name)
    metrics["split"] = split_name
    metrics["threshold"] = threshold
    metrics["horizon"] = horizon
    metrics["model"] = model_label
    env.close()

    # Reset state for next run
    state["entry_step"] = -1

    return metrics


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(output_dir, asset, eval_results, backtest_results,
                 models_by_horizon, feature_cols, predictions_store):
    """Save CSV tables, report, models, and predictions."""
    os.makedirs(output_dir, exist_ok=True)

    # ── Regression metrics CSV ──
    eval_df = pd.DataFrame(eval_results)
    eval_df.to_csv(os.path.join(output_dir, "regression_metrics.csv"),
                   index=False)

    # ── Threshold sweep CSV ──
    if backtest_results:
        bt_df = pd.DataFrame(backtest_results)
        bt_df.to_csv(os.path.join(output_dir, "threshold_sweep.csv"),
                     index=False)

    # ── Feature importance per model/horizon ──
    for h, models in models_by_horizon.items():
        for model_name, model in models.items():
            imp = model.feature_importances_
            imp_df = pd.DataFrame({
                "feature": feature_cols,
                "importance": imp,
            }).sort_values("importance", ascending=False)
            imp_df.to_csv(
                os.path.join(output_dir,
                             f"feature_importance_{model_name}_h{h}.csv"),
                index=False)

    # ── Save models ──
    for h, models in models_by_horizon.items():
        for model_name, model in models.items():
            if model_name == "xgb":
                model.save_model(
                    os.path.join(output_dir, f"xgb_h{h}.json"))
            elif model_name == "lgbm":
                model.booster_.save_model(
                    os.path.join(output_dir, f"lgbm_h{h}.txt"))

    # ── Save predictions ──
    for (model_name, h, split_name), preds in predictions_store.items():
        np.save(
            os.path.join(output_dir,
                         f"predictions_{model_name}_{split_name}_h{h}.npy"),
            preds)

    # ── Summary report ──
    report_path = os.path.join(output_dir, "summary_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"  PHASE I: SUPERVISED SIGNAL — {asset.upper()}\n")
        f.write("=" * 80 + "\n\n")

        # Regression quality
        f.write("REGRESSION QUALITY\n")
        f.write("-" * 80 + "\n")
        f.write(f"  {'Model':<16} {'Split':<6} {'R2':>10} {'MAE(bps)':>10} "
                f"{'DirAcc':>8} {'IC':>8} {'N':>8}\n")
        f.write(f"  {'-'*70}\n")
        for m in eval_results:
            f.write(f"  {m['model']:<16} {m['split']:<6} {m['r2']:>10.6f} "
                    f"{m['mae']:>10.3f} {m['directional_accuracy']:>8.4f} "
                    f"{m['ic']:>8.4f} {m['n_samples']:>8d}\n")

        # Feature importance (top 10 per model/horizon)
        for h, models in models_by_horizon.items():
            for model_name, model in models.items():
                imp = model.feature_importances_
                idx = np.argsort(imp)[::-1][:10]
                f.write(f"\nTOP 10 FEATURES ({model_name} H{h})\n")
                f.write("-" * 40 + "\n")
                for i in idx:
                    f.write(f"  {feature_cols[i]:<25} {imp[i]:.4f}\n")

        # Backtest results
        if backtest_results:
            f.write("\n\nTHRESHOLD BACKTEST RESULTS\n")
            f.write("=" * 80 + "\n")
            f.write(f"  {'Model':<16} {'T':>5} {'Split':<5} {'PF':>7} "
                    f"{'Sharpe':>8} {'Trades':>6} {'Return':>9} "
                    f"{'MaxDD':>8} {'WinRate':>7}\n")
            f.write(f"  {'-'*76}\n")
            for r in backtest_results:
                f.write(f"  {r.get('model','?'):<16} "
                        f"{r.get('threshold',0):>5.2f} "
                        f"{r.get('split','?'):<5} "
                        f"{r['profit_factor']:>7.3f} "
                        f"{r['sharpe']:>8.2f} "
                        f"{r['trade_count']:>6d} "
                        f"{r['total_return']*100:>8.2f}% "
                        f"{r['max_drawdown']*100:>7.2f}% "
                        f"{r.get('win_rate',0):>6.1f}%\n")

            # Best results summary
            f.write("\n\nBEST RESULTS (>= 20 trades)\n")
            f.write("=" * 80 + "\n")
            qualified = [r for r in backtest_results
                         if r.get("trade_count", 0) >= 20]

            for split_name in ["val", "test"]:
                split_res = [r for r in qualified
                             if r.get("split") == split_name]
                if not split_res:
                    f.write(f"\n  {split_name}: No results with >= 20 trades\n")
                    continue

                best = max(split_res, key=lambda r: r["profit_factor"])
                f.write(f"\n  {split_name}: {best['model']} "
                        f"T={best['threshold']:.2f}  "
                        f"PF={best['profit_factor']:.3f}  "
                        f"Trades={best['trade_count']}  "
                        f"Return={best['total_return']*100:.2f}%\n")

                pf = best["profit_factor"]
                if pf >= 1.10:
                    f.write("    -> STRONG: PF >= 1.10 — ship supervised\n")
                elif pf >= 1.02:
                    f.write("    -> MARGINAL: PF 1.02-1.10 — "
                            "proceed to J1 (IQN distributional RL)\n")
                else:
                    f.write("    -> WEAK: PF < 1.02 — "
                            "signal too thin at 5-min\n")

    print(f"\n  Results saved to {output_dir}/")
    return report_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase I: Supervised Gradient Boosting Signal Pipeline")
    parser.add_argument("--asset", type=str, default="gold",
                        choices=["gold", "btc"],
                        help="Asset to run (default: gold)")
    parser.add_argument("--config", type=str, default=None,
                        help="Override config path (default: per-asset)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: results/phase_i_supervised_signal/<asset>)")
    parser.add_argument("--horizons", nargs="*", type=int, default=[1, 3],
                        help="Prediction horizons in bars (default: 1 3)")
    parser.add_argument("--thresholds", nargs="*", type=float, default=None,
                        help="Threshold values (bps). Default: 0.5 to 5.0 step 0.25")
    parser.add_argument("--no_backtest", action="store_true",
                        help="Skip env-based threshold backtests")
    args = parser.parse_args()

    # Resolve config
    config_path = args.config or ASSET_DEFAULTS[args.asset]["config"]
    output_dir = args.output_dir or \
        f"results/phase_i_supervised_signal/{args.asset}"

    # Default threshold sweep: 0.5 to 5.0 step 0.25 (19 values)
    thresholds = args.thresholds or \
        list(np.arange(0.5, 5.25, 0.25))

    print("=" * 70)
    print(f"  PHASE I: SUPERVISED GRADIENT BOOSTING — {args.asset.upper()}")
    print(f"  Config: {config_path}")
    print(f"  Horizons: {args.horizons}")
    print(f"  Thresholds: {len(thresholds)} values "
          f"[{min(thresholds):.2f} .. {max(thresholds):.2f}]")
    print(f"  Backtest: {'ENABLED' if not args.no_backtest else 'DISABLED'}")
    print("=" * 70)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    file_path = data_cfg["file_path"]

    # ── 1. Load RAW features ──
    print(f"\n[1/4] Loading RAW features ({args.asset})...")

    if args.asset == "gold":
        load_fn = load_raw_features_gold
    else:
        load_fn = load_raw_features_btc

    splits = {}
    for split_name, start, end in [
        ("train", data_cfg["train_start_date"], data_cfg["train_end_date"]),
        ("val",   data_cfg["val_start_date"],   data_cfg["val_end_date"]),
        ("test",  data_cfg["test_start_date"],  data_cfg["test_end_date"]),
    ]:
        print(f"\n  {split_name}:")
        X, mid, feature_cols = load_fn(file_path, start, end)
        splits[split_name] = {
            "X": X, "mid": mid,
            "start_date": start, "end_date": end,
        }

    n_features = splits["train"]["X"].shape[1]
    print(f"\n  Feature set: {n_features} features ({args.asset})")

    # ── 2. Train & evaluate for each horizon ──
    print("\n[2/4] Training regressors...")

    all_eval_results = []
    models_by_horizon = {}
    predictions_store = {}  # (model_name, horizon, split_name) -> preds

    for h in args.horizons:
        print(f"\n  --- Horizon H{h} ---")

        # Construct targets
        for split_name in ["train", "val", "test"]:
            y, valid = construct_regression_target(
                splits[split_name]["mid"], horizon=h)
            splits[split_name][f"y_h{h}"] = y
            splits[split_name][f"valid_h{h}"] = valid

            n_valid = valid.sum()
            y_std = np.std(y[valid]) if n_valid > 0 else 0
            print(f"  Target {split_name}: {n_valid} valid, "
                  f"mean={np.mean(y[valid]):.3f} bps, "
                  f"std={y_std:.3f} bps")

        # Train on valid-target rows only
        train_valid = splits["train"][f"valid_h{h}"]
        val_valid = splits["val"][f"valid_h{h}"]

        models = train_regressors(
            splits["train"]["X"][train_valid],
            splits["train"][f"y_h{h}"][train_valid],
            splits["val"]["X"][val_valid],
            splits["val"][f"y_h{h}"][val_valid],
            horizon=h,
            feature_names=feature_cols,
        )
        models_by_horizon[h] = models

        # Evaluate on all splits
        for model_name, model in models.items():
            for split_name in ["train", "val", "test"]:
                y = splits[split_name][f"y_h{h}"]
                valid = splits[split_name][f"valid_h{h}"]

                metrics, full_preds = evaluate_regression(
                    model, splits[split_name]["X"], y, valid,
                    split_name, f"{model_name}_h{h}",
                    feature_names=feature_cols,
                )
                metrics["horizon"] = h
                all_eval_results.append(metrics)
                predictions_store[(model_name, h, split_name)] = full_preds

    # ── 3. Threshold backtests ──
    backtest_results = []
    if not args.no_backtest:
        print("\n[3/4] Threshold backtests (taker execution)...")

        norm_cutoffs = {
            "val": data_cfg["val_start_date"],
            "test": data_cfg["test_start_date"],
        }

        total_runs = (len(models_by_horizon)
                      * 2  # xgb + lgbm
                      * 2  # val + test
                      * len(thresholds))
        run_idx = 0

        for h in args.horizons:
            for model_name, model in models_by_horizon[h].items():
                for split_name in ["val", "test"]:
                    sd = splits[split_name]
                    preds = predictions_store[
                        (model_name, h, split_name)]

                    for thresh in thresholds:
                        run_idx += 1
                        try:
                            m = run_regression_backtest(
                                config, preds, split_name,
                                sd["start_date"], sd["end_date"],
                                norm_cutoffs[split_name],
                                thresh, h,
                                f"{model_name}_h{h}",
                            )
                            backtest_results.append(m)

                            pf = m["profit_factor"]
                            tc = m["trade_count"]
                            sr = m["sharpe"]
                            ret = m["total_return"]
                            print(f"  [{run_idx}/{total_runs}] "
                                  f"{model_name:>4} H{h} "
                                  f"T={thresh:>5.2f} {split_name:<4}: "
                                  f"PF={pf:.3f}  Sharpe={sr:>7.2f}  "
                                  f"Trades={tc:>4d}  "
                                  f"Return={ret*100:>7.2f}%")
                        except Exception as e:
                            print(f"  [{run_idx}/{total_runs}] "
                                  f"{model_name} H{h} T={thresh:.2f} "
                                  f"{split_name}: FAILED — {e}")
    else:
        print("\n[3/4] Threshold backtests skipped (--no_backtest)")

    # ── 4. Save results ──
    print("\n[4/4] Saving results...")
    report_path = save_results(
        output_dir, args.asset,
        all_eval_results, backtest_results,
        models_by_horizon, feature_cols, predictions_store,
    )

    # ── Print summary ──
    print(f"\n{'=' * 70}")
    print(f"  PHASE I SUMMARY — {args.asset.upper()}")
    print(f"{'=' * 70}")

    print("\n  Regression Quality (val/test):")
    for m in all_eval_results:
        if m["split"] in ("val", "test"):
            marker = ""
            if m["directional_accuracy"] > 0.51:
                marker = " *"
            if m["directional_accuracy"] > 0.52:
                marker = " **"
            print(f"    {m['model']:<16} {m['split']:<5}: "
                  f"R2={m['r2']:.6f}  MAE={m['mae']:.3f}  "
                  f"DirAcc={m['directional_accuracy']:.4f}  "
                  f"IC={m['ic']:.4f}{marker}")

    if backtest_results:
        qualified = [r for r in backtest_results
                     if r.get("trade_count", 0) >= 20]
        print("\n  Best Backtests (>= 20 trades):")
        for split_name in ["val", "test"]:
            split_res = [r for r in qualified
                         if r.get("split") == split_name]
            if not split_res:
                print(f"    {split_name}: No qualified results")
                continue

            best = max(split_res, key=lambda r: r["profit_factor"])
            print(f"    {split_name}: {best['model']} "
                  f"T={best['threshold']:.2f}  "
                  f"PF={best['profit_factor']:.3f}  "
                  f"Trades={best['trade_count']}  "
                  f"Return={best['total_return']*100:.2f}%")

        # Decision gate
        test_qualified = [r for r in qualified
                          if r.get("split") == "test"]
        if test_qualified:
            best_test = max(test_qualified,
                            key=lambda r: r["profit_factor"])
            pf = best_test["profit_factor"]
            print("\n  DECISION GATE:")
            if pf >= 1.10:
                print(f"    PF={pf:.3f} >= 1.10 -> SHIP SUPERVISED")
            elif pf >= 1.02:
                print(f"    PF={pf:.3f} in [1.02, 1.10) -> "
                      f"Proceed to J1 (IQN distributional RL)")
            else:
                print(f"    PF={pf:.3f} < 1.02 -> "
                      f"Signal too thin at 5-min")

    print(f"\n  Full report: {report_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
