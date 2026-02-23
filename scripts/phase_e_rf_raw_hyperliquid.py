#!/usr/bin/env python
"""
Phase E Priority 1: RF on RAW Features + H1 Taker + Hyperliquid Fees
=====================================================================
Combines every positive finding from E1/E1c/methodology stress test:
  - RAW features (no EMA-Z normalization) — +1.9% AUC vs normalized
  - Fee-aware target — filters sub-fee noise from training labels
  - H1 horizon — E1c proved H1 is best for taker execution
  - Taker-only at Hyperliquid fees (2.5bps each way = 5bps RT)

Compares against Binance fees (5bps each way = 10bps RT) as control.

Usage:
    python scripts/phase_e_rf_raw_hyperliquid.py
    python scripts/phase_e_rf_raw_hyperliquid.py --no_backtest
    python scripts/phase_e_rf_raw_hyperliquid.py --thresholds 0.50 0.52 0.54 0.56 0.58 0.60
"""
import argparse
import os
import sys
import time
import warnings
import yaml
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "scripts"))
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve

from finrl_pro_ds.data.feature_engineering import MICRO_FEATURE_COLS, MACRO_FEATURE_COLS

# ── Feature columns ──
# Exclude fev3 cross-TF features (only use v2 baseline 45 features)
FEV3_NAMES = {
    "obi_burst", "obi_trend", "dofi_burst", "microprice_range", "spread_max",
    "obi_accel", "dofi_accel", "microprice_accel", "spread_velocity", "depth_drain",
}
ALL_COLS = list(MICRO_FEATURE_COLS) + list(MACRO_FEATURE_COLS)
V2_COLS = [c for c in ALL_COLS if c not in FEV3_NAMES]

# ── Fee structures ──
VENUES = {
    "Binance_VIP0": {"maker_fee": 0.0002, "taker_fee": 0.0005},   # 2bps maker, 5bps taker (10bps RT)
    "Hyperliquid":  {"maker_fee": -0.0002, "taker_fee": 0.00025}, # -2bps rebate, 2.5bps taker (5bps RT)
}


# ============================================================================
# RAW FEATURE LOADING (adapted from fev_methodology_test.py)
# ============================================================================

def load_raw_features(file_path, start_date, end_date):
    """Load raw parquet and compute features WITHOUT EMA-Z normalization.

    Uses the same formulas as feature_engineering.py but skips the
    SymLog -> EMA-Z -> tanh normalization pipeline. RF can split on
    raw values directly — normalization suppresses discriminative tails.

    Returns:
        X: (N, 45) raw feature matrix (v2 columns only)
        mid: (N,) mid-price array
        feature_cols: list of column names
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

    # ── Micro features (raw) ──
    bv1 = df["bid_vol_1"].values.astype(np.float64)
    av1 = df["ask_vol_1"].values.astype(np.float64)
    vol_sum = bv1 + av1

    # microprice_basis (bps)
    microprice = (bp1 * av1 + ap1 * bv1) / (vol_sum + 1e-8)
    mp_safe = np.where(microprice > 0, microprice, mid_safe)
    microprice_basis = np.log(mp_safe / mid_safe) * 10000.0

    # DOFI per level
    dofi = {}
    dofi_int = {}
    for i in range(1, 6):
        bp = df[f"bid_price_{i}"].values.astype(np.float64)
        bv = df[f"bid_vol_{i}"].values.astype(np.float64)
        ap_i = df[f"ask_price_{i}"].values.astype(np.float64)
        av_i = df[f"ask_vol_{i}"].values.astype(np.float64)

        bp_prev = np.roll(bp, 1); bp_prev[0] = bp[0]
        bv_prev = np.roll(bv, 1); bv_prev[0] = bv[0]
        ap_prev = np.roll(ap_i, 1); ap_prev[0] = ap_i[0]
        av_prev = np.roll(av_i, 1); av_prev[0] = av_i[0]

        w_b = np.where(bp > bp_prev, bv,
                np.where(bp < bp_prev, -bv_prev, bv - bv_prev))
        w_a = np.where(ap_i < ap_prev, av_i,
                np.where(ap_i > ap_prev, -av_prev, av_i - av_prev))
        dofi[i] = w_b - w_a
        dofi_int[i] = np.cumsum(dofi[i])

    total_dofi = sum(dofi.values())

    # OBI per level
    obi = {}
    total_bid = np.zeros(n, dtype=np.float64)
    total_ask = np.zeros(n, dtype=np.float64)
    for i in range(1, 6):
        bv = df[f"bid_vol_{i}"].values.astype(np.float64)
        av_i = df[f"ask_vol_{i}"].values.astype(np.float64)
        obi[i] = np.clip((bv - av_i) / (bv + av_i + 1e-8), -1, 1)
        total_bid += bv
        total_ask += av_i
    total_obi = np.clip((total_bid - total_ask) / (total_bid + total_ask + 1e-8), -1, 1)

    # Distance features (bps from mid)
    dist_bid = {}
    dist_ask = {}
    for i in range(1, 6):
        bp = df[f"bid_price_{i}"].values.astype(np.float64)
        ap_i = df[f"ask_price_{i}"].values.astype(np.float64)
        dist_bid[i] = (mid_safe - bp) / mid_safe * 10000.0
        dist_ask[i] = (ap_i - mid_safe) / mid_safe * 10000.0

    # Slope asymmetry
    bid_depths = np.column_stack([dist_bid[i] for i in range(1, 6)])
    ask_depths = np.column_stack([dist_ask[i] for i in range(1, 6)])
    slope_asym = np.clip(
        (bid_depths[:, -1] - bid_depths[:, 0]) - (ask_depths[:, -1] - ask_depths[:, 0]),
        -1, 1
    ) / (bid_depths[:, -1] + ask_depths[:, -1] + 1e-8)

    # Spread (bps)
    spread_bps = ((ap1 - bp1) / mid_safe) * 10000.0

    # DOFI velocity
    dofi_vel = np.diff(total_dofi, prepend=0.0)

    # ── Macro features (raw) ──
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
        hl = np.log(df["high"].values.astype(np.float64) / (df["low"].values.astype(np.float64) + 1e-9))
        parkinson = pd.Series(hl**2).rolling(20, min_periods=1).mean().values
        parkinson = np.sqrt(parkinson / (4 * np.log(2)))
    else:
        parkinson = np.abs(logret_1) * 0.01

    # Vol regime ratio
    vol_15 = pd.Series(np.abs(logret_1)).rolling(15, min_periods=1).std().values
    vol_60 = pd.Series(np.abs(logret_1)).rolling(60, min_periods=1).std().values
    vol_regime = np.clip(vol_15 / (vol_60 + 1e-8) - 1.0, -1.0, 1.0)

    # CVD proxy
    sign_ret = np.sign(logret_1)
    if "volume" in df.columns:
        vol = df["volume"].values.astype(np.float64)
    else:
        vol = total_bid + total_ask
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

    # Time features
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        minutes = ts.dt.hour * 60 + ts.dt.minute
        session_sin = np.sin(2 * np.pi * minutes / 1440)
        session_cos = np.cos(2 * np.pi * minutes / 1440)
        funding_minutes = minutes % 480
        funding_sin = np.sin(2 * np.pi * funding_minutes / 480)
        funding_cos = np.cos(2 * np.pi * funding_minutes / 480)
    else:
        session_sin = session_cos = funding_sin = funding_cos = np.zeros(n)

    # ── Assemble in ALL_COLS order, then select V2 subset ──
    raw_arrays = {
        "microprice_basis": microprice_basis,
        "total_obi": total_obi,
        "slope_asym": slope_asym,
        "spread_bps": spread_bps,
        "dofi_velocity": dofi_vel,
        "logret_1": logret_1, "logret_3": logret_3,
        "logret_5": logret_5, "logret_15": logret_15,
        "parkinson_vol": parkinson, "vol_regime_ratio": vol_regime,
        "cvd_proxy_5": cvd_5, "cvd_proxy_15": cvd_15,
        "rvol": rvol, "rsi_14": rsi, "bbpct_20": bbpct,
        "funding_sin": funding_sin, "funding_cos": funding_cos,
        "session_sin": session_sin, "session_cos": session_cos,
    }
    for i in range(1, 6):
        raw_arrays[f"dofi_{i}"] = dofi[i]
        raw_arrays[f"dofi_int_{i}"] = dofi_int[i]
        raw_arrays[f"obi_{i}"] = obi[i]
        raw_arrays[f"dist_bid_{i}"] = dist_bid[i]
        raw_arrays[f"dist_ask_{i}"] = dist_ask[i]

    # Build full matrix in ALL_COLS order, then filter to V2
    col_to_idx = {c: i for i, c in enumerate(ALL_COLS)}
    v2_idx = [col_to_idx[c] for c in V2_COLS]

    X_full = np.column_stack([
        raw_arrays.get(c, np.zeros(n)) for c in ALL_COLS
    ]).astype(np.float32)
    X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)

    X_v2 = X_full[:, v2_idx]

    return X_v2, mid, V2_COLS


# ============================================================================
# TARGET CONSTRUCTION
# ============================================================================

def binary_target(mid, horizon=1):
    """Standard: 1 if price goes UP in `horizon` steps."""
    n = len(mid)
    y = np.zeros(n, dtype=np.int32)
    valid = np.ones(n, dtype=bool)
    if horizon >= n:
        valid[:] = False
        return y, valid
    y[:-horizon] = (mid[horizon:] > mid[:-horizon]).astype(np.int32)
    valid[-horizon:] = False
    return y, valid


def fee_aware_target(mid, horizon=1, fee_bps=2.5):
    """Fee-aware: 1 if return > fee (profitable long), 0 if < -fee (profitable short).

    Excludes sub-fee moves (|return| < fee_bps) from training.
    For Hyperliquid taker: fee_bps=2.5 (one-way), so RT=5bps.

    Args:
        fee_bps: one-way taker fee in bps. Move must exceed this to be profitable.
    """
    n = len(mid)
    mid_safe = np.where(mid > 0, mid, 1e-9)
    ret_bps = np.zeros(n, dtype=np.float64)
    if horizon >= n:
        return np.zeros(n, dtype=np.int32), np.zeros(n, dtype=bool)
    ret_bps[:-horizon] = ((mid[horizon:] - mid[:-horizon]) / mid_safe[:-horizon]) * 10000.0

    y = np.zeros(n, dtype=np.int32)
    valid = np.zeros(n, dtype=bool)

    # For taker RT: need to cover entry + exit fee = 2 * fee_bps
    rt_fee = 2.0 * fee_bps
    long_mask = ret_bps > rt_fee
    short_mask = ret_bps < -rt_fee

    y[long_mask] = 1
    y[short_mask] = 0
    valid[long_mask | short_mask] = True
    valid[-horizon:] = False

    return y, valid


# ============================================================================
# RF TRAINING & EVALUATION
# ============================================================================

def train_rf(X_train, y_train, valid_train, label=""):
    """Train Random Forest on valid training samples."""
    X_fit = X_train[valid_train]
    y_fit = y_train[valid_train]

    n_pos = y_fit.sum()
    n_neg = len(y_fit) - n_pos
    print(f"  [{label}] Training RF on {len(y_fit)} samples "
          f"(pos={n_pos}, neg={n_neg}, P(up)={n_pos/len(y_fit):.3f})")

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
    return rf


def evaluate_rf(rf, X, y, valid, split_name, label):
    """Return AUC, accuracy, and full P(up) predictions."""
    X_eval = X[valid]
    y_eval = y[valid]

    probs = rf.predict_proba(X_eval)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_eval, probs) if len(np.unique(y_eval)) > 1 else 0.5
    acc = accuracy_score(y_eval, preds)
    brier = brier_score_loss(y_eval, probs)

    try:
        frac_pos, mean_pred = calibration_curve(y_eval, probs, n_bins=10, strategy='uniform')
        cal_error = np.mean(np.abs(frac_pos - mean_pred))
    except Exception:
        cal_error = float('nan')

    print(f"  [{label}] {split_name}: AUC={auc:.4f}, Acc={acc:.4f}, "
          f"Brier={brier:.4f}, CalErr={cal_error:.4f}, N={len(y_eval)}")

    # Return full probs for ALL rows (including invalid target rows)
    full_probs = rf.predict_proba(X)[:, 1]

    return {
        "split": split_name, "label": label,
        "auc": auc, "accuracy": acc, "brier": brier, "cal_error": cal_error,
        "n_samples": len(y_eval),
    }, full_probs


# ============================================================================
# THRESHOLD BACKTEST (through DeepScalper env for honest accounting)
# ============================================================================

def run_threshold_backtest(config, probs, split_name, start_date, end_date,
                           norm_cutoff, threshold, horizon, fee_overrides,
                           venue_name):
    """Run RF threshold strategy through the env with specified fees.

    Policy:
      - Buy (TAKER_BUY) when P(up) > threshold
      - Sell (TAKER_SELL) when P(up) < (1 - threshold)
      - Hold otherwise
      - Close position after `horizon` steps (timeout exit)
    """
    from run_baselines import make_env, run_backtest

    TAKER_BUY = 0
    HOLD = 2
    TAKER_SELL = 5

    env = make_env(config, start_date, end_date, norm_cutoff,
                   fee_overrides=fee_overrides)

    state = {"entry_step": -1}

    def rf_policy(obs, info, step, env_ref):
        current_step = env_ref.current_step
        pos = info.get("position", 0)
        if isinstance(pos, np.ndarray):
            pos = float(pos.flat[0])
        has_position = abs(pos) > 1e-6

        # Timeout exit
        if has_position and state["entry_step"] >= 0:
            if current_step - state["entry_step"] >= horizon:
                state["entry_step"] = -1
                return TAKER_SELL if pos > 0 else TAKER_BUY
            return HOLD

        # Reset stale entry
        if not has_position and state["entry_step"] >= 0:
            if current_step - state["entry_step"] > 2:
                state["entry_step"] = -1

        if state["entry_step"] >= 0:
            return HOLD

        # RF signal
        if current_step >= len(probs):
            return HOLD

        p = probs[current_step]
        if p > threshold:
            state["entry_step"] = current_step
            return TAKER_BUY
        elif p < (1.0 - threshold):
            state["entry_step"] = current_step
            return TAKER_SELL
        return HOLD

    name = f"RF_RAW_{venue_name}_T{threshold:.2f}_H{horizon}_{split_name}"
    metrics = run_backtest(env, rf_policy, name)
    metrics["split"] = split_name
    metrics["threshold"] = threshold
    metrics["horizon"] = horizon
    metrics["venue"] = venue_name
    env.close()

    # Reset state for next run
    state["entry_step"] = -1

    return metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase E Priority 1: RF-RAW + H1 Taker + Hyperliquid")
    parser.add_argument("--config", type=str,
                        default="configs/phase_f1_fev3_validation.yaml")
    parser.add_argument("--output_dir", type=str,
                        default="results/rf_raw_hyperliquid")
    parser.add_argument("--no_backtest", action="store_true",
                        help="Skip env-based threshold backtests (AUC only)")
    parser.add_argument("--thresholds", nargs="*", type=float,
                        default=[0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.58, 0.60],
                        help="Threshold values for P(up) sweep")
    parser.add_argument("--horizon", type=int, default=1,
                        help="Prediction/exit horizon (default: 1 = next bar)")
    args = parser.parse_args()

    print("=" * 70)
    print("  PHASE E PRIORITY 1: RF-RAW + H1 Taker + Hyperliquid")
    print("  Combines: RAW features + fee-aware target + taker execution")
    print("=" * 70)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    file_path = data_cfg["file_path"]

    # ── 1. Load RAW features ──
    print("\n[1/4] Loading RAW features (no EMA-Z normalization)...")

    splits = {}
    for split_name, start, end in [
        ("train", data_cfg["train_start_date"], data_cfg["train_end_date"]),
        ("val",   data_cfg["val_start_date"],   data_cfg["val_end_date"]),
        ("test",  data_cfg["test_start_date"],  data_cfg["test_end_date"]),
    ]:
        print(f"\n  {split_name}:")
        X, mid, feature_cols = load_raw_features(file_path, start, end)
        splits[split_name] = {
            "X": X, "mid": mid,
            "start_date": start, "end_date": end,
        }

    n_features = splits["train"]["X"].shape[1]
    print(f"\n  Feature set: {n_features} features (v2 baseline)")

    # ── 2. Construct targets ──
    print(f"\n[2/4] Constructing targets (H{args.horizon})...")

    targets = {}
    for target_name, target_fn, kwargs in [
        ("binary", binary_target, {"horizon": args.horizon}),
        ("fee_aware_HL", fee_aware_target, {"horizon": args.horizon, "fee_bps": 2.5}),  # Hyperliquid 2.5bps taker
        ("fee_aware_BN", fee_aware_target, {"horizon": args.horizon, "fee_bps": 5.0}),  # Binance 5bps taker
    ]:
        targets[target_name] = {}
        for split_name, sd in splits.items():
            y, valid = target_fn(sd["mid"], **kwargs)
            targets[target_name][split_name] = {"y": y, "valid": valid}
            n_valid = valid.sum()
            n_pos = y[valid].sum() if n_valid > 0 else 0
            print(f"  {target_name} {split_name}: {n_valid} valid "
                  f"({100*n_valid/len(y):.1f}%), P(up)={n_pos/max(n_valid,1):.3f}")

    # ── 3. Train RF models ──
    print(f"\n[3/4] Training Random Forest classifiers...")

    models = {}
    eval_results = []

    for target_name in ["binary", "fee_aware_HL", "fee_aware_BN"]:
        print(f"\n  --- Target: {target_name} ---")
        train_data = splits["train"]
        train_target = targets[target_name]["train"]

        rf = train_rf(train_data["X"], train_target["y"],
                      train_target["valid"], label=target_name)
        models[target_name] = rf

        # Evaluate on all splits
        for split_name, sd in splits.items():
            target_data = targets[target_name][split_name]
            metrics, full_probs = evaluate_rf(
                rf, sd["X"], target_data["y"], target_data["valid"],
                split_name, target_name)
            eval_results.append(metrics)
            splits[split_name][f"probs_{target_name}"] = full_probs

    # ── Feature importance for best model ──
    best_model = models["fee_aware_HL"]
    importances = best_model.feature_importances_
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print(f"\n  Top 10 features (fee_aware_HL):")
    for _, row in imp_df.head(10).iterrows():
        print(f"    {row['feature']:<25} {row['importance']:.4f}")

    # ── 4. Threshold backtests ──
    backtest_results = []
    if not args.no_backtest:
        print(f"\n[4/4] Threshold backtests (taker execution, H{args.horizon})...")

        # Norm cutoff dates for env
        norm_cutoffs = {
            "val": data_cfg["val_start_date"],
            "test": data_cfg["test_start_date"],
        }

        for target_name, rf_model in models.items():
            for venue_name, fees in VENUES.items():
                for split_name in ["val", "test"]:
                    sd = splits[split_name]
                    probs = sd[f"probs_{target_name}"]

                    for thresh in args.thresholds:
                        try:
                            m = run_threshold_backtest(
                                config, probs, split_name,
                                sd["start_date"], sd["end_date"],
                                norm_cutoffs[split_name],
                                thresh, args.horizon,
                                fee_overrides=fees,
                                venue_name=venue_name,
                            )
                            m["target"] = target_name
                            backtest_results.append(m)
                            pf = m["profit_factor"]
                            tc = m["trade_count"]
                            sr = m["sharpe"]
                            ret = m["total_return"]
                            print(f"  {target_name:>14} {venue_name:>14} "
                                  f"T={thresh:.2f} {split_name:<4}: "
                                  f"PF={pf:.3f}  Sharpe={sr:>7.2f}  "
                                  f"Trades={tc:>4d}  Return={ret*100:>7.2f}%")
                        except Exception as e:
                            print(f"  {target_name} {venue_name} T={thresh:.2f} "
                                  f"{split_name}: FAILED — {e}")
    else:
        print("\n[4/4] Threshold backtests skipped (--no_backtest)")

    # ── Save results ──
    os.makedirs(args.output_dir, exist_ok=True)

    # Save evaluation report
    report_path = os.path.join(args.output_dir, "rf_raw_hyperliquid_results.txt")
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("  PHASE E PRIORITY 1: RF-RAW + H1 TAKER + HYPERLIQUID\n")
        f.write("=" * 80 + "\n\n")

        f.write("RF SIGNAL QUALITY (AUC)\n")
        f.write("-" * 60 + "\n")
        f.write(f"  {'Target':<16} {'Split':<6} {'AUC':>7} {'Acc':>7} "
                f"{'Brier':>7} {'N':>7}\n")
        f.write(f"  {'-'*55}\n")
        for m in eval_results:
            f.write(f"  {m['label']:<16} {m['split']:<6} {m['auc']:>7.4f} "
                    f"{m['accuracy']:>7.4f} {m['brier']:>7.4f} {m['n_samples']:>7d}\n")

        f.write(f"\nTOP 10 FEATURES (fee_aware_HL model)\n")
        f.write("-" * 40 + "\n")
        for _, row in imp_df.head(10).iterrows():
            f.write(f"  {row['feature']:<25} {row['importance']:.4f}\n")

        if backtest_results:
            f.write(f"\n\nTHRESHOLD BACKTEST RESULTS\n")
            f.write("=" * 80 + "\n")
            f.write(f"  {'Target':<14} {'Venue':<14} {'T':>5} {'Split':<5} "
                    f"{'PF':>7} {'Sharpe':>8} {'Trades':>6} {'Return':>9} "
                    f"{'MaxDD':>8} {'WinRate':>7}\n")
            f.write(f"  {'-'*78}\n")
            for r in backtest_results:
                f.write(f"  {r.get('target','?'):<14} {r.get('venue','?'):<14} "
                        f"{r.get('threshold',0):>5.2f} {r.get('split','?'):<5} "
                        f"{r['profit_factor']:>7.3f} {r['sharpe']:>8.2f} "
                        f"{r['trade_count']:>6d} {r['total_return']*100:>8.2f}% "
                        f"{r['max_drawdown']*100:>7.2f}% "
                        f"{r.get('win_rate',0):>6.1f}%\n")

            # Summary: best results per venue
            f.write(f"\n\nBEST RESULTS SUMMARY\n")
            f.write("=" * 80 + "\n")
            for venue_name in VENUES:
                venue_results = [r for r in backtest_results
                                 if r.get("venue") == venue_name
                                 and r.get("trade_count", 0) >= 20]
                if not venue_results:
                    f.write(f"\n  {venue_name}: No results with >= 20 trades\n")
                    continue

                # Best val PF with >= 20 trades
                val_results = [r for r in venue_results if r.get("split") == "val"]
                test_results = [r for r in venue_results if r.get("split") == "test"]

                if val_results:
                    best_val = max(val_results, key=lambda r: r["profit_factor"])
                    # Find matching test result
                    matching_test = [r for r in test_results
                                     if r.get("threshold") == best_val.get("threshold")
                                     and r.get("target") == best_val.get("target")]
                    test_pf = matching_test[0]["profit_factor"] if matching_test else "N/A"
                    test_tc = matching_test[0]["trade_count"] if matching_test else "N/A"

                    f.write(f"\n  {venue_name}:\n")
                    f.write(f"    Best val: {best_val['target']} T={best_val['threshold']:.2f} "
                            f"PF={best_val['profit_factor']:.3f} "
                            f"({best_val['trade_count']} trades)\n")
                    f.write(f"    Matching test: PF={test_pf} ({test_tc} trades)\n")

                    # Verdict
                    if matching_test:
                        tp = matching_test[0]["profit_factor"]
                        if tp >= 1.2:
                            f.write(f"    VERDICT: STRONG PASS (PF >= 1.2)\n")
                        elif tp >= 1.05:
                            f.write(f"    VERDICT: MARGINAL PASS (PF 1.05-1.2)\n")
                        elif tp >= 1.0:
                            f.write(f"    VERDICT: BREAKEVEN (PF 1.0-1.05)\n")
                        else:
                            f.write(f"    VERDICT: FAIL (PF < 1.0)\n")

    print(f"\n  Results saved to {report_path}")

    # Save feature importances
    imp_df.to_csv(os.path.join(args.output_dir, "feature_importance.csv"), index=False)

    # Save RF model
    try:
        import joblib
        joblib.dump(models["fee_aware_HL"],
                    os.path.join(args.output_dir, "rf_raw_fee_aware_HL_h1.joblib"))
        print(f"  Model saved to {args.output_dir}/rf_raw_fee_aware_HL_h1.joblib")
    except ImportError:
        print("  joblib not available, skipping model save")

    # ── Print summary ──
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")

    print("\n  RF AUC (val/test):")
    for m in eval_results:
        if m["split"] in ("val", "test"):
            marker = " ***" if m["split"] == "val" and m["auc"] > 0.53 else ""
            print(f"    {m['label']:<16} {m['split']:<5}: AUC={m['auc']:.4f}{marker}")

    if backtest_results:
        print(f"\n  Best Backtest Results (>= 20 trades):")
        for venue_name in VENUES:
            venue_res = [r for r in backtest_results
                         if r.get("venue") == venue_name
                         and r.get("trade_count", 0) >= 20]
            if not venue_res:
                continue

            val_res = [r for r in venue_res if r.get("split") == "val"]
            test_res = [r for r in venue_res if r.get("split") == "test"]

            if val_res:
                best_v = max(val_res, key=lambda r: r["profit_factor"])
                print(f"\n    {venue_name} (val):  T={best_v['threshold']:.2f} "
                      f"[{best_v['target']}] PF={best_v['profit_factor']:.3f} "
                      f"Trades={best_v['trade_count']} "
                      f"Return={best_v['total_return']*100:.2f}%")

            if test_res:
                best_t = max(test_res, key=lambda r: r["profit_factor"])
                print(f"    {venue_name} (test): T={best_t['threshold']:.2f} "
                      f"[{best_t['target']}] PF={best_t['profit_factor']:.3f} "
                      f"Trades={best_t['trade_count']} "
                      f"Return={best_t['total_return']*100:.2f}%")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
