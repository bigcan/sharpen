#!/usr/bin/env python
"""Methodology stress-test: do evaluation choices mask real feature signal?

Tests 3 axes:
  1. Normalization: EMA-Z+tanh (current) vs RAW features
  2. Target: binary up/down vs fee-aware ternary (long if >5bps, short if <-5bps)
  3. Model: RF vs XGBoost (gradient boosting, deeper interactions)

Runs v1 (7d) vs v2 full (45d) under each condition to see if methodology
changes reveal feature value that the original evaluation missed.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import yaml

sys.path.append(os.getcwd())
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.ensemble import (  # noqa: E402
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import roc_auc_score  # noqa: E402

from finrl_pro_ds.data.feature_engineering import (  # noqa: E402
    MACRO_FEATURE_COLS,
    MICRO_FEATURE_COLS,
    DeepScalperFeatureEngineer,
)
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler  # noqa: E402

# ── Feature subsets ──
FEV3_NAMES = {
    "obi_burst", "obi_trend", "dofi_burst", "microprice_range", "spread_max",
    "obi_accel", "dofi_accel", "microprice_accel", "spread_velocity", "depth_drain",
}
ALL_COLS = list(MICRO_FEATURE_COLS) + list(MACRO_FEATURE_COLS)
V2_COLS = [c for c in ALL_COLS if c not in FEV3_NAMES]
V1_COLS = ["obi_1", "obi_2", "obi_3", "obi_4", "obi_5", "spread_bps", "logret_1"]


def load_normalized(config, start, end, cutoff=None):
    """Load via ParquetDataHandler (EMA-Z + tanh normalized)."""
    data_cfg = config["data"]
    h = ParquetDataHandler(
        file_path=data_cfg["file_path"],
        ticker=data_cfg.get("ticker", "BTCUSDT"),
        feature_config=config.get("features", {}),
        start_date=start, end_date=end, norm_cutoff_date=cutoff,
    )
    X = np.column_stack([
        h._data_arrays[c] if c in h._data_arrays
        else np.zeros(h._len, dtype=np.float32)
        for c in ALL_COLS
    ])
    mid = h._data_arrays["mid_price"].copy()
    h.close()
    return X, mid


def load_raw(file_path, start, end):
    """Load raw parquet and compute features WITHOUT EMA-Z normalization.

    Uses the same feature formulas but skips the normalization pipeline.
    Returns raw feature values that RF/XGB can split on directly.
    """
    df = pd.read_parquet(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    mask = (df["timestamp"] >= pd.Timestamp(start)) & (df["timestamp"] <= pd.Timestamp(end))
    df = df[mask].reset_index(drop=True)

    DeepScalperFeatureEngineer(config={})
    # process_micro computes features AND normalizes in-place
    # We need the raw values BEFORE normalization
    # Strategy: compute raw features manually from LOB columns

    bp1 = df["bid_price_1"].values.astype(np.float64)
    ap1 = df["ask_price_1"].values.astype(np.float64)
    mid = ((bp1 + ap1) / 2.0).astype(np.float32)
    mid_safe = np.where(mid > 0, mid, 1e-9)

    # ── Raw micro features (same formulas, no normalization) ──
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

        bp_prev = np.roll(bp, 1)
        bp_prev[0] = bp[0]
        bv_prev = np.roll(bv, 1)
        bv_prev[0] = bv[0]
        ap_prev = np.roll(ap_i, 1)
        ap_prev[0] = ap_i[0]
        av_prev = np.roll(av_i, 1)
        av_prev[0] = av_i[0]

        w_b = np.where(bp > bp_prev, bv,
                np.where(bp < bp_prev, -bv_prev, bv - bv_prev))
        w_a = np.where(ap_i < ap_prev, av_i,
                np.where(ap_i > ap_prev, -av_prev, av_i - av_prev))
        dofi[i] = w_b - w_a
        dofi_int[i] = np.cumsum(dofi[i])

    total_dofi = sum(dofi.values())

    # OBI per level
    obi = {}
    total_bid = np.zeros(len(df), dtype=np.float64)
    total_ask = np.zeros(len(df), dtype=np.float64)
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
    np.polyfit(np.arange(5), bid_depths.mean(axis=0), 1)[0] if len(df) > 5 else 0
    # Per-row slope is expensive; use simplified version
    slope_asym = np.clip(
        (bid_depths[:, -1] - bid_depths[:, 0]) - (ask_depths[:, -1] - ask_depths[:, 0]),
        -1, 1,
    ) / (bid_depths[:, -1] + ask_depths[:, -1] + 1e-8)

    # Spread (bps)
    spread_bps = ((ap1 - bp1) / mid_safe) * 10000.0

    # DOFI velocity
    dofi_vel = np.diff(total_dofi, prepend=0.0)

    # ── Raw macro features ──
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
        parkinson = np.abs(logret_1) * 0.01  # fallback

    # Vol regime ratio (simplified)
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

    # RSI (14-bar)
    gains = np.maximum(logret_1, 0)
    losses = np.maximum(-logret_1, 0)
    avg_gain = pd.Series(gains).ewm(span=14, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = (2 * rs / (1 + rs) - 1.0)  # centered to [-1, 1]

    # Bollinger %B (20-bar)
    mid_series = pd.Series(mid.astype(np.float64))
    sma20 = mid_series.rolling(20, min_periods=1).mean().values
    std20 = mid_series.rolling(20, min_periods=1).std().values + 1e-8
    bbpct = ((mid - sma20) / (2 * std20))  # centered ~[-1, 1]
    bbpct = np.clip(bbpct, -1, 1)

    # Time features
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        minutes = ts.dt.hour * 60 + ts.dt.minute
        session_sin = np.sin(2 * np.pi * minutes / 1440)
        session_cos = np.cos(2 * np.pi * minutes / 1440)
        # Funding rate clock (8-hour cycle)
        funding_minutes = minutes % 480
        funding_sin = np.sin(2 * np.pi * funding_minutes / 480)
        funding_cos = np.cos(2 * np.pi * funding_minutes / 480)
    else:
        session_sin = session_cos = funding_sin = funding_cos = np.zeros(len(df))

    # ── Assemble feature matrix in SAME column order as ALL_COLS ──
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

    # fev3 cross-TF from parquet (if available)
    for col in ["obi_burst", "obi_trend", "dofi_burst", "microprice_range", "spread_max"]:
        if col in df.columns:
            raw_arrays[col] = df[col].values.astype(np.float64)
        else:
            raw_arrays[col] = np.zeros(len(df), dtype=np.float64)

    # fev3 acceleration
    obi_d1 = np.diff(total_obi, prepend=0.0)
    raw_arrays["obi_accel"] = np.diff(obi_d1, prepend=0.0)
    raw_arrays["dofi_accel"] = np.diff(dofi_vel, prepend=0.0)
    mp_d1 = np.diff(microprice_basis, prepend=0.0)
    raw_arrays["microprice_accel"] = np.diff(mp_d1, prepend=0.0)
    raw_arrays["spread_velocity"] = np.diff(spread_bps, prepend=0.0)
    near_liq = np.zeros(len(df), dtype=np.float64)
    for i in range(1, 4):
        near_liq += df[f"bid_vol_{i}"].values.astype(np.float64)
        near_liq += df[f"ask_vol_{i}"].values.astype(np.float64)
    near_liq /= 6.0
    raw_arrays["depth_drain"] = np.diff(near_liq, prepend=0.0)

    X = np.column_stack([raw_arrays.get(c, np.zeros(len(df))) for c in ALL_COLS]).astype(np.float32)
    # Replace NaN/Inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, mid


def binary_target(mid):
    """Standard: price goes up in 1 step."""
    y = np.zeros(len(mid), dtype=np.int32)
    y[:-1] = (mid[1:] > mid[:-1]).astype(np.int32)
    valid = np.ones(len(mid), dtype=bool)
    valid[-1] = False
    return y, valid


def fee_aware_target(mid, fee_bps=5.0):
    """Fee-aware: 1 if return > fee, 0 if return < -fee, excluded otherwise."""
    n = len(mid)
    mid_safe = np.where(mid > 0, mid, 1e-9)
    ret_bps = np.zeros(n, dtype=np.float64)
    ret_bps[:-1] = ((mid[1:] - mid[:-1]) / mid_safe[:-1]) * 10000.0

    y = np.zeros(n, dtype=np.int32)
    valid = np.zeros(n, dtype=bool)

    long_mask = ret_bps > fee_bps
    short_mask = ret_bps < -fee_bps

    y[long_mask] = 1
    y[short_mask] = 0
    valid[long_mask | short_mask] = True
    valid[-1] = False

    return y, valid


def main():
    with open("configs/phase_f1_fev3_validation.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    data_cfg = config["data"]
    file_path = data_cfg["file_path"]

    col_to_idx = {c: i for i, c in enumerate(ALL_COLS)}
    v1_idx = [col_to_idx[c] for c in V1_COLS]
    v2_idx = [col_to_idx[c] for c in V2_COLS]

    # ── Load data both ways ──
    print("Loading NORMALIZED splits (EMA-Z + tanh)...")
    Xn_tr, mid_tr = load_normalized(config, data_cfg["train_start_date"], data_cfg["train_end_date"])
    Xn_val, mid_val = load_normalized(config, data_cfg["val_start_date"], data_cfg["val_end_date"], data_cfg["val_start_date"])
    Xn_test, mid_test = load_normalized(config, data_cfg["test_start_date"], data_cfg["test_end_date"], data_cfg["test_start_date"])

    print("\nLoading RAW splits (no normalization)...")
    Xr_tr, _ = load_raw(file_path, data_cfg["train_start_date"], data_cfg["train_end_date"])
    Xr_val, _ = load_raw(file_path, data_cfg["val_start_date"], data_cfg["val_end_date"])
    Xr_test, _ = load_raw(file_path, data_cfg["test_start_date"], data_cfg["test_end_date"])

    # Align row counts — normalized pipeline slices ~200 warm-up rows
    # Truncate raw to match normalized (trim from front)
    for label, Xn, Xr_ref in [("train", Xn_tr, "Xr_tr"), ("val", Xn_val, "Xr_val"), ("test", Xn_test, "Xr_test")]:
        Xr = locals()[Xr_ref]
        diff = len(Xr) - len(Xn)
        if diff > 0:
            locals()[Xr_ref] = Xr[diff:]
            print(f"  {label}: trimmed {diff} warm-up rows from raw ({len(Xr)} -> {len(Xn)})")
    Xr_tr = Xr_tr[len(Xr_tr) - len(Xn_tr):]
    Xr_val = Xr_val[len(Xr_val) - len(Xn_val):]
    Xr_test = Xr_test[len(Xr_test) - len(Xn_test):]
    # Also align mid arrays for raw (use same mid from normalized load)
    print(f"\nTrain: {len(Xn_tr)} | Val: {len(Xn_val)} | Test: {len(Xn_test)}")

    # ── Targets ──
    y_bin_tr, v_bin_tr = binary_target(mid_tr)
    y_bin_val, v_bin_val = binary_target(mid_val)
    y_bin_test, v_bin_test = binary_target(mid_test)

    y_fee_tr, v_fee_tr = fee_aware_target(mid_tr, fee_bps=5.0)
    y_fee_val, v_fee_val = fee_aware_target(mid_val, fee_bps=5.0)
    y_fee_test, v_fee_test = fee_aware_target(mid_test, fee_bps=5.0)

    print(f"Binary target: train pos={y_bin_tr[v_bin_tr].mean():.3f} | "
          f"val pos={y_bin_val[v_bin_val].mean():.3f}")
    print(f"Fee-aware target: train valid={v_fee_tr.sum()}/{len(v_fee_tr)} "
          f"({100*v_fee_tr.mean():.1f}%), pos={y_fee_tr[v_fee_tr].mean():.3f} | "
          f"val valid={v_fee_val.sum()}/{len(v_fee_val)} "
          f"({100*v_fee_val.mean():.1f}%), pos={y_fee_val[v_fee_val].mean():.3f}")

    # ── Models ──
    def make_rf():
        return RandomForestClassifier(
            n_estimators=500, max_depth=12, min_samples_leaf=100,
            max_features="sqrt", class_weight="balanced",
            n_jobs=-1, random_state=42,
        )

    def make_xgb():
        return GradientBoostingClassifier(
            n_estimators=500, max_depth=6, min_samples_leaf=100,
            learning_rate=0.05, subsample=0.8, max_features="sqrt",
            random_state=42,
        )

    # ── Run all combinations ──
    print(f"\n{'='*90}")
    print("  METHODOLOGY STRESS TEST")
    print("  Does evaluation methodology mask real feature signal?")
    print(f"{'='*90}\n")
    print(f"  {'Condition':<45} {'Dims':>4}  {'Tr AUC':>7}  {'Val':>7}  {'Test':>7}  {'Time':>6}")
    print(f"  {'-'*85}")

    experiments = []

    for norm_label, X_tr, X_val, X_test in [
        ("NORM", Xn_tr, Xn_val, Xn_test),
        ("RAW", Xr_tr, Xr_val, Xr_test),
    ]:
        for target_label, y_tr, v_tr, y_val, v_val, y_test, v_test in [
            ("binary", y_bin_tr, v_bin_tr, y_bin_val, v_bin_val, y_bin_test, v_bin_test),
            ("fee-aware", y_fee_tr, v_fee_tr, y_fee_val, v_fee_val, y_fee_test, v_fee_test),
        ]:
            for feat_label, idx in [("v1(7d)", v1_idx), ("v2(45d)", v2_idx)]:
                for model_label, model_fn in [("RF", make_rf), ("XGB", make_xgb)]:
                    tag = f"{norm_label} | {target_label} | {feat_label} | {model_label}"

                    X_fit = X_tr[:, idx][v_tr]
                    y_fit = y_tr[v_tr]

                    if len(np.unique(y_fit)) < 2:
                        print(f"  {tag:<45} SKIP (single class)")
                        continue

                    clf = model_fn()
                    t0 = time.time()
                    clf.fit(X_fit, y_fit)
                    elapsed = time.time() - t0

                    aucs = {}
                    for name, Xs, ys, vs in [
                        ("train", X_tr[:, idx], y_tr, v_tr),
                        ("val", X_val[:, idx], y_val, v_val),
                        ("test", X_test[:, idx], y_test, v_test),
                    ]:
                        xs_v = Xs[vs]
                        ys_v = ys[vs]
                        if len(np.unique(ys_v)) < 2:
                            aucs[name] = 0.5
                            continue
                        probs = clf.predict_proba(xs_v)[:, 1]
                        aucs[name] = roc_auc_score(ys_v, probs)

                    print(f"  {tag:<45} {len(idx):>4}  {aucs['train']:>.4f}  "
                          f"{aucs['val']:>.4f}  {aucs['test']:>.4f}  {elapsed:>5.1f}s")

                    experiments.append({
                        "norm": norm_label, "target": target_label,
                        "features": feat_label, "model": model_label,
                        "dims": len(idx),
                        "train_auc": aucs["train"],
                        "val_auc": aucs["val"],
                        "test_auc": aucs["test"],
                    })

    # ── Analysis ──
    print(f"\n{'='*90}")
    print("  ANALYSIS: Which methodology axis matters most?")
    print(f"{'='*90}")

    # Effect of normalization (averaged across other axes)
    norm_effect = {}
    for e in experiments:
        key = (e["target"], e["features"], e["model"])
        norm_effect.setdefault(key, {})
        norm_effect[key][e["norm"]] = e["val_auc"]

    print("\n  Normalization effect (RAW - NORM):")
    for key, vals in sorted(norm_effect.items()):
        if "NORM" in vals and "RAW" in vals:
            delta = vals["RAW"] - vals["NORM"]
            print(f"    {str(key):<50} {delta:+.4f}")

    # Effect of target
    target_effect = {}
    for e in experiments:
        key = (e["norm"], e["features"], e["model"])
        target_effect.setdefault(key, {})
        target_effect[key][e["target"]] = e["val_auc"]

    print("\n  Target effect (fee-aware - binary):")
    for key, vals in sorted(target_effect.items()):
        if "binary" in vals and "fee-aware" in vals:
            delta = vals["fee-aware"] - vals["binary"]
            print(f"    {str(key):<50} {delta:+.4f}")

    # Effect of features (v2 - v1)
    feat_effect = {}
    for e in experiments:
        key = (e["norm"], e["target"], e["model"])
        feat_effect.setdefault(key, {})
        feat_effect[key][e["features"]] = e["val_auc"]

    print("\n  Feature effect (v2 - v1):")
    for key, vals in sorted(feat_effect.items()):
        if "v1(7d)" in vals and "v2(45d)" in vals:
            delta = vals["v2(45d)"] - vals["v1(7d)"]
            print(f"    {str(key):<50} {delta:+.4f}")

    # Effect of model (XGB - RF)
    model_effect = {}
    for e in experiments:
        key = (e["norm"], e["target"], e["features"])
        model_effect.setdefault(key, {})
        model_effect[key][e["model"]] = e["val_auc"]

    print("\n  Model effect (XGB - RF):")
    for key, vals in sorted(model_effect.items()):
        if "RF" in vals and "XGB" in vals:
            delta = vals["XGB"] - vals["RF"]
            print(f"    {str(key):<50} {delta:+.4f}")

    # Best overall
    best = max(experiments, key=lambda e: e["val_auc"])
    print(f"\n  BEST: {best['norm']} | {best['target']} | {best['features']} | "
          f"{best['model']} → val AUC = {best['val_auc']:.4f}")

    worst = min(experiments, key=lambda e: e["val_auc"])
    print(f"  WORST: {worst['norm']} | {worst['target']} | {worst['features']} | "
          f"{worst['model']} → val AUC = {worst['val_auc']:.4f}")
    print(f"  RANGE: {best['val_auc'] - worst['val_auc']:.4f}")

    print(f"\n{'='*90}")


if __name__ == "__main__":
    main()
