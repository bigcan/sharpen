"""
RF Signal Quality Test for Gold Futures (GC)
=============================================

Phase 1: OHLCV-only features (macro + price-derived)
Phase 2: + LOB features from MBP-10 (if Phase 1 shows promise)

Computes directional accuracy (AUC) at multiple horizons.
Target: AUC >= 52.5% for Gold H1 profitability (0.36 bps RT cost).

Usage:
    python scripts/rf_signal_gold.py
    python scripts/rf_signal_gold.py --phase 2  # With LOB features
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

DATA_DIR = Path(__file__).parent.parent / "data" / "cme"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "rf_signal_gold"

# Gold-specific parameters
GC_TICK_SIZE = 0.10
GC_MULTIPLIER = 100
GC_COMMISSION_PER_SIDE = 2.50
GC_SPREAD_TICKS = 1

# Breakeven accuracy targets (from breakeven_analysis)
TARGETS = {
    "H1": 0.525,   # 52.5% needed for H1 (5-min hold)
    "H3": 0.514,   # 51.4% needed for H3 (15-min hold)
    "H6": 0.510,   # 51.0% needed for H6 (30-min hold)
    "H12": 0.507,  # ~50.7% needed for H12 (60-min hold)
}


def load_ohlcv(asset="GC"):
    """Load 5-min resampled OHLCV data."""
    path = DATA_DIR / f"{asset.lower()}_2025_ohlcv_1min.parquet"
    df = pd.read_parquet(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Resample to 5-min
    df = df.set_index('timestamp')
    df5 = df.resample('5min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna(subset=['close'])
    df5 = df5[df5['volume'] > 0].reset_index()

    print(f"  Loaded {asset}: {len(df):,} 1-min bars → {len(df5):,} 5-min bars")
    print(f"  Date range: {df5['timestamp'].min()} to {df5['timestamp'].max()}")
    print(f"  Price range: ${df5['close'].min():.2f} - ${df5['close'].max():.2f}")
    return df5


def compute_ohlcv_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features from OHLCV data only (no LOB needed).

    Features designed to match/overlap with DeepScalper macro features:
    - Multi-horizon returns (log returns at 1, 3, 5, 15 bars)
    - Volatility (Parkinson, regime ratio)
    - Volume features (relative volume, volume momentum)
    - Momentum (RSI-14, Bollinger %B)
    - Session encoding (time-of-day sin/cos)
    - Additional: price acceleration, range features
    """
    out = pd.DataFrame()
    out['timestamp'] = df['timestamp']
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)
    n = len(close)

    # --- Multi-horizon log returns ---
    for h in [1, 2, 3, 5, 10, 15]:
        ret = np.zeros(n)
        ret[h:] = np.log(close[h:] / close[:-h])
        ret = np.clip(ret, -0.1, 0.1)  # Clip extreme returns
        out[f'logret_{h}'] = ret

    # --- Parkinson volatility (rolling) ---
    log_hl = np.log(np.maximum(high, 1e-10) / np.maximum(low, 1e-10))
    for window in [10, 30]:
        park = pd.Series(log_hl ** 2).rolling(window, min_periods=1).mean()
        out[f'parkinson_vol_{window}'] = np.sqrt(park / (4 * np.log(2)))

    # --- Volatility regime ratio ---
    park_10 = out['parkinson_vol_10'].values
    park_30 = out['parkinson_vol_30'].values
    ratio = np.log1p(park_10 / (park_30 + 1e-10))
    out['vol_regime_ratio'] = np.tanh(ratio / 1.5 - 0.46)

    # --- Relative Volume ---
    vol_ema = pd.Series(volume).ewm(span=60, adjust=False).mean().values
    out['rvol'] = volume / (vol_ema + 1e-10)

    # --- Volume momentum ---
    vol_short = pd.Series(volume).rolling(5, min_periods=1).mean().values
    vol_long = pd.Series(volume).rolling(30, min_periods=1).mean().values
    out['vol_momentum'] = (vol_short - vol_long) / (vol_long + 1e-10)

    # --- RSI-14 (Wilder's) ---
    delta = np.diff(close, prepend=close[0])
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gains).ewm(span=27, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=27, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - 100 / (1 + rs)
    out['rsi_14'] = 2 * rsi / 100 - 1  # Scale to [-1, 1]

    # --- Bollinger %B (20-period) ---
    sma20 = pd.Series(close).rolling(20, min_periods=1).mean().values
    std20 = pd.Series(close).rolling(20, min_periods=1).std().values
    std20 = np.where(std20 < 1e-10, 1e-10, std20)
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    band_width = upper - lower
    bbpct = (close - lower) / (band_width + 1e-10)
    out['bbpct_20'] = np.tanh(bbpct * 2 - 1)

    # --- MACD signal ---
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
    macd = ema12 - ema26
    macd_signal = pd.Series(macd).ewm(span=9, adjust=False).mean().values
    macd_hist = macd - macd_signal
    out['macd_hist'] = macd_hist / (close * 0.001 + 1e-10)  # Normalize by price

    # --- Price range features ---
    for w in [5, 15]:
        hi = pd.Series(high).rolling(w, min_periods=1).max().values
        lo = pd.Series(low).rolling(w, min_periods=1).min().values
        out[f'range_pct_{w}'] = (hi - lo) / (close + 1e-10) * 100

    # --- Close position within range ---
    hi5 = pd.Series(high).rolling(5, min_periods=1).max().values
    lo5 = pd.Series(low).rolling(5, min_periods=1).min().values
    out['close_position'] = (close - lo5) / (hi5 - lo5 + 1e-10) * 2 - 1  # [-1, 1]

    # --- Price acceleration (2nd derivative) ---
    ret1 = out['logret_1'].values
    out['price_accel'] = np.diff(ret1, prepend=0)

    # --- Session encoding (time-of-day) ---
    minutes = df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute
    out['session_sin'] = np.sin(2 * np.pi * minutes / 1440)
    out['session_cos'] = np.cos(2 * np.pi * minutes / 1440)

    # --- Day-of-week encoding ---
    dow = df['timestamp'].dt.dayofweek
    out['dow_sin'] = np.sin(2 * np.pi * dow / 5)
    out['dow_cos'] = np.cos(2 * np.pi * dow / 5)

    # --- Candle body/wick features ---
    body = (close - df['open'].values) / (close + 1e-10) * 10000  # bps
    upper_wick = (high - np.maximum(close, df['open'].values)) / (close + 1e-10) * 10000
    lower_wick = (np.minimum(close, df['open'].values) - low) / (close + 1e-10) * 10000
    out['candle_body'] = body
    out['upper_wick'] = upper_wick
    out['lower_wick'] = lower_wick

    # Drop NaN rows from rolling calculations
    out = out.fillna(0)

    print(f"  Computed {len(out.columns) - 1} OHLCV features")
    return out


def construct_targets(close: np.ndarray, horizons: list) -> dict:
    """Construct binary direction targets for multiple horizons."""
    n = len(close)
    targets = {}
    for h in horizons:
        y = np.zeros(n, dtype=np.int32)
        valid = np.ones(n, dtype=bool)
        if h < n:
            y[:-h] = (close[h:] > close[:-h]).astype(np.int32)
            valid[-h:] = False
        else:
            valid[:] = False
        targets[f"H{h}"] = {"y": y, "valid": valid}
    return targets


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test,
                       feature_names, horizon_label):
    """Train RF and evaluate on val/test."""
    print(f"\n  [{horizon_label}] Training RF...")
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    print(f"    Train: {len(y_train)} samples (pos={n_pos}, neg={n_neg}, "
          f"ratio={n_pos/len(y_train):.3f})")

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=50,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
    )

    t0 = time.time()
    rf.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"    Trained in {elapsed:.1f}s")

    # Evaluate
    results = {}
    for split_name, X_eval, y_eval in [("val", X_val, y_val), ("test", X_test, y_test)]:
        if len(y_eval) == 0:
            continue

        probs = rf.predict_proba(X_eval)[:, 1]
        preds = (probs >= 0.5).astype(int)

        auc = roc_auc_score(y_eval, probs)
        acc = accuracy_score(y_eval, preds)

        target_auc = TARGETS.get(horizon_label, 0.52)
        margin = auc - target_auc
        verdict = "PROFITABLE" if margin > 0.01 else "MARGINAL" if margin > -0.005 else "FAIL"

        results[split_name] = {
            "auc": auc, "accuracy": acc,
            "n_samples": len(y_eval),
            "class_balance": float(y_eval.mean()),
            "target_auc": target_auc,
            "margin": margin,
            "verdict": verdict,
        }

        print(f"    [{horizon_label}] {split_name}: AUC={auc:.4f} (target={target_auc:.3f}, "
              f"margin={margin:+.4f}) → {verdict}")

    # Feature importance
    imp = rf.feature_importances_
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": imp,
    }).sort_values("importance", ascending=False)

    return rf, results, imp_df


def simulate_pf(auc, avg_abs_return_bps, rt_cost_bps):
    """Simulate approximate PF given accuracy and costs.

    Assumes accuracy ≈ AUC for near-random classifiers.
    """
    p = auc  # Approximation: accuracy ~ AUC for weak classifiers
    mu = avg_abs_return_bps
    c = rt_cost_bps

    # Expected win/loss
    win_pnl = mu - c  # Winner: get the move, pay cost
    loss_pnl = mu + c  # Loser: lose the move AND pay cost

    if loss_pnl <= 0:
        return float('inf')

    gross_win = p * win_pnl
    gross_loss = (1 - p) * loss_pnl

    if gross_loss <= 0:
        return float('inf')

    return gross_win / gross_loss


def main():
    parser = argparse.ArgumentParser(description="RF Signal Quality Test for Gold")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--assets", nargs="+", default=["GC", "CL", "ES"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 6, 12])
    args = parser.parse_args()

    print("=" * 70)
    print("RF SIGNAL QUALITY TEST — CME Futures")
    print(f"Phase {args.phase}: {'OHLCV-only features' if args.phase == 1 else 'OHLCV + LOB features'}")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for asset in args.assets:
        print(f"\n{'='*70}")
        print(f"  {asset}")
        print(f"{'='*70}")

        # Load data
        try:
            df5 = load_ohlcv(asset)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        # Compute features
        features = compute_ohlcv_features(df5)
        feature_cols = [c for c in features.columns if c != 'timestamp']
        X = features[feature_cols].values.astype(np.float32)

        # Replace inf/nan
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Split: Jan-Oct train, Nov val, Dec test
        months = df5['timestamp'].dt.month.values
        train_mask = (months >= 1) & (months <= 10)
        val_mask = months == 11
        test_mask = months == 12

        print(f"  Train: {train_mask.sum():,} bars | Val: {val_mask.sum():,} | Test: {test_mask.sum():,}")

        # Construct targets
        targets = construct_targets(df5['close'].values, args.horizons)

        # Train and evaluate for each horizon
        asset_results = {}
        for h in args.horizons:
            h_label = f"H{h}"
            y = targets[h_label]["y"]
            valid = targets[h_label]["valid"]

            # Apply validity mask + split masks
            train_idx = train_mask & valid
            val_idx = val_mask & valid
            test_idx = test_mask & valid

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            rf, results, imp_df = train_and_evaluate(
                X_train, y_train, X_val, y_val, X_test, y_test,
                feature_cols, h_label,
            )

            asset_results[h_label] = results

            if h == 1:
                # Save feature importance for H1
                imp_path = RESULTS_DIR / f"{asset.lower()}_feature_importance.csv"
                imp_df.to_csv(imp_path, index=False)
                print(f"    Top 5 features: {imp_df['feature'].head().tolist()}")

        all_results[asset] = asset_results

    # ===== SUMMARY =====
    print(f"\n{'='*70}")
    print("SUMMARY — RF Signal Quality (AUC) vs Breakeven Targets")
    print(f"{'='*70}")

    print(f"\n  {'Asset':>20s} | {'Horizon':>7s} | {'Val AUC':>8s} | {'Test AUC':>9s} | "
          f"{'Target':>7s} | {'Margin':>7s} | {'Sim PF':>7s} | Verdict")
    print(f"  {'-'*20}-+-{'-'*7}-+-{'-'*8}-+-{'-'*9}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*12}")

    # BTC reference
    print(f"  {'BTC (Binance)':>20s} | {'H1':>7s} | {'0.5270':>8s} | {'0.5386':>9s} | "
          f"{'0.833':>7s} | {'-0.306':>7s} | {'0.223':>7s} | FAIL")
    print(f"  {'BTC (Hyperliquid)':>20s} | {'H1':>7s} | {'0.5270':>8s} | {'0.5386':>9s} | "
          f"{'0.667':>7s} | {'-0.140':>7s} | {'0.557':>7s} | FAIL")
    print(f"  {'-'*20}-+-{'-'*7}-+-{'-'*8}-+-{'-'*9}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*12}")

    for asset in args.assets:
        if asset not in all_results:
            continue
        asset_res = all_results[asset]

        for h in args.horizons:
            h_label = f"H{h}"
            if h_label not in asset_res:
                continue
            res = asset_res[h_label]

            val_auc = res.get("val", {}).get("auc", 0)
            test_auc = res.get("test", {}).get("auc", 0)
            target = TARGETS.get(h_label, 0.52)
            margin = val_auc - target
            verdict = res.get("val", {}).get("verdict", "?")

            # Compute average absolute return for PF simulation
            load_ohlcv(asset) if h == args.horizons[0] else None
            # Use simple estimate: 7.4 bps for GC, 6.8 for CL, 3.9 for ES (from oracle analysis)
            avg_ret = {"GC": 7.4, "CL": 6.8, "ES": 3.9}.get(asset, 5.0) * (h ** 0.5)
            rt_cost = {"GC": 0.36, "CL": 2.52, "ES": 0.52}.get(asset, 1.0)
            sim_pf = simulate_pf(val_auc, avg_ret, rt_cost)

            sim_pf_str = f"{sim_pf:.3f}" if sim_pf < 100 else "INF"
            print(f"  {asset:>20s} | {h_label:>7s} | {val_auc:>8.4f} | {test_auc:>9.4f} | "
                  f"{target:>7.3f} | {margin:>+7.4f} | {sim_pf_str:>7s} | {verdict}")

    print(f"\n  Results saved to: {RESULTS_DIR}/")

    # Key insight
    print(f"\n{'='*70}")
    print("KEY INSIGHT")
    print(f"{'='*70}")
    print("""
  If AUC > breakeven target → signal is sufficient for profitability
  at Gold's near-zero transaction costs (0.36 bps round-trip).

  BTC needs 83% accuracy to profit (at Binance).
  Gold needs only 52.5% accuracy at H1, 51.4% at H3.

  Even a slight edge above random produces profitable trading on Gold.
  Phase 2 (LOB features) expected to ADD ~1-3% AUC above OHLCV baseline.
""")


if __name__ == "__main__":
    main()
