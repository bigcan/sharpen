"""
Fetch GC MBP-1 (top of book) + compute LOB features + RF signal test.

MBP-1 is tiny (~$0.10/year) and gives us the most important LOB features:
- microprice_basis (top feature in BTC RF)
- obi_1 (order book imbalance level 1)
- spread_bps (bid-ask spread)
- dofi_1 (depth-of-flow indicator)

These were the TOP 4 features in our BTC model. Level 1 carries most of the signal.

Usage:
    python scripts/fetch_gc_mbp1_and_test.py
"""
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import databento as db
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

API_KEY = os.environ.get('DATABENTO_API_KEY')
DATA_DIR = Path(__file__).parent.parent / "data" / "cme"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "rf_signal_gold_lob"

# Breakeven targets for Gold
TARGETS = {"H1": 0.525, "H3": 0.514, "H6": 0.510, "H12": 0.507}


def fetch_mbp1():
    """Fetch MBP-1 (top of book) for GC.c.2 full year."""
    out_path = DATA_DIR / "gc_2025_mbp1_1min.parquet"
    if out_path.exists():
        print(f"  MBP-1 parquet already exists: {out_path}")
        return pd.read_parquet(out_path)

    print(f"  Fetching MBP-1 for GC.c.2 (full year 2025)...")
    client = db.Historical(API_KEY)

    t0 = time.time()
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=["GC.c.2"],
        stype_in="continuous",
        schema="mbp-1",
        start="2025-01-01",
        end="2025-12-31",
    )
    df = data.to_df()
    elapsed = time.time() - t0
    print(f"  Fetched {len(df):,} tick records in {elapsed:.0f}s")

    # Resample to 1-minute: take LAST snapshot per minute
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Extract LOB columns
    lob_cols = ['bid_px_00', 'ask_px_00', 'bid_sz_00', 'ask_sz_00']
    available_lob = [c for c in lob_cols if c in df.columns]
    print(f"  LOB columns: {available_lob}")

    # Also get trade price/size for OHLCV
    has_price = 'price' in df.columns
    has_size = 'size' in df.columns

    # Resample LOB: last snapshot per minute
    df_1min = df[available_lob].resample('1min').last()

    # Resample OHLCV from trade prices
    if has_price:
        trades = df[df['price'] > 0]
        if len(trades) > 0:
            ohlcv = trades['price'].resample('1min').agg(['first', 'max', 'min', 'last'])
            ohlcv.columns = ['open', 'high', 'low', 'close']
            if has_size:
                ohlcv['volume'] = trades['size'].resample('1min').sum()
            else:
                ohlcv['volume'] = 0
            df_1min = df_1min.join(ohlcv, how='left')

    # Drop empty minutes
    df_1min = df_1min.dropna(subset=[available_lob[0]] if available_lob else ['close'])

    # Rename to DeepScalper format
    rename = {
        'bid_px_00': 'bid_price_1',
        'ask_px_00': 'ask_price_1',
        'bid_sz_00': 'bid_vol_1',
        'ask_sz_00': 'ask_vol_1',
    }
    df_1min = df_1min.rename(columns=rename)

    # Add mid_price
    if 'bid_price_1' in df_1min.columns and 'ask_price_1' in df_1min.columns:
        df_1min['mid_price'] = (df_1min['bid_price_1'] + df_1min['ask_price_1']) / 2

    # Use mid_price for OHLCV if trade-based OHLCV is missing
    if 'close' not in df_1min.columns or df_1min['close'].isna().mean() > 0.5:
        print("  Trade-based OHLCV sparse, using mid_price...")
        df_1min['close'] = df_1min.get('close', df_1min['mid_price'])
        df_1min['close'] = df_1min['close'].fillna(df_1min['mid_price'])
        df_1min['open'] = df_1min.get('open', df_1min['mid_price'])
        df_1min['open'] = df_1min['open'].fillna(df_1min['mid_price'])
        df_1min['high'] = df_1min.get('high', df_1min['mid_price'])
        df_1min['high'] = df_1min['high'].fillna(df_1min['mid_price'])
        df_1min['low'] = df_1min.get('low', df_1min['mid_price'])
        df_1min['low'] = df_1min['low'].fillna(df_1min['mid_price'])
        df_1min['volume'] = df_1min.get('volume', 0)
        df_1min['volume'] = df_1min['volume'].fillna(0)

    df_1min = df_1min.reset_index()
    ts_col = 'ts_event' if 'ts_event' in df_1min.columns else df_1min.columns[0]
    df_1min = df_1min.rename(columns={ts_col: 'timestamp'})

    # Save
    df_1min.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path} ({len(df_1min):,} rows, "
          f"{out_path.stat().st_size / 1e6:.1f} MB)")

    return df_1min


def compute_lob_features_level1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute LOB micro features from level-1 data only.
    Matches the most important features from DeepScalper feature engineering.
    """
    out = pd.DataFrame()
    out['timestamp'] = df['timestamp']

    bp1 = df['bid_price_1'].values.astype(np.float64)
    ap1 = df['ask_price_1'].values.astype(np.float64)
    bv1 = df['bid_vol_1'].values.astype(np.float64)
    av1 = df['ask_vol_1'].values.astype(np.float64)
    mid = (bp1 + ap1) / 2.0
    mid_safe = np.where(mid > 0, mid, 1e-9)
    n = len(df)

    # 1. Microprice basis (bps) — TOP FEATURE in BTC RF
    vol_sum = bv1 + av1
    microprice = (bp1 * av1 + ap1 * bv1) / (vol_sum + 1e-8)
    mp_safe = np.where(microprice > 0, microprice, mid_safe)
    out['microprice_basis'] = np.log(mp_safe / mid_safe) * 10000.0

    # 2. OBI level 1 — ORDER BOOK IMBALANCE (bounded [-1, 1])
    out['obi_1'] = np.clip((bv1 - av1) / (bv1 + av1 + 1e-8), -1.0, 1.0)

    # 3. Spread in bps
    out['spread_bps'] = ((ap1 - bp1) / mid_safe) * 10000.0

    # 4. DOFI level 1 — DEPTH OF FLOW INDICATOR
    bp_prev = np.roll(bp1, 1); bp_prev[0] = bp1[0]
    bv_prev = np.roll(bv1, 1); bv_prev[0] = bv1[0]
    ap_prev = np.roll(ap1, 1); ap_prev[0] = ap1[0]
    av_prev = np.roll(av1, 1); av_prev[0] = av1[0]

    w_b = np.where(bp1 > bp_prev, bv1,
            np.where(bp1 < bp_prev, -bv_prev, bv1 - bv_prev))
    w_a = np.where(ap1 < ap_prev, av1,
            np.where(ap1 > ap_prev, -av_prev, av1 - av_prev))
    out['dofi_1'] = w_b - w_a

    # 5. Integrated DOFI (5-min rolling sum)
    out['dofi_int_1'] = pd.Series(out['dofi_1'].values).rolling(5, min_periods=1).sum().values

    # 6. DOFI velocity (1st derivative)
    dofi_vals = out['dofi_1'].values
    out['dofi_velocity'] = np.diff(dofi_vals, prepend=dofi_vals[0])

    # 7. Bid/Ask volume ratio
    out['vol_ratio'] = bv1 / (av1 + 1e-8)
    out['log_vol_ratio'] = np.log(out['vol_ratio'].values + 1e-8)

    # 8. Microprice momentum (change in microprice basis)
    mp_basis = out['microprice_basis'].values
    out['mp_momentum'] = np.diff(mp_basis, prepend=mp_basis[0])

    # 9. OBI momentum
    obi = out['obi_1'].values
    out['obi_momentum'] = np.diff(obi, prepend=obi[0])

    # 10. OBI acceleration (2nd derivative)
    obi_mom = out['obi_momentum'].values
    out['obi_accel'] = np.diff(obi_mom, prepend=obi_mom[0])

    # 11. Spread change
    spread = out['spread_bps'].values
    out['spread_velocity'] = np.diff(spread, prepend=spread[0])

    # 12. Volume imbalance EMA (short-term trend)
    out['obi_ema5'] = pd.Series(obi).ewm(span=5, adjust=False).mean().values
    out['obi_ema20'] = pd.Series(obi).ewm(span=20, adjust=False).mean().values

    # 13. Microprice basis EMA
    out['mp_ema5'] = pd.Series(mp_basis).ewm(span=5, adjust=False).mean().values
    out['mp_ema20'] = pd.Series(mp_basis).ewm(span=20, adjust=False).mean().values

    out = out.fillna(0)
    print(f"  Computed {len(out.columns) - 1} LOB level-1 features")
    return out


def compute_ohlcv_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute OHLCV-derived features (same as Phase 1)."""
    out = pd.DataFrame()
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)
    n = len(close)

    for h in [1, 2, 3, 5, 10, 15]:
        ret = np.zeros(n)
        ret[h:] = np.log(np.maximum(close[h:], 1e-10) / np.maximum(close[:-h], 1e-10))
        out[f'logret_{h}'] = np.clip(ret, -0.1, 0.1)

    log_hl = np.log(np.maximum(high, 1e-10) / np.maximum(low, 1e-10))
    for w in [10, 30]:
        park = pd.Series(log_hl ** 2).rolling(w, min_periods=1).mean()
        out[f'parkinson_vol_{w}'] = np.sqrt(park / (4 * np.log(2)))

    park_10 = out['parkinson_vol_10'].values
    park_30 = out['parkinson_vol_30'].values
    out['vol_regime_ratio'] = np.tanh(np.log1p(park_10 / (park_30 + 1e-10)) / 1.5 - 0.46)

    vol_ema = pd.Series(volume).ewm(span=60, adjust=False).mean().values
    out['rvol'] = volume / (vol_ema + 1e-10)

    delta = np.diff(close, prepend=close[0])
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gains).ewm(span=27, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=27, adjust=False).mean().values
    rsi = 100 - 100 / (1 + avg_gain / (avg_loss + 1e-10))
    out['rsi_14'] = 2 * rsi / 100 - 1

    sma20 = pd.Series(close).rolling(20, min_periods=1).mean().values
    std20 = pd.Series(close).rolling(20, min_periods=1).std().values
    std20 = np.where(std20 < 1e-10, 1e-10, std20)
    bbpct = (close - (sma20 - 2 * std20)) / (4 * std20 + 1e-10)
    out['bbpct_20'] = np.tanh(bbpct * 2 - 1)

    for w in [5, 15]:
        hi = pd.Series(high).rolling(w, min_periods=1).max().values
        lo = pd.Series(low).rolling(w, min_periods=1).min().values
        out[f'range_pct_{w}'] = (hi - lo) / (close + 1e-10) * 100

    minutes = df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute
    out['session_sin'] = np.sin(2 * np.pi * minutes / 1440)
    out['session_cos'] = np.cos(2 * np.pi * minutes / 1440)

    return out.fillna(0)


def resample_5min(df):
    """Resample 1-min data to 5-min bars, preserving LOB columns."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # OHLCV aggregation
    ohlcv_agg = {
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }

    # LOB columns: take LAST snapshot per 5-min bar
    lob_cols = [c for c in df.columns if any(x in c for x in ['bid_price', 'ask_price', 'bid_vol', 'ask_vol', 'mid_price'])]
    lob_agg = {c: 'last' for c in lob_cols if c in df.columns}

    agg_dict = {**ohlcv_agg, **lob_agg}
    # Only aggregate columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

    df5 = df.resample('5min').agg(agg_dict).dropna(subset=['close'])
    df5 = df5[df5['close'] > 0]

    return df5.reset_index()


def main():
    print("=" * 70)
    print("GOLD RF SIGNAL TEST — Phase 2: OHLCV + LOB Level-1 Features")
    print("=" * 70)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Fetch MBP-1 data
    print("\n[1/5] Fetching MBP-1 (top of book) data...")
    df_1min = fetch_mbp1()
    print(f"  Got {len(df_1min):,} 1-min bars with LOB data")

    # Step 2: Resample to 5-min
    print("\n[2/5] Resampling to 5-min bars...")
    df_5min = resample_5min(df_1min)
    print(f"  Got {len(df_5min):,} 5-min bars")

    # Step 3: Compute features
    print("\n[3/5] Computing features...")
    lob_features = compute_lob_features_level1(df_5min)
    ohlcv_features = compute_ohlcv_features(df_5min)

    # Merge all features
    lob_cols = [c for c in lob_features.columns if c != 'timestamp']
    ohlcv_cols = [c for c in ohlcv_features.columns]
    all_feature_cols = lob_cols + ohlcv_cols

    X = np.column_stack([
        lob_features[lob_cols].values,
        ohlcv_features[ohlcv_cols].values,
    ]).astype(np.float32)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Total features: {X.shape[1]} ({len(lob_cols)} LOB + {len(ohlcv_cols)} OHLCV)")

    # Step 4: Split and train
    print("\n[4/5] Training RF classifiers...")
    months = df_5min['timestamp'].dt.month.values
    close = df_5min['close'].values
    train_mask = (months >= 1) & (months <= 10)
    val_mask = months == 11
    test_mask = months == 12

    print(f"  Train: {train_mask.sum():,} | Val: {val_mask.sum():,} | Test: {test_mask.sum():,}")

    horizons = [1, 3, 6, 12]
    results = {}

    for h in horizons:
        h_label = f"H{h}"
        # Target: direction
        y = np.zeros(len(close), dtype=np.int32)
        valid = np.ones(len(close), dtype=bool)
        if h < len(close):
            y[:-h] = (close[h:] > close[:-h]).astype(np.int32)
            valid[-h:] = False

        train_idx = train_mask & valid
        val_idx = val_mask & valid
        test_idx = test_mask & valid

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        n_pos = y_train.sum()
        print(f"\n  [{h_label}] Training on {len(y_train)} samples "
              f"(pos={n_pos}, ratio={n_pos/len(y_train):.3f})")

        rf = RandomForestClassifier(
            n_estimators=500, max_depth=12, min_samples_leaf=50,
            max_features='sqrt', class_weight='balanced',
            n_jobs=-1, random_state=42,
        )
        t0 = time.time()
        rf.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f"    Trained in {elapsed:.1f}s")

        h_results = {}
        for split_name, X_eval, y_eval in [("val", X_val, y_val), ("test", X_test, y_test)]:
            if len(y_eval) == 0:
                continue
            probs = rf.predict_proba(X_eval)[:, 1]
            auc = roc_auc_score(y_eval, probs)
            acc = accuracy_score(y_eval, (probs >= 0.5).astype(int))
            target = TARGETS.get(h_label, 0.52)
            margin = auc - target
            verdict = "PROFITABLE" if margin > 0.01 else "MARGINAL" if margin > -0.005 else "FAIL"

            h_results[split_name] = {"auc": auc, "acc": acc, "margin": margin, "verdict": verdict}
            print(f"    [{h_label}] {split_name}: AUC={auc:.4f} (target={target:.3f}, "
                  f"margin={margin:+.4f}) → {verdict}")

        results[h_label] = h_results

        # Feature importance for H1
        if h == 1:
            imp = rf.feature_importances_
            imp_df = pd.DataFrame({
                "feature": all_feature_cols,
                "importance": imp,
            }).sort_values("importance", ascending=False)
            imp_df.to_csv(RESULTS_DIR / "gc_lob_feature_importance.csv", index=False)
            print(f"    Top 10 features:")
            for _, row in imp_df.head(10).iterrows():
                source = "LOB" if row['feature'] in lob_cols else "OHLCV"
                print(f"      {row['feature']:<25s} {row['importance']:.4f} ({source})")

    # Step 5: Summary
    print(f"\n{'='*70}")
    print("SUMMARY — Gold RF Signal Quality (OHLCV + LOB Level-1)")
    print(f"{'='*70}")

    print(f"\n  Phase 1 (OHLCV only) vs Phase 2 (OHLCV + LOB):")
    print(f"  {'Horizon':>7s} | {'P1 Val':>8s} | {'P2 Val':>8s} | {'Delta':>7s} | "
          f"{'P2 Test':>8s} | {'Target':>7s} | {'Margin':>7s} | Verdict")
    print(f"  {'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*10}")

    # Phase 1 baselines (from previous run)
    p1_baselines = {"H1": 0.5032, "H3": 0.5053, "H6": 0.5017, "H12": 0.4834}

    for h_label in ["H1", "H3", "H6", "H12"]:
        p1 = p1_baselines.get(h_label, 0.50)
        r = results.get(h_label, {})
        p2_val = r.get("val", {}).get("auc", 0)
        p2_test = r.get("test", {}).get("auc", 0)
        delta = p2_val - p1
        target = TARGETS.get(h_label, 0.52)
        margin = p2_val - target
        verdict = r.get("val", {}).get("verdict", "?")

        print(f"  {h_label:>7s} | {p1:>8.4f} | {p2_val:>8.4f} | {delta:>+7.4f} | "
              f"{p2_test:>8.4f} | {target:>7.3f} | {margin:>+7.4f} | {verdict}")

    print(f"\n  LOB features should add meaningful signal above OHLCV baseline.")
    print(f"  Results saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
