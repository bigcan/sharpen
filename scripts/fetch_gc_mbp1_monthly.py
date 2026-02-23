"""
Fetch GC MBP-1 month by month to avoid memory issues.
183M records/year = ~15M records/month = ~500MB in memory.

Processes each month to 1-min LOB snapshots, then concatenates.
Then runs the RF signal quality test.
"""
import os
import sys
import time
import gc as garbage_collect
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
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
TARGETS = {"H1": 0.525, "H3": 0.514, "H6": 0.510, "H12": 0.507}


def fetch_and_resample_month(client, start_str, end_str):
    """Fetch MBP-1 for one month, resample to 1-min snapshots."""
    t0 = time.time()
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=["GC.c.0"],
        stype_in="continuous",
        schema="mbp-1",
        start=start_str,
        end=end_str,
    )
    df = data.to_df()
    elapsed = time.time() - t0

    if len(df) == 0:
        return pd.DataFrame()

    print(f"    {start_str}: {len(df):,} ticks ({elapsed:.0f}s)", end="")

    # Remove timezone
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # LOB columns
    lob_cols = [c for c in ['bid_px_00', 'ask_px_00', 'bid_sz_00', 'ask_sz_00']
                if c in df.columns]

    # Take last snapshot per minute
    df_1min = df[lob_cols].resample('1min').last()

    # OHLCV from trade prices
    if 'price' in df.columns:
        trades = df[df['price'] > 0][['price']]
        if 'size' in df.columns:
            trades = df[df['price'] > 0][['price', 'size']]

        if len(trades) > 0:
            ohlcv = trades['price'].resample('1min').agg(
                open='first', high='max', low='min', close='last'
            )
            if 'size' in trades.columns:
                ohlcv['volume'] = trades['size'].resample('1min').sum()
            else:
                ohlcv['volume'] = 0
            df_1min = df_1min.join(ohlcv, how='left')

    # Drop empty minutes
    df_1min = df_1min.dropna(subset=lob_cols[:1])

    # Rename
    rename = {
        'bid_px_00': 'bid_price_1', 'ask_px_00': 'ask_price_1',
        'bid_sz_00': 'bid_vol_1', 'ask_sz_00': 'ask_vol_1',
    }
    df_1min = df_1min.rename(columns=rename)
    df_1min['mid_price'] = (df_1min['bid_price_1'] + df_1min['ask_price_1']) / 2

    # Fill OHLCV gaps with mid_price
    for col in ['close', 'open', 'high', 'low']:
        if col not in df_1min.columns:
            df_1min[col] = df_1min['mid_price']
        else:
            df_1min[col] = df_1min[col].fillna(df_1min['mid_price'])
    if 'volume' not in df_1min.columns:
        df_1min['volume'] = 0
    df_1min['volume'] = df_1min['volume'].fillna(0)

    df_1min = df_1min.reset_index()
    ts_col = 'ts_event' if 'ts_event' in df_1min.columns else df_1min.columns[0]
    df_1min = df_1min.rename(columns={ts_col: 'timestamp'})

    print(f" → {len(df_1min):,} 1-min bars")

    # Free memory
    del df, data
    garbage_collect.collect()

    return df_1min


def fetch_all_months():
    """Fetch MBP-1 month by month for full year."""
    out_path = DATA_DIR / "gc_2025_lob1_1min_front.parquet"
    if out_path.exists():
        print(f"  Already exists: {out_path}")
        return pd.read_parquet(out_path)

    client = db.Historical(API_KEY)
    all_months = []

    print("  Fetching GC.c.0 (front-month) MBP-1 month by month...")
    start = datetime(2025, 1, 1)
    end = datetime(2026, 1, 1)

    current = start
    while current < end:
        next_month = current + relativedelta(months=1)
        try:
            df_month = fetch_and_resample_month(
                client,
                current.strftime("%Y-%m-%d"),
                next_month.strftime("%Y-%m-%d"),
            )
            if len(df_month) > 0:
                all_months.append(df_month)
        except Exception as e:
            print(f"    {current.strftime('%Y-%m')}: ERROR - {e}")
        current = next_month

    if not all_months:
        print("  ERROR: No data fetched")
        return pd.DataFrame()

    result = pd.concat(all_months, ignore_index=True)
    result = result.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    result.to_parquet(out_path, index=False)
    print(f"\n  Saved: {out_path} ({len(result):,} rows, {out_path.stat().st_size / 1e6:.1f} MB)")
    return result


def compute_all_features(df_5min):
    """Compute LOB + OHLCV features on 5-min data."""
    bp1 = df_5min['bid_price_1'].values.astype(np.float64)
    ap1 = df_5min['ask_price_1'].values.astype(np.float64)
    bv1 = df_5min['bid_vol_1'].values.astype(np.float64)
    av1 = df_5min['ask_vol_1'].values.astype(np.float64)
    mid = (bp1 + ap1) / 2.0
    mid_safe = np.where(mid > 0, mid, 1e-9)
    close = df_5min['close'].values.astype(np.float64)
    high = df_5min['high'].values.astype(np.float64)
    low = df_5min['low'].values.astype(np.float64)
    volume = df_5min['volume'].values.astype(np.float64)
    n = len(df_5min)

    features = {}

    # --- LOB Level-1 Features ---
    vol_sum = bv1 + av1
    microprice = (bp1 * av1 + ap1 * bv1) / (vol_sum + 1e-8)
    mp_safe = np.where(microprice > 0, microprice, mid_safe)
    features['microprice_basis'] = np.log(mp_safe / mid_safe) * 10000.0
    features['obi_1'] = np.clip((bv1 - av1) / (bv1 + av1 + 1e-8), -1.0, 1.0)
    features['spread_bps'] = ((ap1 - bp1) / mid_safe) * 10000.0

    bp_prev = np.roll(bp1, 1); bp_prev[0] = bp1[0]
    bv_prev = np.roll(bv1, 1); bv_prev[0] = bv1[0]
    ap_prev = np.roll(ap1, 1); ap_prev[0] = ap1[0]
    av_prev = np.roll(av1, 1); av_prev[0] = av1[0]
    w_b = np.where(bp1 > bp_prev, bv1, np.where(bp1 < bp_prev, -bv_prev, bv1 - bv_prev))
    w_a = np.where(ap1 < ap_prev, av1, np.where(ap1 > ap_prev, -av_prev, av1 - av_prev))
    features['dofi_1'] = w_b - w_a
    features['dofi_int_1'] = pd.Series(features['dofi_1']).rolling(5, min_periods=1).sum().values
    features['dofi_velocity'] = np.diff(features['dofi_1'], prepend=features['dofi_1'][0])
    features['log_vol_ratio'] = np.log(bv1 / (av1 + 1e-8) + 1e-8)
    features['mp_momentum'] = np.diff(features['microprice_basis'], prepend=features['microprice_basis'][0])
    features['obi_momentum'] = np.diff(features['obi_1'], prepend=features['obi_1'][0])
    features['obi_accel'] = np.diff(features['obi_momentum'], prepend=features['obi_momentum'][0])
    features['spread_velocity'] = np.diff(features['spread_bps'], prepend=features['spread_bps'][0])
    features['obi_ema5'] = pd.Series(features['obi_1']).ewm(span=5, adjust=False).mean().values
    features['obi_ema20'] = pd.Series(features['obi_1']).ewm(span=20, adjust=False).mean().values
    features['mp_ema5'] = pd.Series(features['microprice_basis']).ewm(span=5, adjust=False).mean().values

    # --- OHLCV Features ---
    for h in [1, 2, 3, 5, 10, 15]:
        ret = np.zeros(n); ret[h:] = np.log(np.maximum(close[h:], 1e-10) / np.maximum(close[:-h], 1e-10))
        features[f'logret_{h}'] = np.clip(ret, -0.1, 0.1)

    log_hl = np.log(np.maximum(high, 1e-10) / np.maximum(low, 1e-10))
    for w in [10, 30]:
        features[f'parkinson_vol_{w}'] = np.sqrt(pd.Series(log_hl**2).rolling(w, min_periods=1).mean().values / (4*np.log(2)))

    p10, p30 = features['parkinson_vol_10'], features['parkinson_vol_30']
    features['vol_regime_ratio'] = np.tanh(np.log1p(p10 / (p30 + 1e-10)) / 1.5 - 0.46)

    vol_ema = pd.Series(volume).ewm(span=60, adjust=False).mean().values
    features['rvol'] = volume / (vol_ema + 1e-10)

    delta = np.diff(close, prepend=close[0])
    gains, losses = np.where(delta > 0, delta, 0), np.where(delta < 0, -delta, 0)
    avg_g = pd.Series(gains).ewm(span=27, adjust=False).mean().values
    avg_l = pd.Series(losses).ewm(span=27, adjust=False).mean().values
    features['rsi_14'] = 2 * (100 - 100 / (1 + avg_g / (avg_l + 1e-10))) / 100 - 1

    sma20 = pd.Series(close).rolling(20, min_periods=1).mean().values
    std20 = np.where(pd.Series(close).rolling(20, min_periods=1).std().values < 1e-10, 1e-10,
                      pd.Series(close).rolling(20, min_periods=1).std().values)
    features['bbpct_20'] = np.tanh((close - (sma20 - 2*std20)) / (4*std20 + 1e-10) * 2 - 1)

    for w in [5, 15]:
        hi = pd.Series(high).rolling(w, min_periods=1).max().values
        lo = pd.Series(low).rolling(w, min_periods=1).min().values
        features[f'range_pct_{w}'] = (hi - lo) / (close + 1e-10) * 100

    minutes = df_5min['timestamp'].dt.hour * 60 + df_5min['timestamp'].dt.minute
    features['session_sin'] = np.sin(2 * np.pi * minutes / 1440).values
    features['session_cos'] = np.cos(2 * np.pi * minutes / 1440).values

    feature_names = list(features.keys())
    X = np.column_stack([features[k] for k in feature_names]).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    lob_count = sum(1 for k in feature_names if k in [
        'microprice_basis', 'obi_1', 'spread_bps', 'dofi_1', 'dofi_int_1',
        'dofi_velocity', 'log_vol_ratio', 'mp_momentum', 'obi_momentum',
        'obi_accel', 'spread_velocity', 'obi_ema5', 'obi_ema20', 'mp_ema5'])
    print(f"  Features: {X.shape[1]} total ({lob_count} LOB + {X.shape[1] - lob_count} OHLCV)")

    return X, feature_names


def main():
    print("=" * 70)
    print("GOLD RF SIGNAL TEST — Phase 2: OHLCV + LOB Level-1")
    print("=" * 70)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Fetch data
    print("\n[1/4] Fetching MBP-1 data (month by month)...")
    df_1min = fetch_all_months()
    if len(df_1min) == 0:
        print("  No data. Exiting.")
        return
    print(f"  Total: {len(df_1min):,} 1-min bars")

    # Step 2: Resample to 5-min
    print("\n[2/4] Resampling to 5-min...")
    df = df_1min.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    for c in ['bid_price_1', 'ask_price_1', 'bid_vol_1', 'ask_vol_1', 'mid_price']:
        if c in df.columns:
            agg[c] = 'last'

    df5 = df.resample('5min').agg(agg).dropna(subset=['close'])
    df5 = df5[df5['close'] > 0].reset_index()
    print(f"  5-min bars: {len(df5):,}")

    del df_1min, df
    garbage_collect.collect()

    # Step 3: Features
    print("\n[3/4] Computing features...")
    X, feature_names = compute_all_features(df5)

    # Split
    months = df5['timestamp'].dt.month.values
    close = df5['close'].values
    train_mask = (months >= 1) & (months <= 10)
    val_mask = months == 11
    test_mask = months == 12
    print(f"  Train: {train_mask.sum():,} | Val: {val_mask.sum():,} | Test: {test_mask.sum():,}")

    # Step 4: Train & evaluate
    print("\n[4/4] Training RF classifiers...")
    results = {}
    for h in [1, 3, 6, 12]:
        h_label = f"H{h}"
        y = np.zeros(len(close), dtype=np.int32)
        valid = np.ones(len(close), dtype=bool)
        y[:-h] = (close[h:] > close[:-h]).astype(np.int32)
        valid[-h:] = False

        X_tr, y_tr = X[train_mask & valid], y[train_mask & valid]
        X_va, y_va = X[val_mask & valid], y[val_mask & valid]
        X_te, y_te = X[test_mask & valid], y[test_mask & valid]

        n_pos = y_tr.sum()
        print(f"\n  [{h_label}] {len(y_tr)} train (pos={n_pos}, ratio={n_pos/len(y_tr):.3f})")

        rf = RandomForestClassifier(
            n_estimators=500, max_depth=12, min_samples_leaf=50,
            max_features='sqrt', class_weight='balanced', n_jobs=-1, random_state=42)
        t0 = time.time()
        rf.fit(X_tr, y_tr)
        print(f"    Trained in {time.time()-t0:.1f}s")

        h_results = {}
        for sn, Xe, ye in [("val", X_va, y_va), ("test", X_te, y_te)]:
            if len(ye) == 0: continue
            probs = rf.predict_proba(Xe)[:, 1]
            auc = roc_auc_score(ye, probs)
            target = TARGETS[h_label]
            margin = auc - target
            verdict = "PROFITABLE" if margin > 0.01 else "MARGINAL" if margin > -0.005 else "FAIL"
            h_results[sn] = {"auc": auc, "margin": margin, "verdict": verdict}
            print(f"    {sn}: AUC={auc:.4f} (target={target:.3f}, margin={margin:+.4f}) → {verdict}")
        results[h_label] = h_results

        if h == 1:
            imp_df = pd.DataFrame({"feature": feature_names, "importance": rf.feature_importances_})
            imp_df = imp_df.sort_values("importance", ascending=False)
            imp_df.to_csv(RESULTS_DIR / "gc_lob1_feature_importance.csv", index=False)
            print(f"    Top 10:")
            for _, row in imp_df.head(10).iterrows():
                print(f"      {row['feature']:<25s} {row['importance']:.4f}")

    # Summary
    print(f"\n{'='*70}")
    print("COMPARISON: Phase 1 (OHLCV) vs Phase 2 (OHLCV + LOB L1)")
    print(f"{'='*70}")
    p1 = {"H1": 0.5032, "H3": 0.5053, "H6": 0.5017, "H12": 0.4834}

    print(f"  {'H':>4} | {'P1(OHLCV)':>10} | {'P2(+LOB)':>10} | {'Δ AUC':>7} | {'Target':>7} | {'Margin':>7} | Verdict")
    print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*10}")
    for hl in ["H1", "H3", "H6", "H12"]:
        p1v = p1[hl]
        p2v = results.get(hl, {}).get("val", {}).get("auc", 0)
        delta = p2v - p1v
        target = TARGETS[hl]
        margin = p2v - target
        verdict = results.get(hl, {}).get("val", {}).get("verdict", "?")
        print(f"  {hl:>4} | {p1v:>10.4f} | {p2v:>10.4f} | {delta:>+7.4f} | {target:>7.3f} | {margin:>+7.4f} | {verdict}")

    print(f"\n  Results: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
