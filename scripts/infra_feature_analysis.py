"""
Infrastructure Diagnostic 3: Feature Predictiveness
=====================================================
Tests if LOB + macro features can predict next-step price direction
using simple logistic regression (no RL needed).

If accuracy ≈ 50% → features carry no directional signal.
If train accuracy >> test accuracy → overfitting / distribution shift.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


DATA_PATH = "data/processed/btc_2025_full_year.parquet"

# Feature lists — same as what the env observes
MICRO_FEATURES = []
for i in range(1, 6):
    MICRO_FEATURES.extend([
        f'n_bid_price_{i}', f'n_bid_vol_{i}',
        f'n_ask_price_{i}', f'n_ask_vol_{i}',
    ])
MICRO_FEATURES.extend(['n_spread', 'log_ret'])
for i in range(1, 6):
    MICRO_FEATURES.append(f'n_ofi_{i}')

MACRO_FEATURES = ['z_open', 'z_high', 'z_low', 'z_close', 'z_volume',
                   'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30']

PERIODS = {
    "TRAIN (Jan-Oct)": ("2025-01-01", "2025-10-31 23:59:59"),
    "VAL   (Nov)":     ("2025-11-01", "2025-11-30 23:59:59"),
    "TEST  (Dec)":     ("2025-12-01", "2025-12-31 23:59:59"),
}


def hr(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def load_and_prepare():
    """Load data through feature engineering pipeline."""
    from finrl_pro_ds.data.parquet_handler import ParquetDataHandler

    handler = ParquetDataHandler(
        file_path=DATA_PATH,
        ticker="BTCUSDT",
        feature_config={"volatility_horizon": 0},  # skip vol target
        start_date="2025-01-01 00:00:00",
        end_date="2025-12-31 23:59:59",
    )

    # Build DataFrame from handler arrays
    available_cols = list(handler._data_arrays.keys())
    data = {}
    for col in available_cols:
        arr = handler._data_arrays[col]
        if isinstance(arr, np.ndarray) and arr.ndim == 1:
            data[col] = arr
    
    df = pd.DataFrame(data)
    
    if 'timestamp' in handler._data_arrays:
        ts = handler._data_arrays['timestamp']
        if isinstance(ts, np.ndarray):
            df['timestamp'] = pd.to_datetime(ts)
    
    handler.close()
    return df, available_cols


def create_target(df, horizon=1, threshold_bps=0.0):
    """Create binary target: next mid-price goes up (1) or down (0)."""
    if 'mid_price' in df.columns:
        mid = df['mid_price'].values
    elif 'bid_price_1' in df.columns and 'ask_price_1' in df.columns:
        mid = (df['bid_price_1'].values + df['ask_price_1'].values) / 2.0
    else:
        raise ValueError("No mid-price data found")
    
    future_ret = np.zeros(len(mid))
    future_ret[:-horizon] = (mid[horizon:] - mid[:-horizon]) / mid[:-horizon] * 10000  # in bps
    future_ret[-horizon:] = 0
    
    # Binary: 1 if up by > threshold, 0 otherwise
    target = (future_ret > threshold_bps).astype(int)
    return target, future_ret


def get_features(df, available_cols):
    """Select available features from the feature lists."""
    features = []
    for f in MICRO_FEATURES + MACRO_FEATURES:
        if f in df.columns:
            features.append(f)
    
    # Also try vol_imbalance variants
    for i in range(1, 6):
        col = f'vol_imbalance_{i}'
        if col in df.columns and col not in features:
            features.append(col)
    
    return features


def run_predictiveness_test(df, features, target, train_mask, test_mask, label):
    """Train logistic regression and report metrics."""
    X_train = df.loc[train_mask, features].values
    y_train = target[train_mask]
    X_test = df.loc[test_mask, features].values
    y_test = target[test_mask]
    
    # Clean NaN/Inf
    valid_train = np.all(np.isfinite(X_train), axis=1)
    valid_test = np.all(np.isfinite(X_test), axis=1)
    X_train, y_train = X_train[valid_train], y_train[valid_train]
    X_test, y_test = X_test[valid_test], y_test[valid_test]
    
    if len(X_train) < 100 or len(X_test) < 100:
        print(f"  {label}: INSUFFICIENT DATA (train={len(X_train)}, test={len(X_test)})")
        return None
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    model.fit(X_train_s, y_train)
    
    train_pred = model.predict(X_train_s)
    test_pred = model.predict(X_test_s)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    # AUC
    train_proba = model.predict_proba(X_train_s)[:, 1]
    test_proba = model.predict_proba(X_test_s)[:, 1]
    train_auc = roc_auc_score(y_train, train_proba)
    test_auc = roc_auc_score(y_test, test_proba)
    
    print(f"\n  {label}:")
    print(f"    Train: {len(X_train):,} samples | Acc={train_acc:.4f} | AUC={train_auc:.4f}")
    print(f"    Test:  {len(X_test):,} samples | Acc={test_acc:.4f} | AUC={test_auc:.4f}")
    print(f"    Gap:   Acc={train_acc - test_acc:+.4f} | AUC={train_auc - test_auc:+.4f}")
    
    # Class balance
    print(f"    Target balance (train): {y_train.mean():.3f} up  |  Test: {y_test.mean():.3f} up")
    
    # Feature importance (top 10)
    coef = np.abs(model.coef_[0])
    top_idx = np.argsort(coef)[-10:][::-1]
    print(f"    Top 10 features:")
    for idx in top_idx:
        print(f"      {features[idx]:<25} coef={model.coef_[0][idx]:+.4f}  |abs|={coef[idx]:.4f}")
    
    return {
        'train_acc': train_acc, 'test_acc': test_acc,
        'train_auc': train_auc, 'test_auc': test_auc,
        'model': model, 'features': features,
    }


if __name__ == "__main__":
    hr("INFRASTRUCTURE DIAGNOSTIC 3: FEATURE PREDICTIVENESS")
    print(f"Data: {DATA_PATH}")
    
    df, available_cols = load_and_prepare()
    print(f"Loaded: {len(df):,} rows  |  Available columns: {len(available_cols)}")
    
    features = get_features(df, available_cols)
    print(f"Selected features ({len(features)}): {features[:10]}...")
    
    target, future_ret = create_target(df, horizon=1, threshold_bps=0.0)
    print(f"Target: next-step direction (up vs down)")
    print(f"  Overall up rate: {target.mean():.3f}")
    
    # Create period masks
    if 'timestamp' in df.columns:
        masks = {}
        for name, (start, end) in PERIODS.items():
            masks[name] = (df['timestamp'] >= start) & (df['timestamp'] <= end)
    else:
        # Fall back to index-based splitting
        n = len(df)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        masks = {
            "TRAIN": pd.Series([True]*train_end + [False]*(n-train_end)),
            "VAL":   pd.Series([False]*train_end + [True]*(val_end-train_end) + [False]*(n-val_end)),
            "TEST":  pd.Series([False]*val_end + [True]*(n-val_end)),
        }
    
    # ---- Test 1: Train on TRAIN, test on VAL ----
    hr("TEST 1: Train→Val (in-distribution)")
    train_mask = masks[list(masks.keys())[0]]
    val_mask = masks[list(masks.keys())[1]]
    r1 = run_predictiveness_test(df, features, target, train_mask, val_mask, "Train→Val")
    
    # ---- Test 2: Train on TRAIN, test on TEST ----
    hr("TEST 2: Train→Test (out-of-distribution)")
    test_mask = masks[list(masks.keys())[2]]
    r2 = run_predictiveness_test(df, features, target, train_mask, test_mask, "Train→Test")
    
    # ---- Test 3: Train on each period separately ----
    hr("TEST 3: Within-period predictiveness")
    for name, mask in masks.items():
        period_df = df[mask].copy().reset_index(drop=True)
        period_target = target[mask.values]
        n = len(period_df)
        half = n // 2
        train_half = pd.Series([True]*half + [False]*(n-half))
        test_half = pd.Series([False]*half + [True]*(n-half))
        run_predictiveness_test(period_df, features, period_target, train_half, test_half, f"Within {name}")
    
    # ---- Test 4: Multi-step horizons ----
    hr("TEST 4: Predictiveness at multiple horizons")
    print(f"\n  {'Horizon':>10} {'Train Acc':>12} {'Test Acc':>12} {'AUC':>8}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*8}")
    for h in [1, 5, 10, 30, 60]:
        t, _ = create_target(df, horizon=h, threshold_bps=0.0)
        X_train = df.loc[train_mask, features].values
        y_train = t[train_mask.values]
        X_test = df.loc[test_mask, features].values
        y_test = t[test_mask.values]
        
        valid_tr = np.all(np.isfinite(X_train), axis=1)
        valid_te = np.all(np.isfinite(X_test), axis=1)
        X_train, y_train = X_train[valid_tr], y_train[valid_tr]
        X_test, y_test = X_test[valid_te], y_test[valid_te]
        
        if len(X_train) > 100 and len(X_test) > 100:
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)
            X_te_s = scaler.transform(X_test)
            m = LogisticRegression(max_iter=500, solver='lbfgs')
            m.fit(X_tr_s, y_train)
            tr_acc = accuracy_score(y_train, m.predict(X_tr_s))
            te_acc = accuracy_score(y_test, m.predict(X_te_s))
            te_auc = roc_auc_score(y_test, m.predict_proba(X_te_s)[:, 1])
            print(f"  {h:>10}m {tr_acc:>12.4f} {te_acc:>12.4f} {te_auc:>8.4f}")
    
    hr("VERDICT")
    print("""
  INTERPRETATION GUIDE:
  - Accuracy ~ 50%, AUC ~ 0.50: Features have NO predictive power for direction.
    → Observation space needs rethinking (different features or horizons).
  - Train accuracy >> Test accuracy: Overfitting / distribution shift.
    → Rolling normalization may be leaking, or market regime changed.
  - AUC 0.51-0.53: Weak but real signal (typical for HFT features).
    → RL agent should be able to extract this with proper architecture.
  - Longer horizons more predictable: Consider increasing action step duration.
    """)
    hr("DIAGNOSTIC 3 COMPLETE")
