"""
Infrastructure Diagnostic 3: Feature Predictiveness (V2)
=========================================================
Tests if V2 LOB micro + macro features can predict next-step price
direction using logistic regression AND random forest.

Key outputs:
  - Per-feature AUC (logistic regression)
  - Per-feature Mutual Information with target
  - Multi-horizon predictiveness scan
  - Feature importance ranking (Random Forest)

If accuracy ≈ 50% → features carry no directional signal.
If train accuracy >> test accuracy → overfitting / distribution shift.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

# Import canonical feature lists from V2 pipeline
from finrl_pro_ds.data.feature_engineering import MICRO_FEATURE_COLS, MACRO_FEATURE_COLS

# ─── Configuration ──────────────────────────────────────────────────────
DATA_PATH = "data/processed/btc_2025_jan_jun.parquet"

PERIODS = {
    "TRAIN (Jan-Apr)": ("2025-01-01", "2025-04-30 23:59:59"),
    "VAL   (May)":     ("2025-05-01", "2025-05-31 23:59:59"),
    "TEST  (Jun)":     ("2025-06-01", "2025-06-30 23:59:59"),
}

ALL_FEATURES = MICRO_FEATURE_COLS + MACRO_FEATURE_COLS
RESULTS_DIR = "results/feature_audit"


def hr(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def load_and_prepare():
    """Load data through V2 feature engineering pipeline."""
    from finrl_pro_ds.data.parquet_handler import ParquetDataHandler

    handler = ParquetDataHandler(
        file_path=DATA_PATH,
        ticker="BTCUSDT",
        feature_config={"volatility_horizon": 0},
        start_date="2025-01-01 00:00:00",
        end_date="2025-06-30 23:59:59",
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

    target = (future_ret > threshold_bps).astype(int)
    return target, future_ret


def get_available_features(df):
    """Select features available in the DataFrame from canonical lists."""
    features = []
    missing = []
    for f in ALL_FEATURES:
        if f in df.columns:
            features.append(f)
        else:
            missing.append(f)

    if missing:
        print(f"  ⚠ Missing {len(missing)} features: {missing[:5]}...")
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
    print(f"    Top 10 features (|coef|):")
    for idx in top_idx:
        print(f"      {features[idx]:<25} coef={model.coef_[0][idx]:+.4f}  |abs|={coef[idx]:.4f}")

    return {
        'train_acc': train_acc, 'test_acc': test_acc,
        'train_auc': train_auc, 'test_auc': test_auc,
        'model': model, 'features': features,
        'X_train': X_train_s, 'y_train': y_train,
        'X_test': X_test_s, 'y_test': y_test,
    }


def run_per_feature_auc(df, features, target, train_mask, test_mask):
    """Compute AUC for each feature individually (univariate signal)."""
    hr("PER-FEATURE UNIVARIATE AUC")
    
    results = []
    for f in features:
        X_train = df.loc[train_mask, [f]].values
        y_train = target[train_mask]
        X_test = df.loc[test_mask, [f]].values
        y_test = target[test_mask]

        valid_train = np.all(np.isfinite(X_train), axis=1)
        valid_test = np.all(np.isfinite(X_test), axis=1)
        X_tr, y_tr = X_train[valid_train], y_train[valid_train]
        X_te, y_te = X_test[valid_test], y_test[valid_test]

        if len(X_tr) < 100 or len(X_te) < 100:
            results.append((f, 0.5, 0.5))
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = LogisticRegression(max_iter=500, solver='lbfgs')
        model.fit(X_tr_s, y_tr)
        
        tr_auc = roc_auc_score(y_tr, model.predict_proba(X_tr_s)[:, 1])
        te_auc = roc_auc_score(y_te, model.predict_proba(X_te_s)[:, 1])
        results.append((f, tr_auc, te_auc))

    # Sort by test AUC descending
    results.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n  {'Feature':<25} {'Train AUC':>12} {'Test AUC':>12} {'Signal':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
    for f, tr_auc, te_auc in results:
        # Signal classification
        if te_auc > 0.52:
            signal = "★★★"
        elif te_auc > 0.51:
            signal = "★★"
        elif te_auc > 0.505:
            signal = "★"
        else:
            signal = "—"
        print(f"  {f:<25} {tr_auc:>12.4f} {te_auc:>12.4f} {signal:>10}")
    
    return results


def run_mutual_information(df, features, target, train_mask):
    """Compute MI between each feature and the target."""
    hr("MUTUAL INFORMATION ANALYSIS")
    
    X = df.loc[train_mask, features].values
    y = target[train_mask]
    
    valid = np.all(np.isfinite(X), axis=1)
    X, y = X[valid], y[valid]
    
    # Subsample for speed if > 500k rows
    if len(X) > 500_000:
        idx = np.random.RandomState(42).choice(len(X), 500_000, replace=False)
        X, y = X[idx], y[idx]
    
    print(f"  Computing MI on {len(X):,} samples...")
    mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    
    mi_ranked = sorted(zip(features, mi_scores), key=lambda x: x[1], reverse=True)
    
    print(f"\n  {'Feature':<25} {'MI Score':>12} {'Signal':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*10}")
    for f, mi in mi_ranked:
        signal = "★★★" if mi > 0.005 else ("★★" if mi > 0.002 else ("★" if mi > 0.001 else "—"))
        print(f"  {f:<25} {mi:>12.6f} {signal:>10}")
    
    return mi_ranked


def run_random_forest(df, features, target, train_mask, test_mask):
    """Random Forest for non-linear signal detection."""
    hr("RANDOM FOREST (NON-LINEAR SIGNAL)")
    
    X_train = df.loc[train_mask, features].values
    y_train = target[train_mask]
    X_test = df.loc[test_mask, features].values
    y_test = target[test_mask]

    valid_train = np.all(np.isfinite(X_train), axis=1)
    valid_test = np.all(np.isfinite(X_test), axis=1)
    X_train, y_train = X_train[valid_train], y_train[valid_train]
    X_test, y_test = X_test[valid_test], y_test[valid_test]

    # Subsample train for speed
    if len(X_train) > 200_000:
        idx = np.random.RandomState(42).choice(len(X_train), 200_000, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
    
    print(f"  Training RF on {len(X_train):,} samples...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=100,
                                 n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    
    train_proba = rf.predict_proba(X_train)[:, 1]
    test_proba = rf.predict_proba(X_test)[:, 1]
    
    train_auc = roc_auc_score(y_train, train_proba)
    test_auc = roc_auc_score(y_test, test_proba)
    train_acc = accuracy_score(y_train, rf.predict(X_train))
    test_acc = accuracy_score(y_test, rf.predict(X_test))
    
    print(f"\n  Results:")
    print(f"    Train: {len(X_train):,} samples | Acc={train_acc:.4f} | AUC={train_auc:.4f}")
    print(f"    Test:  {len(X_test):,} samples | Acc={test_acc:.4f} | AUC={test_auc:.4f}")
    print(f"    Gap:   Acc={train_acc - test_acc:+.4f} | AUC={train_auc - test_auc:+.4f}")
    
    # Feature importance
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[-15:][::-1]
    print(f"\n  Top 15 RF Feature Importances:")
    for idx in top_idx:
        print(f"    {features[idx]:<25} importance={importances[idx]:.4f}")
    
    return {'train_auc': train_auc, 'test_auc': test_auc, 'model': rf}


def save_results(per_feature_auc, mi_results, lr_result, rf_result):
    """Save results to CSV for cross-session analysis."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Per-feature summary
    rows = []
    mi_dict = dict(mi_results)
    for f, tr_auc, te_auc in per_feature_auc:
        rows.append({
            'feature': f,
            'type': 'micro' if f in MICRO_FEATURE_COLS else 'macro',
            'train_auc': tr_auc,
            'test_auc': te_auc,
            'mi_score': mi_dict.get(f, 0.0),
        })
    
    df_results = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "feature_audit_results.csv")
    df_results.to_csv(path, index=False)
    print(f"\n  ✅ Per-feature results saved to {path}")
    
    # Summary
    summary = {
        'lr_train_auc': lr_result['train_auc'] if lr_result else None,
        'lr_test_auc': lr_result['test_auc'] if lr_result else None,
        'rf_train_auc': rf_result['train_auc'] if rf_result else None,
        'rf_test_auc': rf_result['test_auc'] if rf_result else None,
        'n_features': len(per_feature_auc),
        'n_features_auc_gt_051': sum(1 for _, _, auc in per_feature_auc if auc > 0.51),
    }
    summary_path = os.path.join(RESULTS_DIR, "audit_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"  ✅ Summary saved to {summary_path}")


if __name__ == "__main__":
    hr("INFRASTRUCTURE DIAGNOSTIC 3: FEATURE PREDICTIVENESS (V2)")
    print(f"Data: {DATA_PATH}")
    print(f"Feature lists: {len(MICRO_FEATURE_COLS)} micro + {len(MACRO_FEATURE_COLS)} macro = {len(ALL_FEATURES)} total")

    df, available_cols = load_and_prepare()
    print(f"Loaded: {len(df):,} rows  |  Available columns: {len(available_cols)}")

    features = get_available_features(df)
    print(f"Selected {len(features)}/{len(ALL_FEATURES)} features")

    target, future_ret = create_target(df, horizon=1, threshold_bps=0.0)
    print(f"Target: next-step direction (up vs down)")
    print(f"  Overall up rate: {target.mean():.3f}")

    # Create period masks
    if 'timestamp' in df.columns:
        masks = {}
        for name, (start, end) in PERIODS.items():
            masks[name] = (df['timestamp'] >= start) & (df['timestamp'] <= end)
    else:
        n = len(df)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        masks = {
            "TRAIN": pd.Series([True]*train_end + [False]*(n-train_end)),
            "VAL":   pd.Series([False]*train_end + [True]*(val_end-train_end) + [False]*(n-val_end)),
            "TEST":  pd.Series([False]*val_end + [True]*(n-val_end)),
        }

    train_mask = masks[list(masks.keys())[0]]
    val_mask = masks[list(masks.keys())[1]]
    test_mask = masks[list(masks.keys())[2]]

    # ---- Test 1: Logistic Regression Train→Val ----
    hr("TEST 1: Logistic Regression Train→Val")
    r_val = run_predictiveness_test(df, features, target, train_mask, val_mask, "Train→Val")

    # ---- Test 2: Logistic Regression Train→Test ----
    hr("TEST 2: Logistic Regression Train→Test")
    r_test = run_predictiveness_test(df, features, target, train_mask, test_mask, "Train→Test")

    # ---- Test 3: Per-Feature Univariate AUC ----
    per_feature = run_per_feature_auc(df, features, target, train_mask, test_mask)

    # ---- Test 4: Mutual Information ----
    mi_results = run_mutual_information(df, features, target, train_mask)

    # ---- Test 5: Random Forest (non-linear) ----
    rf_result = run_random_forest(df, features, target, train_mask, test_mask)

    # ---- Test 6: Multi-step horizons ----
    hr("MULTI-HORIZON PREDICTIVENESS (Logistic Regression)")
    print(f"\n  {'Horizon':>10} {'Train Acc':>12} {'Test Acc':>12} {'Test AUC':>10}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*10}")
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
            print(f"  {h:>10} {tr_acc:>12.4f} {te_acc:>12.4f} {te_auc:>10.4f}")

    # ---- Save results ----
    save_results(per_feature, mi_results, r_test, rf_result)

    hr("VERDICT")
    if r_test:
        auc = r_test['test_auc']
        if auc < 0.505:
            verdict = "🔴 COIN FLIP — Features have NO directional signal. Feature overhaul REQUIRED."
        elif auc < 0.52:
            verdict = "🟡 WEAK — Real but minimal signal. Feature improvements recommended."
        elif auc < 0.55:
            verdict = "🟢 MODERATE — Signal exists. RL agent should exploit with proper architecture."
        else:
            verdict = "🟢🟢 STRONG — Good signal level. Focus on agent tuning, not features."
        print(f"\n  Combined feature AUC (LR, test): {auc:.4f}")
        print(f"  {verdict}")

    if rf_result:
        rf_auc = rf_result['test_auc']
        lr_auc = r_test['test_auc'] if r_test else 0.5
        nl_gain = rf_auc - lr_auc
        print(f"\n  Non-linear gain (RF - LR): {nl_gain:+.4f}")
        if nl_gain > 0.01:
            print(f"  → Significant non-linear interactions exist. MLP/TCN can exploit.")
        else:
            print(f"  → Minimal non-linear gain. Features need enrichment.")

    hr("DIAGNOSTIC 3 COMPLETE")
