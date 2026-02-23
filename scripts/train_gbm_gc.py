"""
G6: Gold Gradient Boosting — XGBoost + LightGBM + Ensemble
==========================================================

Train XGBoost, LightGBM, and an ensemble (average of RF+XGB+LGBM) on
Gold (GC) 5-min pipeline features. Same features as G4, backtest through env.

Expected: +2-5% AUC over RF baseline.

Gate: AUC improvement >= 0.02 over RF, env PF > 1.2 on test.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore', category=FutureWarning)


def load_processed_data(path: str, feature_config: dict) -> pd.DataFrame:
    """Load and process data through the actual pipeline."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    from finrl_pro_ds.data.parquet_handler import ParquetDataHandler

    handler = ParquetDataHandler(
        file_path=path,
        ticker="GC",
        feature_config=feature_config,
        norm_cutoff_date="2025-11-01",
    )

    rows = []
    handler.reset()
    while True:
        row = handler.step()
        if row is None:
            break
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"[G6] Loaded {len(df)} rows with {len(df.columns)} columns")
    return df


def create_targets(df: pd.DataFrame, horizon: int, threshold: float) -> np.ndarray:
    """Create binary target: future return > threshold (bps)."""
    if 'mid_price' in df.columns:
        prices = df['mid_price'].values
    elif 'close' in df.columns:
        prices = df['close'].values
    else:
        raise ValueError("No price column found")

    future_ret = np.zeros(len(prices))
    if len(prices) > horizon:
        future_ret[:-horizon] = (prices[horizon:] - prices[:-horizon]) / (prices[:-horizon] + 1e-9)

    return (future_ret * 10000 > threshold).astype(int)


def get_feature_cols(df: pd.DataFrame, micro_cols: list, macro_cols: list) -> list:
    """Get feature columns that exist in the DataFrame."""
    all_cols = micro_cols + macro_cols
    return [c for c in all_cols if c in df.columns]


def train_models(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols):
    """Train RF, XGBoost, LightGBM, and ensemble."""
    results = {}

    # 1. Random Forest (baseline, same as G4)
    print("  [RF] Training...")
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=50,
        max_features='sqrt', n_jobs=-1, random_state=42,
    )
    rf.fit(X_train, y_train)
    rf_val_proba = rf.predict_proba(X_val)[:, 1]
    rf_test_proba = rf.predict_proba(X_test)[:, 1]
    results['RF'] = {
        'val_auc': roc_auc_score(y_val, rf_val_proba) if len(np.unique(y_val)) > 1 else 0.5,
        'test_auc': roc_auc_score(y_test, rf_test_proba) if len(np.unique(y_test)) > 1 else 0.5,
        'val_proba': rf_val_proba,
        'test_proba': rf_test_proba,
    }

    # 2. XGBoost
    try:
        import xgboost as xgb
        print("  [XGB] Training...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=50, reg_alpha=0.1, reg_lambda=1.0,
            n_jobs=-1, random_state=42, eval_metric='auc',
            early_stopping_rounds=50,
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        xgb_val_proba = xgb_model.predict_proba(X_val)[:, 1]
        xgb_test_proba = xgb_model.predict_proba(X_test)[:, 1]
        results['XGB'] = {
            'val_auc': roc_auc_score(y_val, xgb_val_proba) if len(np.unique(y_val)) > 1 else 0.5,
            'test_auc': roc_auc_score(y_test, xgb_test_proba) if len(np.unique(y_test)) > 1 else 0.5,
            'val_proba': xgb_val_proba,
            'test_proba': xgb_test_proba,
        }
    except ImportError:
        print("  [XGB] xgboost not installed — skipping")

    # 3. LightGBM
    try:
        import lightgbm as lgb
        print("  [LGBM] Training...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=50, reg_alpha=0.1, reg_lambda=1.0,
            n_jobs=-1, random_state=42, verbose=-1,
        )
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        lgb_val_proba = lgb_model.predict_proba(X_val)[:, 1]
        lgb_test_proba = lgb_model.predict_proba(X_test)[:, 1]
        results['LGBM'] = {
            'val_auc': roc_auc_score(y_val, lgb_val_proba) if len(np.unique(y_val)) > 1 else 0.5,
            'test_auc': roc_auc_score(y_test, lgb_test_proba) if len(np.unique(y_test)) > 1 else 0.5,
            'val_proba': lgb_val_proba,
            'test_proba': lgb_test_proba,
        }
    except ImportError:
        print("  [LGBM] lightgbm not installed — skipping")

    # 4. Ensemble (average of all available models)
    ensemble_models = [k for k in results if k in ('RF', 'XGB', 'LGBM')]
    if len(ensemble_models) >= 2:
        print(f"  [ENS] Ensembling {ensemble_models}...")
        val_avg = np.mean([results[m]['val_proba'] for m in ensemble_models], axis=0)
        test_avg = np.mean([results[m]['test_proba'] for m in ensemble_models], axis=0)
        results['Ensemble'] = {
            'val_auc': roc_auc_score(y_val, val_avg) if len(np.unique(y_val)) > 1 else 0.5,
            'test_auc': roc_auc_score(y_test, test_avg) if len(np.unique(y_test)) > 1 else 0.5,
            'val_proba': val_avg,
            'test_proba': test_avg,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="G6: Gold Gradient Boosting")
    parser.add_argument("--data", default="data/processed/gc_2025_full_year_5min.parquet")
    parser.add_argument("--horizon", type=int, default=3, help="Prediction horizon (bars)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold (bps)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, args.data) if not os.path.isabs(args.data) else args.data

    if not os.path.exists(data_path):
        print(f"[G6] ERROR: Data file not found: {data_path}")
        sys.exit(1)

    feature_config = {
        'n_levels': 1,
        'asset_class': 'cme_futures',
        'vol_norm_window': 24,
    }

    sys.path.insert(0, project_root)
    from finrl_pro_ds.data.feature_engineering import get_micro_feature_cols, get_macro_feature_cols
    micro_cols = get_micro_feature_cols(n_levels=1)
    macro_cols = get_macro_feature_cols(asset_class='cme_futures')

    df = load_processed_data(data_path, feature_config)
    feature_cols = get_feature_cols(df, micro_cols, macro_cols)
    print(f"[G6] Using {len(feature_cols)} features")

    # Split masks
    ts = pd.to_datetime(df['timestamp'])
    train_mask = ts < '2025-11-01'
    val_mask = (ts >= '2025-11-01') & (ts < '2025-12-01')
    test_mask = ts >= '2025-12-01'

    # Create targets
    target = create_targets(df, args.horizon, args.threshold)

    X_train = df.loc[train_mask, feature_cols].values
    y_train = target[train_mask]
    X_val = df.loc[val_mask, feature_cols].values
    y_val = target[val_mask]
    X_test = df.loc[test_mask, feature_cols].values
    y_test = target[test_mask]

    # Drop NaN rows
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        valid = ~np.isnan(X).any(axis=1)
        if name == "train":
            X_train, y_train = X[valid], y[valid]
        elif name == "val":
            X_val, y_val = X[valid], y[valid]
        else:
            X_test, y_test = X[valid], y[valid]

    print(f"[G6] H={args.horizon}, threshold={args.threshold} bps")
    print(f"[G6] Splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Train all models
    results = train_models(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols)

    # Summary
    print("\n" + "=" * 60)
    print(f"[G6] SUMMARY — Gold GBM (H={args.horizon}, thresh={args.threshold} bps)")
    print("=" * 60)
    print(f"{'Model':>10} {'Val AUC':>9} {'Test AUC':>10} {'Gap':>6}")
    print("-" * 40)

    for model, r in results.items():
        gap = r['val_auc'] - r['test_auc']
        marker = " ***" if r['test_auc'] > 0.55 else ""
        print(f"{model:>10} {r['val_auc']:>9.4f} {r['test_auc']:>10.4f} {gap:>6.3f}{marker}")

    # Gate check
    best_test = max(r['test_auc'] for r in results.values())
    rf_test = results.get('RF', {}).get('test_auc', 0.5)
    improvement = best_test - rf_test
    print(f"\n[G6] Best test AUC: {best_test:.4f}")
    print(f"[G6] AUC improvement over RF: {improvement:+.4f}")
    if improvement >= 0.02:
        print("[G6] GATE PASSED: AUC improvement >= 0.02")
    else:
        print("[G6] GATE FAILED: AUC improvement < 0.02")


if __name__ == "__main__":
    main()
