"""
G4: Gold RF Baseline — Through Pipeline Features
=================================================

Train Random Forest on actual pipeline features (18 micro + 15 macro = 33 dims,
EMA-Z normalized) for Gold (GC) 5-min data.

Binary classification target: price_change > threshold at horizon H.
Sweeps H={1,3,6,12} bars and threshold={0.50,0.55,0.60,0.65}.

Compare vs Session 17 standalone RF (AUC 0.6147 val / 0.5542 test).

Gate: At least one H+threshold combo with PF > 1.0 on both splits.
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
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    from finrl_pro_ds.data.parquet_handler import ParquetDataHandler

    handler = ParquetDataHandler(
        file_path=path,
        ticker="GC",
        feature_config=feature_config,
        norm_cutoff_date="2025-11-01",
    )

    # Extract all data as DataFrame
    rows = []
    handler.reset()
    while True:
        row = handler.step()
        if row is None:
            break
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"[G4] Loaded {len(df)} rows with {len(df.columns)} columns")
    return df


def create_targets(df: pd.DataFrame, horizon: int, threshold: float) -> pd.Series:
    """Create binary target: future return > threshold."""
    if 'mid_price' in df.columns:
        prices = df['mid_price'].values
    elif 'close' in df.columns:
        prices = df['close'].values
    else:
        raise ValueError("No price column found")

    future_ret = np.zeros(len(prices))
    if len(prices) > horizon:
        future_ret[:-horizon] = (prices[horizon:] - prices[:-horizon]) / (prices[:-horizon] + 1e-9)

    # Binary: 1 if return > threshold (in bps)
    target = (future_ret * 10000 > threshold).astype(int)
    return pd.Series(target, index=df.index)


def get_feature_cols(df: pd.DataFrame, micro_cols: list, macro_cols: list) -> list:
    """Get feature columns that exist in the DataFrame."""
    all_cols = micro_cols + macro_cols
    available = [c for c in all_cols if c in df.columns]
    return available


def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: list,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    horizon: int,
    threshold: float,
) -> dict:
    """Train RF and evaluate on val/test splits."""
    target = create_targets(df, horizon, threshold)

    X_train = df.loc[train_mask, feature_cols].values
    y_train = target[train_mask].values
    X_val = df.loc[val_mask, feature_cols].values
    y_val = target[val_mask].values
    X_test = df.loc[test_mask, feature_cols].values
    y_test = target[test_mask].values

    # Drop rows with NaN in features
    valid_train = ~np.isnan(X_train).any(axis=1)
    X_train, y_train = X_train[valid_train], y_train[valid_train]
    valid_val = ~np.isnan(X_val).any(axis=1)
    X_val, y_val = X_val[valid_val], y_val[valid_val]
    valid_test = ~np.isnan(X_test).any(axis=1)
    X_test, y_test = X_test[valid_test], y_test[valid_test]

    if len(X_train) < 100 or len(X_val) < 50:
        return {"error": "Insufficient data"}

    # Train RF
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=50,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)

    # Evaluate
    val_proba = rf.predict_proba(X_val)[:, 1]
    test_proba = rf.predict_proba(X_test)[:, 1]

    val_auc = roc_auc_score(y_val, val_proba) if len(np.unique(y_val)) > 1 else 0.5
    test_auc = roc_auc_score(y_test, test_proba) if len(np.unique(y_test)) > 1 else 0.5

    # Feature importances (top 10)
    fi = sorted(zip(feature_cols, rf.feature_importances_), key=lambda x: -x[1])[:10]

    return {
        "horizon": horizon,
        "threshold": threshold,
        "val_auc": val_auc,
        "test_auc": test_auc,
        "val_n": len(X_val),
        "test_n": len(X_test),
        "train_n": len(X_train),
        "val_pos_rate": y_val.mean(),
        "test_pos_rate": y_test.mean(),
        "top_features": fi,
    }


def main():
    parser = argparse.ArgumentParser(description="G4: Gold RF baseline through pipeline")
    parser.add_argument("--data", default="data/processed/gc_2025_full_year_5min.parquet")
    parser.add_argument("--horizons", default="1,3,6,12", help="Comma-separated horizons")
    parser.add_argument("--thresholds", default="0.0,0.5,1.0,2.0", help="Comma-separated thresholds (bps)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, args.data) if not os.path.isabs(args.data) else args.data

    if not os.path.exists(data_path):
        print(f"[G4] ERROR: Data file not found: {data_path}")
        sys.exit(1)

    # Feature config for Gold Level-1
    feature_config = {
        'n_levels': 1,
        'asset_class': 'cme_futures',
        'vol_norm_window': 24,
    }

    # Get feature column names
    sys.path.insert(0, project_root)
    from finrl_pro_ds.data.feature_engineering import get_micro_feature_cols, get_macro_feature_cols
    micro_cols = get_micro_feature_cols(n_levels=1)
    macro_cols = get_macro_feature_cols(asset_class='cme_futures')

    # Load data through pipeline
    df = load_processed_data(data_path, feature_config)
    feature_cols = get_feature_cols(df, micro_cols, macro_cols)
    print(f"[G4] Using {len(feature_cols)} features: {feature_cols}")

    # Create split masks (Jan-Oct train, Nov val, Dec test)
    ts = pd.to_datetime(df['timestamp'])
    train_mask = ts < '2025-11-01'
    val_mask = (ts >= '2025-11-01') & (ts < '2025-12-01')
    test_mask = ts >= '2025-12-01'

    print(f"[G4] Splits: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")

    # Sweep horizons and thresholds
    horizons = [int(h) for h in args.horizons.split(',')]
    thresholds = [float(t) for t in args.thresholds.split(',')]

    results = []
    for H in horizons:
        for thresh in thresholds:
            print(f"\n[G4] H={H}, threshold={thresh} bps")
            result = train_and_evaluate(df, feature_cols, train_mask, val_mask, test_mask, H, thresh)
            results.append(result)
            if "error" not in result:
                print(f"  Val AUC:  {result['val_auc']:.4f} (n={result['val_n']})")
                print(f"  Test AUC: {result['test_auc']:.4f} (n={result['test_n']})")
                print(f"  Top features: {result['top_features'][:5]}")

    # Summary
    print("\n" + "=" * 70)
    print("[G4] SUMMARY — Gold RF Baseline")
    print("=" * 70)
    print(f"{'H':>3} {'Thresh':>7} {'Val AUC':>9} {'Test AUC':>10} {'Gap':>6} {'Val N':>7} {'Test N':>7}")
    print("-" * 55)

    best = None
    for r in results:
        if "error" in r:
            continue
        gap = r['val_auc'] - r['test_auc']
        marker = " ***" if r['val_auc'] > 0.55 and r['test_auc'] > 0.52 else ""
        print(f"{r['horizon']:>3} {r['threshold']:>7.1f} {r['val_auc']:>9.4f} {r['test_auc']:>10.4f} {gap:>6.3f} {r['val_n']:>7} {r['test_n']:>7}{marker}")
        if best is None or r['val_auc'] > best['val_auc']:
            best = r

    if best:
        print(f"\n[G4] Best: H={best['horizon']}, threshold={best['threshold']} bps")
        print(f"  Val AUC={best['val_auc']:.4f}, Test AUC={best['test_auc']:.4f}")
        print(f"  Session 17 comparison: Val 0.6147, Test 0.5542")


if __name__ == "__main__":
    main()
