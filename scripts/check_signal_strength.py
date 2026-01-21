import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from finrl_pro_ds.data.loader_pro import ProFeatureAssembler

def main():
    print("Loading data...")
    df = pd.read_parquet("data/sp500_daily_2016_2025.parquet")
    df = df.rename(columns={'timestamp': 'date', 'ticker': 'tic'})
    
    # Configuration for "Combo" mode
    features_cfg = {
        "families": {
            "trend": False,
            "momentum": False,
            "vol": True,
            "volume": False
        },
        "advanced": {
            "fracdiff": {"enable": False},
            "wavelet": {
                "enable": True,
                "wavelet": "db4",
                "level": 2,
                "window": 256,
                "cols": ["close"],
                "denoise": True
            }
        },
        "stockstats_overrides": [] # Let assembler resolve defaults or families
    }
    
    print("Generating features...")
    assembler = ProFeatureAssembler()
    # We need a dummy dataset_hash
    assembly = assembler.assemble_from_df(
        df=df,
        features_cfg=features_cfg,
        dataset_hash="file://data/sp500_daily_2016_2025.parquet"
    )
    
    # Extract Features and Prices
    # tech_ary shape: (Time, Ticker * Features)
    # Since we have only 1 ticker (SPY), shape is (Time, Features)
    X = assembly.tech_ary
    prices = assembly.price_ary # Shape (Time, 1)
    
    # Compute Target: Next Day Return
    # returns[t] = ln(P[t+1] / P[t])
    # We shift returns backward by 1 to align with features at t
    log_prices = np.log(prices)
    returns = np.diff(log_prices, axis=0) # length T-1
    
    # X needs to be trimmed to length T-1
    X = X[:-1]
    y = returns.flatten()
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature names (approx): {assembly.feature_list}")

    # Split Train/Test (Simple time split)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training on {len(X_train)} samples...")
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("Evaluating...")
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)
    
    mse_train = mean_squared_error(y_train, preds_train)
    mse_test = mean_squared_error(y_test, preds_test)
    
    ic_train, _ = pearsonr(y_train, preds_train)
    ic_test, _ = pearsonr(y_test, preds_test)
    
    print("\nRESULTS")
    print("=======")
    print(f"Train MSE: {mse_train:.6f} | IC: {ic_train:.4f}")
    print(f"Test  MSE: {mse_test:.6f} | IC: {ic_test:.4f}")
    
    if ic_test < 0.02:
        print("\nCONCLUSION: Signal is WEAK or NON-EXISTENT. (IC < 0.02)")
    else:
        print("\nCONCLUSION: Signal DETECTED. (IC >= 0.02)")

if __name__ == "__main__":
    main()
