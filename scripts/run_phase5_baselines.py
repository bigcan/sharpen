
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from finrl_pro.data.loader_pro import ProFeatureAssembler

SNAPSHOT_ID = "36629e7c-ff6a-4585-9fcf-284058413028"

LIQUID_20 = [
    'SPY', 'QQQ', 'IWM',
    'XLK', 'XLF', 'XLE', 'XLV',
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'JPM', 'XOM', 'JNJ', 'PG', 'V',
    'GLD', 'TLT'
]

def calculate_metrics(daily_returns):
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    
    cum_returns = (1 + daily_returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_dd = drawdown.min()
    
    return {
        'Sharpe': sharpe,
        'MaxDD': max_dd,
        'Cumulative': cum_returns.iloc[-1] - 1
    }

def run_baselines():
    print(f"Loading Snapshot {SNAPSHOT_ID}...")
    assembler = ProFeatureAssembler()
    # Bypass cache again to be safe, we just want prices
    features_cfg = {
        "stockstats_overrides": ["close"],
        "dataset_hash_source": "baselines_v1"
    }
    
    assembly = assembler.assemble_from_snapshot(
        snapshot_id=SNAPSHOT_ID,
        features_cfg=features_cfg
    )
    
    df_price = pd.DataFrame(assembly.price_ary, index=assembly.dates, columns=assembly.tickers)
    
    # Filter for Liquid 20
    df = df_price[LIQUID_20].copy()
    print(f"Universe filtered to {len(df.columns)} assets: {df.columns.tolist()}")
    
    # Split (Test Set only for Report)
    test_start = '2021-01-01'
    df_test = df[df.index >= test_start].copy()
    
    print(f"\nRunning Baselines on Test Set ({test_start} to {df.index[-1].date()})...")
    
    # 1. Equal Weight (Daily Rebalancing)
    # Return = Mean of asset returns
    returns = df_test.pct_change().fillna(0)
    ew_returns = returns.mean(axis=1)
    
    # 2. Risk Parity (Inverse Vol)
    # Calculate rolling volatility (60 day lookback) on the full dataset, then slice
    full_returns = df.pct_change().fillna(0)
    vol = full_returns.rolling(window=60).std()
    inv_vol = 1 / vol
    
    # Normalize to sum to 1
    rp_weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    
    # Align with Test Set
    rp_weights_test = rp_weights[rp_weights.index >= test_start]
    # Shift weights by 1 day (weights calculated at close t applied to return t+1)
    rp_weights_test = rp_weights_test.shift(1).fillna(0)
    
    # Portfolio Return = sum(weights * returns)
    # We need to match columns
    rp_returns = (rp_weights_test * returns).sum(axis=1)
    
    # 3. SPY Buy & Hold
    spy_returns = df_price['SPY'].pct_change().fillna(0)
    spy_returns_test = spy_returns[spy_returns.index >= test_start]

    # Metrics
    metrics = {
        'Equal Weight': calculate_metrics(ew_returns),
        'Risk Parity': calculate_metrics(rp_returns),
        'SPY Benchmark': calculate_metrics(spy_returns_test)
    }
    
    print("\n--- Baseline Performance (Test Set) ---")
    res_df = pd.DataFrame(metrics).T
    print(res_df)
    
    # Export for Experiment Comparison
    res_df.to_csv('data/phase5_baselines.csv')
    print("\nBaseline metrics saved to data/phase5_baselines.csv")
    
    # Plot Equity Curves
    plt.figure(figsize=(12, 6))
    (1 + ew_returns).cumprod().plot(label='Equal Weight')
    (1 + rp_returns).cumprod().plot(label='Risk Parity')
    (1 + spy_returns_test).cumprod().plot(label='SPY', color='black', linestyle='--')
    
    plt.title('Phase 5 Baselines (2021-2025)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figs/phase5/baselines.png')
    print("Plot saved to figs/phase5/baselines.png")

if __name__ == "__main__":
    run_baselines()
