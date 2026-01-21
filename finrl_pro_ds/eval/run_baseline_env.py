"""Run baselines (Buy&Hold, SMA) through ProStockEnv with Phase 4 costs.

Validates the environment and cost model by running simple strategies.
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from finrl_pro_ds.data.yahoo_loader import YahooLoader
from finrl_pro_ds.envs.pro_stock_env import ProStockEnv
from finrl_pro_ds.agents.baselines import BuyAndHold, SMACrossover
from finrl_pro_ds.eval.statistics import sharpe_ratio

def max_drawdown(returns: np.ndarray) -> float:
    """Calculate Maximum Drawdown from returns sequence."""
    if len(returns) == 0:
        return 0.0
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    return float(drawdown.min())

def run_strategy(env, actions, strategy_name):
    """Run a strategy in the environment."""
    state, _ = env.reset()
    total_assets = []
    
    # actions is a DataFrame or Series aligned with the environment steps
    # Env step 0 corresponds to data index 0.
    # But env.step(action) executes action at step t, and returns state at t+1.
    # The loop usually goes from 0 to max_step.
    
    # We need to ensure actions align with the env's internal day.
    # env.day starts at 0.
    
    for i in range(env.max_step):
        # Get action for current step
        # Assuming actions is a numpy array or list aligned with env steps
        if i < len(actions):
            act = np.array([actions[i]])
        else:
            act = np.array([0.0])
            
        state, reward, done, truncated, info = env.step(act)
        total_assets.append(env.total_asset)
        
        if done:
            break
            
    return total_assets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--fee", type=float, default=1e-4, help="Transaction fee (1bp = 1e-4)")
    parser.add_argument("--slippage", type=float, default=1e-4, help="Slippage (1bp = 1e-4)")
    parser.add_argument("--out_dir", default="reports/baselines_env")
    args = parser.parse_args()

    # 1. Load Data
    loader = YahooLoader()
    df = loader.fetch(tickers=[args.ticker], start=args.start, end=args.end)
    if df.empty:
        raise RuntimeError("No data found")
    
    # Ensure sorted
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Prepare Env Arrays
    # ProStockEnv expects price_ary as (T, stock_dim)
    # We use 'close' for execution price.
    price_ary = df[["close"]].values.astype(np.float32)
    
    # Tech ary - just use close price as a dummy feature
    tech_ary = df[["close"]].values.astype(np.float32)
    
    # 2. Generate Actions
    # Buy & Hold
    bh_strategy = BuyAndHold()
    bh_actions_df = bh_strategy.predict(df)
    # BuyAndHold returns 1s. 
    # In ProStockEnv, we want to buy max at start, then hold.
    # If we pass 1 every time, it tries to buy more if cash is available. This is fine (reinvest dividends/cash).
    bh_actions = bh_actions_df["action"].values

    # SMA
    sma_strategy = SMACrossover(short_window=20, long_window=50, price_col="close")
    sma_actions_df = sma_strategy.predict(df)
    sma_actions = sma_actions_df["action"].values
    
    # 3. Initialize Env
    # Note: max_stock=1e5 to allow large positions (SPY ~500 -> 1e5 shares = 50M capacity, initial is 1M)
    env = ProStockEnv(
        price_ary=price_ary,
        tech_ary=tech_ary,
        buy_cost_pct=args.fee + args.slippage, # Total cost on entry
        sell_cost_pct=args.fee + args.slippage, # Total cost on exit
        max_stock=1e5,
        initial_capital=1e6
    )
    
    results = {}
    
    # 4. Run Buy & Hold
    print("Running Buy & Hold...")
    bh_assets = run_strategy(env, bh_actions, "Buy&Hold")
    bh_returns = pd.Series(bh_assets).pct_change().fillna(0.0).values
    
    results["BuyAndHold"] = {
        "sharpe": float(sharpe_ratio(bh_returns)),
        "max_dd": float(max_drawdown(bh_returns)),
        "final_value": float(bh_assets[-1]),
        "returns": bh_returns.tolist(),
        "assets": bh_assets
    }
    
    # 5. Run SMA
    print("Running SMA(20/50)...")
    sma_assets = run_strategy(env, sma_actions, "SMA")
    sma_returns = pd.Series(sma_assets).pct_change().fillna(0.0).values
    
    results["SMA"] = {
        "sharpe": float(sharpe_ratio(sma_returns)),
        "max_dd": float(max_drawdown(sma_returns)),
        "final_value": float(sma_assets[-1]),
        "returns": sma_returns.tolist(),
        "assets": sma_assets
    }
    
    # 6. Save Results
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    with open(out_path / "metrics.json", "w") as f:
        # Exclude full arrays from json for readability, or keep them?
        # Let's keep metrics only in json, and save csvs for curves
        metrics_only = {k: {m: v for m, v in val.items() if m not in ["returns", "assets"]} for k, val in results.items()}
        json.dump(metrics_only, f, indent=2)
        
    # Save curves
    pd.DataFrame({
        "BuyAndHold": results["BuyAndHold"]["assets"],
        "SMA": results["SMA"]["assets"]
    }).to_csv(out_path / "equity_curves.csv")
    
    print("Results:")
    print(json.dumps(metrics_only, indent=2))

if __name__ == "__main__":
    main()
