import pandas as pd
import numpy as np
from finrl_pro.envs.pro_stock_env import ProStockEnv

def trace_episode():
    csv_path = "tmp/features_to_validate.csv"
    df = pd.read_csv(csv_path)
    
    # Prepare data
    # Assuming single asset
    price_ary = df[["close"]].values
    
    # Use 'ma20_lag' and 'fd_close_d0p5_w256' as tech
    tech_cols = ["ma20_lag", "fd_close_d0p5_w256"]
    # Fill NaNs for env execution (env expects valid float32)
    df[tech_cols] = df[tech_cols].fillna(0.0)
    tech_ary = df[tech_cols].values
    
    env = ProStockEnv(
        price_ary=price_ary,
        tech_ary=tech_ary,
        buy_cost_pct=1e-3,
        sell_cost_pct=1e-3,
        initial_capital=100000.0
    )
    
    obs, _ = env.reset()
    print("Step | Action | Stock (Hold) | Price | Amount (Cash) | Asset Value | Reward")
    print("-" * 80)
    
    for i in range(50):
        # Simple deterministic action: alternate buy/sell
        if i % 10 < 5:
            action = [1.0] # Buy max
        else:
            action = [-1.0] # Sell max
            
        obs, reward, done, truncated, info = env.step(action)
        
        price = env.price_ary[env.day]
        stocks = env.stocks[0]
        amount = env.amount
        asset = env.total_asset
        
        print(f"{i:4d} | {action[0]:6.2f} | {stocks:12.2f} | {price[0]:8.2f} | {amount:12.2f} | {asset:12.2f} | {reward:8.4f}")
        
        if done:
            break

if __name__ == "__main__":
    trace_episode()
