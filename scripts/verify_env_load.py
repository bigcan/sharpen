"""
Verify Environment Load
Smoke test to check if DeepScalperEnv can load the generated dataset.
"""
from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler
import pandas as pd
import logging
import time

def test_env():
    print("Testing data load...")
    file_path = "c:/data/btc_lob_jan2023.parquet"
    
    # 1. Check Data
    df = pd.read_parquet(file_path)
    print(f"Data Shape: {df.shape}")
    
    # 2. Config
    config = {
        'ticker': 'BTCUSDT',
        'tech_indicator_list': ['rsi_14', 'MACD_12_26_9', 'atr_14'], # Just a subset to check
        'initial_amount': 100000,
        'reward_type': 'pnl',
        'transaction_cost_pct': 0.0004
    }
    
    # 3. Init Handler
    # Note: DeepScalperEnv might expect a specific handler class or path
    # In my codebase, DeepScalperEnv takes data_handler object.
    
    handler = ParquetDataHandler(file_path, ticker='BTCUSDT')
    print("Handler initialized.")
    
    # 4. Init Env
    env = DeepScalperEnv(config=config, data_handler=handler)
    print("Environment initialized.")
    
    # 5. Reset & Step
    obs, info = env.reset()
    print("Reset successful. Obs shape:", {k: v.shape for k, v in obs.items()})
    
    # Step loop
    start = time.time()
    for i in range(100):
        action = [0, 0, 0] # Hold, Price Level 0, 100% (or similar, depending on env)
        obs, reward, terminated, truncated, info = env.step(action)
        if i % 20 == 0:
            print(f"Step {i}: Reward={reward:.4f}, Portfolio={info['portfolio_value']:.2f}")
        if terminated or truncated:
            break
            
    print(f"100 steps took {time.time() - start:.2f}s")
    print("Test Passed!")

if __name__ == "__main__":
    test_env()
