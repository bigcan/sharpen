
import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import logging
from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler

logging.basicConfig(level=logging.INFO)

def main():
    # Mock Data Handler logic or use real data
    # Let's use real data if available, else mock
    file_path = "c:/data/btc_lob_jan2023.parquet"
    if not os.path.exists(file_path):
        print("Data not found, using mock logic via Mock Data Handler not implemented, using real config")
        return

    config = {
        "symbol": "BTCUSDT",
        "data": {"file_path": file_path, "ticker": "BTCUSDT"},
        "reward": {"scaling": 1e-4, "risk_penalty": 0.05}
    }
    
    handler = ParquetDataHandler(file_path=file_path, ticker="BTCUSDT")
    env = DeepScalperEnv(config=config, data_handler=handler)
    
    obs, _ = env.reset()
    print("Env Reset.")
    
    # Step 10 times
    for i in range(10):
        action = np.array([1, 0, 0]) # Buy, Price 0, Vol 0
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: Reward={reward}, Term={terminated}, Info={info}")
        if terminated: break

if __name__ == "__main__":
    main()
