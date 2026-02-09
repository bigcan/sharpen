
import yaml
import sys
import os
import torch
import numpy as np

# Add CWD to path
sys.path.append(os.getcwd())

from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler

def main():
    print("DEBUG: Starting Env Check")
    config_path = "configs/deepscalper_rtx5090_production.yaml"
    
    if not os.path.exists(config_path):
        print(f"ERROR: Config not found at {config_path}")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print("DEBUG: Config Loaded. Data Path:", config.get('data_file_path'))
    
    try:
        # 1. Init Data Handler
        print("DEBUG: Initializing ParquetHandler...")
        # Resolve data path relative to workspace or absolute if needed
        # Assuming script runs in root, data is in data/
        data_path = config.get('data_file_path', 'data/btc_lob_jan2023.parquet')
        if not os.path.exists(data_path):
             print(f"WARNING: Data path {data_path} not found. Trying absolute...")
             data_path = "/workspace/DeepScalper/data/btc_lob_jan2023.parquet"
        
        handler = ParquetDataHandler(
            file_path=data_path,
            ticker=config.get('symbol', 'BTCUSDT'),
            feature_config=config.get('feature_engineering', {}),
            start_date=config.get('train_start_date'),
            end_date=config.get('train_end_date')
        )
        print(f"DEBUG: Handler Initialized. Len={handler._len}")

        # 2. Init Env
        print("DEBUG: Initializing Env...")
        env = DeepScalperEnv(config, data_handler=handler)
        print("DEBUG: Env Initialized")
        
        # 3. Reset
        print("DEBUG: Resetting Env...")
        obs, info = env.reset()
        print(f"DEBUG: Env Reset. Obs Keys: {list(obs.keys())}")
        
        # Introspect Handler
        dh = env.handler
        print(f"DEBUG: Handler Internal State:")
        print(f"  _len: {dh._len}")
        print(f"  _ptr: {dh._ptr}")
        print(f"  start_date: {dh.start_date}")
        print(f"  end_date: {dh.end_date}")
        print(f"  timestamps[0]: {dh._timestamps[0] if dh._timestamps else 'EMPTY'}")
        print(f"  timestamps[-1]: {dh._timestamps[-1] if dh._timestamps else 'EMPTY'}")
        
        # 3. Step
        print("DEBUG: Stepping Env...")
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        print(f"DEBUG: Env Step Done. Reward: {reward}")
        
    except Exception as e:
        print(f"CRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
