
import pandas as pd
import numpy as np
import torch
import os
import shutil
from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler
from finrl_pro_ds.agents.deepscalper.networks import DeepScalperNetwork

def create_dummy_parquet(path):
    dates = pd.date_range(start='2023-01-01', periods=200, freq='1min')
    data = {
        'timestamp': dates,
        'mid_price': np.linspace(100, 105, 200),
        'close': np.linspace(100, 105, 200),
        'open': np.linspace(100, 105, 200),
        'high': np.linspace(101, 106, 200),
        'low': np.linspace(99, 104, 200),
        'volume': np.random.rand(200) * 100
    }
    
    # 5 levels LOB
    for i in range(1, 6):
        data[f'bid_price_{i}'] = data['mid_price'] - i * 0.1
        data[f'ask_price_{i}'] = data['mid_price'] + i * 0.1
        data[f'bid_vol_{i}'] = np.random.rand(200) * 10
        data[f'ask_vol_{i}'] = np.random.rand(200) * 10
        
    df = pd.DataFrame(data)
    df.to_parquet(path, engine='fastparquet')
    print(f"Created dummy parquet at {path}")
    return df

def test_pipeline():
    parquet_path = "dummy_test.parquet"
    try:
        create_dummy_parquet(parquet_path)
        
        # Config
        config = {
            "symbol": "TEST",
            "window_size": 50,
            "reward": {"volatility_horizon": 10}
        }
        
        handler = ParquetDataHandler(parquet_path, "TEST", feature_config={"volatility_horizon": 10})
        env = DeepScalperEnv(config, data_handler=handler)
        
        print("Resetting Env...")
        obs, _ = env.reset()
        
        micro_shape = obs['micro'].shape
        macro_shape = obs['macro'].shape
        private_shape = obs['private'].shape
        
        print(f"Micro Shape: {micro_shape}")
        print(f"Macro Shape: {macro_shape}")
        print(f"Private Shape: {private_shape}")
        
        expected_micro = (50, 27)
        assert micro_shape == expected_micro, f"Micro shape mismatch! Expected {expected_micro}, got {micro_shape}"
        
        # Check Maco Features non-zero (scaling check)
        print(f"Macro Sample: {obs['macro'][:5]}")
        # If scaling worked, these should be > 1e-4 roughly (if price moves)
        
        # Network Check
        print("Initializing Network...")
        net_config = {
            "micro_config": {"input_size": 27, "hidden_size": 128},
            "macro_config": {"input_size": 11, "hidden_sizes": (128,)},
            "fusion_dim": 128
        }
        net = DeepScalperNetwork(**net_config)
        
        # Forward Pass
        micro_t = torch.tensor(obs['micro']).unsqueeze(0) # Batch 1
        macro_t = torch.tensor(obs['macro']).unsqueeze(0)
        private_t = torch.tensor(obs['private']).unsqueeze(0)
        
        print("Running Forward Pass...")
        q_dir, q_price, q_vol, v, vol_pred = net(micro_t, private_t, macro_t)
        
        print("Forward Pass Successful!")
        print(f"Q_dir shape: {q_dir.shape}")
        print(f"Vol Pred: {vol_pred.item()}")
        
    finally:
        if os.path.exists(parquet_path):
            os.remove(parquet_path)
            
if __name__ == "__main__":
    test_pipeline()
