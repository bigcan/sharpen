
import yaml
import os
import sys
import gymnasium as gym
import numpy as np
import time
import functools
import traceback
import faulthandler

# Enable faulthandler to catch segfaults
faulthandler.enable()

# Add CWD to path
sys.path.append(os.getcwd())

from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler

def make_env(config, start_date=None, end_date=None):
    print(f"[Worker {os.getpid()}] make_env started.")
    try:
        data_config = config.get("data", {})
        file_path = data_config.get("file_path", "data/btc_lob_jan2023.parquet")
        
        # Absolute path fallback
        if not os.path.exists(file_path):
            file_path = "/workspace/DeepScalper/data/btc_lob_jan2023.parquet"
            
        handler = ParquetDataHandler(
            file_path=file_path,
            ticker="BTCUSDT",
            feature_config=config.get("features", {}),
            start_date=start_date or data_config.get("train_start_date"),
            end_date=end_date or data_config.get("train_end_date")
        )
        print(f"[Worker {os.getpid()}] ParquetHandler loaded {handler._len} rows.")
        
        env = DeepScalperEnv(config=config.get("env", {}), data_handler=handler)
        print(f"[Worker {os.getpid()}] Env created.")
        return env
        
    except Exception as e:
        print(f"[Worker {os.getpid()}] CRITICAL INIT FAILURE: {e}")
        traceback.print_exc()
        raise e

def main():
    print("DEBUG: Starting Multiprocessing Check (Single Worker)")
    config_path = "configs/deepscalper_rtx5090_production.yaml"
    
    if not os.path.exists(config_path):
        # absolute fallback
        config_path = "/workspace/DeepScalper/configs/deepscalper_rtx5090_production.yaml"
        
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Force params to match production but reduce num_envs
    num_envs = 12 # RETRY 12
    use_shm = False # Forced false
    
    print(f"DEBUG: Creating AsyncVectorEnv with num_envs={num_envs}, context='spawn', shared_memory={use_shm}")
    
    env_factory = functools.partial(make_env, config=config)
    
    try:
        vec_env = gym.vector.AsyncVectorEnv(
            [env_factory for _ in range(num_envs)],
            context="spawn",
            shared_memory=use_shm
        )
        
        print("DEBUG: VectorEnv Created. Resetting...")
        t0 = time.time()
        obs, info = vec_env.reset()
        print(f"DEBUG: VectorEnv Reset Done in {time.time() - t0:.2f}s")
        print(f"DEBUG: Obs Keys: {list(obs.keys())}")
        
        print("DEBUG: Stepping...")
        actions = vec_env.action_space.sample()
        obs, reward, term, trunc, info = vec_env.step(actions)
        print("DEBUG: Step Done.")
        
        vec_env.close()
        print("DEBUG: Success.")
        
    except Exception as e:
        print(f"DEBUG: MAIN PROCESS CRASHED: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
