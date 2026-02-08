
import yaml
import sys
import os
import torch
import numpy as np

# Add CWD to path
sys.path.append(os.getcwd())

from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv

def main():
    print("DEBUG: Starting Env Check")
    config_path = "configs/deepscalper_rtx5090_production.yaml"
    
    if not os.path.exists(config_path):
        print(f"ERROR: Config not found at {config_path}")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print("DEBUG: Config Loaded")
    
    try:
        # 1. Init
        print("DEBUG: Initializing Env...")
        env = DeepScalperEnv(config)
        print("DEBUG: Env Initialized")
        
        # 2. Reset
        print("DEBUG: Resetting Env...")
        obs, info = env.reset()
        print("DEBUG: Env Reset. Obs Keys:", obs.keys())
        
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
