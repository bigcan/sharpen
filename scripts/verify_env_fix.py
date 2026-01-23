import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import gymnasium as gym
from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv

def test_env_step_typo_fix():
    print("Testing DeepScalperEnv step() return values...")
    config = {
        "symbol": "BTCUSDT",
        "ticker": "BTCUSDT",
        "data": {"file_path": "c:/data/btc_lob_jan2023.parquet"},
        "reward": {"scaling": 1e-4}
    }
    
    # We can pass None as handler for a quick structure check if the env handles it gracefully
    # The code shows "if self.handler and step_data is None:" -> terminated = True
    # So if we pass no handler, step_data is None?
    # Let's check step():
    # if self.handler: step_data = self.handler.step()
    # else: step_data = None
    # if self.handler and step_data is None: terminated = True
    
    # If handler is None, step_data is None.
    # self._update_state(step_data) -> _build_frame(None) -> returns zeros.
    # This should run without erroring on variable names.
    
    env = DeepScalperEnv(config, data_handler=None)
    obs, info = env.reset()
    
    action = [0, 0, 0] # Hold
    try:
        obs, reward, terminated, truncated, info = env.step(action)
        print("SUCCESS: env.step() returned 5 values correctly.")
        print(f"Terminated: {terminated}, Truncated: {truncated}")
    except ValueError as e:
        print(f"FAILURE: ValueError unpacking: {e}")
    except Exception as e:
        print(f"FAILURE: Exception: {e}")

if __name__ == "__main__":
    test_env_step_typo_fix()
