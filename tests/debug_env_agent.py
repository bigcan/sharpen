"""Module: debug_env_agent
Purpose: Minimal script to debug ProStockEnv and SB3 agent interaction."""

import pandas as pd
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from finrl_pro.features.engineering import FeatureEngineer
from finrl_pro.envs.pro_stock_env import ProStockEnv
# Assuming wrappers are commented out in run_phase3.py, so not using them here either for isolation

def make_debug_env():
    # Create dummy data with a longer range
    dates = pd.date_range(start='2015-01-01', end='2017-12-31', freq='D')
    dates = dates[dates.dayofweek < 5] 
    data = pd.DataFrame({
        'date': dates,
        'tic': 'SPY',
        'open': np.random.uniform(100, 400, size=len(dates)),
        'high': np.random.uniform(100, 400, size=len(dates)),
        'low': np.random.uniform(100, 400, size=len(dates)),
        'close': np.random.uniform(100, 400, size=len(dates)),
        'volume': np.random.uniform(1000000, 5000000, size=len(dates))
    })
    data = data.sort_values('date')
    
    fe = FeatureEngineer()
    df_processed = fe.preprocess_data(data)
    print(f"DEBUG: df_processed length after FE: {len(df_processed)}")
    
    if df_processed.empty:
        raise ValueError("df_processed is empty after feature engineering. Adjust dummy data range.")
    
    price_ary = df_processed[['close']].values
    tech_cols = [c for c in df_processed.columns if '_shifted' in c]
    if not tech_cols:
        tech_cols = ['close', 'volume']
    tech_ary = df_processed[tech_cols].values
    
    env = ProStockEnv(
        price_ary=price_ary,
        tech_ary=tech_ary,
        initial_capital=100000,
        buy_cost_pct=1e-4,
        sell_cost_pct=1e-4,
        reward_scaling=1e-4
    )
    return env

if __name__ == "__main__":
    print("--- Debugging ProStockEnv and SB3 Agent Interaction ---")
    
    # 1. Initialize environment
    vec_env = DummyVecEnv([make_debug_env])
    
    # 2. Initialize agent
    model = PPO("MlpPolicy", vec_env, verbose=0, device='cpu', seed=42)
    
    # 3. Reset environment
    obs = vec_env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Observation space: {vec_env.observation_space}")
    print(f"Action space: {vec_env.action_space}")
    
    # 4. Call predict and inspect output
    print("\n--- Testing model.predict() ---")
    try:
        predict_output = model.predict(obs, deterministic=True)
        print(f"Type of predict output: {type(predict_output)}")
        print(f"Length of predict output: {len(predict_output)}")
        
        action, _states = predict_output
        print(f"Action type: {type(action)}, shape: {action.shape}")
        
        # 5. Test env.step()
        print("\n--- Testing env.step() ---")
        next_obs, reward, done, info = vec_env.step(action)
        print(f"Next observation shape: {next_obs.shape}, Reward: {reward}, Done: {done}")
        
        print("\n--- Debugging successful: predict and step calls did not raise exceptions ---")
        
    except Exception as e:
        print(f"An error occurred during predict/step: {e}")
