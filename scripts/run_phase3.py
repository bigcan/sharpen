"""Module: run_phase3
Purpose: Execute Phase 3 Algorithm Exploration.
Iterates over PPO, SAC, TD3, DDPG, A2C agents to compare performance."""

import pandas as pd
import numpy as np
import gymnasium as gym
import os
from stable_baselines3 import PPO, SAC, TD3, DDPG, A2C
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise

from finrl_pro.data.splitter import DataSplitter
from finrl_pro.features.engineering import FeatureEngineer
from finrl_pro.envs.pro_stock_env import ProStockEnv
from finrl_pro.envs.wrappers import RiskAwareWrapper, SlippageWrapper
from finrl_pro.mlops.risk import RiskControlPolicy

def make_env(df, mode='train'):
    # Convert DF to arrays for ProStockEnv
    price_ary = df[['close']].values  # (T, 1)
    # Features: close_zscore_shifted, volume_zscore_shifted, log_return_shifted, etc.
    tech_cols = [c for c in df.columns if '_shifted' in c]
    if not tech_cols:
        # Fallback if no shifted features found (shouldn't happen with correct FE) 
        tech_cols = ['close', 'volume'] 
    
    tech_ary = df[tech_cols].values # (T, n_features)
    
    env = ProStockEnv(
        price_ary=price_ary,
        tech_ary=tech_ary,
        initial_capital=100000,
        buy_cost_pct=1e-4, # 1 bp
        sell_cost_pct=1e-4, # 1 bp
        reward_scaling=1e-4
    )
    
    # Wrappers
    if mode == 'train':
        # Apply Risk Control
        risk_policy = RiskControlPolicy(max_drawdown=0.25, max_exposure=1.0)
        env = RiskAwareWrapper(env, risk_policy)
        # Apply Slippage (1 bp)
        env = SlippageWrapper(env, slippage_bps=1.0)
        
    return env

def run_agent(agent_name, train_env, val_env, total_timesteps=30000):
    print(f"--- Training {agent_name} ---")
    
    # Stop if no improvement after 5 eval periods
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=3, verbose=0)
    eval_callback = EvalCallback(val_env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=0)
    
    if agent_name == "PPO":
        agent = PPO("MlpPolicy", train_env, verbose=0, seed=42)
    elif agent_name == "A2C":
        agent = A2C("MlpPolicy", train_env, verbose=0, seed=42)
    elif agent_name == "SAC":
        agent = SAC("MlpPolicy", train_env, verbose=0, seed=42)
    elif agent_name == "TD3":
        n_actions = train_env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        agent = TD3("MlpPolicy", train_env, action_noise=action_noise, verbose=0, seed=42)
    elif agent_name == "DDPG":
        n_actions = train_env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        agent = DDPG("MlpPolicy", train_env, action_noise=action_noise, verbose=0, seed=42)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")
        
    agent.learn(total_timesteps=total_timesteps, callback=eval_callback)
    return agent

def run_phase3():
    print("--- Starting Phase 3 Algorithm Exploration ---")
    
    # 1. Load Data & Engineer Features (Mocking for speed/demo)
    dates = pd.date_range(start='2016-01-01', end='2025-12-31', freq='D')
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
    
    splitter = DataSplitter(df_processed)
    train_df, val_df, test_df = splitter.split_by_date(
        train_start='2016-01-01', train_end='2021-12-31',
        val_start='2022-01-01', val_end='2022-12-31',
        test_start='2023-01-01', test_end='2025-12-31',
        embargo=21
    )
    
    # 2. Setup Environments
    train_env = DummyVecEnv([lambda: make_env(train_df, 'train')])
    val_env = DummyVecEnv([lambda: make_env(val_df, 'val')])
    
    # 3. Iterate over Agents
    agents_to_test = ["PPO", "A2C", "SAC", "TD3", "DDPG"]
    results = {}
    
    for agent_name in agents_to_test:
        try:
            model = run_agent(agent_name, train_env, val_env)
            
            # Simple evaluation on Validation set
            obs = val_env.reset()
            total_reward = 0
            for _ in range(len(val_df)):
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, dones, info = val_env.step(action)
                total_reward += rewards[0]
                if dones[0]:
                    break
            results[agent_name] = total_reward
            print(f"{agent_name} Val Reward: {total_reward:.4f}")
            
        except Exception as e:
            print(f"Failed to train {agent_name}: {e}")
            results[agent_name] = -np.inf

    print("\n--- Phase 3 Results ---")
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for name, reward in sorted_results:
        print(f"{name}: {reward:.4f}")
        
    print(f"\nBest Agent: {sorted_results[0][0]}")

if __name__ == "__main__":
    run_phase3()
