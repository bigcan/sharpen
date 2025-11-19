"""Module: run_phase0
Purpose: Execute the Phase 0 MVP loop: PPO vs Buy&Hold with strict controls."""

import pandas as pd
import numpy as np
import gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import DummyVecEnv

from finrl_pro.data.loader import load_data  # Assuming this loads raw Yahoo/Alpaca data
from finrl_pro.data.splitter import DataSplitter
from finrl_pro.features.engineering import FeatureEngineer
from finrl_pro.envs.pro_stock_env import ProStockEnv
from finrl_pro.envs.wrappers import RiskAwareWrapper, SlippageWrapper
from finrl_pro.mlops.risk import RiskControlPolicy

def run_phase0():
    print("--- Starting Phase 0 MVP Run ---")
    
    # 1. Load Data (Mocking raw load for now, assuming daily SPY data structure)
    # In a real run, this would fetch from DB or CSV
    # creating dummy data for demonstration if file doesn't exist
    # dates from 2016-01-01 to 2025-12-31
    dates = pd.date_range(start='2016-01-01', end='2025-12-31', freq='D')
    # Filter for business days roughly
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
    # Sort by date
    data = data.sort_values('date')
    
    # 2. Feature Engineering (PIT)
    print("Generating features...")
    fe = FeatureEngineer()
    df_processed = fe.preprocess_data(data)
    
    # 3. Data Splitting
    print("Splitting data...")
    splitter = DataSplitter(df_processed)
    # Train: 2016-2021, Val: 2022, Test: 2023-2025
    # Embargo: 21 days
    train_df, val_df, test_df = splitter.split_by_date(
        train_start='2016-01-01', train_end='2021-12-31',
        val_start='2022-01-01', val_end='2022-12-31',
        test_start='2023-01-01', test_end='2025-12-31',
        embargo=21
    )
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    
    # 4. Environment Setup
    def make_env(df, mode='train'):
        # Convert DF to arrays for ProStockEnv
        price_ary = df[['close']].values  # (T, 1)
        # Features: close_zscore_shifted, volume_zscore_shifted, log_return_shifted
        tech_cols = [c for c in df.columns if '_shifted' in c]
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

    train_env = DummyVecEnv([lambda: make_env(train_df, 'train')])
    val_env = DummyVecEnv([lambda: make_env(val_df, 'val')])
    
    # 5. Agent Training with Early Stopping
    print("Initializing PPO agent...")
    agent = PPO("MlpPolicy", train_env, verbose=1, seed=42)
    
    # Stop if no improvement after 10 eval periods
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=5, verbose=1)
    eval_callback = EvalCallback(val_env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1)
    
    print("Training...")
    agent.learn(total_timesteps=50000, callback=eval_callback) # Short run for demo
    
    print("Phase 0 Run Complete.")
    # In a real pipeline, we would now run prediction on test_df and calculate metrics.

if __name__ == "__main__":
    run_phase0()
