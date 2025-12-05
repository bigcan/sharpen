import sys
import os
import pandas as pd
import numpy as np
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finrl_pro.data.feature_factory import MarketingFeatureFactory
from finrl_pro.envs.pro_stock_env import ProStockEnv

# ---------------------------------------------------------
# Data Loading & Prep
# ---------------------------------------------------------
def load_and_process_data(ticker='AAPL'):
    """Loads data and applies Feature Factory (mRMR selection)."""
    file_path = f"data/sp500_multi_2016_2025/{ticker}.parquet"
    if not os.path.exists(file_path):
        raise FileNotFoundError("Data not found.")
    
    df = pd.read_parquet(file_path)
    if 'timestamp' in df.columns: df = df.rename(columns={'timestamp': 'date'})
    df.columns = [c.lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    
    # Add tic
    if 'tic' not in df.columns: df['tic'] = ticker
    
    # 1. Feature Factory
    print("Generating features...")
    factory = MarketingFeatureFactory(windows=[14, 30])
    df_transformed = factory.transform(df)
    df_transformed['date'] = df['date'] # Restore date
    
    # 2. Selection
    print("Selecting features...")
    stationary = factory.check_stationarity(df_transformed)
    selected = factory.select_features_mrmr(df_transformed, k=20)
    features = [f for f in selected if f in stationary]
    if not features: features = stationary[:20]
    
    print(f"Selected {len(features)} features: {features}")
    return df_transformed, features

# ---------------------------------------------------------
# Objective Function
# ---------------------------------------------------------
def objective(trial):
    # 1. Data Split (Train 2016-2018, Val 2019)
    # Note: We load data globally to avoid re-processing every trial
    global GLOBAL_DF, GLOBAL_FEATURES
    
    train_df = GLOBAL_DF[(GLOBAL_DF['date'] >= '2016-01-01') & (GLOBAL_DF['date'] < '2019-01-01')]
    val_df = GLOBAL_DF[(GLOBAL_DF['date'] >= '2019-01-01') & (GLOBAL_DF['date'] < '2020-01-01')]
    
    # 2. Suggest Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    gamma = trial.suggest_float("gamma", 0.90, 0.9999)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    
    # 3. Setup Env
    def make_env(df):
        price_ary = df[['close']].values.astype(np.float32)
        tech_ary = df[GLOBAL_FEATURES].values.astype(np.float32)
        return ProStockEnv(price_ary=price_ary, tech_ary=tech_ary, 
                           initial_capital=100000, reward_scaling=1e-4, gamma=gamma)
    
    train_env = make_env(train_df)
    val_env = make_env(val_df)
    
    # 4. Train Agent
    try:
        model = PPO(
            "MlpPolicy", 
            train_env, 
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_steps=n_steps,
            gamma=gamma,
            ent_coef=ent_coef,
            clip_range=clip_range,
            verbose=0,
            seed=42
        )
        model.learn(total_timesteps=15000) # Fast training
        
        # 5. Evaluate
        obs, _ = val_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, _, done, _, _ = val_env.step(action)
            
        final_val = val_env.total_asset
        
        # Metric: Total Return on Val
        return final_val
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0

if __name__ == "__main__":
    # Load Data Once
    GLOBAL_DF, GLOBAL_FEATURES = load_and_process_data(ticker='AAPL')
    
    print("Starting Optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)
    
    print("\n--- Optimization Results ---")
    print("Best Value (Portfolio $):", study.best_value)
    print("Best Params:", study.best_params)
    
    # Save Best Params
    import yaml
    with open("marketing_best_params.yaml", "w") as f:
        yaml.dump(study.best_params, f)
