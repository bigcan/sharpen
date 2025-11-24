import mlflow
import torch
import pandas as pd
import numpy as np
import os
from finrl_pro.agents.ppo import PPOAgent
from finrl_pro.data.loader_pro import ProFeatureAssembler
from finrl_pro.envs.factory import make_pro_env
from finrl_pro.data.loader import DataLoader

# Run IDs (Hardcoded from training step)
RUN_IDS = {
    "bull": "74324eece6cd446c837e055cf34f6415",
    "bear": "b3bd16fc4b5349a9a267a1fc84427e61",
    "sideways": "cf31189fd7884338be0958d6761056dc"
}

TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", "V", "JNJ",
    "WMT", "PG", "XOM", "UNH", "MA", "HD", "CVX", "MRK", "ABBV", "KO"
]

def load_agent(run_id, state_dim, action_dim, device="cpu"):
    print(f"Loading agent from run {run_id}...")
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model/temp_model.pth")
    
    # Recreate Agent (Params must match training config, or be close enough)
    # We assume standard params for inference structure
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    agent.policy.load_state_dict(torch.load(local_path, map_location=device))
    return agent

def get_regime(date, df):
    """
    Determine regime based on market data up to date.
    Simple Heuristic: Average distance from SMA60 across universe.
    """
    # This is slow if done per step naively. 
    # We pre-compute regimes for the whole dataframe.
    pass

def precompute_regimes(df):
    """
    Add 'regime' column: 0=Sideways, 1=Bull, -1=Bear
    """
    # Calculate SMA60 for each ticker
    # We need pivot
    piv = df.pivot(index="date", columns="tic", values="close")
    sma60 = piv.rolling(60).mean()
    
    # Deviation
    dev = (piv - sma60) / sma60
    
    # Mean Deviation across market
    market_score = dev.mean(axis=1)
    
    # Define thresholds
    # > 0.02 -> Bull
    # < -0.02 -> Bear
    # Else -> Sideways
    regimes = pd.Series(0, index=market_score.index) # Default Sideways
    regimes[market_score > 0.02] = 1
    regimes[market_score < -0.02] = -1
    
    return regimes

def main():
    # 1. Load Data (Test Set: 2023-2025)
    print("Loading Test Data...")
    df = DataLoader.resolve_dataset("file://data/sp500_full_2010_2025.parquet")
    
    # Filter Tickers
    if "tic" in df.columns:
        df = df[df["tic"].isin(TICKERS)]
        
    df["date"] = pd.to_datetime(df["date"])
    test_df = df[(df["date"] >= "2023-01-04") & (df["date"] <= "2024-12-31")]
    
    # 2. Precompute Regimes
    print("Detecting Regimes...")
    regimes = precompute_regimes(df) # Compute on full history to have SMA ready
    test_df = df[(df["date"] >= "2023-01-04") & (df["date"] <= "2024-12-31")]
    
    print(f"DEBUG: Test Data Shape: {test_df.shape}")
    print(f"DEBUG: Test Data Head:\n{test_df.head()}")
    
    # 3. Assemble Features
    # Force specific features to match training
    features_cfg = {
        "stockstats_overrides": [
            "macd", "boll_ub", "boll_lb", "rsi_30", "dx_30", "close_30_sma", "close_60_sma"
        ],
        "use_turbulence": True
    }
    assembler = ProFeatureAssembler()
    asm = assembler.assemble_from_df(df=test_df, features_cfg=features_cfg, dataset_hash="test_v2")
    
    print(f"DEBUG: Tickers: {len(asm.tickers)}")
    print(f"DEBUG: Feature List: {asm.feature_list}")
    print(f"DEBUG: Feature Dim: {len(asm.feature_list)}")
    
    # 4. Create Env
    env = make_pro_env(asm, initial_capital=1000000, reward_scaling=1e-4)
    print(f"DEBUG: Env State Dim: {env.observation_space.shape[0]}")
    print(f"DEBUG: Env Action Dim: {env.action_space.shape[0]}")
    
    # 5. Load Agents
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agents = {}
    try:
        agents["bull"] = load_agent(RUN_IDS["bull"], state_dim, action_dim)
        agents["bear"] = load_agent(RUN_IDS["bear"], state_dim, action_dim)
        agents["sideways"] = load_agent(RUN_IDS["sideways"], state_dim, action_dim)
    except Exception as e:
        print(f"Failed to load agents: {e}")
        return

    # 6. Run Backtest
    print("Running Ensemble Backtest...")
    obs, _ = env.reset()
    done = False
    
    rewards = []
    regime_history = []
    
    # Helper to map date to regime
    # ProStockEnv tracks 'day' index. asm.dates[day] gives date.
    
    while not done:
        # Get current date
        # env.day is the index in asm.dates
        # But env might handle day internally.
        # ProStockEnv: self.day
        current_date = asm.dates[env.day] # Assuming env.day is accessible and synced
        
        # Get Regime
        # date might be timestamp, regimes index is timestamp
        regime_val = regimes.get(current_date, 0) # Default 0
        regime_history.append(regime_val)
        
        # Select Agent
        if regime_val == 1:
            agent = agents["bull"]
        elif regime_val == -1:
            agent = agents["bear"]
        else:
            agent = agents["sideways"]
            
        # Act
        action, _ = agent.select_action(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        
        if truncated: done = True

    # 7. Metrics
    print(f"Backtest Complete. Steps: {len(rewards)}")
    returns = np.array(rewards) # Approx
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    print(f"Ensemble Sharpe Ratio: {sharpe:.4f}")
    
    # Regime Stats
    print(f"Regime Distribution: Bull={regime_history.count(1)}, Bear={regime_history.count(-1)}, Sideways={regime_history.count(0)}")

if __name__ == "__main__":
    main()
