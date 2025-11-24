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

from finrl_pro.data.regimes import HMMRegimeDetector, MarketRegime

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

def precompute_regimes_hmm(df):
    """
    Use HMM to detect regimes on the market average return.
    Returns a Series mapping Date -> MarketRegime (Enum Int).
    """
    print("Computing Market Proxy Returns...")
    # Pivot to get matrix of closes
    piv = df.pivot(index="date", columns="tic", values="close")
    
    # Calculate Daily Returns
    returns = piv.pct_change().fillna(0)
    
    # Create Market Proxy (Equal Weight Index)
    market_returns = returns.mean(axis=1)
    
    # Fit HMM
    print("Fitting HMMRegimeDetector (3 Components)...")
    # 3 Components: Bull, Sideways, Crisis (mapped to Bear)
    detector = HMMRegimeDetector(n_components=3, random_state=42)
    
    # Fit on the entire history provided (2010-2025) to establish global regimes
    # Then we will lookup the date in the backtest loop
    regime_vals = detector.fit_predict(market_returns.values)
    
    regime_series = pd.Series(regime_vals, index=market_returns.index)
    
    # Log distribution
    counts = regime_series.value_counts()
    print(f"Regime Distribution (Global):\n{counts}")
    
    return regime_series

def main():
    # 1. Load Data (Test Set: 2023-2025)
    print("Loading Test Data...")
    df = DataLoader.resolve_dataset("file://data/sp500_full_2010_2025.parquet")
    
    # Filter Tickers
    if "tic" in df.columns:
        df = df[df["tic"].isin(TICKERS)]
        
    df["date"] = pd.to_datetime(df["date"])
    
    # 2. Precompute Regimes (HMM)
    print("Detecting Regimes with HMM...")
    # We use the full dataset for fitting to get stable regimes, then slice for backtest
    regimes = precompute_regimes_hmm(df)
    
    # Slice Test Data
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
    print("Running Ensemble Backtest (HMM Switching)...")
    obs, _ = env.reset()
    done = False
    
    rewards = []
    regime_history = []
    
    while not done:
        # Get current date
        current_date = asm.dates[env.day]
        
        # Get Regime from HMM Series
        # regime_val is MarketRegime enum (int)
        regime_val = regimes.get(current_date, MarketRegime.SIDEWAYS) 
        regime_history.append(regime_val)
        
        # Map Regime to Agent
        if regime_val == MarketRegime.BULL:
            agent = agents["bull"]
        elif regime_val == MarketRegime.BEAR or regime_val == MarketRegime.CRISIS:
            agent = agents["bear"] # Crisis -> Bear Agent
        else:
            agent = agents["sideways"] # Sideways or default
            
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
    bull_days = regime_history.count(MarketRegime.BULL)
    bear_days = regime_history.count(MarketRegime.BEAR) + regime_history.count(MarketRegime.CRISIS)
    side_days = regime_history.count(MarketRegime.SIDEWAYS)
    
    print(f"Regime Distribution (Test): Bull={bull_days}, Bear/Crisis={bear_days}, Sideways={side_days}")


if __name__ == "__main__":
    main()
