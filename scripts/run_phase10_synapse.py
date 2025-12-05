"""
Phase 10: Synapse Arbitrator Backtest Driver.
Loads trained specialist agents and runs the Synapse Arbitrator on the test set.
"""

import mlflow
import torch
import pandas as pd
import numpy as np
import os
from finrl_pro.agents.ppo import PPOAgent
from finrl_pro.data.loader_pro import ProFeatureAssembler
from finrl_pro.envs.factory import make_pro_env
from finrl_pro.data.loader import DataLoader
from finrl_pro.execution.arbitrator import SynapseArbitrator

# Run IDs (Reused from Phase 6)
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
    try:
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model/temp_model.pth")
        agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, device=device)
        agent.policy.load_state_dict(torch.load(local_path, map_location=device))
        return agent
    except Exception as e:
        print(f"Error loading agent {run_id}: {e}")
        # Return a Mock Agent if loading fails (for testing without MLflow server)
        print("Returning Mock Agent (Random Policy) for testing...")
        return PPOAgent(state_dim=state_dim, action_dim=action_dim, device=device)

def main():
    # 1. Load Data (Test Set: 2023-2025)
    print("Loading Test Data...")
    df = DataLoader.resolve_dataset("file://data/sp500_full_2010_2025.parquet")
    
    if "tic" in df.columns:
        df = df[df["tic"].isin(TICKERS)]
        
    df["date"] = pd.to_datetime(df["date"])
    
    # Slice Test Data
    test_df = df[(df["date"] >= "2023-01-04") & (df["date"] <= "2024-12-31")]
    
    # 2. Assemble Features
    features_cfg = {
        "stockstats_overrides": [
            "macd", "boll_ub", "boll_lb", "rsi_30", "dx_30", "close_30_sma", "close_60_sma"
        ],
        "use_turbulence": True
    }
    assembler = ProFeatureAssembler()
    asm = assembler.assemble_from_df(df=test_df, features_cfg=features_cfg, dataset_hash="test_v2")
    
    # 3. Create Env
    env = make_pro_env(asm, initial_capital=1000000, reward_scaling=1e-4)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 4. Load Agents
    agents = []
    # Order matters? Synapse treats them as a pool.
    # We load Bull, Bear, Sideways.
    agents.append(load_agent(RUN_IDS["bull"], state_dim, action_dim))
    agents.append(load_agent(RUN_IDS["bear"], state_dim, action_dim))
    agents.append(load_agent(RUN_IDS["sideways"], state_dim, action_dim))
    
    # 5. Initialize Synapse Arbitrator
    print("Initializing Synapse Arbitrator...")
    arbitrator = SynapseArbitrator(agents, n_samples=100, window_size=20)
    
    # 6. Run Backtest
    print("Running Synapse Backtest...")
    obs, _ = env.reset()
    done = False
    rewards = []
    weights_history = []
    
    while not done:
        # Synapse Prediction
        action, _ = arbitrator.predict(obs)
        
        # Step
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        
        # Update Arbitrator with Realized Reward (Profit Scoring)
        arbitrator.update(reward)
        
        # Log weights
        weights_history.append(arbitrator.get_weights())
        
        if truncated: done = True
        
    # 7. Metrics
    print(f"Backtest Complete. Steps: {len(rewards)}")
    returns = np.array(rewards)
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    print(f"Synapse Sharpe Ratio: {sharpe:.4f}")
    
    # Save Weights History
    weights_df = pd.DataFrame(weights_history, columns=["Bull", "Bear", "Sideways"])
    weights_df.to_csv("results/phase10_synapse_weights.csv", index=False)
    print("Weights history saved to results/phase10_synapse_weights.csv")

if __name__ == "__main__":
    main()
