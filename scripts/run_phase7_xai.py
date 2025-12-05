import mlflow
import torch
import pandas as pd
import numpy as np
import shap
from finrl_pro.agents.ppo import PPOAgent
from finrl_pro.data.loader_pro import ProFeatureAssembler
from finrl_pro.data.loader import DataLoader
from finrl_pro.explainability.ppo_explainer import PPOExplainer

# Hardcoded from training
RUN_ID_BULL = "74324eece6cd446c837e055cf34f6415"
TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", "V", "JNJ",
    "WMT", "PG", "XOM", "UNH", "MA", "HD", "CVX", "MRK", "ABBV", "KO"
]

def load_agent(run_id, state_dim, action_dim, device="cpu"):
    print(f"Loading agent from run {run_id}...")
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model/temp_model.pth")
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    agent.policy.load_state_dict(torch.load(local_path, map_location=device))
    return agent

def main():
    print("Loading Data for Explainability...")
    # Load Bull Market Data (2016-2018) where the specialist was trained
    df = DataLoader.resolve_dataset("file://data/sp500_full_2010_2025.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["tic"].isin(TICKERS)]
    
    # Focus on a Bull Period
    bull_df = df[(df["date"] >= "2017-01-01") & (df["date"] <= "2017-06-01")]
    
    # Assemble
    features_cfg = {
        "stockstats_overrides": [
            "macd", "boll_ub", "boll_lb", "rsi_30", "dx_30", "close_30_sma", "close_60_sma"
        ],
        "use_turbulence": True
    }
    assembler = ProFeatureAssembler()
    asm = assembler.assemble_from_df(df=bull_df, features_cfg=features_cfg, dataset_hash="xai_bull")
    
    # Prepare Inputs
    # Env State: [Amount, Turb, TurbBool, Prices(N), Stocks(N), CD(N), Tech(N*F)]
    # We need to manually construct the state array from Assembly for SHAP
    # Or use a dummy env to step through and collect states.
    
    print("Simulating environment to collect states...")
    from finrl_pro.envs.factory import make_pro_env
    env = make_pro_env(asm, initial_capital=1000000)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = load_agent(RUN_ID_BULL, state_dim, action_dim)
    
    states = []
    obs, _ = env.reset()
    for _ in range(200): # Collect 200 steps
        states.append(obs)
        action, _ = agent.select_action(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        if done: break
        
    states = np.array(states)
    print(f"Collected {states.shape[0]} states.")
    
    # Run SHAP
    print("Running SHAP Analysis...")
    # Background: First 50 states
    # Explain: Next 50 states
    background = states[:50]
    to_explain = states[50:100]
    
    explainer = PPOExplainer(agent, background)
    shap_values = explainer.explain(to_explain)
    
    print("SHAP Values computed.")
    print(f"Shape: {np.array(shap_values).shape}") # (actions, samples, features) or (samples, features) depending on output
    
    # Summary
    # shap_values shape: (Samples, InputFeatures, OutputDim)
    # We want importance per InputFeature
    # Aggregate absolute values over Samples (0) and Outputs (2)
    
    if isinstance(shap_values, list):
        # If multiple outputs, sum them
        # (Samples, Features)
        importances = np.sum([np.mean(np.abs(s), axis=0) for s in shap_values], axis=0)
    else:
        # (Samples, Features, Outputs) -> (Features,)
        # Sum over Outputs, Mean over Samples
        importances = np.mean(np.sum(np.abs(shap_values), axis=2), axis=0)
        
    print(f"Feature Importances Shape: {importances.shape}") # Should be (183,)
        
    # Map back to feature names (Approximate)
    # State Structure: 
    # 0: Amount
    # 1: Turb
    # 2: TurbBool
    # 3..3+N: Prices
    # ...
    # Last chunk: Tech Features
    
    # For simplicity, print top indices
    top_indices = np.argsort(importances)[-10:][::-1]
    print("\nTop 10 Important Feature Indices:")
    for idx in top_indices:
        print(f"Index {idx}: Importance {importances[idx]:.4f}")

if __name__ == "__main__":
    main()
