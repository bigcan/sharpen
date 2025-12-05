import mlflow
import torch
import pandas as pd
import numpy as np
import os
from finrl_pro.agents.ppo import PPOAgent
from finrl_pro.data.loader_pro import ProFeatureAssembler
from finrl_pro.envs.factory import make_pro_env
from finrl_pro.data.loader import DataLoader
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

def precompute_regimes_hmm(df, hysteresis_window=3):
    """
    Use HMM to detect regimes on the market average return and apply hysteresis.
    Returns a DataFrame of probabilities (indexed by date), where columns are MarketRegime enum values.
    """
    print("Computing Market Proxy Returns...")
    # Pivot to get matrix of closes
    piv = df.pivot(index="date", columns="tic", values="close")
    
    # Calculate Daily Returns
    returns = piv.pct_change().fillna(0)
    
    # Create Market Proxy (Equal Weight Index)
    market_returns = returns.mean(axis=1).dropna() # Drop NaNs from pct_change
    
    # Fit HMM for rolling probabilities
    print("Fitting HMMRegimeDetector for rolling probabilities (3 Components)...")
    detector = HMMRegimeDetector(n_components=3, random_state=42)
    
    # rolling_predict_proba is PIT-safe and returns DataFrame of probabilities
    # Columns are MarketRegime.value (0=BEAR, 1=BULL, 2=CRISIS, 3=SIDEWAYS)
    regime_probs_df = detector.rolling_predict_proba(market_returns, window=252, min_periods=60)
    
    # Apply Hysteresis: rolling mean on probabilities
    if hysteresis_window > 1:
        print(f"Applying {hysteresis_window}-day hysteresis to HMM probabilities...")
        # Fill any NaNs from rolling (beginning of series) before applying mean
        regime_probs_df = regime_probs_df.fillna(method='ffill').fillna(method='bfill')
        regime_probs_df = regime_probs_df.rolling(window=hysteresis_window).mean().dropna()
        regime_probs_df = regime_probs_df.div(regime_probs_df.sum(axis=1), axis=0) # Re-normalize after rolling mean
    
    # Log distribution
    print(f"HMM Probabilities Head (with Hysteresis):\n{regime_probs_df.head()}")
    
    return regime_probs_df

def main():
    # 1. Load Data (Test Set: 2023-2025)
    print("Loading Test Data...")
    df = DataLoader.resolve_dataset("file://data/sp500_full_2010_2025.parquet")
    
    # Filter Tickers
    if "tic" in df.columns:
        df = df[df["tic"].isin(TICKERS)]
        
    df["date"] = pd.to_datetime(df["date"])
    
    # 2. Precompute Regimes (HMM Probabilities with Hysteresis)
    print("Detecting Regimes with HMM Probabilities...")
    # Use the full dataset for fitting to establish global regimes
    regime_probs_df = precompute_regimes_hmm(df)
    
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
        agents[MarketRegime.BULL] = load_agent(RUN_IDS["bull"], state_dim, action_dim)
        agents[MarketRegime.BEAR] = load_agent(RUN_IDS["bear"], state_dim, action_dim)
        agents[MarketRegime.CRISIS] = agents[MarketRegime.BEAR] # Map CRISIS to BEAR agent
        agents[MarketRegime.SIDEWAYS] = load_agent(RUN_IDS["sideways"], state_dim, action_dim)
    except Exception as e:
        print(f"Failed to load agents: {e}")
        return

    # 6. Run Backtest with Soft Voting
    print("Running Ensemble Backtest (Soft Voting with HMM Probabilities)...")
    obs, _ = env.reset()
    done = False
    
    rewards = []
    
    while not done:
        current_date = asm.dates[env.day]
        
        # Get current regime probabilities, use equal weights if not found (e.g., initial days of hysteresis)
        current_probs = regime_probs_df.loc[current_date] if current_date in regime_probs_df.index else pd.Series([1/len(MarketRegime)] * len(MarketRegime), index=[r.value for r in MarketRegime])
        
        # Normalize probabilities to ensure they sum to 1, especially after slicing
        current_probs = current_probs / current_probs.sum()
        
        # Collect actions from all relevant agents
        individual_actions = {}
        for regime_enum_val in [MarketRegime.BEAR, MarketRegime.BULL, MarketRegime.CRISIS, MarketRegime.SIDEWAYS]:
            if regime_enum_val in agents: # Ensure we have an agent for this regime
                action, _ = agents[regime_enum_val].select_action(obs, deterministic=True)
                individual_actions[regime_enum_val] = action
            
        # Compute weighted average action (soft voting)
        weighted_action = np.zeros(action_dim)
        for regime_enum_val, prob in current_probs.items():
            if regime_enum_val in individual_actions: # Ensure we have an action for this regime
                weighted_action += prob * individual_actions[MarketRegime(regime_enum_val)]
                
        # Perform step with the weighted action
        obs, reward, done, truncated, info = env.step(weighted_action)
        rewards.append(reward)
        
        if truncated: done = True

    # 7. Metrics
    print(f"Backtest Complete. Steps: {len(rewards)}")
    returns = np.array(rewards) # Approx
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    print(f"Ensemble Sharpe Ratio: {sharpe:.4f}")
    
    # Note: Regime distribution is now a blend, not distinct counts
    print("Regime distribution was applied as soft probabilities during backtest.")

if __name__ == "__main__":
    main()
