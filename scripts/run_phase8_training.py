import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch
import mlflow
from stable_baselines3.common.vec_env import DummyVecEnv

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finrl_pro_ds.data.loader_pro import ProFeatureAssembler
from finrl_pro_ds.envs.factory import make_pro_env
from finrl_pro_ds.agents.ppo import PPOAgent
from finrl_pro_ds.data.regimes import HMMRegimeDetector, MarketRegime

# Configuration
CONFIG_PATH = "finrl_pro_ds/configs/experiments/phase8_comprehensive.yaml"
DATA_PATH = "data/sp500_full_2010_2025.parquet"
MODEL_DIR = "models/phase8"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def get_regime_labels(df, train_end_date):
    """
    Fit HMM on SPY training data and predict for full dataset.
    Returns a Series of regime labels aligned with df index.
    """
    print("Detecting Market Regimes based on SPY...")
    
    # Filter SPY
    spy_df = df[df['tic'] == 'SPY'].copy()
    spy_df = spy_df.sort_values('date').set_index('date')
    
    # Calculate Returns
    returns = spy_df['close'].pct_change().dropna()
    
    # Split Train/Test for HMM Fit
    train_returns = returns[returns.index <= train_end_date]
    
    # Fit HMM
    detector = HMMRegimeDetector(n_components=3, random_state=42)
    detector.fit_predict(train_returns.values)
    
    # Predict for ALL data
    all_regimes = detector.predict(returns.values)
    regime_series = pd.Series(all_regimes, index=returns.index)
    
    # Map Regimes to Semantic Labels (Bull/Bear/Sideways)
    # We use the average return of each state to identify them
    state_means = []
    for i in range(3):
        mask = (all_regimes == i)
        mean_ret = returns.values[mask].mean()
        state_means.append((i, mean_ret))
        
    # Sort by mean return
    state_means.sort(key=lambda x: x[1])
    
    # Lowest return -> Bear (0)
    # Middle return -> Sideways (1)
    # Highest return -> Bull (2)
    
    mapping = {
        state_means[0][0]: 'Bear',
        state_means[1][0]: 'Sideways',
        state_means[2][0]: 'Bull'
    }
    
    print(f"Regime Mapping (State -> Label): {mapping}")
    print(f"Regime Means: {state_means}")
    
    semantic_regimes = regime_series.map(mapping)
    
    # Re-align with original DF (forward fill for tickers other than SPY)
    # The regimes are daily, so we map by date.
    return semantic_regimes

def train_specialist(label, df_regime, features_cfg, train_params):
    print(f"\n=== Training {label} Specialist ===")
    print(f"Data points: {len(df_regime)}")
    
    if len(df_regime) < 1000:
        print("Warning: Not enough data for this regime. Skipping.")
        return None

    # Assemble Features
    assembler = ProFeatureAssembler()
    # Note: We must use the same feature config
    dataset = assembler.assemble_from_df(df=df_regime, features_cfg=features_cfg, dataset_hash=f"phase8_{label}")
    
    # Create Env
    env = make_pro_env(dataset, 
                       initial_capital=train_params['training']['environment']['initial_capital'],
                       reward_scaling=1e-4) # Scaling helps convergence
    
    # Init Agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent_params = train_params['training']['agent']['params']
    agent = PPOAgent(state_dim, action_dim, 
                     lr=agent_params['learning_rate'],
                     gamma=agent_params['gamma'],
                     gae_lambda=agent_params['gae_lambda'],
                     clip_ratio=agent_params['clip_range'], # Config uses clip_range
                     entropy_coef=agent_params['ent_coef'], # Config uses ent_coef
                     batch_size=agent_params['batch_size'],
                     n_epochs=10)
    
    # Train
    total_timesteps = train_params['training']['total_timesteps']
    n_steps = train_params['training']['agent']['params']['n_steps']
    print(f"Training for {total_timesteps} steps (Batch size: {n_steps})...")
    
    # Training Loop
    obs, _ = env.reset()
    for step in range(total_timesteps):
        action, log_prob = agent.select_action(obs)
        next_obs, reward, done, truncated, _ = env.step(action)
        
        agent.store_transition(obs, action, reward, done, next_obs, log_prob)
        
        obs = next_obs
        
        if done:
            obs, _ = env.reset()
            
        # Update if buffer is full
        if (step + 1) % n_steps == 0:
            loss_metrics = agent.update()
            agent.reset_buffer()
            if (step + 1) % (n_steps * 5) == 0:
                print(f"Step {step+1}: {loss_metrics}")
    
    # Final update if residual
    if len(agent.buffer['states']) > 0:
        agent.update()
    
    # Save
    save_path = os.path.join(MODEL_DIR, f"{label.lower()}.pth")
    agent.save(save_path)
    print(f"Saved model to {save_path}")
    return agent

def run_ensemble_backtest(agents, df_test, semantic_regimes, features_cfg):
    print("\n=== Running Ensemble Backtest (2024-2025) ===")
    
    # Assemble Test Env
    assembler = ProFeatureAssembler()
    dataset = assembler.assemble_from_df(df=df_test, features_cfg=features_cfg, dataset_hash="phase8_test")
    env = make_pro_env(dataset)
    
    obs, _ = env.reset()
    done = False
    
    # Metrics
    portfolio_values = []
    dates = dataset.dates # Assembly is a dataclass
    
    # We need to track which agent was used
    regime_log = []
    
    step_idx = 0
    while not done:
        current_date = pd.to_datetime(dates[step_idx])
        
        # Determine Regime
        # Look up in semantic_regimes series
        # If missing (e.g. first day of test might not have returns for HMM?), ffill
        try:
            regime = semantic_regimes.asof(current_date)
        except:
            regime = 'Sideways' # Default
            
        if pd.isna(regime): regime = 'Sideways'
        
        regime_log.append(regime)
        
        # Select Agent
        # Fallback Logic: Sideways -> Bear (Conservative) -> Bull -> First Available
        agent = agents.get(regime.lower())
        if not agent:
            agent = agents.get('sideways')
        if not agent:
             agent = agents.get('bear')
        if not agent:
             agent = agents.get('bull')
        if not agent:
             # Should not happen if at least one trained
             agent = list(agents.values())[0]
        
        # Act
        action, _ = agent.select_action(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        # Handle Vectorized Info
        if isinstance(info, list):
            step_info = info[0]
        else:
            step_info = info
            
        val = step_info.get('total_assets', step_info.get('account_value'))
        if val is None:
             # Fallback if key missing (should not happen in standard env)
             print(f"Warning: 'total_assets' missing in info: {step_info.keys()}")
             val = 0.0
             
        portfolio_values.append(val)
        step_idx += 1
        
        if step_idx >= len(dates) - 1:
            break
            
    # Calculate Sharpe
    portfolio_values = np.array(portfolio_values)
    returns = pd.Series(portfolio_values).pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    
    print(f"Ensemble Test Sharpe: {sharpe:.4f}")
    print(f"Final Portfolio Value: {portfolio_values[-1]:.2f}")
    
    # Save Results
    results_df = pd.DataFrame({
        'date': dates[:len(portfolio_values)],
        'account_value': portfolio_values,
        'regime': regime_log[:len(portfolio_values)]
    })
    results_df.to_csv("results/phase8_ensemble_test.csv", index=False)
    print("Saved results to results/phase8_ensemble_test.csv")

def main():
    cfg = load_config()
    
    # 1. Load Data
    print("Loading Dataset...")
    df = pd.read_parquet(DATA_PATH)
    # Rename columns if needed (already verified as 'ticker', 'timestamp' -> 'date')
    df = df.rename(columns={'timestamp': 'date', 'ticker': 'tic'})
    df['date'] = pd.to_datetime(df['date'])
    
    # 2. Split Train/Test
    train_end = pd.to_datetime(cfg['training']['end_date'])
    df_train = df[df['date'] <= train_end]
    df_test = df[df['date'] > train_end]
    
    print(f"Train Range: {df_train['date'].min()} - {df_train['date'].max()}")
    print(f"Test Range: {df_test['date'].min()} - {df_test['date'].max()}")
    
    # 3. Detect Regimes
    regimes = get_regime_labels(df, train_end)
    
    # 4. Train Specialists
    agents = {}
    for label in ['Bull', 'Bear', 'Sideways']:
        # Filter Data for this regime
        # We need to filter the Multi-Asset DF based on the Date's regime
        # Get dates where regime == label
        valid_dates = regimes[regimes == label].index
        
        # Filter training data
        df_subset = df_train[df_train['date'].isin(valid_dates)]
        
        # Train
        agent = train_specialist(label, df_subset, cfg['features'], cfg)
        if agent:
            agents[label.lower()] = agent
            
    # 5. Backtest Ensemble
    # Ensure we have all agents, or fallback
    if not agents:
        print("No agents trained!")
        return
        
    run_ensemble_backtest(agents, df_test, regimes, cfg['features'])

if __name__ == "__main__":
    main()
