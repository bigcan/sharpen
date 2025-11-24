
import sys
import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from finrl_pro.data.loader_pro import ProFeatureAssembler
from finrl_pro.envs.portfolio_allocation import PortfolioAllocationEnv
from finrl_pro.agents.ppo import PPOAgent

SNAPSHOT_ID = "36629e7c-ff6a-4585-9fcf-284058413028"

LIQUID_20 = [
    'SPY', 'QQQ', 'IWM', 'XLK', 'XLF', 'XLE', 'XLV',
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'JPM', 'XOM', 'JNJ', 'PG', 'V',
    'GLD', 'TLT'
]
MACRO = ['^VIX', '^TNX', 'DX-Y.NYB']

def split_arrays(assembly):
    """Separate Liquid 20 and Macro from the assembly."""
    df_price = pd.DataFrame(assembly.price_ary, index=assembly.dates, columns=assembly.tickers)
    df_tech = pd.DataFrame(assembly.tech_ary, index=assembly.dates) # Columns are flat
    
    # Extract Liquid 20 Price
    liq_price = df_price[sorted(LIQUID_20)].values
    
    # Extract Macro Features (Using Price as Feature)
    # Note: We might want tech indicators for macro too, but for now just raw levels/returns
    macro_price = df_price[sorted(MACRO)].values
    
    # Extract Liquid 20 Tech
    # Tech array is (T, N*F).
    # We need to filter columns corresponding to Liquid 20 tickers.
    # The assembler sorts tickers.
    # If ticker i is in Liquid 20, we keep block i.
    
    all_tickers = assembly.tickers
    n_tickers = len(all_tickers)
    n_tech_features = assembly.tech_ary.shape[1] // n_tickers
    
    liq_tech_blocks = []
    for i, tic in enumerate(all_tickers):
        if tic in LIQUID_20:
            # Extract block i
            start = i * n_tech_features
            end = (i + 1) * n_tech_features
            liq_tech_blocks.append(assembly.tech_ary[:, start:end])
            
    liq_tech = np.hstack(liq_tech_blocks)
    
    return liq_price, liq_tech, macro_price

def train_stage(agent, env, n_episodes, stage_name):
    print(f"\n--- Starting {stage_name} ({n_episodes} episodes) ---")
    rewards_history = []
    
    for e in tqdm(range(n_episodes)):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            
            agent.store_transition(state, action, reward, done, log_prob=log_prob)
            
            state = next_state
            ep_reward += reward
            
            if len(agent.buffer['states']) >= 256:
                agent.update()
                agent.reset_buffer()
        
        # End of episode update
        if len(agent.buffer['states']) > 0:
            agent.update()
            agent.reset_buffer()
            
        rewards_history.append(ep_reward)
        if (e+1) % 10 == 0:
             print(f"Ep {e+1}: Reward={ep_reward:.4f}, PF Value={info['portfolio_value']:.2f}")
             
    return rewards_history

def main():
    print("Loading Data...")
    assembler = ProFeatureAssembler()
    features_cfg = {
        "stockstats_overrides": ["close", "rsi_14", "macd"], # Minimal tech features
        "dataset_hash_source": "training_v3"
    }
    assembly = assembler.assemble_from_snapshot(
        snapshot_id=SNAPSHOT_ID,
        features_cfg=features_cfg
    )
    
    liq_price, liq_tech, macro_price = split_arrays(assembly)
    dates = assembly.dates
    
    # Define Date Ranges
    # Stage 1: 2009-2019
    stage1_mask = (np.array(dates) >= pd.Timestamp('2009-01-01', tz='UTC')) & (np.array(dates) < pd.Timestamp('2019-01-01', tz='UTC'))
    # Stage 2: 2000-2019
    stage2_mask = (np.array(dates) >= pd.Timestamp('2000-01-01', tz='UTC')) & (np.array(dates) < pd.Timestamp('2019-01-01', tz='UTC'))
    # Test: 2021-2025
    test_mask = (np.array(dates) >= pd.Timestamp('2021-01-01', tz='UTC'))
    
    # Create Envs
    env_stage1 = PortfolioAllocationEnv(
        price_ary=liq_price[stage1_mask],
        tech_ary=liq_tech[stage1_mask],
        macro_ary=macro_price[stage1_mask]
    )
    
    env_stage2 = PortfolioAllocationEnv(
        price_ary=liq_price[stage2_mask],
        tech_ary=liq_tech[stage2_mask],
        macro_ary=macro_price[stage2_mask]
    )
    
    env_test = PortfolioAllocationEnv(
        price_ary=liq_price[test_mask],
        tech_ary=liq_tech[test_mask],
        macro_ary=macro_price[test_mask]
    )
    
    # Setup Agent
    state_dim = env_stage1.observation_space.shape[0]
    action_dim = env_stage1.action_space.shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Agent: State Dim={state_dim}, Action Dim={action_dim}, Device={device}")
    
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        device=device,
        batch_size=256,
        n_epochs=10
    )
    
    # --- Stage 1: Bull Market Warmup ---
    train_stage(agent, env_stage1, n_episodes=50, stage_name="Stage 1 (Bull Warmup)")
    os.makedirs('models/phase5', exist_ok=True)
    agent.save('models/phase5/ppo_stage1.pth')
    
    # --- Stage 2: Full History Robustness ---
    train_stage(agent, env_stage2, n_episodes=50, stage_name="Stage 2 (Full History)")
    agent.save('models/phase5/ppo_stage2.pth')
    
    # --- Evaluation ---
    print("\n--- Evaluating on Test Set (2021-2025) ---")
    state, _ = env_test.reset()
    done = False
    portfolio_values = [env_test.portfolio_value]
    
    while not done:
        action, _ = agent.select_action(state, deterministic=True) # Deterministic for Eval
        state, _, done, _, info = env_test.step(action)
        portfolio_values.append(info['portfolio_value'])
        
    # Calculate Metrics
    values = np.array(portfolio_values)
    returns = np.diff(values) / values[:-1]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    max_dd = np.min((values / np.maximum.accumulate(values)) - 1)
    cum_ret = (values[-1] / values[0]) - 1
    
    print(f"Test Results:")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"Max Drawdown: {max_dd:.4f}")
    print(f"Cumulative Return: {cum_ret:.4f}")
    
    # Save Test Equity Curve
    plt.figure(figsize=(10, 6))
    plt.plot(dates[test_mask], values[:-1]) # Match length (values has N+1)
    plt.title(f'Phase 5 Agent Test Performance (Sharpe: {sharpe:.2f})')
    plt.grid(True, alpha=0.3)
    plt.savefig('figs/phase5/agent_test.png')
    print("Plot saved to figs/phase5/agent_test.png")

if __name__ == "__main__":
    main()
