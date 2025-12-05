import pandas as pd
import numpy as np
import torch
import os
import config
from models import TradingEnv, SimpleAgent, SynapseArbitrator

def evaluate(agent, env):
    """Evaluates an agent on a given environment."""
    state, done, rewards = env.reset(), False, []
    while not done:
        # Handle both single agent and Synapse (which has predict method)
        if hasattr(agent, 'act'):
            action = agent.act(state)
        else:
            action, _ = agent.predict(state)
            
        state, reward, done, _ = env.step(action)
        
        # Update Synapse internal state if applicable
        if hasattr(agent, 'update'):
            agent.update(reward)
            
        rewards.append(reward)
        
    # Metrics
    if np.std(rewards) < 1e-8:
        sharpe = 0
    else:
        sharpe = np.mean(rewards) / np.std(rewards) * np.sqrt(252)
        
    total_return = (env.portfolio_values[-1] / env.portfolio_values[0] - 1) * 100
    
    return {
        'sharpe': sharpe,
        'return': total_return,
        'final_value': env.portfolio_values[-1]
    }

def run_backtest(df, windows):
    """Runs the rolling window backtest."""
    results = []
    
    print(f"Starting backtest with {len(windows)} rolling windows...")
    
    for i, w in enumerate(windows):
        print(f"\n=== Window {i+1}/{len(windows)} ===")
        print(f"Train: {w['train_start'].date()} -> {w['train_end'].date()}")
        print(f"Test:  {w['test_start'].date()} -> {w['test_end'].date()}")
        
        # Slice Data
        train_df = df[(df['date'] >= w['train_start']) & (df['date'] <= w['train_end'])].copy()
        test_df = df[(df['date'] >= w['test_start']) & (df['date'] <= w['test_end'])].copy()
        
        if len(train_df) < 100 or len(test_df) < 10:
            print("⚠ Skipping window - insufficient data")
            continue
            
        # Create Environments
        train_env = TradingEnv(train_df)
        test_env = TradingEnv(test_df)
        
        # Initialize Agents
        print("Training agents...")
        agents = []
        for j in range(3): # Ensemble of 3
            agent = SimpleAgent(train_env.state_dim, train_env.action_dim, seed=42+j*100)
            
            # Train
            for ep in range(config.TRAIN_EPOCHS):
                agent.train_episode(train_env)
                
            agents.append(agent)
            # print(f"  Agent {j+1} trained")
            
        # Initialize Synapse
        synapse = SynapseArbitrator(agents)
        
        # Evaluate on Test Set
        print("Evaluating...")
        
        # Single Agent (Best one? Or just the first one as baseline?)
        # Let's use Agent 1 as the representative Single Agent baseline
        single_metrics = evaluate(agents[0], test_env)
        
        # Synapse
        synapse_metrics = evaluate(synapse, test_env)
        
        # Market Baseline (Buy & Hold)
        # We can simulate this by taking the mean return of the universe or just using the env with static weights
        # For simplicity, let's just track the agent metrics for now
        
        print(f"  Single:  Sharpe={single_metrics['sharpe']:.3f}, Return={single_metrics['return']:.1f}%")
        print(f"  Synapse: Sharpe={synapse_metrics['sharpe']:.3f}, Return={synapse_metrics['return']:.1f}%")
        
        results.append({
            'window_id': i+1,
            'train_start': w['train_start'],
            'test_start': w['test_start'],
            'single_sharpe': single_metrics['sharpe'],
            'single_return': single_metrics['return'],
            'single_final_value': single_metrics['final_value'],
            'synapse_sharpe': synapse_metrics['sharpe'],
            'synapse_return': synapse_metrics['return'],
            'synapse_final_value': synapse_metrics['final_value']
        })
        
    # Save Results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(config.RESULTS_DIR, "backtest_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ Backtest complete. Results saved to {results_path}")
    
    return results_df
