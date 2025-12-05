
import sys
import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
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
    df_price = pd.DataFrame(assembly.price_ary, index=assembly.dates, columns=assembly.tickers)
    df_tech = pd.DataFrame(assembly.tech_ary, index=assembly.dates) 
    liq_price = df_price[sorted(LIQUID_20)].values
    macro_price = df_price[sorted(MACRO)].values
    
    all_tickers = assembly.tickers
    n_tickers = len(all_tickers)
    n_tech_features = assembly.tech_ary.shape[1] // n_tickers
    
    liq_tech_blocks = []
    for i, tic in enumerate(all_tickers):
        if tic in LIQUID_20:
            start = i * n_tech_features
            end = (i + 1) * n_tech_features
            liq_tech_blocks.append(assembly.tech_ary[:, start:end])
    liq_tech = np.hstack(liq_tech_blocks)
    return liq_price, liq_tech, macro_price

def calculate_metrics(returns):
    if len(returns) < 2:
        return {'Sharpe': 0, 'MaxDD': 0, 'Return': 0}
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    cum = np.cumprod(1 + returns)
    max_dd = np.min(cum / np.maximum.accumulate(cum) - 1)
    total_ret = cum[-1] - 1
    return {'Sharpe': sharpe, 'MaxDD': max_dd, 'Return': total_ret}

def evaluate():
    print("Loading Data...")
    assembler = ProFeatureAssembler()
    features_cfg = {
        "stockstats_overrides": ["close", "rsi_14", "macd"], 
        "dataset_hash_source": "training_v4"
    }
    assembly = assembler.assemble_from_snapshot(
        snapshot_id=SNAPSHOT_ID,
        features_cfg=features_cfg
    )
    
    liq_price, liq_tech, macro_price = split_arrays(assembly)
    dates = np.array(assembly.dates) # Convert to numpy array immediately
    
    # Load Regimes
    regimes_df = pd.read_csv('data/phase5_regimes.csv', index_col=0, parse_dates=True)
    # Ensure timezone alignment. assembly.dates are UTC. regimes_df index might be naive or UTC.
    # regimes_df index comes from assembly.dates in previous script, but CSV read might make it naive.
    if regimes_df.index.tz is None:
        regimes_df.index = regimes_df.index.tz_localize('UTC')
    
    # Define Test Range (2021-2025)
    test_start = pd.Timestamp('2021-01-01', tz='UTC')
    test_mask = (dates >= test_start)
    
    env_test = PortfolioAllocationEnv(
        price_ary=liq_price[test_mask],
        tech_ary=liq_tech[test_mask],
        macro_ary=macro_price[test_mask]
    )
    
    # Setup Agent
    state_dim = env_test.observation_space.shape[0]
    action_dim = env_test.action_space.shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    agent = PPOAgent(state_dim, action_dim, device=device)
    model_path = 'models/phase5/ppo_stage2.pth'
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        agent.load(model_path)
    else:
        print("Model not found! Cannot evaluate.")
        return

    # Run Backtest
    print("Running Backtest...")
    state, _ = env_test.reset()
    done = False
    
    portfolio_values = [env_test.portfolio_value]
    daily_returns = []
    test_dates = dates[test_mask]
    
    # We need to match returns to dates.
    # Step t produces return for t+1 (realized at t+1).
    # portfolio_values[0] is initial.
    # portfolio_values[1] is after day 0 return.
    # return[0] corresponds to test_dates[0] to test_dates[1].
    # So return[i] happens at date[i+1].
    
    while not done:
        action, _ = agent.select_action(state, deterministic=True)
        state, _, done, _, info = env_test.step(action)
        portfolio_values.append(info['portfolio_value'])
        daily_returns.append(info['return'])
        
    # Create Result DataFrame
    # Length of daily_returns is N_steps.
    # Length of test_dates is N_steps + 1 (prices).
    # Returns are realized on dates[1:].
    
    res_dates = test_dates[1:] # 2021-01-05 onwards (skipping first day open)
    res_df = pd.DataFrame({
        'return': daily_returns,
        'value': portfolio_values[1:]
    }, index=res_dates)
    
    # Join with Regimes
    # Reindex regimes to match result dates (forward fill if needed)
    res_df = res_df.join(regimes_df['regime_label'], how='left')
    res_df['regime_label'] = res_df['regime_label'].fillna('Unknown')
    
    # Overall Metrics
    overall = calculate_metrics(res_df['return'].values)
    print("\n--- Overall Performance (2021-2025) ---")
    print(pd.Series(overall))
    
    # Per-Regime Metrics
    print("\n--- Per-Regime Performance ---")
    regime_stats = []
    for regime, group in res_df.groupby('regime_label'):
        m = calculate_metrics(group['return'].values)
        m['Regime'] = regime
        m['Days'] = len(group)
        regime_stats.append(m)
        
    stats_df = pd.DataFrame(regime_stats).set_index('Regime')
    print(stats_df)
    
    stats_df.to_csv('data/phase5_regime_report.csv')
    print("Report saved to data/phase5_regime_report.csv")
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Equity Curve
    plt.subplot(2, 1, 1)
    plt.plot(res_df.index, res_df['value'], label='Agent Portfolio', color='blue')
    
    # Color background by regime
    # (Simplified: just vertical spans)
    # Find changes in regime
    # ... (visualization logic)
    
    plt.title('Agent Performance & Regimes')
    plt.grid(True, alpha=0.3)
    
    # Drawdown
    plt.subplot(2, 1, 2)
    cum = (1 + res_df['return']).cumprod()
    dd = cum / cum.cummax() - 1
    plt.plot(res_df.index, dd, label='Drawdown', color='red')
    plt.fill_between(res_df.index, dd, 0, color='red', alpha=0.3)
    plt.title('Drawdown Profile')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figs/phase5/eval_report.png')
    print("Plot saved to figs/phase5/eval_report.png")

if __name__ == "__main__":
    evaluate()
