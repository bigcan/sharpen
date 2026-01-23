import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finrl_pro_ds.analytics.wandb_evaluator import generate_wandb_report, WandbFinRLEvaluator

def create_mock_data(start_date='2023-01-01', periods=100, name="Agent"):
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    # Random Walk for Account Value
    returns = np.random.normal(0, 0.01, periods)
    # Add some drift
    if "Ensemble" in name:
        returns += 0.0005
    
    account_value = 10000 * np.cumprod(1 + returns)
    
    # Random Actions (-1, 0, 1)
    actions = np.random.choice([-1, 0, 1], size=periods)
    
    # Enhanced: Price simulation (random walk starting at 100)
    price_changes = np.random.normal(0, 0.5, periods)
    prices = 100 + np.cumsum(price_changes)
    
    # Enhanced: Quantity (scale actions by random share amounts, 0 when no action)
    base_qty = np.random.randint(10, 100, periods)
    quantities = actions * base_qty  # Signed: + for buy, - for sell
    
    df = pd.DataFrame({
        'date': dates,
        'account_value': account_value,
        'actions': actions,
        'price': prices,
        'quantity': quantities,
        'ticker': 'TEST'
    })
    
    return df


def test_wandb_evaluator():
    print("Generating mock data...")
    df_ensemble = create_mock_data(name="Ensemble")
    df_a2c = create_mock_data(name="A2C")
    df_ppo = create_mock_data(name="PPO")
    
    dict_agents = {
        'A2C': df_a2c,
        'PPO': df_ppo
    }
    
    print("Testing WandbFinRLEvaluator initialization...")
    try:
        # Note: We use 'disabled' mode for W&B to avoid needing auth during automated tests
        # In production, this would be 'online' or omitted.
        import wandb
        os.environ["WANDB_MODE"] = "disabled"
        
        evaluator = generate_wandb_report(
            df_ensemble=df_ensemble,
            dict_agents=dict_agents,
            run_name="test_verification_run",
            project_name="finrl_verification",
            benchmark_ticker="^GSPC" # yfinance will try to fetch, might fail if no internet but handled
        )
        
        print("Metric Calculation Check:")
        ens_results = evaluator.results['Ensemble']
        print(f"Ensemble Sharpe: {ens_results['Sharpe_Ratio']:.4f}")
        print(f"Ensemble Total Return: {ens_results['Total_Return']:.4%}")
        
        diversity_corr, lift = evaluator.calculate_diversity()
        print("\nDiversity Analysis:")
        print(f"Ensemble Lift: {lift:.4f}")
        print("Correlation Matrix:")
        print(diversity_corr)
        
        print("\nSUCCESS: WandbFinRLEvaluator verified locally (W&B in disabled mode).")
        
    except Exception as e:
        print(f"\nFAILURE: Verification failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_wandb_evaluator()
