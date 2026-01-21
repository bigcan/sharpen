import sys
import pandas as pd
import numpy as np

def sharpe_ratio(returns, risk_free=0.0, periods=252):
    if len(returns) == 0: return 0.0
    mu = np.mean(returns) - risk_free/periods
    sigma = np.std(returns)
    if sigma == 0: return 0.0
    return (mu / sigma) * np.sqrt(periods)

def main():
    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)
    # Assuming column 'return' or similar
    if 'return' in df.columns:
        rets = df['return'].values
    elif 'daily_return' in df.columns:
        rets = df['daily_return'].values
    else:
        # Try 2nd column
        rets = df.iloc[:, 1].values
        
    sr = sharpe_ratio(rets)
    print(f"Sharpe Ratio: {sr:.4f}")
    print(f"Mean Return: {np.mean(rets)*252:.4f}")
    print(f"Vol: {np.std(rets)*np.sqrt(252):.4f}")

if __name__ == "__main__":
    main()
