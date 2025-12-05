import os
import pandas as pd
import numpy as np

REPORTS_DIR = "reports"
LATEST_FILE = os.path.join(REPORTS_DIR, "LATEST")

def get_latest_report():
    if not os.path.exists(LATEST_FILE):
        print("No LATEST report pointer found. Run 'python scripts/util_index_reports.py' first.")
        return

    with open(LATEST_FILE, "r") as f:
        latest_id = f.read().strip()

    report_path = os.path.join(REPORTS_DIR, latest_id)
    if not os.path.exists(report_path):
        print(f"Report directory {report_path} not found.")
        return

    print(f"Latest Report ID: {latest_id}")
    print(f"Path: {report_path}")
    
    # Calculate metrics from returns.csv
    returns_file = os.path.join(report_path, "returns.csv")
    if os.path.exists(returns_file):
        try:
            df = pd.read_csv(returns_file)
            if 'return' in df.columns:
                returns = df['return'].values
                
                # Annualized Sharpe (assuming daily)
                mean_ret = np.mean(returns)
                std_ret = np.std(returns)
                sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret != 0 else 0.0
                
                # Total Cumulative Return
                cum_ret = (np.prod(1 + returns) - 1) * 100
                
                # Max Drawdown (approximate from returns)
                cum_returns = np.cumprod(1 + returns)
                peak = np.maximum.accumulate(cum_returns)
                drawdown = (cum_returns - peak) / peak
                max_dd = np.min(drawdown) * 100

                print("-" * 30)
                print(f"Sharpe Ratio: {sharpe:.4f}")
                print(f"Total Return: {cum_ret:.2f}%")
                print(f"Max Drawdown: {max_dd:.2f}%")
                print(f"Days:         {len(returns)}")
                print("-" * 30)
            else:
                print("Column 'return' not found in returns.csv")
        except Exception as e:
            print(f"Error reading returns.csv: {e}")
    else:
        print("returns.csv not found.")

if __name__ == "__main__":
    get_latest_report()
