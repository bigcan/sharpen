import sys
import pandas as pd
import numpy as np

def analyze_execution(csv_path):
    df = pd.read_csv(csv_path)
    
    # Map columns
    if 'trade' in df.columns:
        trades = df['trade'].values
        positions = df['position'].values if 'position' in df.columns else np.cumsum(trades)
    else:
        print("Error: 'trade' column not found.")
        return

    # 1. Trade Distribution (Activity)
    print(f"--- Trade Stats (Action Proxy) ---")
    print(f"Mean Trade: {np.mean(trades):.4f}")
    print(f"Std Trade:  {np.std(trades):.4f}")
    print(f"Min/Max:    {np.min(trades):.4f} / {np.max(trades):.4f}")
    print(f"% Buy (>0): {np.mean(trades > 1e-5)*100:.1f}%")
    print(f"% Sell (<0):{np.mean(trades < -1e-5)*100:.1f}%")
    print(f"% Hold (~0):{np.mean(np.abs(trades) <= 1e-5)*100:.1f}%")

    # 2. Position Stats (Exposure)
    print(f"\n--- Position Stats ---")
    print(f"Mean Pos:   {np.mean(positions):.4f}")
    print(f"Std Pos:    {np.std(positions):.4f}")
    print(f"% Long:     {np.mean(positions > 1e-3)*100:.1f}%")
    print(f"% Short:    {np.mean(positions < -1e-3)*100:.1f}%")

    # 3. Turnover & Cost
    if 'turnover' in df.columns:
        turnover = df['turnover'].values
        print(f"\n--- Turnover ---")
        print(f"Mean Daily Turnover: {np.mean(turnover):.4f}")
        print(f"Annualized Turnover: {np.mean(turnover)*252:.2f}x")
    
    if 'transaction_cost' in df.columns:
        total_cost = df['transaction_cost'].sum()
        print(f"\n--- Cost ---")
        print(f"Total Cost: {total_cost:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_exec.py <path_to_execution.csv>")
        sys.exit(1)
    analyze_execution(sys.argv[1])
