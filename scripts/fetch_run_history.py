import wandb
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def fetch_run_history(run_path):
    api = wandb.Api()
    try:
        run = api.run(run_path)
        print(f"Run ID: {run.id}")
        
        # Fetch the last history row
        history = run.history(keys=["Ensemble_Sharpe", "Ensemble_Total_Return", "Final_Balance_USDT", "Absolute_PnL_USDT"])
        if not history.empty:
            print("\n--- Final Backtest Metrics (from history) ---")
            print(history.iloc[-1].to_dict())
        else:
            print("\nNo financial metrics found in history.")

        # Check Summary for these keys again, maybe I missed them
        summary = run.summary
        print("\n--- Summary Check ---")
        for k in ["Ensemble_Sharpe", "Ensemble_Total_Return", "Final_Balance_USDT", "Absolute_PnL_USDT"]:
            if k in summary:
                print(f"{k}: {summary[k]}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    RUN_PATH = "bigcan-chiwin-technology/FinRL-Pro-DS/514e17aa"
    fetch_run_history(RUN_PATH)
