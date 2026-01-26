import wandb
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def fetch_backtest_report(run_path):
    api = wandb.Api()
    try:
        run = api.run(run_path)
        print(f"Run ID: {run.id}")
        print(f"Name: {run.name}")
        print(f"Status: {run.state}")
        
        print("\n--- Backtest Summary ---")
        summary = run.summary
        metrics = [
            "Ensemble_Sharpe", "Ensemble_Sortino", "Ensemble_Total_Return",
            "Initial_Balance_USDT", "Final_Balance_USDT", "Absolute_PnL_USDT"
        ]
        for m in metrics:
            if m in summary:
                print(f"{m}: {summary[m]}")
                
        # Also check for individual agent metrics if present
        for k, v in summary.items():
            if "Sharpe" in k or "Return" in k:
                if k not in metrics and not k.startswith("_"):
                    print(f"{k}: {v}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    RUN_PATH = "bigcan-chiwin-technology/FinRL-Pro-DS/qsividfx"
    fetch_backtest_report(RUN_PATH)
