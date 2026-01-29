import wandb
import os
import pandas as pd
from dotenv import load_dotenv
import sys

load_dotenv()

def fetch_run_report(run_path):
    api = wandb.Api()
    try:
        run = api.run(run_path)
        print(f"Run ID: {run.id}")
        print(f"Name: {run.name}")
        print(f"Status: {run.state}")
        
        print("\n--- Summary Metrics ---")
        summary = run.summary
        metrics_of_interest = [
            "Ensemble_Sharpe", "Ensemble_Sortino", "Ensemble_Total_Return",
            "Initial_Balance_USDT", "Final_Balance_USDT", "Absolute_PnL_USDT",
            "train/global_step", "train/episode_reward"
        ]
        for m in metrics_of_interest:
            if m in summary:
                print(f"{m}: {summary[m]}")
        
        # Check for Tables
        print("\n--- Tables ---")
        for artifact in run.logged_artifacts():
            if artifact.type == 'run_table':
                print(f"Found Table Artifact: {artifact.name}")
                # We can't easily print the whole table here without downloading, 
                # but we can check the names.
        
        # In W&B API, tables are often logged as 'wandb.Table' objects in summary
        for k, v in summary.items():
            if isinstance(v, dict) and "_type" in v and v["_type"] == "table":
                print(f"Found Table in Summary: {k}")
                # Try to download and show a snippet
                table_data = run.logged_artifacts()
                # Actually, fetching table data via API:
                # table = run.summary[k]
                # ... too complex for a quick script, let's just note they exist.

        print("\n--- System Metadata ---")
        config = run.config
        for k, v in config.items():
            print(f"{k}: {v}")
            
    except Exception as e:
        print(f"Error fetching run: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        RUN_PATH = sys.argv[1]
    else:
        RUN_PATH = "bigcan-chiwin-technology/FinRL-Pro-DS/514e17aa"
    
    print(f"Fetching summary for: {RUN_PATH}")
    fetch_run_report(RUN_PATH)
