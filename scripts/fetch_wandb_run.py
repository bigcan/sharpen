import wandb
import json
import os

import argparse

def fetch_run_data(run_id, entity="bigcan-chiwin-technology", project="FinRL-Pro-DS"):
    run_path = f"{entity}/{project}/{run_id}"
    output_file = f"c:/FinRL/FinRL-Pro_DS/results/run_data_{run_id}.json"
    
    print(f"Fetching run data for {run_path}...")
    api = wandb.Api()
    try:
        run = api.run(run_path)
        
        # Collect data
        data = {
            "id": run.id,
            "name": run.name,
            "status": run.state,
            "config": run.config,
            "summary": run.summary._json_dict,
            "tags": run.tags,
            "created_at": run.created_at,
            "url": run.url,
            "history": [] # Attempt to fetch history sample
        }
        
        # Fetch some history for performance metrics (SPS)
        # We process 'train/global_step' vs '_runtime' to calc SPS if not logged
        history = run.history(keys=["step", "train/global_step", "_runtime"], samples=100)
        data["history"] = history.to_dict(orient="records")

        
        # Save to JSON (Detail View)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"Successfully saved run data to {output_file}")

        # Update Local Metrics DB (Registry View)
        import sqlite3
        db_path = "c:/FinRL/FinRL-Pro_DS/results/metrics.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                name TEXT,
                status TEXT,
                created_at TEXT,
                sharpe REAL,
                total_return REAL,
                sps REAL,
                tags TEXT,
                config TEXT,
                json_path TEXT
            )
        ''')
        
        # Check if config column exists (migration for existing DB)
        try:
            cursor.execute("SELECT config FROM runs LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE runs ADD COLUMN config TEXT")
        
        # Extract Key Metrics
        sharpe = data.get("summary", {}).get("backtest_test/sharpe", data.get("summary", {}).get("backtest/sharpe", 0.0))
        ret = data.get("summary", {}).get("backtest_test/total_return", data.get("summary", {}).get("backtest/total_return", 0.0))
        sps = data.get("summary", {}).get("train/sps", 0.0)
        
        # Upsert
        cursor.execute('''
            INSERT OR REPLACE INTO runs (id, name, status, created_at, sharpe, total_return, sps, tags, config, json_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (run.id, run.name, run.state, run.created_at, sharpe, ret, sps, json.dumps(run.tags), json.dumps(run.config), output_file))
        
        conn.commit()
        conn.close()
        print(f"Updated Local Metrics DB: {db_path}")

    except Exception as e:
        print(f"Error fetching run data: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True, help="WandB Run ID")
    args = parser.parse_args()
    
    fetch_run_data(args.run_id)
