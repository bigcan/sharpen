import wandb
import json
import os

import argparse


def fetch_latest_run_metrics(entity="bigcan-chiwin-technology", project="FinRL-Pro-DS", tag=None):
    """
    Fetch the most recent run's metrics for goal verification.
    Args:
        tag: Optional tag to filter runs (e.g. "Ralph_Autonomous")
    Returns dict with: state, hpo_trials, validation_sharpe, test_sharpe, checkpoint_saved
    """
    api = wandb.Api()
    
    # Filter by tag if provided to avoid resuming unrelated runs
    filters = {}
    if tag:
        filters = {"tags": {"$in": [tag]}}
        
    runs = api.runs(f"{entity}/{project}", order="-created_at", per_page=1, filters=filters)
    
    if not runs:
        return {"state": "no_runs", "error": "No runs found"}
    
    run = runs[0]
    summary = run.summary._json_dict
    
    # Extract metrics for goal checking
    metrics = {
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,  # "running", "finished", "crashed", "failed"
        "url": run.url,
        
        # HPO metrics
        "hpo_trials": summary.get("hpo/best_trial_number", summary.get("hpo/completed_trials", 0)),
        
        # Training metrics
        "checkpoint_saved": summary.get("train/checkpoint_saved", False) or "checkpoint" in str(summary),
        "total_steps": summary.get("train/global_step", 0),
        
        # Backtest metrics
        "validation_sharpe": summary.get("backtest_validation/sharpe", summary.get("backtest/validation_sharpe")),
        "test_sharpe": summary.get("backtest_test/sharpe", summary.get("backtest/test_sharpe")),
        "validation_return": summary.get("backtest_validation/total_return"),
        "test_return": summary.get("backtest_test/total_return"),
    }
    
    return metrics


def poll_run_until_complete(run_id=None, entity="bigcan-chiwin-technology", project="FinRL-Pro-DS", 
                            poll_interval=300, max_wait=7200, tag=None):
    """
    Poll a WandB run until it completes or times out.
    
    Args:
        run_id: Specific run ID to poll. If None, polls the latest run.
        poll_interval: Seconds between polls (default: 5 minutes)
        max_wait: Maximum seconds to wait (default: 2 hours)
        tag: Optional tag to filter runs (e.g. "Ralph_Autonomous")
    
    Returns:
        Final run state and metrics
    """
    import time
    
    api = wandb.Api()
    start_time = time.time()
    
    # Build filter for tag if provided
    filters = {}
    if tag:
        filters = {"tags": {"$in": [tag]}}
    
    while time.time() - start_time < max_wait:
        if run_id:
            run = api.run(f"{entity}/{project}/{run_id}")
        else:
            # Use tag filter when polling latest
            runs = api.runs(f"{entity}/{project}", order="-created_at", per_page=1, filters=filters)
            if not runs:
                print("No runs found, waiting...")
                time.sleep(poll_interval)
                continue
            run = runs[0]
        
        state = run.state
        print(f"[{time.strftime('%H:%M:%S')}] Run {run.id}: {state}")
        
        if state in ("finished", "crashed", "failed"):
            # Pass tag through to final metrics fetch for consistency
            return fetch_latest_run_metrics(entity, project, tag=tag)
        
        time.sleep(poll_interval)
    
    return {"state": "timeout", "error": f"Run did not complete within {max_wait}s"}


def fetch_run_data(run_id, entity="bigcan-chiwin-technology", project="FinRL-Pro-DS"):
    run_path = f"{entity}/{project}/{run_id}"
    # Use relative path for portability (works on WSL/Windows)
    output_file = os.path.join(os.getcwd(), "results", f"run_data_{run_id}.json")
    
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
        db_path = os.path.join(os.getcwd(), "results", "metrics.db")
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
        raise  # Re-raise to make failures explicit to caller

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True, help="WandB Run ID")
    args = parser.parse_args()
    
    fetch_run_data(args.run_id)
