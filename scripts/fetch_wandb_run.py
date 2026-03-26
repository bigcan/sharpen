import argparse
import json
import os

import paramiko
from dotenv import load_dotenv

import wandb

load_dotenv()

# Remote artifacts to download (filename -> description)
REMOTE_ARTIFACTS = {
    "checkpoint_final.pth": "Trained model checkpoint",
    "hpo.db": "HPO Optuna database",
}
REMOTE_WORKSPACE = "/workspace/DeepScalper"


def fetch_latest_run_metrics(entity="bigcan-chiwin-technology", project="FinRL-Pro-DS", tag=None):
    """
    Fetch the most recent run's metrics for goal verification.
    Args:
        tag: Optional tag to filter runs
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
        "validation_sharpe": summary.get("backtest_val/sharpe", summary.get("backtest_validation/sharpe", summary.get("backtest/validation_sharpe"))),
        "test_sharpe": summary.get("backtest_test/sharpe", summary.get("backtest/test_sharpe")),
        "validation_return": summary.get("backtest_val/total_return", summary.get("backtest_validation/total_return")),
        "test_return": summary.get("backtest_test/total_return"),

        # Institutional metrics (PyfolioAnalyzer)
        "validation_sortino": summary.get("backtest_val/sortino"),
        "test_sortino": summary.get("backtest_test/sortino"),
        "validation_calmar": summary.get("backtest_val/calmar"),
        "test_calmar": summary.get("backtest_test/calmar"),
        "validation_omega": summary.get("backtest_val/omega"),
        "test_omega": summary.get("backtest_test/omega"),
        "validation_max_drawdown": summary.get("backtest_val/max_drawdown"),
        "test_max_drawdown": summary.get("backtest_test/max_drawdown"),
        "validation_win_rate": summary.get("backtest_val/win_rate"),
        "test_win_rate": summary.get("backtest_test/win_rate"),
        "validation_profit_factor": summary.get("backtest_val/profit_factor"),
        "test_profit_factor": summary.get("backtest_test/profit_factor"),
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
        tag: Optional tag to filter runs

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


def download_remote_artifacts(run_id, remote_workspace=REMOTE_WORKSPACE):
    """
    Download run artifacts from remote GPUHub via SFTP.

    Downloads checkpoint, logs, and HPO database to results/{run_id}/.
    Requires GPUHUB_HOST, GPUHUB_PORT, GPUHUB_PASSWORD in .env.

    Args:
        run_id: WandB run ID (used for local directory naming)
        remote_workspace: Remote workspace path

    Returns:
        dict of {filename: local_path} for successfully downloaded files
    """
    host = os.getenv("GPUHUB_HOST")
    port = os.getenv("GPUHUB_PORT")
    password = os.getenv("GPUHUB_PASSWORD")

    if not all([host, port, password]):
        print("WARNING: GPUHUB credentials not found in .env. Skipping remote artifact download.")
        return {}

    local_dir = os.path.join(os.getcwd(), "results", run_id)
    os.makedirs(local_dir, exist_ok=True)

    downloaded = {}

    print(f"Connecting to {host}:{port} for artifact download...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.WarningPolicy())

    try:
        ssh.connect(host, port=int(port), username='root', password=password, timeout=30)
        sftp = ssh.open_sftp()

        for filename, description in REMOTE_ARTIFACTS.items():
            remote_path = f"{remote_workspace}/{filename}"
            local_path = os.path.join(local_dir, filename)

            try:
                remote_attr = sftp.stat(remote_path)
                size_mb = remote_attr.st_size / (1024 * 1024)
                print(f"  Downloading {filename} ({size_mb:.1f} MB) — {description}...")
                sftp.get(remote_path, local_path)
                downloaded[filename] = local_path
                print(f"  ✅ {filename} saved to {local_path}")
            except FileNotFoundError:
                print(f"  ⚠️  {filename} not found on remote (run may have crashed early)")
            except Exception as e:
                print(f"  ⚠️  Failed to download {filename}: {e}")

        # Download the most recent per-run log (run_YYYYMMDD_HHMMSS.log)
        # Falls back to legacy run.log if no timestamped logs exist
        try:
            log_files = []
            for entry in sftp.listdir_attr(remote_workspace):
                if entry.filename.startswith("run_") and entry.filename.endswith(".log"):
                    log_files.append((entry.filename, entry.st_mtime))

            if log_files:
                # Pick the most recently modified log
                log_files.sort(key=lambda x: x[1], reverse=True)
                log_name = log_files[0][0]
                remote_log = f"{remote_workspace}/{log_name}"
                local_log = os.path.join(local_dir, log_name)
                remote_attr = sftp.stat(remote_log)
                size_mb = remote_attr.st_size / (1024 * 1024)
                print(f"  Downloading {log_name} ({size_mb:.1f} MB) — Per-run log (latest)...")
                sftp.get(remote_log, local_log)
                downloaded[log_name] = local_log
                print(f"  ✅ {log_name} saved to {local_log}")
            else:
                # Fallback: try legacy run.log
                remote_log = f"{remote_workspace}/run.log"
                local_log = os.path.join(local_dir, "run.log")
                try:
                    remote_attr = sftp.stat(remote_log)
                    size_mb = remote_attr.st_size / (1024 * 1024)
                    print(f"  Downloading run.log ({size_mb:.1f} MB) — Legacy log (fallback)...")
                    sftp.get(remote_log, local_log)
                    downloaded["run.log"] = local_log
                    print(f"  ✅ run.log saved to {local_log}")
                except FileNotFoundError:
                    print("  ⚠️  No log files found on remote")
        except Exception as e:
            print(f"  ⚠️  Failed to download run log: {e}")

        sftp.close()
    except Exception as e:
        print(f"WARNING: SFTP connection failed: {e}")
    finally:
        ssh.close()

    return downloaded


def fetch_run_data(run_id, entity="bigcan-chiwin-technology", project="FinRL-Pro-DS", full_history=False):
    """
    Fetch run data from WandB and save to local JSON + SQLite registry.

    Args:
        run_id: WandB 8-character run ID
        entity: WandB entity
        project: WandB project name
        full_history: If True, fetch all history rows (up to 10K). Default: 100 samples.
    """
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
            "history": [],
        }

        # Fetch history
        if full_history:
            print("  Fetching full history (scan_history, max 10K rows)...")
            history_rows = []
            for i, row in enumerate(run.scan_history()):
                history_rows.append(row)
                if i >= 9999:
                    print("  WARNING: History capped at 10,000 rows.")
                    break
            data["history"] = history_rows
            print(f"  Fetched {len(history_rows)} history rows.")
        else:
            # Default: 100 sampled points for SPS calculation
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
    parser = argparse.ArgumentParser(description="Fetch WandB run data and optionally download remote artifacts")
    parser.add_argument("--run_id", type=str, required=True, help="WandB Run ID")
    parser.add_argument("--full_history", action="store_true", help="Fetch all history rows (up to 10K) instead of 100 samples")
    parser.add_argument("--download_artifacts", action="store_true", help="Download remote artifacts via SFTP (checkpoint, logs, hpo.db)")
    args = parser.parse_args()

    fetch_run_data(args.run_id, full_history=args.full_history)

    if args.download_artifacts:
        download_remote_artifacts(args.run_id)
