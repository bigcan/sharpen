import json
import argparse
import os
import sqlite3
from datetime import datetime

def generate_report(run_id, output_path):
    # 1. Load Data
    json_path = os.path.join(os.getcwd(), "results", f"run_data_{run_id}.json")
    if not os.path.exists(json_path):
        print(f"Error: Data file not found at {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    summary = data.get("summary", {})
    config = data.get("config", {})
    name = data.get("name", "Unknown Run")
    
    # 2. Extract Metrics
    metrics = {
        "RunID": data.get("id"),
        "Name": name,
        "Date": data.get("created_at"),
        "Status": data.get("status"),
        "Url": data.get("url"),
        "ValSharpe": round(summary.get("backtest_val/sharpe", summary.get("backtest_validation/sharpe", summary.get("backtest/validation_sharpe", 0.0))), 2),
        "TestSharpe": round(summary.get("backtest_test/sharpe", summary.get("backtest/test_sharpe", 0.0)), 2),
        "ValReturn": round(summary.get("backtest_val/total_return", summary.get("backtest_validation/total_return", 0.0)) * 100, 2),
        "TestReturn": round(summary.get("backtest_test/total_return", 0.0) * 100, 2),
        "ValMaxDD": round(summary.get("backtest_val/max_drawdown", summary.get("backtest_validation/max_drawdown", 0.0)) * 100, 2),
        "TestMaxDD": round(summary.get("backtest_test/max_drawdown", 0.0) * 100, 2),
        "TotalSteps": summary.get("train/global_step", 0),
        "NumEnvs": config.get("env", {}).get("num_envs", "N/A"),
        "BatchSize": config.get("training", {}).get("batch_size", "N/A"),
    }

    # 3. Calculate Derived Metrics
    try:
        metrics["DeviationSharpe"] = round(abs(metrics["ValSharpe"] - metrics["TestSharpe"]) / abs(metrics["ValSharpe"]) * 100, 1)
    except ZeroDivisionError:
        metrics["DeviationSharpe"] = 0.0

    # 4. Fill Template
    report = f"""# DeepScalper Production Report: {metrics['Name']}

**Run ID:** `{metrics['RunID']}`
**Date:** {metrics['Date']}
**Status:** {metrics['Status']}
**WandB URL:** [Link]({metrics['Url']})

## 1. Executive Summary
*   **Outcome:** {metrics['Status']}
*   **Performance:** Test Sharpe {metrics['TestSharpe']} (Target: > 1.0)

## 2. Financial Performance (Backtest)
| Metric | Validation | Test | Deviation |
| :--- | :--- | :--- | :--- |
| **Sharpe Ratio** | {metrics['ValSharpe']} | {metrics['TestSharpe']} | {metrics['DeviationSharpe']}% |
| **Total Return** | {metrics['ValReturn']}% | {metrics['TestReturn']}% | - |
| **Max Drawdown** | {metrics['ValMaxDD']}% | {metrics['TestMaxDD']}% | - |

## 3. Configuration
*   **Num Envs:** {metrics['NumEnvs']}
*   **Batch Size:** {metrics['BatchSize']}
*   **Total Steps:** {metrics['TotalSteps']}

## 4. Automated Decision
"""
    if metrics['TestSharpe'] >= 1.0 and metrics['ValSharpe'] >= 1.0:
        report += "\n✅ **GOAL ACHIEVED**: Run meets all production criteria.\n"
    else:
        report += "\n❌ **GOAL MISSED**: Performance below target thresholds.\n"

    # ------------------------------------------------------------------
    # Gap #5: Richer Reports (HPO + Logs)
    # ------------------------------------------------------------------
    artifact_dir = os.path.join(os.getcwd(), "results", run_id)
    
    # HPO Summary
    hpo_db_path = os.path.join(artifact_dir, "hpo.db")
    if os.path.exists(hpo_db_path):
        try:
            conn = sqlite3.connect(f"file:{hpo_db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            
            # Query trials (assuming standard Optuna schema)
            # We want: trial_id, state, value (Sharpe/Reward), params
            # This is a bit complex in raw SQL as params are in check table
            # Simplified: just count trials and get best value if possible
            cursor.execute("SELECT count(*) FROM trials")
            n_trials = cursor.fetchone()[0]
            
            cursor.execute("SELECT count(*) FROM trials where state='COMPLETE'")
            n_complete = cursor.fetchone()[0]
            
            cursor.execute("SELECT max(value) FROM trial_values") # value is usually sharpe or reward
            best_val = cursor.fetchone()[0]
            
            conn.close()
            
            report += f"""
## 5. HPO Summary
*   **Trials:** {n_complete}/{n_trials} completed
*   **Best Objective Value:** {best_val if best_val else 'N/A'}
*   *Detailed trial data available in `results/{run_id}/hpo.db`*
"""
        except Exception as e:
            report += f"\n## 5. HPO Summary\n*Error reading HPO DB: {e}*\n"

    # Log Tail
    log_path = os.path.join(artifact_dir, "run.log")
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                # Read last 2 KB
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 2048))
                tail = f.read()
                
            report += f"""
## 6. Remote Log (Tail)
```text
{tail}
```
"""
        except Exception as e:
            report += f"\n## 6. Remote Log\n*Error reading log: {e}*\n"

    # 5. Save Report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report generated: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    generate_report(args.run_id, args.output)
