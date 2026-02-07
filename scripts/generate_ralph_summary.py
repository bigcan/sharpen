import json
import os
import pandas as pd
from datetime import datetime

STATE_FILE = '.agent/ralph_state.json'
OUTPUT_FILE = 'Ralph_Workflow_Summary_Report.md'

def generate_summary():
    if not os.path.exists(STATE_FILE):
        print(f"State file {STATE_FILE} not found.")
        return

    with open(STATE_FILE, 'r') as f:
        state = json.load(f)

    history = state.get('history', [])
    fixes = state.get('fixes_applied', [])
    goal_criteria = state.get('goal_criteria', {})
    
    # Calculate Summary Metrics
    total_iterations = len(history)
    best_test_sharpe = -float('inf')
    best_run_id = "N/A"
    
    history_rows = []
    for entry in history:
        metrics = entry.get('metrics', {})
        test_sharpe = metrics.get('test', -999)
        if test_sharpe > best_test_sharpe:
            best_test_sharpe = test_sharpe
            best_run_id = entry.get('run_id')
            
        history_rows.append({
            "Iteration": entry.get('iteration'),
            "RunID": entry.get('run_id'),
            "Timestamp": entry.get('timestamp'),
            "Val Sharpe": round(metrics.get('val', 0.0), 4),
            "Test Sharpe": round(test_sharpe, 4),
            "Outcome": entry.get('outcome')
        })
        
    df_hist = pd.DataFrame(history_rows)
    
    # Generate Markdown
    md = f"# Ralph Workflow Summary Report\n\n"
    md += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    md += f"**Status:** {'GOAL MET' if state.get('goal_met') else 'MAX ITERATIONS REACHED'}\n"
    md += f"**Total Iterations:** {total_iterations} / {state.get('max_iterations')}\n"
    md += f"**Best Test Sharpe:** {best_test_sharpe:.4f} (Run `{best_run_id}`)\n"
    md += f"**Goal:** Test Sharpe >= {goal_criteria.get('min_test_sharpe', 1.0)}\n\n"
    
    md += "## 1. Iteration History\n\n"
    if not df_hist.empty:
        md += df_hist.to_markdown(index=False)
    else:
        md += "No history found."
    md += "\n\n"
    
    md += "## 2. Fixes Applied\n\n"
    if fixes:
        md += "| Iteration | Timestamp | Fix Applied | Reason |\n"
        md += "| :--- | :--- | :--- | :--- |\n"
        for fix in fixes:
            md += f"| {fix.get('iteration')} | {fix.get('timestamp')} | {fix.get('fix')} | {fix.get('reason')} |\n"
    else:
        md += "No fixes applied.\n"
        
    md += "\n"
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(md)
        
    print(f"Report generated: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    generate_summary()
