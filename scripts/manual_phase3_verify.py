import json
import os
import sys

# Ensure scripts dir is in path
sys.path.append(os.getcwd())

from scripts.fetch_wandb_run import fetch_latest_run_metrics, fetch_run_data
from scripts.generate_report import generate_report

print("--- Manual Phase 3 Execution ---")

# 1. Fetch Metrics
print("Fetching latest metrics...")
metrics = fetch_latest_run_metrics(tag="Ralph_Autonomous")
run_id = metrics.get('run_id', 'unknown')
print(f"Latest run detected: {run_id}")

# 2. Load State
state_path = '.agent/ralph_state.json'
if os.path.exists(state_path):
    with open(state_path, 'r') as f:
        state = json.load(f)
    iteration = state.get('iteration', 'X')
else:
    print("State file not found!")
    iteration = 'X'
    state = {}

# 3. Generate Iteration Report
print(f'Fetching full data for run {run_id}...')
fetch_run_data(run_id)

os.makedirs('.agent/reports', exist_ok=True)
report_path = f'.agent/reports/ralph_iter_{iteration}_{run_id}.md'
print(f'Generating Report for Iteration {iteration}...')
generate_report(run_id, report_path)
print(f'Report saved to {report_path}')

# 4. Check Status
checkpoint_saved = metrics.get('checkpoint_saved', False)
val_sharpe = metrics.get('validation_sharpe', 0) or 0
test_sharpe = metrics.get('test_sharpe', 0) or 0
run_state = metrics.get('state', 'unknown')

print(f'[METRICS] Val:{val_sharpe} Test:{test_sharpe} State:{run_state}')

# 5. Verification Logic
# Fix for None type comparison
test_sharpe_val = float(test_sharpe) if test_sharpe is not None else 0.0

success = (
    test_sharpe_val >= 1.0 and 
    run_state == 'finished'
)

if success:
    print('GOAL_RESULT: SUCCESS')
    import shutil
    shutil.copy(report_path, '.agent/ralph_success_report.md')
    from datetime import datetime
    state.setdefault('history', []).append({
        'timestamp': datetime.now().isoformat(),
        'iteration': iteration,
        'run_id': run_id,
        'outcome': 'success',
        'metrics': {'val_sharpe': val_sharpe, 'test_sharpe': test_sharpe}
    })
    state['goal_met'] = True
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=4)
else:
    from datetime import datetime
    failure_reason = []
    if not checkpoint_saved: failure_reason.append('no_checkpoint')
    if test_sharpe_val < 1.0: failure_reason.append(f'low_test_sharpe:{test_sharpe}')
    if run_state != 'finished': failure_reason.append(f'run_state:{run_state}')
    
    state.setdefault('history', []).append({
        'timestamp': datetime.now().isoformat(),
        'iteration': iteration,
        'run_id': run_id,
        'outcome': 'failure',
        'failure_reasons': failure_reason,
        'metrics': {'val_sharpe': val_sharpe, 'test_sharpe': test_sharpe}
    })
    # Update last_run_id to match what we just processed
    state['last_run_id'] = run_id
    # Reset status to something that encourages the next phase? 
    # Actually Phase 1 checks if it's running. It's finished. So Phase 1 will proceed to deploy next.
    state['status'] = 'verified' 
    
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=4)
    print('GOAL_RESULT: FAILURE')
