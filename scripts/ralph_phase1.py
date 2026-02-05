import json
import sys
import os

# Add cwd to path to find scripts module
sys.path.append(os.getcwd())

from scripts.fetch_wandb_run import fetch_latest_run_metrics
from scripts.remote_cmd import remote_cmd

# 1. Check for Active Run (Physical & Cloud)
print("Checking remote process state...")
try:
    pid_output = remote_cmd("pgrep -f 'run_full_pipeline.py' || true", timeout=15)
    pids = [p.strip() for p in pid_output.split() if p.strip().isdigit()]
    
    if pids:
        print(f"PHYSICAL RUN DETECTED: PID(s) {pids}")
        is_running_physically = True
    else:
        is_running_physically = False
except Exception as e:
    print(f"CRITICAL WARNING: Remote process check failed: {e}")
    # Fail-safe: Assume running to prevent clobbering
    is_running_physically = True 

# Check WandB
metrics = fetch_latest_run_metrics(tag="Ralph_Autonomous")
wandb_state = metrics.get('state', 'unknown')
run_id = metrics.get('run_id', 'unknown')

if is_running_physically or wandb_state == 'running':
    print(f'RESUME MODE: Active run detected (Physical: {is_running_physically}, WandB: {wandb_state}). Skipping deployment.')
    state_path = '.agent/ralph_state.json'
    with open(state_path, 'r') as f:
        state = json.load(f)
    state['status'] = 'monitoring'
    if run_id != 'unknown':
        state['last_run_id'] = run_id
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=4)
    print('SKIP_DEPLOY')
    sys.exit(0)

# 2. Update State (Deploy)
state_path = '.agent/ralph_state.json'
with open(state_path, 'r') as f:
    state = json.load(f)

# Increment Iteration
state['iteration'] += 1
max_iter = state.get('max_iterations', 10)
print(f'Starting Iteration {state["iteration"]}/{max_iter}...')

if state['iteration'] > max_iter:
    print(f'MAX_ITERATIONS_REACHED ({max_iter})')
    sys.exit(1)

with open(state_path, 'w') as f:
    json.dump(state, f, indent=4)
print('DEPLOY_REQUIRED')
