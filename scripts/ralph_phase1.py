import json
import sys
import os
# Add parent dir to sys.path to allow imports from scripts
sys.path.append(os.getcwd())

from scripts.fetch_wandb_run import fetch_latest_run_metrics
from scripts.remote_cmd import remote_cmd

# 1. Check for Active Run (Physical & Cloud)
# Truth on Ground: Check if process is running on remote
print("Checking remote process state...")
try:
    # pgrep returns PIDs if found. -f matches full command line.
    # We ignore our own transient check command if it appears.
    # Robust check: Get full command line (-a), filter for python, exclude bash/pgrep wrappers
    pid_output = remote_cmd("pgrep -a -f 'run_full_pipeline.py' | grep 'python' | grep -v 'bash' | awk '{print $1}' || true", timeout=15)
    # cleaning output to get just numbers
    pids = [p.strip() for p in pid_output.split() if p.strip().isdigit()]
    
    if pids:
        print(f"PHYSICAL RUN DETECTED: PID(s) {pids}")
        is_running_physically = True
    else:
        is_running_physically = False
except Exception as e:
    print(f"CRITICAL WARNING: Remote process check failed: {e}")
    print("FAIL-SAFE: Assuming process is running to prevent accidental kill.")
    is_running_physically = True  # Safety First!

# Truth in Cloud: Check WandB
# Filter by tag to avoid hijacking unrelated runs
metrics = fetch_latest_run_metrics(tag="Ralph_Autonomous")
wandb_state = metrics.get('state', 'unknown')
run_id = metrics.get('run_id', 'unknown')

# LOGIC: If EITHER is running, we assume it's valid and do not deploy.
# We prioritize physical presence because WandB can lag.
if is_running_physically or wandb_state == 'running':
    print(f'RESUME MODE: Active run detected (Physical: {is_running_physically}, WandB: {wandb_state}). Skipping deployment.')
    # Update state to reflect monitoring status
    state_path = '.agent/ralph_state.json'
    with open(state_path, 'r') as f:
        state = json.load(f)
    state['status'] = 'monitoring'
    if run_id != 'unknown':
        state['last_run_id'] = run_id
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=4)
    print('SKIP_DEPLOY')
    sys.exit(0)  # Exit cleanly, agent will proceed to Phase 2

# 2. Update State (only if deploying)
state_path = '.agent/ralph_state.json'
with open(state_path, 'r') as f:
    state = json.load(f)

state['iteration'] += 1
max_iter = state.get('max_iterations', 10)  # Configurable in ralph_state.json
print(f'Starting Iteration {state["iteration"]}/{max_iter}...')

if state['iteration'] > max_iter:
    print(f'MAX_ITERATIONS_REACHED ({max_iter})')
    print('WORKFLOW TERMINATED: Goal not achieved after max attempts.')
    sys.exit(1)

with open(state_path, 'w') as f:
    json.dump(state, f, indent=4)
print('DEPLOY_REQUIRED')
