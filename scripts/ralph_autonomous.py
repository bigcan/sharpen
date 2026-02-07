#!/usr/bin/env python3
"""
Ralph Autonomous Driver
=======================
Self-correcting loop for DeepScalper HPO/Training pipeline.
Handles the full lifecycle: Deploy -> Monitor -> Verify -> Fix -> Retry.
"""

import sys
import os
import time
import json
import shutil
import argparse
import subprocess
import yaml
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.fetch_wandb_run import fetch_latest_run_metrics, poll_run_until_complete, fetch_run_data
from scripts.remote_cmd import remote_cmd
from scripts.notion_sync import update_mission_control, check_command, log_run
from scripts.generate_report import generate_report

# Constants
STATE_FILE = '.agent/ralph_state.json'
DEFAULT_CONFIG = 'configs/deepscalper_rtx5090_production.yaml'
POLL_INTERVAL = 60  # Check Notion/Status every minute
LOG_INTERVAL = 900  # Update WandB/Logs every 15 min
MAX_WAIT_TIME = 28800 # 8 Hours max per run

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return None

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)

def init_state():
    """Initialize state file if not exists."""
    if not os.path.exists(STATE_FILE):
        print("Initializing Ralph State...")
        state = {
            'iteration': 0, 
            'max_iterations': 10,
            'goal_met': False, 
            'status': 'idle',
            'goal_criteria': {
                'min_test_sharpe': 1.0
            },
            'config_file': DEFAULT_CONFIG,
            'last_metrics': {},
            'last_error': None,
            'fixes_applied': [],
            'started_at': datetime.now().isoformat(),
            'history': []
        }
        save_state(state)
        return state
    return load_state()

def check_active_run():
    """Check if a run is already active (Physical AND WandB running)."""
    print("Checking active runs...")
    update_mission_control("Running", 0, "Checking active runs...")
    
    # 1. Physical Check
    is_running_physically = False
    try:
        pid_output = remote_cmd("pgrep -f '[r]un_full_pipeline.py' || true", timeout=15)
        pids = [p.strip() for p in pid_output.split() if p.strip().isdigit()]
        if pids:
            print(f"PHYSICAL RUN DETECTED: PID(s) {pids}")
            is_running_physically = True
    except Exception as e:
        print(f"Warning: Remote check failed ({e}).")

    # 2. WandB Check
    metrics = fetch_latest_run_metrics(tag="Ralph_Autonomous")
    wandb_state = metrics.get('state', 'unknown')
    run_id = metrics.get('run_id', 'unknown')
    
    # CRITICAL FIX: Only skip deploy if WandB is actively "running"
    # A "finished" WandB run with lingering physical process = stale zombie, deploy new run
    if wandb_state == 'running':
        print(f"Active WandB run detected (State: {wandb_state}, Physical: {is_running_physically})")
        if is_running_physically:
            return True, run_id
        else:
            print("WARNING: WandB says running but no physical process found. Treating as Zombie/Crashed.")
            return False, None
    
    # WandB not running - deploy regardless of physical process (could be zombie)
    if is_running_physically:
        print(f"Stale physical process detected (WandB: {wandb_state}). Killing and deploying fresh.")
        try:
            remote_cmd("pkill -f 'run_full_pipeline.py' || true", timeout=15)
        except:
            pass
    
    return False, None

def deploy(config_file):
    """Deploy the pipeline to bare metal."""
    print(f"Deploying {config_file}...")
    update_mission_control("Running", 0, "Deploying new run...")
    
    # Construct command
    # Use subprocess.run with list args for secure and robust quoting on Windows/Linux
    cmd = [
        sys.executable, "scripts/deploy_bare_metal.py",
        "--config", config_file,
        "--fresh_hpo",
        "--upload_data",
        "--data_file", "btc_lob_jan2023.parquet",
        "--extra_args", "--tags Ralph_Autonomous" 
    ]
    
    # Execute locally to trigger remote deployment
    print(f"Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print("Deployment command sent successfully.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Deployment failed with exit code {e.returncode}")

def monitor_loop():
    """Poll for completion."""
    print("Starting monitoring loop...")
    start_time = time.time()
    last_log_time = 0
    
    while time.time() - start_time < MAX_WAIT_TIME:
        # 1. Notion Control
        try:
            cmd = check_command()
            if cmd == "Pause":
                print("PAUSED by Notion.")
                update_mission_control("Paused", 0, "Paused by user")
                time.sleep(60)
                continue
            elif cmd == "Stop":
                print("STOPPED by Notion.")
                update_mission_control("Stopped", 0, "Stopped by user")
                sys.exit(0)
        except Exception as e:
            print(f"Notion check warning: {e}")

        # 2. WandB Status
        try:
            metrics = fetch_latest_run_metrics(tag="Ralph_Autonomous")
            state = metrics.get('state', 'unknown')
            run_id = metrics.get('run_id', 'unknown')
            val_sharpe = metrics.get('validation_sharpe', 0) or 0
            
            # Log periodically
            if time.time() - last_log_time >= LOG_INTERVAL or state in ['finished', 'crashed', 'failed']:
                log_msg = f"Run {run_id}: {state} | ValSharpe: {val_sharpe:.4f}"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {log_msg}")
                update_mission_control("Running", val_sharpe, log_msg)
                last_log_time = time.time()
                
            if state in ['finished', 'crashed', 'failed']:
                return state, run_id, metrics
                
        except Exception as e:
            print(f"WandB poll error: {e}")
            
        time.sleep(POLL_INTERVAL)
        
    return "timeout", None, {}

def diagnose_and_fix(state, metrics):
    """Analyze failure and apply fixes."""
    iteration = state.get('iteration', 0)
    run_state = metrics.get('state', 'unknown')
    test_sharpe = metrics.get('test_sharpe', 0) or 0
    val_sharpe = metrics.get('validation_sharpe', 0) or 0
    goal_metrics = state.get('goal_criteria', {'min_test_sharpe': 1.0})
    
    print(f"Diagnosing failure (Iter {iteration}). State: {run_state}, Sharpe: {test_sharpe}")
    
    # Retrieve Logs (Tail)
    try:
        print("Fetching remote logs...")
        logs = remote_cmd("tail -n 50 /workspace/DeepScalper/logs/latest.log")
    except:
        logs = "Could not fetch logs."
        
    fix_applied = None
    
    # Logic Tree & Fix Application
    new_config_path = None
    
    # Load current config to modify
    current_config_path = state.get('config_file', DEFAULT_CONFIG)
    try:
        with open(current_config_path, 'r') as f:
            config_data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config {current_config_path}: {e}")
        config_data = {} # Should probably fail hard here, but let's try to proceed or handle gracefully

    # 1. OOM / CUDA Memory
    if "CUDA out of memory" in logs or "OOM" in logs:
        fix_applied = "Reduced num_envs (OOM detected)"
        
        # Action: Reduce num_envs
        current_envs = config_data.get('env', {}).get('num_envs', 12)
        new_envs = max(1, current_envs - 4)
        if 'env' not in config_data: config_data['env'] = {}
        config_data['env']['num_envs'] = new_envs
        
    # 2. Low Sharpe (Underperformance)
    elif run_state == 'finished' and test_sharpe < goal_metrics['min_test_sharpe']:
        # Check history to avoid cycles
        fixes = state.get('fixes_applied', [])
        lr_fixes = sum(1 for f in fixes if 'learning rate' in f['fix'].lower())
        
        if lr_fixes < 2:
             fix_applied = "Adjust HPO: Shift Learning Rate Range"
             # Action: Shift LR range down slightly to exploit
             if 'agents' not in config_data: config_data['agents'] = {'bdq': {}}
             # Simplified: just reduce LR for training phase
             current_lr = config_data.get('agents', {}).get('bdq', {}).get('learning_rate', 0.0001)
             config_data['agents']['bdq']['learning_rate'] = current_lr * 0.5
        else:
             fix_applied = "Increase Batch Size (Stability)"
             current_bs = config_data.get('agents', {}).get('bdq', {}).get('batch_size', 64)
             config_data['agents']['bdq']['batch_size'] = current_bs * 2
             
    # 3. Crash/NaN
    elif run_state in ['crashed', 'failed']:
        if "NaN" in logs:
             fix_applied = "Reduce Learning Rate (NaN detected)"
             current_lr = config_data.get('agents', {}).get('bdq', {}).get('learning_rate', 0.0001)
             if 'agents' not in config_data: config_data['agents'] = {'bdq': {}}
             config_data['agents']['bdq']['learning_rate'] = current_lr * 0.1
        else:
             fix_applied = "Retry (Transient Error or Unknown)"
             # No config change, just retry
             
    if not fix_applied:
        fix_applied = "Retry (General Failure)"

    # Save new config if changed
    if fix_applied and "Retry" not in fix_applied:
        os.makedirs('configs/autogen', exist_ok=True)
        new_config_path = f"configs/autogen/ralph_iter_{iteration+1}.yaml"
        with open(new_config_path, 'w') as f:
            yaml.dump(config_data, f)
        print(f"Generated new config: {new_config_path}")
        state['config_file'] = new_config_path

    # Log fix
    state['fixes_applied'].append({
        'timestamp': datetime.now().isoformat(),
        'iteration': iteration,
        'fix': fix_applied,
        'reason': f"State: {run_state}, TestSharpe: {test_sharpe}",
        'new_config': new_config_path
    })
    
    print(f"Applying Fix: {fix_applied}")
    update_mission_control("Running", 0, f"Applied Fix: {fix_applied}")
    
    return fix_applied

def main():
    print(">>> Starting Ralph Autonomous Driver")
    state = init_state()
    
    try:
        # Main Loop
        while not state['goal_met'] and state['iteration'] < state['max_iterations']:
            
            # Phase 1: Check/Deploy
            active, run_id = check_active_run()
            
            if not active:
                # Increment Iteration if we are starting fresh
                state['iteration'] += 1
                save_state(state)
                
                print(f"\n=== Starting Iteration {state['iteration']} ===")
                deploy(state['config_file'])
                
                # Wait for startup
                time.sleep(60)
                
            else:
                print(f"Resuming monitoring for Run {run_id}...")
                
            # Phase 2: Monitor
            run_state, run_id, metrics = monitor_loop()
            
            if run_state == "timeout":
                print("Run timed out. Aborting.")
                raise TimeoutError("Run monitoring timed out")
                
            # Phase 3: Verify
            print("Run finished. Verifying...")
            
            # Generate Report
            try:
                fetch_run_data(run_id)
                os.makedirs('.agent/reports', exist_ok=True)
                report_path = f".agent/reports/ralph_iter_{state['iteration']}_{run_id}.md"
                generate_report(run_id, report_path)
            except Exception as e:
                print(f"Report generation failed: {e}")
                
            test_sharpe = metrics.get('test_sharpe', 0) or 0
            val_sharpe = metrics.get('validation_sharpe', 0) or 0
            
            success = (run_state == 'finished' and test_sharpe >= state['goal_criteria']['min_test_sharpe'])
            
            # Log History
            hist_entry = {
                'timestamp': datetime.now().isoformat(),
                'iteration': state['iteration'],
                'run_id': run_id,
                'outcome': 'success' if success else 'failure',
                'metrics': {'val': val_sharpe, 'test': test_sharpe, 'state': run_state}
            }
            state['history'].append(hist_entry)
            save_state(state)
            
            if success:
                print(f"GOAL MET! Test Sharpe: {test_sharpe}")
                state['goal_met'] = True
                save_state(state)
                update_mission_control("Stopped", test_sharpe, "Goal Met! Success.")
                
                # Copy success report
                if os.path.exists(report_path):
                    shutil.copy(report_path, '.agent/ralph_success_report.md')
                break
            else:
                print(f"Goal failed. State: {run_state}, Test Sharpe: {test_sharpe}")
                
                # Phase 4: Fix
                diagnose_and_fix(state, metrics)
                save_state(state)
                
                print("Retrying loop...\n")
                time.sleep(10)

        if not state['goal_met']:
            print("Ralph terminated: Max iterations reached.")
            raise RuntimeError("Max iterations reached without success")

    except KeyboardInterrupt:
        print("\nCTRL+C Detected. Stopping...")
        update_mission_control("Stopped", 0, "Interrupt by User")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        update_mission_control("Stopped", 0, f"Error: {str(e)}")
        raise e
        
    finally:
        print("Exiting Ralph Driver.")


if __name__ == "__main__":
    main()
