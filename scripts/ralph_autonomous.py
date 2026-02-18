#!/usr/bin/env python3
"""
Ralph Autonomous Driver — V3 Roadmap Orchestrator
===================================================
Multi-phase roadmap execution with decision gates.

Modes:
  --roadmap            Read ralph_roadmap.yaml. Execute phases sequentially.
  --roadmap --dry-run  Parse roadmap, print plan, exit without executing.
  (no flags)           Legacy single-config retry loop.
"""

import sys
import os
import shutil
import tempfile
import time
import json
import shutil
import argparse
import subprocess
import yaml
import wandb
from datetime import datetime
from copy import deepcopy

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.fetch_wandb_run import fetch_latest_run_metrics, poll_run_until_complete, fetch_run_data
from scripts.remote_cmd import remote_cmd
from scripts.generate_report import generate_report

# Try optional imports — Notion sync may not be needed on all machines
try:
    from scripts.notion_sync import update_mission_control, check_command, log_run
except ImportError:
    def update_mission_control(*a, **kw): pass
    def check_command(): return None
    def log_run(*a, **kw): pass

# ── Constants ────────────────────────────────────────────────────────────
STATE_FILE = '.agent/ralph_state.json'
ROADMAP_FILE = '.agent/ralph_roadmap.yaml'
DEFAULT_CONFIG = 'configs/deepscalper_rtx5090_production.yaml'
WANDB_ENTITY = "bigcan-chiwin-technology"
WANDB_PROJECT = "FinRL-Pro-DS"
POLL_INTERVAL = 60       # Check status every minute (legacy mode)
POLL_INTERVAL_ROAD = 300 # Check status every 5 min (roadmap mode — runs take hours) 
LOG_INTERVAL = 900       # Update Notion/logs every 15 min
MAX_WAIT_TIME = 28800    # 8 hours max per run
DATA_FILE = "btc_2025_jan_jun.parquet"  # Default data file for V5-era configs

# ── Logging ──────────────────────────────────────────────────────────────

def rlog(msg, level="INFO"):
    """Timestamped log line."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    prefix = {"INFO": "ℹ️ ", "OK": "✅", "WARN": "⚠️ ", "ERR": "❌", "PHASE": "🔶"}.get(level, "  ")
    line = f"[{ts}] {prefix} {msg}"
    print(line)
    # Also append to file log
    try:
        os.makedirs('.agent/logs', exist_ok=True)
        with open(f'.agent/logs/ralph_{datetime.now().strftime("%Y%m%d")}.log', 'a') as f:
            f.write(line + "\n")
    except:
        pass

# ═══════════════════════════════════════════════════════════════════════
# SECTION 1 — Legacy Functions (preserved for backward compatibility)
# ═══════════════════════════════════════════════════════════════════════

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
    
    if wandb_state == 'running':
        print(f"Active WandB run detected (State: {wandb_state}, Physical: {is_running_physically})")
        if is_running_physically:
            return True, run_id
        else:
            print("WARNING: WandB says running but no physical process found. Treating as Zombie/Crashed.")
            return False, None
    
    if is_running_physically:
        print(f"Stale physical process detected (WandB: {wandb_state}). Killing and deploying fresh.")
        try:
            remote_cmd("pkill -f 'run_full_pipeline.py' || true", timeout=15)
        except:
            pass
    
    return False, None

def deploy_legacy(config_file, fresh_hpo=True):
    """Deploy the pipeline to bare metal (legacy mode)."""
    print(f"Deploying {config_file}...")
    update_mission_control("Running", 0, "Deploying new run...")
    # Load config to get tags and data file
    with open(config_file, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # Extract data file from config, defaulting to the standard one
    data_file = cfg.get('data', {}).get('file_path', 'btc_2025_jan_jun.parquet')
    
    # Build command
    cmd = [
        "python", "scripts/deploy_bare_metal.py",
        "--config", config_file,
        "--data_file", data_file,  # Use data file from config (RALPH-06)
        "--fresh_hpo" if fresh_hpo else "--resume_hpo",  # logic for resuming? (TODO)
        "--upload_data",  # Always ensure data is there
        "--no_kill",      # CRITICAL: Do not kill other runs (RALPH-02)
        "--extra_args", "--tags Ralph_Autonomous" 
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print("Deployment command sent successfully.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Deployment failed with exit code {e.returncode}")

def monitor_loop_legacy():
    """Poll for completion (legacy mode)."""
    print("Starting monitoring loop...")
    start_time = time.time()
    last_log_time = 0
    
    while time.time() - start_time < MAX_WAIT_TIME:
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

        try:
            metrics = fetch_latest_run_metrics(tag="Ralph_Autonomous")
            state = metrics.get('state', 'unknown')
            run_id = metrics.get('run_id', 'unknown')
            val_sharpe = metrics.get('validation_sharpe', 0) or 0
            
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
    """Analyze failure and apply fixes (legacy mode)."""
    iteration = state.get('iteration', 0)
    run_state = metrics.get('state', 'unknown')
    test_sharpe = metrics.get('test_sharpe', 0) or 0
    val_sharpe = metrics.get('validation_sharpe', 0) or 0
    goal_metrics = state.get('goal_criteria', {'min_test_sharpe': 1.0})
    
    print(f"Diagnosing failure (Iter {iteration}). State: {run_state}, Sharpe: {test_sharpe}")
    
    try:
        print("Fetching remote logs...")
        logs = remote_cmd("tail -n 50 /workspace/DeepScalper/logs/latest.log")
    except:
        logs = "Could not fetch logs."
        
    fix_applied = None
    new_config_path = None
    
    current_config_path = state.get('config_file', DEFAULT_CONFIG)
    try:
        with open(current_config_path, 'r') as f:
            config_data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config {current_config_path}: {e}")
        config_data = {}

    if "CUDA out of memory" in logs or "OOM" in logs:
        fix_applied = "Reduced num_envs (OOM detected)"
        current_envs = config_data.get('env', {}).get('num_envs', 12)
        new_envs = max(1, current_envs - 4)
        if 'env' not in config_data: config_data['env'] = {}
        config_data['env']['num_envs'] = new_envs
        
    elif run_state == 'finished' and test_sharpe < goal_metrics['min_test_sharpe']:
        fixes = state.get('fixes_applied', [])
        lr_fixes = sum(1 for f in fixes if 'learning rate' in f['fix'].lower())
        
        if lr_fixes < 2:
             fix_applied = "Adjust HPO: Shift Learning Rate Range"
             if 'agents' not in config_data: config_data['agents'] = {'bdq': {}}
             current_lr = config_data.get('agents', {}).get('bdq', {}).get('learning_rate', 0.0001)
             config_data['agents']['bdq']['learning_rate'] = current_lr * 0.5
        else:
             fix_applied = "Increase Batch Size (Stability)"
             current_bs = config_data.get('agents', {}).get('bdq', {}).get('batch_size', 64)
             config_data['agents']['bdq']['batch_size'] = current_bs * 2
             
    elif run_state in ['crashed', 'failed']:
        if "NaN" in logs:
             fix_applied = "Reduce Learning Rate (NaN detected)"
             current_lr = config_data.get('agents', {}).get('bdq', {}).get('learning_rate', 0.0001)
             if 'agents' not in config_data: config_data['agents'] = {'bdq': {}}
             config_data['agents']['bdq']['learning_rate'] = current_lr * 0.1
        else:
             fix_applied = "Retry (Transient Error or Unknown)"
             
    if not fix_applied:
        fix_applied = "Retry (General Failure)"

    if fix_applied and "Retry" not in fix_applied:
        os.makedirs('configs/autogen', exist_ok=True)
        new_config_path = f"configs/autogen/ralph_iter_{iteration+1}.yaml"
        with open(new_config_path, 'w') as f:
            yaml.dump(config_data, f)
        print(f"Generated new config: {new_config_path}")
        state['config_file'] = new_config_path

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


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2 — Roadmap Orchestrator (V3)
# ═══════════════════════════════════════════════════════════════════════

def load_roadmap():
    """Load the ralph_roadmap.yaml file."""
    if not os.path.exists(ROADMAP_FILE):
        raise FileNotFoundError(f"Roadmap not found: {ROADMAP_FILE}")
    with open(ROADMAP_FILE, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_roadmap(roadmap):
    """Save the roadmap state to YAML (atomic write)."""
    # Create temp file in same directory to ensure atomic move works
    dir_name = os.path.dirname(ROADMAP_FILE)
    # Windows: NamedTemporaryFile can't be opened twice, so we write and close.
    # We use delete=False so we can move it.
    with tempfile.NamedTemporaryFile('w', delete=False, dir=dir_name, encoding='utf-8', suffix='.tmp') as tf:
        yaml.dump(roadmap, tf, default_flow_style=False, sort_keys=False, width=120, allow_unicode=True)
        temp_name = tf.name
    
    # Atomic replace
    try:
        shutil.move(temp_name, ROADMAP_FILE)
        rlog(f"Roadmap updated → {ROADMAP_FILE}")
    except Exception as e:
        rlog(f"Failed to save roadmap: {e}", "ERR")
        if os.path.exists(temp_name):
            os.remove(temp_name)

def get_phase(roadmap, phase_name):
    """Get a phase definition by name."""
    phases = roadmap.get('phases', {})
    if phase_name not in phases:
        raise KeyError(f"Phase '{phase_name}' not found in roadmap. Available: {list(phases.keys())}")
    return phases[phase_name]

def find_runs_by_config(config_path):
    """Find WandB runs matching a config's tags. Returns list of (run_id, state, summary_dict)."""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    tags = cfg.get('wandb', {}).get('tags', [])
    
    if not tags:
        rlog(f"Config {config_path} has no WandB tags — cannot match runs", "WARN")
        return []
    
    api = wandb.Api()
    # Match ALL tags from the config
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}",
                    filters={"tags": {"$all": tags}},
                    order="-created_at", per_page=5)
    
    results = []
    for r in runs:
        results.append((r.id, r.state, r.summary._json_dict))
    
    rlog(f"Found {len(results)} WandB run(s) matching tags {tags}")
    return results

def fetch_run_metrics(run_id):
    """Fetch metrics for a specific run by ID."""
    api = wandb.Api()
    run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
    summary = run.summary._json_dict
    
    return {
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "url": run.url,
        "total_steps": summary.get("train/global_step", 0),
        # Backtest metrics
        "validation_sharpe": summary.get("backtest_val/sharpe", summary.get("backtest_validation/sharpe")),
        "test_sharpe": summary.get("backtest_test/sharpe", summary.get("backtest/test_sharpe")),
        "validation_return": summary.get("backtest_val/total_return"),
        "test_return": summary.get("backtest_test/total_return"),
        "validation_profit_factor": summary.get("backtest_val/profit_factor"),
        "test_profit_factor": summary.get("backtest_test/profit_factor"),
        "validation_max_drawdown": summary.get("backtest_val/max_drawdown"),
        "test_max_drawdown": summary.get("backtest_test/max_drawdown"),
        "test_sortino": summary.get("backtest_test/sortino"),
        "test_omega": summary.get("backtest_test/omega"),
        "test_win_rate": summary.get("backtest_test/win_rate"),
    }

def wait_for_runs(run_ids, poll_interval=None, max_wait=None):
    """
    Poll multiple WandB runs until ALL are terminal (finished/crashed/failed).
    
    Returns:
        dict mapping run_id → final metrics dict
    """
    if poll_interval is None:
        poll_interval = POLL_INTERVAL_ROAD
    if max_wait is None:
        max_wait = MAX_WAIT_TIME
        
    api = wandb.Api()
    pending = set(run_ids)
    completed = {}
    start = time.time()
    
    rlog(f"Waiting for {len(pending)} run(s): {pending}")
    
    while pending and (time.time() - start < max_wait):
        for rid in list(pending):
            try:
                run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{rid}")
                state = run.state
                
                if state in ('finished', 'crashed', 'failed'):
                    rlog(f"Run {rid}: {state}", "OK" if state == "finished" else "WARN")
                    completed[rid] = fetch_run_metrics(rid)
                    pending.discard(rid)
                else:
                    # RALPH-04: Zombie Check
                    # Verify if process is actually running on remote
                    try:
                        # We need the PID to check specifically, but finding specific run PIDs 
                        # remotely without a mapping is hard. 
                        # However, we can check if *any* python process matches the run config?
                        # Simpler: If WandB says running but last heartbeat > 5 min, suspect zombie.
                        # Even better: remote_cmd("pgrep -f run_full_pipeline.py") 
                        # If NO pipeline is running, but this run is 'running', it's a zombie.
                        if len(pending) == 1: # Only accurate if we are tracking 1 run or we know they share the machine
                             pids = remote_cmd("pgrep -f run_full_pipeline.py || echo ''", timeout=10).strip()
                             if not pids:
                                 rlog(f"Run {rid}: WandB='running' but NO physical process found! Marking CRASHED.", "ERR")
                                 completed[rid] = {"run_id": rid, "state": "crashed", "error": "Zombie process detected"}
                                 pending.discard(rid)
                                 continue
                    except Exception as z_err:
                        rlog(f"Zombie check failed: {z_err}")

                    # Log progress
                    steps = run.summary._json_dict.get("train/global_step", "?")
                    rlog(f"Run {rid}: {state} (steps: {steps})")
            except Exception as e:
                rlog(f"Error polling {rid}: {e}", "WARN")
        
        if pending:
            elapsed = int(time.time() - start)
            rlog(f"Still waiting for {len(pending)} run(s). Elapsed: {elapsed//3600}h{(elapsed%3600)//60}m")
            update_mission_control("Running", 0, 
                f"Monitoring {len(pending)} run(s). Elapsed: {elapsed//3600}h{(elapsed%3600)//60}m")
            time.sleep(poll_interval)
    
    if pending:
        rlog(f"TIMEOUT: {len(pending)} run(s) did not complete: {pending}", "ERR")
        for rid in pending:
            try:
                completed[rid] = fetch_run_metrics(rid)
                completed[rid]['state'] = 'timeout'
            except:
                completed[rid] = {"run_id": rid, "state": "timeout"}
    
    return completed

def collect_and_report(run_id, phase_label=""):
    """Post-run collection: fetch data, download artifacts, generate report."""
    from scripts.collect_run import collect_run
    
    rlog(f"Collecting run {run_id} [{phase_label}]...")
    try:
        result = collect_run(run_id, skip_remote=False, skip_report=False)
        rlog(f"Collection complete for {run_id}", "OK")
        return result
    except Exception as e:
        rlog(f"Collection failed for {run_id}: {e}", "ERR")
        # Still return partial results
        return {"run_id": run_id, "error": str(e)}

def deploy_config(config_path, fresh_hpo=True, extra_tags=None):
    """Deploy a config via deploy_bare_metal.py and return resolved run ID."""
    rlog(f"Deploying config: {config_path}", "PHASE")
    
    # Load config to get tags and data file
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # Extract data file from config, defaulting to the standard one
    data_file = cfg.get('data', {}).get('file_path', 'btc_2025_jan_jun.parquet')
    
    cmd = [
        sys.executable, "scripts/deploy_bare_metal.py",
        "--config", config_path,
        "--data_file", data_file,  # RALPH-06: Use data file from config
        "--fresh_hpo" if fresh_hpo else "--resume_hpo",
        "--upload_data",
        "--no_kill",   # RALPH-02: Do not kill concurrent runs
    ]
    
    # Build tags
    tags_parts = ["Ralph_Roadmap"]
    if extra_tags:
        tags_parts.extend(extra_tags)
    cmd.extend(["--extra_args", f"--tags {' '.join(tags_parts)}"])
    
    rlog(f"Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        rlog("Deployment command succeeded", "OK")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Deployment failed with exit code {e.returncode}")
    
    # Wait for WandB to register the new run
    rlog("Waiting 30s for WandB registration...")
    time.sleep(30)
    
    # RALPH-03: Read definitively resolved Run ID from remote file
    # This avoids the race condition of polling WandB tags
    rlog("Checking remote run_id.txt for resolved ID...")
    try:
        run_id_out = remote_cmd(f"cat /workspace/DeepScalper/run_id.txt", timeout=10)
        run_id_clean = run_id_out.strip()
        if run_id_clean and len(run_id_clean) == 8:
            new_run_id = run_id_clean
            rlog(f"Resolved new run from file: {new_run_id}", "OK")
            return new_run_id
    except Exception as e:
        rlog(f"Could not read run_id.txt: {e}", "WARN")

    # Fallback to polling if file read fails
    metrics = fetch_latest_run_metrics(tag="Ralph_Roadmap")
    new_run_id = metrics.get('run_id')
    if new_run_id:
        rlog(f"Resolved new run (fallback poll): {new_run_id}", "OK")
    else:
        rlog("Could not resolve new run ID from WandB", "WARN")
    
    return new_run_id

def evaluate_decision_gate(phase, results):
    """
    Evaluate a decision gate matrix against collected results.
    
    Returns:
        (next_phase_name, condition_name, diagnosis)
    """
    matrix = phase.get('matrix', {})
    
    rlog(f"Evaluating decision gate: {phase.get('label', '?')}", "PHASE")
    rlog(f"Results available: {list(results.keys())}")
    
    # ── Phase 3c decision gate (BDQ both_profit / fees_only_fail / both_fail) ──
    if 'both_profit' in matrix and 'fees_only_fail' in matrix:
        # Extract profit factors from stored results
        fees_results = results.get('bdq_fees', {})
        zerofee_results = results.get('bdq_zerofee', {})
        
        fees_pf = fees_results.get('test_profit_factor') or 0
        zerofee_pf = zerofee_results.get('test_profit_factor') or 0
        fees_sharpe = fees_results.get('test_sharpe') or 0
        zerofee_sharpe = zerofee_results.get('test_sharpe') or 0
        
        rlog(f"BDQ Fees:     PF={fees_pf:.3f}, Sharpe={fees_sharpe:.3f}")
        rlog(f"BDQ Zero-Fee: PF={zerofee_pf:.3f}, Sharpe={zerofee_sharpe:.3f}")
        
        if fees_pf > 1.0 and zerofee_pf > 1.0 and fees_sharpe > 0.0 and zerofee_sharpe > 0.0:
            entry = matrix['both_profit']
            rlog(f"Decision: BOTH_PROFIT → {entry['next']}", "OK")
            return entry['next'], 'both_profit', entry.get('diagnosis', '')
        
        elif zerofee_pf > 1.0 and fees_pf < 1.0:
            entry = matrix['fees_only_fail']
            rlog(f"Decision: FEES_ONLY_FAIL → {entry['next']}", "WARN")
            return entry['next'], 'fees_only_fail', entry.get('diagnosis', '')
        
        else:
            entry = matrix['both_fail']
            rlog(f"Decision: BOTH_FAIL → {entry['next']}", "ERR")
            return entry['next'], 'both_fail', entry.get('diagnosis', '')
    
    # ── Generic evaluate_results gate (success / failure) ──
    elif 'success' in matrix and 'failure' in matrix:
        # Check the most recent run's results against the parent phase's success criteria
        latest_results = results.get('_latest', {})
        test_pf = latest_results.get('test_profit_factor') or 0
        test_sharpe = latest_results.get('test_sharpe') or 0
        
        rlog(f"Latest run: PF={test_pf:.3f}, Sharpe={test_sharpe:.3f}")
        
        if test_pf > 1.0 and test_sharpe > 0.5:
            entry = matrix['success']
            rlog(f"Decision: SUCCESS → {entry['next']}", "OK")
            return entry['next'], 'success', entry.get('diagnosis', '')
        else:
            entry = matrix['failure']
            rlog(f"Decision: FAILURE → {entry['next']}", "ERR")
            return entry['next'], 'failure', entry.get('diagnosis', '')
    
    else:
        raise ValueError(f"Unknown decision gate structure: {list(matrix.keys())}")

def advance_phase(roadmap, next_phase, phase_results=None):
    """Advance the roadmap to the next phase, storing results."""
    old_phase = roadmap['current_phase']
    
    # Store results for completed phase
    if phase_results:
        if roadmap.get('results') is None:
            roadmap['results'] = {}
        roadmap['results'][old_phase] = {
            'completed_at': datetime.now().isoformat(),
            **phase_results
        }
    
    roadmap['current_phase'] = next_phase
    save_roadmap(roadmap)
    rlog(f"Phase advanced: {old_phase} → {next_phase}", "PHASE")

def print_roadmap_plan(roadmap):
    """Pretty-print the roadmap for dry-run mode."""
    print("\n" + "=" * 70)
    print(f"  RALPH V3 ROADMAP: {roadmap.get('roadmap', '?')}")
    print(f"  Current Phase: {roadmap.get('current_phase', '?')}")
    print(f"  Started: {roadmap.get('started_at', '?')}")
    print("=" * 70)
    
    phases = roadmap.get('phases', {})
    current = roadmap.get('current_phase', '')
    
    for name, phase in phases.items():
        marker = "→ " if name == current else "  "
        ptype = phase.get('type', 'run')
        label = phase.get('label', name)
        
        if ptype == 'decision_gate':
            icon = "🔀"
        elif ptype == 'terminal':
            icon = "🏁"
        elif phase.get('deploy', False):
            icon = "🚀"
        else:
            icon = "👀"
        
        status = ""
        if name in roadmap.get('results', {}):
            status = " [DONE]"
        elif name == current:
            status = " [ACTIVE]"
        
        print(f"  {marker}{icon} {label}{status}")
        
        if ptype == 'decision_gate':
            matrix = phase.get('matrix', {})
            for cond, info in matrix.items():
                print(f"       ├─ {cond}: → {info.get('next', '?')}")
        elif phase.get('on_complete'):
            print(f"       └─ on_complete: {phase['on_complete']}")
    
    print("=" * 70 + "\n")

# ── Phase Handlers ───────────────────────────────────────────────────

def handle_monitor_phase(roadmap, phase_name, phase):
    """Handle a multi-run monitoring phase (e.g. phase_3c_monitor_both)."""
    rlog(f"Phase: {phase.get('label', phase_name)}", "PHASE")
    
    runs_config = phase.get('runs', {})
    run_ids = {}
    
    # Step 1: Discover run IDs from WandB tags
    for run_key, run_info in runs_config.items():
        config_path = run_info.get('config')
        
        # Check if we already have a stored run_id
        stored_id = run_info.get('run_id')
        if stored_id:
            rlog(f"Using stored run_id for {run_key}: {stored_id}")
            run_ids[run_key] = stored_id
            continue
        
        # Discover from WandB
        matches = find_runs_by_config(config_path)
        if matches:
            # Take the most recent run
            rid, state, _ = matches[0]
            run_ids[run_key] = rid
            
            # Persist discovered ID back into roadmap
            roadmap['phases'][phase_name]['runs'][run_key]['run_id'] = rid
            save_roadmap(roadmap)
            
            rlog(f"Discovered {run_key}: {rid} (state: {state})", "OK")
        else:
            rlog(f"No WandB run found for {run_key} (config: {config_path})", "ERR")
    
    if not run_ids:
        raise RuntimeError(f"No runs found for phase {phase_name}. Cannot proceed.")
    
    # Step 2: Wait for all runs to complete
    completed = wait_for_runs(list(run_ids.values()))
    
    # Step 3: Collect results for each run
    phase_results = {}
    for run_key, rid in run_ids.items():
        metrics = completed.get(rid, {})
        
        # Collect data and generate report
        collect_and_report(rid, run_key)
        
        phase_results[run_key] = metrics
        rlog(f"{run_key} ({rid}): state={metrics.get('state', '?')}, "
             f"PF={metrics.get('test_profit_factor', '?')}, "
             f"Sharpe={metrics.get('test_sharpe', '?')}")
    
    # Step 4: Advance to next phase
    next_phase = phase.get('on_complete')
    if next_phase:
        advance_phase(roadmap, next_phase, phase_results)
    
    return phase_results

def handle_deploy_phase(roadmap, phase_name, phase):
    """Handle a deploy-and-monitor phase."""
    rlog(f"Phase: {phase.get('label', phase_name)}", "PHASE")
    
    config_path = phase.get('config')
    if not config_path:
        raise ValueError(f"Phase {phase_name} has deploy=true but no config path")
    
    fresh_hpo = phase.get('fresh_hpo', True)
    
    # Deploy
    run_id = deploy_config(config_path, fresh_hpo=fresh_hpo, extra_tags=[phase_name])
    
    if not run_id:
        raise RuntimeError(f"Deploy failed — could not resolve run ID for {config_path}")
    
    # Store run_id in roadmap
    roadmap['phases'][phase_name]['run_id'] = run_id
    save_roadmap(roadmap)
    
    # Wait for completion
    completed = wait_for_runs([run_id])
    metrics = completed.get(run_id, {})
    
    # Collect and report
    collect_and_report(run_id, phase.get('label', phase_name))
    
    # Check success criteria
    success_criteria = phase.get('success', {})
    min_sharpe = success_criteria.get('min_test_sharpe', 0)
    min_pf = success_criteria.get('min_profit_factor', 0)
    
    test_sharpe = metrics.get('test_sharpe') or 0
    test_pf = metrics.get('test_profit_factor') or 0
    
    met_criteria = (test_sharpe >= min_sharpe and test_pf >= min_pf)
    
    rlog(f"Success criteria: Sharpe≥{min_sharpe} ({test_sharpe:.3f}), PF≥{min_pf} ({test_pf:.3f})")
    rlog(f"Criteria met: {met_criteria}", "OK" if met_criteria else "WARN")
    
    # Store results including _latest for downstream decision gates
    phase_results = {
        'run_id': run_id,
        'met_criteria': met_criteria,
        **metrics
    }
    
    # Also store as _latest for generic decision gates
    if roadmap.get('results') is None:
        roadmap['results'] = {}
    roadmap['results']['_latest'] = metrics
    
    # Advance
    next_phase = phase.get('on_complete')
    if next_phase:
        advance_phase(roadmap, next_phase, phase_results)
    
    return phase_results

def handle_decision_gate(roadmap, phase_name, phase):
    """Handle a decision gate phase."""
    rlog(f"Decision Gate: {phase.get('label', phase_name)}", "PHASE")
    
    # Execute always_do actions (generate reports etc)
    always_do = phase.get('always_do', '')
    if always_do:
        rlog(f"Always-do: {always_do}")
    
    # Gather all results collected so far
    results = roadmap.get('results', {})
    
    # For phase_3c_decision, flatten the monitor_both results
    monitor_results = results.get('phase_3c_monitor_both', {})
    # Merge sub-run results to top level for the gate evaluator
    gate_results = {}
    for key, val in monitor_results.items():
        if isinstance(val, dict) and 'run_id' in val:
            gate_results[key] = val
    # Also include _latest if available
    if '_latest' in results:
        gate_results['_latest'] = results['_latest']
    
    next_phase, condition, diagnosis = evaluate_decision_gate(phase, gate_results)
    
    # Store gate decision
    gate_result = {
        'condition': condition,
        'diagnosis': diagnosis,
        'next_phase': next_phase,
        'evaluated_at': datetime.now().isoformat()
    }
    
    advance_phase(roadmap, next_phase, gate_result)
    return gate_result

def handle_passthrough_phase(roadmap, phase_name, phase):
    """Handle a passthrough phase — log and immediately advance."""
    rlog(f"Passthrough: {phase.get('label', phase_name)}", "PHASE")
    rlog(f"Action: {phase.get('action', '')}")
    
    next_phase = phase.get('on_complete')
    if next_phase:
        advance_phase(roadmap, next_phase, {"passthrough": True, "action": phase.get('action', '')})
    else:
        rlog(f"Passthrough phase {phase_name} has no on_complete — stopping", "WARN")

def handle_terminal_phase(roadmap, phase_name, phase):
    """Handle a terminal phase (end of roadmap)."""
    rlog(f"TERMINAL: {phase.get('label', phase_name)}", "PHASE")
    rlog(f"Action: {phase.get('action', 'No action specified')}")
    
    # Mark roadmap as complete
    roadmap['completed_at'] = datetime.now().isoformat()
    roadmap['outcome'] = phase.get('label', phase_name)
    save_roadmap(roadmap)
    
    update_mission_control("Stopped", 0, f"Roadmap complete: {phase.get('label', phase_name)}")
    
    return {"terminal": True, "label": phase.get('label')}

# ── Main Orchestration Loop ──────────────────────────────────────────

def roadmap_main(dry_run=False):
    """Main roadmap orchestration entry point."""
    roadmap = load_roadmap()
    
    if dry_run:
        print_roadmap_plan(roadmap)
        return
    
    rlog(f"Ralph V3 Roadmap Orchestrator starting", "PHASE")
    rlog(f"Roadmap: {roadmap.get('roadmap', '?')}")
    rlog(f"Current phase: {roadmap.get('current_phase', '?')}")
    print_roadmap_plan(roadmap)
    
    update_mission_control("Running", 0, f"Roadmap: {roadmap.get('roadmap')} | Phase: {roadmap.get('current_phase')}")
    
    max_phases = 20  # Safety: prevent infinite loops
    phases_executed = 0
    
    while phases_executed < max_phases:
        phase_name = roadmap.get('current_phase')
        if not phase_name:
            rlog("No current_phase set. Roadmap complete.", "OK")
            break
        
        phase = get_phase(roadmap, phase_name)
        phase_type = phase.get('type', 'run')
        
        rlog(f"{'='*60}")
        rlog(f"Executing: {phase.get('label', phase_name)} (type: {phase_type})")
        rlog(f"{'='*60}")
        
        try:
            if phase_type == 'terminal':
                handle_terminal_phase(roadmap, phase_name, phase)
                break
            
            elif phase_type == 'passthrough':
                handle_passthrough_phase(roadmap, phase_name, phase)
            
            elif phase_type == 'decision_gate':
                handle_decision_gate(roadmap, phase_name, phase)
            
            elif phase.get('adopt_latest') or (phase.get('runs') and not phase.get('deploy')):
                # Multi-run monitor phase
                handle_monitor_phase(roadmap, phase_name, phase)
            
            elif phase.get('deploy'):
                handle_deploy_phase(roadmap, phase_name, phase)
            
            else:
                rlog(f"Unknown phase type for {phase_name}. Skipping.", "ERR")
                next_p = phase.get('on_complete')
                if next_p:
                    advance_phase(roadmap, next_p, {"skipped": True})
                else:
                    break
            
            phases_executed += 1
            
            # Reload roadmap (it was saved by advance_phase)
            roadmap = load_roadmap()
            
        except KeyboardInterrupt:
            rlog("INTERRUPTED by user (Ctrl+C)", "WARN")
            update_mission_control("Stopped", 0, "Interrupted by user")
            break
            
        except Exception as e:
            rlog(f"PHASE ERROR: {e}", "ERR")
            import traceback
            traceback.print_exc()
            
            # Log error but try to continue if possible
            if roadmap.get('results') is None:
                roadmap['results'] = {}
            roadmap['results'][f'{phase_name}_error'] = {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            save_roadmap(roadmap)
            
            # Don't auto-advance on error — stop and let human decide
            rlog("Stopping due to error. Fix and restart to resume from current phase.", "ERR")
            update_mission_control("Stopped", 0, f"Error in {phase_name}: {str(e)[:100]}")
            break
    
    rlog(f"Ralph V3 finished. Phases executed: {phases_executed}", "PHASE")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3 — Legacy Main Loop
# ═══════════════════════════════════════════════════════════════════════

def main_legacy():
    print(">>> Starting Ralph Autonomous Driver (Legacy Mode)")
    state = init_state()
    
    try:
        while not state['goal_met'] and state['iteration'] < state['max_iterations']:
            
            active, run_id = check_active_run()
            
            if not active:
                state['iteration'] += 1
                save_state(state)
                
                print(f"\n=== Starting Iteration {state['iteration']} ===")
                deploy_legacy(state['config_file'])
                time.sleep(60)
                
            else:
                print(f"Resuming monitoring for Run {run_id}...")
                
            run_state, run_id, metrics = monitor_loop_legacy()
            
            if run_state == "timeout":
                print("Run timed out. Aborting.")
                raise TimeoutError("Run monitoring timed out")
                
            print("Run finished. Verifying...")
            
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
                
                if os.path.exists(report_path):
                    shutil.copy(report_path, '.agent/ralph_success_report.md')
                break
            else:
                print(f"Goal failed. State: {run_state}, Test Sharpe: {test_sharpe}")
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


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4 — CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Ralph Autonomous Driver — V3 Roadmap Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ralph_autonomous.py --roadmap              # Execute roadmap
  python scripts/ralph_autonomous.py --roadmap --dry-run    # Print plan only
  python scripts/ralph_autonomous.py                        # Legacy single-config mode
        """
    )
    parser.add_argument("--roadmap", action="store_true",
                        help="Run in roadmap mode (reads .agent/ralph_roadmap.yaml)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse roadmap and print plan without executing (requires --roadmap)")
    
    args = parser.parse_args()
    
    if args.roadmap or args.dry_run:
        roadmap_main(dry_run=args.dry_run)
    else:
        main_legacy()


if __name__ == "__main__":
    main()
