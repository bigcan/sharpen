#!/usr/bin/env python3
"""
DeepScalper Fleet Monitor
=========================
Unified status dashboard for all GPUHub instances and WandB runs.
Combines hardware metrics (GPU util, memory, processes) with WandB
training metrics (SPS, PF, Q-values, step count) into a single
table-based report.

Anomaly Detection:
  - Stalled run: no WandB step increment in 30+ min
  - Low SPS: steps/sec below threshold for GPU tier
  - Low GPU utilization: <30% on a GPU with an active run
  - Q-divergence: |q_mean| > 10,000
  - NaN/Inf loss
  - Process dead: WandB says "running" but no python process on host

Output: Table-formatted status report to stdout + results/fleet_monitor.log

Usage:
    python scripts/monitor_fleet.py              # Full fleet status
    python scripts/monitor_fleet.py --wandb-only  # WandB runs only (no SSH)
    python scripts/monitor_fleet.py --hw-only     # Hardware only (no WandB)
    python scripts/monitor_fleet.py --json        # JSON output
"""

import sys
import os
import json
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ─── Thresholds ───────────────────────────────────────────────────────────────
Q_DIVERGENCE_THRESHOLD = 1e4
STALL_MINUTES = 30
GPU_UTIL_LOW = 30          # % — warn if GPU < 30% with active run
GPU_MEM_HIGH = 95          # % — warn if GPU memory > 95%

# SPS thresholds by GPU tier and algorithm
# BDQ/IQN are env-step-bound (~500+ SPS); SAC is gradient-bound (~4 SPS)
SPS_THRESHOLDS = {
    "RTX 5090": {"default": 800, "sac": 2},
    "RTX 4090": {"default": 500, "sac": 2},
    "default":  {"default": 400, "sac": 2},
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INSTANCES_FILE = PROJECT_ROOT / "instances.json"
LOG_DIR = PROJECT_ROOT / "results"
LOG_FILE = LOG_DIR / "fleet_monitor.log"
WANDB_PROJECT = "bigcan-chiwin-technology/FinRL-Pro-DS"


# ─── Logging ──────────────────────────────────────────────────────────────────

def log(msg, level="INFO"):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] [{level}] {msg}"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


# ─── Instance Registry ───────────────────────────────────────────────────────

def load_instances():
    """Load GPUHub instance registry."""
    if not INSTANCES_FILE.exists():
        return {}
    with open(INSTANCES_FILE, encoding="utf-8") as f:
        registry = json.load(f)
    return registry.get("instances", {})


# ─── SSH Hardware Probing ─────────────────────────────────────────────────────

def probe_instance(name, inst_config, timeout=30):
    """
    SSH into a GPUHub instance and collect:
      - GPU utilization, memory, temperature
      - Active python processes
      - WandB run ID from run_id.txt
      - Last log line
    Returns a dict with hardware + process info.
    """
    import paramiko

    result = {
        "instance": name,
        "gpus_config": inst_config.get("gpus", []),
        "status": "OK",
        "gpus": [],
        "processes": [],
        "wandb_run_ids": [],
        "alerts": [],
        "error": None,
    }

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(
            inst_config["host"],
            port=inst_config["port"],
            username='root',
            password=inst_config["password"],
            timeout=timeout,
        )

        # 1. GPU metrics
        cmd_gpu = (
            "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu "
            "--format=csv,noheader,nounits 2>/dev/null"
        )
        _, stdout, _ = ssh.exec_command(cmd_gpu, timeout=timeout)
        gpu_lines = stdout.read().decode('utf-8', errors='replace').strip().split('\n')
        for line in gpu_lines:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:
                idx, util, mem_used, mem_total, temp = parts[:5]
                try:
                    mem_used_mb = float(mem_used)
                    mem_total_mb = float(mem_total)
                    mem_pct = (mem_used_mb / mem_total_mb * 100) if mem_total_mb > 0 else 0
                except (ValueError, ZeroDivisionError):
                    mem_pct = 0
                    mem_used_mb = 0
                    mem_total_mb = 0

                gpu_info = {
                    "index": int(idx) if idx.isdigit() else idx,
                    "util_pct": int(util) if util.isdigit() else util,
                    "mem_used_mb": int(mem_used_mb),
                    "mem_total_mb": int(mem_total_mb),
                    "mem_pct": round(mem_pct, 1),
                    "temp_c": int(temp) if temp.isdigit() else temp,
                }
                result["gpus"].append(gpu_info)

        # 2. Active python processes (training-related)
        cmd_ps = (
            "ps aux | grep -E 'python.*(run_full_pipeline|train|optuna|hpo)' "
            "| grep -v grep | awk '{print $2, $11, $12, $13}'"
        )
        _, stdout, _ = ssh.exec_command(cmd_ps, timeout=timeout)
        ps_out = stdout.read().decode('utf-8', errors='replace').strip()
        if ps_out:
            for pline in ps_out.split('\n'):
                parts = pline.strip().split(None, 3)
                if parts:
                    result["processes"].append({
                        "pid": parts[0],
                        "cmd": " ".join(parts[1:]) if len(parts) > 1 else "python",
                    })

        # 3. WandB run IDs from remote — scan wandb/ dirs for all run IDs
        #    run_id.txt only has latest; wandb/run-*/ dirs have all
        cmd_runid = (
            "cat /workspace/DeepScalper/run_id.txt 2>/dev/null; "
            "echo '---'; "
            "ls -1d /workspace/DeepScalper/wandb/run-* 2>/dev/null | "
            "sed 's/.*run-[0-9_]*-//' | sort -u"
        )
        _, stdout, _ = ssh.exec_command(cmd_runid, timeout=timeout)
        runid_out = stdout.read().decode('utf-8', errors='replace').strip()
        seen_ids = set()
        for section in runid_out.split('---'):
            for line in section.strip().split('\n'):
                rid = line.strip()
                if rid and len(rid) == 8 and rid.isalnum() and rid not in seen_ids:
                    result["wandb_run_ids"].append(rid)
                    seen_ids.add(rid)

        # 4. Per-GPU process count (MONITOR-10)
        cmd_compute = (
            "nvidia-smi --query-compute-apps=gpu_bus_id,pid "
            "--format=csv,noheader,nounits 2>/dev/null"
        )
        _, stdout, _ = ssh.exec_command(cmd_compute, timeout=timeout)
        compute_out = stdout.read().decode('utf-8', errors='replace').strip()

        # Map bus_id to GPU index (nvidia-smi reports GPUs in index order)
        cmd_bus = (
            "nvidia-smi --query-gpu=index,gpu_bus_id "
            "--format=csv,noheader,nounits 2>/dev/null"
        )
        _, stdout, _ = ssh.exec_command(cmd_bus, timeout=timeout)
        bus_out = stdout.read().decode('utf-8', errors='replace').strip()
        bus_to_idx = {}
        for bline in bus_out.split('\n'):
            bparts = [p.strip() for p in bline.split(',')]
            if len(bparts) >= 2:
                bus_to_idx[bparts[1]] = int(bparts[0]) if bparts[0].isdigit() else bparts[0]

        # Count PIDs per GPU index
        gpu_proc_counts = {}
        if compute_out:
            for cline in compute_out.split('\n'):
                cparts = [p.strip() for p in cline.split(',')]
                if len(cparts) >= 2:
                    bus_id = cparts[0]
                    gpu_idx = bus_to_idx.get(bus_id)
                    if gpu_idx is not None:
                        gpu_proc_counts[gpu_idx] = gpu_proc_counts.get(gpu_idx, 0) + 1

        # Attach per-GPU process count
        for gpu in result["gpus"]:
            gpu["process_count"] = gpu_proc_counts.get(gpu["index"], 0)

        # 5. Last log line (quick health check)
        cmd_log = "tail -3 /workspace/DeepScalper/run_*.log 2>/dev/null | tail -5"
        _, stdout, _ = ssh.exec_command(cmd_log, timeout=timeout)
        log_tail = stdout.read().decode('utf-8', errors='replace').strip()
        result["log_tail"] = log_tail[-200:] if log_tail else ""

        # 6. Anomaly detection on hardware
        for gpu in result["gpus"]:
            gpu_idx = gpu["index"]
            util = gpu["util_pct"]
            mem_pct = gpu["mem_pct"]

            # Check if there's an active process (heuristic: mem > 2GB = active training)
            has_active_run = gpu["mem_used_mb"] > 2000

            if has_active_run and isinstance(util, int) and util < GPU_UTIL_LOW:
                result["alerts"].append(
                    f"LOW GPU UTIL: GPU {gpu_idx} at {util}% (threshold: {GPU_UTIL_LOW}%)"
                )

            if isinstance(mem_pct, (int, float)) and mem_pct > GPU_MEM_HIGH:
                result["alerts"].append(
                    f"HIGH GPU MEM: GPU {gpu_idx} at {mem_pct:.0f}% ({gpu['mem_used_mb']}/"
                    f"{gpu['mem_total_mb']} MB)"
                )

    except Exception as e:
        result["status"] = "UNREACHABLE"
        result["error"] = f"{type(e).__name__}: {e}"
        result["alerts"].append(f"SSH FAILED: {e}")

    finally:
        ssh.close()

    return result


# ─── WandB Run Monitoring ────────────────────────────────────────────────────

def fetch_wandb_runs():
    """Fetch all running WandB runs with key metrics."""
    import wandb
    api = wandb.Api()

    runs = []
    try:
        active = api.runs(WANDB_PROJECT, filters={"state": "running"}, order="-created_at")
        for run in active:
            runs.append(_extract_wandb_metrics(run))
    except Exception as e:
        log(f"WandB fetch error: {e}", "ERROR")

    return runs


def _format_eta(seconds):
    """Format seconds into a human-readable ETA string."""
    if seconds is None or seconds < 0:
        return "--"
    if seconds < 60:
        return "<1m"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours >= 24:
        days = hours // 24
        hours = hours % 24
        return f"{days}d{hours:02d}h"
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m"


def _count_completed_trials(summary):
    """
    Count completed HPO trials by scanning WandB summary for terminal trial keys.

    Each trial logs one of: hpo/t{N}/completed, hpo/t{N}/killed, hpo/t{N}/error,
    or hpo/t{N}/status (pruned). We find the highest trial number with a terminal
    marker to determine how many trials have finished.
    """
    terminal_suffixes = ('/completed', '/killed', '/error', '/status')
    max_trial = -1
    for key in summary:
        if not key.startswith('hpo/t'):
            continue
        if not any(key.endswith(s) for s in terminal_suffixes):
            continue
        # Extract trial number: "hpo/t5/completed" -> "5"
        parts = key.split('/')
        if len(parts) >= 2:
            trial_str = parts[1][1:]  # strip 't' prefix
            if trial_str.isdigit():
                max_trial = max(max_trial, int(trial_str))
    # Trial numbers are 0-indexed, so max_trial=5 means 6 trials done
    return max_trial + 1 if max_trial >= 0 else 0


def _compute_eta(run_config, summary, step, sps):
    """
    Compute ETA (seconds remaining) based on run phase and progress.

    HPO phase: Count completed trials from summary keys, estimate remaining
               trials * steps_per_trial + full training run.
    Training phase: remaining = total_timesteps - step.
    """
    if not sps or sps <= 0 or step is None:
        return None

    # Detect phase from summary status markers
    hpo_status = summary.get('hpo/status')
    train_status = summary.get('train/status')

    # Try to get config values
    cfg_training = run_config.get('training', {}) or {}
    cfg_hpo = run_config.get('hpo', {}) or {}
    total_timesteps = cfg_training.get('total_timesteps')
    steps_per_trial = cfg_hpo.get('steps_per_trial')
    n_trials = cfg_hpo.get('n_trials')
    hpo_enabled = cfg_hpo.get('enabled', False)

    # HPO phase: hpo/status == "started" and NOT yet "completed"
    if hpo_enabled and hpo_status == 'started' and steps_per_trial and n_trials:
        trials_done = _count_completed_trials(summary)
        remaining_trials = max(0, n_trials - trials_done)
        # Assume current in-progress trial is ~halfway through (conservative)
        remaining_hpo_steps = max(0, remaining_trials - 0.5) * steps_per_trial
        # After HPO completes, full training run follows
        full_train_steps = total_timesteps or 0
        remaining_steps = remaining_hpo_steps + full_train_steps
        return remaining_steps / sps

    # Training phase: train/status == "started", or HPO completed and training underway
    # During training, _step is set explicitly to global_step via wandb.log(step=global_step)
    if train_status == 'started' and total_timesteps and total_timesteps > 0:
        remaining = max(0, total_timesteps - step)
        return remaining / sps

    # Fallback: HPO disabled, no explicit status — try total_timesteps vs step
    if not hpo_enabled and total_timesteps and total_timesteps > 0:
        remaining = max(0, total_timesteps - step)
        return remaining / sps

    return None


def _extract_wandb_metrics(run):
    """Extract key metrics from a WandB run object."""
    summary = run.summary._json_dict
    step = summary.get('_step', summary.get('step', 0))

    # Determine experiment tag
    exp_tags = [t for t in (run.tags or []) if not t.startswith('gpuhub') and not t.startswith('rtx')]
    exp_tag = exp_tags[0].upper() if exp_tags else "?"

    # SPS
    sps = summary.get('train/sps')

    # Q-values
    q_mean = summary.get('agent/q_value/mean', summary.get('agent/q_qty_mean'))
    q_max = summary.get('agent/q_value/max')

    # Loss
    loss = summary.get('agent/loss_total')

    # PF
    hpo_pf = summary.get('hpo/best_profit_factor', summary.get('hpo/best_pf'))
    eval_pf = summary.get('eval/profit_factor')
    best_pf = hpo_pf or eval_pf

    # HPO
    hpo_trials = summary.get('hpo/n_trials')
    hpo_status = summary.get('hpo/status')
    train_status = summary.get('train/status')

    # Staleness
    last_ts = summary.get('_timestamp')
    minutes_since = None
    if last_ts and isinstance(last_ts, (int, float)):
        minutes_since = (time.time() - last_ts) / 60

    # Alerts
    alerts = []
    verdict = "OK"

    # Stall
    if minutes_since and minutes_since > STALL_MINUTES:
        alerts.append(f"STALL: No update for {minutes_since:.0f}m (last step: {step:,})")
        verdict = "CRITICAL"

    # NaN loss
    if loss is not None:
        if str(loss).lower() in ('nan', 'inf') or (isinstance(loss, float) and (loss != loss or abs(loss) == float('inf'))):
            alerts.append(f"NaN/Inf LOSS at step {step:,}")
            verdict = "CRITICAL"

    # Q-divergence
    if q_mean is not None and isinstance(q_mean, (int, float)) and abs(q_mean) > Q_DIVERGENCE_THRESHOLD:
        alerts.append(f"Q-DIVERGE: |q_mean|={abs(q_mean):.0e}")
        verdict = "CRITICAL"

    # Low SPS (check after enough steps)
    if sps is not None and step > 10_000:
        # Determine GPU tier from tags (normalize whitespace for double-space edge case)
        gpu_tag = next((t for t in (run.tags or []) if t.startswith('rtx')), None)
        gpu_name = " ".join(gpu_tag.upper().split()) if gpu_tag else "default"
        gpu_thresholds = SPS_THRESHOLDS.get(gpu_name, SPS_THRESHOLDS["default"])

        # Detect algorithm from tags or config
        run_tags = [t.lower() for t in (run.tags or [])]
        if "sac" in run_tags:
            algo = "sac"
        elif any(t in run_tags for t in ("iqn", "bdq", "ppo")):
            algo = "default"
        else:
            # Fallback: check run config for SAC-specific keys
            run_cfg = run.config or {}
            algo = "sac" if "sac" in run_cfg.get("agents", {}) else "default"

        sps_threshold = gpu_thresholds.get(algo, gpu_thresholds["default"])
        if sps < sps_threshold * 0.5:  # Warn at 50% of expected
            alerts.append(f"LOW SPS: {sps:.0f} (expected >{sps_threshold} for {algo.upper()})")
            if verdict == "OK":
                verdict = "WARNING"

    # ETA computation
    run_config = run.config or {}
    eta_seconds = _compute_eta(run_config, summary, step, sps)
    eta_str = _format_eta(eta_seconds)

    return {
        "run_id": run.id,
        "run_name": run.name,
        "exp_tag": exp_tag,
        "state": run.state,
        "tags": run.tags or [],
        "step": step,
        "sps": round(sps, 1) if sps else None,
        "q_mean": round(q_mean, 2) if q_mean and isinstance(q_mean, (int, float)) else None,
        "q_max": round(q_max, 0) if q_max and isinstance(q_max, (int, float)) else None,
        "loss": round(loss, 4) if loss and isinstance(loss, (int, float)) and loss == loss else loss,
        "best_pf": round(best_pf, 4) if best_pf else None,
        "hpo_trials": hpo_trials,
        "hpo_status": hpo_status,
        "train_status": train_status,
        "minutes_since": round(minutes_since, 1) if minutes_since else None,
        "eta_seconds": round(eta_seconds, 0) if eta_seconds is not None else None,
        "eta_str": eta_str,
        "alerts": alerts,
        "verdict": verdict,
    }


# ─── Cross-Reference Matching ────────────────────────────────────────────────

def match_runs_to_instances(hw_results, wandb_runs):
    """
    Match WandB runs to GPUHub instances using run_id.txt from SSH probe.
    Returns enriched wandb_runs with instance info.
    """
    # Build reverse map: run_id -> instance
    rid_to_instance = {}
    for hw in hw_results:
        for rid in hw.get("wandb_run_ids", []):
            rid_to_instance[rid] = hw["instance"]

    for run in wandb_runs:
        run["instance"] = rid_to_instance.get(run["run_id"], "?")

    return wandb_runs


# ─── Output Formatting ───────────────────────────────────────────────────────

def format_hw_table(hw_results):
    """Format hardware status as a table."""
    lines = []
    lines.append("")
    lines.append("=" * 90)
    lines.append("  FLEET HARDWARE STATUS")
    lines.append("=" * 90)

    header = f"{'Instance':<12} {'GPU':<4} {'Model':<10} {'Util%':>6} {'Mem':>12} {'Mem%':>5} {'Temp':>5} {'Status':<12}"
    lines.append(header)
    lines.append("-" * 90)

    for hw in hw_results:
        if hw["status"] == "UNREACHABLE":
            lines.append(f"{hw['instance']:<12} {'--':<4} {'--':<10} {'--':>6} {'--':>12} {'--':>5} {'--':>5} {'UNREACHABLE':<12}")
            continue

        for i, gpu in enumerate(hw["gpus"]):
            gpu_model = hw["gpus_config"][i] if i < len(hw["gpus_config"]) else "?"
            mem_str = f"{gpu['mem_used_mb']}/{gpu['mem_total_mb']}MB"
            util_str = f"{gpu['util_pct']}%" if isinstance(gpu['util_pct'], int) else str(gpu['util_pct'])
            temp_str = f"{gpu['temp_c']}C" if isinstance(gpu['temp_c'], int) else str(gpu['temp_c'])
            mem_pct_str = f"{gpu['mem_pct']:.0f}%"

            procs = gpu.get("process_count", None)
            if procs is not None:
                status = f"{procs} proc" if procs > 0 else "idle"
            else:
                # Fallback to instance-level count (shared)
                inst_procs = len(hw["processes"])
                status = f"{inst_procs} proc" if inst_procs > 0 else "idle"

            lines.append(
                f"{hw['instance']:<12} {gpu['index']:<4} {gpu_model:<10} "
                f"{util_str:>6} {mem_str:>12} {mem_pct_str:>5} {temp_str:>5} {status:<12}"
            )

    lines.append("")

    # Alerts
    all_alerts = []
    for hw in hw_results:
        for alert in hw.get("alerts", []):
            all_alerts.append(f"  [{hw['instance']}] {alert}")

    if all_alerts:
        lines.append("  HW ALERTS:")
        lines.extend(all_alerts)
        lines.append("")

    return "\n".join(lines)


def format_wandb_table(wandb_runs):
    """Format WandB run status as a table."""
    lines = []
    lines.append("=" * 120)
    lines.append("  WANDB ACTIVE RUNS")
    lines.append("=" * 120)

    header = (
        f"{'Tag':<8} {'RunID':<10} {'Instance':<12} {'Step':>10} "
        f"{'SPS':>6} {'PF':>7} {'Q_mean':>8} {'Loss':>8} "
        f"{'Phase':<10} {'Updated':>8} {'ETA':>8} {'Status':<10}"
    )
    lines.append(header)
    lines.append("-" * 120)

    if not wandb_runs:
        lines.append("  (no active WandB runs)")
    else:
        # Sort: criticals first, then warnings, then OK
        priority = {"CRITICAL": 0, "WARNING": 1, "OK": 2}
        sorted_runs = sorted(wandb_runs, key=lambda r: priority.get(r["verdict"], 3))

        for run in sorted_runs:
            verdict_icon = {"OK": "OK", "WARNING": "WARN", "CRITICAL": "CRIT"}
            status = verdict_icon.get(run["verdict"], run["verdict"])

            step_str = f"{run['step']:,}" if run['step'] else "--"
            sps_str = f"{run['sps']:.0f}" if run['sps'] else "--"
            pf_str = f"{run['best_pf']:.4f}" if run['best_pf'] else "--"
            q_str = f"{run['q_mean']:.1f}" if run['q_mean'] is not None else "--"
            loss_str = f"{run['loss']:.4f}" if run['loss'] and isinstance(run['loss'], (int, float)) else "--"
            phase_str = run['train_status'] or run['hpo_status'] or "--"
            if run['hpo_trials']:
                phase_str = f"HPO {run['hpo_trials']}t"
            updated_str = f"{run['minutes_since']:.0f}m" if run['minutes_since'] is not None else "--"

            eta_str = run.get('eta_str', '--')

            lines.append(
                f"{run['exp_tag']:<8} {run['run_id']:<10} {run['instance']:<12} "
                f"{step_str:>10} {sps_str:>6} {pf_str:>7} {q_str:>8} {loss_str:>8} "
                f"{phase_str:<10} {updated_str:>8} {eta_str:>8} {status:<10}"
            )

    lines.append("")

    # Alerts
    all_alerts = []
    for run in wandb_runs:
        for alert in run.get("alerts", []):
            all_alerts.append(f"  [{run['exp_tag']} {run['run_id']}] {alert}")

    if all_alerts:
        lines.append("  RUN ALERTS:")
        lines.extend(all_alerts)
        lines.append("")

    return "\n".join(lines)


def format_summary(hw_results, wandb_runs):
    """One-line summary."""
    total_gpus = sum(len(hw["gpus"]) for hw in hw_results)
    active_gpus = sum(
        1 for hw in hw_results for gpu in hw["gpus"]
        if gpu["mem_used_mb"] > 1000
    )
    unreachable = sum(1 for hw in hw_results if hw["status"] == "UNREACHABLE")

    n_runs = len(wandb_runs)
    n_crit = sum(1 for r in wandb_runs if r["verdict"] == "CRITICAL")
    n_warn = sum(1 for r in wandb_runs if r["verdict"] == "WARNING")
    n_ok = sum(1 for r in wandb_runs if r["verdict"] == "OK")

    hw_alerts = sum(len(hw.get("alerts", [])) for hw in hw_results)
    run_alerts = sum(len(r.get("alerts", [])) for r in wandb_runs)

    parts = [
        f"GPUs: {active_gpus}/{total_gpus} active",
        f"Instances: {len(hw_results) - unreachable}/{len(hw_results)} reachable",
        f"Runs: {n_runs} (OK:{n_ok} WARN:{n_warn} CRIT:{n_crit})",
        f"Alerts: {hw_alerts + run_alerts}",
    ]

    lines = [
        "-" * 120,
        f"  SUMMARY: {' | '.join(parts)}",
        f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "-" * 120,
    ]
    return "\n".join(lines)


def format_json(hw_results, wandb_runs):
    """JSON output for programmatic consumption."""
    return json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": hw_results,
        "runs": wandb_runs,
    }, indent=2, default=str)


# ─── Anomaly: Orphan Detection ───────────────────────────────────────────────

def detect_orphans(hw_results, wandb_runs):
    """
    Detect orphan states:
      1. WandB says "running" but no process on any instance (ghost run)
      2. GPU has active process but no WandB run (untracked run)
         Compares active GPU count vs tracked run count per instance.
    """
    alerts = []

    unmatched_ids = {r["run_id"] for r in wandb_runs if r.get("instance") == "?"}

    # Check for unmatched WandB runs (ghost runs)
    for rid in unmatched_ids:
        run = next(r for r in wandb_runs if r["run_id"] == rid)
        alerts.append(
            f"GHOST RUN: WandB run {run['exp_tag']} ({rid}) is 'running' but not found on any instance"
        )

    # Check for GPUs with high memory but no WandB run
    # Compare active GPU count vs tracked run count per instance
    for hw in hw_results:
        if hw["status"] == "UNREACHABLE":
            continue
        active_gpus = [g for g in hw["gpus"] if g["mem_used_mb"] > 2000]
        instance_runs = [r for r in wandb_runs if r.get("instance") == hw["instance"]]
        # Also count run IDs found via SSH
        total_tracked = max(len(instance_runs), len(hw.get("wandb_run_ids", [])))
        if len(active_gpus) > total_tracked:
            for gpu in active_gpus[total_tracked:]:
                alerts.append(
                    f"UNTRACKED: {hw['instance']} GPU {gpu['index']} using "
                    f"{gpu['mem_used_mb']}MB but no WandB run found"
                )

    return alerts


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DeepScalper Fleet Monitor")
    parser.add_argument("--wandb-only", action="store_true", help="Skip hardware probing (no SSH)")
    parser.add_argument("--hw-only", action="store_true", help="Skip WandB queries")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--timeout", type=int, default=30, help="SSH timeout in seconds")
    args = parser.parse_args()

    hw_results = []
    wandb_runs = []

    # 1. Probe hardware
    if not args.wandb_only:
        instances = load_instances()
        if not instances:
            print("No instances found in instances.json")
        else:
            for name, config in instances.items():
                hw_results.append(probe_instance(name, config, timeout=args.timeout))

    # 2. Fetch WandB runs
    if not args.hw_only:
        wandb_runs = fetch_wandb_runs()

    # 3. Cross-reference
    if hw_results and wandb_runs:
        wandb_runs = match_runs_to_instances(hw_results, wandb_runs)

    # 4. Orphan detection
    orphan_alerts = []
    if hw_results and wandb_runs:
        orphan_alerts = detect_orphans(hw_results, wandb_runs)

    # 5. Output
    if args.json:
        output = format_json(hw_results, wandb_runs)
    else:
        parts = []
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        parts.append(f"\n  FLEET MONITOR -- {timestamp}")

        if hw_results:
            parts.append(format_hw_table(hw_results))
        if wandb_runs or not args.hw_only:
            parts.append(format_wandb_table(wandb_runs))
        if orphan_alerts:
            parts.append("  ORPHAN ALERTS:")
            parts.extend(f"    {a}" for a in orphan_alerts)
            parts.append("")
        if hw_results or wandb_runs:
            parts.append(format_summary(hw_results, wandb_runs))

        output = "\n".join(parts)

    print(output)

    # Log
    log(f"Fleet check: {len(hw_results)} instances, {len(wandb_runs)} runs, "
        f"{sum(len(r.get('alerts', [])) for r in wandb_runs)} run alerts, "
        f"{sum(len(h.get('alerts', [])) for h in hw_results)} hw alerts")

    # Exit code: 2 for criticals, 1 for warnings, 0 for clean
    has_critical = any(r["verdict"] == "CRITICAL" for r in wandb_runs) or orphan_alerts
    has_warning = any(r["verdict"] == "WARNING" for r in wandb_runs)
    has_hw_alert = any(hw.get("alerts") for hw in hw_results)

    if has_critical:
        return 2
    elif has_warning or has_hw_alert:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
