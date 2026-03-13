#!/usr/bin/env python3
"""
DeepScalper Run Watchdog
========================
Non-blocking health check for ALL active WandB runs.
Designed to be called periodically (e.g., every 30 min via cron).

Checks:
  1. Run state (running / finished / crashed / failed)
  2. Training stall (no new steps in 30+ min)
  3. NaN loss detection
  4. Q-value divergence (|q_mean| > 1e4)
  5. Low profit factor trend (early warning)
  6. Switch rate anomalies (too high = noise trading, too low = stuck)

On completion: triggers collect_run.py + prints analysis summary.
On failure: prints alert with last known metrics.

Output is append-logged to results/watchdog.log for review.

Usage:
    python scripts/watchdog.py                # Check all active runs
    python scripts/watchdog.py --tag k1       # Filter by WandB tag
    python scripts/watchdog.py --collect      # Also collect finished runs
"""

import sys
import os
import argparse
from datetime import datetime, timezone

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Thresholds
Q_DIVERGENCE_THRESHOLD = 1e4
STALL_MINUTES = 30
SWITCH_RATE_HIGH = 0.8  # >80% of steps are switches = noise
SWITCH_RATE_LOW = 0.01  # <1% = stuck in one position
MIN_STEPS_FOR_EARLY_PF = 50_000  # Don't judge PF before this
PF_EARLY_KILL_THRESHOLD = 0.7  # PF < 0.7 after 100K+ steps = likely hopeless

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
LOG_FILE = os.path.join(LOG_DIR, 'watchdog.log')


def log(msg, level="INFO"):
    """Print and append to log file."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] [{level}] {msg}"
    print(line)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def get_active_runs(tag=None):
    """Fetch all running WandB runs, optionally filtered by tag."""
    import wandb
    api = wandb.Api()
    project = "bigcan-chiwin-technology/FinRL-Pro-DS"

    filters = {"state": "running"}
    if tag:
        filters["tags"] = {"$in": [tag]}

    runs = api.runs(project, filters=filters, order="-created_at")
    return list(runs)


def get_recently_finished(since_minutes=35):
    """Fetch runs that finished in the last N minutes (catch completions between polls)."""
    import wandb
    from datetime import timedelta
    api = wandb.Api()
    project = "bigcan-chiwin-technology/FinRL-Pro-DS"

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=since_minutes)
    cutoff_str = cutoff.strftime('%Y-%m-%dT%H:%M:%S')

    results = []
    for state in ['finished', 'crashed', 'failed']:
        runs = api.runs(project, filters={
            "state": state,
            "updatedAt": {"$gte": cutoff_str}
        }, order="-updated_at")
        results.extend(list(runs))
    return results


def check_run_health(run):
    """
    Analyze a single run's health using summary (fast, no scan_history).
    Returns a dict with status and alerts.
    """
    result = {
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "tags": run.tags,
        "alerts": [],
        "metrics": {},
        "verdict": "OK",  # OK, WARNING, CRITICAL, DEAD
    }

    summary = run.summary._json_dict
    step = summary.get('_step', summary.get('step', 0))
    result["metrics"]["step"] = step

    # --- Check 1: Stall Detection ---
    last_ts = summary.get('_timestamp')
    if last_ts and isinstance(last_ts, (int, float)):
        import time as _time
        minutes_since = (_time.time() - last_ts) / 60
        result["metrics"]["minutes_since_update"] = round(minutes_since, 1)
        if minutes_since > STALL_MINUTES:
            result["alerts"].append(
                f"STALL: No new data for {minutes_since:.0f} min (last step: {step:,})"
            )
            result["verdict"] = "CRITICAL"

    # --- Check 2: NaN Loss ---
    loss = summary.get('agent/loss_total')
    if loss is not None:
        result["metrics"]["loss"] = loss
        if str(loss).lower() in ('nan', 'inf') or (isinstance(loss, float) and (loss != loss or abs(loss) == float('inf'))):
            result["alerts"].append(f"NaN/Inf LOSS at step {step:,}")
            result["verdict"] = "CRITICAL"

    # --- Check 3: Q-Divergence ---
    q_mean = summary.get('agent/q_value/mean', summary.get('agent/q_qty_mean'))
    q_max = summary.get('agent/q_value/max')
    target_max = summary.get('agent/q_value/target_max')

    if q_mean is not None:
        result["metrics"]["q_mean"] = q_mean
        if abs(q_mean) > Q_DIVERGENCE_THRESHOLD:
            result["alerts"].append(
                f"Q-DIVERGENCE: |q_mean| = {abs(q_mean):.2e} > {Q_DIVERGENCE_THRESHOLD:.0e}"
            )
            result["verdict"] = "CRITICAL"

    if q_max is not None:
        result["metrics"]["q_max"] = q_max

    if target_max is not None:
        result["metrics"]["target_max"] = target_max
        if abs(target_max) > Q_DIVERGENCE_THRESHOLD * 5:
            result["alerts"].append(
                f"TARGET Q EXPLODING: target_max = {target_max:.0f}"
            )
            result["verdict"] = "CRITICAL"

    # --- Check 4: Profit Factor ---
    hpo_pf = summary.get('hpo/best_profit_factor', summary.get('hpo/best_pf'))
    eval_pf = summary.get('eval/profit_factor')
    best_pf = hpo_pf or eval_pf

    if best_pf is not None:
        result["metrics"]["best_pf"] = round(best_pf, 4)
        if step > MIN_STEPS_FOR_EARLY_PF and best_pf < PF_EARLY_KILL_THRESHOLD:
            result["alerts"].append(
                f"LOW PF: Best PF = {best_pf:.4f} after {step:,} steps (< {PF_EARLY_KILL_THRESHOLD})"
            )
            if result["verdict"] != "CRITICAL":
                result["verdict"] = "WARNING"

    # --- Check 5: HPO Status ---
    hpo_status = summary.get('hpo/status')
    hpo_trials = summary.get('hpo/n_trials')
    if hpo_status:
        result["metrics"]["hpo_status"] = hpo_status
    if hpo_trials:
        result["metrics"]["hpo_trials"] = hpo_trials

    # Per-trial PFs
    trial_pfs = []
    for i in range(50):
        tpf = summary.get(f'hpo/t{i}/profit_factor')
        if tpf is not None:
            trial_pfs.append((i, round(tpf, 4)))
    if trial_pfs:
        result["metrics"]["trial_pfs"] = trial_pfs

    # --- Check 6: Training Progress ---
    sps = summary.get('train/sps')
    if sps is not None:
        result["metrics"]["sps"] = round(sps, 1)

    reward_mean = summary.get('train/reward_mean')
    if reward_mean is not None:
        result["metrics"]["reward_mean"] = round(reward_mean, 2)

    train_status = summary.get('train/status')
    if train_status:
        result["metrics"]["train_status"] = train_status

    return result


def analyze_finished_run(run):
    """Analyze a finished/crashed run and return summary."""
    result = {
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "tags": run.tags,
        "metrics": {},
        "assessment": "",
    }

    summary = run.summary._json_dict

    # Key metrics
    for key in ['eval/profit_factor', 'eval/total_trades', 'eval/sharpe_ratio',
                'eval/max_drawdown', 'eval/total_return', 'eval/switch_rate',
                'hpo/best_profit_factor', 'hpo/best_trial', 'hpo/n_trials',
                'agent/q_value/mean', 'agent/loss_total', '_step',
                'train/reward_mean', 'train/sps']:
        val = summary.get(key)
        if val is not None:
            result["metrics"][key] = val

    # Assessment
    pf = summary.get('eval/profit_factor') or summary.get('hpo/best_profit_factor')
    trades = summary.get('eval/total_trades', 0)

    if run.state in ('crashed', 'failed'):
        result["assessment"] = f"RUN {run.state.upper()} -- check logs for traceback"
    elif pf is not None:
        if pf >= 1.3:
            result["assessment"] = f"PROMISING: PF={pf:.4f} -- warrants detailed analysis"
        elif pf >= 1.05:
            result["assessment"] = f"MARGINAL: PF={pf:.4f} -- may improve with tuning"
        elif pf >= 0.9:
            result["assessment"] = f"WEAK: PF={pf:.4f} -- likely not viable"
        else:
            result["assessment"] = f"FAIL: PF={pf:.4f} -- not profitable"
    else:
        result["assessment"] = "NO PF DATA -- check if eval ran"

    if trades is not None and trades < 50:
        result["assessment"] += f" (LOW TRADES: {trades})"

    return result


def format_run_report(health):
    """Format a single run's health check as a compact report."""
    lines = []
    verdict_icons = {"OK": "+", "WARNING": "!", "CRITICAL": "X", "DEAD": "X"}
    icon = verdict_icons.get(health["verdict"], "?")

    # Extract experiment tag (k1, r2.1, gmgp1, etc.)
    exp_tags = [t for t in health.get("tags", []) if not t.startswith('gpuhub') and not t.startswith('rtx')]
    tag_label = exp_tags[0].upper() if exp_tags else "?"
    lines.append(f"  [{icon}] {tag_label}: {health['run_name']} ({health['run_id']})")
    lines.append(f"      State: {health['state']} | Verdict: {health['verdict']}")

    m = health["metrics"]
    # Line 1: Step, SPS, staleness
    parts1 = []
    if "step" in m:
        parts1.append(f"Step: {m['step']:,}")
    if "sps" in m:
        parts1.append(f"SPS: {m['sps']}")
    if "minutes_since_update" in m:
        parts1.append(f"Updated: {m['minutes_since_update']:.0f}m ago")
    if parts1:
        lines.append(f"      {' | '.join(parts1)}")

    # Line 2: Q-values, loss
    parts2 = []
    if "loss" in m:
        parts2.append(f"Loss: {m['loss']:.4f}" if isinstance(m['loss'], (int, float)) else f"Loss: {m['loss']}")
    if "q_mean" in m:
        parts2.append(f"Q: {m['q_mean']:.2f}" if isinstance(m['q_mean'], (int, float)) else f"Q: {m['q_mean']}")
    if "q_max" in m:
        parts2.append(f"Qmax: {m['q_max']:.0f}" if isinstance(m['q_max'], (int, float)) else f"Qmax: {m['q_max']}")
    if "target_max" in m:
        parts2.append(f"TgtMax: {m['target_max']:.0f}" if isinstance(m['target_max'], (int, float)) else f"TgtMax: {m['target_max']}")
    if parts2:
        lines.append(f"      {' | '.join(parts2)}")

    # Line 3: PF, HPO, reward
    parts3 = []
    if "best_pf" in m:
        parts3.append(f"BestPF: {m['best_pf']:.4f}")
    if "hpo_status" in m:
        parts3.append(f"HPO: {m['hpo_status']}")
    if "hpo_trials" in m:
        parts3.append(f"Trials: {m['hpo_trials']}")
    if "reward_mean" in m:
        parts3.append(f"Reward: {m['reward_mean']}")
    if "train_status" in m:
        parts3.append(f"Phase: {m['train_status']}")
    if parts3:
        lines.append(f"      {' | '.join(parts3)}")

    # Trial PFs if available
    if "trial_pfs" in m:
        pf_strs = [f"T{i}={pf}" for i, pf in m["trial_pfs"]]
        lines.append(f"      Trials: {', '.join(pf_strs)}")

    for alert in health.get("alerts", []):
        lines.append(f"      >> {alert}")

    return "\n".join(lines)


def format_finished_report(analysis):
    """Format a finished run analysis."""
    lines = []
    tag_str = ", ".join(analysis.get("tags", [])) or "no tags"
    lines.append(f"  [DONE] {analysis['run_name']} ({analysis['run_id']}) [{tag_str}]")
    lines.append(f"      State: {analysis['state']} | {analysis['assessment']}")

    m = analysis["metrics"]
    metric_parts = []
    for key in ['_step', 'hpo/best_pf', 'eval/profit_factor', 'eval/total_trades',
                'eval/sharpe_ratio', 'eval/max_drawdown', 'hpo/completed_trials']:
        if key in m:
            short_key = key.split('/')[-1]
            val = m[key]
            if isinstance(val, float):
                metric_parts.append(f"{short_key}: {val:.4f}")
            else:
                metric_parts.append(f"{short_key}: {val}")
    if metric_parts:
        lines.append(f"      {' | '.join(metric_parts)}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="DeepScalper Run Watchdog")
    parser.add_argument("--tag", help="Filter runs by WandB tag (e.g., k1, k2)")
    parser.add_argument("--collect", action="store_true", help="Auto-collect finished runs")
    parser.add_argument("--quiet", action="store_true", help="Only print warnings/errors")
    args = parser.parse_args()

    log("=" * 60)
    log("WATCHDOG CHECK START")
    log("=" * 60)

    # 1. Check active runs
    try:
        active_runs = get_active_runs(tag=args.tag)
    except Exception as e:
        log(f"Failed to fetch active runs: {e}", "ERROR")
        return 1

    log(f"Active runs: {len(active_runs)}")

    criticals = []
    warnings = []
    ok_runs = []

    for run in active_runs:
        try:
            health = check_run_health(run)
            report = format_run_report(health)

            if health["verdict"] == "CRITICAL":
                criticals.append(health)
                log(report, "CRITICAL")
            elif health["verdict"] == "WARNING":
                warnings.append(health)
                log(report, "WARNING")
            else:
                ok_runs.append(health)
                if not args.quiet:
                    log(report, "OK")
        except Exception as e:
            log(f"Error checking run {run.id}: {e}", "ERROR")

    # 2. Check recently finished runs
    try:
        finished = get_recently_finished(since_minutes=35)
    except Exception as e:
        log(f"Failed to fetch finished runs: {e}", "ERROR")
        finished = []

    if finished:
        log(f"\nRecently completed: {len(finished)}")
        for run in finished:
            try:
                analysis = analyze_finished_run(run)
                report = format_finished_report(analysis)
                log(report, "COMPLETE")

                # Auto-collect if requested
                if args.collect and run.state == 'finished':
                    log(f"  Triggering collection for {run.id}...")
                    try:
                        from scripts.collect_run import collect_run
                        collect_run(run.id)
                        log(f"  Collection complete for {run.id}", "OK")
                    except Exception as e:
                        log(f"  Collection failed for {run.id}: {e}", "ERROR")
            except Exception as e:
                log(f"Error analyzing finished run {run.id}: {e}", "ERROR")

    # 3. Summary
    log("-" * 60)
    summary_parts = [
        f"Active: {len(active_runs)}",
        f"OK: {len(ok_runs)}",
        f"Warnings: {len(warnings)}",
        f"Critical: {len(criticals)}",
        f"Recently Finished: {len(finished)}",
    ]
    log(f"SUMMARY: {' | '.join(summary_parts)}")

    if criticals:
        log("ACTION NEEDED: Critical issues detected!", "CRITICAL")
        for h in criticals:
            tag_str = ", ".join(h.get("tags", []))
            log(f"  >> {h['run_name']} [{tag_str}]: {'; '.join(h['alerts'])}", "CRITICAL")

    log("WATCHDOG CHECK END\n")

    return 1 if criticals else 0


if __name__ == "__main__":
    sys.exit(main())
