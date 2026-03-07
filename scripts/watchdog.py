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
import json
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
    Analyze a single run's health. Returns a dict with status and alerts.
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

    # Get recent history (last 20 logged points)
    try:
        history_keys = [
            '_step', '_timestamp',
            'train/loss', 'train/q_mean', 'train/q_std',
            'eval/profit_factor', 'eval/switch_rate', 'eval/total_trades',
            'eval/reward_mean', 'hpo/best_pf',
        ]
        history = list(run.scan_history(keys=history_keys, page_size=20))
        recent = history[-20:] if history else []
    except Exception as e:
        result["alerts"].append(f"Could not fetch history: {e}")
        result["verdict"] = "WARNING"
        return result

    if not recent:
        result["alerts"].append("No history data yet (run may be initializing)")
        return result

    latest = recent[-1]
    step = latest.get('_step', 0)
    result["metrics"]["step"] = step

    # --- Check 1: Stall Detection ---
    last_ts = latest.get('_timestamp')
    if last_ts:
        try:
            if isinstance(last_ts, (int, float)):
                last_update = datetime.fromtimestamp(last_ts, tz=timezone.utc)
            else:
                last_update = datetime.fromisoformat(str(last_ts).replace('Z', '+00:00'))
            minutes_since = (datetime.now(timezone.utc) - last_update).total_seconds() / 60
            result["metrics"]["minutes_since_update"] = round(minutes_since, 1)
            if minutes_since > STALL_MINUTES:
                result["alerts"].append(
                    f"STALL: No new data for {minutes_since:.0f} min (last step: {step})"
                )
                result["verdict"] = "CRITICAL"
        except Exception:
            pass

    # --- Check 2: NaN Loss ---
    loss = latest.get('train/loss')
    if loss is not None:
        result["metrics"]["loss"] = loss
        if str(loss).lower() == 'nan' or (isinstance(loss, float) and loss != loss):
            result["alerts"].append(f"NaN LOSS at step {step}")
            result["verdict"] = "CRITICAL"

    # --- Check 3: Q-Divergence ---
    q_mean = latest.get('train/q_mean')
    if q_mean is not None:
        result["metrics"]["q_mean"] = q_mean
        if abs(q_mean) > Q_DIVERGENCE_THRESHOLD:
            result["alerts"].append(
                f"Q-DIVERGENCE: |q_mean| = {abs(q_mean):.2e} > {Q_DIVERGENCE_THRESHOLD:.0e} at step {step}"
            )
            result["verdict"] = "CRITICAL"

    # Q trend: check if growing rapidly
    if len(recent) >= 5:
        q_values = [r.get('train/q_mean') for r in recent[-5:] if r.get('train/q_mean') is not None]
        if len(q_values) >= 3:
            q_range = max(abs(v) for v in q_values) - min(abs(v) for v in q_values)
            if q_range > 1000:
                result["alerts"].append(
                    f"Q-TREND: Q-values swinging by {q_range:.0f} over recent logs"
                )
                if result["verdict"] != "CRITICAL":
                    result["verdict"] = "WARNING"

    # --- Check 4: Profit Factor Early Warning ---
    pf = latest.get('eval/profit_factor') or summary.get('eval/profit_factor')
    hpo_pf = latest.get('hpo/best_pf') or summary.get('hpo/best_pf')
    best_pf = hpo_pf or pf

    if best_pf is not None:
        result["metrics"]["best_pf"] = best_pf
        if step > MIN_STEPS_FOR_EARLY_PF and best_pf < PF_EARLY_KILL_THRESHOLD:
            result["alerts"].append(
                f"LOW PF: Best PF = {best_pf:.4f} after {step:,} steps (< {PF_EARLY_KILL_THRESHOLD})"
            )
            if result["verdict"] != "CRITICAL":
                result["verdict"] = "WARNING"

    # --- Check 5: Switch Rate Anomaly ---
    switch_rate = latest.get('eval/switch_rate') or summary.get('eval/switch_rate')
    if switch_rate is not None:
        result["metrics"]["switch_rate"] = switch_rate
        if switch_rate > SWITCH_RATE_HIGH:
            result["alerts"].append(
                f"HIGH SWITCH RATE: {switch_rate:.2%} (noise trading?)"
            )
            if result["verdict"] != "CRITICAL":
                result["verdict"] = "WARNING"
        elif switch_rate < SWITCH_RATE_LOW and step > 50_000:
            result["alerts"].append(
                f"LOW SWITCH RATE: {switch_rate:.2%} (stuck in one position?)"
            )
            if result["verdict"] != "CRITICAL":
                result["verdict"] = "WARNING"

    # --- Check 6: Total Trades ---
    total_trades = latest.get('eval/total_trades') or summary.get('eval/total_trades')
    if total_trades is not None:
        result["metrics"]["total_trades"] = total_trades

    # --- Reward ---
    reward_mean = latest.get('eval/reward_mean') or summary.get('eval/reward_mean')
    if reward_mean is not None:
        result["metrics"]["reward_mean"] = round(reward_mean, 4)

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
                'hpo/best_pf', 'hpo/best_trial_number', 'hpo/completed_trials',
                'train/q_mean', 'train/loss', '_step']:
        val = summary.get(key)
        if val is not None:
            result["metrics"][key] = val

    # Assessment
    pf = summary.get('eval/profit_factor') or summary.get('hpo/best_pf')
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

    tag_str = ", ".join(health.get("tags", [])) or "no tags"
    lines.append(f"  [{icon}] {health['run_name']} ({health['run_id']}) [{tag_str}]")
    lines.append(f"      State: {health['state']} | Verdict: {health['verdict']}")

    m = health["metrics"]
    metric_parts = []
    if "step" in m:
        metric_parts.append(f"Step: {m['step']:,}")
    if "loss" in m:
        metric_parts.append(f"Loss: {m['loss']:.4f}" if isinstance(m['loss'], (int, float)) else f"Loss: {m['loss']}")
    if "q_mean" in m:
        metric_parts.append(f"Q: {m['q_mean']:.2f}" if isinstance(m['q_mean'], (int, float)) else f"Q: {m['q_mean']}")
    if "best_pf" in m:
        metric_parts.append(f"PF: {m['best_pf']:.4f}")
    if "switch_rate" in m:
        metric_parts.append(f"SwRate: {m['switch_rate']:.2%}")
    if "reward_mean" in m:
        metric_parts.append(f"Reward: {m['reward_mean']}")
    if "minutes_since_update" in m:
        metric_parts.append(f"LastUpdate: {m['minutes_since_update']:.0f}m ago")

    if metric_parts:
        lines.append(f"      {' | '.join(metric_parts)}")

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
