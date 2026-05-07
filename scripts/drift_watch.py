"""Drift-baseline staleness watcher (S535-R3 part b).

Polls live WandB runs for persistent `drift/status == "WARN"` and prints a
ready-to-paste invocation of `scripts/recal_drift_baseline.py` for any run
whose WARN streak exceeds a threshold. Optionally emits a Telegram alert.

Composes with `scripts/recal_drift_baseline.py` (S535-R3 part a). The two
together close the systemic gap that produced 2 manual recals in 48h
(sg1-xauusd S535-R1, sg1-btc S535-cont-3).

Typical operator flow:
    python scripts/drift_watch.py                    # one-shot scan, stdout only
    python scripts/drift_watch.py --warn-hours 8     # only flag 8h+ streaks
    python scripts/drift_watch.py --alert            # also Telegram-ping fleet bot

Drives off the same WandB schema as ActionDriftTracker
(`drift/{status,bucket,deadband_frac_live,saturation_frac_live,n_bars}`).
Live runs are identified via tag="live" + state="running"; override the tag
with --tag if a different fleet schema applies.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
WANDB_ENTITY = "bigcan-chiwin-technology"
WANDB_PROJECT = "FinRL-Pro-DS"
DRIFT_KEYS = (
    "drift/status",
    "drift/bucket",
    "drift/n_bars",
    "drift/deadband_frac_live",
    "drift/saturation_frac_live",
    "_timestamp",
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("drift_watch")


@dataclass
class WatchReport:
    run_id: str
    run_name: str
    run_path: str
    state: str
    n_drift_rows: int
    n_warn_total: int
    n_crit_total: int
    last_status: Optional[str]
    last_step: Optional[int]
    last_ts: Optional[float]
    warn_streak_n: int
    warn_streak_hours: Optional[float]
    warn_streak_first_step: Optional[int]
    warn_streak_last_step: Optional[int]


def _list_live_runs(api, tag: str, hours: int) -> list[Any]:
    """Active live runs: state=running OR updatedAt within `hours`."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    filters = {
        "tags": {"$in": [tag]},
        "$or": [
            {"state": "running"},
            {"updatedAt": {"$gte": cutoff.isoformat()}},
        ],
    }
    runs = api.runs(
        f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters=filters,
        order="-updated_at",
    )
    return list(runs)


def _trailing_warn_streak(df: pd.DataFrame) -> tuple[int, Optional[float], Optional[int], Optional[int]]:
    """Count trailing consecutive WARN rows from end of df.

    Returns (streak_n, streak_hours, first_step, last_step). hours is None if
    `_timestamp` missing.
    """
    if df.empty or "drift/status" not in df.columns:
        return 0, None, None, None
    statuses = df["drift/status"].astype(str).tolist()
    n = 0
    for s in reversed(statuses):
        if s == "WARN":
            n += 1
        else:
            break
    if n == 0:
        return 0, None, None, None
    streak_df = df.iloc[-n:]
    first_step = int(streak_df["_step"].iloc[0]) if "_step" in streak_df else None
    last_step = int(streak_df["_step"].iloc[-1]) if "_step" in streak_df else None
    hours = None
    if "_timestamp" in streak_df and not streak_df["_timestamp"].isna().all():
        ts = pd.to_numeric(streak_df["_timestamp"], errors="coerce").dropna()
        if len(ts) >= 2:
            hours = round(float(ts.iloc[-1] - ts.iloc[0]) / 3600.0, 2)
        elif len(ts) == 1:
            hours = 0.0
    return n, hours, first_step, last_step


def _watch_one(api, run, lookback_hours: int) -> WatchReport:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    cutoff_ts = cutoff.timestamp()
    try:
        hist = run.history(keys=list(DRIFT_KEYS), pandas=True, x_axis="_step")
    except Exception as exc:
        log.debug(f"  {run.id}: history fetch failed ({exc})")
        hist = pd.DataFrame()

    if not hist.empty and "_timestamp" in hist:
        hist = hist.loc[pd.to_numeric(hist["_timestamp"], errors="coerce") >= cutoff_ts]

    n_warn = int((hist["drift/status"].astype(str) == "WARN").sum()) if not hist.empty else 0
    n_crit = int((hist["drift/status"].astype(str) == "CRIT").sum()) if not hist.empty else 0
    last_status = (
        str(hist["drift/status"].iloc[-1]) if not hist.empty and "drift/status" in hist
        else None
    )
    last_step = int(hist["_step"].iloc[-1]) if not hist.empty and "_step" in hist else None
    last_ts = float(hist["_timestamp"].iloc[-1]) if not hist.empty and "_timestamp" in hist else None

    streak_n, streak_h, first_step, last_step_in_streak = _trailing_warn_streak(hist)

    return WatchReport(
        run_id=run.id,
        run_name=run.name or run.id,
        run_path=f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run.id}",
        state=run.state,
        n_drift_rows=len(hist),
        n_warn_total=n_warn,
        n_crit_total=n_crit,
        last_status=last_status,
        last_step=last_step,
        last_ts=last_ts,
        warn_streak_n=streak_n,
        warn_streak_hours=streak_h,
        warn_streak_first_step=first_step,
        warn_streak_last_step=last_step_in_streak,
    )


def _resolve_baseline_for_run(run) -> Optional[str]:
    """Best-effort: pull `drift.baseline_path` from the run config so the
    operator-recommended invocation is fully resolved.
    """
    try:
        cfg = dict(run.config) if run.config else {}
    except Exception:
        return None
    drift = cfg.get("drift") or {}
    bp = drift.get("baseline_path")
    if isinstance(bp, str) and bp:
        if bp.startswith("/app/"):
            local = REPO_ROOT / bp[len("/app/"):]
            if local.is_file():
                return str(local.relative_to(REPO_ROOT)).replace("\\", "/")
        return bp
    return None


def _format_recal_command(rep: WatchReport, baseline_hint: Optional[str]) -> str:
    baseline = baseline_hint or "<baselines/.../ensemble_report.json>"
    lo = rep.warn_streak_first_step or 0
    hi = rep.warn_streak_last_step or 0
    reason = (
        f"Persistent drift WARN streak ~{rep.warn_streak_hours}h "
        f"(_step {lo}-{hi}, n={rep.warn_streak_n}); recal against live regime."
    )
    return (
        "python scripts/recal_drift_baseline.py \\\n"
        f"  --baseline {baseline} \\\n"
        f"  --source-run {rep.run_path} \\\n"
        f"  --step-range {lo},{hi} \\\n"
        f'  --reason "{reason}" \\\n'
        "  --dry-run"
    )


def _send_telegram(message: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        env_path = REPO_ROOT / "docker" / "live" / ".env"
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if "=" not in line or line.startswith("#"):
                    continue
                k, _, v = line.partition("=")
                if k.strip() == "TELEGRAM_BOT_TOKEN" and not token:
                    token = v.strip()
                elif k.strip() == "TELEGRAM_CHAT_ID" and not chat:
                    chat = v.strip()
    if not token or not chat:
        log.warning("Telegram creds not configured (TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID); printing instead")
        log.warning(message)
        return False
    try:
        import requests
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception as exc:
        log.warning(f"Telegram send failed: {exc}")
        return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hours", type=int, default=24,
                   help="Lookback for drift history per run (default: 24h)")
    p.add_argument("--warn-hours", type=float, default=4.0,
                   help="WARN streak threshold for actionable flag (default: 4h)")
    p.add_argument("--tag", default="live",
                   help="WandB tag identifying live runs (default: 'live')")
    p.add_argument("--include-state", default="running",
                   choices=("running", "any"),
                   help="'running': only state=running. 'any': running + recently-updated. Default: running")
    p.add_argument("--alert", action="store_true",
                   help="Send Telegram alert for actionable runs (uses TELEGRAM_BOT_TOKEN/CHAT_ID or docker/live/.env)")
    p.add_argument("--quiet", action="store_true",
                   help="Only print actionable runs")
    args = p.parse_args()

    try:
        import wandb
    except ImportError:
        log.error("wandb not installed (pip install wandb)")
        return 2

    api = wandb.Api()
    log.info(f"Querying WandB for tag='{args.tag}' runs (lookback {args.hours}h)...")
    candidates = _list_live_runs(api, tag=args.tag, hours=args.hours)
    if args.include_state == "running":
        candidates = [r for r in candidates if r.state == "running"]
    log.info(f"Found {len(candidates)} candidate run(s)")

    reports: list[WatchReport] = []
    for run in candidates:
        rep = _watch_one(api, run, lookback_hours=args.hours)
        reports.append(rep)

    actionable = [r for r in reports
                  if r.warn_streak_hours is not None
                  and r.warn_streak_hours >= args.warn_hours]

    if not args.quiet:
        log.info("\n%-44s %-9s %-8s %-9s %-10s %s" % (
            "RUN", "STATE", "STATUS", "STREAK_N", "STREAK_H", "RUN_ID"))
        log.info("-" * 110)
        for r in reports:
            status = r.last_status or "-"
            streak_h = f"{r.warn_streak_hours:.1f}" if r.warn_streak_hours is not None else "-"
            mark = "  ACTION" if r in actionable else ""
            log.info("%-44s %-9s %-8s %-9d %-10s %s%s" % (
                r.run_name[:44], r.state, status, r.warn_streak_n,
                streak_h, r.run_id, mark))

    if not actionable:
        log.info(f"\nNo runs meet the actionable threshold (>={args.warn_hours}h WARN streak).")
        return 0

    log.info(f"\n=== {len(actionable)} actionable run(s) — recommended recal commands ===\n")
    for r in actionable:
        baseline_hint = _resolve_baseline_for_run(api.run(r.run_path))
        cmd = _format_recal_command(r, baseline_hint)
        log.info(f"# {r.run_name} ({r.run_path})")
        log.info(f"# WARN streak: {r.warn_streak_n} emissions over ~{r.warn_streak_hours}h "
                 f"(steps {r.warn_streak_first_step}-{r.warn_streak_last_step})")
        log.info(cmd)
        log.info("")

    if args.alert:
        msg_lines = [f"<b>Drift watch — {len(actionable)} actionable run(s)</b>"]
        for r in actionable:
            msg_lines.append(
                f"• <code>{r.run_name}</code>: WARN ~{r.warn_streak_hours}h "
                f"(n={r.warn_streak_n}, last_step={r.warn_streak_last_step})"
            )
        msg_lines.append("Run scripts/drift_watch.py for the recal commands.")
        ok = _send_telegram("\n".join(msg_lines))
        log.info(f"Telegram alert: {'sent' if ok else 'NOT sent'}")

    return 0 if not actionable else 3


if __name__ == "__main__":
    sys.exit(main())
