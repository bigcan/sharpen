#!/usr/bin/env python3
"""Docker Watchdog — monitor trading container health and send alerts.

Modes:
    1. Event-driven: subscribe to Docker health_status events (instant)
    2. Periodic sweep: every 5 min, check all finrl.monitor=true containers
    3. Periodic WandB check: every 30 min, run watchdog health checks

Alerting:
    - Telegram via Bot API (TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID env vars)
    - Falls back to stdout logging if Telegram not configured

Usage:
    # Direct (for testing)
    python scripts/watchdog_docker.py

    # Docker (production)
    docker compose --profile monitoring up -d watchdog
"""

from __future__ import annotations

import html
import json
import logging
import os
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import docker
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("watchdog_docker")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SWEEP_INTERVAL = int(os.environ.get("SWEEP_INTERVAL", "300"))  # 5 min
WANDB_CHECK_INTERVAL = int(os.environ.get("WANDB_CHECK_INTERVAL", "1800"))  # 30 min
XVAL_INTERVAL = int(os.environ.get("XVAL_INTERVAL", "900"))  # 15 min
XVAL_PROMETHEUS_URL = os.environ.get("XVAL_PROMETHEUS_URL", "http://prometheus:9090")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
ALERT_COOLDOWN = int(os.environ.get("ALERT_COOLDOWN", "300"))  # 5 min per container

# Auto-restart on persistent running/unhealthy (S489 fix, issue #4).
# After N consecutive unhealthy sweeps, `docker restart <name>` is issued.
# Rate-limited to avoid loops when the container flaps.
AUTO_RESTART_ENABLED = os.environ.get("AUTO_RESTART_UNHEALTHY", "true").lower() == "true"
AUTO_RESTART_AFTER_N = int(os.environ.get("AUTO_RESTART_AFTER_N", "3"))
AUTO_RESTART_MAX_PER_HOUR = int(os.environ.get("AUTO_RESTART_MAX_PER_HOUR", "3"))

# Per-container state, keyed by container name.
_unhealthy_streak: dict[str, int] = {}
_auto_restart_history: dict[str, list[float]] = {}

# Comma-separated container names to ignore (intentionally stopped / shelved workstreams).
# Empty string means monitor everything (backward compatible).
_IGNORE_CONTAINERS_RAW = os.environ.get("IGNORE_CONTAINERS", "")
IGNORE_CONTAINERS: set[str] = {
    name.strip()
    for name in _IGNORE_CONTAINERS_RAW.split(",")
    if name.strip()
}

# Comma-separated WandB run IDs to suppress alerts for (known false-positive stalls,
# e.g. ALPHASEEK HPO parent runs whose _step is intentionally silent — child Optuna
# trials log locally, not to WandB). See project_alphaseek_false_crit.md.
_IGNORE_WANDB_RUNS_RAW = os.environ.get("IGNORE_WANDB_RUNS", "")
IGNORE_WANDB_RUNS: set[str] = {
    rid.strip()
    for rid in _IGNORE_WANDB_RUNS_RAW.split(",")
    if rid.strip()
}

def _is_ignored(container_name: str) -> bool:
    """Return True if container_name is in the ignore list."""
    if container_name in IGNORE_CONTAINERS:
        logger.debug(f"Skipping ignored container: {container_name}")
        return True
    return False


# TradFi instrument markers on WandB run tags. Presence of any one of these
# means the run trades an instrument with scheduled market closes (weekend,
# daily session break) — stalls during those windows are expected, not failure.
_TRADFI_TAGS = {"Gold", "XAUUSD", "MGC", "GC", "cTrader", "IB", "COMEX", "FTMO", "CFD"}

# Container-name prefixes for TradFi strategies — used for sweep-path suppression
# (sweep has no WandB tags available, only container names).
_TRADFI_CONTAINER_PREFIXES = ("gmgp1-gold", "gmgp1-xauusd", "gmgp2-xauusd", "sg1-gold")


def _rollover_hours_utc(dt_utc: datetime, tz_name: str, local_break_hour: int) -> set[int]:
    """UTC hours covered by a broker's daily rollover + 1-hour post-rollover lag.

    `local_break_hour` is the hour-of-day (0-23) in the broker's server timezone
    when the daily break begins. IC Markets (Athens) breaks at server 00:00;
    CME Globex (Chicago) breaks at server 16:00. The returned set includes the
    break hour itself plus the hour immediately after, to cover WandB stall
    detection lag while the first post-break bar is still pending.
    """
    server_dt = dt_utc.astimezone(ZoneInfo(tz_name))
    offset = server_dt.utcoffset()
    # ZoneInfo always returns a non-None offset for aware datetimes.
    assert offset is not None, f"ZoneInfo({tz_name}) returned None utcoffset"
    offset_hours = int(offset.total_seconds()) // 3600
    break_start = (local_break_hour - offset_hours) % 24
    return {break_start, (break_start + 1) % 24}


def _tradfi_market_closed_utc(now_utc: time.struct_time | None = None) -> bool:
    """Approximate market-closed predicate covering both cTrader XAUUSD and CME MGC.

    Closed windows (UTC):
      - Weekend:          Fri 21:00  →  Sun 22:00
      - Daily rollover:   Mon-Thu, union of IC Markets (Athens, break at
                          server 00:00-01:00) and CME Globex (Chicago, break
                          at server 16:00-17:00). Both schedules are computed
                          per-tz to stay correct during the ~2-week windows
                          each year when Europe and US DST offsets diverge.

    Covers XAUUSD (Fri ~21Z close / Sun 22Z reopen) and MGC/CME (Fri 22Z /
    Sun 23Z reopen — we suppress slightly ahead, reopening Mon stalls still
    alert because bar_count resumes within STALL_MINUTES). Daily rollover
    added S490 after false STALL alert for live_XAUUSD_ctrader at 22:06 UTC
    during 21-22 UTC rollover gap.
    """
    t = now_utc or time.gmtime()
    wday = t.tm_wday  # Mon=0 .. Sun=6
    hour = t.tm_hour
    if wday == 5:  # Saturday
        return True
    if wday == 4 and hour >= 21:  # Friday 21:00+
        return True
    if wday == 6 and hour < 22:  # Sunday before 22:00
        return True

    # Daily rollover — Mon-Thu only (Fri rollover = start of weekend, already
    # covered above). Union IC Markets (Athens) + CME Globex (Chicago) so
    # MGC stays covered during DST transition weeks where Europe and US are
    # temporarily offset by 1h relative to each other.
    if wday in (0, 1, 2, 3):
        if now_utc is not None:
            dt_utc = datetime(
                year=t.tm_year, month=t.tm_mon, day=t.tm_mday,
                hour=t.tm_hour, minute=t.tm_min, second=t.tm_sec,
                tzinfo=timezone.utc,
            )
        else:
            dt_utc = datetime.now(timezone.utc)
        rollover_hours = (
            _rollover_hours_utc(dt_utc, "Europe/Athens", 0)
            | _rollover_hours_utc(dt_utc, "America/Chicago", 16)
        )
        if hour in rollover_hours:
            return True

    return False


def _is_stall_only(alerts: list[str]) -> bool:
    """True if every CRITICAL alert is a stall — real problems must still fire."""
    return bool(alerts) and all(a.startswith("STALL:") for a in alerts)


def _should_suppress_tradfi_weekend(health: dict) -> bool:
    """Suppress a STALL-only CRITICAL on a TradFi run during market-closed UTC window."""
    tags = set(health.get("tags") or [])
    if not (tags & _TRADFI_TAGS):
        return False
    if not _is_stall_only(health.get("alerts") or []):
        return False
    return _tradfi_market_closed_utc()


# ---------------------------------------------------------------------------
# Telegram alerting
# ---------------------------------------------------------------------------
_alert_timestamps: dict[str, float] = {}


def send_telegram(message: str) -> bool:
    """Send a Telegram message. Returns True on success."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.info(f"[ALERT] {message}")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML",
            },
            timeout=10,
        )
        if resp.status_code == 200:
            return True
        logger.warning(f"Telegram API returned {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")
    return False


def alert(container_name: str, message: str) -> None:
    """Send alert with per-container cooldown to avoid spam."""
    now = time.time()
    last = _alert_timestamps.get(container_name, 0)
    if now - last < ALERT_COOLDOWN:
        logger.debug(f"Alert suppressed (cooldown): {container_name}")
        return

    _alert_timestamps[container_name] = now
    send_telegram(message)


# ---------------------------------------------------------------------------
# Container inspection
# ---------------------------------------------------------------------------

def get_health_detail(container) -> str:
    """Read /tmp/health_status.json from container for alert context."""
    try:
        # Read health file via docker exec
        exit_code, output = container.exec_run(
            "cat /tmp/health_status.json",
            demux=True,
        )
        if exit_code == 0 and output[0]:
            status = json.loads(output[0].decode())
            parts = []
            if "last_bar_time" in status:
                parts.append(f"Last bar: {status['last_bar_time']}")
            if "position" in status:
                parts.append(f"Position: {status['position']}")
            if "drawdown_pct" in status:
                parts.append(f"Drawdown: {status['drawdown_pct']:.2%}")
            if "portfolio_value" in status:
                parts.append(f"PV: ${status['portfolio_value']:,.2f}")
            if "consecutive_errors" in status:
                parts.append(f"Errors: {status['consecutive_errors']}")
            if "broker_connected" in status:
                parts.append(
                    f"Broker: {'OK' if status['broker_connected'] else 'DISCONNECTED'}",
                )
            return "\n  ".join(parts)
    except Exception:
        pass
    return "Health file unavailable"


def inspect_container(container) -> dict:
    """Get container health status and metadata."""
    container.reload()
    state = container.attrs.get("State", {})
    health = state.get("Health", {})

    return {
        "name": container.name,
        "status": container.status,
        "health": health.get("Status", "unknown"),
        "running": state.get("Running", False),
        "strategy": container.labels.get("finrl.strategy", container.name),
        "broker": container.labels.get("finrl.broker", "unknown"),
    }


# ---------------------------------------------------------------------------
# Event listener
# ---------------------------------------------------------------------------

def handle_health_event(client: docker.DockerClient, event: dict) -> None:
    """Handle a container health_status event."""
    container_id = event.get("id", "")[:12]
    action = event.get("Action", "")  # health_status: healthy / unhealthy
    actor = event.get("Actor", {})
    attrs = actor.get("Attributes", {})
    container_name = attrs.get("name", container_id)

    # Only care about our trading containers
    if "finrl.monitor" not in attrs:
        return

    if _is_ignored(container_name):
        return

    strategy = attrs.get("finrl.strategy", container_name)

    if "unhealthy" in action:
        # Get detailed health info
        try:
            container = client.containers.get(container_id)
            detail = get_health_detail(container)
        except Exception:
            detail = "Container not accessible"

        msg = (
            f"<b>[UNHEALTHY] {strategy}</b>\n"
            f"Container: {container_name}\n"
            f"  {detail}\n"
            f"Docker will attempt restart (unless-stopped policy)"
        )
        alert(container_name, msg)
        logger.warning(f"UNHEALTHY: {container_name} — {detail}")

    elif "healthy" in action:
        # Recovery alert (only if we previously alerted)
        if container_name in _alert_timestamps:
            msg = f"<b>[RECOVERED] {strategy}</b>\nContainer: {container_name}"
            alert(container_name, msg)
            logger.info(f"RECOVERED: {container_name}")


def _detect_stop_reason(container_name: str, exit_code) -> str:
    """Scan recent container logs for 'Stop requested: <reason>' (Session 426).

    Returns the reason if found, empty otherwise. Only meaningful for exit=0 —
    on exit=137 (OOM) the engine had no chance to log.
    """
    try:
        code = int(exit_code)
    except (TypeError, ValueError):
        return ""
    if code != 0:
        return ""
    try:
        client = docker.from_env()
        container = client.containers.get(container_name)
        logs = container.logs(tail=200).decode(errors="replace")
    except Exception:
        return ""
    marker = "Stop requested: "
    idx = logs.rfind(marker)
    if idx < 0:
        return ""
    return logs[idx + len(marker):].splitlines()[0].strip()[:64]


def handle_container_event(event: dict) -> None:
    """Handle container lifecycle events (die, start, restart)."""
    action = event.get("Action", "")
    actor = event.get("Actor", {})
    attrs = actor.get("Attributes", {})
    container_name = attrs.get("name", "unknown")

    if "finrl.monitor" not in attrs:
        return

    if _is_ignored(container_name):
        return

    strategy = attrs.get("finrl.strategy", container_name)

    if action == "die":
        exit_code = attrs.get("exitCode", "?")
        # Session 426: a clean exit (0) from a trading container usually means
        # the engine self-stopped via _request_stop(). Tag the reason from
        # recent logs so execution_error / roll_failed / risk_halt are visible
        # instead of blending in with normal stops.
        stop_reason = _detect_stop_reason(container_name, exit_code)
        tag = "DIED"
        if stop_reason:
            # F-01: escape to prevent log content from breaking Telegram HTML parse.
            tag = f"DIED: SELF-STOP ({html.escape(stop_reason)})"
        msg = (
            f"<b>[{tag}] {strategy}</b>\n"
            f"Container: {container_name}\n"
            f"Exit code: {exit_code}"
        )
        alert(container_name, msg)
        logger.warning(f"{tag}: {container_name} (exit={exit_code})")

    elif action == "start":
        logger.info(f"STARTED: {container_name}")

    elif action == "restart":
        msg = (
            f"<b>[RESTARTED] {strategy}</b>\n"
            f"Container: {container_name}"
        )
        alert(container_name, msg)
        logger.info(f"RESTARTED: {container_name}")


# ---------------------------------------------------------------------------
# Periodic sweep
# ---------------------------------------------------------------------------

def _should_auto_restart(container_name: str, streak: int, now: float | None = None) -> bool:
    """Decide whether to auto-restart a persistently unhealthy container.

    Returns True when the streak has reached the configured threshold AND
    the container has not exceeded the per-hour restart cap.
    """
    if not AUTO_RESTART_ENABLED:
        return False
    if streak < AUTO_RESTART_AFTER_N:
        return False
    now = now if now is not None else time.time()
    cutoff = now - 3600.0
    history = [t for t in _auto_restart_history.get(container_name, []) if t >= cutoff]
    _auto_restart_history[container_name] = history
    return len(history) < AUTO_RESTART_MAX_PER_HOUR


def _record_auto_restart(container_name: str, now: float | None = None) -> None:
    """Append an auto-restart timestamp and prune entries older than 1h."""
    now = now if now is not None else time.time()
    cutoff = now - 3600.0
    history = [t for t in _auto_restart_history.get(container_name, []) if t >= cutoff]
    history.append(now)
    _auto_restart_history[container_name] = history


def _attempt_auto_restart(container, container_name: str) -> bool:
    """Try `container.restart()`; return True on success, False otherwise."""
    try:
        container.restart()
        return True
    except Exception as e:  # noqa: BLE001
        logger.error(f"Auto-restart failed for {container_name}: {e}")
        return False


def sweep(client: docker.DockerClient) -> None:
    """Check all finrl.monitor containers. Catch issues missed by events."""
    containers = client.containers.list(
        all=True,
        filters={"label": "finrl.monitor=true"},
    )

    if not containers:
        logger.info("Sweep: no monitored containers found")
        return

    for container in containers:
        info = inspect_container(container)
        name = info["name"]

        if _is_ignored(name):
            continue

        status_str = f"{name}: {info['status']} / {info['health']}"

        if info["health"] == "unhealthy":
            if name.startswith(_TRADFI_CONTAINER_PREFIXES) and _tradfi_market_closed_utc():
                # Suppressed: don't advance the streak during market close either.
                _unhealthy_streak.pop(name, None)
                logger.info(
                    f"Sweep UNHEALTHY suppressed (TradFi market closed): {status_str}"
                )
                continue

            streak = _unhealthy_streak.get(name, 0) + 1
            _unhealthy_streak[name] = streak

            if _should_auto_restart(name, streak):
                logger.warning(
                    f"Auto-restart: {name} after {streak}x unhealthy "
                    f"(threshold={AUTO_RESTART_AFTER_N})",
                )
                if _attempt_auto_restart(container, name):
                    _record_auto_restart(name)
                    _unhealthy_streak.pop(name, None)  # Give it a fresh window
                    alert(
                        name,
                        f"<b>[AUTO-RESTART] {info['strategy']}</b>\n"
                        f"Container: {name}\n"
                        f"Reason: {streak}x consecutive unhealthy sweeps",
                    )
                    continue

            detail = get_health_detail(container)
            msg = (
                f"<b>[SWEEP: UNHEALTHY] {info['strategy']}</b>\n"
                f"Container: {name}\n"
                f"  {detail}"
            )
            alert(name, msg)
            logger.warning(f"Sweep: {status_str}")
        elif not info["running"]:
            _unhealthy_streak.pop(name, None)  # docker restart policy handles this
            msg = (
                f"<b>[SWEEP: NOT RUNNING] {info['strategy']}</b>\n"
                f"Container: {name}\n"
                f"Status: {info['status']}"
            )
            alert(name, msg)
            logger.warning(f"Sweep: {status_str}")
        else:
            _unhealthy_streak.pop(name, None)  # Back to healthy — reset streak
            logger.info(f"Sweep: {status_str}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Docker Watchdog starting")
    logger.info(
        f"Config: sweep={SWEEP_INTERVAL}s, wandb_check={WANDB_CHECK_INTERVAL}s, "
        f"telegram={'configured' if TELEGRAM_BOT_TOKEN else 'disabled'}",
    )
    if AUTO_RESTART_ENABLED:
        logger.info(
            f"Auto-restart: enabled after {AUTO_RESTART_AFTER_N}x unhealthy sweeps "
            f"(max {AUTO_RESTART_MAX_PER_HOUR}/hour per container)",
        )
    else:
        logger.info("Auto-restart: disabled (AUTO_RESTART_UNHEALTHY=false)")
    if IGNORE_CONTAINERS:
        logger.info(f"Ignoring containers: {sorted(IGNORE_CONTAINERS)}")
    else:
        logger.info("No containers in ignore list — monitoring everything")

    client = docker.from_env()

    # Initial sweep
    sweep(client)

    last_sweep = time.time()
    last_wandb_check = time.time()
    last_xval_check = time.time()

    while True:
        try:
            # FIX WD-01: Use since/until windowed event polling instead of an
            # infinite blocking iterator. This ensures periodic tasks (sweep,
            # WandB check) run even when no Docker events occur.
            since = time.time()
            poll_interval = min(SWEEP_INTERVAL, 60)  # Poll at most every 60s

            while True:
                until = time.time()
                try:
                    events = client.events(
                        decode=True,
                        since=since,
                        until=until,
                        filters={
                            "type": ["container"],
                            "event": ["health_status", "die", "start", "restart"],
                        },
                    )
                    for event in events:
                        action = event.get("Action", "")
                        if "health_status" in action:
                            handle_health_event(client, event)
                        elif action in ("die", "start", "restart"):
                            handle_container_event(event)
                except docker.errors.APIError:
                    pass  # Transient — will retry next poll

                since = until

                # Run periodic tasks unconditionally
                now = time.time()
                if now - last_sweep >= SWEEP_INTERVAL:
                    sweep(client)
                    last_sweep = now

                if now - last_wandb_check >= WANDB_CHECK_INTERVAL:
                    _run_wandb_check()
                    last_wandb_check = now

                if now - last_xval_check >= XVAL_INTERVAL:
                    _run_xval_check()
                    last_xval_check = now

                time.sleep(poll_interval)

        except docker.errors.APIError as e:
            logger.error(f"Docker API error: {e} — reconnecting in 10s")
            time.sleep(10)
            try:
                client = docker.from_env()
            except Exception:
                pass
        except KeyboardInterrupt:
            logger.info("Watchdog stopped (SIGINT)")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e} — retrying in 10s")
            time.sleep(10)


def _run_wandb_check() -> None:
    """Run WandB health checks (reuse existing watchdog.py logic)."""
    try:
        from scripts.watchdog import get_active_runs, check_run_health

        runs = get_active_runs()
        suppressed = 0
        for run in runs:
            health = check_run_health(run)
            if health["verdict"] == "CRITICAL":
                if health["run_id"] in IGNORE_WANDB_RUNS:
                    logger.info(
                        f"WandB CRITICAL suppressed (in IGNORE_WANDB_RUNS): "
                        f"{health['run_name']} ({health['run_id']})"
                    )
                    suppressed += 1
                    continue
                if _should_suppress_tradfi_weekend(health):
                    logger.info(
                        f"WandB CRITICAL suppressed (TradFi market closed): "
                        f"{health['run_name']} ({health['run_id']}) — "
                        f"{'; '.join(health['alerts'])}"
                    )
                    suppressed += 1
                    continue
                alerts_str = "; ".join(health["alerts"])
                msg = (
                    f"<b>[WANDB CRITICAL] {health['run_name']}</b>\n"
                    f"Run: {health['run_id']}\n"
                    f"{alerts_str}"
                )
                alert(health["run_id"], msg)
                logger.warning(f"WandB CRITICAL: {health['run_name']} — {alerts_str}")

        logger.info(
            f"WandB check: {len(runs)} active runs scanned"
            + (f" ({suppressed} suppressed)" if suppressed else "")
        )
    except ImportError:
        logger.debug("WandB check skipped (watchdog module not available)")
    except Exception as e:
        logger.warning(f"WandB check failed: {e}")


def _run_xval_check() -> None:
    """XVal Layer 3: Run multi-channel cross-validation and alert on CRIT."""
    try:
        from scripts.cross_validate_live import CrossValidator

        validator = CrossValidator(
            prometheus_url=XVAL_PROMETHEUS_URL,
        )
        report = validator.run()

        for strategy, sr in report.strategies.items():
            if sr.verdict == "CRIT":
                issues = [
                    f"{c.pair}/{c.field_name}: delta={c.delta:.4f}"
                    for c in sr.comparisons
                    if c.verdict == "CRIT"
                ]
                issues_str = "; ".join(issues)
                msg = (
                    f"<b>[XVAL CRIT] {strategy}</b>\n"
                    f"Channel divergence detected:\n"
                    f"{issues_str}"
                )
                alert(f"xval-{strategy}", msg)
                logger.warning(f"XVal CRIT: {strategy} — {issues_str}")
            elif sr.verdict == "WARN":
                issues = [
                    f"{c.pair}/{c.field_name}: delta={c.delta:.4f}"
                    for c in sr.comparisons
                    if c.verdict == "WARN"
                ]
                logger.info(f"XVal WARN: {strategy} — {'; '.join(issues)}")

        logger.info(
            f"XVal check: {len(report.strategies)} strategies, "
            f"verdict={report.verdict}",
        )
    except ImportError:
        logger.debug("XVal check skipped (cross_validate_live module not available)")
    except Exception as e:
        logger.warning(f"XVal check failed: {e}")


if __name__ == "__main__":
    main()
