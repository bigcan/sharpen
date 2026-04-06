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

import json
import logging
import os
import time

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
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
ALERT_COOLDOWN = int(os.environ.get("ALERT_COOLDOWN", "300"))  # 5 min per container

# Comma-separated container names to ignore (intentionally stopped / shelved workstreams).
# Empty string means monitor everything (backward compatible).
_IGNORE_CONTAINERS_RAW = os.environ.get("IGNORE_CONTAINERS", "")
IGNORE_CONTAINERS: set[str] = {
    name.strip()
    for name in _IGNORE_CONTAINERS_RAW.split(",")
    if name.strip()
}

def _is_ignored(container_name: str) -> bool:
    """Return True if container_name is in the ignore list."""
    if container_name in IGNORE_CONTAINERS:
        logger.debug(f"Skipping ignored container: {container_name}")
        return True
    return False


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
        msg = (
            f"<b>[DIED] {strategy}</b>\n"
            f"Container: {container_name}\n"
            f"Exit code: {exit_code}"
        )
        alert(container_name, msg)
        logger.warning(f"DIED: {container_name} (exit={exit_code})")

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

        if _is_ignored(info["name"]):
            continue

        status_str = f"{info['name']}: {info['status']} / {info['health']}"

        if info["health"] == "unhealthy":
            detail = get_health_detail(container)
            msg = (
                f"<b>[SWEEP: UNHEALTHY] {info['strategy']}</b>\n"
                f"Container: {info['name']}\n"
                f"  {detail}"
            )
            alert(info["name"], msg)
            logger.warning(f"Sweep: {status_str}")
        elif not info["running"]:
            msg = (
                f"<b>[SWEEP: NOT RUNNING] {info['strategy']}</b>\n"
                f"Container: {info['name']}\n"
                f"Status: {info['status']}"
            )
            alert(info["name"], msg)
            logger.warning(f"Sweep: {status_str}")
        else:
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
    if IGNORE_CONTAINERS:
        logger.info(f"Ignoring containers: {sorted(IGNORE_CONTAINERS)}")
    else:
        logger.info("No containers in ignore list — monitoring everything")

    client = docker.from_env()

    # Initial sweep
    sweep(client)

    last_sweep = time.time()
    last_wandb_check = time.time()

    while True:
        try:
            # Subscribe to Docker events with a timeout so we can do periodic work
            events = client.events(
                decode=True,
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

                # Check if periodic tasks are due
                now = time.time()
                if now - last_sweep >= SWEEP_INTERVAL:
                    sweep(client)
                    last_sweep = now

                if now - last_wandb_check >= WANDB_CHECK_INTERVAL:
                    _run_wandb_check()
                    last_wandb_check = now

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
        for run in runs:
            health = check_run_health(run)
            if health["verdict"] == "CRITICAL":
                alerts_str = "; ".join(health["alerts"])
                msg = (
                    f"<b>[WANDB CRITICAL] {health['run_name']}</b>\n"
                    f"Run: {health['run_id']}\n"
                    f"{alerts_str}"
                )
                alert(health["run_id"], msg)
                logger.warning(f"WandB CRITICAL: {health['run_name']} — {alerts_str}")

        logger.info(f"WandB check: {len(runs)} active runs scanned")
    except ImportError:
        logger.debug("WandB check skipped (watchdog module not available)")
    except Exception as e:
        logger.warning(f"WandB check failed: {e}")


if __name__ == "__main__":
    main()
