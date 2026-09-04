#!/bin/bash
# Watchdog container entrypoint. Persists env for cron-launched jobs, starts
# the cron daemon in the background, then exec's the existing watchdog_docker.py
# as PID 1 so docker stop signals propagate cleanly.
set -e

# Snapshot env vars needed by drift_watch.py (cron jobs do not inherit them).
printenv \
    | grep -E '^(WANDB_API_KEY|TELEGRAM_BOT_TOKEN|TELEGRAM_CHAT_ID|TZ)=' \
    > /etc/container_env

# Start cron daemon (backgrounds by default on Debian).
cron

exec python scripts/watchdog_docker.py
