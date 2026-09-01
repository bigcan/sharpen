#!/bin/bash
# Wrapper invoked by /etc/cron.d/drift_watch. Cron strips the container env,
# so we re-source the snapshot persisted by watchdog_entrypoint.sh and then
# redirect output to PID 1's stdio so `docker logs watchdog` surfaces alerts.
set -e

if [ -f /etc/container_env ]; then
    set -a
    . /etc/container_env
    set +a
fi

cd /app
exec python scripts/drift_watch.py --alert --warn-hours 4 > /proc/1/fd/1 2> /proc/1/fd/2
