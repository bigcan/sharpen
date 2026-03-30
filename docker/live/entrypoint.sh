#!/usr/bin/env bash
# Entrypoint for GMGP1-v5 live/paper trading engine.
# 1. Waits for IB Gateway to be reachable
# 2. Overrides exchange.host in config from $IB_HOST if set
# 3. Launches the trading engine
set -euo pipefail

IB_HOST="${IB_HOST:-127.0.0.1}"
IB_PORT="${IB_PORT:-4002}"
WAIT_TIMEOUT="${IB_WAIT_TIMEOUT:-120}"

echo "=== GMGP1-v5 Trading Engine ==="
echo "  IB Gateway: ${IB_HOST}:${IB_PORT}"
echo "  Wait timeout: ${WAIT_TIMEOUT}s"

# --- Wait for IB Gateway ---
elapsed=0
until curl -sf --connect-timeout 2 "telnet://${IB_HOST}:${IB_PORT}" 2>/dev/null || \
      (echo >/dev/tcp/"${IB_HOST}"/"${IB_PORT}") 2>/dev/null; do
    if [ "$elapsed" -ge "$WAIT_TIMEOUT" ]; then
        echo "ERROR: IB Gateway not reachable at ${IB_HOST}:${IB_PORT} after ${WAIT_TIMEOUT}s"
        exit 1
    fi
    echo "Waiting for IB Gateway at ${IB_HOST}:${IB_PORT}... (${elapsed}s/${WAIT_TIMEOUT}s)"
    sleep 5
    elapsed=$((elapsed + 5))
done
echo "IB Gateway reachable at ${IB_HOST}:${IB_PORT}"

# --- Override exchange.host in config if IB_HOST is set ---
# Work on a runtime copy so the bind-mounted original stays untouched
CONFIG_SRC="${1:-configs/live_gmgp1_gc_ib.yaml}"
# Strip --config prefix if passed via CMD
CONFIG_SRC="${CONFIG_SRC#--config }"
if [[ "$1" == "--config" ]]; then
    CONFIG_SRC="$2"
    shift 2
else
    shift || true
fi

CONFIG_RUNTIME="/tmp/live_config_runtime.yaml"
cp "$CONFIG_SRC" "$CONFIG_RUNTIME"

if [ "$IB_HOST" != "127.0.0.1" ]; then
    sed -i "s|host:.*\"127.0.0.1\"|host: \"${IB_HOST}\"|" "$CONFIG_RUNTIME"
    echo "Overrode exchange.host -> ${IB_HOST}"
fi

# Override port if non-default
if [ "$IB_PORT" != "4002" ]; then
    sed -i "s|paper_port:.*4002|paper_port: ${IB_PORT}|" "$CONFIG_RUNTIME"
    echo "Overrode paper_port -> ${IB_PORT}"
fi

echo "Config: ${CONFIG_RUNTIME}"
echo "Starting trading engine..."

# --- Launch ---
exec python scripts/run_live_ib.py --config "$CONFIG_RUNTIME" "$@"
