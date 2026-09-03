#!/usr/bin/env bash
# Strategy-agnostic entrypoint for all live/paper trading strategies.
#
# Environment variables:
#   BROKER_TYPE       — "ib", "crypto", or "ctrader" (default: "crypto")
#   STRATEGY_RUNNER   — Python script path, e.g. "scripts/run_live_ib.py"
#   STRATEGY_CONFIG   — Config YAML path, e.g. "configs/live_gmgp1_gc_ib.yaml"
#   IB_HOST           — IB Gateway hostname (default: "127.0.0.1")
#   IB_PORT           — IB Gateway port (default: "4002")
#   IB_WAIT_TIMEOUT   — Seconds to wait for IB Gateway (default: 120)
set -euo pipefail

BROKER_TYPE="${BROKER_TYPE:-crypto}"
RUNNER="${STRATEGY_RUNNER:?STRATEGY_RUNNER must be set}"
CONFIG="${STRATEGY_CONFIG:?STRATEGY_CONFIG must be set}"

IB_HOST="${IB_HOST:-127.0.0.1}"
IB_PORT="${IB_PORT:-4002}"
WAIT_TIMEOUT="${IB_WAIT_TIMEOUT:-120}"

STRATEGY_NAME="${STRATEGY_NAME:-unknown}"
echo "=== FinRL Strategy: ${STRATEGY_NAME} ==="
echo "  Broker:  ${BROKER_TYPE}"
echo "  Runner:  ${RUNNER}"
echo "  Config:  ${CONFIG}"

# --- Pre-flight checks ---
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG (is the volume mounted?)"
    exit 1
fi
if [ ! -f "$RUNNER" ]; then
    echo "ERROR: Runner not found: $RUNNER"
    exit 1
fi

# --- Wait for IB Gateway (only for IB strategies) ---
if [ "$BROKER_TYPE" = "ib" ]; then
    echo "  IB Gateway: ${IB_HOST}:${IB_PORT} (timeout ${WAIT_TIMEOUT}s)"
    elapsed=0
    until (echo >/dev/tcp/"${IB_HOST}"/"${IB_PORT}") 2>/dev/null; do
        if [ "$elapsed" -ge "$WAIT_TIMEOUT" ]; then
            echo "ERROR: IB Gateway not reachable at ${IB_HOST}:${IB_PORT} after ${WAIT_TIMEOUT}s"
            exit 1
        fi
        echo "Waiting for IB Gateway... (${elapsed}s/${WAIT_TIMEOUT}s)"
        sleep 5
        elapsed=$((elapsed + 5))
    done
    echo "IB Gateway reachable at ${IB_HOST}:${IB_PORT}"
fi

# --- Prepare runtime config (bind mount is read-only) ---
CONFIG_RUNTIME="/tmp/strategy_config_runtime.yaml"
cp "$CONFIG" "$CONFIG_RUNTIME"

# Apply config override if present (workaround for Docker Desktop Windows
# bind mount caching that can serve stale file contents)
CONFIG_OVERRIDE="/tmp/config_override.yaml"
if [ -f "$CONFIG_OVERRIDE" ]; then
    echo "Applying config override from $CONFIG_OVERRIDE"
    cp "$CONFIG_OVERRIDE" "$CONFIG_RUNTIME"
fi

# Override IB host/port in config if non-default (uses PyYAML for safe YAML mutation)
if [ "$BROKER_TYPE" = "ib" ]; then
    if [ "$IB_HOST" != "127.0.0.1" ] || [ "$IB_PORT" != "4002" ]; then
        python3 -c "
import yaml, sys
with open('${CONFIG_RUNTIME}') as f:
    cfg = yaml.safe_load(f)
host, port = '${IB_HOST}', int('${IB_PORT}')
if host != '127.0.0.1':
    cfg.setdefault('exchange', {})['host'] = host
    print(f'Overrode exchange.host -> {host}')
if port != 4002:
    cfg.setdefault('exchange', {})['paper_port'] = port
    print(f'Overrode exchange.paper_port -> {port}')
with open('${CONFIG_RUNTIME}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
"
    fi
fi

# --- Wait for PRISM API (soft check — strategy starts regardless) ---
if python3 -c "
import yaml, sys
with open('${CONFIG_RUNTIME}') as f:
    cfg = yaml.safe_load(f)
sys.exit(0 if cfg.get('prism', {}).get('enabled', False) else 1)
" 2>/dev/null; then
    PRISM_URL="${PRISM_BASE_URL:-http://prism-api:8001}"
    PRISM_TIMEOUT="${PRISM_WAIT_TIMEOUT:-300}"
    echo "PRISM overlay enabled — waiting for ${PRISM_URL} (timeout ${PRISM_TIMEOUT}s)..."
    elapsed=0
    while ! python3 -c "import urllib.request; urllib.request.urlopen('${PRISM_URL}/health')" 2>/dev/null; do
        if [ "$elapsed" -ge "$PRISM_TIMEOUT" ]; then
            echo "WARNING: PRISM API not ready after ${PRISM_TIMEOUT}s — starting with fallback mode"
            break
        fi
        echo "PRISM not ready... (${elapsed}s/${PRISM_TIMEOUT}s)"
        sleep 10
        elapsed=$((elapsed + 10))
    done
    if [ "$elapsed" -lt "$PRISM_TIMEOUT" ]; then
        echo "PRISM API ready at ${PRISM_URL}"
    fi
fi

echo "Starting ${STRATEGY_NAME}..."
exec python3 "$RUNNER" --config "$CONFIG_RUNTIME"
