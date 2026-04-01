#!/usr/bin/env bash
# Docker HEALTHCHECK for trading engine.
#
# Primary: read /tmp/health_status.json (written by LiveTradingEngine every bar).
#   Checks staleness, broker connection, consecutive errors, stop flag.
# Fallback: if JSON file missing (engine still bootstrapping), use pgrep.
#
# Env vars:
#   MAX_STALE_SECONDS  — max age of health file before unhealthy (default: 600)
#   HEALTH_FILE        — path to health status JSON (default: /tmp/health_status.json)
set -euo pipefail

HEALTH_FILE="${HEALTH_FILE:-/tmp/health_status.json}"
MAX_STALE="${MAX_STALE_SECONDS:-600}"

# --- Fallback: if health file doesn't exist yet, check process ---
if [ ! -f "$HEALTH_FILE" ]; then
    # Engine may still be bootstrapping (loading checkpoint, waiting for IB).
    # Fall back to process check during start_period.
    pgrep -f "run_live|_runner" > /dev/null 2>&1 || exit 1
    exit 0
fi

# --- Read health file with jq-free parsing (python is available in our image) ---
STATUS=$(python3 -c "
import json, sys, time

try:
    with open('${HEALTH_FILE}') as f:
        s = json.load(f)
except Exception as e:
    print(f'READ_ERROR: {e}', file=sys.stderr)
    sys.exit(1)

age = time.time() - s.get('timestamp', 0)

# Check staleness
if age > ${MAX_STALE}:
    print(f'STALE: {age:.0f}s > ${MAX_STALE}s', file=sys.stderr)
    sys.exit(1)

# Check stop flag
if s.get('should_stop', False):
    print('STOPPING', file=sys.stderr)
    sys.exit(1)

# Check consecutive errors
if s.get('consecutive_errors', 0) >= 3:
    print(f'ERRORS: {s[\"consecutive_errors\"]} consecutive', file=sys.stderr)
    sys.exit(1)

# Check broker connection
if not s.get('broker_connected', True):
    print('BROKER_DISCONNECTED', file=sys.stderr)
    sys.exit(1)

# Healthy
sys.exit(0)
") || exit 1

exit 0
