#!/usr/bin/env bash
# Docker HEALTHCHECK for trading engine.
# Checks that the Python trading process is alive.
# Start period is 5 minutes (bootstrap takes ~3 min).
set -euo pipefail

# Check if the main trading process is running
pgrep -f "run_live_ib.py" > /dev/null 2>&1 || exit 1

exit 0
