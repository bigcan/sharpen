#!/usr/bin/env bash
# Docker HEALTHCHECK for trading engine.
# Checks that the Python trading process is alive.
# Start period is 5 minutes (bootstrap takes ~3 min).
set -euo pipefail

# Check if any trading engine process is alive (IB or crypto runner)
pgrep -f "run_live|_runner" > /dev/null 2>&1 || exit 1

exit 0
