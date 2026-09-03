#!/bin/bash
# Final checkpoint collection from gpuhub-1 before expiry (2026-04-02 22:39)
# Run this ~1h before expiry to grab the latest state.

set -euo pipefail

SSH_HOST="<GPU_HOST>"
SSH_PORT=15844
SSH_USER="root"
REMOTE_DIR="/workspace/DeepScalper"
LOCAL_DIR="/c/FinRL/FinRL-Pro_DS/collected/gpuhub1_btc_15min_20260330"
SSH_CMD="ssh -o StrictHostKeyChecking=no -p ${SSH_PORT} ${SSH_USER}@${SSH_HOST}"

echo "=== gpuhub-1 Final Collection — $(date) ==="

# 1. Grab latest run log (has all trial results + hyperparams)
echo "[1/4] Downloading run log..."
scp -o StrictHostKeyChecking=no -P ${SSH_PORT} \
    ${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/run_20260330_165938.log \
    "${LOCAL_DIR}/run_20260330_165938.log"

# 2. Grab latest checkpoints (current trial's model weights)
echo "[2/4] Downloading checkpoints..."
scp -o StrictHostKeyChecking=no -P ${SSH_PORT} -r \
    ${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/checkpoints/sac_run/ \
    "${LOCAL_DIR}/checkpoints/"

# 3. Grab WandB local data for run 8v4mueuf
echo "[3/4] Downloading WandB local data..."
mkdir -p "${LOCAL_DIR}/wandb"
scp -o StrictHostKeyChecking=no -P ${SSH_PORT} -r \
    ${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/wandb/run-20260330_165945-8v4mueuf/ \
    "${LOCAL_DIR}/wandb/" 2>/dev/null || echo "  (WandB dir may not exist locally)"

# 4. Extract trial summary
echo "[4/4] Extracting trial summary..."
grep -E "^INFO:DeepScalperPipeline:Trial" "${LOCAL_DIR}/run_20260330_165938.log" \
    > "${LOCAL_DIR}/trial_summary.txt" 2>/dev/null || true

TRIAL_COUNT=$(wc -l < "${LOCAL_DIR}/trial_summary.txt" 2>/dev/null || echo 0)
BEST_TRIAL=$(sort -t= -k2 -rn "${LOCAL_DIR}/trial_summary.txt" 2>/dev/null | head -1)

echo ""
echo "=== Collection Complete ==="
echo "  Trials collected: ${TRIAL_COUNT} / 50"
echo "  Best trial: ${BEST_TRIAL}"
echo "  Local dir: ${LOCAL_DIR}"
echo "  Checkpoint size: $(du -sh "${LOCAL_DIR}/checkpoints/" 2>/dev/null | cut -f1)"
echo ""
echo "To resume on gpuhub-2, the best HPs can be extracted from the log or WandB."
