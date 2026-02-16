#!/bin/bash
# Launch BDQ A/B test in parallel workspace
set -e

export PATH=/root/miniconda3/bin:$PATH

cd /workspace/DeepScalper_BDQ

# Source WANDB key from main workspace
source /workspace/DeepScalper/.env 2>/dev/null || true
export WANDB_API_KEY

# Install package
pip install -e . -q 2>&1 | tail -3

# Launch BDQ with fresh HPO
echo "Launching BDQ A/B test..."
nohup python -u scripts/run_full_pipeline.py \
    --config configs/deepscalper_dev.yaml \
    --fresh_hpo \
    --run_name "DeepScalper_BDQ_AB_20260216" \
    --tags v2_bdq_ab_test \
    > run_bdq.log 2>&1 &

echo "BDQ_PID=$!"
echo $! > run_bdq.pid
echo "Launch complete. Checking process..."
sleep 2
ps -p $! -o pid,cmd || echo "WARNING: Process died"
