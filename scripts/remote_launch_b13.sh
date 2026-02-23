#!/bin/bash
cd /workspace/DeepScalper
mkdir -p logs
export PATH=/root/miniconda3/bin:$PATH
export WANDB_API_KEY=$(grep WANDB_API_KEY .env | cut -d= -f2 | tr -d '[:space:]')
export WANDB_MODE=online

# Kill any leftover python training processes
pkill -f run_full_pipeline || true
sleep 2

# Launch B13 only
nohup python -u scripts/run_full_pipeline.py \
  --config configs/phase_b13_inventory_penalty_5min.yaml \
  --run_name DeepScalper_V1_GPUHub_20260222_B13 \
  > logs/b13_inventory_penalty.log 2>&1 &
echo $! > b13.pid
echo "B13_PID=$(cat b13.pid)"

# Wait for training to actually start
sleep 30

echo "=== Process check ==="
ps aux | grep run_full_pipeline | grep -v grep

echo "=== GPU check ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

echo "=== Log tail ==="
tail -10 logs/b13_inventory_penalty.log
