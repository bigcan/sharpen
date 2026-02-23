#!/bin/bash
cd /workspace/DeepScalper
mkdir -p logs checkpoints
export PATH=/root/miniconda3/bin:$PATH
export WANDB_API_KEY=$(grep WANDB_API_KEY .env | cut -d= -f2 | tr -d '[:space:]')

# Kill any leftover training processes
pkill -f run_full_pipeline 2>/dev/null || true
sleep 2

echo "=== Launching B13 (Inventory Penalty) ==="
nohup python -u scripts/run_full_pipeline.py \
  --config configs/phase_b13_inventory_penalty_5min.yaml \
  --run_name DeepScalper_V1_GPUHub_20260222_B13 \
  > logs/b13_inventory_penalty.log 2>&1 &
B13PID=$!
echo $B13PID > b13.pid
echo "B13 PID=$B13PID"

sleep 5

echo "=== Launching B14 (CRRA Utility) ==="
nohup python -u scripts/run_full_pipeline.py \
  --config configs/phase_b14_crra_utility_5min.yaml \
  --run_name DeepScalper_V1_GPUHub_20260222_B14 \
  > logs/b14_crra_utility.log 2>&1 &
B14PID=$!
echo $B14PID > b14.pid
echo "B14 PID=$B14PID"

# Wait for both to initialize
sleep 40

echo ""
echo "=== Process Check ==="
ps aux | grep run_full_pipeline | grep -v grep

echo ""
echo "=== GPU Check ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

echo ""
echo "=== B13 Log Tail ==="
tail -5 logs/b13_inventory_penalty.log

echo ""
echo "=== B14 Log Tail ==="
tail -5 logs/b14_crra_utility.log
