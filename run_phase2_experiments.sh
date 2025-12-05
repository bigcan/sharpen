#!/bin/bash
set -e

echo "Starting Phase 2 Experiments..."
echo "--------------------------------"

echo "[1/6] Running Standard Baseline (Seed 41)..."
python3 -m finrl_pro.training.run_experiment --config finrl_pro/configs/experiments/phase2_revamp/standard_baseline_seed_41.yaml > logs_std_41.txt 2>&1
echo "Done. Log: logs_std_41.txt"

echo "[2/6] Running Standard Baseline (Seed 42)..."
python3 -m finrl_pro.training.run_experiment --config finrl_pro/configs/experiments/phase2_revamp/standard_baseline_seed_42.yaml > logs_std_42.txt 2>&1
echo "Done. Log: logs_std_42.txt"

echo "[3/6] Running Standard Baseline (Seed 43)..."
python3 -m finrl_pro.training.run_experiment --config finrl_pro/configs/experiments/phase2_revamp/standard_baseline_seed_43.yaml > logs_std_43.txt 2>&1
echo "Done. Log: logs_std_43.txt"

echo "[4/6] Running Hybrid Baseline (Seed 41)..."
python3 -m finrl_pro.training.run_experiment --config finrl_pro/configs/experiments/phase2_revamp/hybrid_baseline_seed_41.yaml > logs_hybrid_41.txt 2>&1
echo "Done. Log: logs_hybrid_41.txt"

echo "[5/6] Running Hybrid Baseline (Seed 42)..."
python3 -m finrl_pro.training.run_experiment --config finrl_pro/configs/experiments/phase2_revamp/hybrid_baseline_seed_42.yaml > logs_hybrid_42.txt 2>&1
echo "Done. Log: logs_hybrid_42.txt"

echo "[6/6] Running Hybrid Baseline (Seed 43)..."
python3 -m finrl_pro.training.run_experiment --config finrl_pro/configs/experiments/phase2_revamp/hybrid_baseline_seed_43.yaml > logs_hybrid_43.txt 2>&1
echo "Done. Log: logs_hybrid_43.txt"

echo "--------------------------------"
echo "All experiments completed successfully."