#!/bin/bash
set -e

echo "=== FinRL Pro Phase 6: The Real Data Tournament ==="
echo "Dataset: data/sp500_full_2010_2025.parquet"
echo "Period: 2010-2020 (Training)"

echo "---------------------------------------------------"
echo "1. Running PPO Agent..."
python -m finrl_pro.training.run_experiment --config finrl_pro/configs/experiments/phase6_tournament_ppo.yaml

echo "---------------------------------------------------"
echo "2. Running SAC Agent..."
python -m finrl_pro.training.run_experiment --config finrl_pro/configs/experiments/phase6_tournament_sac.yaml

echo "---------------------------------------------------"
echo "3. Running DDPG Agent..."
python -m finrl_pro.training.run_experiment --config finrl_pro/configs/experiments/phase6_tournament_ddpg.yaml

echo "---------------------------------------------------"
echo "Tournament Complete. Check 'reports/' for results."
