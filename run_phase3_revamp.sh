#!/bin/bash

echo "Starting Phase 3 Revamp Experiments (Algorithm Exploration)..."
echo "------------------------------------------------------------"

echo "[1/11] PPO Hybrid Baseline..."
python3 -m finrl_pro.training.run_experiment --config finrl_pro/configs/experiments/phase3_revamp/ppo_hybrid_baseline.yaml > logs_p3_ppo_base.txt 2>&1
echo "Done."

echo "[2/11] PPO Clip 0.15..."
python3 -m finrl_pro.training.run_experiment --config finrl_pro/configs/experiments/phase3_revamp/ppo_hybrid_clip_0_15.yaml > logs_p3_ppo_clip15.txt 2>&1
echo "Done."

echo "[3/11] PPO Clip 0.30..."
python3 -m finrl_pro.training.run_experiment --config finrl_pro/configs/experiments/phase3_revamp/ppo_hybrid_clip_0_30.yaml > logs_p3_ppo_clip30.txt 2>&1
echo "Done."

echo "[4/11] PPO GAE 0.98..."
python3 -m finrl_pro.training.run_experiment --config finrl_pro/configs/experiments/phase3_revamp/ppo_hybrid_gae_0_98.yaml > logs_p3_ppo_gae98.txt 2>&1
echo "Done."

echo "[5/11] PPO Ent 0.005..."
python3 -m finrl_pro.training.run_experiment --config finrl_pro/configs/experiments/phase3_revamp/ppo_hybrid_ent_0_005.yaml > logs_p3_ppo_ent005.txt 2>&1
echo "Done."

echo "[6/11] PPO Ent 0.02..."
python3 -m finrl_pro.training.run_experiment --config finrl_pro/configs/experiments/phase3_revamp/ppo_hybrid_ent_0_02.yaml > logs_p3_ppo_ent020.txt 2>&1
echo "Done."

echo "[7/11] SAC Auto Alpha..."
python3 -m finrl_pro.training.run_experiment --config finrl_pro/configs/experiments/phase3_revamp/sac_hybrid_auto.yaml > logs_p3_sac_auto.txt 2>&1
echo "Done."

echo "[8/11] SAC Fixed Alpha 0.05..."
python3 -m finrl_pro.training.run_experiment --config finrl_pro/configs/experiments/phase3_revamp/sac_hybrid_fixed_0_05.yaml > logs_p3_sac_fix05.txt 2>&1
echo "Done."

echo "[9/11] SAC Fixed Alpha 0.20..."
python3 -m finrl_pro.training.run_experiment --config finrl_pro/configs/experiments/phase3_revamp/sac_hybrid_fixed_0_20.yaml > logs_p3_sac_fix20.txt 2>&1
echo "Done."

echo "[10/11] TD3 Noise 0.10..."
python3 -m finrl_pro.training.run_experiment --config finrl_pro/configs/experiments/phase3_revamp/td3_hybrid_noise_0_10.yaml > logs_p3_td3_noise10.txt 2>&1
echo "Done."

echo "[11/11] TD3 Noise 0.25..."
python3 -m finrl_pro.training.run_experiment --config finrl_pro/configs/experiments/phase3_revamp/td3_hybrid_noise_0_25.yaml > logs_p3_td3_noise25.txt 2>&1
echo "Done."

echo "------------------------------------------------------------"
echo "Phase 3 Revamp Experiments Complete."
