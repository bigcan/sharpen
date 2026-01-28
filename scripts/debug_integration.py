import argparse
import sys
import os
import yaml
import numpy as np

# Replicate Import Order of train_deepscalper.py (approx)
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import pandas as pd
import wandb # Import wandb to match trainer deps (but no login)

from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.agents.deepscalper.dqn_agent import DeepScalperDQN
from finrl_pro_ds.agents.deepscalper.policy_agents import DeepScalperPPO, DeepScalperA2C
from finrl_pro_ds.agents.deepscalper.ensemble import DeepScalperEnsemble, SynapseGatingNetwork
from finrl_pro_ds.training.deepscalper_trainer import DeepScalperTrainer
from finrl_pro_ds.mlops.logger import MLOpsLogger

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("Starting Integration Test (Torch + Env + Agents)...", flush=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_name", type=str)
    args = parser.parse_args()

    # Device Check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    # Load Config
    config = load_config(args.config)
    print("Config Loaded.", flush=True)

    # Create Env
    print("Creating Environment...", flush=True)
    try:
        env = DeepScalperEnv(config)
        obs, info = env.reset()
        print("Environment Created and Reset.", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Env Creation FAILED: {e}", flush=True)
        return

    # Create Agents
    print("Creating Agents...", flush=True)
    try:
        # Mocking or using config
        # We need correct input dims
        micro_shape = (env.window_size, len(config['features']['micro_features'])) 
        macro_shape = (len(config['features']['macro_features']),)
        private_shape = (env.window_size, 2)
        
        # DeepScalperEnsemble handles init
        ensemble = DeepScalperEnsemble(config, device=device)
        print("Agents Created.", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Agent Creation FAILED: {e}", flush=True)
        return

    # Create Trainer
    print("Creating Trainer...", flush=True)
    try:
        logger = MLOpsLogger()
        trainer = DeepScalperTrainer(env, ensemble, config, logger, device=device)
        print("Trainer Created.", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Trainer Creation FAILED: {e}", flush=True)
        return
    
    # Train 1 Step
    print("Running Trainer.train() for 1 step logic (loop start)...", flush=True)
    try:
        # We mock total_timesteps to 10 to exit quickly
        trainer.total_timesteps = 10 
        avg_reward = trainer.train()
        print(f"Training Loop Finished. Avg Reward: {avg_reward}", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Training Loop FAILED: {e}", flush=True)
        return

    print("Integration Debug SUCCESS.", flush=True)

if __name__ == "__main__":
    main()
