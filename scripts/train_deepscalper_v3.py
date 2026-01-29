import argparse
import sys
import os

# Critical Fix: Import Torch BEFORE Pandas/PyArrow to prevent ABI conflicts
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Data imports after Torch
import pandas as pd
import yaml
import logging
import wandb
from pathlib import Path

# Imports
from finrl_pro_ds.training.deepscalper_trainer import DeepScalperTrainer
import gymnasium as gym
from finrl_pro_ds.agents.deepscalper.dqn_agent import DeepScalperDQN
from finrl_pro_ds.agents.deepscalper.policy_agents import DeepScalperPPO, DeepScalperA2C
from finrl_pro_ds.agents.deepscalper.ensemble import DeepScalperEnsemble
from finrl_pro_ds.networks.gating import DeepScalperGatingNetwork
from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler

# Helper functions
def make_env(config):
    data_config = config.get("data", {})
    file_path = data_config.get("file_path")
    ticker = data_config.get("ticker", "BTCUSDT")
    
    if not file_path or not os.path.exists(file_path):
        raise ValueError(f"Invalid data file path: {file_path}")
        
    handler = ParquetDataHandler(
        file_path=file_path,
        ticker=ticker,
        feature_config=config.get("features", {}),
        shared_memory_config=config.get("data", {}).get("shared_memory_config")
    )
    
    env_config = config.get("env", {})
    env_config["reward"] = config.get("reward", {})
    
    env = DeepScalperEnv(config=env_config, data_handler=handler)
    return env

def create_env(config, debug=False):
    # Simplified factory
    return make_env(config)

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# Main
def main():
    parser = argparse.ArgumentParser(description="Train DeepScalper V3 (Fixed)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    print("Starting DeepScalper V3 (Import Order Fixed)...", flush=True)

    # Load Config
    config = load_config(args.config)
    
    # Initialize WandB
    wandb_config = config.get("wandb", {})
    # Use CLI run_name if provided, else from config, else None (auto-generated)
    run_name = args.run_name 
    
    print(f"Initializing WandB run: {run_name}", flush=True)
    wandb.init(
        project=wandb_config.get("project", "FinRL-Pro-DS"),
        entity=wandb_config.get("entity"),
        mode=wandb_config.get("mode", "online"),
        tags=wandb_config.get("tags", []),
        name=run_name,
        config=config
    )
    
    if args.debug:
        config["torch_compile"] = False
        config["training"]["torch_compile"] = False
        print("DEBUG MODE: torch.compile disabled.", flush=True)

    # Environment
    print("Creating Environment...", flush=True)
    try:
        if config.get("env", {}).get("num_envs", 1) > 1 and not args.debug:
             # Just use SyncVectorEnv to avoid MP crashes for now if > 1
             import functools
             env_factory = functools.partial(create_env, config=config, debug=args.debug)
             num_envs = config.get("env", {})["num_envs"]
             env = gym.vector.SyncVectorEnv([env_factory for _ in range(num_envs)])
        else:
             env = create_env(config, debug=args.debug)
        print("Environment Created.", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FATAL: Environment Creation Failed: {e}", flush=True)
        return

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    
    # Agents Init (From V1)
    print("Initializing Agents...", flush=True)
    try:
        raw_net_config = config.get("network", {})
        if "micro_config" not in raw_net_config:
            hidden_size = raw_net_config.get("hidden_size", 64)
            net_config = {
                "micro_config": {
                    "input_size": raw_net_config.get("micro_input_size", 20),
                    "private_input_size": raw_net_config.get("private_input_size", 2),
                    "hidden_size": hidden_size
                },
                "macro_config": {
                    "input_size": raw_net_config.get("macro_input_size", 11),
                    "hidden_sizes": [hidden_size]
                }
            }
        else:
            net_config = raw_net_config
        
        agent_net_config = net_config.copy()
        if "ensemble_config" in agent_net_config:
            del agent_net_config["ensemble_config"]
            
        agents_config = config.get("agents", {})
        
        # Sanitize
        def sanitize_config(cfg):
            for k, v in cfg.items():
                if isinstance(v, dict): protocol = sanitize_config(v)
                elif k in ["learning_rate", "gamma", "entropy_coef", "gae_lambda", "clip_epsilon", "max_grad_norm"]:
                    try: cfg[k] = float(v)
                    except: pass
        sanitize_config(agents_config)
        
        dqn_config = agents_config.get("dqn", {})
        dqn_lr = float(dqn_config.get("learning_rate", 1e-4))
        dqn_gamma = float(dqn_config.get("gamma", 0.99))
        
        dqn_kwargs = {}
        key_map = {
            "dqn_batch_size": "batch_size",
            "dqn_buffer_size": "buffer_size",
            "dqn_target_update_freq": "target_update_freq",
            "dqn_epsilon_start": "epsilon_start",
             "dqn_epsilon_end": "epsilon_end", 
            "dqn_epsilon_decay": "epsilon_decay"
        }
        for k, v in dqn_config.items():
            if k in ["learning_rate", "gamma"]: continue
            if k in key_map: dqn_kwargs[key_map[k]] = v
            else: dqn_kwargs[k] = v
            
        dqn = DeepScalperDQN(agent_net_config, lr=dqn_lr, gamma=dqn_gamma, device=device, **dqn_kwargs)
        ppo = DeepScalperPPO(agent_net_config, device=device)
        a2c = DeepScalperA2C(agent_net_config, device=device)
        
        ensemble_config = net_config.get("ensemble_config", {})
        gating_input = ensemble_config.get("input_size", net_config["macro_config"]["input_size"])
        gating = DeepScalperGatingNetwork(input_dim=gating_input, hidden_dim=64)
        
        ensemble = DeepScalperEnsemble(dqn, ppo, a2c, gating, device=device)
        print("Agents Initialized.", flush=True)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FATAL: Agent Init Failed: {e}", flush=True)
        return

    # Trainer
    try:
        training_config = config.get("training", {})
        training_config["agents"] = config.get("agents", {})
        
        # Disable torch compile for safety
        config["torch_compile"] = False
        
        trainer = DeepScalperTrainer(env, ensemble, training_config, device=device)
        print("Trainer Initialized. Starting Training Loop...", flush=True)
        
        trainer.train()
        
    except KeyboardInterrupt:
        print("Interrupted.", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FATAL: Training Failed: {e}", flush=True)
    finally:
        print("Cleaning up...", flush=True)
        env.close()

if __name__ == "__main__":
    main()
