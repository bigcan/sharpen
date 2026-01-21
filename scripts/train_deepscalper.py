import argparse
import yaml
import os
import torch
import logging
import numpy as np
from pathlib import Path

from finrl_pro_ds.training.deepscalper_trainer import DeepScalperTrainer
from finrl_pro_ds.agents.deepscalper.dqn_agent import DeepScalperDQN
from finrl_pro_ds.agents.deepscalper.policy_agents import DeepScalperPPO, DeepScalperA2C
from finrl_pro_ds.agents.deepscalper.ensemble import DeepScalperEnsemble, SynapseGatingNetwork
from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler

# Mock Env for initial testing if real env fails or for debugging
class MockDeepScalperEnv:
    def __init__(self):
        self.observation_space = None # Placeholder
        self.action_space = None 
    
    def reset(self):
        # Micro: (20, 20), Macro: (11,)
        return {
            "micro": np.random.randn(20, 20),
            "macro": np.random.randn(11)
        }, {}
    
    def step(self, action):
        return self.reset()[0], 1.0, False, False, {}

def make_env(config):
    """Factory to create DeepScalperEnv with Real Data"""
    # 1. Get Data Config
    data_config = config.get("data", {})
    file_path = data_config.get("file_path")
    ticker = data_config.get("ticker", "BTCUSDT")
    
    if not file_path or not os.path.exists(file_path):
        raise ValueError(f"Invalid data file path: {file_path}")
        
    # 2. Init Data Handler
    handler = ParquetDataHandler(
        file_path=file_path,
        ticker=ticker,
        feature_config=config.get("features", {})
    )
    
    # 3. Init Env
    env_config = config.get("env", {})
    # Inject reward config if it exists at top level, or ensure it's passed
    env_config["reward"] = config.get("reward", {})
    
    env = DeepScalperEnv(config=env_config, data_handler=handler)
    return env

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Train DeepScalper Agent")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--debug", action="store_true", help="Use mock environment")
    args = parser.parse_args()

    # Load Config
    config = load_config(args.config)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Environment
    if args.debug:
        print("DEBUG MODE: Using Mock Environment")
        env = MockDeepScalperEnv()
    else:
        print("Loading Real Environment...")
        env = make_env(config)
        print("Real Environment Loaded.")

    # Network Configs

    # Network Configs
    net_config = config.get("network", {
        "micro_config": {"input_size": 20, "hidden_size": 64},
        "macro_config": {"input_size": 11, "hidden_sizes": [64]}
    })
    
    # Initialize Agents with Specific Configs
    agents_config = config.get("agents", {})
    dqn_config = agents_config.get("dqn", {})
    
    # DQN expects network_config + its own params. 
    # We combine them or pass specific args. 
    # DQN signature: (network_config, lr, gamma, etc.)
    # We can pass kwargs from dqn_config
    
    # Extract known args for DQN
    dqn_lr = dqn_config.get("learning_rate", 1e-4) # Fallback
    dqn_gamma = dqn_config.get("gamma", 0.99)
    # Passed as kwargs to dqn
    dqn_kwargs = {k:v for k,v in dqn_config.items() if k not in ["learning_rate", "gamma"]}
    
    dqn = DeepScalperDQN(
        network_config=net_config, 
        lr=dqn_lr,
        gamma=dqn_gamma,
        device=device,
        **dqn_kwargs
    )
    
    # PPO/A2C currently take net_config and device. LRs are handled in Trainer now.
    ppo = DeepScalperPPO(net_config, device=device)
    a2c = DeepScalperA2C(net_config, device=device)
    
    # Initialize Ensemble
    gating = SynapseGatingNetwork(input_dim=net_config["macro_config"]["input_size"])
    ensemble = DeepScalperEnsemble(dqn, ppo, a2c, gating, device=device)
    
    # Initialize Trainer
    trainer = DeepScalperTrainer(
        env=env,
        ensemble_agent=ensemble,
        config=config.get("training", {}),
        device=device
    )
    
    # Start Training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted. Saving checkpoint...")
        trainer.save_checkpoint("interrupted_checkpoint.pth")

if __name__ == "__main__":
    main()
