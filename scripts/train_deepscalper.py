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
        # TODO: Load real environment using factory
        # from finrl_pro_ds.envs.factory import make_env
        # env = make_env(config['env'])
        raise NotImplementedError("Real environment loading not yet integrated. Use --debug to run with MockEnv.")

    # Network Configs
    net_config = config.get("network", {
        "micro_config": {"input_size": 20, "hidden_size": 64},
        "macro_config": {"input_size": 11, "hidden_sizes": [64]}
    })
    
    # Initialize Agents
    dqn = DeepScalperDQN(net_config, device=device)
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
