import argparse
import yaml
import os
import torch
import logging
import numpy as np
from pathlib import Path

from finrl_pro_ds.training.deepscalper_trainer import DeepScalperTrainer
import gymnasium as gym
import numpy as np
from finrl_pro_ds.agents.deepscalper.dqn_agent import DeepScalperDQN
from finrl_pro_ds.agents.deepscalper.policy_agents import DeepScalperPPO, DeepScalperA2C
from finrl_pro_ds.agents.deepscalper.ensemble import DeepScalperEnsemble, SynapseGatingNetwork
from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler

# Mock Env for initial testing if real env fails or for debugging
class MockDeepScalperEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(self):
        super().__init__()
        # Define spaces matching DeepScalperEnv
        self.window_size = 50
        self.micro_dim = 20
        self.observation_space = gym.spaces.Dict({
            "micro": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, self.micro_dim), dtype=np.float32),
            "macro": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32),
            "private": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        })
        self.action_space = gym.spaces.MultiDiscrete([3, 5, 5])
    
    def reset(self, seed=None, options=None):
        return {
            "micro": np.random.randn(50, 20).astype(np.float32),
            "macro": np.random.randn(11).astype(np.float32),
            "private": np.zeros(2, dtype=np.float32)
        }, {}
    
    def step(self, action):
        # Return proper types
        obs = self.reset()[0]
        reward = 1.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

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

import wandb

def setup_wandb(config):
    wandb_config = config.get("wandb", {})
    project = wandb_config.get("project", "DeepScalper_Pilot")
    tags = wandb_config.get("tags", [])
    mode = wandb_config.get("mode", "online")
    
    # User requested entity: bigcan-chiwin-technology
    # Ideally should be in config, but I will hardcode default if missing or pass it here.
    # The prompt explicitly asked to start run in this project.
    entity = wandb_config.get("entity", "bigcan-chiwin-technology")
    
    print(f"Initializing WandB: Project={project}, Entity={entity}, Mode={mode}")
    wandb.init(
        project=project,
        entity=entity,
        config=config,
        tags=tags,
        mode=mode,
        name=wandb_config.get("name", None) # Optional run name
    )

def main():
    parser = argparse.ArgumentParser(description="Train DeepScalper Agent")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--debug", action="store_true", help="Use mock environment")
    parser.add_argument("--run_name", type=str, default=None, help="WandB Run Name")
    args = parser.parse_args()

    # Load Config
    config = load_config(args.config)
    
    if args.debug:
        config["torch_compile"] = False
        print("DEBUG MODE: torch.compile disabled.")
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
    
    # Override Run Name if provided
    if args.run_name:
        if "wandb" not in config: config["wandb"] = {}
        config["wandb"]["name"] = args.run_name
    
    # Setup WandB
    setup_wandb(config)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    


    # Environment
    
    # Check for num_envs in config
    env_config = config.get("env", {})
    num_envs = env_config.get("num_envs", 1)
    
    # Define Factory
    def env_factory():
        if args.debug:
            return MockDeepScalperEnv()
        else:
            return make_env(config)
            
    if num_envs > 1:
        print(f"Vectorizing {'Mock' if args.debug else 'Real'} Environment: {num_envs} Envs")
        
        # Use Gymnasium AsyncVectorEnv (Real) or SyncVectorEnv (Debug/Fallback)
        # MockEnv is hard to pickle on Windows from __main__, so use Sync for debug.
        if args.debug:
             print("Debug Mode: Forcing SyncVectorEnv.")
             env = gym.vector.SyncVectorEnv([env_factory for _ in range(num_envs)])
        else:
            try:
                env = gym.vector.AsyncVectorEnv([env_factory for _ in range(num_envs)])
            except Exception as e:
                print(f"Failed to create AsyncVectorEnv: {e}. Fallback to Sync.")
                env = gym.vector.SyncVectorEnv([env_factory for _ in range(num_envs)])
            
        print("Vector Environment Loaded.")
    else:
        env = env_factory()
        print(f"{'Mock' if args.debug else 'Real'} Environment Loaded.")

    # Network Configs
    net_config = config.get("network", {
        "micro_config": {"input_size": 20, "hidden_size": 64},
        "macro_config": {"input_size": 11, "hidden_sizes": [64]}
    })
    
    # Create clean config for agents (remove ensemble_config if present)
    agent_net_config = net_config.copy()
    if "ensemble_config" in agent_net_config:
        del agent_net_config["ensemble_config"]
    
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
        network_config=agent_net_config, 
        lr=dqn_lr,
        gamma=dqn_gamma,
        device=device,
        **dqn_kwargs
    )
    
    # PPO/A2C currently take net_config and device. LRs are handled in Trainer now.
    ppo = DeepScalperPPO(agent_net_config, device=device)
    a2c = DeepScalperA2C(agent_net_config, device=device)
    
    # Initialize Ensemble
    ensemble_config = net_config.get("ensemble_config", {})
    gating_input = ensemble_config.get("input_size", net_config["macro_config"]["input_size"])
    gating_hidden = ensemble_config.get("hidden_size", 64)
    
    gating = SynapseGatingNetwork(input_dim=gating_input, hidden_dim=gating_hidden)
    ensemble = DeepScalperEnsemble(dqn, ppo, a2c, gating, device=device)
    
    # Initialize Trainer
    # Inject 'agents' config into 'training' config so Trainer can find it
    training_config = config.get("training", {})
    training_config["agents"] = config.get("agents", {})
    
    # Check if we should enable torch.compile (passed via config or args)
    # The config file has training.torch_compile
    
    trainer = DeepScalperTrainer(
        env=env,
        ensemble_agent=ensemble,
        config=training_config,
        device=device
    )
    
    # Start Training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted. Saving checkpoint...")
        trainer.save_checkpoint("checkpoints/interrupted_checkpoint.pth")
    finally:
        env.close()

if __name__ == "__main__":
    import gymnasium as gym # Lazy import for vector envs
    main()

