import argparse
import yaml
import os
from datetime import datetime
import pandas as pd # Explicitly import pandas/pyarrow BEFORE torch to avoid ABI crash
# import pyarrow # REMOVED: Testing if this caused conflict
import torch
import logging
import numpy as np
import atexit
from pathlib import Path

from finrl_pro_ds.training.deepscalper_trainer import DeepScalperTrainer
import gymnasium as gym
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
            "private": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, 2), dtype=np.float32)
        })
        self.action_space = gym.spaces.MultiDiscrete([3, 5, 5])
    
    def reset(self, seed=None, options=None):
        return {
            "micro": np.random.randn(50, 20).astype(np.float32),
            "macro": np.random.randn(11).astype(np.float32),
            "private": np.zeros((50, 2), dtype=np.float32)
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
        feature_config=config.get("features", {}),
        shared_memory_config=config.get("data", {}).get("shared_memory_config")
    )
    
    # 3. Init Env
    env_config = config.get("env", {})
    # Inject reward config if it exists at top level, or ensure it's passed
    env_config["reward"] = config.get("reward", {})
    
    env = DeepScalperEnv(config=env_config, data_handler=handler)
    return env

def create_env(config, debug=False):
    """Top-level wrapper for env creation to ensure picklability."""
    if debug:
        return MockDeepScalperEnv()
    return make_env(config)

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
    entity = wandb_config.get("entity", "bigcan-chiwin-technology")
    
    # Enforce Naming Convention
    run_name = wandb_config.get("name")
    if not run_name:
        from finrl_pro_ds.utils.naming import generate_run_name
        run_name = generate_run_name(version="V1", platform="GPUHub")
        print(f"Auto-generated Canonical Run Name: {run_name}")
    
    print(f"Initializing WandB: Project={project}, Entity={entity}, Mode={mode}, Name={run_name}")
    wandb.init(
        project=project,
        entity=entity,
        config=config,
        tags=tags,
        mode=mode,
        name=run_name
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
        if "training" in config:
            config["training"]["torch_compile"] = False
            
        print("DEBUG MODE: torch.compile disabled.")
        # import torch._dynamo  <-- removed to prevent shadowing
        if hasattr(torch, "_dynamo"):
            torch._dynamo.config.suppress_errors = True
    
    # Override Run Name if provided
    if args.run_name:
        if "wandb" not in config: config["wandb"] = {}
        config["wandb"]["name"] = args.run_name
    
    # Setup WandB
    setup_wandb(config)

    # Pre-load Data for Shared Memory (Optimization)
    data_loader = None
    
    # Emergency cleanup for abnormal exits (SIGTERM, etc.)
    def _emergency_shm_cleanup():
        if data_loader:
            print("[atexit] Cleaning up Shared Memory...")
            data_loader.close_shared_memory(unlink=True)
    atexit.register(_emergency_shm_cleanup)
    
    # Respect use_shm config (default False if not specified)
    use_shm_config = config.get("training", {}).get("use_shm", False)
    
    if config.get("env", {}).get("num_envs", 1) > 1 and not args.debug and use_shm_config:
        print("Initializing Shared Memory for Vector Env...")
        data_config = config.get("data", {})
        file_path = data_config.get("file_path")
        
        # Load once in main process
        data_loader = ParquetDataHandler(
            file_path=file_path, 
            ticker=data_config.get("ticker", "BTCUSDT"), 
            feature_config=config.get("features", {})
        )
        # Create SHM and inject into config
        shm_config = data_loader.create_shared_memory()
        data_config["shared_memory_config"] = shm_config
        print(f"Shared Memory Initialized. Config injected.")
    
    # Device
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {device}")
    # MOVED DOWN after Env creation
    


    # Environment
    
    # Check for num_envs in config
    env_config = config.get("env", {})
    num_envs = env_config.get("num_envs", 1)
    
    # Define Factory
    # Define Factory using functools.partial (Must be picklable for spawn)
    import functools
    
    # Use the top-level create_env helper
    env_factory = functools.partial(create_env, config=config, debug=args.debug)
            
    # CRITICAL FIX for Multiprocessing with CUDA:
    # 1. Do not initialize CUDA before forking/spawning if possible.
    # 2. Use 'spawn' context to avoid CUDA context corruption in workers.
    # We delay device init until after env creation (though we still assign it later).
    
    if num_envs > 1:
        print(f"Vectorizing {'Mock' if args.debug else 'Real'} Environment: {num_envs} Envs")
        
        if args.debug:
             print("Debug Mode: Forcing SyncVectorEnv.")
             env = gym.vector.SyncVectorEnv([env_factory for _ in range(num_envs)])
        else:
            try:
                # Staggered spawning to avoid CPU/RAM spike that causes BrokenPipeError
                # Each env waits based on its batch before loading data
                batch_size = 6
                delay_per_batch = 2.0  # seconds between batches
                
                def make_staggered_factory(index, base_factory, batch_size, delay_per_batch):
                    """Create a factory that delays based on batch index to stagger data loading."""
                    def staggered_factory():
                        import time
                        batch_num = index // batch_size
                        delay = batch_num * delay_per_batch
                        if delay > 0:
                            time.sleep(delay)
                        return base_factory()
                    return staggered_factory
                
                staggered_factories = [
                    make_staggered_factory(i, env_factory, batch_size, delay_per_batch)
                    for i in range(num_envs)
                ]
                
                total_batches = (num_envs + batch_size - 1) // batch_size
                est_time = (total_batches - 1) * delay_per_batch
                print(f"Using staggered AsyncVectorEnv: {num_envs} envs in {total_batches} batches of {batch_size}")
                print(f"Estimated stagger time: {est_time:.1f}s to spread CPU/RAM load...")
                
                env = gym.vector.AsyncVectorEnv(staggered_factories, context="spawn", shared_memory=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Failed to create AsyncVectorEnv: {e}. Fallback to Sync.")
                env = gym.vector.SyncVectorEnv([env_factory for _ in range(num_envs)])
            
        print("Vector Environment Loaded.")
    else:
        env = env_factory()
        print(f"{'Mock' if args.debug else 'Real'} Environment Loaded.")

    # Device Setup - moved AFTER Env creation to check for CUDA safely
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Network Configs
    # Network Configs
    raw_net_config = config.get("network", {})
    
    # Handle Flat YAML Config -> Nested Config for DeepScalperNetwork
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
            },
            # Preserve potential ensemble settings
            "ensemble_config": raw_net_config.get("ensemble_config", {})
        }
    else:
        net_config = raw_net_config
    
    # Create clean config for agents (remove ensemble_config if present)
    agent_net_config = net_config.copy()
    if "ensemble_config" in agent_net_config:
        del agent_net_config["ensemble_config"]
    
    # Initialize Agents with Specific Configs
    agents_config = config.get("agents", {})
    
    # Sanitize Config Types (Fix for YAML string parsing issues)
    def sanitize_config(cfg):
        for k, v in cfg.items():
            if isinstance(v, dict):
                sanitize_config(v)
            elif k in ["learning_rate", "gamma", "entropy_coef", "gae_lambda", "clip_epsilon", "max_grad_norm"]:
                try:
                    cfg[k] = float(v)
                except:
                    pass
    
    sanitize_config(agents_config)
    
    dqn_config = agents_config.get("dqn", {})
    
    # DQN expects network_config + its own params. 
    # We combine them or pass specific args. 
    # DQN signature: (network_config, lr, gamma, etc.)
    # We can pass kwargs from dqn_config
    
    # Extract known args for DQN
    dqn_lr = float(dqn_config.get("learning_rate", 1e-4)) # Fallback
    dqn_gamma = float(dqn_config.get("gamma", 0.99))
    # Passed as kwargs to dqn
    # Map prefixed config keys to agent init args
    dqn_kwargs = {}
    
    # Key Mapping (Config -> Init Arg)
    key_map = {
        "dqn_batch_size": "batch_size",
        "dqn_buffer_size": "buffer_size",
        "dqn_target_update_freq": "target_update_freq",
        "dqn_epsilon_start": "epsilon_start",
        "dqn_epsilon_end": "epsilon_end", 
        "dqn_epsilon_decay": "epsilon_decay"
    }

    for k, v in dqn_config.items():
        if k in ["learning_rate", "gamma"]:
            continue
            
        if k in key_map:
            dqn_kwargs[key_map[k]] = v
        else:
            # Pass through other keys 
            dqn_kwargs[k] = v
    
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
        if data_loader:
            print("Cleaning up Shared Memory...")
            data_loader.close_shared_memory(unlink=True)

if __name__ == "__main__":
    import gymnasium as gym # Lazy import for vector envs
    main()

