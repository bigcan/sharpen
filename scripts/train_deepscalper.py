import argparse
import yaml
import os
import torch
import logging
import numpy as np
import atexit
import wandb
import sys
import gymnasium as gym
import pickle  # FIX: Added for exception handling

from finrl_pro_ds.training.deepscalper_trainer import DeepScalperTrainer
from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def setup_wandb(config, run_name=None):
    wandb_config = config.get("wandb", {})
    project = wandb_config.get("project", "FinRL-Pro-DS")
    tags = wandb_config.get("tags", [])
    mode = wandb_config.get("mode", "online")
    entity = wandb_config.get("entity", "bigcan-chiwin-technology")
    
    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  CANONICAL NAMING: DeepScalper_V{version}_{Platform}_{YYYYMMDD}_{HHMM}║
    # ║  NO SUFFIXES - use WandB tags for metadata (Pilot, HPO, etc.)        ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    from finrl_pro_ds.utils.naming import generate_run_name, validate_run_name
    
    if not run_name:
        # os is imported at top of file
        platform = "GPUHub" if os.path.exists("/workspace") else "Local"
        run_name = generate_run_name(version="V1", platform=platform)
    else:
        # Validate user-provided name follows canonical format
        # This will raise ValueError if name has suffixes
        validate_run_name(run_name, raise_on_fail=True)
    
    wandb_config["name"] = run_name
        
    wandb.init(
        project=project,
        entity=entity,
        config=config,
        tags=tags,
        mode=mode,
        name=run_name,
        resume="allow"
    )
    return wandb.run.name

def make_env(config):
    """Factory to create DeepScalperEnv with Real Data"""
    # 1. Get Data Config
    data_config = config.get("data", {})
    file_path = data_config.get("file_path")
    ticker = data_config.get("ticker", "BTCUSDT")
    
    if not file_path or not os.path.exists(file_path):
        # Fallback for testing if file doesn't exist? Or crash.
        # Check if debug mode might imply mock data
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
    # Inject reward config
    env_config["reward"] = config.get("env", {}).get("reward", {})
    
    env = DeepScalperEnv(config=env_config, data_handler=handler)
    return env

def create_env(config, debug=False):
    if debug:
        # Simple Mock
        class MockEnv(gym.Env):
            metadata = {"render_modes": ["human"]}
            def __init__(self):
                self.observation_space = gym.spaces.Dict({
                    "micro": gym.spaces.Box(-np.inf, np.inf, (50, 20)),
                    "macro": gym.spaces.Box(-np.inf, np.inf, (11,)),
                    "private": gym.spaces.Box(-np.inf, np.inf, (50, 2))
                })
                self.action_space = gym.spaces.MultiDiscrete([3, 5, 5])
            def reset(self, **kwargs):
                return {
                    "micro": np.random.randn(50, 20).astype(np.float32),
                    "macro": np.random.randn(11).astype(np.float32),
                    "private": np.zeros((50, 2), dtype=np.float32)
                }, {}
            def step(self, action):
                return self.reset()[0], 1.0, False, False, {"volatility_target": 0.001}
        return MockEnv()
    return make_env(config)

def main():
    parser = argparse.ArgumentParser(description="Train DeepScalper Single BDQ")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--debug", action="store_true", help="Use mock environment")
    parser.add_argument("--run_name", type=str, default=None, help="WandB Run Name")
    parser.add_argument("--tags", nargs="*", default=[], help="List of WandB tags")
    parser.add_argument("--steps", type=int, default=None, help="Override total_timesteps (e.g. for smoke test)")
    args = parser.parse_args()

    # Load Config
    config = load_config(args.config)
    
    # Merge CLI args
    if args.steps:
        print(f"Overriding total_timesteps: {args.steps}")
        config["training"]["total_timesteps"] = args.steps
        
    # Inject CLI tags into config for setup_wandb
    if args.tags:
        config.setdefault("wandb", {})["tags"] = args.tags

    run_name = args.run_name  # Can be None, will be auto-generated if so
    
    # Setup WandB
    run_name = setup_wandb(config, run_name=args.run_name)
    print(f"WANDB_RUN_ID: {wandb.run.id}")
    sys.stdout.flush()
    
    # Pre-load Data for Shared Memory (Optimization)
    data_loader = None
    use_shm = config.get("training", {}).get("use_shm", False)
    num_envs = config.get("env", {}).get("num_envs", 1)
    
    if num_envs > 1 and not args.debug and use_shm:
        print("Initializing Shared Memory for Vector Env...")
        data_config = config.get("data", {})
        file_path = data_config.get("file_path")
        
        data_loader = ParquetDataHandler(
            file_path=file_path, 
            ticker=data_config.get("ticker", "BTCUSDT"), 
            feature_config=config.get("features", {})
        )
        shm_config = data_loader.create_shared_memory()
        data_config["shared_memory_config"] = shm_config
        
        # Register cleanup
        def cleanup():
            if data_loader:
                data_loader.close_shared_memory(unlink=True)
        atexit.register(cleanup)
        
    # Env Factory
    import functools
    env_factory = functools.partial(create_env, config=config, debug=args.debug)
    
    if num_envs > 1:
        if args.debug:
             env = gym.vector.SyncVectorEnv([env_factory for _ in range(num_envs)])
        else:
             # Use Sync for stability unless Async needed
             # For 16 envs, Async 'spawn' is better but harder to setup without proper main guard
             # We use Sync for simplicity/robustness in this update unless performance is blocked
             # But let's try Async with spawn context if possible, or Sync.
             # Given previous code had complex staggering, let's stick to Sync for 16 envs on Windows/PyTorch to avoid Pickle hell?
             # Actually, config said "num_envs: 16". Sync might be slow.
             # Using Async with 'spawn' context.
             try:
                 env = gym.vector.AsyncVectorEnv(
                     [env_factory for _ in range(num_envs)], 
                     context="spawn", 
                     shared_memory=use_shm
                 )
             except (RuntimeError, pickle.PicklingError, AttributeError) as e:
                 print(f"AsyncVectorEnv failed ({type(e).__name__}: {e}), falling back to Sync.")
                 env = gym.vector.SyncVectorEnv([env_factory for _ in range(num_envs)])
    else:
        # Force SyncVectorEnv to maintain (1, ...) shapes and array rewards
        env = gym.vector.SyncVectorEnv([env_factory])
        
    print(f"Environment Loaded: {env}")
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Trainer
    trainer = DeepScalperTrainer(env, config, device=device, run_name=run_name)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        env.close()
        if data_loader:
            data_loader.close_shared_memory(unlink=True)

if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
