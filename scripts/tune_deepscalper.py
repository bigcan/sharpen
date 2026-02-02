#!/usr/bin/env python
"""
DeepScalper HPO using Optuna with TPE Sampler.
Optimizes hindsight parameters and auxiliary loss weight.
"""
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import yaml
import argparse
import os
import sys
import copy
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
import functools
import logging

# Needed to import project modules
sys.path.append(os.getcwd())

from finrl_pro_ds.training.deepscalper_trainer import DeepScalperTrainer
from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HPO")


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env(config, start_date=None, end_date=None):
    """Factory to create DeepScalperEnv with Real Data for HPO."""
    data_config = config.get("data", {})
    file_path = data_config.get("file_path")
    ticker = data_config.get("ticker", "BTCUSDT")
    
    if not file_path or not os.path.exists(file_path):
        raise ValueError(f"Invalid data file path: {file_path}")
    
    # Use provided dates or fall back to config
    sd = start_date or data_config.get("train_start_date")
    ed = end_date or data_config.get("train_end_date")
    
    handler = ParquetDataHandler(
        file_path=file_path,
        ticker=ticker,
        feature_config=config.get("features", {}),
        start_date=sd,
        end_date=ed
    )
    
    env_config = config.get("env", {})
    env_config["reward"] = config.get("env", {}).get("reward", {})
    
    env = DeepScalperEnv(config=env_config, data_handler=handler)
    return env


def create_env_factory(config, start_date=None, end_date=None):
    """Create a picklable env factory for VectorEnv."""
    return functools.partial(make_env, config=config, start_date=start_date, end_date=end_date)


def evaluate_agent(env, agent, num_episodes=1, max_steps=10000):
    """Evaluate agent and compute Sharpe-like metric."""
    episode_returns = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        step = 0
        
        while not done and step < max_steps:
            # Extract tensors
            micro = torch.tensor(obs["micro"], dtype=torch.float32).unsqueeze(0).to(agent.device)
            private = torch.tensor(obs["private"], dtype=torch.float32).unsqueeze(0).to(agent.device)
            macro = torch.tensor(obs["macro"], dtype=torch.float32).unsqueeze(0).to(agent.device)
            
            action = agent.predict(micro, private, macro, deterministic=True)
            obs, reward, term, trunc, _ = env.step(action[0])
            done = term or trunc
            episode_reward += reward
            step += 1
        
        episode_returns.append(episode_reward)
    
    # Compute Sharpe-like metric (mean/std of returns)
    returns = np.array(episode_returns)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns)
    else:
        sharpe = np.mean(returns) / 100.0  # Fallback for single episode
    
    return sharpe


def objective(trial, base_config, args):
    """Optuna objective function for a single trial."""
    # === PAPER-ALIGNED: Hindsight Params ===
    hindsight_horizon = trial.suggest_categorical("hindsight_horizon", [60, 120, 180, 240])
    hindsight_weight = trial.suggest_float("hindsight_weight", 0.001, 0.1, log=True)
    auxiliary_weight = trial.suggest_categorical("auxiliary_weight", [0.5, 1.0])
    
    # === GUIDE-RECOMMENDED: Critical Agent Params ===
    # Rank 1: Learning Rate (Very High sensitivity)
    learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
    
    # Rank 2: Target Update Interval (High sensitivity)
    target_update_interval = trial.suggest_categorical("target_update_interval", [5000, 10000, 15000])
    
    # Rank 3: Exploration Fraction (High sensitivity)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.05, 0.20)
    
    # Rank 4: Batch Size (High sensitivity)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    
    # Rank 5: Discount Factor (Medium sensitivity)
    gamma = trial.suggest_categorical("gamma", [0.99, 0.995])
    
    # Rank 6: Final Exploration Epsilon (Medium sensitivity)
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.10)
    
    # Merge into config
    config = copy.deepcopy(base_config)
    config["env"]["reward"]["hindsight_horizon"] = hindsight_horizon
    config["env"]["reward"]["hindsight_weight"] = hindsight_weight
    config["agents"]["bdq"]["auxiliary_weight"] = auxiliary_weight
    config["agents"]["bdq"]["learning_rate"] = learning_rate
    config["agents"]["bdq"]["gamma"] = gamma
    config["agents"]["bdq"]["batch_size"] = batch_size
    config["training"]["target_update_interval"] = target_update_interval
    config["training"]["exploration_fraction"] = exploration_fraction
    config["training"]["exploration_final_eps"] = exploration_final_eps

    # Reduce training steps for HPO
    hpo_steps = args.steps if args.steps else 50000
    config["training"]["total_timesteps"] = hpo_steps
    
    # Enable basic WandB logging for HPO trials
    wandb_project = config.get("wandb", {}).get("project", "DeepScalper-HPO")
    wandb.init(
        project=wandb_project,
        name=f"HPO_trial_{trial.number}",
        tags=["HPO"] + (args.tags or []),
        config={
            "trial_number": trial.number,
            "hindsight_horizon": hindsight_horizon,
            "hindsight_weight": hindsight_weight,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gamma": gamma,
        },
        reinit=True  # Allow multiple inits in same process
    )
    
    logger.info(f"Trial {trial.number}: h={hindsight_horizon}, w={hindsight_weight:.4f}, aux={auxiliary_weight}")
    
    try:
        # === DATA SPLITTING LOGIC (3-SPLIT) ===
        # Use explicit Train/Val splits from config
        train_start = config["data"]["train_start_date"]
        train_end = config["data"]["train_end_date"]
        val_start = config["data"]["val_start_date"]
        val_end = config["data"]["val_end_date"]
        
        logger.info(f"Train: {train_start} to {train_end}")
        logger.info(f"Val:   {val_start} to {val_end}")

        # Create Environments with explicit dates
        train_env_fn = create_env_factory(config, start_date=train_start, end_date=train_end)
        train_env = gym.vector.SyncVectorEnv([train_env_fn])
        
        val_env_fn = create_env_factory(config, start_date=val_start, end_date=val_end)
        # Validate val env is not empty (check dates)
        val_env = gym.vector.SyncVectorEnv([val_env_fn])
        
        # Create trainer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = DeepScalperTrainer(
            env=train_env,
            config=config,
            device=device,
            run_name=f"hpo_trial_{trial.number}"
        )
        
        # === PROPER EARLY STOPPING ===
        # Train in chunks and report intermediate metrics for pruning
        checkpoint_interval = 10000  # Report every 10k steps
        total_steps = 0
        is_first_chunk = True
        
        while total_steps < hpo_steps:
            # Train for one chunk
            remaining_steps = hpo_steps - total_steps
            chunk_steps = min(checkpoint_interval, remaining_steps)
            target_step_count = total_steps + chunk_steps
            
            # CRITICAL FIX: Manually update trainer's target steps
            trainer.total_timesteps = target_step_count
            
            # CRITICAL FIX: Pass start_step and skip_reset to preserve trajectory state
            trainer.train(start_step=total_steps, skip_reset=not is_first_chunk)
            is_first_chunk = False
            
            total_steps = target_step_count
            
            # Evaluate on VALIDATION env (3 episodes for statistical validity)
            intermediate_sharpe = evaluate_agent(
                val_env, trainer.agent, num_episodes=3, max_steps=hpo_steps // 10
            )
            trial.report(intermediate_sharpe, step=total_steps)
            
            logger.info(f"Trial {trial.number} @ {total_steps}: Val Sharpe={intermediate_sharpe:.4f}")
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Final evaluation (3 episodes)
        val_sharpe = evaluate_agent(val_env, trainer.agent, num_episodes=3, max_steps=hpo_steps // 10)
        
        logger.info(f"Trial {trial.number} completed: Val Sharpe={val_sharpe:.4f}")
        wandb.log({"hpo/final_val_sharpe": val_sharpe})
        wandb.finish()
        return val_sharpe
        
    except optuna.TrialPruned:
        wandb.log({"hpo/pruned": True})
        wandb.finish()
        raise
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        wandb.finish()
        raise optuna.TrialFailed(f"Trial crashed: {e}")
    finally:
        # P0 FIX: Always cleanup resources
        if 'train_env' in locals():
            train_env.close()
        if 'val_env' in locals():
            val_env.close()


def main():
    parser = argparse.ArgumentParser(description="DeepScalper HPO with Optuna")
    parser.add_argument("--config", type=str, default="configs/deepscalper_rtx5090.yaml")
    parser.add_argument("--output", type=str, default="configs/best_params.yaml")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--steps", type=int, default=50000, help="Training steps per trial")
    parser.add_argument("--tags", nargs="*", default=["HPO"])
    parser.add_argument("--resume", action="store_true", help="Resume existing study")
    args = parser.parse_args()
    
    print(f"Loading config from {args.config}")
    base_config = load_config(args.config)
    
    # Study storage
    storage = "sqlite:///hpo.db"
    study_name = "deepscalper_hpo"
    
    # Clean old study if not resuming
    if not args.resume and os.path.exists("hpo.db"):
        print("Removing stale HPO database...")
        os.remove("hpo.db")
        if os.path.exists("hpo.db-journal"):
            os.remove("hpo.db-journal")
    
    # Create study with TPE sampler
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=10000)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=args.resume
    )
    
    print(f"\n{'='*50}")
    print(f">>> STARTING HPO: {args.trials} trials, {args.steps} steps each")
    print(f">>> Sampler: TPE | Pruner: Median")
    print(f"{'='*50}\n")
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, base_config, args),
        n_trials=args.trials,
        show_progress_bar=True
    )
    
    # Export best params
    print(f"\n{'='*50}")
    print(">>> HPO COMPLETE")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Sharpe: {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")
    print(f"{'='*50}\n")
    
    best_params = {
        "env": {
            "reward": {
                "hindsight_horizon": study.best_params["hindsight_horizon"],
                "hindsight_weight": study.best_params["hindsight_weight"]
            }
        },
        "agents": {
            "bdq": {
                "auxiliary_weight": study.best_params["auxiliary_weight"],
                "learning_rate": study.best_params["learning_rate"],
                "gamma": study.best_params["gamma"],
                "batch_size": study.best_params["batch_size"]
            }
        },
        "training": {
            "target_update_interval": study.best_params["target_update_interval"],
            "exploration_fraction": study.best_params["exploration_fraction"],
            "exploration_final_eps": study.best_params["exploration_final_eps"]
        }
    }
    
    with open(args.output, "w") as f:
        yaml.dump(best_params, f)
    
    print(f"Best params saved to {args.output}")


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
