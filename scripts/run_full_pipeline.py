#!/usr/bin/env python
"""
DeepScalper Unified Pipeline - Single BDQ Agent
Consolidated: HPO, Training, Backtesting in one process.
"""
import yaml
import argparse
import os
import sys
import copy
import logging
import functools
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import gymnasium as gym
import wandb
import optuna
from optuna.samplers import TPESampler

# Project imports
sys.path.append(os.getcwd())
from finrl_pro_ds.training.deepscalper_trainer import DeepScalperTrainer
from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler
from finrl_pro_ds.agents.deepscalper.bdq_agent import DeepScalperBDQ
from finrl_pro_ds.utils.naming import generate_run_name, validate_run_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeepScalperPipeline")


# ============================================================================
# CONFIG UTILS
# ============================================================================
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(base, overrides):
    """Deep merge dictionaries."""
    for k, v in overrides.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            merge_configs(base[k], v)
        else:
            base[k] = v
    return base


# ============================================================================
# ENVIRONMENT FACTORY
# ============================================================================
def make_env(config, start_date=None, end_date=None, shm_config=None):
    """Factory to create DeepScalperEnv with real data.
    
    Note: shm_config is passed explicitly to avoid stale references.
    """
    data_config = config.get("data", {})
    file_path = data_config.get("file_path")
    ticker = data_config.get("ticker", "BTCUSDT")
    
    if not file_path or not os.path.exists(file_path):
        raise ValueError(f"Invalid data file path: {file_path}")
    
    sd = start_date or data_config.get("train_start_date")
    ed = end_date or data_config.get("train_end_date")
    
    handler = ParquetDataHandler(
        file_path=file_path,
        ticker=ticker,
        feature_config=config.get("features", {}),
        start_date=sd,
        end_date=ed,
        shared_memory_config=shm_config  # Explicit, not from config dict
    )
    
    env_config = config.get("env", {})
    env_config["reward"] = config.get("env", {}).get("reward", {})
    
    return DeepScalperEnv(config=env_config, data_handler=handler)


def create_vector_env(config, num_envs, start_date=None, end_date=None, shm_config=None):
    """Create vectorized environment for training.
    
    Note: Using SyncVectorEnv only. AsyncVectorEnv has race conditions with
    SHM cleanup and memory issues when workers load large datasets individually.
    """
    env_factory = functools.partial(make_env, config=config, start_date=start_date, end_date=end_date, shm_config=shm_config)
    
    # Always use SyncVectorEnv to avoid subprocess issues
    env = gym.vector.SyncVectorEnv([env_factory for _ in range(num_envs)])
    
    return env


# ============================================================================
# PHASE 1: HYPERPARAMETER OPTIMIZATION
# ============================================================================
def evaluate_for_hpo(env, agent, max_steps=5000):
    """Quick evaluation for HPO - returns Sharpe ratio."""
    all_returns = []
    
    try:
        obs, info = env.reset()
        done = False
        step = 0
        
        prev_val = info.get("portfolio_value", 100000.0)
        if hasattr(prev_val, "__len__"):
            prev_val = prev_val[0]
        prev_val = float(prev_val) if prev_val > 0 else 1e-6
        
        while not done and step < max_steps:
            micro = torch.tensor(obs["micro"], dtype=torch.float32).to(agent.device)
            private = torch.tensor(obs["private"], dtype=torch.float32).to(agent.device)
            macro = torch.tensor(obs["macro"], dtype=torch.float32).to(agent.device)
            
            action = agent.predict(micro, private, macro, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            
            t_val = term[0] if hasattr(term, "__len__") else term
            tr_val = trunc[0] if hasattr(trunc, "__len__") else trunc
            done = bool(t_val or tr_val)
            
            curr_val = info.get("portfolio_value", prev_val)
            if hasattr(curr_val, "__len__"):
                curr_val = curr_val[0]
            curr_val = float(curr_val)
            
            step_return = (curr_val - prev_val) / prev_val if prev_val > 0 else 0
            all_returns.append(step_return)
            prev_val = curr_val
            step += 1
            
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return 0.0
    
    returns = np.array(all_returns)
    if len(returns) > 1 and np.std(returns) > 1e-9:
        ann_factor = np.sqrt(525600)  # Minute data, 24/7
        sharpe = (np.mean(returns) / np.std(returns)) * ann_factor
    else:
        sharpe = 0.0
    
    return sharpe


def run_hpo(base_config, n_trials, steps_per_trial, device):
    """Phase 1: Hyperparameter Optimization with Optuna."""
    logger.info(f"Starting HPO: {n_trials} trials, {steps_per_trial} steps each")
    wandb.log({"hpo/status": "started", "hpo/n_trials": n_trials})
    
    def objective(trial):
        trial_prefix = f"hpo/t{trial.number}"
        wandb.log({f"{trial_prefix}/started": True})
        
        # Sample hyperparameters
        hindsight_horizon = trial.suggest_categorical("hindsight_horizon", [60, 120, 180, 240])
        hindsight_weight = trial.suggest_float("hindsight_weight", 0.001, 0.1, log=True)
        auxiliary_weight = trial.suggest_categorical("auxiliary_weight", [0.5, 1.0])
        learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
        target_update_interval = trial.suggest_categorical("target_update_interval", [5000, 10000, 15000])
        exploration_fraction = trial.suggest_float("exploration_fraction", 0.05, 0.20)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        gamma = trial.suggest_categorical("gamma", [0.99, 0.995])
        exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.10)
        
        # Create trial config
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
        config["training"]["total_timesteps"] = steps_per_trial
        
        wandb.log({
            f"{trial_prefix}/hindsight_horizon": hindsight_horizon,
            f"{trial_prefix}/learning_rate": learning_rate,
            f"{trial_prefix}/batch_size": batch_size,
        })
        
        # Create env and train
        env = None
        try:
            env = create_vector_env(config, num_envs=config["env"].get("num_envs", 12))
            trainer = DeepScalperTrainer(env, config, device=device, hpo_mode=True)
            trainer.train()
            
            # Evaluate
            sharpe = evaluate_for_hpo(env, trainer.agent, max_steps=5000)
            
            wandb.log({f"{trial_prefix}/sharpe": sharpe, f"{trial_prefix}/completed": True})
            logger.info(f"Trial {trial.number}: Sharpe={sharpe:.4f}")
            
            return sharpe
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            wandb.log({f"{trial_prefix}/error": str(e)})
            return 0.0
        finally:
            if env:
                env.close()
            # FIX A: Force garbage collection to release PyArrow mmap/FD handles
            import gc
            gc.collect()
    

    # Run optimization
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials)
    
    # Log best results
    best = study.best_trial
    best_params = {
        "env": {"reward": {}},
        "agents": {"bdq": {}},
        "training": {}
    }
    
    for key, val in best.params.items():
        if key in ["hindsight_horizon", "hindsight_weight"]:
            best_params["env"]["reward"][key] = val
        elif key in ["auxiliary_weight", "learning_rate", "gamma", "batch_size"]:
            best_params["agents"]["bdq"][key] = val
        else:
            best_params["training"][key] = val
    
    wandb.log({
        "hpo/best_sharpe": best.value,
        "hpo/best_trial": best.number,
        "hpo/status": "completed"
    })
    
    logger.info(f"HPO Complete. Best Sharpe: {best.value:.4f}")
    return best_params


# ============================================================================
# PHASE 2: TRAINING
# ============================================================================
def run_training(config, run_name, device):
    """Phase 2: Full training with optimized hyperparameters."""
    logger.info("Starting Training Phase")
    wandb.log({"train/status": "started"})
    
    num_envs = config.get("env", {}).get("num_envs", 12)
    data_loader = None
    env = None
    
    try:
        # Setup shared memory if enabled
        shm_config = None
        use_shm = config.get("training", {}).get("use_shm", False)
        if num_envs > 1 and use_shm:
            logger.info("Initializing Shared Memory...")
            data_config = config.get("data", {})
            data_loader = ParquetDataHandler(
                file_path=data_config.get("file_path"),
                ticker=data_config.get("ticker", "BTCUSDT"),
                feature_config=config.get("features", {}),
                start_date=data_config.get("train_start_date"),
                end_date=data_config.get("train_end_date")
            )
            shm_config = data_loader.create_shared_memory()
            # Note: Don't store in config dict - pass explicitly to avoid stale refs
        
        # Create environment (pass shm_config explicitly)
        env = create_vector_env(config, num_envs, shm_config=shm_config)
        logger.info(f"Environment ready: {num_envs} workers")
        
        # Train
        trainer = DeepScalperTrainer(env, config, device=device, run_name=run_name)
        trainer.train()
        
        # Find checkpoint
        checkpoints_dir = f"checkpoints/{run_name}"
        checkpoint_path = None
        if os.path.exists(checkpoints_dir):
            ckpts = sorted([f for f in os.listdir(checkpoints_dir) if f.endswith(".pth") or f.endswith(".pt")])
            if ckpts:
                checkpoint_path = os.path.join(checkpoints_dir, ckpts[-1])
        
        wandb.log({"train/status": "completed", "train/checkpoint": checkpoint_path or "none"})
        logger.info(f"Training complete. Checkpoint: {checkpoint_path}")
        
        return checkpoint_path
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        wandb.log({"train/status": "failed", "train/error": str(e)})
        raise
    finally:
        if env:
            env.close()
        if data_loader:
            try:
                data_loader.close_shared_memory(unlink=True)
            except:
                pass


# ============================================================================
# PHASE 3: BACKTESTING
# ============================================================================
def run_backtest(config, checkpoint_path, device, start_date=None, end_date=None, prefix="backtest"):
    """Phase 3: Backtest on specified data range."""
    mode = "Test" if prefix == "backtest" else "Validation"
    logger.info(f"Starting {mode} Phase")
    wandb.log({f"{prefix}/status": "started"})
    
    data_config = config.get("data", {})
    if not start_date:
        start_date = data_config.get("test_start_date")
    if not end_date:
        end_date = data_config.get("test_end_date")
        
    if not start_date or not end_date:
        logger.warning(f"No dates configured for {prefix}, using validation dates as fallback")
        start_date = data_config.get("val_start_date")
        end_date = data_config.get("val_end_date")
    
    env = None
    try:
        env = make_env(config, start_date=start_date, end_date=end_date)
        
        # Create agent
        sample_obs, _ = env.reset()
        # Create agent for backtest
        bdq_config = config.get("agents", {}).get("bdq", {})
        # Fix: Network config is at top level, not inside agents.bdq
        network_config = config.get("network", {
            "micro_config": {
                "input_size": 20,
                "private_input_size": 2,
                "hidden_size": 128,
                "rnn_type": "LSTM"
            },
            "macro_config": {
                "input_size": 11,
                "hidden_sizes": [128, 128]
            }
        })
        agent = DeepScalperBDQ(
            network_config=network_config,
            lr=bdq_config.get("learning_rate", 1e-4),
            gamma=bdq_config.get("gamma", 0.99),
            batch_size=bdq_config.get("batch_size", 64),
            device=device
        )
        
        # Load checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            agent.load(checkpoint_path)
        else:
            logger.warning("No checkpoint found, using random policy")
        
        # Run backtest
        obs, info = env.reset()
        portfolio_values = []
        positions = []
        done = False
        step = 0
        
        while not done and step < 200000:
            micro = torch.tensor(obs["micro"], dtype=torch.float32).unsqueeze(0).to(device)
            private = torch.tensor(obs["private"], dtype=torch.float32).unsqueeze(0).to(device)
            macro = torch.tensor(obs["macro"], dtype=torch.float32).unsqueeze(0).to(device)
            
            action = agent.predict(micro, private, macro, deterministic=True)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            portfolio_values.append(info.get("portfolio_value", 100000))
            positions.append(info.get("position", 0))
            
            if step % 5000 == 0:
                logger.info(f"Backtest step {step}: Value={portfolio_values[-1]:.2f}")
            step += 1
        
        # Compute metrics
        pv = np.array(portfolio_values)
        pos_arr = np.array(positions)
        returns = np.diff(pv) / pv[:-1]
        
        total_return = (pv[-1] - pv[0]) / pv[0] if len(pv) > 0 else 0
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(525600) if np.std(returns) > 1e-9 else 0
        max_dd = np.min(pv / np.maximum.accumulate(pv)) - 1 if len(pv) > 0 else 0
        
        # Trade Stats
        trade_count = np.sum(np.abs(np.diff(pos_arr)) > 1e-6)
        market_exposure = np.mean(np.abs(pos_arr) > 1e-6)
        
        metrics = {
            f"{prefix}/total_return": total_return,
            f"{prefix}/sharpe": sharpe,
            f"{prefix}/max_drawdown": max_dd,
            f"{prefix}/final_value": pv[-1] if len(pv) > 0 else 0,
            f"{prefix}/steps": step,
            f"{prefix}/trade_count": int(trade_count),
            f"{prefix}/market_exposure": market_exposure,
            f"{prefix}/status": "completed"
        }
        
        wandb.log(metrics)
        logger.info(f"{mode} Complete. Return={total_return*100:.2f}%, Sharpe={sharpe:.2f}, MaxDD={max_dd*100:.2f}%")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        wandb.log({f"{prefix}/status": "failed", f"{prefix}/error": str(e)})
        raise
    finally:
        if env:
            env.close()


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="DeepScalper Single BDQ Pipeline")
    parser.add_argument("--config", type=str, default="configs/deepscalper_rtx5090.yaml")
    parser.add_argument("--tags", nargs="*", default=["Pipeline"], help="WandB Tags")
    parser.add_argument("--run_name", type=str, default=None, help="Override WandB Run Name")
    parser.add_argument("--trials", type=int, default=None, help="Number of HPO trials")
    parser.add_argument("--steps", type=int, default=None, help="Training steps override")
    args = parser.parse_args()

    base_config = load_config(args.config)
    
    # =========================================================================
    # 0. Setup Run Name & WandB
    # =========================================================================
    if args.run_name:
        validate_run_name(args.run_name, raise_on_fail=True)
        run_name = args.run_name
    else:
        platform = "GPUHub" if os.path.exists("/workspace") else "Local"
        run_name = generate_run_name(version="V1", platform=platform)
    
    logger.info(f"Pipeline Run: {run_name}")
    
    # Initialize single WandB run for entire pipeline
    wandb_config = base_config.get("wandb", {})
    wandb.init(
        project=wandb_config.get("project", "FinRL-Pro-DS"),
        entity=wandb_config.get("entity", "bigcan-chiwin-technology"),
        name=run_name,
        tags=args.tags,
        config=base_config
    )
    logger.info(f"WandB Run: {wandb.run.url}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    try:
        # =====================================================================
        # PHASE 1: HPO
        # =====================================================================
        print("\n" + "="*60)
        print(">>> PHASE 1: HYPERPARAMETER OPTIMIZATION")
        print("="*60 + "\n")
        
        hpo_config = base_config.get("hpo", {})
        final_config = copy.deepcopy(base_config)
        
        if hpo_config.get("enabled", True):
            n_trials = args.trials or hpo_config.get("n_trials", 20)
            steps_per_trial = hpo_config.get("steps_per_trial", 50000)
            
            best_params = run_hpo(base_config, n_trials, steps_per_trial, device)
            
            # Merge best params into config
            final_config = merge_configs(final_config, best_params)
        else:
            logger.info("HPO disabled in config. Using base configuration.")
        
        # Override training steps if specified
        if args.steps:
            final_config["training"]["total_timesteps"] = args.steps
        
        # =====================================================================
        # PHASE 2: TRAINING
        # =====================================================================
        print("\n" + "="*60)
        print(">>> PHASE 2: TRAINING (Full Run with Best Params)")
        print("="*60 + "\n")
        
        checkpoint_path = run_training(final_config, run_name, device)
        
        if checkpoint_path:
            # PHASE 3a: Validation Backtest (for Overfitting Check)
            print(">>> PHASE 3a: VALIDATION BACKTEST")
            data_config = final_config.get("data", {})
            run_backtest(
                final_config, 
                checkpoint_path, 
                device, 
                start_date=data_config.get("val_start_date"), 
                end_date=data_config.get("val_end_date"),
                prefix="backtest_val"
            )

            # PHASE 3b: Test Backtest (for Final Evaluation)
            print(">>> PHASE 3b: TEST BACKTEST")
            run_backtest(
                final_config, 
                checkpoint_path, 
                device,
                prefix="backtest_test"
            )
        else:
            logger.warning("No checkpoint found, skipping backtests")
        
        print("\n" + "="*60)
        print(">>> PIPELINE COMPLETE")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        wandb.log({"pipeline/status": "failed", "pipeline/error": str(e)})
        raise
    finally:
        wandb.finish()
        logger.info("WandB run finalized.")


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
