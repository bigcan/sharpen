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
from finrl_pro_ds.training.ppo_trainer import PPOTrainer
from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler
from finrl_pro_ds.agents.deepscalper.bdq_agent import DeepScalperBDQ
from finrl_pro_ds.agents.ppo_scalper.ppo_agent import PPOAgent
from finrl_pro_ds.utils.naming import generate_run_name, validate_run_name
from finrl_pro_ds.analytics.pyfolio_analyzer import PyfolioAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeepScalperPipeline")


# ============================================================================
# CONFIG UTILS
# ============================================================================
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
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
def make_env(config, start_date=None, end_date=None, shm_config=None, norm_cutoff_date=None):
    """Factory to create DeepScalperEnv with real data.
    
    Args:
        norm_cutoff_date: FIX LEAK-1 — When set, rolling normalization statistics
            (z-scores, SMAs) are reset at this date boundary so that training
            data does not leak into val/test feature statistics.
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
        shared_memory_config=shm_config,
        norm_cutoff_date=norm_cutoff_date  # FIX LEAK-1
    )
    
    env_config = config.get("env", {})
    env_config["reward"] = config.get("env", {}).get("reward", {})
    
    return DeepScalperEnv(config=env_config, data_handler=handler)


def create_vector_env(config, num_envs, start_date=None, end_date=None, shm_config=None, gym_shm=True, use_sync=False):
    """Create vectorized environment for training.
    
    Note: Using AsyncVectorEnv for parallel data loading. Context 'spawn' is used
    for CUDA/PyTorch safety. Set use_sync=True to use SyncVectorEnv (no subprocesses),
    which avoids IPC/FD limits on constrained containers.
    """
    env_factory = functools.partial(make_env, config=config, start_date=start_date, end_date=end_date, shm_config=shm_config)
    
    if use_sync:
        # SyncVectorEnv: all envs run in main process. No pipes, no FD issues.
        # Slower but reliable on containers with restricted ulimits.
        env = gym.vector.SyncVectorEnv(
            [env_factory for _ in range(num_envs)]
        )
    else:
        # AsyncVectorEnv for production training (parallel data loading)
        env = gym.vector.AsyncVectorEnv(
            [env_factory for _ in range(num_envs)],
            context="spawn",
            shared_memory=False
        )
    
    return env


# ============================================================================
# PHASE 1: HYPERPARAMETER OPTIMIZATION
# ============================================================================
def evaluate_for_hpo(env, agent, max_steps=5000):
    """Quick evaluation for HPO - returns Sharpe ratio.
    
    IMPORTANT: This function handles both single envs and VectorEnvs.
    VectorEnv returns info as a dict of arrays, or for newer Gymnasium versions,
    as a tuple (info_dict, final_info_dict).
    """
    all_returns = []
    action_counts = {0: 0, 1: 0, 2: 0}  # Track action distribution
    
    def extract_portfolio_value(info, env_idx=0, default=100000.0):
        """Extract portfolio_value handling both single env and VectorEnv info structures."""
        if info is None:
            return default
        
        # Handle VectorEnv info structure (Gymnasium >= 0.26)
        # VectorEnv returns: info = {'final_info': [...], 'final_observation': [...], ...}
        # with per-step metrics potentially in different places
        
        # Try direct access first (single env case)
        pv = info.get("portfolio_value")
        
        if pv is not None:
            # If it's an array (VectorEnv), get the env_idx element
            if hasattr(pv, "__len__") and not isinstance(pv, str):
                return float(pv[env_idx])
            return float(pv)
        
        # For VectorEnv, the step info might be nested
        # Try '_all_info' key which some versions use
        if "_all_info" in info:
            per_env_info = info["_all_info"]
            if per_env_info and len(per_env_info) > env_idx:
                env_info = per_env_info[env_idx]
                if env_info and "portfolio_value" in env_info:
                    return float(env_info["portfolio_value"])
        
        # For final_info (used on episode termination)
        if "final_info" in info:
            final = info["final_info"]
            if final and len(final) > env_idx and final[env_idx]:
                if "portfolio_value" in final[env_idx]:
                    return float(final[env_idx]["portfolio_value"])
        
        return default
    
    info_logged = False  # Only log once
    
    # BUG-B: Save and restore LSTM hidden state around eval.
    # Training uses num_envs=12 → hidden=(1,12,256), eval uses num_envs=1.
    # Must restore after eval to avoid corrupting training state.
    _saved_hidden = agent._hidden_state if hasattr(agent, '_hidden_state') else None
    if hasattr(agent, "reset_hidden_state"):
        agent.reset_hidden_state()
    
    try:
        obs, info = env.reset()
        done = False
        step = 0
        
        prev_val = extract_portfolio_value(info, env_idx=0, default=100000.0)
        
        while not done and step < max_steps:
            micro = torch.tensor(obs["micro"], dtype=torch.float32).to(agent.device)
            private = torch.tensor(obs["private"], dtype=torch.float32).to(agent.device)
            macro = torch.tensor(obs["macro"], dtype=torch.float32).to(agent.device)
            
            action = agent.predict(micro, private, macro, deterministic=True)
            
            # Track action distribution (first env if vectorized)
            first_action = action[0] if len(action.shape) > 1 else action
            direction = int(first_action[0]) if hasattr(first_action, "__len__") else int(first_action)
            action_counts[direction] = action_counts.get(direction, 0) + 1
            
            obs, reward, term, trunc, info = env.step(action)
            
            # Log info structure on first step to diagnose VectorEnv format
            if not info_logged:
                info_keys = list(info.keys()) if isinstance(info, dict) else str(type(info))
                sample_values = {}
                if isinstance(info, dict):
                    for k, v in list(info.items())[:5]:  # First 5 keys
                        if hasattr(v, "shape"):
                            sample_values[k] = f"array{v.shape}"
                        elif hasattr(v, "__len__") and len(v) > 0:
                            sample_values[k] = f"list[{len(v)}]:{type(v[0]).__name__}"
                        else:
                            sample_values[k] = str(type(v).__name__)
                wandb.log({"_debug/info_keys": str(info_keys), "_debug/info_sample": str(sample_values)})
                info_logged = True
            
            t_val = term[0] if hasattr(term, "__len__") else term
            tr_val = trunc[0] if hasattr(trunc, "__len__") else trunc
            done = bool(t_val or tr_val)
            
            curr_val = extract_portfolio_value(info, env_idx=0, default=prev_val)
            
            step_return = (curr_val - prev_val) / prev_val if prev_val > 0 else 0
            all_returns.append(step_return)
            prev_val = curr_val
            step += 1
            
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        import traceback
        wandb.log({"_debug/eval_error": str(e), "_debug/eval_traceback": traceback.format_exc()})
        # BUG-B: Restore training hidden state even on error
        agent._hidden_state = _saved_hidden
        return 0.0
    
    returns = np.array(all_returns)
    
    # Diagnostic: log what we computed including action distribution
    diag = {
        "_debug/eval_steps": step,
        "_debug/eval_returns_len": len(returns),
        "_debug/eval_returns_std": float(np.std(returns)) if len(returns) > 0 else 0.0,
        "_debug/eval_returns_mean": float(np.mean(returns)) if len(returns) > 0 else 0.0,
        "_debug/eval_final_pv": prev_val,
        "_debug/eval_action_hold": action_counts.get(0, 0),
        "_debug/eval_action_buy": action_counts.get(1, 0),
        "_debug/eval_action_sell": action_counts.get(2, 0),
    }
    wandb.log(diag)
    
    if len(returns) > 1 and np.std(returns) > 1e-9:
        raw_ratio = np.mean(returns) / np.std(returns)
        # Per-minute Sharpe (for reporting only)
        sharpe_minute = raw_ratio * np.sqrt(525600)  # 365.25 × 24 × 60
        # Hourly-aggregated Sharpe (research comparison)
        n_per_hour = 60
        hourly_returns = np.add.reduceat(returns, np.arange(0, len(returns), n_per_hour))
        if len(hourly_returns) > 1 and np.std(hourly_returns) > 1e-9:
            sharpe_hourly = (np.mean(hourly_returns) / np.std(hourly_returns)) * np.sqrt(365 * 24)
        else:
            sharpe_hourly = 0.0
        wandb.log({
            "_research/sharpe_minute": sharpe_minute,
            "_research/sharpe_hourly": sharpe_hourly,
            "_research/raw_ratio_per_step": raw_ratio,
        })
    else:
        raw_ratio = 0.0
        sharpe_minute = 0.0
        logger.warning(f"Zero Sharpe: steps={step}, len={len(returns)}, std={np.std(returns) if len(returns) > 0 else 'N/A'}, actions={action_counts}")
    
    # BUG-B: Restore training hidden state after eval
    agent._hidden_state = _saved_hidden
    
    # FIX: Return RAW (non-annualized) ratio for HPO optimization.
    # sqrt(525600) ≈ 725x amplification makes all trials look equally catastrophic,
    # preventing Optuna's TPE sampler from differentiating between trial quality.
    # Annualization is applied only during final backtesting (Phase 3).
    return raw_ratio


def run_hpo(base_config, n_trials, steps_per_trial, device):
    """Phase 1: Hyperparameter Optimization with Optuna."""
    logger.info(f"Starting HPO: {n_trials} trials, {steps_per_trial} steps each")
    wandb.log({"hpo/status": "started", "hpo/n_trials": n_trials})
    
    def objective(trial):
        trial_prefix = f"hpo/t{trial.number}"
        wandb.log({f"{trial_prefix}/started": True})
        
        # Sample hyperparameters (8 dimensions)
        # Paper-aligned reward params (Section 3.2 + 4.2)
        hindsight_horizon = trial.suggest_categorical("hindsight_horizon", [30, 60, 90, 120, 150, 180])
        hindsight_weight = trial.suggest_float("hindsight_weight", 0.05, 0.2, log=True)
        # DSR REMOVED: Sprint 5 showed sharpe_weight=0.815 → agent optimized for
        # variance minimization instead of profit. Pure paper reward only.
        # Agent params
        auxiliary_weight = trial.suggest_float("auxiliary_weight", 0.5, 1.5, log=True)
        learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
        gamma = trial.suggest_categorical("gamma", [0.99, 0.995])
        epsilon_end = trial.suggest_float("epsilon_end", 0.01, 0.10)
        tau = trial.suggest_float("tau", 0.001, 0.01, log=True)
        # NOTE: target_update_freq removed — Polyak (tau) runs every step, freq is dead code.
        
        # Create trial config
        config = copy.deepcopy(base_config)
        config["env"]["reward"]["hindsight_horizon"] = hindsight_horizon
        config["env"]["reward"]["hindsight_weight"] = hindsight_weight
        config["env"]["reward"]["sharpe_weight"] = 0.0  # Pure paper reward
        config["agents"]["bdq"]["auxiliary_weight"] = auxiliary_weight
        config["agents"]["bdq"]["learning_rate"] = learning_rate
        config["agents"]["bdq"]["gamma"] = gamma
        config["agents"]["bdq"]["batch_size"] = batch_size
        config["agents"]["bdq"]["epsilon_end"] = epsilon_end
        config["agents"]["bdq"]["tau"] = tau
        config["training"]["total_timesteps"] = steps_per_trial
        
        wandb.log({
            f"{trial_prefix}/hindsight_horizon": hindsight_horizon,
            f"{trial_prefix}/hindsight_weight": hindsight_weight,
            f"{trial_prefix}/learning_rate": learning_rate,
            f"{trial_prefix}/batch_size": batch_size,
            f"{trial_prefix}/auxiliary_weight": auxiliary_weight,
            f"{trial_prefix}/epsilon_end": epsilon_end,
            f"{trial_prefix}/tau": tau,
        })
        
        # Create env and train
        env = None
        eval_env = None
        try:
            # FIX: Use SyncVectorEnv for HPO to avoid AsyncVectorEnv pipe crashes
            # on containers where ulimit -n is blocked. SyncVectorEnv runs all
            # envs in the main process — slower but no IPC/FD issues.
            hpo_num_envs = min(config["env"].get("num_envs", 12), 12)  # Cap at 12 for HPO throughput
            env = create_vector_env(config, num_envs=hpo_num_envs, gym_shm=False, use_sync=True)
            trainer = DeepScalperTrainer(env, config, device=device, hpo_mode=True)
            
            # Create DEDICATED eval env for pruning (P1b fix: prevents training state corruption)
            eval_env = create_vector_env(config, num_envs=1, gym_shm=False, use_sync=True)
            
            # Define Pruning Callback with dedicated eval env
            def pruning_callback():
                return evaluate_for_hpo(eval_env, trainer.agent, max_steps=3000)

            trainer.train(optuna_trial=trial, pruning_callback=pruning_callback)
            
            # FIX: Evaluate on dedicated eval_env (clean state), not training env
            sharpe = evaluate_for_hpo(eval_env, trainer.agent, max_steps=15000)
            
            wandb.log({f"{trial_prefix}/sharpe": sharpe, f"{trial_prefix}/completed": True})
            logger.info(f"Trial {trial.number}: Sharpe(raw)={sharpe:.6f}")
            
            return sharpe
            
        except optuna.TrialPruned:
            logger.info(f"Trial {trial.number} pruned.")
            wandb.log({f"{trial_prefix}/status": "pruned"})
            raise  # Re-raise original exception to preserve traceback
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            wandb.log({f"{trial_prefix}/error": str(e)})
            return 0.0
        finally:
            # FIX: AsyncVectorEnv pipes may already be dead → BrokenPipeError
            # during close(). This is harmless (cleanup of dead workers) but
            # kills the entire HPO study if uncaught.
            if env:
                try:
                    env.close()
                except (BrokenPipeError, EOFError, ConnectionResetError):
                    pass
            # Close dedicated eval env
            if eval_env:
                try:
                    eval_env.close()
                except (BrokenPipeError, EOFError, ConnectionResetError):
                    pass
            # FIX A: Force garbage collection to release PyArrow mmap/FD handles
            import gc
            gc.collect()
    

    # Run optimization
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        # FIX AUDIT-5: Increased min_resource from 5K to 15K.
        # At 5K steps, learning_starts=5K means the agent has done ~0 gradient updates.
        # The LSTM micro-encoder needs ~10-20K steps to warm up hidden states.
        # Pruning at 5K discards trials before they can demonstrate learning.
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=15000, 
            max_resource=steps_per_trial, 
            reduction_factor=3
        )
    )
    study.optimize(objective, n_trials=n_trials)
    
    # Log best results
    best = study.best_trial
    best_params = {
        "env": {"reward": {}},
        "agents": {"bdq": {}},
        "training": {}
    }
    
    # Explicit routing for ALL HPO params to prevent silent mis-routing.
    reward_params = {"hindsight_horizon", "hindsight_weight"}
    agent_params = {"auxiliary_weight", "learning_rate", "gamma", "batch_size", "epsilon_end", "tau"}
    
    # Key name mapping: Optuna param name -> config key name
    reward_key_map = {
        "hindsight_horizon": "hindsight_horizon",
        "hindsight_weight": "hindsight_weight",
    }
    
    for key, val in best.params.items():
        if key in reward_params:
            config_key = reward_key_map.get(key, key)
            best_params["env"]["reward"][config_key] = val
        elif key in agent_params:
            best_params["agents"]["bdq"][key] = val
        else:
            raise ValueError(f"HPO param '{key}' has no routing rule. Add to reward_params or agent_params in run_hpo().")
    
    # best.value is raw (non-annualized) ratio; also report annualized for human reference
    annualized_sharpe = best.value * np.sqrt(525600) if best.value is not None else 0.0
    wandb.log({
        "hpo/best_sharpe_raw": best.value,
        "hpo/best_sharpe": annualized_sharpe,  # Human-readable annualized
        "hpo/best_trial": best.number,
        "hpo/best_params": str(best.params),
        "hpo/status": "completed"
    })
    
    logger.info(f"HPO Complete. Best Raw Ratio: {best.value:.6f} (Annualized: {annualized_sharpe:.2f})")
    return best_params


# ============================================================================
# PHASE 2: TRAINING
# ============================================================================
def run_training(config, run_name, device, agent_type="bdq"):
    """Phase 2: Full training with optimized hyperparameters."""
    logger.info(f"Starting Training Phase (agent={agent_type})")
    wandb.log({"train/status": "started", "train/agent_type": agent_type})
    
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
        # Also pass use_shm to Gymnasium to disable internal SHM if needed
        env = create_vector_env(config, num_envs, shm_config=shm_config, gym_shm=use_shm)
        logger.info(f"Environment ready: {num_envs} workers")
        
        # Train — dispatch based on agent type
        if agent_type == "earnhft":
            # EarnHFT uses its own 3-stage pipeline (Q-Teacher → Pool → Router)
            # instead of the standard vector-env training loop
            from finrl_pro_ds.training.earnhft_trainer import EarnHFTTrainer
            data_config = config.get("data", {})
            import pandas as pd
            train_df = pd.read_parquet(
                data_config.get("file_path"),
            )
            # Filter to training date range
            if "timestamp" in train_df.columns:
                train_df = train_df[
                    (train_df["timestamp"] >= data_config.get("train_start_date", "")) &
                    (train_df["timestamp"] <= data_config.get("train_end_date", ""))
                ]
            earnhft_trainer = EarnHFTTrainer(
                config=config,
                data_df=train_df,
                device=device,
                run_name=run_name,
                checkpoint_dir=f"checkpoints/{run_name}",
            )
            earnhft_trainer.train()
        elif agent_type == "ppo":
            trainer = PPOTrainer(env, config, device=device, run_name=run_name)
            trainer.train()
        else:
            trainer = DeepScalperTrainer(env, config, device=device, run_name=run_name)
            trainer.train()
        
        # Find checkpoint
        checkpoints_dir = f"checkpoints/{run_name}"
        checkpoint_path = None
        if os.path.exists(checkpoints_dir):
            if agent_type == "earnhft":
                # EarnHFT saves a manifest describing all sub-checkpoints
                manifest = os.path.join(checkpoints_dir, "pool_manifest.json")
                if os.path.exists(manifest):
                    checkpoint_path = manifest
            else:
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
            try:
                env.close()
            except (BrokenPipeError, EOFError, ConnectionResetError):
                pass
        if data_loader:
            try:
                data_loader.close_shared_memory(unlink=True)
            except:
                pass


# ============================================================================
# PHASE 3: BACKTESTING
# ============================================================================
def run_backtest(config, checkpoint_path, device, start_date=None, end_date=None, prefix="backtest", agent_type="bdq", norm_cutoff_date=None):
    """Phase 3: Backtest on specified data range.
    
    Args:
        norm_cutoff_date: FIX LEAK-1 — Reset normalization statistics at this
            date boundary (e.g. train_end_date for val, val_end_date for test).
    """
    mode = "Test" if prefix == "backtest" else "Validation"
    logger.info(f"Starting {mode} Phase (agent={agent_type})")
    if norm_cutoff_date:
        logger.info(f"[LEAK-1] Normalization cutoff: {norm_cutoff_date}")
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
        env = make_env(config, start_date=start_date, end_date=end_date, norm_cutoff_date=norm_cutoff_date)
        
        # Create agent
        sample_obs, _ = env.reset()
        network_config = config.get("network")
        if not network_config:
            raise ValueError("Config missing 'network' section — cannot reconstruct agent for backtest")
        
        # Read action dims from config (mirrors trainer L24-29)
        action_config = config.get("env", {}).get("action", {})
        signed_qty_props = action_config.get(
            "signed_qty_proportions", [-0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5]
        )
        action_dims = (
            action_config.get("price_bins", 5),
            len(signed_qty_props)
        )
        
        # Dispatch agent creation based on type
        if agent_type == "earnhft":
            # EarnHFT: load from manifest with multi-agent checkpoint
            from finrl_pro_ds.agents.earnhft.earnhft_agent import EarnHFTAgent
            import json as _json
            import re
            
            if not checkpoint_path or not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"EarnHFT manifest not found: {checkpoint_path}")
            
            with open(checkpoint_path, "r") as f:
                manifest = _json.load(f)
            
            # Extract pool checkpoint paths from manifest, sorted by beta value (float)
            pool_info = manifest.get("pool", {})
            def parse_beta_key(k):
                # Extract float from "beta_0.0", "beta_1.0" etc
                m = re.search(r"beta_([\d\.]+)", k)
                return float(m.group(1)) if m else -1.0
                
            sorted_keys = sorted(
                [k for k in pool_info.keys() if "checkpoint" in pool_info[k]],
                key=parse_beta_key
            )
            pool_paths = [pool_info[k]["checkpoint"] for k in sorted_keys]
            
            router_path = manifest.get("router", {}).get("checkpoint")
            router_obs_dim = manifest.get("router", {}).get("obs_dim", 7)
            
            # Reconstruct network_config structure required by DiscretePPOAgent
            # (EarnHFTTrainer builds this manually from flat config)
            net_cfg = config.get("network", {})
            earnhft_cfg = config.get("agents", {}).get("earnhft", {})
            earnhft_network_config = {
                "micro_config": {
                    "input_size": net_cfg.get("micro_input_size", 20),
                    "private_input_size": net_cfg.get("private_input_size", 3),
                    "hidden_size": net_cfg.get("micro_hidden_size", 128),
                    "rnn_type": net_cfg.get("rnn_type", "LSTM"),
                },
                "macro_config": {
                    "input_size": net_cfg.get("macro_input_size", 15),
                    "hidden_sizes": net_cfg.get("macro_hidden_sizes", [128, 64]),
                },
                "fusion_dim": net_cfg.get("fusion_dim", 128),
                "num_actions": earnhft_cfg.get("num_actions", 5),
            }
            
            agent = EarnHFTAgent.from_checkpoints(
                pool_checkpoint_paths=pool_paths,
                router_checkpoint_path=router_path,
                network_config=earnhft_network_config,
                router_obs_dim=router_obs_dim,
                num_actions=earnhft_cfg.get("num_actions", 5),
                max_holding=earnhft_cfg.get("max_holding", 1.0),
                minute_interval=earnhft_cfg.get("minute_interval", 60),
                device=device,
            )
            logger.info(f"Loaded EarnHFT agent from manifest: {len(pool_paths)} pool agents + router")
        elif agent_type == "ppo":
            ppo_cfg = config.get("agents", {}).get("ppo", {})
            agent = PPOAgent(
                network_config=network_config,
                lr=ppo_cfg.get("learning_rate", 3e-4),
                gamma=ppo_cfg.get("gamma", 0.99),
                action_dims=action_dims,
                use_amp=config.get("training", {}).get("use_amp", False),
                device=device,
            )
        else:
            # BDQ agent (existing behavior)
            bdq_config = config.get("agents", {}).get("bdq", {})
            agent = DeepScalperBDQ(
                network_config=network_config,
                lr=bdq_config.get("learning_rate", 1e-4),
                gamma=bdq_config.get("gamma", 0.99),
                epsilon_start=bdq_config.get("epsilon_start", 1.0),
                epsilon_end=bdq_config.get("epsilon_end", 0.01),
                buffer_size=bdq_config.get("buffer_size", 100000),
                batch_size=bdq_config.get("batch_size", 64),
                target_update_freq=bdq_config.get("target_update_freq", 100),
                auxiliary_weight=bdq_config.get("auxiliary_weight", 1.0),
                epsilon_decay=bdq_config.get("epsilon_decay", 0.99999),
                action_dims=action_dims,
                use_amp=config.get("training", {}).get("use_amp", False),
                device=device
            )
        
        # Load checkpoint (EarnHFT already loaded above via from_checkpoints)
        if agent_type != "earnhft":
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
        minute_features = None  # For EarnHFT router decisions
        
        while not done and step < 200000:
            micro = torch.tensor(obs["micro"], dtype=torch.float32).unsqueeze(0).to(device)
            private = torch.tensor(obs["private"], dtype=torch.float32).unsqueeze(0).to(device)
            
            if agent_type == "earnhft":
                # EarnHFT predict returns a dict; router uses macro as minute features
                if step % 60 == 0 and "macro" in obs:
                    minute_features = obs["macro"]
                pred = agent.predict(
                    micro, private,
                    minute_features=minute_features,
                    deterministic=True,
                )
                action = pred["action"]
            else:
                macro = torch.tensor(obs["macro"], dtype=torch.float32).unsqueeze(0).to(device)
                pred = agent.predict(micro, private, macro, deterministic=True)
                if isinstance(pred, tuple):
                    action = pred[0][0]  # PPO: (actions, log_probs, values)
                else:
                    action = pred[0]    # BDQ: actions array
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            portfolio_values.append(info.get("portfolio_value", 100000))
            positions.append(info.get("position", 0))
            
            if step % 50000 == 0:
                logger.info(f"Backtest step {step}: Value={portfolio_values[-1]:.2f}")
            step += 1
        
        # Compute metrics
        pv = np.array(portfolio_values)
        pos_arr = np.array(positions)
        returns = np.diff(pv) / pv[:-1]
        
        total_return = (pv[-1] - pv[0]) / pv[0] if len(pv) > 0 else 0
        
        # Dual Sharpe: per-minute (canonical) + hourly-aggregated (research)
        sharpe = 0.0
        sharpe_hourly = 0.0
        if np.std(returns) > 1e-9:
            raw_ratio = np.mean(returns) / np.std(returns)
            sharpe = raw_ratio * np.sqrt(525600)  # Per-minute (canonical)
            # Hourly aggregation
            n_per_hour = 60
            hourly_returns = np.add.reduceat(returns, np.arange(0, len(returns), n_per_hour))
            if len(hourly_returns) > 1 and np.std(hourly_returns) > 1e-9:
                sharpe_hourly = (np.mean(hourly_returns) / np.std(hourly_returns)) * np.sqrt(365 * 24)
        
        max_dd = np.min(pv / np.maximum.accumulate(pv)) - 1 if len(pv) > 0 else 0
        
        # Trade Stats
        trade_count = np.sum(np.abs(np.diff(pos_arr)) > 1e-6)
        market_exposure = np.mean(np.abs(pos_arr) > 1e-6)
        
        # ── Institutional Metrics via PyfolioAnalyzer (Blueprint mandate) ──
        returns_series = pd.Series(returns)
        try:
            analyzer = PyfolioAnalyzer(returns_series)
            pyfolio_metrics = analyzer.get_audit_metrics()
        except Exception as e:
            logger.warning(f"PyfolioAnalyzer failed, using fallback: {e}")
            pyfolio_metrics = {}
        
        # Manual: Profit Factor & Avg Win/Loss Ratio
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
        gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 1e-12 else 0.0
        avg_win = np.mean(wins) if len(wins) > 0 else 0.0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0.0
        avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 1e-12 else 0.0
        
        metrics = {
            # ── Core metrics (manual, crypto-specific annualization) ──
            f"{prefix}/total_return": total_return,
            f"{prefix}/sharpe": sharpe,
            f"{prefix}/sharpe_hourly": sharpe_hourly,
            f"{prefix}/max_drawdown": max_dd,
            f"{prefix}/final_value": pv[-1] if len(pv) > 0 else 0,
            f"{prefix}/steps": step,
            f"{prefix}/trade_count": int(trade_count),
            f"{prefix}/market_exposure": market_exposure,
            # ── Institutional metrics (PyfolioAnalyzer / empyrical) ──
            f"{prefix}/sortino": pyfolio_metrics.get("sortino_ratio", 0.0),
            f"{prefix}/calmar": pyfolio_metrics.get("calmar_ratio", 0.0),
            f"{prefix}/omega": pyfolio_metrics.get("omega_ratio", 0.0),
            f"{prefix}/stability": pyfolio_metrics.get("stability", 0.0),
            f"{prefix}/daily_var": pyfolio_metrics.get("daily_value_at_risk", 0.0),
            f"{prefix}/win_rate": pyfolio_metrics.get("win_rate", 0.0),
            f"{prefix}/annual_return": pyfolio_metrics.get("annual_return", 0.0),
            # ── Trade quality metrics (manual) ──
            f"{prefix}/profit_factor": profit_factor,
            f"{prefix}/avg_win_loss_ratio": avg_win_loss_ratio,
            f"{prefix}/status": "completed"
        }
        
        wandb.log(metrics)
        logger.info(
            f"{mode} Complete. Return={total_return*100:.2f}%, Sharpe={sharpe:.2f}, "
            f"Sortino={pyfolio_metrics.get('sortino_ratio', 0):.2f}, "
            f"MaxDD={max_dd*100:.2f}%, WinRate={pyfolio_metrics.get('win_rate', 0):.1f}%"
        )
        
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
    parser = argparse.ArgumentParser(description="DeepScalper Pipeline (BDQ / PPO)")
    parser.add_argument("--config", type=str, default="configs/deepscalper_rtx5090.yaml")
    parser.add_argument("--agent", type=str, default="bdq", choices=["bdq", "ppo", "earnhft"],
                        help="Agent type: bdq (default), ppo, or earnhft")
    parser.add_argument("--tags", nargs="*", default=["Pipeline"], help="WandB Tags")
    parser.add_argument("--run_name", type=str, default=None, help="Override WandB Run Name")
    parser.add_argument("--trials", type=int, default=None, help="Number of HPO trials")
    parser.add_argument("--steps", type=int, default=None, help="Training steps override")
    parser.add_argument("--version", type=str, default="V1", help="Version tag")
    args = parser.parse_args()
    
    agent_type = args.agent
    logger.info(f"Agent type: {agent_type}")

    base_config = load_config(args.config)
    
    # =========================================================================
    # 0. Setup Run Name & WandB
    # =========================================================================
    if args.run_name:
        validate_run_name(args.run_name, raise_on_fail=True)
        run_name = args.run_name
    else:
        platform = "GPUHub" if os.path.exists("/workspace") else "Local"
        run_name = generate_run_name(version=args.version, platform=platform)
    
    logger.info(f"Pipeline Run: {run_name}")
    
    # Initialize single WandB run for entire pipeline
    wandb_config = base_config.get("wandb", {})
    wandb.init(
        project=wandb_config.get("project", "FinRL-Pro-DS"),
        entity=wandb_config.get("entity", "bigcan-chiwin-technology"),
        name=run_name,
        tags=wandb_config.get("tags", []) + args.tags,
        config=base_config
    )
    logger.info(f"WandB Run: {wandb.run.url}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    # Enable TF32 for 5th Gen Tensor Cores (RTX 5090 Blackwell)
    # Uses 10-bit mantissa — sufficient for financial signals, ~8x faster than IEEE FP32
    torch.set_float32_matmul_precision('high')
    logger.info("TF32 enabled: torch.set_float32_matmul_precision('high')")
    
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
            # Silence Optuna INFO logs (Start/Finish trial) to avoid WandB console spam
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
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

        # Update WandB config to reflect final parameters (Crucial for reproducibility)
        wandb.config.update(final_config, allow_val_change=True)
        logger.info(f"Final Config for Training: {final_config}")
        
        # =====================================================================
        # PHASE 2: TRAINING
        # =====================================================================
        print("\n" + "="*60)
        print(">>> PHASE 2: TRAINING (Full Run with Best Params)")
        print("="*60 + "\n")
        
        checkpoint_path = run_training(final_config, run_name, device, agent_type=agent_type)
        
        if checkpoint_path:
            # PHASE 3a: Validation Backtest (for Overfitting Check)
            # FIX LEAK-1: Pass train_end_date as cutoff so validation z-scores
            # don't include training data in their rolling windows.
            print(">>> PHASE 3a: VALIDATION BACKTEST")
            data_config = final_config.get("data", {})
            run_backtest(
                final_config, 
                checkpoint_path, 
                device, 
                start_date=data_config.get("val_start_date"), 
                end_date=data_config.get("val_end_date"),
                prefix="backtest_val",
                agent_type=agent_type,
                norm_cutoff_date=data_config.get("val_start_date")  # Reset stats at val boundary
            )

            # PHASE 3b: Test Backtest (for Final Evaluation)
            # FIX LEAK-1: Pass val_end_date as cutoff so test z-scores
            # don't include val/train data in their rolling windows.
            print(">>> PHASE 3b: TEST BACKTEST")
            run_backtest(
                final_config, 
                checkpoint_path, 
                device,
                prefix="backtest_test",
                agent_type=agent_type,
                norm_cutoff_date=data_config.get("test_start_date")  # Reset stats at test boundary
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
