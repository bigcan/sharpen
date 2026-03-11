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

import json
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
import wandb
import optuna
from optuna.samplers import TPESampler, RandomSampler
from urllib.request import Request, urlopen

# Project imports
sys.path.append(os.getcwd())
from finrl_pro_ds.training.deepscalper_trainer import DeepScalperTrainer
from finrl_pro_ds.training.ppo_trainer import PPOTrainer
from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.envs.swing_scalper_env import SwingScalperEnv
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

    env_config = config.get("env", {})
    env_config["reward"] = config.get("env", {}).get("reward", {})
    # Forward network config so env can read micro_config.input_size for fev3 compat
    env_config["network"] = config.get("network", {})
    # Forward features config for include_spread, n_levels, asset_class
    env_config["features"] = config.get("features", {})

    # V6 swing MDP: binary direction-switching (Phase K)
    # V7 continuous swing MDP: SAC position control (GMGP1)
    mdp_version = env_config.get("mdp_version", "v5")
    if mdp_version == "v7":
        from finrl_pro_ds.envs.continuous_swing_env import ContinuousSwingEnv
        from finrl_pro_ds.data.multiscale_handler import MultiScaleOHLCVHandler
        features_cfg = config.get("features", {})
        ms_handler = MultiScaleOHLCVHandler(
            file_path=file_path,
            ticker=ticker,
            feature_config=features_cfg,
            start_date=sd,
            end_date=ed,
            norm_cutoff_date=norm_cutoff_date,
        )
        return ContinuousSwingEnv(config=env_config, data_handler=ms_handler)

    handler = ParquetDataHandler(
        file_path=file_path,
        ticker=ticker,
        feature_config=config.get("features", {}),
        start_date=sd,
        end_date=ed,
        shared_memory_config=shm_config,
        norm_cutoff_date=norm_cutoff_date  # FIX LEAK-1
    )

    if mdp_version == "v6":
        return SwingScalperEnv(config=env_config, data_handler=handler)
    return DeepScalperEnv(config=env_config, data_handler=handler)


def create_vector_env(config, num_envs, start_date=None, end_date=None, shm_config=None, gym_shm=True, use_sync=False, norm_cutoff_date=None):
    """Create vectorized environment for training.

    Note: Using AsyncVectorEnv for parallel data loading. Context 'spawn' is used
    for CUDA/PyTorch safety. Set use_sync=True to use SyncVectorEnv (no subprocesses),
    which avoids IPC/FD limits on constrained containers.
    """
    # FIX BUG-02: Forward norm_cutoff_date to individual envs for normalization isolation
    env_factory = functools.partial(make_env, config=config, start_date=start_date, end_date=end_date, shm_config=shm_config, norm_cutoff_date=norm_cutoff_date)

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
def evaluate_for_hpo(env, agent, max_steps=5000, bar_minutes=1):
    """Evaluate agent for HPO — returns (profit_factor, trade_count).

    V4.2: Changed from raw Sharpe to profit_factor to prevent specification gaming.
    Also tracks trade_count for the activity constraint (min 100 trades).

    IMPORTANT: This function handles both single envs and VectorEnvs.
    VectorEnv returns info as a dict of arrays, or for newer Gymnasium versions,
    as a tuple (info_dict, final_info_dict).
    """
    all_returns = []
    positions = []  # V4.2: Track positions for trade counting
    action_counts = {0: 0, 1: 0, 2: 0}  # Track action distribution
    trade_pnls = []  # FIX GMO1-06: Track PnL per completed swing

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

    # FIX GMO1-02: Detect discrete_dims BEFORE the eval loop.
    # Previously defined after the loop, causing NameError on action tracking.
    if hasattr(env.action_space, 'shape') and env.action_space.shape and not hasattr(env.action_space, 'n'):
        discrete_dims = -1  # V7: continuous action space
    elif hasattr(env.action_space, 'n'):
        discrete_dims = env.action_space.n
    elif hasattr(env.action_space, 'nvec'):
        discrete_dims = int(env.action_space.nvec[0])
    else:
        discrete_dims = 6

    try:
        obs, info = env.reset()
        done = False
        step = 0

        prev_val = extract_portfolio_value(info, env_idx=0, default=100000.0)
        prev_realized_pnl = 0.0  # Track cumulative realized pnl to get per-trade delta
        # FIX GMO1-01: Track direction for fee_threshold context
        current_direction = None  # Set after first step from info["direction"]

        while not done and step < max_steps:
            # Dispatch obs keys based on agent type
            if "scale_0" in obs:
                # SAC / V7 multi-scale obs (positional keys: scale_0, scale_1, ...)
                scale_tensors = []
                for si in range(100):  # find all scale_N keys
                    sk = f"scale_{si}"
                    if sk not in obs:
                        break
                    scale_tensors.append(
                        torch.tensor(obs[sk], dtype=torch.float32).to(agent.device, non_blocking=True)
                    )
                priv = torch.tensor(obs["private"], dtype=torch.float32).to(agent.device, non_blocking=True)
                pred = agent.predict(scale_tensors, priv, deterministic=True)
            else:
                micro = torch.tensor(obs["micro"], dtype=torch.float32).to(agent.device, non_blocking=True)
                private = torch.tensor(obs["private"], dtype=torch.float32).to(agent.device, non_blocking=True)
                macro = torch.tensor(obs["macro"], dtype=torch.float32).to(agent.device, non_blocking=True)
                # FIX GMO1-01: Pass direction context for fee_threshold filtering
                ctx = {"current_direction": current_direction} if current_direction is not None else None
                pred = agent.predict(micro, private, macro, deterministic=True, context=ctx)

            # PPO returns (actions, log_probs, values), BDQ returns just actions
            if isinstance(pred, tuple):
                action = pred[0]  # PPO: extract actions from tuple
            else:
                action = pred  # BDQ: already just actions

            # Track action distribution (first env if vectorized)
            first_action = action[0] if hasattr(action, 'shape') and len(action.shape) > 1 else action
            if discrete_dims == -1:
                # V7 continuous: bucket into long/flat/short
                act_val = float(first_action[0]) if hasattr(first_action, "__len__") else float(first_action)
                if act_val > 0.1:
                    action_counts[0] = action_counts.get(0, 0) + 1  # long
                elif act_val < -0.1:
                    action_counts[2] = action_counts.get(2, 0) + 1  # short
                else:
                    action_counts[1] = action_counts.get(1, 0) + 1  # flat
            else:
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

            # FIX GMO1-01: Update direction context for fee_threshold filtering
            dir_val = info.get("direction")
            if dir_val is not None:
                current_direction = np.array([float(dir_val[0])]) if hasattr(dir_val, "__len__") else np.array([float(dir_val)])

            # V4.2: Track position for trade counting
            pos = info.get("position")
            if pos is not None:
                if hasattr(pos, "__len__") and not isinstance(pos, str):
                    positions.append(float(pos[0]))
                else:
                    positions.append(float(pos))

            # FIX GMO1-06: Trade-level PF via realized_pnl and switched flag
            switched = info.get("switched")
            if switched is not None:
                # Handle VectorEnv format
                is_switch = bool(switched[0]) if hasattr(switched, "__len__") and not isinstance(switched, str) else bool(switched)
                if is_switch:
                    rpnl = info.get("realized_pnl")
                    if rpnl is not None:
                        curr_rpnl = float(rpnl[0]) if hasattr(rpnl, "__len__") and not isinstance(rpnl, str) else float(rpnl)
                        trade_pnl = curr_rpnl - prev_realized_pnl
                        trade_pnls.append(trade_pnl)
                        prev_realized_pnl = curr_rpnl

            step_return = (curr_val - prev_val) / prev_val if prev_val > 0 else 0
            all_returns.append(step_return)
            prev_val = curr_val
            step += 1

    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        import traceback
        wandb.log({"_debug/eval_error": str(e), "_debug/eval_traceback": traceback.format_exc()})
        # BUG-B: Restore training hidden state even on error
        # AUDIT FIX C1: Guard to prevent phantom attribute on stateless agents
        if hasattr(agent, '_hidden_state'):
            agent._hidden_state = _saved_hidden
        return 0.0, 0  # V4.2: (profit_factor, trade_count)

    returns = np.array(all_returns)
    pos_arr = np.array(positions) if positions else np.array([0.0])

    # discrete_dims already detected before the eval loop (FIX GMO1-02)

    # V4.2: Count trades (position changes) — same logic as backtest
    # FIX K03: V6 (always-in-market) double-counts: base_count and sign_flips both fire on switch.
    # For V6, sign_flips alone is the correct trade count.
    pos_deltas = np.abs(np.diff(pos_arr))
    base_count = int(np.sum(pos_deltas > 1e-6))
    sign_flips = int(np.sum((pos_arr[:-1] * pos_arr[1:]) < -1e-9))
    if discrete_dims == -1:
        trade_count = base_count  # V7: continuous, count all position changes
    elif discrete_dims == 2:
        trade_count = sign_flips  # V6: always in market, sign flip = one switch
    else:
        trade_count = base_count + sign_flips

    # FIX GMO1-06: Compute profit_factor based on trade-level PnL, not bar-by-bar returns
    trade_pnls_arr = np.array(trade_pnls)
    if len(trade_pnls_arr) > 0:
        positive_pnls = trade_pnls_arr[trade_pnls_arr > 0]
        negative_pnls = trade_pnls_arr[trade_pnls_arr < 0]
        gross_profit = float(np.sum(positive_pnls))
        gross_loss = float(np.abs(np.sum(negative_pnls)))
        profit_factor = gross_profit / gross_loss if gross_loss > 1e-12 else (10.0 if gross_profit > 1e-12 else 0.0)
    else:
        # Fallback if no trades or no info["switched"] support
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        gross_profit = float(np.sum(positive_returns)) if len(positive_returns) > 0 else 0.0
        gross_loss = float(np.abs(np.sum(negative_returns))) if len(negative_returns) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 1e-12 else (10.0 if gross_profit > 1e-12 else 0.0)

    # Diagnostic: log what we computed including action distribution
    # discrete_dims already detected above for trade counting
    if discrete_dims == -1:
        action_labels = {0: "long", 1: "flat", 2: "short"}
    elif discrete_dims == 2:
        action_labels = {0: "long", 1: "short"}
    elif discrete_dims == 3:
        action_labels = {0: "taker_buy", 1: "hold", 2: "taker_sell"}
    else:
        action_labels = {0: "taker_buy", 1: "maker_buy", 2: "hold", 3: "cancel", 4: "maker_sell", 5: "taker_sell"}
    diag = {
        "_debug/eval_steps": step,
        "_debug/eval_returns_len": len(returns),
        "_debug/eval_returns_std": float(np.std(returns)) if len(returns) > 0 else 0.0,
        "_debug/eval_returns_mean": float(np.mean(returns)) if len(returns) > 0 else 0.0,
        "_debug/eval_final_pv": prev_val,
        "_debug/eval_trade_count": trade_count,
        "_debug/eval_profit_factor": profit_factor,
    }
    for idx, label in action_labels.items():
        diag[f"_debug/eval_action_{idx}_{label}"] = action_counts.get(idx, 0)
    wandb.log(diag)

    # Also log Sharpe for research tracking (not used for HPO scoring)
    if len(returns) > 1 and np.std(returns) > 1e-9:
        raw_ratio = np.mean(returns) / np.std(returns)
        # FIX R7-AUD-03: Use correct annualization for bar duration (not hardcoded 1-min)
        bars_per_year = 525600 / bar_minutes
        sharpe_minute = raw_ratio * np.sqrt(bars_per_year)
        # FIX R7-AUD-04: Correct hourly aggregation for bar duration
        n_per_hour = max(1, 60 // bar_minutes)
        hourly_returns = np.add.reduceat(returns, np.arange(0, len(returns), n_per_hour))
        # FIX BUG-05: Drop last partial bucket to avoid upward Sharpe bias
        # A partial bucket with fewer than 60 samples has lower variance, inflating Sharpe.
        if len(returns) % n_per_hour != 0 and len(hourly_returns) > 1:
            hourly_returns = hourly_returns[:-1]
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
        logger.warning(f"Zero Sharpe: steps={step}, len={len(returns)}, std={np.std(returns) if len(returns) > 0 else 'N/A'}, actions={action_counts}")

    # BUG-B: Restore training hidden state after eval
    # AUDIT FIX C1: Guard to prevent phantom attribute on stateless agents
    if hasattr(agent, '_hidden_state'):
        agent._hidden_state = _saved_hidden

    # V4.2: Return profit_factor + trade_count for anti-specification-gaming.
    # Profit factor is immune to the Sharpe smoothness hack. Trade count
    # enables the activity constraint (min 100 trades to pass).
    return profit_factor, trade_count


def _create_sampler(hpo_config: dict):
    """Create Optuna sampler from config. Supports 'tpe' (default) and 'random'."""
    sampler_type = hpo_config.get("sampler", "tpe").lower()
    seed = hpo_config.get("sampler_seed", 42)
    if sampler_type == "random":
        logger.info(f"Using RandomSampler (seed={seed})")
        return RandomSampler(seed=seed)
    # Default: TPE with multivariate correlation modeling
    logger.info(f"Using TPESampler (seed={seed}, n_startup_trials=10, multivariate=True)")
    return TPESampler(seed=seed, n_startup_trials=10, multivariate=True)


def run_hpo(base_config, n_trials, steps_per_trial, device, agent_type="bdq"):
    """Phase 1: Hyperparameter Optimization with Optuna. Supports BDQ and PPO agents."""
    logger.info(f"Starting HPO: {n_trials} trials, {steps_per_trial} steps each")
    wandb.log({"hpo/status": "started", "hpo/n_trials": n_trials})

    def objective(trial):
        trial_prefix = f"hpo/t{trial.number}"
        wandb.log({f"{trial_prefix}/started": True})

        # Create trial config
        config = copy.deepcopy(base_config)
        config["training"]["total_timesteps"] = steps_per_trial

        # -----------------------------------------------------------
        # Agent-specific HPO hyperparameter sampling
        # V4.2: OPTIMIZER HPs ONLY — MDP/reward params are LOCKED.
        # Rationale: Tuning reward params (gamma, sharpe_weight) allows
        # Optuna to change the game definition, causing specification gaming.
        # See: expert DRL audit, "Goodhart's Law" in RL.
        # -----------------------------------------------------------
        if agent_type == "sac":
            # SAC hyperparams — actor/critic LR, tau, alpha, deadband
            lr_actor = trial.suggest_float("lr_actor", 1e-4, 1e-3, log=True)
            lr_critic = trial.suggest_float("lr_critic", 1e-4, 1e-3, log=True)
            tau = trial.suggest_float("tau", 0.001, 0.01, log=True)
            initial_alpha = trial.suggest_float("initial_alpha", 0.05, 0.5, log=True)
            deadband = trial.suggest_categorical("deadband_threshold", [0.15, 0.25, 0.35])

            config["agents"]["sac"]["lr_actor"] = lr_actor
            config["agents"]["sac"]["lr_critic"] = lr_critic
            config["agents"]["sac"]["tau"] = tau
            config["agents"]["sac"]["initial_alpha"] = initial_alpha
            config["env"]["deadband_threshold"] = deadband

            wandb.log({
                f"{trial_prefix}/lr_actor": lr_actor,
                f"{trial_prefix}/lr_critic": lr_critic,
                f"{trial_prefix}/tau": tau,
                f"{trial_prefix}/initial_alpha": initial_alpha,
                f"{trial_prefix}/deadband_threshold": deadband,
            })
        elif agent_type == "ppo":
            # === OPTIMIZER HPs (tunable) ===
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True)
            ent_coef = trial.suggest_float("ent_coef", 0.005, 0.1, log=True)
            gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.98)
            n_epochs = trial.suggest_categorical("n_epochs", [3, 5, 8])
            target_kl = trial.suggest_float("target_kl", 0.01, 0.04)
            max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.5, 1.0, 5.0])
            clip_eps = trial.suggest_categorical("clip_eps", [0.1, 0.2])

            config["agents"]["ppo"]["learning_rate"] = learning_rate
            config["agents"]["ppo"]["ent_coef"] = ent_coef
            config["agents"]["ppo"]["gae_lambda"] = gae_lambda
            config["agents"]["ppo"]["n_epochs"] = n_epochs
            config["agents"]["ppo"]["target_kl"] = target_kl
            config["agents"]["ppo"]["max_grad_norm"] = max_grad_norm
            config["agents"]["ppo"]["clip_eps"] = clip_eps
            # NOTE: gamma, batch_size, sharpe_weight, hindsight_* are
            # READ FROM CONFIG and never overridden by HPO.

            wandb.log({
                f"{trial_prefix}/learning_rate": learning_rate,
                f"{trial_prefix}/ent_coef": ent_coef,
                f"{trial_prefix}/gae_lambda": gae_lambda,
                f"{trial_prefix}/n_epochs": n_epochs,
                f"{trial_prefix}/target_kl": target_kl,
                f"{trial_prefix}/max_grad_norm": max_grad_norm,
                f"{trial_prefix}/clip_eps": clip_eps,
            })
        elif agent_type == "iqn":
            # IQN hyperparams — optimizer HPs + gamma (discount horizon)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
            num_quantiles = trial.suggest_categorical("num_quantiles", [8, 16, 32, 64])
            noisy_sigma0 = trial.suggest_float("noisy_sigma0", 0.3, 0.7)
            tau = trial.suggest_float("tau", 0.001, 0.01, log=True)

            config["env"]["reward"]["sharpe_weight"] = 0.0
            config["env"]["reward"]["hindsight_weight"] = 0.0
            config["agents"]["iqn"]["learning_rate"] = learning_rate
            config["agents"]["iqn"]["num_quantiles"] = num_quantiles
            config["agents"]["iqn"]["noisy_sigma0"] = noisy_sigma0
            config["agents"]["iqn"]["tau"] = tau

            # Gamma routing: multi_horizon has separate short/long gammas;
            # single-horizon tunes one gamma. Base gamma must equal gamma_long
            # to keep N-step buffer consistent with long-horizon Bellman targets.
            is_multi_horizon = config["agents"]["iqn"].get("multi_horizon", False)
            if is_multi_horizon:
                gamma_short = trial.suggest_float("gamma_short", 0.90, 0.98)
                gamma_long = trial.suggest_float("gamma_long", 0.95, 0.999)
                config["agents"]["iqn"]["gamma_short"] = gamma_short
                config["agents"]["iqn"]["gamma_long"] = gamma_long
                config["agents"]["iqn"]["gamma"] = gamma_long  # Sync base gamma = long
            else:
                gamma = trial.suggest_float("gamma", 0.93, 0.999)
                config["agents"]["iqn"]["gamma"] = gamma

            # CRRA risk-aversion coefficient (reward shaping, NOT reward structure)
            # Only tune if config has crra_gamma > 0 (opt-in per experiment)
            if config["env"].get("reward", {}).get("crra_gamma", 0.0) > 0:
                crra_gamma = trial.suggest_float("crra_gamma", 0.0, 1.5)
                config["env"]["reward"]["crra_gamma"] = crra_gamma

            # GMO1 opt-in HPO expansions (config-gated, backward compatible)
            # stay_reward_weight: only when switch_centric reward mode
            if config["env"].get("reward", {}).get("mode") == "switch_centric":
                stay_reward_weight = trial.suggest_float("stay_reward_weight", 0.05, 0.3)
                config["env"]["reward"]["stay_reward_weight"] = stay_reward_weight

            # n_step HPO: categorical sweep {3,5,7,10}
            if config["agents"]["iqn"].get("n_step_hpo", False):
                n_step = trial.suggest_categorical("n_step", [3, 5, 7, 10])
                config["agents"]["iqn"]["n_step"] = n_step

            # buffer_size HPO: log-uniform [500K, 2M] (scaled for 5090 32GB VRAM)
            if config["agents"]["iqn"].get("buffer_size_hpo", False):
                buffer_size = trial.suggest_int("buffer_size", 500000, 2000000, log=True)
                config["agents"]["iqn"]["buffer_size"] = buffer_size

            # hidden_dim HPO: categorical {64, 128, 256}
            if config.get("network", {}).get("hidden_dim_hpo", False):
                hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
                config["network"]["micro_config"]["hidden_size"] = hidden_dim
                # Scale macro hidden sizes proportionally
                config["network"]["macro_config"]["hidden_sizes"] = [hidden_dim, hidden_dim // 2]

            hpo_log = {
                f"{trial_prefix}/learning_rate": learning_rate,
                f"{trial_prefix}/num_quantiles": num_quantiles,
                f"{trial_prefix}/noisy_sigma0": noisy_sigma0,
                f"{trial_prefix}/tau": tau,
            }
            if config["env"].get("reward", {}).get("crra_gamma", 0.0) > 0:
                hpo_log[f"{trial_prefix}/crra_gamma"] = crra_gamma
            if is_multi_horizon:
                hpo_log[f"{trial_prefix}/gamma_short"] = gamma_short
                hpo_log[f"{trial_prefix}/gamma_long"] = gamma_long
            else:
                hpo_log[f"{trial_prefix}/gamma"] = gamma
            if config["env"].get("reward", {}).get("mode") == "switch_centric":
                hpo_log[f"{trial_prefix}/stay_reward_weight"] = stay_reward_weight
            if config["agents"]["iqn"].get("n_step_hpo", False):
                hpo_log[f"{trial_prefix}/n_step"] = n_step
            if config["agents"]["iqn"].get("buffer_size_hpo", False):
                hpo_log[f"{trial_prefix}/buffer_size"] = buffer_size
            if config.get("network", {}).get("hidden_dim_hpo", False):
                hpo_log[f"{trial_prefix}/hidden_dim"] = hidden_dim
            wandb.log(hpo_log)
        else:
            # BDQ hyperparams (4 dimensions) — optimizer HPs only
            # FIX BUG-01: MDP-defining params (gamma, reward) are LOCKED in config.
            # gamma locked at 0.95 per T1.2 signal horizon alignment.
            # PERF-OPT: batch_size LOCKED (not tuned). It has 4.7x SPS impact and
            # fundamentally changes the hardware profile — tuning it conflates
            # learning quality with throughput. Read from config instead.
            auxiliary_weight = trial.suggest_float("auxiliary_weight", 0.5, 1.5, log=True)
            learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
            epsilon_end = trial.suggest_float("epsilon_end", 0.01, 0.10)
            tau = trial.suggest_float("tau", 0.001, 0.01, log=True)

            config["env"]["reward"]["sharpe_weight"] = 0.0  # Pure paper reward for BDQ
            config["env"]["reward"]["hindsight_weight"] = 0.0  # FIX FIND-V3-05: Lock hindsight (BUG-01 invariant)
            config["agents"]["bdq"]["auxiliary_weight"] = auxiliary_weight
            config["agents"]["bdq"]["learning_rate"] = learning_rate
            # NOTE: gamma read from config (locked), NOT tuned by HPO
            # NOTE: batch_size read from config (locked), NOT tuned by HPO (PERF-OPT)
            config["agents"]["bdq"]["epsilon_end"] = epsilon_end
            config["agents"]["bdq"]["tau"] = tau

            wandb.log({
                f"{trial_prefix}/learning_rate": learning_rate,
                f"{trial_prefix}/auxiliary_weight": auxiliary_weight,
                f"{trial_prefix}/epsilon_end": epsilon_end,
                f"{trial_prefix}/tau": tau,
            })

        # -----------------------------------------------------------
        # Create env and train (agent-agnostic)
        # -----------------------------------------------------------
        env = None
        eval_env = None
        try:
            # FIX: Use SyncVectorEnv for HPO to avoid AsyncVectorEnv pipe crashes
            hpo_num_envs = min(config.get("training", {}).get("num_envs",
                              config["env"].get("num_envs", 12)), 12)
            env = create_vector_env(config, num_envs=hpo_num_envs, gym_shm=False, use_sync=True)

            if agent_type == "sac":
                from finrl_pro_ds.training.sac_trainer import SACTrainer
                trainer = SACTrainer(env, config, device=device, hpo_mode=True)
            elif agent_type == "ppo":
                trainer = PPOTrainer(env, config, device=device, hpo_mode=True)
            else:
                # DeepScalperTrainer handles both BDQ and IQN (auto-detects from config)
                trainer = DeepScalperTrainer(env, config, device=device, hpo_mode=True)

            # V4.2: Evaluate on VALIDATION set (anti-overfitting)
            # Training happens on Jan-Apr, evaluation on May.
            data_cfg = config.get("data", {})
            # FIX AUD-S129-01: HPO eval must use final fee from fee_schedule, not
            # the initial 0.0 from fee curriculum. Same pattern as R2-AUD-03 for backtest.
            eval_config = copy.deepcopy(config)
            fee_schedule = eval_config.get("env", {}).get("fee_schedule")
            if fee_schedule:
                final_tier = fee_schedule[-1]
                final_fee = final_tier.get("ramp_to", final_tier.get("taker_fee", 0.0))
                eval_config["env"]["taker_fee"] = final_fee
                logger.info(f"[AUD-S129-01] HPO eval fee overridden from fee_schedule: {final_fee:.6f}")
            eval_env = create_vector_env(
                eval_config, num_envs=1, gym_shm=False, use_sync=True,
                start_date=data_cfg.get("val_start_date"),
                end_date=data_cfg.get("val_end_date"),
                # FIX BUG-02: HPO eval env must have normalization isolation.
                # Without this, EMA stats from training data leak into validation scoring.
                norm_cutoff_date=data_cfg.get("val_start_date"),
            )

            # FIX R7-AUD-03: Compute bar_minutes for correct Sharpe annualization
            mdp_ver = config.get("env", {}).get("mdp_version", "v5")
            hpo_scales = config.get("features", {}).get("scales", [])
            if mdp_ver == "v7" and hpo_scales:
                hpo_bar_minutes = min(hpo_scales)
            elif "15min" in config.get("data", {}).get("file_path", ""):
                hpo_bar_minutes = 15
            elif "3min" in config.get("data", {}).get("file_path", ""):
                hpo_bar_minutes = 3
            elif "5min" in config.get("data", {}).get("file_path", ""):
                hpo_bar_minutes = 5
            else:
                hpo_bar_minutes = 1

            # Define Pruning Callback (also uses validation env)
            def pruning_callback():
                pf, tc = evaluate_for_hpo(eval_env, trainer.agent, max_steps=3000, bar_minutes=hpo_bar_minutes)
                return pf  # Optuna pruner expects a single float

            trainer.train(optuna_trial=trial, pruning_callback=pruning_callback)

            # V4.2: Multi-seed eval for robust PF measurement (3 seeds, median)
            pf_values = []
            tc_values = []
            for eval_seed in [42, 123, 7]:
                eval_env.reset(seed=eval_seed)
                pf, tc = evaluate_for_hpo(eval_env, trainer.agent, max_steps=50000, bar_minutes=hpo_bar_minutes)
                pf_values.append(pf)
                tc_values.append(tc)
            profit_factor = float(np.median(pf_values))
            trade_count = int(np.median(tc_values))

            # V4.2: Activity constraint — kill lazy holding agents
            # FIX HPO-3: Lowered from 100 to 30 for swing MDP. Binary {Long, Short}
            # with cooldown_bars=2-3 naturally produces fewer trades per eval window.
            # K1 killed trials at 54 and 78 trades — both were active agents.
            min_trades = 30
            if trade_count < min_trades:
                wandb.log({f"{trial_prefix}/killed": "lazy_agent", f"{trial_prefix}/trades": trade_count})
                logger.info(f"Trial {trial.number}: KILLED (only {trade_count} trades, min={min_trades})")
                return -999.0

            wandb.log({
                f"{trial_prefix}/profit_factor": profit_factor,
                f"{trial_prefix}/trade_count": trade_count,
                f"{trial_prefix}/completed": True,
            })
            logger.info(f"Trial {trial.number}: PF={profit_factor:.4f}, Trades={trade_count}")

            return profit_factor

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
    logger.info(f"Creating Optuna study (storage={base_config.get('hpo', {}).get('storage')})...")
    study = optuna.create_study(
        direction="maximize",
        storage=base_config.get("hpo", {}).get("storage"),
        study_name=f"hpo_{agent_type}",
        load_if_exists=True,
        sampler=_create_sampler(base_config.get("hpo", {})),
        # FIX HPO-1: Disable inter-trial pruning for swing MDP.
        # HyperbandPruner was killing trials before IQN+NoisyNets could converge
        # (needs 200K+ steps for signal). All 5 trials pruned in K2/K4 runs.
        # Let all trials run to completion so we get clean data points to
        # distinguish "bad hyperparams" from "bad MDP design".
        pruner=optuna.pruners.NopPruner()
    )
    study.optimize(objective, n_trials=n_trials)

    # Log best results
    best = study.best_trial
    agent_key = agent_type if agent_type in ("ppo", "iqn", "sac") else "bdq"
    best_params = {
        "env": {"reward": {}},
        "agents": {agent_key: {}},
        "training": {}
    }

    # V4.2: Explicit routing for ALL HPO params to prevent silent mis-routing.
    if agent_type == "sac":
        reward_params = set()
        agent_params = {"lr_actor", "lr_critic", "tau", "initial_alpha"}
        # deadband routes to env, not agent
        env_params = {"deadband_threshold"}
    elif agent_type == "ppo":
        # V4.2: PPO locks reward params — only optimizer HPs are tunable
        reward_params = set()
        agent_params = {"learning_rate", "ent_coef", "gae_lambda", "n_epochs", "target_kl", "max_grad_norm", "clip_eps"}
    elif agent_type == "iqn":
        # FIX GMO1-03: Added stay_reward_weight, n_step, buffer_size routing
        reward_params = {"crra_gamma", "stay_reward_weight"}
        agent_params = {"learning_rate", "num_quantiles", "noisy_sigma0", "tau",
                        "gamma", "gamma_short", "gamma_long", "n_step", "buffer_size"}
    else:
        # FIX BUG-01+BUG-10: BDQ reward/MDP params are LOCKED (gamma read from config)
        # PERF-OPT: batch_size removed — locked in config (hardware-profile param, not learning param)
        reward_params = set()
        agent_params = {"auxiliary_weight", "learning_rate", "epsilon_end", "tau", "gamma"}

    # FIX GMO1-03: hidden_dim routes to network config, not agents
    network_params = {"hidden_dim"}

    for key, val in best.params.items():
        if key in reward_params:
            best_params["env"]["reward"][key] = val
        elif key in agent_params:
            best_params["agents"][agent_key][key] = val
        elif agent_type == "sac" and key in env_params:
            best_params["env"][key] = val
        elif key in network_params:
            if "network" not in best_params:
                best_params["network"] = {"micro_config": {}, "macro_config": {}}
            best_params["network"]["micro_config"]["hidden_size"] = val
            best_params["network"]["macro_config"] = {"hidden_sizes": [val, val // 2]}
        else:
            raise ValueError(f"HPO param '{key}' has no routing rule. Add to reward_params or agent_params in run_hpo().")

    # V4.2: best.value is now profit_factor (not Sharpe)
    wandb.log({
        "hpo/best_profit_factor": best.value,
        "hpo/best_trial": best.number,
        "hpo/best_params": str(best.params),
        "hpo/status": "completed"
    })

    logger.info(f"HPO Complete. Best Profit Factor: {best.value:.4f} (Trial #{best.number})")
    return best_params


# ============================================================================
# PHASE 2: TRAINING
# ============================================================================
def run_training(config, run_name, device, agent_type="bdq", warm_start=None):
    """Phase 2: Full training with optimized hyperparameters."""
    logger.info(f"Starting Training Phase (agent={agent_type})")
    wandb.log({"train/status": "started", "train/agent_type": agent_type})

    # FIX R6-AUD-02: num_envs may be under "training" or "env" depending on config version.
    # Check training first (GMGP1 configs), then env (legacy), then default.
    num_envs = config.get("training", {}).get("num_envs",
               config.get("env", {}).get("num_envs", 12))
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
        use_sync = config.get("training", {}).get("use_sync", False)
        env = create_vector_env(config, num_envs, shm_config=shm_config, gym_shm=use_shm, use_sync=use_sync)
        logger.info(f"Environment ready: {num_envs} workers")

        # Train — dispatch based on agent type
        if agent_type == "sac":
            from finrl_pro_ds.training.sac_trainer import SACTrainer
            trainer = SACTrainer(env, config, device=device, run_name=run_name)
        elif agent_type == "ppo":
            trainer = PPOTrainer(env, config, device=device, run_name=run_name)
        else:
            trainer = DeepScalperTrainer(env, config, device=device, run_name=run_name)

        # Warm-start: load pretrained weights (strict=False for architecture mismatch,
        # e.g. bandit K8 single-head → K7 multi-horizon dual-head transfer)
        if warm_start:
            if agent_type == "sac":
                trainer.agent.load(warm_start)
            else:
                trainer.load_checkpoint(warm_start, strict=False)
            logger.info(f"[Warm-Start] Loaded weights from {warm_start}")

        if agent_type == "sac":
            trainer.train()
        else:
            trainer.train()

        # Find checkpoint
        checkpoints_dir = f"checkpoints/{run_name}"
        checkpoint_path = None
        if os.path.exists(checkpoints_dir):
            # Prefer checkpoint_final.pth (saved after all training steps)
            final_ckpt = os.path.join(checkpoints_dir, "checkpoint_final.pth")
            if os.path.exists(final_ckpt):
                checkpoint_path = final_ckpt
            else:
                # Fallback: highest step number (numerical sort, not alphabetical)
                step_ckpts = [f for f in os.listdir(checkpoints_dir)
                              if f.startswith("checkpoint_step_") and f.endswith(".pth")]
                if step_ckpts:
                    step_ckpts.sort(key=lambda f: int(f.replace("checkpoint_step_", "").replace(".pth", "")))
                    checkpoint_path = os.path.join(checkpoints_dir, step_ckpts[-1])

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
        # Fix Issue #1: Disable Private State Augmentation during backtest
        # We need deterministic starting states (Initial Balance, Pos=0), not random ones.
        backtest_config = copy.deepcopy(config)
        if "env" not in backtest_config: backtest_config["env"] = {}
        backtest_config["env"]["private_state_augment_prob"] = 0.0
        # FIX BUG-03: Disable hindsight reward during backtest — it uses future prices
        # which inflates evaluation metrics. Hindsight is a training-only shaping signal.
        if "reward" not in backtest_config["env"]: backtest_config["env"]["reward"] = {}
        backtest_config["env"]["reward"]["hindsight_weight"] = 0.0
        # FIX BUG-17: Override daily episode settings for backtest — must evaluate
        # full test period sequentially, not a single random day.
        backtest_config["env"]["episode_length"] = 0   # Full dataset
        backtest_config["env"]["random_start"] = False  # Sequential from start
        # FIX R2-AUD-03: Backtest must use final fee from fee_schedule, not initial 0.
        # Fee curriculum is a training concept; backtest evaluates at production fee level.
        fee_schedule = backtest_config.get("env", {}).get("fee_schedule")
        if fee_schedule:
            final_tier = fee_schedule[-1]
            final_fee = final_tier.get("ramp_to", final_tier.get("taker_fee", 0.0))
            backtest_config["env"]["taker_fee"] = final_fee
            logger.info(f"[R2-AUD-03] Backtest fee overridden from fee_schedule: {final_fee:.6f}")

        env = make_env(backtest_config, start_date=start_date, end_date=end_date, norm_cutoff_date=norm_cutoff_date)

        # Create agent
        sample_obs, _ = env.reset()
        network_config = dict(config.get("network", {}))
        if not network_config:
            raise ValueError("Config missing 'network' section — cannot reconstruct agent for backtest")

        # FIX R6-AUD-03: Inject n_scales (dynamically derived in SACTrainer but missing here).
        # Without this, SACAgent defaults to n_scales=3 which silently breaks for != 3 scales.
        bt_scales = config.get("features", {}).get("scales",
                    config.get("env", {}).get("scales", [3, 15, 60]))
        network_config["n_scales"] = len(bt_scales)

        # Read action dims from config (mirrors trainer logic exactly)
        # FIX BUG-15: PPO uses Discrete(N) — action_dims MUST be int, not tuple.
        # FIX BUG-16: Must check size_dims BEFORE discrete_dims (parity with trainer).
        action_config = config.get("env", {}).get("action", {})
        if agent_type == "ppo":
            # PPO: Discrete(N) — same default as ppo_trainer.py:60
            action_dims = action_config.get("discrete_dims", 6)
        elif "size_dims" in action_config and int(action_config["size_dims"]) > 0:
            # H2: Leverage-Aware Sizing — MultiDiscrete([size_dims, direction_dims])
            action_dims = (
                int(action_config["size_dims"]),
                int(action_config.get("direction_dims", 3))
            )
        elif "discrete_dims" in action_config:
            action_dims = action_config["discrete_dims"]
        else:
            # BDQ: Legacy MultiDiscrete (price_bins, qty_bins)
            signed_qty_props = action_config.get(
                "signed_qty_proportions", [-0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5]
            )
            action_dims = (
                action_config.get("price_bins", 5),
                len(signed_qty_props)
            )
        # FIX: Inject action_space_dims into network_config (mirrors trainer logic).
        # Without this, the BDQ/PPO agent assertion falls back to default (5,9).
        network_config["action_space_dims"] = action_dims

        # Dispatch agent creation based on type
        if agent_type == "sac":
            from finrl_pro_ds.agents.sac.sac_agent import SACAgent
            sac_cfg = config.get("agents", {}).get("sac", {})
            agent = SACAgent(
                network_config=network_config,
                lr_actor=sac_cfg.get("lr_actor", 3e-4),
                lr_critic=sac_cfg.get("lr_critic", 3e-4),
                lr_alpha=sac_cfg.get("lr_alpha", 3e-4),
                gamma=sac_cfg.get("gamma", 0.99),
                tau=sac_cfg.get("tau", 0.005),
                batch_size=sac_cfg.get("batch_size", 256),
                buffer_size=100,  # FIX R2-AUD-06: Minimal buffer for backtest (never used)
                initial_alpha=sac_cfg.get("initial_alpha", 0.2),
                device=device,
            )
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
        elif agent_type == "iqn":
            from finrl_pro_ds.agents.deepscalper.iqn_agent import IQNAgent
            iqn_cfg = config.get("agents", {}).get("iqn", {})
            agent = IQNAgent(
                network_config=network_config,
                lr=iqn_cfg.get("learning_rate", 3e-4),
                gamma=iqn_cfg.get("gamma", 0.99),
                tau=iqn_cfg.get("tau", 0.005),
                buffer_size=100,  # FIX R8-AUD-03: backtest doesn't use replay buffer
                num_quantiles=iqn_cfg.get("num_quantiles", 32),
                embedding_dim=iqn_cfg.get("embedding_dim", 64),
                noisy_sigma0=iqn_cfg.get("noisy_sigma0", 0.5),
                fee_threshold=iqn_cfg.get("fee_threshold", 0.0),  # FIX GMO1-05
                multi_horizon=iqn_cfg.get("multi_horizon", False),  # FIX GMO1-09
                gamma_short=iqn_cfg.get("gamma_short", 0.95),  # FIX GMO1-09
                gamma_long=iqn_cfg.get("gamma_long", 0.99),  # FIX GMO1-09
                horizon_alpha=iqn_cfg.get("horizon_alpha", 0.5),  # FIX GMO1-09
                action_dims=action_dims,
                use_amp=config.get("training", {}).get("use_amp", False),
                amp_dtype=config.get("training", {}).get("amp_dtype", "float16"),
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
                buffer_size=100,  # FIX R8-AUD-03: backtest doesn't use replay buffer
                batch_size=bdq_config.get("batch_size", 64),
                target_update_freq=bdq_config.get("target_update_freq", 100),
                auxiliary_weight=bdq_config.get("auxiliary_weight", 1.0),
                epsilon_decay=bdq_config.get("epsilon_decay", 0.99999),
                action_dims=action_dims,
                use_amp=config.get("training", {}).get("use_amp", False),
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
        # FIX GMO1-01: Track direction for fee_threshold context
        current_direction = None

        while not done and step < 200000:
            if agent_type == "sac":
                scale_tensors = []
                for si in range(100):
                    sk = f"scale_{si}"
                    if sk not in obs:
                        break
                    scale_tensors.append(
                        torch.tensor(obs[sk], dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)
                    )
                priv = torch.tensor(obs["private"], dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)
                pred = agent.predict(scale_tensors, priv, deterministic=True)
                action = pred[0].cpu().numpy()  # (1, 1) → numpy scalar
            else:
                micro = torch.tensor(obs["micro"], dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)
                private = torch.tensor(obs["private"], dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)
                macro = torch.tensor(obs["macro"], dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)
                # FIX GMO1-01: Pass direction context for fee_threshold filtering
                ctx = {"current_direction": current_direction} if current_direction is not None else None
                pred = agent.predict(micro, private, macro, deterministic=True, context=ctx)
                if isinstance(pred, tuple):
                    action = pred[0][0]  # PPO: (actions, log_probs, values)
                else:
                    action = pred[0]    # BDQ: actions array

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            portfolio_values.append(info.get("portfolio_value", 100000))
            positions.append(info.get("position", 0))
            # FIX GMO1-01: Update direction for next step's fee_threshold context
            dir_val = info.get("direction")
            if dir_val is not None:
                current_direction = np.array([float(dir_val)])

            if step % 50000 == 0:
                logger.info(f"Backtest step {step}: Value={portfolio_values[-1]:.2f}")
            step += 1

        # Compute metrics
        pv = np.array(portfolio_values)
        pos_arr = np.array(positions)
        returns = np.diff(pv) / pv[:-1]

        total_return = (pv[-1] - pv[0]) / pv[0] if len(pv) > 0 else 0

        # Dual Sharpe: per-bar (canonical) + hourly-aggregated (research)
        # FIX K04 + AUD-S129-02: Detect bar duration for correct annualization.
        # For multi-scale (v7), the base scale IS the bar duration. For v5/v6,
        # infer from the data file path.
        mdp_version = config.get("env", {}).get("mdp_version", "v5")
        scales = config.get("features", {}).get("scales", [])
        data_file = config.get("data", {}).get("file_path", "")
        if mdp_version == "v7" and scales:
            bar_minutes = scales[0]  # First scale is the base (decision) timeframe
        elif "15min" in data_file:
            bar_minutes = 15
        elif "3min" in data_file:
            bar_minutes = 3
        elif "5min" in data_file:
            bar_minutes = 5
        else:
            bar_minutes = 1  # Default: 1-min bars
        bars_per_year = 525600 / bar_minutes
        n_per_hour = 60 // bar_minutes
        sharpe = 0.0
        sharpe_hourly = 0.0
        if np.std(returns) > 1e-9:
            raw_ratio = np.mean(returns) / np.std(returns)
            sharpe = raw_ratio * np.sqrt(bars_per_year)
            hourly_returns = np.add.reduceat(returns, np.arange(0, len(returns), n_per_hour))
            # FIX FIND-V3-02c: Drop last partial bucket (matches HPO eval BUG-05 fix)
            if len(returns) % n_per_hour != 0 and len(hourly_returns) > 1:
                hourly_returns = hourly_returns[:-1]
            if len(hourly_returns) > 1 and np.std(hourly_returns) > 1e-9:
                sharpe_hourly = (np.mean(hourly_returns) / np.std(hourly_returns)) * np.sqrt(365 * 24)

        peak = np.maximum.accumulate(pv) if len(pv) > 0 else np.array([1.0])
        max_dd = np.min(pv / np.maximum(peak, 1e-12)) - 1 if len(pv) > 0 else 0

        # Trade Stats
        # FIX BUG-P2: Count position-change legs (flips = 2 counts).
        # Base count: any change > epsilon.
        # Sign change: crossing zero implies 2 legs (Close + Open).
        # FIX K03: V6 (always-in-market) double-counts — sign_flips alone is correct.
        pos_deltas = np.abs(np.diff(pos_arr))
        base_count = np.sum(pos_deltas > 1e-6)
        sign_flips = np.sum((pos_arr[:-1] * pos_arr[1:]) < -1e-9)
        mdp_ver = config.get("env", {}).get("mdp_version", "v5")
        if mdp_ver == "v7":
            # V7: continuous positions, count all position changes past deadband
            trade_count = base_count
        elif mdp_ver == "v6":
            trade_count = sign_flips
        else:
            trade_count = base_count + sign_flips
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
        # FIX BUG-14: Zero losses with gains = excellent (cap at 10, not 0)
        profit_factor = gross_profit / gross_loss if gross_loss > 1e-12 else (10.0 if gross_profit > 1e-12 else 0.0)
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
            f"{prefix}/minute_var": pyfolio_metrics.get("minute_value_at_risk", 0.0),
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
# NOTIFICATIONS
# ============================================================================
def _notify_discord(title: str, message: str, color: int = 0x00FF00):
    """Send a Discord webhook notification. Silent no-op if URL not configured."""
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        return
    payload = json.dumps({"content": "<@792547494453575693>", "embeds": [{"title": title, "description": message, "color": color}]})
    try:
        req = Request(url, data=payload.encode(), headers={"Content-Type": "application/json", "User-Agent": "DeepScalper/1.0"})
        urlopen(req, timeout=10)
    except Exception as e:
        print(f"[notify] Discord webhook failed: {e}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="DeepScalper Pipeline (BDQ / PPO)")
    parser.add_argument("--config", type=str, default="configs/deepscalper_rtx5090.yaml")
    parser.add_argument("--agent", type=str, default="bdq", choices=["bdq", "ppo", "iqn", "sac"],
                        help="Agent type: bdq (default), ppo, iqn, or sac")
    parser.add_argument("--tags", nargs="*", default=["Pipeline"], help="WandB Tags")
    parser.add_argument("--run_name", type=str, default=None, help="Override WandB Run Name")
    parser.add_argument("--trials", type=int, default=None, help="Number of HPO trials")
    parser.add_argument("--steps", type=int, default=None, help="Training steps override")
    parser.add_argument("--version", type=str, default="V1", help="Version tag")
    parser.add_argument("--hpo_storage", type=str, default=None, help="Optuna storage URL (e.g. sqlite:///hpo.db)")
    parser.add_argument("--backtest_only", action="store_true", help="Skip HPO and training, run backtest only (requires --checkpoint)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for --backtest_only mode")
    parser.add_argument("--warm_start", type=str, default=None,
                        help="Path to checkpoint for warm-starting training (encoder weights)")
    args = parser.parse_args()

    base_config = load_config(args.config)

    # Auto-detect agent type from config if --agent was not explicitly provided.
    # This prevents KeyError when deploying with e.g. deepscalper_ppo_dev.yaml
    # but forgetting to pass --agent ppo on the command line.
    agent_type = args.agent
    if agent_type == "bdq":  # default value — check if config says otherwise
        agents_section = base_config.get("agents", {})
        if "sac" in agents_section:
            agent_type = "sac"
            logger.info(f"Agent type auto-detected from config: {agent_type}")
        elif "iqn" in agents_section:
            agent_type = "iqn"
            logger.info(f"Agent type auto-detected from config: {agent_type}")
        elif "ppo" in agents_section and "bdq" not in agents_section:
            agent_type = "ppo"
            logger.info(f"Agent type auto-detected from config: {agent_type}")
        else:
            logger.info(f"Agent type: {agent_type}")
    else:
        logger.info(f"Agent type (explicit): {agent_type}")

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

    # cuDNN auto-tuner: caches fastest kernel for fixed input shapes (our envs have constant obs dims)
    torch.backends.cudnn.benchmark = True
    logger.info("cuDNN benchmark enabled")

    try:
        # =====================================================================
        # PHASE 1: HPO
        # =====================================================================
        print("\n" + "="*60)
        print(">>> PHASE 1: HYPERPARAMETER OPTIMIZATION")
        print("="*60 + "\n")

        hpo_config = base_config.get("hpo", {})
        final_config = copy.deepcopy(base_config)

        if args.backtest_only:
            logger.info("Backtest-only mode: skipping HPO.")
        elif hpo_config.get("enabled", True):
            # Silence Optuna INFO logs (Start/Finish trial) to avoid WandB console spam
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            n_trials = args.trials or hpo_config.get("n_trials", 20)
            steps_per_trial = hpo_config.get("steps_per_trial", 50000)

            best_params = run_hpo(base_config, n_trials, steps_per_trial, device, agent_type=agent_type)

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
        if args.backtest_only:
            if not args.checkpoint:
                raise ValueError("--backtest_only requires --checkpoint <path>")
            checkpoint_path = args.checkpoint
            logger.info(f"Backtest-only mode: using checkpoint {checkpoint_path}")
        else:
            print("\n" + "="*60)
            print(">>> PHASE 2: TRAINING (Full Run with Best Params)")
            print("="*60 + "\n")

            checkpoint_path = run_training(final_config, run_name, device, agent_type=agent_type,
                                           warm_start=args.warm_start)

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

        # Notify on completion
        run_name = wandb.run.name if wandb.run else "unknown"
        _notify_discord("Pipeline Complete", f"**{run_name}** finished successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if wandb.run:
            wandb.log({"pipeline/status": "failed", "pipeline/error": str(e)})
        run_name = wandb.run.name if wandb.run else "unknown"
        _notify_discord("Pipeline FAILED", f"**{run_name}** crashed: {e}", color=0xFF0000)
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
