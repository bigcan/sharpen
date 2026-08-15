#!/usr/bin/env python
"""
DeepScalper Unified Pipeline - Single BDQ Agent
Consolidated: HPO, Training, Backtesting in one process.
"""
import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import optuna
import pandas as pd
import torch
import yaml

import wandb

# Project imports
sys.path.append(os.getcwd())
from finrl_pro_ds.agents.deepscalper.bdq_agent import DeepScalperBDQ
from finrl_pro_ds.logging import init_wandb, is_consolidated
from finrl_pro_ds.agents.ppo_scalper.ppo_agent import PPOAgent
from finrl_pro_ds.analytics.pyfolio_analyzer import PyfolioAnalyzer
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler
from finrl_pro_ds.training.deepscalper_trainer import DeepScalperTrainer
from finrl_pro_ds.training.ppo_trainer import PPOTrainer
from finrl_pro_ds.utils.naming import generate_run_name, validate_run_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeepScalperPipeline")


# ============================================================================
# CONFIG UTILS
# ============================================================================
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


from finrl_pro_ds.config_utils import deep_merge as _deep_merge  # noqa: E402


def merge_configs(base, overrides):
    """Deep merge dictionaries (legacy signature: mutates ``base`` in place).

    Thin wrapper around ``finrl_pro_ds.config_utils.deep_merge``; retained for
    backward compatibility with scripts that expect in-place mutation.
    """
    return _deep_merge(base, overrides, _mutate=True)


# _parse_frequency_to_minutes moved to finrl_pro_ds.hpo.objective (imported above)


# ============================================================================
# ENVIRONMENT FACTORY — delegated to finrl_pro_ds.hpo.env_factory
# ============================================================================
from finrl_pro_ds.hpo.env_factory import create_vector_env, make_env  # noqa: E402


# ============================================================================
# PHASE 1: HYPERPARAMETER OPTIMIZATION
# ============================================================================
# Evaluation & correlation analysis — delegated to finrl_pro_ds.hpo
from finrl_pro_ds.hpo.evaluate import analyze_hpo_correlation as _analyze_hpo_correlation  # noqa: E402
from finrl_pro_ds.hpo.objective import _parse_frequency_to_minutes  # noqa: E402
from finrl_pro_ds.hpo.objective import make_objective  # noqa: E402
from finrl_pro_ds.hpo.sampler import create_sampler  # noqa: E402




def _resolve_hpo_storage(storage):
    """Turn a Postgres URL into an RDBStorage that survives a long, DB-idle trial.

    THIS EXISTS BECAUSE IT KILLED A FULL OVERNIGHT RUN. The shared backend is Neon
    (serverless Postgres), which suspends compute and terminates idle connections. Optuna
    only touches the DB at trial start and trial end, so a 104-minute trial leaves the
    connection idle for the whole of it; the commit at the end then died with

        psycopg.errors.AdminShutdown: terminating connection due to administrator command

    which Optuna surfaces as StorageInternalError and `_run_trial` converts into
    `assert False, "Should not reach."` — taking the worker process down. All 6 workers of
    the gmgp1-spx500 A/B died this way at 2026-08-15 ~23:20, each having completed its full
    400K training steps first, so ~10 GPU-hours produced zero recorded trials.

    Two independent defences, because either alone leaves a hole:
      * `pool_pre_ping` + short `pool_recycle` — SQLAlchemy validates (and transparently
        reopens) a connection before use, so a connection killed while idle is replaced
        instead of raising.
      * `heartbeat_interval` — Optuna writes periodically DURING a trial, which both keeps
        the serverless instance from idling out and lets a genuinely dead worker's trial be
        reclaimed rather than sitting RUNNING forever (exactly the 6 orphaned RUNNING trials
        this failure left behind).

    Non-Postgres storages (the sqlite default, or an already-constructed storage object)
    pass through untouched, so no existing workstream changes behaviour.
    """
    if not isinstance(storage, str) or not storage.startswith("postgresql"):
        return storage
    return optuna.storages.RDBStorage(
        url=storage,
        engine_kwargs={
            "pool_pre_ping": True,   # revalidate before use — the core fix
            "pool_recycle": 280,     # under Neon's idle cutoff
            "connect_args": {"connect_timeout": 30},
        },
        heartbeat_interval=60,
        grace_period=600,
        failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=1),
    )


def run_hpo(base_config, n_trials, steps_per_trial, device, agent_type="bdq"):
    """Phase 1: Hyperparameter Optimization with Optuna. Supports BDQ and PPO agents."""
    logger.info(f"Starting HPO: {n_trials} trials, {steps_per_trial} steps each")
    wandb.log({"hpo/status": "started", "hpo/n_trials": n_trials})

    trial_records = []  # Per-trial data for post-HPO correlation analysis

    # Objective function now lives in finrl_pro_ds.hpo.objective (shared with distributed HPO)
    objective = make_objective(base_config, steps_per_trial, agent_type, device, trial_records)


    # Run optimization
    logger.info(f"Creating Optuna study (storage={base_config.get('hpo', {}).get('storage')})...")
    # STUDY NAME MUST BE CONFIGURABLE, and this was a latent data-corruption bug on shared
    # storage. It was hardcoded to f"hpo_{agent_type}" with load_if_exists=True, which is
    # harmless against the default per-run sqlite but silently WRONG against the shared
    # Postgres in DISTRIBUTED_HPO_DB_URL: every SAC workstream would converge on one study
    # named "hpo_sac", so two configs run against that storage would pool their trials and
    # best-trial selection would range over BOTH. That is fatal for a paired A/B such as
    # gmgp1-spx500 long-short vs long-only, where the whole design rests on the two searches
    # being independent. Falls back to the old name when unset, so every existing caller and
    # every sqlite-backed run is byte-identical.
    _hpo_cfg = base_config.get("hpo", {}) or {}
    _study_name = _hpo_cfg.get("study_name") or f"hpo_{agent_type}"
    logger.info("Optuna study_name=%s", _study_name)
    study = optuna.create_study(
        direction="maximize",
        storage=_resolve_hpo_storage(_hpo_cfg.get("storage")),
        study_name=_study_name,
        load_if_exists=True,
        sampler=create_sampler(base_config.get("hpo", {})),
        # FIX HPO-1: Disable inter-trial pruning for swing MDP.
        # HyperbandPruner was killing trials before IQN+NoisyNets could converge
        # (needs 200K+ steps for signal). All 5 trials pruned in K2/K4 runs.
        # Let all trials run to completion so we get clean data points to
        # distinguish "bad hyperparams" from "bad MDP design".
        pruner=optuna.pruners.NopPruner(),
    )
    # Narrow-mode HPO: seed Trial 0 with baseline HPs from config (guaranteed baseline)
    if base_config.get("hpo", {}).get("enqueue_baseline", False) and agent_type == "sac":
        bs = base_config["agents"]["sac"]
        be = base_config["env"]
        be_r = be["reward"]
        baseline_params = {
            "lr_actor": bs["lr_actor"],
            "lr_critic": bs["lr_critic"],
            "lr_alpha": bs["lr_alpha"],
            "tau": bs["tau"],
            "gamma": bs["gamma"],
            "initial_alpha": bs["initial_alpha"],
            "deadband_threshold": be["deadband_threshold"],
            "dsr_eta": be_r["dsr_eta"],
            "gradient_clip": bs["gradient_clip"],
        }
        # B3 fix: anchor max_leverage=1.0 in trial 0 when the search space
        # declares it. Without this, Optuna samples max_leverage even for
        # baseline anchor → no within-study leverage=1 control point.
        if "max_leverage" in (base_config.get("hpo", {}).get("search_space", {}) or {}):
            baseline_params["max_leverage"] = 1.0
        study.enqueue_trial(baseline_params)
        logger.info(f"Enqueued baseline trial with config HPs: {baseline_params}")

    study.optimize(objective, n_trials=n_trials)

    # Post-HPO: Reward-PF rank correlation analysis
    _analyze_hpo_correlation(trial_records, agent_type)

    # Log best results
    best = study.best_trial
    agent_key = agent_type if agent_type in ("ppo", "iqn", "sac") else "bdq"
    best_params = {
        "env": {"reward": {}},
        "agents": {agent_key: {}},
        "training": {},
    }

    # V4.2: Explicit routing for ALL HPO params to prevent silent mis-routing.
    if agent_type == "sac":
        reward_params = {"dsr_eta", "reward_mode"}
        agent_params = {"lr_actor", "lr_critic", "lr_alpha", "tau", "initial_alpha", "gamma", "gradient_clip", "batch_size",
                        "learning_rate", "buffer_size",
                        "cvar_alpha", "n_quantiles", "kappa"}
        # deadband + v6 hard constraints + v9 MM params route to env, not agent
        env_params = {"deadband_threshold", "stop_loss_bps", "max_holding_bars",
                      "base_spread_bps", "max_skew_bps",
                      "reward_scaling", "lambda_delta"}
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
        if key == "reward_mode":
            # V8 MM: reward_mode maps to env.reward.mode (not env.reward.reward_mode)
            best_params["env"]["reward"]["mode"] = val
        elif key in reward_params:
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
        "hpo/status": "completed",
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
                end_date=data_config.get("train_end_date"),
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
            except Exception:
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
        if "env" not in backtest_config:
            backtest_config["env"] = {}
        backtest_config["env"]["private_state_augment_prob"] = 0.0
        # FIX BUG-03: Disable hindsight reward during backtest — it uses future prices
        # which inflates evaluation metrics. Hindsight is a training-only shaping signal.
        if "reward" not in backtest_config["env"]:
            backtest_config["env"]["reward"] = {}
        backtest_config["env"]["reward"]["hindsight_weight"] = 0.0
        # FIX BUG-17: Override daily episode settings for backtest — must evaluate
        # full test period sequentially, not a single random day.
        backtest_config["env"]["episode_length"] = 0   # Full dataset
        backtest_config["env"]["random_start"] = False  # Sequential from start
        # FIX R2-AUD-03: Backtest must use final fee from fee_schedule, not initial 0.
        # Fee curriculum is a training concept; backtest evaluates at production fee level.
        # FIX BUG-12: V8/V9 (MarketMakingEnv) reads `maker_fee`, not `taker_fee`.
        fee_schedule = backtest_config.get("env", {}).get("fee_schedule")
        if fee_schedule:
            final_tier = fee_schedule[-1]
            mdp_ver_fee = backtest_config.get("env", {}).get("mdp_version", "v5")
            if mdp_ver_fee in ("v8", "v9"):
                final_fee = final_tier.get("ramp_to", final_tier.get("maker_fee", 0.0))
                backtest_config["env"]["maker_fee"] = final_fee
            else:
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
                int(action_config.get("direction_dims", 3)),
            )
        elif "discrete_dims" in action_config:
            action_dims = action_config["discrete_dims"]
        else:
            # BDQ: Legacy MultiDiscrete (price_bins, qty_bins)
            signed_qty_props = action_config.get(
                "signed_qty_proportions", [-0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5],
            )
            action_dims = (
                action_config.get("price_bins", 5),
                len(signed_qty_props),
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
                exploration_mode=iqn_cfg.get("exploration_mode", "noisy"),  # OPT-C
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
                device=device,
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
        # FIX XMATH-13: Track trade-level PnL for dual PF reporting (PF-XCHECK)
        _bt_trade_pnls = []
        _bt_prev_rpnl = 0.0

        while not done and step < 200000:
            if agent_type == "sac":
                n_scales = sum(1 for si in range(100) if f"scale_{si}" in obs)
                obs_mode = getattr(agent, '_obs_mode', 'window')

                if obs_mode == "summary_stats":
                    # v6: Concat scale summaries + private into flat (1, D)
                    parts = [obs[f"scale_{i}"] for i in range(n_scales)]
                    parts.append(obs["private"])
                    flat_np = np.concatenate(parts)
                    scale_stack = torch.as_tensor(flat_np, dtype=torch.float32).unsqueeze(0).to(
                        device, non_blocking=True,
                    )
                    priv = None
                else:
                    # Window mode: stack all scales into (1,N,W,F) — single H2D transfer
                    scale_np = np.stack([obs[f"scale_{i}"] for i in range(n_scales)], axis=0)
                    scale_stack = torch.as_tensor(scale_np, dtype=torch.float32).unsqueeze(0).to(
                        device, non_blocking=True,
                    )
                    priv = torch.as_tensor(obs["private"], dtype=torch.float32).unsqueeze(0).to(
                        device, non_blocking=True,
                    )
                # Optional LOB microstructure features (Market Making V8)
                lob = None
                if "lob" in obs:
                    lob = torch.as_tensor(obs["lob"], dtype=torch.float32).unsqueeze(0).to(
                        device, non_blocking=True,
                    )
                pred = agent.predict(scale_stack, priv, deterministic=True, lob=lob)
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
            positions.append(info.get("position", info.get("inventory", 0)))
            # FIX GMO1-01: Update direction for next step's fee_threshold context
            dir_val = info.get("direction")
            if dir_val is not None:
                current_direction = np.array([float(dir_val)])

            # FIX XMATH-13: Track trade-level PnL for dual PF reporting
            switched = info.get("switched")
            if switched is not None:
                is_switch = bool(switched)
                if is_switch:
                    rpnl = info.get("realized_pnl")
                    if rpnl is not None:
                        curr_rpnl = float(rpnl)
                        trade_pnl = curr_rpnl - _bt_prev_rpnl
                        _bt_trade_pnls.append(trade_pnl)
                        _bt_prev_rpnl = curr_rpnl

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
        # FIX BUG-12: V8 with 10s bars has scales=[1] but actual bar is 10 seconds.
        # Check bar_duration_seconds first for sub-minute bars.
        mdp_version = config.get("env", {}).get("mdp_version", "v5")
        scales = config.get("features", {}).get("scales", [])
        data_file = config.get("data", {}).get("file_path", "")
        bar_duration_seconds = config.get("env", {}).get("bar_duration_seconds")
        if bar_duration_seconds is not None:
            bar_minutes = bar_duration_seconds / 60.0
        elif mdp_version == "cmgp1":
            bar_minutes = _parse_frequency_to_minutes(config.get("data", {}).get("frequency", "1h"))
        elif mdp_version in ("v7", "v8", "v9") and scales:
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
        n_per_hour = max(1, int(60 / bar_minutes))
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
        if mdp_ver in ("v7", "v8", "v9", "cmgp1"):
            # V7/V8/V9/CMGP1: continuous positions, count all position changes past deadband
            trade_count = base_count
        elif mdp_ver == "v6":
            trade_count = sign_flips
        else:
            trade_count = base_count + sign_flips
        market_exposure = np.mean(np.abs(pos_arr) > 1e-6)

        # ── Institutional Metrics via PyfolioAnalyzer (Blueprint mandate) ──
        # FIX BUG-10: Pass bar_minutes for correct annualization (was hardcoded 1-min)
        returns_series = pd.Series(returns)
        try:
            analyzer = PyfolioAnalyzer(returns_series, bar_minutes=bar_minutes)
            pyfolio_metrics = analyzer.get_audit_metrics()
        except Exception as e:
            logger.warning(f"PyfolioAnalyzer failed, using fallback: {e}")
            pyfolio_metrics = {}

        # Manual: Profit Factor & Avg Win/Loss Ratio
        # Bar-level PF (portfolio value changes)
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
        gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 0.0
        # FIX BUG-14: Zero losses with gains = excellent (cap at 10, not 0)
        profit_factor = gross_profit / gross_loss if gross_loss > 1e-12 else (10.0 if gross_profit > 1e-12 else 0.0)

        # FIX XMATH-13: Trade-level PF for PF-XCHECK (matches HPO objective granularity)
        trade_pnls_arr = np.array(_bt_trade_pnls)
        if len(trade_pnls_arr) > 0:
            tp_pos = trade_pnls_arr[trade_pnls_arr > 0]
            tp_neg = trade_pnls_arr[trade_pnls_arr < 0]
            tp_profit = float(np.sum(tp_pos))
            tp_loss = float(np.abs(np.sum(tp_neg)))
            profit_factor_trade = tp_profit / tp_loss if tp_loss > 1e-12 else (10.0 if tp_profit > 1e-12 else 0.0)
        else:
            profit_factor_trade = profit_factor  # Fallback to bar-level

        avg_win = np.mean(wins) if len(wins) > 0 else 0.0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0.0
        avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 1e-12 else 0.0

        # FIX BUG-10: Add daily-equivalent Sharpe for industry-standard comparison
        raw_ratio = sharpe / np.sqrt(bars_per_year) if bars_per_year > 0 else 0.0
        sharpe_daily = raw_ratio * np.sqrt(252)

        metrics = {
            # ── Core metrics (manual, crypto-specific annualization) ──
            f"{prefix}/total_return": total_return,
            f"{prefix}/sharpe": sharpe,
            f"{prefix}/sharpe_daily": sharpe_daily,
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
            f"{prefix}/profit_factor_trade": profit_factor_trade,
            f"{prefix}/avg_win_loss_ratio": avg_win_loss_ratio,
            f"{prefix}/status": "completed",
        }

        wandb.log(metrics)
        logger.info(
            f"{mode} Complete. Return={total_return*100:.2f}%, Sharpe={sharpe:.2f}, "
            f"Sortino={pyfolio_metrics.get('sortino_ratio', 0):.2f}, "
            f"MaxDD={max_dd*100:.2f}%, WinRate={pyfolio_metrics.get('win_rate', 0):.1f}%",
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
def _load_telegram_creds() -> tuple[str | None, str | None]:
    """Read TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID from env, falling back to docker/live/.env."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat_id:
        return token, chat_id
    env_path = Path(__file__).resolve().parent.parent / "docker" / "live" / ".env"
    if not env_path.exists():
        return token, chat_id
    try:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            v = v.strip().strip('"').strip("'")
            if k.strip() == "TELEGRAM_BOT_TOKEN" and not token:
                token = v
            elif k.strip() == "TELEGRAM_CHAT_ID" and not chat_id:
                chat_id = v
    except Exception:
        pass
    return token, chat_id


def _notify_telegram(title: str, message: str, ok: bool = True):
    """Send a Telegram bot notification. Silent no-op if token/chat not configured."""
    token, chat_id = _load_telegram_creds()
    if not token or not chat_id:
        return
    icon = "✅" if ok else "❌"
    text = f"{icon} <b>{title}</b>\n{message}"
    payload = json.dumps({"chat_id": chat_id, "text": text, "parse_mode": "HTML"})
    try:
        req = Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=payload.encode(),
            headers={"Content-Type": "application/json"},
        )
        urlopen(req, timeout=10)
    except Exception as e:
        print(f"[notify] Telegram send failed: {e}")


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
    parser.add_argument("--hpo_storage", type=str, default=None, help="Optuna storage URL (e.g. sqlite:///hpo.db)")
    parser.add_argument("--backtest_only", action="store_true", help="Skip HPO and training, run backtest only (requires --checkpoint)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for --backtest_only mode")
    parser.add_argument("--warm_start", type=str, default=None,
                        help="Path to checkpoint for warm-starting training (encoder weights)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Global random seed for reproducibility (torch, numpy, random, env)")
    parser.add_argument("--stage", type=str, default=None,
                        help="Protocol v2 stage (data-prep|hpo|l1-multiseed|ensemble-confirm|"
                             "wf|oos|paper-deploy). When set, the config is validated via "
                             "scripts/validate_config.py before any training; a FAIL aborts.")
    parser.add_argument("--skip_validate", action="store_true",
                        help="Skip the protocol-v2 config-validation gate (NOT for CI/scheduled jobs).")
    args = parser.parse_args()

    base_config = load_config(args.config)

    # audit F10: run the protocol-v2 config validator before any training so the
    # fee-curriculum ban, XPARAM, hindsight, drift/safe-mode and health-key gates
    # actually enforce. Previously run_full_pipeline never invoked validate_config,
    # so a config could silently reproduce e.g. the frictionless-HPO artifact.
    # --stage names the stage; omitting it warns (backward compatible).
    if args.stage and not args.skip_validate:
        import subprocess
        _validator = str(Path(__file__).resolve().parent / "validate_config.py")
        _vc = subprocess.run(
            [sys.executable, _validator, "--config", args.config, "--stage", args.stage],
        )
        if _vc.returncode != 0:
            logger.error(
                "Config validation FAILED for stage '%s' (validate_config.py exit %d). "
                "Aborting before training. Use --skip_validate to override (not for CI).",
                args.stage, _vc.returncode,
            )
            sys.exit(_vc.returncode)
        logger.info("Protocol v2 config validation PASSED for stage '%s'.", args.stage)
    elif not args.stage:
        logger.warning(
            "No --stage given: Protocol v2 config validation SKIPPED. CI/scheduled "
            "jobs MUST name the stage (see CLAUDE.md Training Protocol v2).",
        )

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
        run_name = generate_run_name(args.config)

    logger.info(f"Pipeline Run: {run_name}")

    # Initialize WandB: consolidated child (attaches to parent run via
    # FINRL_WANDB_RUN_ID + FINRL_WANDB_NAMESPACE env vars) or standalone.
    init_wandb(base_config, fallback_name=run_name, tags=list(args.tags))
    if is_consolidated():
        logger.info(
            "WandB consolidated run: %s (namespace=%s)",
            wandb.run.url if wandb.run else "?",
            os.environ.get("FINRL_WANDB_NAMESPACE"),
        )
    else:
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

    # Global seed for reproducibility (multi-seed stability tests)
    if args.seed is not None:
        import random
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logger.info(f"Global seed set: {args.seed}")

    try:
        # =====================================================================
        # PHASE 1: HPO
        # =====================================================================
        print("\n" + "="*60)
        print(">>> PHASE 1: HYPERPARAMETER OPTIMIZATION")
        print("="*60 + "\n")

        hpo_config = base_config.get("hpo", {})
        # --hpo_storage was DECLARED but never applied — a dead flag, so the only way to
        # reach shared storage was to commit a credentialed URL into a tracked YAML. Wire it
        # here so the Postgres URL is supplied at launch time from .env instead. Config value
        # still wins when the flag is absent.
        if args.hpo_storage:
            hpo_config = dict(hpo_config)
            hpo_config["storage"] = args.hpo_storage
            base_config["hpo"] = hpo_config
            logger.info("HPO storage overridden from --hpo_storage (shared study backend)")
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
                norm_cutoff_date=data_config.get("val_start_date"),  # Reset stats at val boundary
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
                norm_cutoff_date=data_config.get("test_start_date"),  # Reset stats at test boundary
            )
        else:
            logger.warning("No checkpoint found, skipping backtests")

        print("\n" + "="*60)
        print(">>> PIPELINE COMPLETE")
        print("="*60 + "\n")

        # Notify on completion
        run_name = wandb.run.name if wandb.run else "unknown"
        _notify_telegram("Pipeline Complete", f"<code>{run_name}</code> finished successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if wandb.run:
            wandb.log({"pipeline/status": "failed", "pipeline/error": str(e)})
        run_name = wandb.run.name if wandb.run else "unknown"
        _notify_telegram("Pipeline FAILED", f"<code>{run_name}</code> crashed: {e}", ok=False)
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
