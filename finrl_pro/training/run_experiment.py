"""Run a FinRL Pro experiment from a YAML config."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import pandas as pd
import yaml

from finrl_pro.configs.fingerprint_store import FingerprintStore
from finrl_pro.data.cache import feature_cache_key
from finrl_pro.data.loader import DataLoader
from finrl_pro.data.loader_pro import ProFeatureAssembler
from finrl_pro.envs.factory import make_pro_env
from finrl_pro.mlops.logger import MLOpsLogger
from finrl_pro.mlops.risk_controls import RiskControlPolicy
from finrl_pro.mlops.risk_profiles import load_risk_profile
from finrl_pro.training.trainer import Trainer
from finrl_pro.envs.wrappers import TurnoverPenaltyWrapper, ActionSmoothingWrapper, SoftmaxAllocationWrapper

# Import agents
from finrl_pro.agents.ppo import PPOAgent
from finrl_pro.agents.sac import SACAgent
from finrl_pro.agents.td3 import TD3Agent
from finrl_pro.agents.ddpg import DDPGAgent
from finrl_pro.utils.naming import generate_experiment_name


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _stable_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _seed_from_inputs(seed: int, dataset_hash: str, config_path: Path) -> int:
    payload = f"{seed}:{dataset_hash}:{config_path}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF


@dataclass(slots=True)
class SimulatedRunArtifacts:
    returns: list[float]
    equity: list[float]
    drawdowns: list[float]
    positions: list[float]
    trades: list[float]
    turnover: list[float]
    transaction_costs: list[float]
    metrics: dict[str, float]


def _compute_equity_and_metrics(returns: list[float]) -> tuple[list[float], list[float], float, float, float]:
    equity: list[float] = []
    drawdowns: list[float] = []
    cum = 1.0
    peak = 1.0
    for r in returns:
        cum *= (1.0 + r)
        equity.append(cum)
        peak = max(peak, cum)
        drawdowns.append((cum / peak) - 1.0)

    mean_daily = statistics.fmean(returns)
    std_daily = statistics.pstdev(returns) or 1e-9
    sharpe = (mean_daily / std_daily) * math.sqrt(252)
    vol_realized = std_daily * math.sqrt(252)
    max_dd = abs(min(drawdowns)) if drawdowns else 0.0
    return equity, drawdowns, sharpe, vol_realized, max_dd


def _simulate_training_outputs(
    *,
    cfg_path: Path,
    training_cfg: dict[str, Any],
) -> SimulatedRunArtifacts:
    """Simulate a training run by generating deterministic returns + metrics."""
    module_versions = dict(training_cfg.get("module_versions", {}) or {})
    agent = str(module_versions.get("agent", "PPO")).upper()
    action_space = str(module_versions.get("action.space", "CONTINUOUS")).upper()

    base_metrics = dict(training_cfg.get("metrics", {}) or {})
    target_sharpe = _stable_float(base_metrics.get("sharpe_ratio"), 1.0)
    target_vol = _stable_float(base_metrics.get("volatility"), 0.20)

    # Adjust heuristic targets by agent family to create diversity.
    if agent == "TD3":
        target_sharpe += 0.05
        target_vol = max(target_vol - 0.02, 0.10)
    elif agent == "SAC":
        target_sharpe -= 0.05
        target_vol = min(target_vol + 0.03, 0.35)

    dataset_hash = str(training_cfg.get("dataset_hash", ""))
    base_seed = int(training_cfg.get("seed", 0))
    rng_seed = _seed_from_inputs(base_seed, dataset_hash, cfg_path)
    rng = random.Random(rng_seed)

    sigma_daily = target_vol / math.sqrt(252)
    sigma_daily = max(sigma_daily, 1e-4)
    mu_daily = target_sharpe * sigma_daily / math.sqrt(252)

    horizon = int(training_cfg.get("steps", 756) or 756)
    horizon = max(horizon, 128)

    returns: list[float] = []
    positions: list[float] = []
    trades: list[float] = []
    turnover: list[float] = []
    txn_costs: list[float] = []
    position = 0.0
    cost_cfg = dict(training_cfg.get("costs", {}) or {})
    per_turnover_bps = float(cost_cfg.get("per_turnover_bps", 2.0))
    cost_rate = per_turnover_bps / 10_000.0
    action_vol = 0.15 if "CONTINUOUS" in action_space else 0.05

    drift = 0.0
    for _ in range(horizon):
        noise = rng.gauss(mu_daily + drift, sigma_daily)
        # Inject occasional shocks to create realistic drawdowns.
        if rng.random() < 0.015:
            shock = abs(rng.gauss(0.0, 3.0 * sigma_daily))
            noise -= shock
            drift = -shock * 0.25
        else:
            drift *= 0.90
        noise = max(min(noise, 0.25), -0.60)
        returns.append(noise)
        target_pos = max(min(position + rng.gauss(0.0, action_vol), 1.0), -1.0)
        trade = target_pos - position
        position = target_pos
        positions.append(position)
        trades.append(trade)
        abs_trade = abs(trade)
        turnover.append(abs_trade)
        fee = abs_trade * cost_rate
        txn_costs.append(fee)
        returns[-1] -= fee

    target_dd = max(_stable_float(base_metrics.get("max_drawdown"), 0.18), 0.05)
    for _ in range(3):
        equity, drawdowns, sharpe, vol_realized, max_dd = _compute_equity_and_metrics(returns)
        if max_dd <= (target_dd + 0.01):
            break
        scale = max(min(target_dd / max_dd, 1.0), 0.25)
        returns = [max(min(r * scale, 0.25), -0.60) for r in returns]
    else:
        equity, drawdowns, sharpe, vol_realized, max_dd = _compute_equity_and_metrics(returns)

    capital_at_risk = min(max_dd * 0.9, 0.09)
    leverage = min(1.0 + (0.2 if "CONTINUOUS" in action_space else 0.05), 1.5)

    total_turnover = float(sum(turnover))
    avg_turnover = total_turnover / max(1, len(turnover))
    annualized_turnover = avg_turnover * (252.0 / max(1, horizon))
    total_cost = float(sum(txn_costs))

    metrics = {
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "volatility": float(vol_realized),
        "capital_at_risk": float(capital_at_risk),
        "leverage": float(leverage),
        "avg_turnover": float(avg_turnover),
        "total_turnover": float(total_turnover),
        "annualized_turnover": float(annualized_turnover),
        "transaction_costs": float(total_cost),
        "transaction_costs_bps": float(total_cost * 10_000.0),
    }
    return SimulatedRunArtifacts(
        returns=returns,
        equity=equity,
        drawdowns=drawdowns,
        positions=positions,
        trades=trades,
        turnover=turnover,
        transaction_costs=txn_costs,
        metrics=metrics,
    )

import numpy as np
import torch # needed for agent

def _run_real_training(
    *,
    training_cfg: dict[str, Any],
    logger: MLOpsLogger,
    tickers: list[str] | None = None,
) -> SimulatedRunArtifacts:
    """Execute a real training run with an RL agent."""
    dataset_hash = str(training_cfg.get("dataset_hash", ""))
    if not dataset_hash:
        raise ValueError("dataset_hash must be provided for real training.")

    # 1. Load Data
    df = DataLoader.resolve_dataset(dataset_hash)
    
    # Filter by Date Range
    start_date = training_cfg.get("start_date")
    end_date = training_cfg.get("end_date")
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']) # Ensure datetime
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
            
    # Filter by Tickers
    if tickers and 'tic' in df.columns:
        df = df[df['tic'].isin(tickers)]
        logger.log_event("finrl_pro.training.tickers_filtered", context={"count": len(tickers)})
            
    if df.empty:
        raise ValueError(f"No data found for range {start_date} to {end_date}")

    logger.log_event("finrl_pro.training.real_data_loaded", context={"dataset_hash": dataset_hash, "rows": len(df), "start": str(start_date), "end": str(end_date)})

    # 2. Assemble Features
    feat_cfg = training_cfg.get("features", {})
    assembler = ProFeatureAssembler()
    asm = assembler.assemble_from_df(
        df=df,
        features_cfg=feat_cfg,
        dataset_hash=dataset_hash, # Pass the original dataset hash for cache key
    )
    logger.log_event("finrl_pro.training.features_assembled", context={"feature_set_id": asm.feature_set_id})


    # 3. Create Environment
    # 3. Create Environment
    env_cfg = training_cfg.get("environment", {})
    env = make_pro_env(asm=asm, **env_cfg)
    
    # Apply Turnover Penalty Wrapper
    turnover_penalty = float(training_cfg.get("turnover_penalty", 0.0))
    if turnover_penalty > 0.0:
        env = TurnoverPenaltyWrapper(env, penalty_coef=turnover_penalty)
        logger.log_event("finrl_pro.training.wrapper_applied", context={"wrapper": "TurnoverPenaltyWrapper", "coef": turnover_penalty})

    # Apply Softmax Allocation Wrapper
    module_versions = training_cfg.get("module_versions", {})
    if module_versions.get("action.space") == "ALLOCATION_VECTOR_LONG_ONLY":
        env = SoftmaxAllocationWrapper(env)
        logger.log_event("finrl_pro.training.wrapper_applied", context={"wrapper": "SoftmaxAllocationWrapper"})

    # Apply Action Smoothing Wrapper
    action_smoothing = float(training_cfg.get("action_smoothing", 0.0))
    if action_smoothing > 0.0:
        env = ActionSmoothingWrapper(env, smooth_factor=action_smoothing)
        logger.log_event("finrl_pro.training.wrapper_applied", context={"wrapper": "ActionSmoothingWrapper", "smooth_factor": action_smoothing})

    logger.log_event("finrl_pro.training.environment_created", context={"env_name": "ProStockEnv"})


    # 4. Instantiate Agent
    agent_name = training_cfg.get("agent.name", "PPOAgent") # Default to PPO
    agent_params = training_cfg.get("agent.params", {})

    # Configure Action Adapter for Allocation
    if module_versions.get("action.space") == "ALLOCATION_VECTOR_LONG_ONLY":
        agent_params["action_adapter"] = "identity"
        logger.log_event("finrl_pro.training.agent_configured", context={"action_adapter": "identity"})

    # Dynamically select agent
    current_agent = None
    
    # Resolve dimensions (unwrap to access custom attributes or use spaces)
    if hasattr(env, "state_dim"):
        s_dim = env.state_dim
    elif hasattr(env.unwrapped, "state_dim"):
        s_dim = env.unwrapped.state_dim
    else:
        s_dim = env.observation_space.shape[0]
        
    if hasattr(env, "action_dim"):
        a_dim = env.action_dim
    elif hasattr(env.unwrapped, "action_dim"):
        a_dim = env.unwrapped.action_dim
    else:
        a_dim = env.action_space.shape[0]

    if agent_name == "PPOAgent":
        current_agent = PPOAgent(state_dim=s_dim, action_dim=a_dim, **agent_params)
    elif agent_name == "SACAgent":
        current_agent = SACAgent(state_dim=s_dim, action_dim=a_dim, **agent_params)
    elif agent_name == "TD3Agent":
        current_agent = TD3Agent(state_dim=s_dim, action_dim=a_dim, **agent_params)
    elif agent_name == "DDPGAgent":
        current_agent = DDPGAgent(state_dim=s_dim, action_dim=a_dim, **agent_params)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")
    logger.log_event("finrl_pro.training.agent_instantiated", context={"agent_name": agent_name})


    # 5. Training Loop
    total_timesteps = training_cfg.get("total_timesteps", 100000)
    stress_tests = training_cfg.get("stress_tests", {})
    
    # Execution Gap Stress Test Config
    exec_gap_enabled = stress_tests.get("execution_gap", {}).get("enable", False)
    exec_gap_ticks = stress_tests.get("execution_gap", {}).get("ticks", 0)
    
    # Input Noise Stress Test Config
    input_noise_enabled = stress_tests.get("input_noise", {}).get("enable", False)
    input_noise_std = stress_tests.get("input_noise", {}).get("std", 0.0)

    obs, info = env.reset() # Initial reset
    
    rewards_history = []
    # Buffer to simulate execution delay (gap)
    action_buffer = []
    
    # Metrics collection
    positions_history = []
    turnover_history = []
    costs_history = []
    asset_values = [] # Track portfolio value
    
    # Initial asset value
    if hasattr(env, "unwrapped"):
        asset_values.append(env.unwrapped.total_asset)
    else:
        asset_values.append(1e6) # Fallback

    for t in range(total_timesteps):
        
        # Apply Input Noise Stress Test
        if input_noise_enabled:
            noise = np.random.normal(0, input_noise_std, size=obs.shape)
            obs = obs + noise

        action_output = current_agent.select_action(obs)
        if isinstance(action_output, tuple):
            action, log_prob = action_output
        else:
            action = action_output
            log_prob = 0.0 # Dummy for Off-Policy agents

        # Apply Execution Gap Stress Test
        if exec_gap_enabled:
            action_buffer.append(action)
            if len(action_buffer) > exec_gap_ticks:
                executed_action = action_buffer.pop(0)
            else:
                # During warm-up of the delay buffer, execute a neutral action (e.g., 0)
                # Or simply repeat the first action. Let's assume zero vector (hold)
                executed_action = np.zeros_like(action)
        else:
            executed_action = action

        next_obs, reward, terminated, truncated, info = env.step(executed_action)
        
        rewards_history.append(reward) # Collect rewards
        
        # Collect metrics
        # Try to get from info or env
        if hasattr(env, "unwrapped"):
             positions_history.append(str(env.unwrapped.stocks))
             asset_values.append(env.unwrapped.total_asset)
        else:
             positions_history.append("N/A")
             # Estimate asset value from reward if unwrapped not available (imprecise)
             # prev = asset_values[-1]
             # asset_values.append(prev + reward / 2**-13) # Assuming default scaling
             asset_values.append(asset_values[-1]) # Fallback
             
        turnover_history.append(info.get("turnover_penalty", 0.0)) # Proxy if real turnover not available
        costs_history.append(0.0) # Placeholder as Env doesn't expose cost explicitly in info yet

        # Store transition (using the *intended* action for training, or executed?)
        # Standard RL assumes transition (s, a, r, s') where 'a' caused 'r'.  
        # If there's a gap, the reward is response to the delayed action.
        # For robustness *testing*, we usually freeze weights (no training), just evaluation.
        # But if training, we store what actually happened.
        
        current_agent.store_transition(obs, executed_action, reward, terminated, next_state=next_obs, log_prob=log_prob)
        
        # Off-Policy Update (SAC/TD3/DDPG)
        is_off_policy = agent_name in ["SACAgent", "TD3Agent", "DDPGAgent"]
        
        if is_off_policy:
            metrics_from_update = current_agent.update()
            # Log sparingly or aggregate? For now relying on agent internal checks
        
        obs = next_obs
        
        if terminated or truncated:
            if not is_off_policy:
                # Update agent after episode (On-Policy PPO)
                metrics_from_update = current_agent.update() # Returns dict of losses
                logger.log_event("finrl_pro.training.agent_updated", context=metrics_from_update)
                current_agent.reset_buffer() # Clear buffer for next episode
            elif 'metrics_from_update' in locals() and metrics_from_update:
                 # Log last metrics for off-policy
                 logger.log_event("finrl_pro.training.agent_updated", context=metrics_from_update)

            obs, info = env.reset()
            action_buffer = [] # Reset gap buffer
            if hasattr(env, "unwrapped"):
                 asset_values.append(env.unwrapped.total_asset) # New episode start

    # Calculate Returns from Asset Values
    # returns[t] = (asset[t+1] - asset[t]) / asset[t]
    # We captured asset_values at t=0 and after each step.
    # asset_values has length T+1 (or more if resets happened)
    # We need to align with rewards_history which has length T.
    # We'll compute simple returns for the contiguous segments.
    
    calculated_returns = []
    # This simple diff ignores reset jumps, assuming single episode or we handle it.
    # For simplicity in this MVP, we just diff the list.
    for i in range(1, len(asset_values)):
        if i > len(rewards_history): break
        prev = asset_values[i-1]
        curr = asset_values[i]
        if prev > 0:
            ret = (curr - prev) / prev
        else:
            ret = 0.0
        calculated_returns.append(ret)

    # 6. Collect and Compute Metrics
    if calculated_returns:
        # Assuming rewards_history are daily returns
        equity, drawdowns, sharpe, vol_realized, max_dd = _compute_equity_and_metrics(calculated_returns)
        final_metrics = {
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "volatility": float(vol_realized),
            # Add other metrics as needed
        }
    else:
        final_metrics = {"sharpe_ratio": 0.0, "max_drawdown": 0.0, "volatility": 0.0} # Fallback

    # Save agent model for MLflow logging
    try:
        current_agent.save("temp_model.pth")
    except Exception as e:
        print(f"Warning: Failed to save temp model: {e}")

    return SimulatedRunArtifacts(
        returns=calculated_returns,
        equity=equity if calculated_returns else [],
        drawdowns=drawdowns if calculated_returns else [],
        positions=positions_history, 
        trades=[0.0] * len(positions_history), # Placeholder
        turnover=turnover_history, 
        transaction_costs=costs_history, 
        metrics=final_metrics,
    )


def _persist_artifacts(fingerprint_id: str, sim: SimulatedRunArtifacts) -> list[str]:
    """Write simulated artifacts to disk for downstream evaluation."""
    out_dir = Path("reports") / fingerprint_id
    out_dir.mkdir(parents=True, exist_ok=True)

    returns_path = out_dir / "returns.csv"
    with returns_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "return"])
        for idx, value in enumerate(sim.returns):
            writer.writerow([idx, value])

    equity_path = out_dir / "equity_curve.csv"
    with equity_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "equity"])
        for idx, value in enumerate(sim.equity):
            writer.writerow([idx, value])

    dd_path = out_dir / "drawdown.csv"
    with dd_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "drawdown"])
        for idx, value in enumerate(sim.drawdowns):
            writer.writerow([idx, value])

    exec_path = out_dir / "execution.csv"
    with exec_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "position", "trade", "turnover", "transaction_cost"])
        for idx, (pos, trade, tov, cost) in enumerate(
            zip(sim.positions, sim.trades, sim.turnover, sim.transaction_costs)
        ):
            writer.writerow([idx, pos, trade, tov, cost])

    return [str(path) for path in (returns_path, equity_path, dd_path, exec_path)]


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="finrl_pro.training.run_experiment",
        description="Execute a training run using an experiment YAML.",
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg_path = Path(args.config)
    cfg = _load_yaml(cfg_path)

    manifest = Path(cfg.get("fingerprint_manifest", "finrl_pro/configs/fingerprints.yaml"))
    store = FingerprintStore(manifest_path=manifest)
    store.load()

    risk_file = Path(cfg["risk_profile_file"]) if cfg.get("risk_profile_file") else None
    risk_id = cfg.get("risk_profile_id", "default")
    risk_policy = None
    if risk_file and risk_file.exists():
        profile = load_risk_profile(risk_file, risk_id)
        risk_policy = RiskControlPolicy(profile)

    logger = MLOpsLogger()
    trainer = Trainer(
        fingerprint_store=store,
        logger=logger,
        risk_policy=risk_policy,
    )

    tcfg = cfg["training"]
    tickers = cfg.get("data", {}).get("tickers")
    
    real_training_enabled = bool(tcfg.get("real_training", False))

    # Generate Experiment Name
    exp_meta = cfg.get("experiment", {})
    run_name = generate_experiment_name(
        category=exp_meta.get("category", "VALI"),
        system=exp_meta.get("system", "FinRLPro"),
        description=exp_meta.get("description", "experiment"),
        experiment_id=exp_meta.get("id")
    )

    import mlflow
    # Start MLflow run explicitly to capture artifacts from real training
    with mlflow.start_run(run_name=run_name) as active_run:
        if real_training_enabled:
            sim = _run_real_training(training_cfg=tcfg, logger=logger, tickers=tickers)
            # Log model artifact if generated
            if Path("temp_model.pth").exists():
                mlflow.log_artifact("temp_model.pth", artifact_path="model")
                # Clean up temp file
                try:
                    # Path("temp_model.pth").unlink()
                    pass
                except:
                    pass
        else:
            sim = _simulate_training_outputs(cfg_path=cfg_path, training_cfg=tcfg)

        # Optional dataset resolution check for snapshot-backed datasets
        ds_hash = str(tcfg.get("dataset_hash", ""))
        # This block is for logging and feature assembly, handled in _run_real_training
        # if real_training_enabled.
        if ds_hash.startswith("snapshot://") and not real_training_enabled:
            try:
                # from finrl_pro.data.loader import DataLoader  # noqa: WPS433 (already imported)
                df = DataLoader.resolve_dataset(ds_hash)
                logger.log_event(
                    "finrl_pro.training.dataset_resolved",
                    context={"dataset_hash": ds_hash, "rows": int(df.shape[0])},
                )
            except Exception as e:  # noqa: BLE001
                logger.log_event(
                    "finrl_pro.training.dataset_resolve_error",
                    context={"dataset_hash": ds_hash, "error": str(e)},
                )
                raise
        
        # Compose module versions with feature cache key for reproducibility
        mv = dict(tcfg.get("module_versions", {}))
        feat_cfg = cfg.get("features", {})
        # Skip feature assembly if real training is enabled, as it's handled in _run_real_training
        if not real_training_enabled:
            try:
                fkey = feature_cache_key(str(tcfg.get("dataset_hash", "")), feat_cfg)
                mv["features.cache_key"] = fkey
            except Exception:
                # Keep going if features block is malformed
                pass

            # Optional: assemble features from DB and log feature_set_id (snapshot datasets only)
            use_pro_env = bool(tcfg.get("use_pro_env", False))
            mv["features.env_mode"] = "B" if use_pro_env else "A"
            if ds_hash.startswith("snapshot://") and feat_cfg:
                try:
                    # from finrl_pro.data.loader_pro import ProFeatureAssembler  # noqa: WPS433 (already imported)
                    snapshot_id = ds_hash.split("//", 1)[1]
                    assembler = ProFeatureAssembler()
                    asm = assembler.assemble_from_snapshot(snapshot_id=snapshot_id, features_cfg=feat_cfg)
                    mv["features.feature_set_id"] = asm.feature_set_id
                except Exception:  # noqa: BLE001
                    # Non-fatal: continue without feature_set_id if assembly not available
                    pass

        fingerprint = trainer.run(
            config_path=str(tcfg["config_path"]),
            dataset_hash=str(tcfg["dataset_hash"]),
            seed=int(tcfg.get("seed", 0)),
            module_versions=mv,
            metrics=sim.metrics,
            artifact_uris=list(tcfg.get("artifact_uris", [])),
            baseline_reference=str(tcfg["baseline_reference"]),
            sandbox_enabled=bool(cfg.get("sandbox_enabled", False)),
        )

        # Register model if configured
        if tcfg.get("register_model", False):
            model_name = str(tcfg.get("model_name", "finrl_pro_model"))
            trainer.register_model(fingerprint.mlflow_run_id, model_name)

    artifact_paths = _persist_artifacts(fingerprint.fingerprint_id, sim)
    fingerprint.artifact_uris = artifact_paths
    store.register(fingerprint)
    store.save()

    print(json.dumps({
        "fingerprint_id": fingerprint.fingerprint_id,
        "manifest": str(manifest),
        "config": str(cfg_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
