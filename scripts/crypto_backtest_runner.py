"""Crypto Backtest Runner — Orchestrator for Synapse Crypto 1H.

End-to-end pipeline:
1. Load/fetch data via CryptoDataPipeline
2. Compute crypto-specific features
3. Build walk-forward windows
4. Train Lean Trinity agents (SAC, A2C, PPO_GAE) per window
5. Evaluate with Softmax Arbitrator
6. Report results

Usage:
    python scripts/crypto_backtest_runner.py
    python scripts/crypto_backtest_runner.py --config configs/synapse_crypto_1h.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from finrl_pro_ds.crypto.data.crypto_loader import (  # noqa: E402
    WalkForwardCoverageValidator,
    fetch_crypto_data,
)
from finrl_pro_ds.crypto.data.crypto_array_builder import build_env_arrays  # noqa: E402
from finrl_pro_ds.crypto.features.crypto_features import compute_crypto_features  # noqa: E402
from finrl_pro_ds.crypto.envs.crypto_perp_env import CryptoPerpEnv  # noqa: E402
from finrl_pro_ds.crypto.execution.arbitrator import SoftmaxArbitrator  # noqa: E402

logger = logging.getLogger(__name__)


def load_config(config_path: str | None = None) -> dict:
    """Load experiment configuration from YAML."""
    if config_path is None:
        config_path = str(
            PROJECT_ROOT / "configs" / "synapse_crypto_1h.yaml"
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    strategy_id = config['strategy'].get('id', config['strategy']['name'])
    logger.info(f"Loaded config: {strategy_id} ({config['strategy']['name']} v{config['strategy']['version']})")
    return config


def prepare_data(config: dict) -> dict:
    """Fetch and prepare data through the full pipeline.

    Returns dict with keys: ohlcv, funding, features, walk_forward
    """
    data_cfg = config["data"]
    universe_cfg = config["universe"]

    assets = universe_cfg["assets"]
    start = data_cfg["start_date"]
    end = data_cfg.get("end_date")
    exchange = universe_cfg["data_exchange"]

    logger.info(f"Fetching data: {len(assets)} assets, {start} → {end or 'now'}")

    # Run Bronze → Silver pipeline
    data = fetch_crypto_data(
        assets=assets,
        start=start,
        end=end,
        exchange=exchange,
        cache_dir=data_cfg["cache_dir"],
    )

    ohlcv = data["ohlcv"]
    funding = data["funding"]

    # Compute crypto-specific features
    logger.info("Computing crypto-specific features...")
    crypto_feats = compute_crypto_features(
        ohlcv_df=ohlcv,
        funding_df=funding,
        correlation_window=config["features"]["tech_window"],
    )

    # Validate walk-forward coverage
    wf_cfg = config["walk_forward"]
    wf_validator = WalkForwardCoverageValidator(
        train_bars=wf_cfg["train_bars"],
        val_bars=wf_cfg["val_bars"],
        test_bars=wf_cfg["test_bars"],
        step_bars=wf_cfg["step_bars"],
        embargo_bars=wf_cfg["embargo_bars"],
        min_windows=wf_cfg.get("min_windows", 10),
    )
    wf_result = wf_validator.validate(ohlcv)

    return {
        "ohlcv": ohlcv,
        "funding": funding,
        "crypto_features": crypto_feats,
        "walk_forward": wf_result,
    }


def create_env(arrays: dict, config: dict) -> CryptoPerpEnv:
    """Create a CryptoPerpEnv from prepared arrays and config."""
    env_cfg = config["environment"]

    return CryptoPerpEnv(
        price_ary=arrays["price_ary"],
        tech_ary=arrays["tech_ary"],
        funding_rate_ary=arrays["funding_rate_ary"],
        volume_ary=arrays["volume_ary"],
        timestamps=arrays["timestamps"],
        initial_capital=env_cfg["initial_capital"],
        maker_fee_pct=env_cfg["maker_fee_pct"],
        taker_fee_pct=env_cfg["taker_fee_pct"],
        slippage_base_bps=env_cfg["slippage_base_bps"],
        slippage_impact_bps=env_cfg["slippage_impact_bps"],
        max_gross_exposure=env_cfg["max_gross_exposure"],
        max_net_short_exposure=float(env_cfg.get("max_net_short_exposure", -0.50)),
        turnover_penalty=env_cfg["turnover_penalty"],
        reward_type=env_cfg["reward_type"],
        sortino_window=env_cfg.get("sortino_window", 168),
        reward_scaling=float(env_cfg.get("reward_scaling", 1.0)),
        reward_clip_range=tuple(env_cfg.get("reward_clip_range", [-5.0, 5.0])),
        min_trade_pct=float(env_cfg.get("min_trade_pct", 0.005)),
        circuit_breaker_threshold=float(env_cfg.get("circuit_breaker_threshold", 0.1)),
        enable_trade_log=True,
        action_ema_alpha=float(env_cfg.get("action_ema_alpha", 0.0)),
        random_start=bool(env_cfg.get("random_start", False)),
        random_start_pct=float(env_cfg.get("random_start_pct", 0.1)),
        long_only=bool(env_cfg.get("long_only", False)),
    )


def run_backtest(config: dict, max_windows: int | None = None, hpo_results_dir: str | None = None) -> dict:
    """Run the full backtest pipeline.

    Returns a dict of metrics and results.
    """
    # Step 1: Prepare data
    logger.info("=" * 60)
    logger.info("PHASE 1: Data Preparation")
    logger.info("=" * 60)
    data = prepare_data(config)

    wf = data["walk_forward"]
    if not wf["passed"]:
        logger.error("Walk-forward coverage check FAILED")
        for issue in wf["issues"]:
            logger.error(f"  {issue}")
        return {"status": "FAILED", "reason": "insufficient_data"}

    logger.info(f"Walk-forward: {wf['n_windows']} windows available")

    if max_windows is not None:
        wf["window_schedule"] = wf["window_schedule"][:max_windows]
        logger.info(f"Pilot mode: limiting to first {max_windows} window(s)")

    # Step 2: Walk-forward loop
    logger.info("=" * 60)
    logger.info("PHASE 2: Walk-Forward Training & Evaluation")
    logger.info("=" * 60)

    assets = config["universe"]["assets"]
    window_results = []

    for window in wf["window_schedule"]:
        w_idx = window["window"]
        logger.info(f"\n--- Window {w_idx} ---")
        logger.info(
            f"  Train: {window['train_start']} → {window['train_end']}"
        )
        logger.info(
            f"  Val:   {window['val_start']} → {window['val_end']}"
        )
        logger.info(
            f"  Test:  {window['test_start']} → {window['test_end']}"
        )

        try:
            # Thread norm_window from features config to array builder
            norm_window = config.get("features", {}).get("norm_window", 720)

            # Build training arrays (vectorized envs created inside _train_and_evaluate)
            train_arrays = build_env_arrays(
                data["ohlcv"], data["crypto_features"], data["funding"],
                assets, window["train_start"], window["train_end"],
                norm_window=norm_window,
            )

            # Build validation environment (for model selection / early stopping)
            # Val/test envs must NOT use random_start (deterministic eval)
            val_arrays = build_env_arrays(
                data["ohlcv"], data["crypto_features"], data["funding"],
                assets, window["val_start"], window["val_end"],
                norm_window=norm_window,
            )
            eval_config = copy.deepcopy(config)
            eval_config["environment"]["random_start"] = False
            val_env = create_env(val_arrays, eval_config)

            # Build test environment
            test_arrays = build_env_arrays(
                data["ohlcv"], data["crypto_features"], data["funding"],
                assets, window["test_start"], window["test_end"],
                norm_window=norm_window,
            )
            test_env = create_env(test_arrays, eval_config)

            # F1: Pass train_arrays (not train_env) — vectorized envs built inside
            test_result = _train_and_evaluate(
                train_arrays, val_env, test_env, config,
                hpo_results_dir=hpo_results_dir, window_idx=w_idx,
            )

            window_results.append({
                "window": w_idx,
                "train_start": str(window["train_start"]),
                "test_end": str(window["test_end"]),
                **test_result,
            })

            logger.info(
                f"  Window {w_idx} result: "
                f"return={test_result['total_return']:.2%}, "
                f"sharpe={test_result['sharpe']:.3f}"
            )
            _wandb_log({
                f"window/{w_idx}/return": test_result["total_return"],
                f"window/{w_idx}/sharpe": test_result["sharpe"],
                f"window/{w_idx}/max_dd": test_result["max_drawdown"],
            })

        except Exception as e:
            logger.error(f"  Window {w_idx} FAILED: {e}")
            window_results.append({
                "window": w_idx,
                "train_start": str(window["train_start"]),
                "test_end": str(window["test_end"]),
                "total_return": float("nan"),
                "sharpe": float("nan"),
                "max_drawdown": float("nan"),
                "n_steps": 0,
                "final_value": float("nan"),
                "error": str(e),
            })

    # Step 3: Aggregate results
    logger.info("=" * 60)
    logger.info("PHASE 3: Results Summary")
    logger.info("=" * 60)

    results_df = pd.DataFrame(window_results)

    # Guard against all-failed windows producing NaN medians
    if results_df.empty or results_df["sharpe"].isna().all():
        return {
            "status": "FAILED",
            "reason": "all_windows_failed",
            "n_windows": len(window_results),
            "results": results_df.to_dict("records") if not results_df.empty else [],
        }

    summary = {
        "status": "COMPLETED",
        "n_windows": len(window_results),
        "median_sharpe": float(results_df["sharpe"].median()),
        "median_return": float(results_df["total_return"].median()),
        "median_max_dd": float(results_df["max_drawdown"].median()),
        "results": results_df.to_dict("records"),
    }

    logger.info(f"  Windows:       {summary['n_windows']}")
    logger.info(f"  Median Sharpe: {summary['median_sharpe']:.3f}")
    logger.info(f"  Median Return: {summary['median_return']:.2%}")
    logger.info(f"  Median Max DD: {summary['median_max_dd']:.2%}")

    return summary


def _init_wandb(config: dict, args) -> bool:
    """Initialize WandB run for the crypto backtest.

    Returns True if WandB is active, False if unavailable/disabled.
    """
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed — training metrics will not be logged")
        return False

    if os.environ.get("WANDB_DISABLED"):
        return False

    from finrl_pro_ds.utils.naming import generate_run_name
    run_name = args.run_name or generate_run_name(args.config or "sync_1h")
    tags = list(args.tags or []) + ["sync-1h"]

    wandb.init(
        project=config.get("wandb", {}).get("project", "FinRL-Pro-DS"),
        entity=config.get("wandb", {}).get("entity", None),
        name=run_name,
        tags=tags,
        config={
            "strategy": config.get("strategy", {}),
            "environment": config.get("environment", {}),
            "agents": config.get("agents", {}),
            "walk_forward": config.get("walk_forward", {}),
        },
        reinit=True,
    )
    return True


def _wandb_log(metrics: dict, step: int | None = None, commit: bool = True) -> None:
    """Log metrics to WandB if available. No-op if wandb not active."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step, commit=commit)
    except Exception:
        pass


def _make_vec_env(arrays: dict, config: dict, n_envs: int):
    """Create a vectorized training environment with n_envs copies.

    F1 fix: SB3 needs vectorized envs for sample diversity (especially
    on-policy A2C/PPO). Each sub-env gets the same data arrays but
    independent internal state (positions, margin, etc.).
    """
    from stable_baselines3.common.vec_env import DummyVecEnv

    def _make_fn(a=arrays, c=config):
        return create_env(a, c)

    return DummyVecEnv([_make_fn for _ in range(n_envs)])


def _make_sb3_agent(agent_type: str, env, agent_cfg: dict):
    """Instantiate an SB3 agent from config.

    Parameters
    ----------
    agent_type : str
        One of "sac", "a2c", "ppo_gae".
    env : VecEnv or CryptoPerpEnv
        Training environment (vectorized or single).
    agent_cfg : dict
        Agent-specific hyperparameters from YAML.
    """
    from stable_baselines3 import SAC, A2C, PPO

    cls_map = {"sac": SAC, "a2c": A2C, "ppo_gae": PPO}
    cls = cls_map[agent_type]

    # Map config keys to SB3 constructor kwargs
    params = {}
    for key in ("learning_rate", "gamma", "tau", "buffer_size", "batch_size",
                "n_steps", "n_epochs", "gae_lambda", "clip_range",
                "vf_coef", "ent_coef", "max_grad_norm"):
        if key in agent_cfg:
            params[key] = agent_cfg[key]

    # Forward custom network architecture if specified in agent config
    net_arch = agent_cfg.get("network_arch")
    if net_arch:
        params["policy_kwargs"] = {"net_arch": list(net_arch)}

    return cls("MlpPolicy", env, verbose=0, **params)


class _WandbStepCallback:
    """Lightweight SB3 callback that logs training progress to WandB.

    Logs every `log_interval` steps: current step, episode reward stats,
    and a heartbeat so the fleet monitor doesn't flag stalls.
    """

    def __init__(self, agent_name: str, total_timesteps: int, log_interval: int = 10_000):
        from stable_baselines3.common.callbacks import BaseCallback

        self.agent_name = agent_name
        self.total_timesteps = total_timesteps
        self.log_interval = log_interval
        self._last_log_step = 0
        self._start_time = time.time()

        class _Callback(BaseCallback):
            def __init__(cb_self, outer=self):
                super().__init__(verbose=0)
                cb_self.outer = outer

            def _on_step(cb_self) -> bool:
                step = cb_self.num_timesteps
                outer = cb_self.outer
                if step - outer._last_log_step >= outer.log_interval:
                    outer._last_log_step = step
                    elapsed = time.time() - outer._start_time
                    sps = step / elapsed if elapsed > 0 else 0
                    metrics = {
                        f"train/{outer.agent_name}/step": step,
                        f"train/{outer.agent_name}/progress": step / outer.total_timesteps,
                        f"train/{outer.agent_name}/sps": sps,
                        "_step": step,
                    }
                    # Pull episode reward from SB3's logger if available
                    if cb_self.model.ep_info_buffer and len(cb_self.model.ep_info_buffer) > 0:
                        ep_rewards = [ep["r"] for ep in cb_self.model.ep_info_buffer]
                        metrics[f"train/{outer.agent_name}/ep_reward_mean"] = float(np.mean(ep_rewards))
                        metrics[f"train/{outer.agent_name}/ep_reward_std"] = float(np.std(ep_rewards))
                    _wandb_log(metrics)
                return True

        self.callback = _Callback()


def _evaluate_agent_on_env(model, env: CryptoPerpEnv) -> tuple[dict, list[float]]:
    """Run a trained model through an env, returning metrics and per-step returns.

    Returns:
        (metrics_dict, step_returns_list)
    """
    obs, _ = env.reset()
    done = False
    portfolio_values = [env.initial_capital]
    step_returns = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
        step_returns.append(info["step_return"])
        done = terminated or truncated

    return _compute_result_metrics(portfolio_values, step_returns), step_returns


def _evaluate_arbitrator_on_env(
    arbitrator: SoftmaxArbitrator,
    agents_dict: dict,
    env: CryptoPerpEnv,
) -> dict:
    """Run the arbitrator ensemble through an env with per-agent return tracking.

    C1 fix: Each agent's predicted action is scored against actual price
    movement to produce per-agent returns, enabling the arbitrator to
    differentiate agent performance and adapt weights.

    Actions are computed once per agent per step (avoids double-predict).
    """
    obs, _ = env.reset()
    done = False
    portfolio_values = [env.initial_capital]
    step_returns = []

    while not done:
        # Get each agent's individual action (computed once)
        per_agent_actions = {}
        for name, agent in agents_dict.items():
            action_i, _ = agent.predict(obs, deterministic=True)
            per_agent_actions[name] = np.asarray(action_i, dtype=np.float64)

        # Combine using arbitrator weights (reuse already-computed actions
        # instead of calling arbitrator.predict which re-predicts internally)
        weights = arbitrator.get_weights()
        w_vec = []
        action_stack = []
        for name in agents_dict:
            if name in weights and name in per_agent_actions:
                w_vec.append(weights[name])
                action_stack.append(per_agent_actions[name])
        w_arr = np.array(w_vec, dtype=np.float64)
        w_arr /= w_arr.sum()
        combined_action = np.tensordot(w_arr, np.stack(action_stack), axes=([0], [0]))

        # Snapshot prices before step for per-agent return estimation
        pre_price = env.price_ary[env.step_idx]

        obs, reward, terminated, truncated, info = env.step(combined_action)
        portfolio_values.append(info["portfolio_value"])
        step_returns.append(info["step_return"])

        # C1 fix: Estimate each agent's return using its action against
        # actual price movement, giving the arbitrator differentiated
        # per-agent performance signals.
        post_price = env.price_ary[env.step_idx]
        price_returns = (post_price - pre_price) / (pre_price + 1e-10)

        for name in arbitrator.agent_names:
            if name in per_agent_actions:
                agent_action = per_agent_actions[name]
                # Estimated return = sum(weight_i * price_return_i)
                agent_return = float(np.dot(agent_action, price_returns))
                arbitrator.record_return(name, agent_return)
            else:
                arbitrator.record_return(name, info["step_return"])
        arbitrator.step()

        done = terminated or truncated

    return _compute_result_metrics(portfolio_values, step_returns)


def _compute_result_metrics(portfolio_values: list, step_returns: list) -> dict:
    """Compute standard metrics from portfolio values and returns."""
    pv = np.array(portfolio_values)
    returns = np.array(step_returns) if step_returns else np.diff(pv) / np.maximum(pv[:-1], 1e-10)
    total_return = pv[-1] / pv[0] - 1.0

    sharpe = 0.0
    if len(returns) > 1 and np.std(returns, ddof=1) > 1e-10:
        sharpe = float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(8760))

    peak = np.maximum.accumulate(pv)
    drawdown = 1.0 - pv / np.where(peak == 0, 1.0, peak)
    max_dd = float(drawdown.max())  # Positive fraction (e.g. 0.25 = -25%)

    return {
        "total_return": float(total_return),
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_steps": len(returns),
        "final_value": float(pv[-1]),
    }


def _train_and_evaluate(
    train_arrays: dict,
    val_env: CryptoPerpEnv,
    test_env: CryptoPerpEnv,
    config: dict,
    hpo_results_dir: str | None = None,
    window_idx: int = 0,
) -> dict:
    """Train Lean Trinity agents, select best via validation, and evaluate
    the Softmax Arbitrator ensemble on the test set.

    Steps:
    1. Train each agent (SAC, A2C, PPO) on vectorized train env
    2. Evaluate each on val_env → pick Sortino scores
    3. Initialize SoftmaxArbitrator with val performance
    4. Run arbitrator ensemble on test_env → return metrics
    """
    agents_cfg = config.get("agents", {})
    arb_cfg = config.get("arbitrator", {})
    total_timesteps = agents_cfg.get("total_timesteps", 2_000_000)
    lean_trinity = agents_cfg.get("lean_trinity", ["sac", "a2c", "ppo_gae"])
    n_envs = agents_cfg.get("n_envs", 1)

    # Load HPO best params if available for this window
    hpo_sac_params = None
    if hpo_results_dir:
        hpo_path = Path(hpo_results_dir) / f"w{window_idx}_best_sac.json"
        if hpo_path.exists():
            with open(hpo_path) as f:
                hpo_data = json.load(f)
            hpo_sac_params = hpo_data.get("agent_params", {})
            logger.info(f"    Loaded HPO params for SAC (window {window_idx}): {hpo_sac_params}")

    # --- Step 1: Train each agent on vectorized env ---
    trained_agents = {}
    # Top-level network architecture applies to all agents unless overridden
    default_net_arch = agents_cfg.get("network_arch")

    for agent_type in lean_trinity:
        agent_specific_cfg = agents_cfg.get(agent_type, {}).copy()
        # Override SAC with HPO best params if available
        if agent_type == "sac" and hpo_sac_params:
            agent_specific_cfg.update(hpo_sac_params)
        # Inherit top-level network_arch unless agent has its own
        if default_net_arch and "network_arch" not in agent_specific_cfg:
            agent_specific_cfg["network_arch"] = default_net_arch
        logger.info(f"    Training {agent_type.upper()} ({total_timesteps} timesteps, {n_envs} envs)...")

        try:
            # F1: Vectorized training — n_envs parallel copies for sample diversity
            vec_env = _make_vec_env(train_arrays, config, n_envs)
            model = _make_sb3_agent(agent_type, vec_env, agent_specific_cfg)
            # WandB callback for training progress + heartbeat
            wb_cb = _WandbStepCallback(agent_type, total_timesteps)
            model.learn(total_timesteps=total_timesteps, callback=wb_cb.callback)
            trained_agents[agent_type] = model
            vec_env.close()
            logger.info(f"    {agent_type.upper()} training complete")
            _wandb_log({f"train/{agent_type}/status": "complete"})
        except Exception as e:
            logger.error(f"    {agent_type.upper()} training failed: {e}")
            _wandb_log({f"train/{agent_type}/status": "failed", f"train/{agent_type}/error": str(e)})

    if not trained_agents:
        logger.error("    All agents failed to train — falling back to random baseline")
        return _compute_result_metrics(
            [test_env.initial_capital], []
        )

    # --- Step 2: Evaluate each agent on validation env ---
    # H2 fix: Single evaluation pass per agent (was running val env twice)
    val_scores = {}
    val_returns_per_agent = {}
    for name, model in trained_agents.items():
        from finrl_pro_ds.crypto.eval.statistics import sortino_ratio
        val_result, val_rets = _evaluate_agent_on_env(model, val_env)
        val_sortino = 0.0
        if val_rets:
            val_sortino = sortino_ratio(val_rets, periods_per_year=8760)
        val_scores[name] = val_sortino
        val_returns_per_agent[name] = val_rets
        logger.info(
            f"    {name.upper()} val: sortino={val_sortino:.3f}, "
            f"return={val_result['total_return']:.2%}"
        )
        _wandb_log({
            f"val/{name}/sortino": val_sortino,
            f"val/{name}/return": val_result["total_return"],
            f"val/{name}/sharpe": val_result["sharpe"],
            f"val/{name}/max_dd": val_result["max_drawdown"],
        })

    # --- Step 3: Initialize Softmax Arbitrator ---
    arbitrator = SoftmaxArbitrator.from_config(arb_cfg)
    for name in trained_agents:
        arbitrator.register_agent(name)

    # C2 fix: Seed arbitrator buffers with ACTUAL validation returns
    # instead of synthetic values derived from Sortino ratios.
    # Use up to the last 72 bars of each agent's validation returns.
    seed_bars = min(72, arbitrator.lookback_bars)
    for name in trained_agents:
        rets = val_returns_per_agent.get(name, [])
        # Use the tail of validation returns for seeding
        seed_rets = rets[-seed_bars:] if len(rets) >= seed_bars else rets
        for r in seed_rets:
            arbitrator.record_return(name, r)
    arbitrator.step()

    weights = arbitrator.get_weights()
    logger.info(f"    Arbitrator weights: {weights}")
    _wandb_log({f"arbitrator/weight_{k}": v for k, v in weights.items()})

    # --- Step 4: Evaluate arbitrator ensemble on test env ---
    logger.info("    Running arbitrator ensemble on test set...")
    test_result = _evaluate_arbitrator_on_env(
        arbitrator, trained_agents, test_env
    )
    test_result["agent_weights"] = weights
    test_result["n_agents_trained"] = len(trained_agents)

    _wandb_log({
        "test/return": test_result["total_return"],
        "test/sharpe": test_result["sharpe"],
        "test/max_dd": test_result["max_drawdown"],
        "test/final_value": test_result["final_value"],
    })

    return test_result


def main():
    parser = argparse.ArgumentParser(description="Synapse Crypto 1H Backtest Runner")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to experiment config YAML"
    )
    parser.add_argument(
        "--max_windows", type=int, default=None,
        help="Limit walk-forward to first N windows (pilot mode)"
    )
    parser.add_argument(
        "--agents", nargs="*", default=None,
        help="Override which agents to train (e.g. --agents sac a2c). Default: from config lean_trinity"
    )
    parser.add_argument(
        "--hpo_results_dir", type=str, default=None,
        help="Directory with HPO best params (w{N}_best_sac.json). If present, SAC uses HPO params."
    )
    # Compatibility with deploy_bare_metal.py injected args (ignored)
    parser.add_argument("--run_name", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--version", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--hpo_storage", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--tags", nargs="*", default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)

    # Override lean_trinity if --agents specified
    if args.agents:
        config.setdefault("agents", {})["lean_trinity"] = args.agents
        logger.info(f"Agent override: {args.agents}")

    _init_wandb(config, args)
    try:
        results = run_backtest(config, max_windows=args.max_windows, hpo_results_dir=args.hpo_results_dir)

        # Log final summary to WandB
        if results.get("status") == "COMPLETED":
            _wandb_log({
                "summary/median_sharpe": results["median_sharpe"],
                "summary/median_return": results["median_return"],
                "summary/median_max_dd": results["median_max_dd"],
                "summary/n_windows": results["n_windows"],
            })

        # R6 fix: Persist results to disk for downstream consumption
        out_dir = Path("results") / "crypto_backtest"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "backtest_results.json"
        # Convert numpy/pandas types for JSON serialization
        def _serialize(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, pd.Timestamp):
                return str(obj)
            return obj

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=_serialize)
        logger.info(f"Results saved to {out_path}")
    except Exception:
        logger.exception("Backtest crashed")
        raise
    finally:
        # Always finalize WandB — even on crash/OOM/signal
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass

    if results["status"] == "COMPLETED":
        logger.info("Backtest COMPLETED successfully")
    else:
        logger.error(f"Backtest FAILED: {results.get('reason', 'unknown')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
