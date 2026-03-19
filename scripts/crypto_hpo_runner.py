"""Crypto HPO Runner — Optuna-based hyperparameter optimization for Synapse Crypto 1H.

Per-window SAC HPO with Sortino objective, followed by full training of
SAC (best params) + A2C (defaults) with Softmax Arbitrator ensemble.

Usage:
    # Smoke test (2 trials, 1K steps)
    python scripts/crypto_hpo_runner.py --max_windows 1 --n_trials 2 --hpo_timesteps 1000

    # Production HPO (50 trials, 200K steps, window 0 only)
    python scripts/crypto_hpo_runner.py --max_windows 1

    # Full walk-forward (all windows)
    python scripts/crypto_hpo_runner.py
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from scripts.crypto_backtest_runner import (  # noqa: E402
    load_config,
    prepare_data,
    create_env,
    _make_sb3_agent,
    _evaluate_agent_on_env,
    _compute_result_metrics,
    _wandb_log,
    _WandbStepCallback,
)
from finrl_pro_ds.crypto.data.crypto_array_builder import build_env_arrays  # noqa: E402
from finrl_pro_ds.crypto.eval.statistics import sortino_ratio  # noqa: E402
from finrl_pro_ds.crypto.execution.arbitrator import SoftmaxArbitrator  # noqa: E402
from finrl_pro_ds.crypto.analytics.crypto_report import CryptoPerformanceReport  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HPO search space
# ---------------------------------------------------------------------------

def define_sac_search_space(trial, base_cfg: dict) -> dict:
    """Define the 8-dimensional SAC search space.

    Returns a dict with keys split into:
      - agent_params: SB3 SAC constructor kwargs
      - env_overrides: params to patch into env config per trial
    """
    agent_params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "buffer_size": trial.suggest_int("buffer_size", 100_000, 1_000_000, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "tau": trial.suggest_float("tau", 0.001, 0.02, log=True),
        "learning_starts": trial.suggest_int("learning_starts", 1000, 10000, log=True),
    }

    env_overrides = {
        "turnover_penalty": trial.suggest_float("turnover_penalty", 0.005, 0.05, log=True),
        "action_ema_alpha": trial.suggest_float("action_ema_alpha", 0.1, 0.5),
    }

    return {"agent_params": agent_params, "env_overrides": env_overrides}


def _create_env_with_overrides(arrays: dict, config: dict, overrides: dict):
    """Create a CryptoPerpEnv with env param overrides applied.

    Always forces random_start=False — this is used for val/test evaluation
    which must be deterministic.
    """
    cfg = copy.deepcopy(config)
    for k, v in overrides.items():
        cfg["environment"][k] = v
    cfg["environment"]["random_start"] = False
    return create_env(arrays, cfg)


def _make_vec_env_with_overrides(arrays: dict, config: dict, n_envs: int, overrides: dict):
    """Create vectorized env with per-trial env overrides."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    cfg = copy.deepcopy(config)
    for k, v in overrides.items():
        cfg["environment"][k] = v

    def _make_fn(a=arrays, c=cfg):
        return create_env(a, c)

    return DummyVecEnv([_make_fn for _ in range(n_envs)])


# ---------------------------------------------------------------------------
# Full-tracking evaluation (for backtest report)
# ---------------------------------------------------------------------------

def _evaluate_ensemble_with_tracking(
    arbitrator: SoftmaxArbitrator,
    agents_dict: dict,
    env,
) -> dict:
    """Run arbitrator ensemble and track positions, funding, costs per step.

    Returns dict with standard metrics + arrays for CryptoPerformanceReport.
    """
    obs, _ = env.reset()
    done = False
    portfolio_values = [env.initial_capital]
    step_returns = []
    positions_history = []
    funding_costs = []
    transaction_costs = []

    prev_cum_fees = 0.0
    prev_cum_funding = 0.0

    while not done:
        # Per-agent actions
        per_agent_actions = {}
        for name, agent in agents_dict.items():
            action_i, _ = agent.predict(obs, deterministic=True)
            per_agent_actions[name] = np.asarray(action_i, dtype=np.float64)

        # Weighted combination
        weights = arbitrator.get_weights()
        w_vec, action_stack = [], []
        for name in agents_dict:
            if name in weights and name in per_agent_actions:
                w_vec.append(weights[name])
                action_stack.append(per_agent_actions[name])
        w_arr = np.array(w_vec, dtype=np.float64)
        w_arr /= w_arr.sum()
        combined_action = np.tensordot(w_arr, np.stack(action_stack), axes=([0], [0]))

        # Snapshot for per-agent return
        pre_price = env.price_ary[env.step_idx]

        obs, reward, terminated, truncated, info = env.step(combined_action)

        portfolio_values.append(info["portfolio_value"])
        step_returns.append(info["step_return"])
        positions_history.append(env.positions.copy())

        # Incremental funding/transaction costs this step
        cur_fees = info["cumulative_fees"]
        cur_funding = info["cumulative_funding"]
        funding_costs.append(cur_funding - prev_cum_funding)
        transaction_costs.append(cur_fees - prev_cum_fees)
        prev_cum_fees = cur_fees
        prev_cum_funding = cur_funding

        # Arbitrator update
        post_price = env.price_ary[env.step_idx]
        price_returns = (post_price - pre_price) / (pre_price + 1e-10)
        for name in arbitrator.agent_names:
            if name in per_agent_actions:
                agent_return = float(np.dot(per_agent_actions[name], price_returns))
                arbitrator.record_return(name, agent_return)
            else:
                arbitrator.record_return(name, info["step_return"])
        arbitrator.step()

        done = terminated or truncated

    metrics = _compute_result_metrics(portfolio_values, step_returns)
    metrics["_tracking"] = {
        "portfolio_values": np.array(portfolio_values),
        "step_returns": np.array(step_returns),
        "positions_history": np.array(positions_history),
        "funding_costs": np.array(funding_costs),
        "transaction_costs": np.array(transaction_costs),
        "timestamps": env.timestamps[: len(step_returns)],
        "trade_log": env.trade_log,
    }
    return metrics


# ---------------------------------------------------------------------------
# HPO objective
# ---------------------------------------------------------------------------

def hpo_objective(
    trial,
    train_arrays: dict,
    val_arrays: dict,
    config: dict,
    hpo_timesteps: int,
) -> float:
    """Optuna objective: train SAC with trial params, return val Sortino."""
    search = define_sac_search_space(trial, config)
    agent_params = search["agent_params"]
    env_overrides = search["env_overrides"]

    agents_cfg = config.get("agents", {})
    n_envs = agents_cfg.get("n_envs", 20)

    # Locked params (BUG-01)
    network_arch = agents_cfg.get("network_arch", [256, 256])

    try:
        # Build vectorized train env with trial's env overrides
        vec_env = _make_vec_env_with_overrides(train_arrays, config, n_envs, env_overrides)

        # Build SAC with trial's agent params + locked network
        sac_cfg = {**agent_params, "network_arch": network_arch}
        model = _make_sb3_agent("sac", vec_env, sac_cfg)

        # Train
        model.learn(total_timesteps=hpo_timesteps)

        # Evaluate on val
        val_env = _create_env_with_overrides(val_arrays, config, env_overrides)
        _, val_rets = _evaluate_agent_on_env(model, val_env)

        if not val_rets or len(val_rets) < 50:
            logger.warning(f"Trial {trial.number}: too few val steps ({len(val_rets) if val_rets else 0})")
            return -999.0

        val_sort = sortino_ratio(val_rets, periods_per_year=8760)

        # Log to WandB
        _wandb_log({
            f"hpo/trial_{trial.number}/sortino": val_sort,
            f"hpo/trial_{trial.number}/lr": agent_params["learning_rate"],
            f"hpo/trial_{trial.number}/gamma": agent_params["gamma"],
            f"hpo/trial_{trial.number}/turnover": env_overrides["turnover_penalty"],
        })

        # Cleanup
        vec_env.close()
        del model, vec_env
        gc.collect()

        logger.info(f"Trial {trial.number}: val_sortino={val_sort:.4f}")
        return val_sort

    except Exception as e:
        logger.error(f"Trial {trial.number} FAILED: {e}")
        gc.collect()
        return -999.0


# ---------------------------------------------------------------------------
# Per-window HPO + full training
# ---------------------------------------------------------------------------

def _generate_backtest_report(
    w_idx: int,
    tracking: dict,
    config: dict,
    out_dir: Path,
) -> None:
    """Generate CryptoPerformanceReport tearsheet from tracked eval data."""
    try:
        assets = config["universe"]["assets"]

        # Build benchmark
        n_steps = len(tracking["step_returns"])
        benchmark_returns = {
            "CashBenchmark": np.zeros(n_steps),
        }

        report = CryptoPerformanceReport(
            returns=tracking["step_returns"],
            benchmark_returns=benchmark_returns,
            portfolio_values=tracking["portfolio_values"],
            positions_history=tracking["positions_history"],
            funding_costs=tracking["funding_costs"],
            transaction_costs=tracking["transaction_costs"],
            asset_names=assets,
            timestamps=tracking["timestamps"],
            annualization_factor=8760,
        )

        tearsheet_dir = str(out_dir / f"w{w_idx}_tearsheet")
        report.generate_tearsheet(tearsheet_dir)

        metrics = report.compute_metrics()
        metrics_path = out_dir / f"w{w_idx}_report_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(
                metrics, f, indent=2,
                default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x,
            )

        logger.info(f"  Tearsheet: {tearsheet_dir}")
        logger.info(
            f"  Report metrics: Sharpe={metrics.get('sharpe_ratio', 0):.3f}, "
            f"Sortino={metrics.get('sortino_ratio', 0):.3f}, "
            f"MaxDD={metrics.get('max_drawdown', 0):.2%}"
        )
    except Exception as e:
        logger.warning(f"  Backtest report generation failed (non-fatal): {e}")


def run_hpo_for_window(
    w_idx: int,
    train_arrays: dict,
    val_arrays: dict,
    test_arrays: dict,
    config: dict,
    n_trials: int,
    hpo_timesteps: int,
    out_dir: Path,
) -> dict:
    """Run HPO for a single walk-forward window, then full train + eval.

    Steps:
    1. HPO: n_trials of SAC → best params (maximize val Sortino)
    2. Full train: SAC (best params) + A2C (defaults) × total_timesteps
    3. Eval: val Sortino → arbitrator → test ensemble → metrics
    """
    import optuna

    logger.info(f"=== Window {w_idx}: HPO Phase ({n_trials} trials × {hpo_timesteps} steps) ===")

    # --- HPO Phase ---
    storage_path = out_dir / f"hpo_sync1h_w{w_idx}.db"
    study = optuna.create_study(
        study_name=f"sync1h_w{w_idx}_sac",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=42, n_startup_trials=10, multivariate=True,
        ),
        pruner=optuna.pruners.NopPruner(),
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
    )

    def _objective(trial):
        return hpo_objective(trial, train_arrays, val_arrays, config, hpo_timesteps)

    # Skip already-completed trials (resumability)
    remaining = n_trials - len(study.trials)
    if remaining > 0:
        study.optimize(
            _objective,
            n_trials=remaining,
            timeout=600 * remaining,  # 10 min per trial cap
            gc_after_trial=True,
        )

    # Extract best params
    best = study.best_trial
    best_params = best.params
    best_sortino = best.value
    logger.info(f"Window {w_idx} HPO COMPLETE: best_sortino={best_sortino:.4f} (trial {best.number})")
    logger.info(f"  Best params: {best_params}")

    # Split best params into agent vs env
    agent_keys = {"learning_rate", "buffer_size", "batch_size", "gamma", "tau", "learning_starts"}
    env_keys = {"turnover_penalty", "action_ema_alpha"}
    best_agent_params = {k: v for k, v in best_params.items() if k in agent_keys}
    best_env_overrides = {k: v for k, v in best_params.items() if k in env_keys}

    # Save best params
    out_dir.mkdir(parents=True, exist_ok=True)
    best_sac_path = out_dir / f"w{w_idx}_best_sac.json"
    with open(best_sac_path, "w") as f:
        json.dump({
            "window": w_idx,
            "best_trial": best.number,
            "best_sortino": best_sortino,
            "agent_params": best_agent_params,
            "env_overrides": best_env_overrides,
            "all_params": best_params,
        }, f, indent=2)
    logger.info(f"  Saved best params to {best_sac_path}")

    _wandb_log({
        f"hpo/w{w_idx}/best_sortino": best_sortino,
        f"hpo/w{w_idx}/best_trial": best.number,
        **{f"hpo/w{w_idx}/best_{k}": v for k, v in best_params.items()},
    })

    # --- Full Training Phase ---
    logger.info(f"=== Window {w_idx}: Full Training Phase ===")

    agents_cfg = config.get("agents", {})
    total_timesteps = agents_cfg.get("total_timesteps", 2_000_000)
    n_envs = agents_cfg.get("n_envs", 20)
    network_arch = agents_cfg.get("network_arch", [256, 256])
    lean_trinity = agents_cfg.get("lean_trinity", ["sac", "a2c"])

    trained_agents = {}

    for agent_type in lean_trinity:
        logger.info(f"  Training {agent_type.upper()} ({total_timesteps} steps, {n_envs} envs)...")

        try:
            if agent_type == "sac":
                # Use HPO best params + locked network
                agent_cfg = {**best_agent_params, "network_arch": network_arch}
                vec_env = _make_vec_env_with_overrides(
                    train_arrays, config, n_envs, best_env_overrides,
                )
            else:
                # A2C/others: use defaults from config
                agent_cfg = agents_cfg.get(agent_type, {}).copy()
                if network_arch and "network_arch" not in agent_cfg:
                    agent_cfg["network_arch"] = network_arch
                vec_env = _make_vec_env_with_overrides(
                    train_arrays, config, n_envs, best_env_overrides,
                )

            model = _make_sb3_agent(agent_type, vec_env, agent_cfg)
            wb_cb = _WandbStepCallback(agent_type, total_timesteps)
            model.learn(total_timesteps=total_timesteps, callback=wb_cb.callback)
            trained_agents[agent_type] = model
            vec_env.close()
            logger.info(f"  {agent_type.upper()} training complete")
            _wandb_log({f"train/{agent_type}/status": "complete"})
        except Exception as e:
            logger.error(f"  {agent_type.upper()} training failed: {e}")
            _wandb_log({f"train/{agent_type}/status": "failed"})

    if not trained_agents:
        logger.error(f"  Window {w_idx}: all agents failed")
        return {"window": w_idx, "status": "FAILED", "best_sortino": best_sortino}

    # --- Eval Phase ---
    logger.info(f"=== Window {w_idx}: Evaluation Phase ===")

    # Val eval for arbitrator seeding
    val_env = _create_env_with_overrides(val_arrays, config, best_env_overrides)
    val_scores = {}
    val_returns_per_agent = {}
    for name, model in trained_agents.items():
        val_result, val_rets = _evaluate_agent_on_env(model, val_env)
        val_sort = sortino_ratio(val_rets, periods_per_year=8760) if val_rets else 0.0
        val_scores[name] = val_sort
        val_returns_per_agent[name] = val_rets
        logger.info(f"    {name.upper()} val: sortino={val_sort:.3f}, return={val_result['total_return']:.2%}")
        _wandb_log({
            f"val/{name}/sortino": val_sort,
            f"val/{name}/return": val_result["total_return"],
            f"val/{name}/sharpe": val_result["sharpe"],
        })

    # Arbitrator
    arb_cfg = config.get("arbitrator", {})
    arbitrator = SoftmaxArbitrator.from_config(arb_cfg)
    for name in trained_agents:
        arbitrator.register_agent(name)

    seed_bars = min(72, arbitrator.lookback_bars)
    for name in trained_agents:
        rets = val_returns_per_agent.get(name, [])
        for r in rets[-seed_bars:]:
            arbitrator.record_return(name, r)
    arbitrator.step()

    weights = arbitrator.get_weights()
    logger.info(f"    Arbitrator weights: {weights}")
    _wandb_log({f"arbitrator/weight_{k}": v for k, v in weights.items()})

    # Test eval with full tracking
    test_env = _create_env_with_overrides(test_arrays, config, best_env_overrides)
    test_result = _evaluate_ensemble_with_tracking(arbitrator, trained_agents, test_env)
    tracking = test_result.pop("_tracking")
    test_result["agent_weights"] = weights
    test_result["n_agents_trained"] = len(trained_agents)

    _wandb_log({
        f"window/{w_idx}/test_return": test_result["total_return"],
        f"window/{w_idx}/test_sharpe": test_result["sharpe"],
        f"window/{w_idx}/test_max_dd": test_result["max_drawdown"],
    })

    logger.info(
        f"  Window {w_idx} RESULT: return={test_result['total_return']:.2%}, "
        f"sharpe={test_result['sharpe']:.3f}, max_dd={test_result['max_drawdown']:.2%}"
    )

    # --- Backtest Report ---
    _generate_backtest_report(
        w_idx, tracking, config, out_dir,
    )

    # Save trade log CSV
    if tracking["trade_log"]:
        trade_log_path = out_dir / f"w{w_idx}_trade_log.csv"
        pd.DataFrame(tracking["trade_log"]).to_csv(trade_log_path, index=False)
        logger.info(f"  Trade log: {trade_log_path} ({len(tracking['trade_log'])} trades)")

    # Save window results
    window_result = {
        "window": w_idx,
        "status": "COMPLETED",
        "hpo_best_sortino": best_sortino,
        "hpo_best_params": best_params,
        "val_scores": {k: float(v) for k, v in val_scores.items()},
        "agent_weights": {k: float(v) for k, v in weights.items()},
        **test_result,
    }

    results_path = out_dir / f"w{w_idx}_results.json"
    with open(results_path, "w") as f:
        json.dump(window_result, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)
    logger.info(f"  Saved results to {results_path}")

    # Cleanup
    del trained_agents
    gc.collect()

    return window_result


# ---------------------------------------------------------------------------
# Walk-forward orchestrator
# ---------------------------------------------------------------------------

def run_full_hpo_wf(
    config: dict,
    max_windows: int | None,
    n_trials: int,
    hpo_timesteps: int,
    out_dir: Path,
) -> dict:
    """Run HPO + full training across walk-forward windows."""
    # Prepare data
    logger.info("=" * 60)
    logger.info("PHASE 1: Data Preparation")
    logger.info("=" * 60)
    data = prepare_data(config)

    wf = data["walk_forward"]
    if not wf["passed"]:
        logger.error("Walk-forward coverage check FAILED")
        return {"status": "FAILED", "reason": "insufficient_data"}

    schedule = wf["window_schedule"]
    if max_windows is not None:
        schedule = schedule[:max_windows]
    logger.info(f"Walk-forward: {len(schedule)} windows (of {wf['n_windows']} available)")

    assets = config["universe"]["assets"]
    window_results = []

    for window in schedule:
        w_idx = window["window"]
        logger.info(f"\n{'='*60}")
        logger.info(f"WINDOW {w_idx}: {window['train_start']} → {window['test_end']}")
        logger.info(f"{'='*60}")

        try:
            norm_window = config.get("features", {}).get("norm_window", 720)
            train_arrays = build_env_arrays(
                data["ohlcv"], data["crypto_features"], data["funding"],
                assets, window["train_start"], window["train_end"],
                norm_window=norm_window,
            )
            val_arrays = build_env_arrays(
                data["ohlcv"], data["crypto_features"], data["funding"],
                assets, window["val_start"], window["val_end"],
                norm_window=norm_window,
            )
            test_arrays = build_env_arrays(
                data["ohlcv"], data["crypto_features"], data["funding"],
                assets, window["test_start"], window["test_end"],
                norm_window=norm_window,
            )

            result = run_hpo_for_window(
                w_idx, train_arrays, val_arrays, test_arrays,
                config, n_trials, hpo_timesteps, out_dir,
            )
            window_results.append(result)

        except Exception as e:
            logger.error(f"Window {w_idx} FAILED: {e}", exc_info=True)
            window_results.append({
                "window": w_idx,
                "status": "FAILED",
                "error": str(e),
            })

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    completed = [r for r in window_results if r.get("status") == "COMPLETED"]
    if not completed:
        return {"status": "FAILED", "reason": "all_windows_failed", "results": window_results}

    returns = [r["total_return"] for r in completed]
    sharpes = [r["sharpe"] for r in completed]
    sortinos = [r["hpo_best_sortino"] for r in completed]

    summary = {
        "status": "COMPLETED",
        "n_windows": len(completed),
        "n_windows_total": len(schedule),
        "median_return": float(np.median(returns)),
        "median_sharpe": float(np.median(sharpes)),
        "median_hpo_sortino": float(np.median(sortinos)),
        "results": window_results,
    }

    logger.info(f"  Completed: {summary['n_windows']}/{summary['n_windows_total']} windows")
    logger.info(f"  Median return: {summary['median_return']:.2%}")
    logger.info(f"  Median Sharpe: {summary['median_sharpe']:.3f}")
    logger.info(f"  Median HPO Sortino: {summary['median_hpo_sortino']:.3f}")

    _wandb_log({
        "summary/median_return": summary["median_return"],
        "summary/median_sharpe": summary["median_sharpe"],
        "summary/median_hpo_sortino": summary["median_hpo_sortino"],
        "summary/n_windows": summary["n_windows"],
    })

    # Save summary
    summary_path = out_dir / "hpo_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)
    logger.info(f"  Summary saved to {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Synapse Crypto 1H - HPO Runner")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--max_windows", type=int, default=None, help="Limit to first N windows")
    parser.add_argument("--n_trials", type=int, default=None, help="HPO trials per window (default: from config)")
    parser.add_argument("--hpo_timesteps", type=int, default=None, help="Steps per HPO trial (default: from config)")
    parser.add_argument("--out_dir", type=str, default="hpo_results", help="Output directory")
    # WandB / deploy compat
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument("--version", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--hpo_storage", default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Suppress Optuna's verbose trial logging
    logging.getLogger("optuna").setLevel(logging.WARNING)

    config = load_config(args.config)

    # Resolve HPO params: CLI > config > defaults
    wf_cfg = config.get("walk_forward", {})
    hpo_cfg = config.get("hpo", {})
    n_trials = args.n_trials or hpo_cfg.get("n_trials", wf_cfg.get("n_hpo_trials", 50))
    hpo_timesteps = args.hpo_timesteps or hpo_cfg.get("hpo_timesteps", wf_cfg.get("hpo_timesteps", 200_000))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Init WandB
    try:
        import wandb
        if not os.environ.get("WANDB_DISABLED"):
            from finrl_pro_ds.utils.naming import generate_run_name
            run_name = args.run_name or generate_run_name(args.config or "sync_1h_hpo")
            tags = list(args.tags or []) + ["sync-1h", "hpo", "s166"]
            wandb.init(
                project=config.get("wandb", {}).get("project", "FinRL-Pro-DS"),
                entity=config.get("wandb", {}).get("entity", None),
                name=run_name,
                tags=tags,
                config={
                    "strategy": config.get("strategy", {}),
                    "hpo": {"n_trials": n_trials, "hpo_timesteps": hpo_timesteps},
                    "agents": config.get("agents", {}),
                },
                reinit=True,
            )
    except ImportError:
        logger.warning("wandb not installed")

    logger.info(f"HPO Config: {n_trials} trials × {hpo_timesteps} steps/trial")
    logger.info(f"Output: {out_dir}")

    results = run_full_hpo_wf(config, args.max_windows, n_trials, hpo_timesteps, out_dir)

    # Finish WandB
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass

    if results["status"] == "COMPLETED":
        logger.info("HPO walk-forward COMPLETED")
    else:
        logger.error(f"HPO walk-forward FAILED: {results.get('reason', 'unknown')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
