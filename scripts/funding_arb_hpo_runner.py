"""Funding Rate Arbitrage HPO Runner — Optuna-based walk-forward pipeline.

Per-window SAC HPO with total_return objective, followed by full 2M training.
No ensemble/arbitrator — single SAC agent per window.

Usage:
    # Smoke test (2 trials, 1K steps)
    python scripts/funding_arb_hpo_runner.py \
      --config configs/funding_arb_sac_5assets_hpo.yaml \
      --max_windows 1 --n_trials 2 --hpo_timesteps 1000

    # Production HPO (50 trials, 500K steps, 5 windows)
    python scripts/funding_arb_hpo_runner.py \
      --config configs/funding_arb_sac_5assets_hpo.yaml
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
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.funding_arb_runner import (  # noqa: E402
    load_config,
    prepare_data,
    create_env,
    _make_sb3_agent,
    _evaluate_agent_on_env,
)
from finrl_pro_ds.crypto.data.crypto_array_builder import build_funding_arb_arrays  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WandB helpers
# ---------------------------------------------------------------------------

def _wandb_log(metrics: dict) -> None:
    """Log metrics to WandB with commit=True (heartbeat fix)."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, commit=True)
    except Exception:
        pass


class _WandbStepCallback:
    """SB3 callback for WandB heartbeat + SPS logging."""

    def __init__(self, agent_name: str, total_timesteps: int, log_interval: int = 5000):
        self.agent_name = agent_name
        self.total_timesteps = total_timesteps
        self.log_interval = log_interval
        self._start_time = time.time()
        self._last_log_step = 0

    def callback(self, locals_dict: dict, globals_dict: dict) -> bool:
        step = locals_dict.get("self", None)
        if step is None:
            return True
        n_calls = getattr(step, "num_timesteps", 0)
        if n_calls - self._last_log_step >= self.log_interval:
            elapsed = time.time() - self._start_time
            sps = n_calls / max(elapsed, 1e-6)
            _wandb_log({
                f"train/{self.agent_name}/step": n_calls,
                f"train/{self.agent_name}/sps": sps,
                f"train/{self.agent_name}/progress": n_calls / max(self.total_timesteps, 1),
            })
            self._last_log_step = n_calls
        return True


# ---------------------------------------------------------------------------
# Vectorized env factory
# ---------------------------------------------------------------------------

def _make_vec_funding_arb_env(
    arrays: dict, config: dict, n_envs: int, env_overrides: dict | None = None,
):
    """Create DummyVecEnv wrapping N FundingArbEnv instances."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    cfg = copy.deepcopy(config)
    if env_overrides:
        for k, v in env_overrides.items():
            cfg["environment"][k] = v

    def _make_fn(a=arrays, c=cfg):
        return create_env(a, c)

    return DummyVecEnv([_make_fn for _ in range(n_envs)])


def _create_eval_env(arrays: dict, config: dict, env_overrides: dict | None = None):
    """Create single FundingArbEnv for deterministic evaluation."""
    cfg = copy.deepcopy(config)
    if env_overrides:
        for k, v in env_overrides.items():
            cfg["environment"][k] = v
    return create_env(arrays, cfg)


# ---------------------------------------------------------------------------
# HPO search space
# ---------------------------------------------------------------------------

def define_funding_arb_search_space(trial) -> dict:
    """Define 8-dimensional funding-arb SAC search space.

    Returns dict with:
      - agent_params: SB3 SAC constructor kwargs
      - env_overrides: params to patch into env config per trial
    """
    agent_params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "buffer_size": trial.suggest_int("buffer_size", 50_000, 500_000, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "tau": trial.suggest_float("tau", 0.001, 0.02, log=True),
    }

    env_overrides = {
        "reward_scaling": trial.suggest_float("reward_scaling", 1000.0, 50000.0, log=True),
        "lambda_delta": trial.suggest_float("lambda_delta", 10.0, 500.0, log=True),
        "deadband_threshold": trial.suggest_float("deadband_threshold", 0.005, 0.05),
    }

    return {"agent_params": agent_params, "env_overrides": env_overrides}


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
    """Optuna objective: train SAC with trial params, return val total_return."""
    search = define_funding_arb_search_space(trial)
    agent_params = search["agent_params"]
    env_overrides = search["env_overrides"]

    agents_cfg = config.get("agents", {})
    n_envs = agents_cfg.get("n_envs", 4)
    network_arch = agents_cfg.get("sac", {}).get("network_arch", [256, 256])

    try:
        # Build vectorized train env with trial's env overrides
        vec_env = _make_vec_funding_arb_env(
            train_arrays, config, n_envs, env_overrides,
        )

        # Build SAC with trial's agent params + locked network
        sac_cfg = {**agent_params, "network_arch": network_arch}
        model = _make_sb3_agent("sac", vec_env, sac_cfg)

        # Train
        model.learn(total_timesteps=hpo_timesteps)

        # Evaluate on val
        val_env = _create_eval_env(val_arrays, config, env_overrides)
        val_metrics = _evaluate_agent_on_env(model, val_env)

        total_return = val_metrics["total_return"]

        # Log to WandB
        _wandb_log({
            f"hpo/trial_{trial.number}/total_return": total_return,
            f"hpo/trial_{trial.number}/sharpe": val_metrics.get("sharpe", 0.0),
            f"hpo/trial_{trial.number}/lr": agent_params["learning_rate"],
            f"hpo/trial_{trial.number}/gamma": agent_params["gamma"],
            f"hpo/trial_{trial.number}/reward_scaling": env_overrides["reward_scaling"],
        })

        # Cleanup
        vec_env.close()
        del model, vec_env
        gc.collect()

        logger.info(
            f"Trial {trial.number}: total_return={total_return:.4f}, "
            f"sharpe={val_metrics.get('sharpe', 0):.3f}"
        )
        return total_return

    except Exception as e:
        logger.error(f"Trial {trial.number} FAILED: {e}")
        gc.collect()
        return -999.0


# ---------------------------------------------------------------------------
# Per-window HPO + full training
# ---------------------------------------------------------------------------

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
    1. HPO: n_trials of SAC -> best params (maximize val total_return)
    2. Full train: SAC (best params) x total_timesteps
    3. Eval: test metrics
    """
    import optuna

    logger.info(f"=== Window {w_idx}: HPO Phase ({n_trials} trials x {hpo_timesteps} steps) ===")

    # --- HPO Phase ---
    storage_path = out_dir / f"hpo_funding_arb_w{w_idx}.db"
    study = optuna.create_study(
        study_name=f"funding_arb_w{w_idx}_sac",
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
    best_return = best.value
    logger.info(f"Window {w_idx} HPO COMPLETE: best_return={best_return:.4f} (trial {best.number})")
    logger.info(f"  Best params: {best_params}")

    # Split best params into agent vs env
    agent_keys = {"learning_rate", "buffer_size", "batch_size", "gamma", "tau"}
    env_keys = {"reward_scaling", "lambda_delta", "deadband_threshold"}
    best_agent_params = {k: v for k, v in best_params.items() if k in agent_keys}
    best_env_overrides = {k: v for k, v in best_params.items() if k in env_keys}

    # Save best params
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"w{w_idx}_best_sac.json"
    with open(best_path, "w") as f:
        json.dump({
            "window": w_idx,
            "best_trial": best.number,
            "best_return": best_return,
            "agent_params": best_agent_params,
            "env_overrides": best_env_overrides,
            "all_params": best_params,
        }, f, indent=2)
    logger.info(f"  Saved best params to {best_path}")

    _wandb_log({
        f"hpo/w{w_idx}/best_return": best_return,
        f"hpo/w{w_idx}/best_trial": best.number,
        **{f"hpo/w{w_idx}/best_{k}": v for k, v in best_params.items()},
    })

    # --- Full Training Phase ---
    logger.info(f"=== Window {w_idx}: Full Training Phase ===")

    agents_cfg = config.get("agents", {})
    total_timesteps = agents_cfg.get("total_timesteps", 2_000_000)
    n_envs = agents_cfg.get("n_envs", 4)
    network_arch = agents_cfg.get("sac", {}).get("network_arch", [256, 256])

    try:
        sac_cfg = {**best_agent_params, "network_arch": network_arch}
        vec_env = _make_vec_funding_arb_env(
            train_arrays, config, n_envs, best_env_overrides,
        )
        model = _make_sb3_agent("sac", vec_env, sac_cfg)
        wb_cb = _WandbStepCallback("sac", total_timesteps)
        model.learn(total_timesteps=total_timesteps, callback=wb_cb.callback)
        vec_env.close()
        logger.info(f"  SAC full training complete ({total_timesteps} steps)")
        _wandb_log({f"train/sac/w{w_idx}/status": "complete"})
    except Exception as e:
        logger.error(f"  SAC full training failed: {e}")
        _wandb_log({f"train/sac/w{w_idx}/status": "failed"})
        return {"window": w_idx, "status": "FAILED", "best_return": best_return, "error": str(e)}

    # --- Eval Phase ---
    logger.info(f"=== Window {w_idx}: Evaluation Phase ===")

    # Val eval
    val_env = _create_eval_env(val_arrays, config, best_env_overrides)
    val_metrics = _evaluate_agent_on_env(model, val_env)
    logger.info(
        f"  Val: return={val_metrics['total_return']:.2%}, "
        f"sharpe={val_metrics.get('sharpe', 0):.3f}, "
        f"funding/costs={val_metrics.get('funding_vs_costs_ratio', 0):.2f}"
    )
    _wandb_log({
        f"val/w{w_idx}/return": val_metrics["total_return"],
        f"val/w{w_idx}/sharpe": val_metrics.get("sharpe", 0),
    })

    # Test eval
    test_env = _create_eval_env(test_arrays, config, best_env_overrides)
    test_metrics = _evaluate_agent_on_env(model, test_env)
    logger.info(
        f"  Test: return={test_metrics['total_return']:.2%}, "
        f"sharpe={test_metrics.get('sharpe', 0):.3f}, "
        f"funding/costs={test_metrics.get('funding_vs_costs_ratio', 0):.2f}, "
        f"max_dd={test_metrics.get('max_drawdown', 0):.2%}"
    )

    _wandb_log({
        f"window/{w_idx}/test_return": test_metrics["total_return"],
        f"window/{w_idx}/test_sharpe": test_metrics.get("sharpe", 0),
        f"window/{w_idx}/test_max_dd": test_metrics.get("max_drawdown", 0),
        f"window/{w_idx}/test_funding_vs_costs": test_metrics.get("funding_vs_costs_ratio", 0),
    })

    # Save trade log
    if test_env.trade_log:
        trade_log_path = out_dir / f"w{w_idx}_trade_log.csv"
        pd.DataFrame(test_env.trade_log).to_csv(trade_log_path, index=False)
        logger.info(f"  Trade log: {trade_log_path} ({len(test_env.trade_log)} trades)")

    # Save window results
    window_result = {
        "window": w_idx,
        "status": "COMPLETED",
        "hpo_best_return": best_return,
        "hpo_best_params": best_params,
        "val_return": val_metrics["total_return"],
        "val_sharpe": val_metrics.get("sharpe", 0),
        **test_metrics,
    }

    results_path = out_dir / f"w{w_idx}_results.json"
    with open(results_path, "w") as f:
        json.dump(
            window_result, f, indent=2,
            default=lambda x: float(x) if isinstance(x, (np.floating,)) else x,
        )
    logger.info(f"  Saved results to {results_path}")

    # Cleanup
    del model
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
        logger.info(f"\n{'=' * 60}")
        logger.info(f"WINDOW {w_idx}: {window['train_start']} -> {window['test_end']}")
        logger.info(f"{'=' * 60}")

        try:
            train_arrays = build_funding_arb_arrays(
                data["spot_ohlcv"], data["perp_ohlcv"],
                data["arb_features"], data["funding"],
                assets, window["train_start"], window["train_end"],
            )
            val_arrays = build_funding_arb_arrays(
                data["spot_ohlcv"], data["perp_ohlcv"],
                data["arb_features"], data["funding"],
                assets, window["val_start"], window["val_end"],
            )
            test_arrays = build_funding_arb_arrays(
                data["spot_ohlcv"], data["perp_ohlcv"],
                data["arb_features"], data["funding"],
                assets, window["test_start"], window["test_end"],
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
    sharpes = [r.get("sharpe", 0) for r in completed]
    hpo_returns = [r["hpo_best_return"] for r in completed]

    summary = {
        "status": "COMPLETED",
        "n_windows": len(completed),
        "n_windows_total": len(schedule),
        "median_return": float(np.median(returns)),
        "median_sharpe": float(np.median(sharpes)),
        "median_hpo_return": float(np.median(hpo_returns)),
        "results": window_results,
    }

    logger.info(f"  Completed: {summary['n_windows']}/{summary['n_windows_total']} windows")
    logger.info(f"  Median return: {summary['median_return']:.2%}")
    logger.info(f"  Median Sharpe: {summary['median_sharpe']:.3f}")
    logger.info(f"  Median HPO return: {summary['median_hpo_return']:.4f}")

    _wandb_log({
        "summary/median_return": summary["median_return"],
        "summary/median_sharpe": summary["median_sharpe"],
        "summary/median_hpo_return": summary["median_hpo_return"],
        "summary/n_windows": summary["n_windows"],
    })

    # Save summary
    summary_path = out_dir / "hpo_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            summary, f, indent=2,
            default=lambda x: float(x) if isinstance(x, (np.floating,)) else x,
        )
    logger.info(f"  Summary saved to {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Funding Rate Arbitrage - HPO Runner")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--max_windows", type=int, default=None, help="Limit to first N windows")
    parser.add_argument("--n_trials", type=int, default=None, help="HPO trials per window")
    parser.add_argument("--hpo_timesteps", type=int, default=None, help="Steps per HPO trial")
    parser.add_argument("--out_dir", type=str, default="hpo_results/funding_arb", help="Output directory")
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
    hpo_cfg = config.get("hpo", {})
    n_trials = args.n_trials or hpo_cfg.get("n_trials", 50)
    hpo_timesteps = args.hpo_timesteps or hpo_cfg.get("hpo_timesteps", 500_000)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Init WandB
    try:
        import wandb
        if not os.environ.get("WANDB_DISABLED"):
            from finrl_pro_ds.utils.naming import generate_run_name
            run_name = args.run_name or generate_run_name(args.config or "funding_arb_hpo")
            tags = list(args.tags or []) + ["funding-arb", "sac", "hpo"]
            wandb_cfg = config.get("wandb", {})
            wandb.init(
                project=wandb_cfg.get("project", "FinRL-Pro-DS"),
                entity=wandb_cfg.get("entity", None),
                name=run_name,
                tags=tags,
                config={
                    "strategy": config.get("strategy", {}),
                    "hpo": {"n_trials": n_trials, "hpo_timesteps": hpo_timesteps},
                    "agents": config.get("agents", {}),
                    "environment": config.get("environment", {}),
                },
                reinit=True,
            )
    except ImportError:
        logger.warning("wandb not installed")

    logger.info(f"HPO Config: {n_trials} trials x {hpo_timesteps} steps/trial")
    logger.info(f"Output: {out_dir}")

    try:
        results = run_full_hpo_wf(config, args.max_windows, n_trials, hpo_timesteps, out_dir)
    except Exception:
        logger.exception("Funding-arb HPO walk-forward crashed")
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
        logger.info("Funding-arb HPO walk-forward COMPLETED")
    else:
        logger.error(f"Funding-arb HPO walk-forward FAILED: {results.get('reason', 'unknown')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
