"""CMGP1 HPO Runner — SAC-only walk-forward for multi-asset crypto.

Simplified from crypto_hpo_runner.py: drops A2C/PPO, ensemble arbitrator,
warm-start, and SB3 dependency. Uses custom SACTrainer directly.

Usage:
    # Smoke test (2 trials, 1K steps, 1 window)
    python scripts/crypto_sac_hpo_runner.py --config configs/sync_2h_sac_crypto.yaml \
        --max_windows 1 --n_trials 2 --hpo_timesteps 1000

    # Production HPO (30 trials, 500K steps)
    python scripts/crypto_sac_hpo_runner.py --config configs/sync_2h_sac_crypto.yaml \
        --max_windows 1

    # Full walk-forward
    python scripts/crypto_sac_hpo_runner.py --config configs/sync_2h_sac_crypto.yaml
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import wandb  # noqa: E402
from finrl_pro_ds.crypto.data.multiscale_crypto_handler import (  # noqa: E402
    MultiScaleCryptoHandler,
)
from finrl_pro_ds.crypto.envs.crypto_perp_swing_env import CryptoPerpSwingEnv  # noqa: E402
from finrl_pro_ds.training.sac_trainer import SACTrainer  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load YAML config."""
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(config: dict) -> dict:
    """Fetch and prepare crypto data.

    Returns dict with keys: ohlcv, funding, walk_forward_windows
    """
    from finrl_pro_ds.crypto.data.crypto_loader import CryptoLoader, CryptoDataPipeline

    data_cfg = config["data"]
    universe_cfg = config["universe"]

    assets = universe_cfg["assets"]
    exchange = universe_cfg["data_exchange"]
    cache_dir = data_cfg.get("cache_dir", "./data/crypto_cache")

    logger.info(f"Preparing data: {len(assets)} assets from {exchange}")

    # Check for cached data first
    cache_path = Path(cache_dir)
    ohlcv_cache = cache_path / "ohlcv_merged.parquet"
    funding_cache = cache_path / "funding_merged.parquet"

    if ohlcv_cache.exists() and funding_cache.exists():
        logger.info("Loading from cache...")
        ohlcv = pd.read_parquet(ohlcv_cache)
        funding = pd.read_parquet(funding_cache)
    else:
        raise FileNotFoundError(
            f"Data cache not found at {cache_dir}. "
            "Run the crypto data pipeline first to populate the cache.",
        )

    # Compute walk-forward windows
    wf_cfg = config["walk_forward"]
    windows = _compute_walk_forward_windows(ohlcv, wf_cfg)

    return {
        "ohlcv": ohlcv,
        "funding": funding,
        "walk_forward_windows": windows,
    }


def _compute_walk_forward_windows(
    ohlcv: pd.DataFrame,
    wf_cfg: dict,
) -> list[dict]:
    """Compute walk-forward window date ranges."""
    timestamps = pd.to_datetime(ohlcv["timestamp"]).sort_values().unique()
    n_bars = len(timestamps)

    train_bars = wf_cfg["train_bars"]
    val_bars = wf_cfg["val_bars"]
    test_bars = wf_cfg["test_bars"]
    step_bars = wf_cfg["step_bars"]
    embargo_bars = wf_cfg.get("embargo_bars", 0)

    total_window = train_bars + embargo_bars + val_bars + test_bars
    windows = []
    offset = 0

    while offset + total_window <= n_bars:
        train_start = timestamps[offset]
        train_end = timestamps[offset + train_bars - 1]
        val_start = timestamps[offset + train_bars + embargo_bars]
        val_end = timestamps[offset + train_bars + embargo_bars + val_bars - 1]
        test_start = timestamps[offset + train_bars + embargo_bars + val_bars]
        test_end_idx = min(offset + total_window - 1, n_bars - 1)
        test_end = timestamps[test_end_idx]

        windows.append({
            "window_id": len(windows),
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "test_start": test_start,
            "test_end": test_end,
            "norm_cutoff": val_start,  # LEAK-1: normalize up to val boundary
        })
        offset += step_bars

    logger.info(f"Walk-forward: {len(windows)} windows ({train_bars} train, {val_bars} val, {test_bars} test)")
    return windows


# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------

def create_env(
    ohlcv: pd.DataFrame,
    funding: pd.DataFrame,
    config: dict,
    start_date: str,
    end_date: str,
    norm_cutoff_date: str | None = None,
    random_start: bool = True,
) -> CryptoPerpSwingEnv:
    """Create a CryptoPerpSwingEnv with MultiScaleCryptoHandler."""
    assets = config["universe"]["assets"]
    feature_config = config["features"]

    handler = MultiScaleCryptoHandler(
        ohlcv_df=ohlcv,
        funding_df=funding,
        assets=assets,
        feature_config=feature_config,
        start_date=start_date,
        end_date=end_date,
        norm_cutoff_date=norm_cutoff_date,
    )

    env_cfg = copy.deepcopy(config["env"])
    env_cfg["random_start"] = random_start
    env_cfg["n_assets"] = len(assets)

    return CryptoPerpSwingEnv(config=env_cfg, data_handler=handler)


def create_vec_env(
    ohlcv: pd.DataFrame,
    funding: pd.DataFrame,
    config: dict,
    start_date: str,
    end_date: str,
    norm_cutoff_date: str | None = None,
    n_envs: int = 10,
) -> object:
    """Create SyncVectorEnv for training."""
    import gymnasium as gym

    def _make_fn():
        return create_env(
            ohlcv, funding, config, start_date, end_date,
            norm_cutoff_date=norm_cutoff_date,
            random_start=True,
        )

    return gym.vector.SyncVectorEnv([_make_fn for _ in range(n_envs)])


# ---------------------------------------------------------------------------
# HPO
# ---------------------------------------------------------------------------

def define_search_space(trial: optuna.Trial, config: dict) -> dict:
    """Define 8-dimensional SAC search space from config."""
    search_cfg = config["hpo"].get("search_space", {})
    overrides = {}

    for name, spec in search_cfg.items():
        t = spec["type"]
        if t == "float":
            overrides[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        elif t == "int":
            overrides[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif t == "categorical":
            overrides[name] = trial.suggest_categorical(name, spec["choices"])

    return overrides


def apply_overrides(config: dict, overrides: dict) -> dict:
    """Apply HPO overrides to config."""
    cfg = copy.deepcopy(config)

    # Agent params
    for key in ["lr_actor", "lr_critic", "gamma", "tau", "batch_size"]:
        if key in overrides:
            cfg["agents"]["sac"][key] = overrides[key]

    # Env params
    for key in ["deadband_threshold", "stop_loss_bps", "max_holding_bars"]:
        if key in overrides:
            cfg["env"][key] = overrides[key]

    return cfg


def run_hpo_trial(
    trial: optuna.Trial,
    ohlcv: pd.DataFrame,
    funding: pd.DataFrame,
    config: dict,
    window: dict,
) -> float:
    """Run a single HPO trial. Returns negative profit factor (minimize)."""
    overrides = define_search_space(trial, config)
    trial_cfg = apply_overrides(config, overrides)

    # Override training steps for HPO
    trial_cfg["training"]["total_timesteps"] = config["hpo"]["steps_per_trial"]

    # Create training env
    n_envs = trial_cfg["training"].get("num_envs", 10)
    vec_env = create_vec_env(
        ohlcv, funding, trial_cfg,
        start_date=str(window["train_start"]),
        end_date=str(window["train_end"]),
        norm_cutoff_date=str(window["norm_cutoff"]),
        n_envs=n_envs,
    )

    try:
        trainer = SACTrainer(
            env=vec_env,
            config=trial_cfg,
            device="cuda",
            hpo_mode=True,
            run_name=f"cmgp1_hpo_w{window['window_id']}_t{trial.number}",
        )
        trainer.train(optuna_trial=trial)
    except Exception as e:
        logger.warning(f"Trial {trial.number} failed: {e}")
        return float("inf")
    finally:
        vec_env.close()

    # Evaluate on validation set
    val_env = create_env(
        ohlcv, funding, trial_cfg,
        start_date=str(window["val_start"]),
        end_date=str(window["val_end"]),
        norm_cutoff_date=str(window["norm_cutoff"]),
        random_start=False,
    )

    try:
        metrics = evaluate_agent(trainer.agent, val_env)
    finally:
        val_env.close()

    # BUG-01: HPO objective = profit_factor
    pf = metrics.get("profit_factor", 0.0)
    trial.set_user_attr("val_sharpe", metrics.get("sharpe_ratio", 0.0))
    trial.set_user_attr("val_return", metrics.get("total_return", 0.0))
    trial.set_user_attr("val_pf", pf)

    logger.info(
        f"Trial {trial.number}: PF={pf:.3f}, Sharpe={metrics.get('sharpe_ratio', 0.0):.3f}, "
        f"Return={metrics.get('total_return', 0.0):.2%}",
    )

    # Clean up GPU memory
    del trainer
    gc.collect()

    # Optuna minimizes, so negate PF (higher PF = better)
    return -pf


def evaluate_agent(agent, env: CryptoPerpSwingEnv) -> dict:
    """Run deterministic evaluation and compute metrics."""
    obs, _ = env.reset()
    done = False
    portfolio_values = [env.initial_balance]

    while not done:
        # Predict deterministic action
        with __import__("torch").no_grad():
            action = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
        done = terminated or truncated

    pv = np.array(portfolio_values)
    returns = np.diff(pv) / np.maximum(pv[:-1], 1e-9)

    # Compute metrics
    total_return = (pv[-1] / pv[0]) - 1.0
    sharpe = _sharpe(returns)
    sortino = _sortino(returns)
    max_dd = _max_drawdown(pv)
    pf = _profit_factor(returns)

    return {
        "total_return": float(total_return),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(max_dd),
        "profit_factor": float(pf),
        "portfolio_values": pv.tolist(),
    }


def _sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2 or np.std(returns) < 1e-10:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(8760))  # annualized for 1H bars


def _sortino(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    dd = np.sqrt(np.mean(downside ** 2)) if len(downside) > 0 else 1e-10
    return float(np.mean(returns) / max(dd, 1e-10) * np.sqrt(8760))


def _max_drawdown(pv: np.ndarray) -> float:
    peak = np.maximum.accumulate(pv)
    dd = (peak - pv) / np.maximum(peak, 1e-9)
    return float(np.max(dd))


def _profit_factor(returns: np.ndarray) -> float:
    gains = float(np.sum(returns[returns > 0]))
    losses = float(np.abs(np.sum(returns[returns < 0])))
    if losses < 1e-10:
        return 10.0 if gains > 0 else 1.0
    return gains / losses


# ---------------------------------------------------------------------------
# Full training
# ---------------------------------------------------------------------------

def train_full(
    ohlcv: pd.DataFrame,
    funding: pd.DataFrame,
    config: dict,
    window: dict,
    best_params: dict,
    run_name: str,
) -> tuple:
    """Full training with best HPO params. Returns (trainer, metrics)."""
    trial_cfg = apply_overrides(config, best_params)
    n_envs = trial_cfg["training"].get("num_envs", 10)

    vec_env = create_vec_env(
        ohlcv, funding, trial_cfg,
        start_date=str(window["train_start"]),
        end_date=str(window["train_end"]),
        norm_cutoff_date=str(window["norm_cutoff"]),
        n_envs=n_envs,
    )

    trainer = SACTrainer(
        env=vec_env,
        config=trial_cfg,
        device="cuda",
        hpo_mode=False,
        run_name=run_name,
    )

    checkpoint_path = trainer.train()
    vec_env.close()

    # Evaluate on val and test
    results = {}
    for split in ["val", "test"]:
        eval_env = create_env(
            ohlcv, funding, trial_cfg,
            start_date=str(window[f"{split}_start"]),
            end_date=str(window[f"{split}_end"]),
            norm_cutoff_date=str(window["norm_cutoff"]),
            random_start=False,
        )
        results[split] = evaluate_agent(trainer.agent, eval_env)
        eval_env.close()

    return trainer, results, checkpoint_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CMGP1 SAC Walk-Forward HPO")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--max_windows", type=int, default=0, help="Max windows (0=all)")
    parser.add_argument("--n_trials", type=int, default=0, help="Override HPO trials (0=use config)")
    parser.add_argument("--hpo_timesteps", type=int, default=0, help="Override HPO steps (0=use config)")
    parser.add_argument("--skip_hpo", action="store_true", help="Skip HPO, use config defaults")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)

    # Override from CLI
    if args.n_trials > 0:
        config["hpo"]["n_trials"] = args.n_trials
    if args.hpo_timesteps > 0:
        config["hpo"]["steps_per_trial"] = args.hpo_timesteps

    # Prepare data
    data = prepare_data(config)
    ohlcv = data["ohlcv"]
    funding = data["funding"]
    windows = data["walk_forward_windows"]

    if args.max_windows > 0:
        windows = windows[:args.max_windows]

    logger.info(f"Running {len(windows)} walk-forward windows")

    # WandB init
    wandb_cfg = config.get("wandb", {})
    wandb.init(
        project=wandb_cfg.get("project", "FinRL-Pro-DS"),
        name=f"cmgp1-wf-{len(windows)}w",
        config=config,
        tags=wandb_cfg.get("tags", []),
    )

    # Walk-forward loop
    all_results = []

    for window in windows:
        wid = window["window_id"]
        logger.info(f"\n{'='*60}")
        logger.info(f"Window {wid}: train={window['train_start']} → test={window['test_end']}")
        logger.info(f"{'='*60}")

        # Phase 1: HPO
        if not args.skip_hpo:
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(multivariate=True, seed=42 + wid),
                pruner=optuna.pruners.NopPruner(),
            )
            study.optimize(
                lambda trial: run_hpo_trial(trial, ohlcv, funding, config, window),
                n_trials=config["hpo"]["n_trials"],
            )
            best_params = study.best_trial.params
            logger.info(f"Window {wid} HPO best: PF={-study.best_value:.3f}, params={best_params}")

            wandb.log({
                f"hpo/w{wid}_best_pf": -study.best_value,
                f"hpo/w{wid}_best_params": json.dumps(best_params),
            })
        else:
            best_params = {}
            logger.info(f"Window {wid}: Skipping HPO, using config defaults")

        # Phase 2: Full training
        run_name = f"cmgp1_w{wid}"
        trainer, results, ckpt_path = train_full(
            ohlcv, funding, config, window, best_params, run_name,
        )

        # Log results
        for split in ["val", "test"]:
            m = results[split]
            prefix = f"w{wid}/{split}"
            wandb.log({
                f"{prefix}/total_return": m["total_return"],
                f"{prefix}/sharpe_ratio": m["sharpe_ratio"],
                f"{prefix}/sortino_ratio": m["sortino_ratio"],
                f"{prefix}/max_drawdown": m["max_drawdown"],
                f"{prefix}/profit_factor": m["profit_factor"],
            })
            logger.info(
                f"Window {wid} {split}: Return={m['total_return']:.2%}, "
                f"PF={m['profit_factor']:.3f}, Sharpe={m['sharpe_ratio']:.3f}, "
                f"MaxDD={m['max_drawdown']:.2%}",
            )

        all_results.append({
            "window_id": wid,
            "best_params": best_params,
            "val": results["val"],
            "test": results["test"],
            "checkpoint": ckpt_path,
        })

        # Clean up GPU memory between windows
        del trainer
        gc.collect()

    # Summary
    test_returns = [r["test"]["total_return"] for r in all_results]
    test_pfs = [r["test"]["profit_factor"] for r in all_results]
    profitable = sum(1 for r in test_returns if r > 0)

    logger.info(f"\n{'='*60}")
    logger.info(f"WALK-FORWARD SUMMARY ({len(all_results)} windows)")
    logger.info(f"  Median return: {np.median(test_returns):.2%}")
    logger.info(f"  Median PF: {np.median(test_pfs):.3f}")
    logger.info(f"  Profitable windows: {profitable}/{len(all_results)} ({profitable/max(len(all_results),1)*100:.0f}%)")
    logger.info(f"{'='*60}")

    wandb.log({
        "summary/median_return": float(np.median(test_returns)),
        "summary/median_pf": float(np.median(test_pfs)),
        "summary/win_rate": profitable / max(len(all_results), 1),
        "summary/n_windows": len(all_results),
    })

    # Save results
    results_path = f"results/cmgp1_wf_{len(all_results)}w.json"
    os.makedirs("results", exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
