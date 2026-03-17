"""Funding Rate Arbitrage Runner — Walk-Forward Backtest Pipeline.

End-to-end pipeline for delta-neutral spot-perp funding rate arbitrage:
1. Load/fetch spot + perp data via CryptoDataPipeline
2. Compute funding-arb-specific features
3. Build walk-forward windows
4. Train PPO agents per window
5. Evaluate with funding-arb-specific metrics
6. Report results

Usage:
    python scripts/funding_arb_runner.py --config configs/funding_arb_delta_neutral.yaml
    python scripts/funding_arb_runner.py --config configs/funding_arb_delta_neutral.yaml --dry_run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from finrl_pro_ds.crypto.data.crypto_loader import (
    WalkForwardCoverageValidator,
    fetch_crypto_data,
    fetch_spot_data,
)
from finrl_pro_ds.crypto.data.crypto_array_builder import build_funding_arb_arrays
from finrl_pro_ds.crypto.features.funding_arb_features import compute_funding_arb_features
from finrl_pro_ds.crypto.envs.funding_arb_env import FundingArbEnv

logger = logging.getLogger(__name__)


def load_config(config_path: str | None = None) -> dict:
    """Load experiment configuration from YAML."""
    if config_path is None:
        config_path = str(
            PROJECT_ROOT / "configs" / "funding_arb_delta_neutral.yaml"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    strategy_id = config["strategy"].get("id", config["strategy"]["name"])
    logger.info(f"Loaded config: {strategy_id}")
    return config


def prepare_data(config: dict) -> dict:
    """Fetch and prepare spot + perp data through the full pipeline.

    Returns dict with keys: spot_ohlcv, perp_ohlcv, funding, arb_features, walk_forward
    """
    data_cfg = config["data"]
    universe_cfg = config["universe"]

    assets = universe_cfg["assets"]
    start = data_cfg["start_date"]
    end = data_cfg.get("end_date")
    exchange = universe_cfg.get("data_exchange", "binance")
    cache_dir = data_cfg.get("cache_dir", "./data/crypto_cache")

    logger.info(f"Fetching data: {len(assets)} assets, {start} → {end or 'now'}")

    # Fetch perp data (reuse existing pipeline)
    perp_data = fetch_crypto_data(
        assets=assets, start=start, end=end,
        exchange=exchange, cache_dir=cache_dir,
    )
    perp_ohlcv = perp_data["ohlcv"]
    funding = perp_data["funding"]

    # Fetch spot data
    logger.info("Fetching spot OHLCV data...")
    spot_ohlcv = fetch_spot_data(
        assets=assets, start=start, end=end,
        exchange=exchange, cache_dir=cache_dir,
    )

    # Compute funding-arb features
    logger.info("Computing funding arb features...")
    arb_features = compute_funding_arb_features(
        perp_ohlcv=perp_ohlcv,
        spot_ohlcv=spot_ohlcv,
        funding_df=funding,
    )

    # Validate walk-forward coverage
    wf_cfg = config["walk_forward"]
    wf_validator = WalkForwardCoverageValidator(
        train_bars=wf_cfg["train_bars"],
        val_bars=wf_cfg["val_bars"],
        test_bars=wf_cfg["test_bars"],
        step_bars=wf_cfg["step_bars"],
        embargo_bars=wf_cfg.get("embargo_bars", 0),
    )
    wf_result = wf_validator.validate(perp_ohlcv)

    return {
        "spot_ohlcv": spot_ohlcv,
        "perp_ohlcv": perp_ohlcv,
        "funding": funding,
        "arb_features": arb_features,
        "walk_forward": wf_result,
    }


def create_env(arrays: dict, config: dict) -> FundingArbEnv:
    """Create a FundingArbEnv from prepared arrays and config."""
    env_cfg = config["environment"]

    return FundingArbEnv(
        spot_price_ary=arrays["spot_price_ary"],
        perp_price_ary=arrays["perp_price_ary"],
        funding_rate_ary=arrays["funding_rate_ary"],
        spot_volume_ary=arrays["spot_volume_ary"],
        perp_volume_ary=arrays["perp_volume_ary"],
        tech_ary=arrays["tech_ary"],
        timestamps=arrays["timestamps"],
        initial_capital=env_cfg["initial_capital"],
        spot_taker_fee_pct=env_cfg["spot_taker_fee_pct"],
        perp_taker_fee_pct=env_cfg["perp_taker_fee_pct"],
        perp_margin_rate=env_cfg.get("perp_margin_rate", 0.05),
        slippage_base_bps=env_cfg.get("slippage_base_bps", 2.0),
        slippage_impact_bps=env_cfg.get("slippage_impact_bps", 10.0),
        max_gross_exposure=env_cfg["max_gross_exposure"],
        deadband_threshold=env_cfg.get("deadband_threshold", 0.02),
        lambda_basis=env_cfg.get("lambda_basis", 2.0),
        lambda_delta=env_cfg.get("lambda_delta", 5.0),
        lambda_turnover=env_cfg.get("lambda_turnover", 0.003),
        reward_scaling=env_cfg.get("reward_scaling", 100.0),
        reward_clip_range=tuple(env_cfg.get("reward_clip_range", [-5.0, 5.0])),
        circuit_breaker_threshold=env_cfg.get("circuit_breaker_threshold", 0.05),
        enable_trade_log=True,
    )


def _make_sb3_agent(agent_type: str, env: FundingArbEnv, agent_cfg: dict):
    """Instantiate an SB3 agent from config."""
    from stable_baselines3 import PPO, SAC, A2C

    cls_map = {"ppo": PPO, "sac": SAC, "a2c": A2C}
    cls = cls_map[agent_type]

    params = {}
    for key in ("learning_rate", "gamma", "n_steps", "batch_size",
                "n_epochs", "gae_lambda", "clip_range", "ent_coef",
                "buffer_size", "tau"):
        if key in agent_cfg:
            params[key] = agent_cfg[key]

    net_arch = agent_cfg.get("network_arch")
    if net_arch:
        params["policy_kwargs"] = {"net_arch": list(net_arch)}

    return cls("MlpPolicy", env, verbose=0, **params)


def _evaluate_agent_on_env(model, env: FundingArbEnv) -> dict:
    """Run a trained model through the env and return metrics."""
    obs, _ = env.reset()
    done = False
    portfolio_values = [env.initial_capital]
    total_funding = 0.0
    max_delta = 0.0
    active_pairs_history = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
        total_funding = info["total_funding_earned"]
        max_delta = max(max_delta, abs(info["net_delta"]))
        active_pairs_history.append(info["n_active_pairs"])
        done = terminated or truncated

    pv = np.array(portfolio_values)
    returns = np.diff(pv) / np.maximum(pv[:-1], 1e-10)
    total_return = pv[-1] / pv[0] - 1.0

    sharpe = 0.0
    if len(returns) > 1 and np.std(returns, ddof=1) > 1e-10:
        sharpe = float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(8760))

    peak = np.maximum.accumulate(pv)
    drawdown = 1.0 - pv / np.where(peak == 0, 1.0, peak)
    max_dd = float(drawdown.max())

    return {
        "total_return": float(total_return),
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_steps": len(returns),
        "final_value": float(pv[-1]),
        "total_funding_earned": total_funding,
        "total_basis_pnl": float(env._calc_total_unrealized_basis_pnl(
            env.spot_price_ary[env.step_idx],
            env.perp_price_ary[env.step_idx],
        )),
        "cumulative_fees": env.cumulative_fees,
        "funding_vs_costs_ratio": (
            total_funding / (env.cumulative_fees + 1e-10)
        ),
        "max_delta_exposure": max_delta,
        "avg_active_pairs": float(np.mean(active_pairs_history)) if active_pairs_history else 0.0,
    }


def run_backtest(config: dict, dry_run: bool = False) -> dict:
    """Run the full walk-forward backtest pipeline."""
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

    # Step 2: Walk-forward loop
    logger.info("=" * 60)
    logger.info("PHASE 2: Walk-Forward Training & Evaluation")
    logger.info("=" * 60)

    assets = config["universe"]["assets"]
    agents_cfg = config.get("agents", {})
    agent_type = agents_cfg.get("primary", "ppo")
    total_timesteps = agents_cfg.get("total_timesteps", 2_000_000)
    agent_specific = agents_cfg.get(agent_type, {})

    # Inherit top-level network_arch
    if "network_arch" not in agent_specific and "network_arch" in agents_cfg:
        agent_specific["network_arch"] = agents_cfg["network_arch"]

    window_schedule = wf["window_schedule"]
    if dry_run:
        window_schedule = window_schedule[:1]
        total_timesteps = min(total_timesteps, 10_000)
        logger.info("DRY RUN: 1 window, reduced timesteps")

    window_results = []

    for window in window_schedule:
        w_idx = window["window"]
        logger.info(f"\n--- Window {w_idx} ---")
        logger.info(f"  Train: {window['train_start']} → {window['train_end']}")
        logger.info(f"  Val:   {window['val_start']} → {window['val_end']}")
        logger.info(f"  Test:  {window['test_start']} → {window['test_end']}")

        try:
            # Build arrays per window (LEAK-1)
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

            train_env = create_env(train_arrays, config)
            val_env = create_env(val_arrays, config)
            test_env = create_env(test_arrays, config)

            # Train
            logger.info(f"  Training {agent_type.upper()} ({total_timesteps} steps)...")
            model = _make_sb3_agent(agent_type, train_env, agent_specific)
            model.learn(total_timesteps=total_timesteps)

            # Evaluate on val (for logging)
            val_metrics = _evaluate_agent_on_env(model, val_env)
            logger.info(
                f"  Val: return={val_metrics['total_return']:.2%}, "
                f"funding/costs={val_metrics['funding_vs_costs_ratio']:.2f}"
            )

            # Evaluate on test
            test_metrics = _evaluate_agent_on_env(model, test_env)
            window_results.append({
                "window": w_idx,
                "train_start": str(window["train_start"]),
                "test_end": str(window["test_end"]),
                **test_metrics,
            })

            logger.info(
                f"  Test: return={test_metrics['total_return']:.2%}, "
                f"sharpe={test_metrics['sharpe']:.3f}, "
                f"funding/costs={test_metrics['funding_vs_costs_ratio']:.2f}, "
                f"max_delta={test_metrics['max_delta_exposure']:.4f}"
            )

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

    if results_df.empty or results_df.get("sharpe", pd.Series(dtype=float)).isna().all():
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
        "median_funding_vs_costs": float(
            results_df.get("funding_vs_costs_ratio", pd.Series([0.0])).median()
        ),
        "median_max_delta": float(
            results_df.get("max_delta_exposure", pd.Series([0.0])).median()
        ),
        "median_avg_active_pairs": float(
            results_df.get("avg_active_pairs", pd.Series([0.0])).median()
        ),
        "results": results_df.to_dict("records"),
    }

    logger.info(f"  Windows:             {summary['n_windows']}")
    logger.info(f"  Median Sharpe:       {summary['median_sharpe']:.3f}")
    logger.info(f"  Median Return:       {summary['median_return']:.2%}")
    logger.info(f"  Median Max DD:       {summary['median_max_dd']:.2%}")
    logger.info(f"  Median Fund/Costs:   {summary['median_funding_vs_costs']:.2f}")
    logger.info(f"  Median Max Delta:    {summary['median_max_delta']:.4f}")
    logger.info(f"  Median Active Pairs: {summary['median_avg_active_pairs']:.1f}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Funding Rate Arbitrage Backtest Runner")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to experiment config YAML"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Run 1 window with reduced timesteps for validation"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    results = run_backtest(config, dry_run=args.dry_run)

    # Persist results
    out_dir = Path("results") / "funding_arb"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "backtest_results.json"

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

    if results["status"] == "COMPLETED":
        logger.info("Backtest COMPLETED successfully")
    else:
        logger.error(f"Backtest FAILED: {results.get('reason', 'unknown')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
