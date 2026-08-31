"""Retrain Funding-Arb SAC from saved best-params JSON.

Recovers model checkpoints for windows that completed HPO + full-train
but didn't save the trained model (pre-checkpoint-fix runs).

Usage:
    # Retrain window 14 using its saved best params
    python scripts/funding_arb_retrain_from_params.py \
      --config configs/funding_arb_sac_10assets_hpo.yaml \
      --params_json results/funding_arb_10a/w14_best_sac.json \
      --window 14

    # Retrain with custom seed
    python scripts/funding_arb_retrain_from_params.py \
      --config configs/funding_arb_sac_10assets_hpo.yaml \
      --params_json results/funding_arb_10a/w14_best_sac.json \
      --window 14 --seed 456
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sharpen.crypto.data.crypto_array_builder import (  # noqa: E402
    build_funding_arb_arrays,
)
from scripts.funding_arb_runner import (  # noqa: E402
    _evaluate_agent_on_env,
    create_env,
    load_config,
    prepare_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def retrain_window(
    config: dict,
    params_json: dict,
    window_idx: int,
    out_dir: Path,
    seed: int = 42,
) -> dict:
    """Retrain a single window from saved best params and save checkpoint."""
    # Extract params
    agent_params = params_json["agent_params"]
    env_overrides = params_json["env_overrides"]
    logger.info(f"Retraining window {window_idx} with params from trial {params_json.get('best_trial', '?')}")
    logger.info(f"  Agent params: {agent_params}")
    logger.info(f"  Env overrides: {env_overrides}")

    # Prepare data
    logger.info("Preparing data...")
    data = prepare_data(config)
    wf = data["walk_forward"]
    if not wf["passed"]:
        raise RuntimeError("Walk-forward coverage check FAILED")

    schedule = wf["window_schedule"]
    if window_idx >= len(schedule):
        raise ValueError(f"Window {window_idx} out of range (max {len(schedule) - 1})")

    window = schedule[window_idx]
    assets = config["universe"]["assets"]

    # Build arrays for this window
    logger.info(f"Building arrays: {window['train_start']} -> {window['test_end']}")
    train_arrays = build_funding_arb_arrays(
        data["spot_ohlcv"], data["perp_ohlcv"],
        data["arb_features"], data["funding"],
        assets, window["train_start"], window["train_end"],
    )
    test_arrays = build_funding_arb_arrays(
        data["spot_ohlcv"], data["perp_ohlcv"],
        data["arb_features"], data["funding"],
        assets, window["test_start"], window["test_end"],
    )

    # Build vectorized train env with env overrides
    import copy
    from stable_baselines3.common.vec_env import DummyVecEnv

    cfg = copy.deepcopy(config)
    for k, v in env_overrides.items():
        cfg["environment"][k] = v

    n_envs = config.get("agents", {}).get("n_envs", 4)
    network_arch = config.get("agents", {}).get("sac", {}).get("network_arch", [256, 256])
    total_timesteps = config.get("agents", {}).get("total_timesteps", 2_000_000)

    def _make_fn(a=train_arrays, c=cfg):
        return create_env(a, c)

    vec_env = DummyVecEnv([_make_fn for _ in range(n_envs)])

    # Train
    # Note: original HPO runner does not set a seed, so retrain results will
    # differ from the original run. This is a "re-roll" with the same HPs,
    # not exact reproduction. Use different seeds for robustness testing.
    sac_cfg = {**agent_params, "network_arch": network_arch}
    logger.info(f"Training SAC for {total_timesteps:,} steps (seed={seed}, n_envs={n_envs})...")
    from stable_baselines3 import SAC
    model = SAC(
        "MlpPolicy", vec_env, verbose=0, seed=seed,
        **{k: v for k, v in sac_cfg.items() if k != "network_arch"},
        policy_kwargs={"net_arch": list(network_arch)},
    )

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps)
    elapsed = time.time() - t0
    vec_env.close()
    logger.info(f"Training complete in {elapsed:.0f}s ({total_timesteps / elapsed:.0f} SPS)")

    # Save checkpoint
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"w{window_idx}_sac_full.zip"
    model.save(str(checkpoint_path))
    logger.info(f"Checkpoint saved: {checkpoint_path}")

    # Evaluate on test set
    test_env = create_env(test_arrays, cfg)
    test_metrics = _evaluate_agent_on_env(model, test_env)
    logger.info(
        f"Test: return={test_metrics['total_return']:.2%}, "
        f"sharpe={test_metrics.get('sharpe', 0):.3f}, "
        f"funding/costs={test_metrics.get('funding_vs_costs_ratio', 0):.2f}, "
        f"max_dd={test_metrics.get('max_drawdown', 0):.2%}",
    )

    # Save results
    result = {
        "window": window_idx,
        "seed": seed,
        "status": "COMPLETED",
        "checkpoint": str(checkpoint_path),
        "elapsed_seconds": elapsed,
        "sps": total_timesteps / elapsed,
        **test_metrics,
    }
    results_path = out_dir / f"w{window_idx}_retrain_results.json"
    with open(results_path, "w") as f:
        json.dump(
            result, f, indent=2,
            default=lambda x: float(x) if isinstance(x, (np.floating,)) else x,
        )
    logger.info(f"Results saved: {results_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Retrain funding-arb SAC from saved params")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--params_json", required=True, help="Path to w{i}_best_sac.json")
    parser.add_argument("--window", type=int, required=True, help="Walk-forward window index")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--out_dir", default=None, help="Output directory (default: results/funding_arb_retrain)")
    args = parser.parse_args()

    config = load_config(args.config)

    with open(args.params_json, "r") as f:
        params_json = json.load(f)

    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "results" / "funding_arb_retrain"

    result = retrain_window(config, params_json, args.window, out_dir, seed=args.seed)

    if result["status"] == "COMPLETED":
        logger.info(f"SUCCESS: Window {args.window} retrained. Checkpoint: {result['checkpoint']}")
    else:
        logger.error(f"FAILED: Window {args.window}")
        sys.exit(1)


if __name__ == "__main__":
    main()
