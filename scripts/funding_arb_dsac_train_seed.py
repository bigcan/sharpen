"""Funding-Arb DSAC — single-seed training runner (Protocol v2 Stage 2).

Train ONE seed with HPs already baked into the config (typically from a
post-HPO L1 multiseed YAML such as `configs/funding_arb_dsac_l1_multiseed.yaml`).
No HPO. No walk-forward sweep. One window, one seed, one checkpoint.

Designed to slot into the existing launch chain:
    launch_l1_multiseed.py
        → deploy_bare_metal.py (--script scripts/funding_arb_dsac_train_seed.py)
            → this script (--config <yaml> --seed <N> --run_name <str>)

Reuses helpers from funding_arb_hpo_runner.py + funding_arb_runner.py to avoid
duplicating env/agent/eval logic.

Usage:
    python scripts/funding_arb_dsac_train_seed.py \
        --config configs/funding_arb_dsac_l1_multiseed.yaml \
        --seed 42 \
        --run_name funding-arb-dsac-l1-seed42

Outputs (in `checkpoints/<run_name>_<timestamp>/`):
    - checkpoint_final.pth (DSAC agent state)
    - manifest.json (seed, HPs, val/test metrics, env_code_sha, timestamp)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sharpen.crypto.data.crypto_array_builder import (  # noqa: E402
    build_funding_arb_arrays,
)
from scripts.funding_arb_hpo_runner import (  # noqa: E402
    _create_eval_env,
    _DSACModelWrapper,
    _make_dsac_agent,
    _train_dsac_agent,
)
from scripts.funding_arb_runner import (  # noqa: E402
    _evaluate_agent_on_env,
    prepare_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("funding_arb_train_seed")


_AGENT_PARAM_KEYS = ("learning_rate", "buffer_size", "batch_size", "gamma", "tau", "learning_starts")
_DSAC_PARAM_KEYS = ("n_quantiles", "cvar_alpha", "kappa")


def _set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CUDA included if available)."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_hps(config: dict) -> tuple[dict, dict]:
    """Pull agent_params + dsac_params from `agents.sac.*` (already HPO'd in config)."""
    sac_cfg = (config.get("agents") or {}).get("sac") or {}
    agent_params = {k: sac_cfg[k] for k in _AGENT_PARAM_KEYS if k in sac_cfg}
    dsac_params = {k: sac_cfg[k] for k in _DSAC_PARAM_KEYS if k in sac_cfg}
    return agent_params, dsac_params


def _select_window(data: dict, window_index: int) -> dict:
    """Pick one walk-forward window for L1 multiseed (single-window evaluation)."""
    schedule = data["walk_forward"]["window_schedule"]
    if not schedule:
        raise RuntimeError("Walk-forward schedule is empty — data preparation failed")
    if window_index >= len(schedule):
        raise IndexError(
            f"window_index={window_index} out of range; only {len(schedule)} windows available",
        )
    return schedule[window_index]


def _build_arrays_for_window(data: dict, assets: list[str], window: dict) -> tuple[dict, dict, dict]:
    """Build train/val/test arrays for a single walk-forward window."""
    train = build_funding_arb_arrays(
        data["spot_ohlcv"], data["perp_ohlcv"],
        data["arb_features"], data["funding"],
        assets, window["train_start"], window["train_end"],
    )
    val = build_funding_arb_arrays(
        data["spot_ohlcv"], data["perp_ohlcv"],
        data["arb_features"], data["funding"],
        assets, window["val_start"], window["val_end"],
    )
    test = build_funding_arb_arrays(
        data["spot_ohlcv"], data["perp_ohlcv"],
        data["arb_features"], data["funding"],
        assets, window["test_start"], window["test_end"],
    )
    return train, val, test


def _maybe_init_wandb(config: dict, run_name: str, seed: int):
    """Init WandB. Honors S488 consolidation: if FINRL_WANDB_RUN_ID is set,
    attach to the parent run with namespace from FINRL_WANDB_NAMESPACE.
    """
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not available — skipping WandB init")
        return None

    wb_cfg = config.get("wandb") or {}
    project = wb_cfg.get("project", "FinRL-Pro-DS")
    entity = wb_cfg.get("entity")
    tags = list(wb_cfg.get("tags") or []) + [f"seed{seed}"]

    shared_run_id = os.environ.get("FINRL_WANDB_RUN_ID")
    namespace = os.environ.get("FINRL_WANDB_NAMESPACE")

    init_kwargs = dict(
        project=project,
        entity=entity,
        name=run_name,
        tags=tags,
        config={
            "seed": seed,
            "hp_provenance": (config.get("upstream") or {}).get("hpo_study"),
            "hpo_best_trial": (config.get("upstream") or {}).get("hpo_best_trial"),
        },
    )
    if shared_run_id:
        init_kwargs["id"] = shared_run_id
        init_kwargs["resume"] = "allow"
        logger.info("WandB consolidation: attaching to parent run %s as namespace %s",
                    shared_run_id, namespace or f"seed{seed}")

    return wandb.init(**init_kwargs)


def _wandb_log(metrics: dict, namespace: str | None = None) -> None:
    """Log metrics under optional namespace prefix (S488 consolidation)."""
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    if namespace:
        metrics = {f"{namespace}/{k}": v for k, v in metrics.items()}
    wandb.log(metrics)


def train_seed(
    config: dict,
    seed: int,
    run_name: str,
    out_root: Path,
    window_index: int = 0,
    steps_override: int | None = None,
) -> dict:
    """End-to-end: load data, train DSAC, eval val/test, save checkpoint + manifest."""
    _set_seed(seed)

    agent_params, dsac_params = _extract_hps(config)
    if not agent_params or not dsac_params:
        missing = []
        if not agent_params:
            missing.append("agent_params (learning_rate/buffer_size/batch_size/gamma/tau)")
        if not dsac_params:
            missing.append("dsac_params (n_quantiles/cvar_alpha/kappa)")
        raise RuntimeError(
            f"Config missing HPO'd HPs in agents.sac.*: {missing}. "
            f"This runner expects baked-in HPs (Protocol v2 Stage 2).",
        )

    logger.info("Seed %d | agent_params=%s | dsac_params=%s", seed, agent_params, dsac_params)

    # --- Data ---
    logger.info("Preparing funding-arb data (ccxt fetch on first call, cached after)...")
    data = prepare_data(config)
    if not data["walk_forward"]["passed"]:
        raise RuntimeError("Walk-forward coverage check FAILED — see prepare_data logs")

    window = _select_window(data, window_index)
    logger.info("Window %d: train %s -> %s, val %s -> %s, test %s -> %s",
                window["window"],
                window["train_start"], window["train_end"],
                window["val_start"], window["val_end"],
                window["test_start"], window["test_end"])

    assets = config["universe"]["assets"]
    train_arrays, val_arrays, test_arrays = _build_arrays_for_window(data, assets, window)

    # --- Train ---
    train_env = _create_eval_env(train_arrays, config, env_overrides=None)
    agent = _make_dsac_agent(train_env, config, agent_params, dsac_params)

    total_timesteps = steps_override or (config.get("agents") or {}).get("total_timesteps", 2_000_000)
    logger.info("Training DSAC for %d steps (seed=%d, window=%d)...", total_timesteps, seed, window["window"])
    t0 = time.time()
    _train_dsac_agent(agent, train_env, total_timesteps, global_step_offset=0)
    train_secs = time.time() - t0
    logger.info("Training complete in %.1fs (%.1f SPS)", train_secs, total_timesteps / max(train_secs, 1e-6))

    # --- Eval ---
    model = _DSACModelWrapper(agent)
    val_env = _create_eval_env(val_arrays, config, env_overrides=None)
    val_metrics = _evaluate_agent_on_env(model, val_env)
    test_env = _create_eval_env(test_arrays, config, env_overrides=None)
    test_metrics = _evaluate_agent_on_env(model, test_env)

    # Strip non-serializable internal _portfolio_values list before logging
    val_log = {k: v for k, v in val_metrics.items() if not k.startswith("_")}
    test_log = {k: v for k, v in test_metrics.items() if not k.startswith("_")}
    logger.info("VAL  metrics: %s", val_log)
    logger.info("TEST metrics: %s", test_log)

    # --- Persist checkpoint + manifest ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"{run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / "checkpoint_final.pth"
    model.save(str(ckpt_path))
    logger.info("Saved checkpoint to %s", ckpt_path)

    manifest = {
        "run_name": run_name,
        "seed": seed,
        "window_index": window["window"],
        "window_dates": {
            "train_start": window["train_start"], "train_end": window["train_end"],
            "val_start": window["val_start"], "val_end": window["val_end"],
            "test_start": window["test_start"], "test_end": window["test_end"],
        },
        "agent_params": agent_params,
        "dsac_params": dsac_params,
        "total_timesteps": total_timesteps,
        "train_seconds": train_secs,
        "val_metrics": val_log,
        "test_metrics": test_log,
        "upstream": config.get("upstream") or {},
        "timestamp": datetime.now().isoformat(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    # --- WandB final metrics ---
    namespace = os.environ.get("FINRL_WANDB_NAMESPACE")
    _wandb_log({
        "train/seconds": train_secs,
        "train/sps": total_timesteps / max(train_secs, 1e-6),
        **{f"val/{k}": v for k, v in val_log.items()},
        **{f"test/{k}": v for k, v in test_log.items()},
    }, namespace=namespace)

    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path,
                   help="L1 multiseed YAML with HPO'd HPs baked into agents.sac.*")
    p.add_argument("--seed", required=True, type=int, help="Random seed")
    p.add_argument("--run_name", default=None, help="WandB / checkpoint run name (default: derived from config + seed)")
    p.add_argument("--out_dir", default=str(PROJECT_ROOT / "checkpoints"),
                   help="Root checkpoint directory")
    p.add_argument("--window_index", type=int, default=0,
                   help="Which walk-forward window to train on (default 0)")
    p.add_argument("--steps_override", type=int, default=None,
                   help="Override agents.total_timesteps (smoke testing)")
    # Accept and ignore the deploy_bare_metal-injected hpo_storage flag.
    p.add_argument("--hpo_storage", default=None, help=argparse.SUPPRESS)
    p.add_argument("--tags", nargs="*", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    if not args.config.is_file():
        logger.error("Config not found: %s", args.config)
        return 2

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_name = args.run_name or f"{config['strategy']['name']}-seed{args.seed}"
    out_root = Path(args.out_dir)

    _maybe_init_wandb(config, run_name, args.seed)

    try:
        manifest = train_seed(
            config, args.seed, run_name, out_root,
            window_index=args.window_index,
            steps_override=args.steps_override,
        )
        logger.info("DONE — seed=%d val/total_return=%.4f test/total_return=%.4f",
                    args.seed,
                    manifest["val_metrics"].get("total_return", float("nan")),
                    manifest["test_metrics"].get("total_return", float("nan")))
        return 0
    except Exception as e:
        logger.error("Training failed: %s", e, exc_info=True)
        return 1
    finally:
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    sys.exit(main())
