#!/usr/bin/env python
"""
Walk-Forward Evaluation — Rolling Window Backtesting
=====================================================
Orchestrates train → validate → test across multiple temporal folds
using RollingWindowSplitter.  Aggregates per-fold metrics into a
summary table for regime-robustness assessment.

Usage:
    python scripts/run_walk_forward.py --config configs/v95_walk_forward.yaml
    python scripts/run_walk_forward.py --config configs/v95_walk_forward.yaml --dry-run
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import wandb
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finrl_pro_ds.data.splitter import RollingWindowSplitter
from finrl_pro_ds.logging import clear_namespace, set_namespace

logger = logging.getLogger("WalkForward")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load YAML config."""
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Fold Runner
# ---------------------------------------------------------------------------

def run_fold(fold_idx: int, train_range, val_range, test_range, config: dict, dry_run: bool = False,
             seed: int | None = None):
    """
    Execute a single walk-forward fold:
      1. Train agent on train_range
      2. Validate on val_range (optional: early stopping)
      3. Backtest on test_range → collect metrics

    Returns dict of fold metrics.
    """
    fold_id = f"fold_{fold_idx:02d}"
    logger.info(f"{'='*60}")
    logger.info(f"  FOLD {fold_idx}: Train {train_range} | Val {val_range} | Test {test_range}")
    logger.info(f"{'='*60}")

    if dry_run:
        logger.info(f"  [DRY RUN] Skipping actual training/backtest for {fold_id}")
        return {
            "fold": fold_idx,
            "train_range": str(train_range),
            "val_range": str(val_range),
            "test_range": str(test_range),
            "sharpe": None,
            "max_drawdown": None,
            "total_return_pct": None,
            "status": "dry_run",
        }

    # ------------------------------------------------------------------
    # Phase 1: Training
    # ------------------------------------------------------------------
    try:
        import copy

        from scripts.run_full_pipeline import run_backtest, run_training

        fold_config = copy.deepcopy(config)

        # Override date ranges for this fold
        fold_config.setdefault("data", {})
        fold_config["data"]["train_start_date"] = str(train_range[0])
        fold_config["data"]["train_end_date"] = str(train_range[1])
        fold_config["data"]["val_start_date"] = str(val_range[0])
        fold_config["data"]["val_end_date"] = str(val_range[1])
        fold_config["data"]["test_start_date"] = str(test_range[0])
        fold_config["data"]["test_end_date"] = str(test_range[1])

        # Run training
        # Include seed when set so multiseed WF checkpoint dirs
        # (checkpoints/<run_name>/checkpoint_final.pth) don't collide.
        prefix = f"WF_seed{seed}" if seed is not None else "WF"
        run_name = f"{prefix}_{fold_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        device = fold_config.get("agent", {}).get("device", "cuda")
        # Agent type defaults to 'bdq' in run_training. For modern configs
        # that key the agent under `agents.<type>`, pick that key so SAC /
        # DSAC / PPO configs train the right model.
        agent_type = (fold_config.get("agent_type")
                      or next(iter(fold_config.get("agents", {}).keys()), "bdq"))
        checkpoint_path = run_training(fold_config, run_name, device,
                                       agent_type=agent_type)

        # ------------------------------------------------------------------
        # Phase 2: Backtest on test range
        # ------------------------------------------------------------------
        if checkpoint_path and os.path.exists(checkpoint_path):
            metrics = run_backtest(
                fold_config,
                checkpoint_path,
                device,
                start_date=str(test_range[0]),
                end_date=str(test_range[1]),
                prefix=f"wf_{fold_id}",
                agent_type=agent_type,
            )
        else:
            logger.warning(f"  No checkpoint found for {fold_id}, skipping backtest")
            metrics = {}

        return {
            "fold": fold_idx,
            "train_range": str(train_range),
            "val_range": str(val_range),
            "test_range": str(test_range),
            "sharpe": metrics.get("sharpe_ratio"),
            "max_drawdown": metrics.get("max_drawdown"),
            "total_return_pct": metrics.get("total_return_pct"),
            "status": "completed",
        }

    except Exception as e:
        logger.error(f"  Fold {fold_idx} failed: {e}", exc_info=True)
        return {
            "fold": fold_idx,
            "train_range": str(train_range),
            "val_range": str(val_range),
            "test_range": str(test_range),
            "sharpe": None,
            "max_drawdown": None,
            "total_return_pct": None,
            "status": f"error: {e}",
        }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list):
    """Print aggregated walk-forward results."""
    print("\n" + "=" * 80)
    print("  WALK-FORWARD EVALUATION SUMMARY")
    print("=" * 80)
    print(f"  {'Fold':<8} {'Test Range':<30} {'Sharpe':>10} {'MaxDD':>10} {'Return%':>10} {'Status':<12}")
    print("-" * 80)

    sharpes, drawdowns, returns = [], [], []
    for r in results:
        sharpe_str = f"{r['sharpe']:.3f}" if r['sharpe'] is not None else "N/A"
        dd_str = f"{r['max_drawdown']:.2%}" if r['max_drawdown'] is not None else "N/A"
        ret_str = f"{r['total_return_pct']:.2f}%" if r['total_return_pct'] is not None else "N/A"

        print(f"  {r['fold']:<8} {r['test_range']:<30} {sharpe_str:>10} {dd_str:>10} {ret_str:>10} {r['status']:<12}")

        if r["sharpe"] is not None:
            sharpes.append(r["sharpe"])
        if r["max_drawdown"] is not None:
            drawdowns.append(r["max_drawdown"])
        if r["total_return_pct"] is not None:
            returns.append(r["total_return_pct"])

    print("-" * 80)
    if sharpes:
        import numpy as np
        print(f"  {'MEAN':<8} {'':30} {np.mean(sharpes):>10.3f} {np.mean(drawdowns):>10.2%} {np.mean(returns):>10.2f}%")
        print(f"  {'STD':<8} {'':30} {np.std(sharpes):>10.3f} {np.std(drawdowns):>10.2%} {np.std(returns):>10.2f}%")
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Print folds without running")
    parser.add_argument("--separate_runs", action="store_true",
                        help="Legacy: skip consolidated WandB parent run. Folds "
                             "log to whatever run run_training happens to create.")
    parser.add_argument("--run_name_prefix", default="wf",
                        help="WandB run name prefix (default: 'wf')")
    parser.add_argument("--seed", type=int, default=None,
                        help="Global random seed (torch, numpy, random). Applied once at startup.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    if args.seed is not None:
        import random as _random

        import numpy as _np
        import torch as _torch
        _torch.manual_seed(args.seed)
        _np.random.seed(args.seed)
        _random.seed(args.seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(args.seed)
        logger.info(f"Global seed set: {args.seed}")

    config = load_config(args.config)

    # Extract splitter params
    splitter_cfg = config.get("splitter", {})
    data_cfg = config.get("data", {})

    splitter = RollingWindowSplitter(
        train_months=splitter_cfg.get("train_months", 3),
        val_months=splitter_cfg.get("val_months", 1),
        test_months=splitter_cfg.get("test_months", 1),
        step_months=splitter_cfg.get("step_months", 1),
        buffer_days=splitter_cfg.get("buffer_days", 0),
    )

    folds = splitter.split(
        start_date=data_cfg.get("start_date", "2025-01-01"),
        end_date=data_cfg.get("end_date", "2025-06-30"),
    )
    logger.info(f"Generated {len(folds)} walk-forward folds")

    if not folds:
        logger.error("No folds generated! Check date range and splitter config.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Consolidated WandB parent run — one run covers all folds (S488+).
    # Each fold's internal wandb.log calls get prefixed win<i>/* via
    # set_namespace. --separate_runs preserves legacy behavior.
    # ------------------------------------------------------------------
    parent_run = None
    if not args.dry_run and not args.separate_runs:
        wcfg = config.get("wandb", {}) or {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parent_run = wandb.init(
            project=wcfg.get("project", "FinRL-Pro-DS"),
            entity=wcfg.get("entity", "bigcan-chiwin-technology"),
            name=f"{args.run_name_prefix}_{timestamp}",
            job_type="walk_forward",
            tags=list(wcfg.get("tags", []) or []) + ["walk_forward", "consolidated"],
            config={
                "experiment": "walk_forward",
                "n_folds": len(folds),
                "config_path": str(args.config),
                "splitter": splitter_cfg,
            },
        )
        logger.info("Parent WandB run: %s", parent_run.url)

    # Run each fold
    results = []
    for i, fold in enumerate(folds):
        # splitter.split() returns list[dict[str, TimeRange]]; unpack explicitly
        # so we don't iterate dict keys.
        train_range = fold["train"].to_tuple()
        val_range = fold["val"].to_tuple()
        test_range = fold["test"].to_tuple()
        if parent_run is not None:
            set_namespace(f"win{i}")
        fold_result = run_fold(i, train_range, val_range, test_range, config,
                               dry_run=args.dry_run, seed=args.seed)
        results.append(fold_result)

        # Final per-window summary dict logged directly from the driver.
        if parent_run is not None:
            summary = {
                f"win{i}/summary/sharpe": fold_result.get("sharpe"),
                f"win{i}/summary/max_drawdown": fold_result.get("max_drawdown"),
                f"win{i}/summary/total_return_pct": fold_result.get("total_return_pct"),
                f"win{i}/summary/status": fold_result.get("status"),
            }
            wandb.log({k: v for k, v in summary.items() if v is not None})

    if parent_run is not None:
        clear_namespace()
        # Aggregate summary across all folds (WF-level header metrics).
        sharpes = [r["sharpe"] for r in results if r.get("sharpe") is not None]
        dds = [r["max_drawdown"] for r in results if r.get("max_drawdown") is not None]
        if sharpes:
            import numpy as _np
            wandb.summary["wf/sharpe_mean"] = float(_np.mean(sharpes))
            wandb.summary["wf/sharpe_std"] = float(_np.std(sharpes))
        if dds:
            import numpy as _np
            wandb.summary["wf/max_drawdown_mean"] = float(_np.mean(dds))
        wandb.summary["wf/n_folds"] = len(results)
        wandb.summary["wf/n_completed"] = sum(1 for r in results if r.get("status") == "completed")
        parent_run.finish()

    # Print summary
    print_summary(results)

    return results


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
