"""AlphaSeek Phase 6 — Per-Agent HPO + Walk-Forward Training Pipeline.

Each agent type (D3QN, DoubleDQN, TwinD3QN) gets an independent Optuna sweep
per walk-forward window. Best params from each are used for full training,
then agents are combined into an ensemble for test evaluation.

Usage:
    # Smoke test (2 trials, 1K steps, 1 window)
    python scripts/alphaseek_hpo_runner.py \
      --config configs/alphaseek_hpo.yaml \
      --max_windows 1 --n_trials 2 --hpo_break_step 1000 --full_break_step 2000

    # Production HPO (50 trials, 160K steps, all windows)
    python scripts/alphaseek_hpo_runner.py \
      --config configs/alphaseek_hpo.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from finrl_pro_ds.alphaseek.agents import (  # noqa: E402
    AGENT_MAP,
    AgentDoubleDQN,
)
from finrl_pro_ds.alphaseek.lob_trade_simulator import (  # noqa: E402
    EvalLOBTradeSimulator,
    LOBTradeSimulator,
)
from finrl_pro_ds.alphaseek.trainer import AlphaSeekTrainer  # noqa: E402

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


# ---------------------------------------------------------------------------
# Walk-Forward Window Schedule
# ---------------------------------------------------------------------------

def build_window_schedule(
    n_segments: int,
    train_segments: int = 8,
    val_segments: int = 1,
    test_segments: int = 1,
    slide_by: int = 1,
) -> list[dict]:
    """Build segment-based walk-forward window schedule.

    Returns list of dicts with keys: window, train_segs, val_segs, test_segs.
    """
    window_size = train_segments + val_segments + test_segments
    windows = []
    w_idx = 0
    start = 0
    while start + window_size <= n_segments:
        train_end = start + train_segments
        val_end = train_end + val_segments
        test_end = val_end + test_segments
        windows.append({
            "window": w_idx,
            "train_segs": list(range(start, train_end)),
            "val_segs": list(range(train_end, val_end)),
            "test_segs": list(range(val_end, test_end)),
        })
        w_idx += 1
        start += slide_by
    return windows


# ---------------------------------------------------------------------------
# HPO Search Space
# ---------------------------------------------------------------------------

def define_search_space(trial) -> dict:
    """Define Optuna search space for AlphaSeek DQN HPO."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
        "soft_update_tau": trial.suggest_float("soft_update_tau", 1e-6, 5e-3, log=True),
        "reward_scale": trial.suggest_float("reward_scale", 100, 10000, log=True),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "net_dims": trial.suggest_categorical("net_dims", ["128,128,128", "256,256", "128,128"]),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "step_gap": trial.suggest_categorical("step_gap", [1, 2, 4, 8]),
        "explore_rate": trial.suggest_float("explore_rate", 0.005, 0.05, log=True),
        "clip_grad_norm": trial.suggest_float("clip_grad_norm", 1.0, 10.0),
        "stop_loss_thresh": trial.suggest_float("stop_loss_thresh", 5e-5, 5e-3, log=True),
    }


# ---------------------------------------------------------------------------
# HPO Objective
# ---------------------------------------------------------------------------

def hpo_objective(
    trial,
    agent_class: type[AgentDoubleDQN],
    agent_name: str,
    lob_path: str,
    train_segs: list[int],
    val_segs: list[int],
    config: dict,
    hpo_break_step: int,
    gpu_id: int,
    out_dir: str,
    window_idx: int,
) -> float:
    """Optuna objective: train agent → eval on val → return total_return."""
    # Sample hyperparameters
    params = define_search_space(trial)

    # Merge with base config
    merged = dict(config.get("agent", {}))
    merged.update(params)
    merged["num_envs"] = config.get("env", {}).get("num_sims_hpo", 512)

    # Build feature config
    feat_cfg = config.get("features", {})
    env_cfg = config.get("env", {})

    try:
        # Build train simulator with HPO params
        train_sim = LOBTradeSimulator(
            lob_parquet_path=lob_path,
            num_sims=merged["num_envs"],
            step_gap=params["step_gap"],
            slippage=env_cfg.get("slippage", 7e-7),
            max_position=env_cfg.get("max_position", 1),
            num_ignore_step=env_cfg.get("num_ignore_step", 60),
            seq_len=env_cfg.get("seq_len", 3600),
            stop_loss_thresh=params["stop_loss_thresh"],
            norm_span=feat_cfg.get("norm_span", 120),
            momentum_window=feat_cfg.get("momentum_window", 5),
            vol_window=feat_cfg.get("vol_window", 30),
            gpu_id=gpu_id,
            segment_filter=train_segs,
        )

        # Build eval simulator on val segments
        eval_sim = EvalLOBTradeSimulator(
            lob_parquet_path=lob_path,
            num_sims=config.get("env", {}).get("num_sims_eval", 64),
            step_gap=params["step_gap"],
            slippage=env_cfg.get("slippage", 7e-7),
            max_position=env_cfg.get("max_position", 1),
            num_ignore_step=env_cfg.get("num_ignore_step", 60),
            seq_len=env_cfg.get("seq_len", 3600),
            stop_loss_thresh=params["stop_loss_thresh"],
            norm_span=feat_cfg.get("norm_span", 120),
            momentum_window=feat_cfg.get("momentum_window", 5),
            vol_window=feat_cfg.get("vol_window", 30),
            gpu_id=gpu_id,
            segment_filter=val_segs,
        )

        # Train
        trial_cwd = os.path.join(out_dir, f"w{window_idx}", agent_name, f"t{trial.number}")
        trainer = AlphaSeekTrainer(
            agent_class=agent_class,
            train_sim=train_sim,
            eval_sim=eval_sim,
            config=merged,
            gpu_id=gpu_id,
            cwd=trial_cwd,
        )
        metrics = trainer.train(break_step=hpo_break_step)

        total_return = metrics.get("total_return", -float("inf"))

        # Log to WandB
        _wandb_log({
            f"hpo/w{window_idx}/{agent_name}/t{trial.number}/total_return": total_return,
            f"hpo/w{window_idx}/{agent_name}/t{trial.number}/sharpe": metrics.get("sharpe", 0),
            f"hpo/w{window_idx}/{agent_name}/t{trial.number}/pf": metrics.get("profit_factor", 0),
            f"hpo/w{window_idx}/{agent_name}/t{trial.number}/lr": params["learning_rate"],
        })

        logger.info(
            f"  Trial {trial.number} | {agent_name} | "
            f"return={total_return:.6f} | sharpe={metrics.get('sharpe', 0):.4f}",
        )

    except Exception as e:
        logger.error(f"  Trial {trial.number} FAILED: {e}")
        total_return = -float("inf")
    finally:
        gc.collect()
        if gpu_id >= 0:
            import torch
            torch.cuda.empty_cache()

    return total_return


# ---------------------------------------------------------------------------
# Per-Agent HPO
# ---------------------------------------------------------------------------

def run_agent_hpo(
    agent_class: type[AgentDoubleDQN],
    agent_name: str,
    window_idx: int,
    lob_path: str,
    train_segs: list[int],
    val_segs: list[int],
    config: dict,
    n_trials: int,
    hpo_break_step: int,
    gpu_id: int,
    out_dir: str,
) -> dict:
    """Run Optuna HPO for a single agent type on a single window.

    Returns dict with best_params and best_value.
    """
    import optuna
    from optuna.pruners import NopPruner
    from optuna.samplers import TPESampler

    study_name = f"alphaseek_w{window_idx}_{agent_name}"
    db_path = os.path.join(out_dir, f"w{window_idx}", f"hpo_{agent_name}.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=TPESampler(seed=42, n_startup_trials=10, multivariate=True),
        pruner=NopPruner(),
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
    )

    remaining = max(0, n_trials - len(study.trials))
    if remaining > 0:
        logger.info(
            f"[W{window_idx}] {agent_name} HPO: {remaining} trials "
            f"({len(study.trials)} existing)",
        )
        study.optimize(
            lambda trial: hpo_objective(
                trial, agent_class, agent_name, lob_path,
                train_segs, val_segs, config, hpo_break_step, gpu_id,
                out_dir, window_idx,
            ),
            n_trials=remaining,
        )
    else:
        logger.info(
            f"[W{window_idx}] {agent_name} HPO: already complete "
            f"({len(study.trials)} trials)",
        )

    best = study.best_trial
    logger.info(
        f"[W{window_idx}] {agent_name} HPO BEST: "
        f"T{best.number} return={best.value:.6f}",
    )

    # Save best params
    best_params_path = os.path.join(
        out_dir, f"w{window_idx}", f"best_{agent_name}.json",
    )
    with open(best_params_path, "w") as f:
        json.dump({"trial": best.number, "value": best.value, "params": best.params}, f, indent=2)

    return {"best_params": best.params, "best_value": best.value, "best_trial": best.number}


# ---------------------------------------------------------------------------
# Full Training + Ensemble Eval
# ---------------------------------------------------------------------------

def full_train_agent(
    agent_class: type[AgentDoubleDQN],
    agent_name: str,
    best_params: dict,
    window_idx: int,
    lob_path: str,
    train_segs: list[int],
    test_segs: list[int],
    config: dict,
    full_break_step: int,
    gpu_id: int,
    out_dir: str,
) -> dict:
    """Full training with best HPO params, eval on test segments."""
    feat_cfg = config.get("features", {})
    env_cfg = config.get("env", {})

    merged = dict(config.get("agent", {}))
    merged.update(best_params)
    merged["num_envs"] = env_cfg.get("num_sims", 4096)

    train_sim = LOBTradeSimulator(
        lob_parquet_path=lob_path,
        num_sims=merged["num_envs"],
        step_gap=best_params.get("step_gap", env_cfg.get("step_gap", 2)),
        slippage=env_cfg.get("slippage", 7e-7),
        max_position=env_cfg.get("max_position", 1),
        num_ignore_step=env_cfg.get("num_ignore_step", 60),
        seq_len=env_cfg.get("seq_len", 3600),
        stop_loss_thresh=best_params.get("stop_loss_thresh", env_cfg.get("stop_loss_thresh", 1e-3)),
        norm_span=feat_cfg.get("norm_span", 120),
        momentum_window=feat_cfg.get("momentum_window", 5),
        vol_window=feat_cfg.get("vol_window", 30),
        gpu_id=gpu_id,
        segment_filter=train_segs,
    )

    test_sim = EvalLOBTradeSimulator(
        lob_parquet_path=lob_path,
        num_sims=env_cfg.get("num_sims_eval", 64),
        step_gap=best_params.get("step_gap", env_cfg.get("step_gap", 2)),
        slippage=env_cfg.get("slippage", 7e-7),
        max_position=env_cfg.get("max_position", 1),
        num_ignore_step=env_cfg.get("num_ignore_step", 60),
        seq_len=env_cfg.get("seq_len", 3600),
        stop_loss_thresh=best_params.get("stop_loss_thresh", env_cfg.get("stop_loss_thresh", 1e-3)),
        norm_span=feat_cfg.get("norm_span", 120),
        momentum_window=feat_cfg.get("momentum_window", 5),
        vol_window=feat_cfg.get("vol_window", 30),
        gpu_id=gpu_id,
        segment_filter=test_segs,
    )

    cwd = os.path.join(out_dir, f"w{window_idx}", agent_name, "full")
    trainer = AlphaSeekTrainer(
        agent_class=agent_class,
        train_sim=train_sim,
        eval_sim=test_sim,
        config=merged,
        gpu_id=gpu_id,
        cwd=cwd,
    )

    logger.info(f"[W{window_idx}] {agent_name} FULL TRAINING: {full_break_step} steps")
    metrics = trainer.train(break_step=full_break_step)

    logger.info(
        f"[W{window_idx}] {agent_name} TEST: "
        f"return={metrics['total_return']:.6f} | "
        f"sharpe={metrics['sharpe']:.4f} | "
        f"PF={metrics['profit_factor']:.3f}",
    )

    gc.collect()
    if gpu_id >= 0:
        import torch
        torch.cuda.empty_cache()

    return metrics


# ---------------------------------------------------------------------------
# Per-Window Orchestration
# ---------------------------------------------------------------------------

def _run_agent_pipeline(
    agent_name: str,
    agent_class_name: str,
    window: dict,
    config: dict,
    lob_path: str,
    n_trials: int,
    hpo_break_step: int,
    full_break_step: int,
    gpu_id: int,
    out_dir: str,
    result_queue: Queue | None = None,
) -> dict:
    """Run HPO + full train for a single agent on a single GPU.

    Designed to run either in-process (serial) or as a subprocess (parallel).
    When result_queue is provided, pushes result to queue instead of returning.
    """
    from finrl_pro_ds.alphaseek.agents import AGENT_MAP as _AGENT_MAP

    agent_class = _AGENT_MAP[agent_name]
    w_idx = window["window"]
    train_segs = window["train_segs"]
    val_segs = window["val_segs"]
    test_segs = window["test_segs"]

    logger.info(f"[W{w_idx}] {agent_name} starting on GPU {gpu_id}")

    hpo_result = run_agent_hpo(
        agent_class=agent_class,
        agent_name=agent_name,
        window_idx=w_idx,
        lob_path=lob_path,
        train_segs=train_segs,
        val_segs=val_segs,
        config=config,
        n_trials=n_trials,
        hpo_break_step=hpo_break_step,
        gpu_id=gpu_id,
        out_dir=out_dir,
    )

    test_metrics = full_train_agent(
        agent_class=agent_class,
        agent_name=agent_name,
        best_params=hpo_result["best_params"],
        window_idx=w_idx,
        lob_path=lob_path,
        train_segs=train_segs,
        test_segs=test_segs,
        config=config,
        full_break_step=full_break_step,
        gpu_id=gpu_id,
        out_dir=out_dir,
    )

    result = {
        "agent_name": agent_name,
        "hpo_best_trial": hpo_result["best_trial"],
        "hpo_best_value": hpo_result["best_value"],
        "hpo_best_params": hpo_result["best_params"],
        "test_metrics": test_metrics,
    }

    if result_queue is not None:
        result_queue.put(result)
    return result


def run_window(
    window: dict,
    config: dict,
    lob_path: str,
    n_trials: int,
    hpo_break_step: int,
    full_break_step: int,
    gpu_id: int,
    out_dir: str,
    gpu_ids: list[int] | None = None,
    agent_filter: list[str] | None = None,
) -> dict:
    """Run HPO + full train for all agent types on one window.

    Parameters
    ----------
    gpu_ids : list[int] | None
        If provided, agents are distributed across GPUs in parallel.
        e.g. [0, 1] runs 2 agents in parallel on GPU 0 and GPU 1.
    agent_filter : list[str] | None
        If provided, only run these agent types (e.g. ["D3QN"]).
    """
    w_idx = window["window"]
    train_segs = window["train_segs"]
    val_segs = window["val_segs"]
    test_segs = window["test_segs"]

    logger.info(
        f"\n{'='*60}\n"
        f"Window {w_idx}: train={train_segs} val={val_segs} test={test_segs}\n"
        f"{'='*60}",
    )

    agent_names = config.get("ensemble", {}).get("agent_classes", ["D3QN", "DoubleDQN", "TwinD3QN"])
    if agent_filter:
        agent_names = [a for a in agent_names if a in agent_filter]
    window_results = {"window": w_idx, "agents": {}}

    effective_gpu_ids = gpu_ids if gpu_ids else [gpu_id]
    use_parallel = len(effective_gpu_ids) > 1 and len(agent_names) > 1

    if use_parallel:
        # Parallel: distribute agents across GPUs
        logger.info(
            f"[W{w_idx}] Parallel mode: {len(agent_names)} agents across GPUs {effective_gpu_ids}",
        )
        result_queue = Queue()
        processes = []

        for i, agent_name in enumerate(agent_names):
            assigned_gpu = effective_gpu_ids[i % len(effective_gpu_ids)]
            p = Process(
                target=_run_agent_pipeline,
                kwargs={
                    "agent_name": agent_name,
                    "agent_class_name": agent_name,
                    "window": window,
                    "config": config,
                    "lob_path": lob_path,
                    "n_trials": n_trials,
                    "hpo_break_step": hpo_break_step,
                    "full_break_step": full_break_step,
                    "gpu_id": assigned_gpu,
                    "out_dir": out_dir,
                    "result_queue": result_queue,
                },
                name=f"alphaseek-{agent_name}-gpu{assigned_gpu}",
            )
            processes.append(p)
            p.start()
            logger.info(f"[W{w_idx}] Spawned {agent_name} on GPU {assigned_gpu} (PID {p.pid})")

        # Collect results
        for p in processes:
            p.join()
        while not result_queue.empty():
            result = result_queue.get()
            window_results["agents"][result["agent_name"]] = {
                "hpo_best_trial": result["hpo_best_trial"],
                "hpo_best_value": result["hpo_best_value"],
                "hpo_best_params": result["hpo_best_params"],
                "test_metrics": result["test_metrics"],
            }
    else:
        # Serial: single GPU
        for agent_name in agent_names:
            result = _run_agent_pipeline(
                agent_name=agent_name,
                agent_class_name=agent_name,
                window=window,
                config=config,
                lob_path=lob_path,
                n_trials=n_trials,
                hpo_break_step=hpo_break_step,
                full_break_step=full_break_step,
                gpu_id=gpu_id,
                out_dir=out_dir,
            )
            window_results["agents"][agent_name] = {
                "hpo_best_trial": result["hpo_best_trial"],
                "hpo_best_value": result["hpo_best_value"],
                "hpo_best_params": result["hpo_best_params"],
                "test_metrics": result["test_metrics"],
            }

    # Log window summary
    for name, res in window_results["agents"].items():
        m = res["test_metrics"]
        _wandb_log({
            f"window/{w_idx}/{name}/test_return": m["total_return"],
            f"window/{w_idx}/{name}/test_sharpe": m["sharpe"],
            f"window/{w_idx}/{name}/test_pf": m["profit_factor"],
            f"window/{w_idx}/{name}/test_max_dd": m["max_drawdown"],
        })

    # Save window results
    results_path = os.path.join(out_dir, f"w{w_idx}", "window_results.json")
    with open(results_path, "w") as f:
        json.dump(window_results, f, indent=2, default=str)

    return window_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AlphaSeek Phase 6 HPO Runner")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--max_windows", type=int, default=None, help="Max walk-forward windows")
    parser.add_argument("--n_trials", type=int, default=None, help="HPO trials per agent per window")
    parser.add_argument("--hpo_break_step", type=int, default=None, help="Steps per HPO trial")
    parser.add_argument("--full_break_step", type=int, default=None, help="Steps for full training")
    parser.add_argument("--out_dir", type=str, default="outputs/alphaseek_hpo", help="Output directory")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID (single GPU mode)")
    parser.add_argument("--gpu_ids", type=str, default=None, help="Comma-separated GPU IDs for parallel agents (e.g. '0,1,2')")
    parser.add_argument("--agent_filter", type=str, default=None, help="Comma-separated agent names to run (e.g. 'D3QN,DoubleDQN')")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--tags", nargs="*", default=[], help="Additional WandB tags")
    # Compatibility with deploy_bare_metal.py (auto-added flags)
    parser.add_argument("--run_name", type=str, default=None, help="Run name (from deploy script)")
    parser.add_argument("--hpo_storage", type=str, default=None, help="HPO storage path (ignored, uses per-agent DBs)")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override from CLI
    hpo_cfg = config.get("hpo", {})
    n_trials = args.n_trials or hpo_cfg.get("n_trials", 50)
    hpo_break_step = args.hpo_break_step or hpo_cfg.get("hpo_break_step", 160000)
    full_break_step = args.full_break_step or config.get("training", {}).get("full_break_step", 320000)

    lob_path = config["env"]["lob_parquet_path"]

    # Get segment info
    seg_info = LOBTradeSimulator.get_segment_info(lob_path)
    n_segments = len(seg_info)
    logger.info(f"LOB data: {n_segments} segments")
    logger.info(f"\n{seg_info.to_string()}")

    # Build walk-forward windows
    wf_cfg = config.get("walk_forward", {})
    windows = build_window_schedule(
        n_segments=n_segments,
        train_segments=wf_cfg.get("train_segments", 8),
        val_segments=wf_cfg.get("val_segments", 1),
        test_segments=wf_cfg.get("test_segments", 1),
        slide_by=wf_cfg.get("slide_by", 1),
    )

    if args.max_windows:
        windows = windows[:args.max_windows]

    logger.info(f"Walk-forward: {len(windows)} windows")
    for w in windows:
        logger.info(f"  W{w['window']}: train={w['train_segs']} val={w['val_segs']} test={w['test_segs']}")

    # Init WandB
    wandb_cfg = config.get("wandb", {})
    if args.wandb or wandb_cfg.get("enabled", False):
        try:
            import wandb
            tags = wandb_cfg.get("tags", []) + args.tags
            wandb.init(
                entity=wandb_cfg.get("entity", "bigcan-chiwin-technology"),
                project=wandb_cfg.get("project", "FinRL-Pro-DS"),
                name=f"alphaseek-phase6-hpo-{len(windows)}w",
                tags=tags,
                config={
                    "hpo": {"n_trials": n_trials, "hpo_break_step": hpo_break_step},
                    "training": {"full_break_step": full_break_step},
                    "walk_forward": {"n_windows": len(windows)},
                    "agent": config.get("agent", {}),
                    "env": config.get("env", {}),
                },
                reinit=True,
            )
        except Exception as e:
            logger.warning(f"WandB init failed: {e}")

    # Parse multi-GPU and agent filter
    gpu_ids = [int(x) for x in args.gpu_ids.split(",")] if args.gpu_ids else None
    agent_filter = [x.strip() for x in args.agent_filter.split(",")] if args.agent_filter else None

    if gpu_ids:
        logger.info(f"Multi-GPU mode: agents distributed across GPUs {gpu_ids}")
    if agent_filter:
        logger.info(f"Agent filter: only running {agent_filter}")

    # Run windows
    os.makedirs(args.out_dir, exist_ok=True)
    all_results = []
    pipeline_start = time.time()

    for window in windows:
        window_result = run_window(
            window=window,
            config=config,
            lob_path=lob_path,
            n_trials=n_trials,
            hpo_break_step=hpo_break_step,
            full_break_step=full_break_step,
            gpu_id=args.gpu_id,
            out_dir=args.out_dir,
            gpu_ids=gpu_ids,
            agent_filter=agent_filter,
        )
        all_results.append(window_result)

    # Aggregate summary
    pipeline_elapsed = time.time() - pipeline_start
    logger.info(f"\n{'='*60}\nPIPELINE COMPLETE: {pipeline_elapsed:.1f}s\n{'='*60}")

    agent_names = config.get("ensemble", {}).get("agent_classes", ["D3QN", "DoubleDQN", "TwinD3QN"])
    for agent_name in agent_names:
        returns = [
            r["agents"][agent_name]["test_metrics"]["total_return"]
            for r in all_results
            if agent_name in r["agents"]
        ]
        sharpes = [
            r["agents"][agent_name]["test_metrics"]["sharpe"]
            for r in all_results
            if agent_name in r["agents"]
        ]
        if returns:
            logger.info(
                f"{agent_name}: median_return={np.median(returns):.6f} "
                f"median_sharpe={np.median(sharpes):.4f} "
                f"({sum(1 for r in returns if r > 0)}/{len(returns)} profitable)",
            )
            _wandb_log({
                f"summary/{agent_name}/median_return": float(np.median(returns)),
                f"summary/{agent_name}/median_sharpe": float(np.median(sharpes)),
                f"summary/{agent_name}/n_profitable": sum(1 for r in returns if r > 0),
                f"summary/{agent_name}/n_windows": len(returns),
            })

    # Save full results
    results_path = os.path.join(args.out_dir, "all_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
