"""Cross-asset momentum allocator pipeline — Protocol-v2 staged runner.

Phase 4 of the cross-sectional pivot (S553-cont-34). Orchestrates the RL allocator
built in Phases 1-3 over the validated linear TSMOM core:

    data-prep (cross_asset_loader)  →  per-window SAC HPO (objective=Sharpe, ADR-4)
      →  full train  →  RL net-Sharpe eval  →  FROZEN linear-core net-Sharpe eval
      →  RL-beats-linear gate  →  Protocol-v2 stage manifest (status PASS/FAIL)

The decisive gate (``configs/cross_asset_momentum.gates.yaml``): the RL allocator
must beat the frozen monthly linear core OOS by ``rl_beats_linear.min_uplift_vs_baseline``
net Sharpe, evaluated through the SAME env/convention/costs (see
``allocator_factory.evaluate_linear_core``) — else we **ship the linear rule**.

Stages (Protocol v2):
    --stage hpo   window 0 only: HPO + train + eval + gate (Stage-1 decision artifact)
    --stage wf    all walk-forward windows (Stage-3 deploy-gating verdict)

CPU-testable pieces (no GPU/SB3 import at module load): the WF scheduler, the gate
logic, the manifest writer, and the config→env mapping (via allocator_factory).
The SAC HPO/train is the only GPU step; SB3/Optuna are lazy-imported.

Usage:
    # Smoke (1 window, 2 trials, 1k steps) — end-to-end wiring check
    python scripts/cross_asset_pipeline.py --config configs/cross_asset_momentum.yaml --smoke
    # Stage-1 HPO (window 0)
    python scripts/cross_asset_pipeline.py --config configs/cross_asset_momentum.yaml --stage hpo
    # Stage-3 walk-forward
    python scripts/cross_asset_pipeline.py --config configs/cross_asset_momentum.yaml --stage wf
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finrl_pro_ds.data import cross_asset_loader as loader  # noqa: E402
from finrl_pro_ds.envs import allocator_factory as factory  # noqa: E402

logger = logging.getLogger("cross_asset_pipeline")

ANN = 252


# --------------------------------------------------------------------------- #
# Config / gates
# --------------------------------------------------------------------------- #
def load_config(path: str) -> dict:
    """Load a YAML config (utf-8 — the configs carry box-drawing chars; cp950 default
    on Windows crashes plain reads)."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_gate_thresholds(config: dict) -> dict:
    """Read the RL-beats-linear gate thresholds from the gates overlay file
    (gates live in configs, never hardcoded — CLAUDE.md invariant)."""
    gates_file = config.get("ensemble", {}).get("gates_file")
    defaults = {"min_uplift": 0.10, "wf_net_sharpe_floor": 0.40, "max_cost_gap": 0.15}
    if not gates_file:
        logger.warning("no ensemble.gates_file — using gate defaults %s", defaults)
        return defaults
    p = (PROJECT_ROOT / gates_file) if not Path(gates_file).is_absolute() else Path(gates_file)
    g = yaml.safe_load(p.read_text(encoding="utf-8")).get("gates", {})
    return {
        "min_uplift": float(g.get("rl_beats_linear", {}).get("min_uplift_vs_baseline", 0.10)),
        "wf_net_sharpe_floor": float(g.get("wf_net_sharpe_floor", 0.40)),
        "max_cost_gap": float(g.get("g_cost_gap", {}).get("max_frictionless_minus_net_sharpe", 0.15)),
    }


# --------------------------------------------------------------------------- #
# Walk-forward scheduler (daily bars; embargo gaps = LEAK guard)
# --------------------------------------------------------------------------- #
def build_wf_schedule(dates: pd.DatetimeIndex, wf: dict) -> list[dict]:
    """Slice the daily index into [train | embargo | val | embargo | test] windows.

    Bar-count based (train_bars/val_bars/test_bars/step_bars/embargo_bars). Drops
    (and logs) any trailing partial window — never silently truncates coverage.
    """
    train, val, test = int(wf["train_bars"]), int(wf["val_bars"]), int(wf["test_bars"])
    step, embargo = int(wf["step_bars"]), int(wf.get("embargo_bars", 0))
    n = len(dates)
    windows, start, w = [], 0, 0
    while True:
        tr_s, tr_e = start, start + train
        va_s, va_e = tr_e + embargo, tr_e + embargo + val
        te_s, te_e = va_e + embargo, va_e + embargo + test
        if te_e > n:
            break
        windows.append({
            "window": w,
            "train_start": dates[tr_s], "train_end": dates[tr_e - 1],
            "val_start": dates[va_s], "val_end": dates[va_e - 1],
            "test_start": dates[te_s], "test_end": dates[te_e - 1],
        })
        start += step
        w += 1
    logger.info("WF schedule: %d windows from %d bars (train=%d val=%d test=%d step=%d embargo=%d)",
                len(windows), n, train, val, test, step, embargo)
    if windows:
        last_end = windows[-1]["test_end"]
        dropped = int((dates > last_end).sum())
        if dropped:
            logger.info("WF: %d trailing bars after last window's test_end dropped (partial window)", dropped)
    return windows


# --------------------------------------------------------------------------- #
# SB3 SAC (lazy) — allocator HPO + training
# --------------------------------------------------------------------------- #
def _make_sac(vec_env, agent_params: dict, net_arch: list[int]):
    from stable_baselines3 import SAC
    params = {k: agent_params[k] for k in (
        "learning_rate", "buffer_size", "batch_size", "gamma", "tau",
        "learning_starts") if k in agent_params}
    return SAC("MlpPolicy", vec_env, verbose=0,
               policy_kwargs={"net_arch": list(net_arch)}, **params)


def _make_vec_env(arrays: dict, config: dict, n_envs: int, overrides: dict | None = None):
    from stable_baselines3.common.vec_env import DummyVecEnv
    def _fn(a=arrays, c=config, o=overrides):
        return factory.make_allocator_env(a, c, overrides=o)
    return DummyVecEnv([_fn for _ in range(n_envs)])


def _evaluate_model(model, arrays: dict, config: dict, overrides: dict | None = None) -> dict:
    """RL net metrics via deterministic env-native rollout (same convention/costs
    as ``evaluate_linear_core`` → fair gate)."""
    env = factory.make_allocator_env(arrays, config, overrides=overrides, eval_mode=True)
    obs, _ = env.reset()
    step_returns, turnovers, pvs = [], [], [env.initial_capital]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        step_returns.append(info["step_return"])
        turnovers.append(info["turnover"])
        pvs.append(info["portfolio_value"])
        done = terminated or truncated
    return factory._metrics(step_returns, turnovers, pvs)


# --------------------------------------------------------------------------- #
# WandB (optional; liveness for monitor_fleet — avoids the false-CRIT stall)
# --------------------------------------------------------------------------- #
def _wandb_log(metrics: dict) -> None:
    """Best-effort WandB log (no-op if wandb absent / not initialized)."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics)
    except Exception:  # noqa: BLE001
        pass


def _heartbeat_callback(label: str, log_every: int = 10_000):
    """SB3 callback that WandB-logs step/SPS periodically so monitor_fleet sees
    the run is alive (the project's documented false-CRIT silent-stall guard)."""
    from stable_baselines3.common.callbacks import BaseCallback

    class _HB(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self._t0 = time.time()
            self._last = 0

        def _on_step(self) -> bool:
            s = self.num_timesteps
            if s - self._last >= log_every:
                self._last = s
                el = time.time() - self._t0
                _wandb_log({f"train/{label}/step": s,
                            f"train/{label}/sps": s / el if el > 0 else 0.0})
            return True

    return _HB()


def _sac_search_space(trial, config: dict | None = None) -> dict:
    """SAC search space. Bounds for turnover_penalty / learning_starts / buffer_size
    are overridable via ``config.hpo.search`` so an anti-overfit retry can widen the
    turnover-penalty range and shrink learning_starts/buffer without a code edit."""
    s = ((config or {}).get("hpo", {}) or {}).get("search", {}) or {}
    tp_lo, tp_hi = s.get("turnover_penalty", [0.0005, 0.01])
    ls_lo, ls_hi = s.get("learning_starts", [1000, 10000])
    bf_lo, bf_hi = s.get("buffer_size", [100_000, 1_000_000])
    return {
        "agent_params": {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "buffer_size": trial.suggest_int("buffer_size", int(bf_lo), int(bf_hi), log=True),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
            "gamma": trial.suggest_float("gamma", 0.95, 0.999),
            "tau": trial.suggest_float("tau", 0.001, 0.02, log=True),
            "learning_starts": trial.suggest_int("learning_starts", int(ls_lo), int(ls_hi), log=True),
        },
        "env_overrides": {
            "turnover_penalty": trial.suggest_float("turnover_penalty", float(tp_lo), float(tp_hi), log=True),
        },
    }


def _run_hpo(train_arrays, val_arrays, config, n_trials, steps, net_arch,
             *, storage: str | None = None, study_name: str = "xsec_hpo") -> dict:
    """Optuna SAC HPO; objective = val net Sharpe (ADR-4). Returns best params.

    ``storage`` (deploy-injected sqlite URL) makes the study resumable — already
    completed trials are skipped on restart."""
    import optuna

    n_envs = config.get("training", {}).get("num_envs", 8)

    def objective(trial):
        search = _sac_search_space(trial, config)
        try:
            vec_env = _make_vec_env(train_arrays, config, n_envs, search["env_overrides"])
            model = _make_sac(vec_env, search["agent_params"], net_arch)
            model.learn(total_timesteps=steps,
                        callback=_heartbeat_callback(f"hpo_t{trial.number}"))
            m = _evaluate_model(model, val_arrays, config, search["env_overrides"])
            vec_env.close()
            del model, vec_env
            gc.collect()
            if m["n_steps"] < 50:
                return -999.0
            _wandb_log({f"hpo/{study_name}/trial": trial.number,
                        f"hpo/{study_name}/val_net_sharpe": m["net_sharpe"]})
            return m["net_sharpe"]
        except Exception as e:  # noqa: BLE001
            logger.error("HPO trial %d failed: %s", trial.number, e)
            gc.collect()
            return -999.0

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
        pruner=optuna.pruners.NopPruner(),
        storage=storage,
        load_if_exists=bool(storage),
    )
    remaining = max(n_trials - len([t for t in study.trials if t.state.is_finished()]), 0) \
        if storage else n_trials
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    best = study.best_trial
    env_keys = {"turnover_penalty"}
    return {
        "best_value": float(best.value),
        "agent_params": {k: v for k, v in best.params.items() if k not in env_keys},
        "env_overrides": {k: v for k, v in best.params.items() if k in env_keys},
        "best_trial": best.number,
    }


# --------------------------------------------------------------------------- #
# Per-window: HPO → train → RL eval → linear-core eval → gate
# --------------------------------------------------------------------------- #
def run_window(w_idx, window, data, config, n_trials, hpo_steps, train_steps,
               out_dir: Path, gate: dict) -> dict:
    lookbacks = data["lookbacks"]
    norm_window = int(config.get("features", {}).get("norm_window", 252))
    net_arch = config.get("agents", {}).get("sac", {}).get("network_arch", [256, 256])

    def arrays_for(a, b):
        return loader.build_allocator_arrays(
            data["signals"], data["close"], data["volume"], data["assets"],
            a, b, lookbacks=lookbacks, norm_window=norm_window)

    train_arrays = arrays_for(window["train_start"], window["train_end"])
    val_arrays = arrays_for(window["val_start"], window["val_end"])
    test_arrays = arrays_for(window["test_start"], window["test_end"])

    # Per-window Optuna storage: resumable AND contention-free when windows run
    # concurrently across GPUs (a single shared sqlite would lock under parallel writes).
    storage = f"sqlite:///{(out_dir / f'hpo_w{w_idx}.db').resolve().as_posix()}"
    logger.info("=== Window %d: HPO (%d trials × %d steps) ===", w_idx, n_trials, hpo_steps)
    hpo = _run_hpo(train_arrays, val_arrays, config, n_trials, hpo_steps, net_arch,
                   storage=storage, study_name=f"xsec_w{w_idx}")
    logger.info("  best val net Sharpe=%.3f trial=%d params=%s",
                hpo["best_value"], hpo["best_trial"], hpo["agent_params"])

    logger.info("=== Window %d: full train (%d steps) ===", w_idx, train_steps)
    n_envs = config.get("training", {}).get("num_envs", 8)
    vec_env = _make_vec_env(train_arrays, config, n_envs, hpo["env_overrides"])
    model = _make_sac(vec_env, hpo["agent_params"], net_arch)
    model.learn(total_timesteps=train_steps,
                callback=_heartbeat_callback(f"w{w_idx}_train"))
    model_path = out_dir / f"w{w_idx}_sac"
    model.save(str(model_path))
    vec_env.close()

    # Test eval — RL vs frozen linear core (identical env/convention/costs).
    rl_test = _evaluate_model(model, test_arrays, config, hpo["env_overrides"])
    core_test = factory.evaluate_linear_core(test_arrays, config, overrides=hpo["env_overrides"])
    # g_cost_gap (artifact detector): frictionless RL minus net RL Sharpe. A large
    # gap = the edge is cost-fragile (the AlphaSeek/HFT failure mode). Reuse the
    # trained model; only the env costs change (zero fee + zero slippage).
    fric_over = {**hpo["env_overrides"], "taker_fee": 0.0,
                 "slippage_base_bps": 0.0, "slippage_impact_bps": 0.0}
    rl_frictionless = _evaluate_model(model, test_arrays, config, fric_over)
    cost_gap = rl_frictionless["net_sharpe"] - rl_test["net_sharpe"]
    cost_gap_pass = cost_gap <= gate["max_cost_gap"]

    uplift = rl_test["net_sharpe"] - core_test["net_sharpe"]
    window_pass = (uplift >= gate["min_uplift"]) \
        and (rl_test["net_sharpe"] >= gate["wf_net_sharpe_floor"]) \
        and cost_gap_pass

    del model
    gc.collect()

    result = {
        "window": w_idx,
        "train": [str(window["train_start"].date()), str(window["train_end"].date())],
        "test": [str(window["test_start"].date()), str(window["test_end"].date())],
        "hpo_best_val_sharpe": hpo["best_value"],
        "hpo_agent_params": hpo["agent_params"],
        "hpo_env_overrides": hpo["env_overrides"],
        "rl_test": rl_test,
        "linear_core_test": core_test,
        "rl_frictionless_net_sharpe": rl_frictionless["net_sharpe"],
        "uplift_net_sharpe": float(uplift),
        "cost_gap": float(cost_gap),
        "cost_gap_pass": bool(cost_gap_pass),
        "gate_pass": bool(window_pass),
        "model_path": str(model_path),
    }
    (out_dir / f"w{w_idx}_result.json").write_text(json.dumps(result, indent=2))
    _wandb_log({
        f"window/{w_idx}/rl_net_sharpe": rl_test["net_sharpe"],
        f"window/{w_idx}/linear_core_net_sharpe": core_test["net_sharpe"],
        f"window/{w_idx}/uplift_net_sharpe": uplift,
        f"window/{w_idx}/cost_gap": cost_gap,
        f"window/{w_idx}/gate_pass": int(window_pass),
    })
    logger.info("  Window %d: RL=%.3f  core=%.3f  uplift=%.3f  cost_gap=%.3f  gate=%s",
                w_idx, rl_test["net_sharpe"], core_test["net_sharpe"], uplift, cost_gap,
                "PASS" if window_pass else "fail")
    return result


# --------------------------------------------------------------------------- #
# Orchestrator + manifest
# --------------------------------------------------------------------------- #
def _assign_windows_to_gpus(schedule: list, gpus: list[int]) -> dict[int, list]:
    """Round-robin WF windows across GPUs (balanced ±1). Pure + testable."""
    buckets: dict[int, list] = {g: [] for g in gpus}
    for i, w in enumerate(schedule):
        buckets[gpus[i % len(gpus)]].append(w)
    return buckets


def _window_worker(gpu_id, window_dicts, config, out_dir_str, gate,
                   n_trials, hpo_steps, train_steps):
    """Spawn-subprocess entry: pin to ONE GPU, reload data from the (parent-warmed)
    cache, run the assigned windows. Each window writes its own w{idx}_result.json,
    aggregated by the parent. Module-level so it is picklable for the spawn context.
    Sets CUDA_VISIBLE_DEVICES before any torch/SB3 import (those are lazy in this
    module), so each worker sees exactly its GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [gpu{gpu_id}] [%(levelname)s] %(name)s: %(message)s")
    out_dir = Path(out_dir_str)
    data = loader.load_cross_asset_data(config)        # cache hit (parent fetched first)
    for w in window_dicts:
        try:
            run_window(w["window"], w, data, config, n_trials, hpo_steps,
                       train_steps, out_dir, gate)
        except Exception:  # noqa: BLE001
            logger.exception("[gpu%s] window %d FAILED", gpu_id, w["window"])


def run_pipeline(config: dict, *, stage: str, out_dir: Path, max_windows: int | None,
                 n_trials: int, hpo_steps: int, train_steps: int,
                 force_refetch: bool = False, gpus: list[int] | None = None,
                 windows: list[int] | None = None, aggregate_only: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    gate = load_gate_thresholds(config)
    logger.info("Gate thresholds: %s", gate)

    # Parent loads ONCE (fetch+clean+cache+signals) so concurrent workers hit the cache.
    data = loader.load_cross_asset_data(config, force_refetch=force_refetch)
    schedule = build_wf_schedule(data["close"].index, config["walk_forward"])
    if not schedule:
        return _write_manifest(out_dir, stage, config, gate, [], status="FAIL",
                               reason="insufficient_data_for_any_window")

    if stage == "hpo":
        schedule = schedule[:1]              # Stage-1 = window 0 decision artifact
    elif max_windows is not None:
        schedule = schedule[:max_windows]

    # --windows restricts which windows THIS process runs (cross-instance split);
    # --aggregate_only skips running and builds the WF verdict from collected JSONs.
    run_set = [w for w in schedule if windows is None or w["window"] in windows]

    gpus = gpus or []
    if aggregate_only:
        logger.info("aggregate-only: building WF verdict from %d windows in %s",
                    len(schedule), out_dir)
    elif len(gpus) > 1 and len(run_set) > 1:
        # Window-level parallelism: round-robin windows to one spawn-worker per GPU.
        # Windows are fully independent (own per-window study/model/arrays), so this
        # is embarrassingly parallel — no shared Optuna RDB, no cross-window state.
        buckets = _assign_windows_to_gpus(run_set, gpus)
        logger.info("Parallel WF: %d windows across GPUs %s -> %s", len(run_set), gpus,
                    {g: [w["window"] for w in ws] for g, ws in buckets.items()})
        ctx = mp.get_context("spawn")
        procs = []
        for g, ws in buckets.items():
            if not ws:
                continue
            p = ctx.Process(target=_window_worker,
                            args=(g, ws, config, str(out_dir), gate,
                                  n_trials, hpo_steps, train_steps))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
    else:
        for window in run_set:
            try:
                run_window(window["window"], window, data, config, n_trials,
                           hpo_steps, train_steps, out_dir, gate)
            except Exception:  # noqa: BLE001
                logger.exception("Window %d FAILED", window["window"])

    # Aggregate from per-window result JSONs (uniform path for serial + parallel).
    # aggregate-only builds the FULL-schedule verdict (cross-instance collect); a
    # subset run reports only the windows it ran.
    agg_set = schedule if aggregate_only else run_set
    results = []
    for w in agg_set:
        p = out_dir / f"w{w['window']}_result.json"
        if p.exists():
            results.append(json.loads(p.read_text()))
        else:
            results.append({"window": w["window"], "status": "FAILED",
                            "error": "no result json (window crashed)"})
    manifest = _write_manifest(out_dir, stage, config, gate, results, status="PASS")
    _wandb_log({
        "wf/median_uplift_net_sharpe": manifest.get("median_uplift_net_sharpe") or 0.0,
        "wf/median_rl_net_sharpe": manifest.get("median_rl_net_sharpe") or 0.0,
        "wf/n_windows_gate_pass": manifest.get("n_windows_gate_pass") or 0,
    })
    return manifest


def _write_manifest(out_dir: Path, stage: str, config: dict, gate: dict,
                    results: list[dict], *, status: str, reason: str | None = None) -> dict:
    ok = [r for r in results if "uplift_net_sharpe" in r]
    uplifts = [r["uplift_net_sharpe"] for r in ok]
    rl_sharpes = [r["rl_test"]["net_sharpe"] for r in ok]
    core_sharpes = [r["linear_core_test"]["net_sharpe"] for r in ok]
    cost_gaps = [r["cost_gap"] for r in ok if "cost_gap" in r]
    n_pass = sum(1 for r in ok if r["gate_pass"])

    decision = "ship_linear_core"
    if ok:
        median_uplift = float(np.median(uplifts))
        median_rl = float(np.median(rl_sharpes))
        if median_uplift >= gate["min_uplift"] and median_rl >= gate["wf_net_sharpe_floor"]:
            decision = "rl_beats_linear"
    else:
        median_uplift = median_rl = None

    manifest = {
        "stage": stage,
        "strategy": config.get("strategy", {}),
        "status": status,
        "reason": reason,
        "gate_thresholds": gate,
        "gate_decision": decision,
        "n_windows": len(ok),
        "n_windows_gate_pass": n_pass,
        "median_uplift_net_sharpe": median_uplift,
        "median_rl_net_sharpe": median_rl,
        "median_linear_core_net_sharpe": float(np.median(core_sharpes)) if core_sharpes else None,
        "median_cost_gap": float(np.median(cost_gaps)) if cost_gaps else None,
        # g_diversification (corr to existing live sleeves) needs GMGP1/SG-1 live
        # return series — a Stage-3/deploy-gate integration, not wired in this
        # sim-only pipeline. Surfaced (not silently dropped) so the deploy gate
        # knows it is still owed.
        "diversification_gate": "DEFERRED — needs live-sleeve returns (Stage-3/deploy)",
        "windows": results,
    }
    path = out_dir / f"manifest_{stage}.json"
    path.write_text(json.dumps(manifest, indent=2,
                               default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x))
    logger.info("Manifest → %s  (status=%s decision=%s, %d/%d windows pass)",
                path, status, decision, n_pass, len(ok))
    return manifest


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Cross-asset momentum allocator pipeline")
    ap.add_argument("--config", default="configs/cross_asset_momentum.yaml")
    ap.add_argument("--stage", choices=["hpo", "wf"], default="hpo")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--max_windows", type=int, default=None)
    ap.add_argument("--n_trials", type=int, default=None)
    ap.add_argument("--hpo_steps", type=int, default=None)
    ap.add_argument("--train_steps", type=int, default=None)
    ap.add_argument("--force_refetch", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end wiring check (1 window, 2 trials, 1k steps)")
    ap.add_argument("--gpus", default=None,
                    help="comma GPU ids for parallel WF window-dispatch, e.g. '0,1'. "
                         "Deploy WITHOUT --gpu so all are visible; each window-worker pins one.")
    ap.add_argument("--windows", default=None,
                    help="comma window indices to run THIS process (cross-instance split), e.g. '0,1,2'")
    ap.add_argument("--aggregate_only", action="store_true",
                    help="skip running; build the WF manifest from collected w*_result.json in --out_dir")
    # Deploy compatibility (deploy_bare_metal.py injects these into the launch cmd)
    ap.add_argument("--run_name", default=None, help="WandB run name (deploy-injected)")
    ap.add_argument("--hpo_storage", default=None,
                    help="(accepted for deploy compat; ignored — WF uses per-window sqlite)")
    ap.add_argument("--tags", nargs="*", default=None, help="extra WandB tags (deploy-injected)")
    args = ap.parse_args()
    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""] if args.gpus else None
    windows = [int(x) for x in args.windows.split(",") if x.strip() != ""] if args.windows else None

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("optuna").setLevel(logging.WARNING)

    config = load_config(args.config)
    hpo_cfg = config.get("hpo", {})

    if args.smoke:
        n_trials, hpo_steps, train_steps, max_windows = 2, 1000, 2000, 1
    else:
        n_trials = args.n_trials or int(hpo_cfg.get("trials", 40))
        hpo_steps = args.hpo_steps or int(hpo_cfg.get("steps_per_trial", 500_000))
        train_steps = args.train_steps or int(config.get("training", {}).get("total_timesteps", 1_000_000))
        max_windows = args.max_windows

    out_dir = Path(args.out_dir or f"results/cross_asset_allocator/{args.stage}")
    logger.info("Pipeline: stage=%s trials=%d hpo_steps=%d train_steps=%d out=%s",
                args.stage, n_trials, hpo_steps, train_steps, out_dir)

    # WandB (optional) — liveness for monitor_fleet; never let it block the run.
    wb_cfg = config.get("wandb", {})
    base_tags = list(wb_cfg.get("tags", [])) + list(args.tags or []) + [f"stage-{args.stage}"]
    # 64-char tag guard (pydantic-validated; an over-limit tag silently crashes
    # wandb.init before the run row exists — see feedback_wandb_tag_64_char_limit).
    tags = [t for t in base_tags if len(str(t)) <= 64]
    try:
        import wandb
        if not os.environ.get("WANDB_DISABLED"):
            wandb.init(
                project=wb_cfg.get("project", "FinRL-Pro-DS"),
                entity=wb_cfg.get("entity"),
                name=args.run_name or f"xsec-allocator-{args.stage}",
                tags=tags,
                config={"stage": args.stage, "strategy": config.get("strategy", {}),
                        "n_trials": n_trials, "hpo_steps": hpo_steps,
                        "train_steps": train_steps},
                reinit=True,
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("WandB init skipped (%s) — proceeding without run tracking", e)

    try:
        manifest = run_pipeline(
            config, stage=args.stage, out_dir=out_dir, max_windows=max_windows,
            n_trials=n_trials, hpo_steps=hpo_steps, train_steps=train_steps,
            force_refetch=args.force_refetch, gpus=gpus, windows=windows,
            aggregate_only=args.aggregate_only)
    finally:
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except Exception:  # noqa: BLE001
            pass

    if manifest["status"] != "PASS":
        logger.error("Pipeline stage did not complete: %s", manifest.get("reason"))
        sys.exit(1)
    logger.info("Stage %s complete — decision: %s", args.stage, manifest["gate_decision"])


if __name__ == "__main__":
    main()
