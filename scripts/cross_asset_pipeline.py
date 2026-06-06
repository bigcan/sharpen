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
import sys
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


def _sac_search_space(trial) -> dict:
    return {
        "agent_params": {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "buffer_size": trial.suggest_int("buffer_size", 100_000, 1_000_000, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
            "gamma": trial.suggest_float("gamma", 0.95, 0.999),
            "tau": trial.suggest_float("tau", 0.001, 0.02, log=True),
            "learning_starts": trial.suggest_int("learning_starts", 1000, 10000, log=True),
        },
        "env_overrides": {
            "turnover_penalty": trial.suggest_float("turnover_penalty", 0.0005, 0.01, log=True),
        },
    }


def _run_hpo(train_arrays, val_arrays, config, n_trials, steps, net_arch) -> dict:
    """Optuna SAC HPO; objective = val net Sharpe (ADR-4). Returns best params."""
    import optuna

    n_envs = config.get("training", {}).get("num_envs", 8)

    def objective(trial):
        search = _sac_search_space(trial)
        try:
            vec_env = _make_vec_env(train_arrays, config, n_envs, search["env_overrides"])
            model = _make_sac(vec_env, search["agent_params"], net_arch)
            model.learn(total_timesteps=steps)
            m = _evaluate_model(model, val_arrays, config, search["env_overrides"])
            vec_env.close()
            del model, vec_env
            gc.collect()
            if m["n_steps"] < 50:
                return -999.0
            return m["net_sharpe"]
        except Exception as e:  # noqa: BLE001
            logger.error("HPO trial %d failed: %s", trial.number, e)
            gc.collect()
            return -999.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
        pruner=optuna.pruners.NopPruner(),
    )
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
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

    logger.info("=== Window %d: HPO (%d trials × %d steps) ===", w_idx, n_trials, hpo_steps)
    hpo = _run_hpo(train_arrays, val_arrays, config, n_trials, hpo_steps, net_arch)
    logger.info("  best val net Sharpe=%.3f trial=%d params=%s",
                hpo["best_value"], hpo["best_trial"], hpo["agent_params"])

    logger.info("=== Window %d: full train (%d steps) ===", w_idx, train_steps)
    n_envs = config.get("training", {}).get("num_envs", 8)
    vec_env = _make_vec_env(train_arrays, config, n_envs, hpo["env_overrides"])
    model = _make_sac(vec_env, hpo["agent_params"], net_arch)
    model.learn(total_timesteps=train_steps)
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
    logger.info("  Window %d: RL=%.3f  core=%.3f  uplift=%.3f  cost_gap=%.3f  gate=%s",
                w_idx, rl_test["net_sharpe"], core_test["net_sharpe"], uplift, cost_gap,
                "PASS" if window_pass else "fail")
    return result


# --------------------------------------------------------------------------- #
# Orchestrator + manifest
# --------------------------------------------------------------------------- #
def run_pipeline(config: dict, *, stage: str, out_dir: Path, max_windows: int | None,
                 n_trials: int, hpo_steps: int, train_steps: int,
                 force_refetch: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    gate = load_gate_thresholds(config)
    logger.info("Gate thresholds: %s", gate)

    data = loader.load_cross_asset_data(config, force_refetch=force_refetch)
    schedule = build_wf_schedule(data["close"].index, config["walk_forward"])
    if not schedule:
        return _write_manifest(out_dir, stage, config, gate, [], status="FAIL",
                               reason="insufficient_data_for_any_window")

    if stage == "hpo":
        schedule = schedule[:1]              # Stage-1 = window 0 decision artifact
    elif max_windows is not None:
        schedule = schedule[:max_windows]

    results = []
    for window in schedule:
        try:
            results.append(run_window(
                window["window"], window, data, config, n_trials, hpo_steps,
                train_steps, out_dir, gate))
        except Exception as e:  # noqa: BLE001
            logger.exception("Window %d FAILED", window["window"])
            results.append({"window": window["window"], "status": "FAILED", "error": str(e)})

    return _write_manifest(out_dir, stage, config, gate, results, status="PASS")


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
    args = ap.parse_args()

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

    manifest = run_pipeline(
        config, stage=args.stage, out_dir=out_dir, max_windows=max_windows,
        n_trials=n_trials, hpo_steps=hpo_steps, train_steps=train_steps,
        force_refetch=args.force_refetch)

    if manifest["status"] != "PASS":
        logger.error("Pipeline stage did not complete: %s", manifest.get("reason"))
        sys.exit(1)
    logger.info("Stage %s complete — decision: %s", args.stage, manifest["gate_decision"])


if __name__ == "__main__":
    main()
