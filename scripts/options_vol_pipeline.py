"""Options-VRP harvester pipeline — Protocol-v2 staged runner (Phase 5).

Phase 5 of the market-making redesign (S553-cont-37). Orchestrates the RL
``OptionsVolHarvestEnv`` (Phase 3) over the validated Phase-1 linear short-straddle
core (``results/options_vrp/verdict.json``, BTC per-asset net Sharpe **1.10**):

    data-prep (deribit_options_loader → options_array_builder)
      →  per-window SAC HPO (objective = val net Sharpe, ADR-5)
      →  full train  →  RL net-Sharpe eval  →  FROZEN linear-core (neutral action) eval
      →  rl_beats_linear + cost-gap + tail gates  →  Protocol-v2 stage manifest

The decisive gate (``configs/options_vol_harvest.gates.yaml``): the RL must beat the
frozen linear short-vol core OOS by ``rl_beats_linear.min_uplift_vs_baseline`` net
Sharpe, evaluated through the SAME env/convention/costs (the neutral action
``[neutral_conviction, 0.0]`` reproduces the linear core bit-for-bit — that is the
keystone parity guarantee, ``tests/envs/test_options_vol_harvest.py``) — else we
**ship the linear rule** (RL = thin overlay, ADR-3/ADR-4).

Annualization is **365** (crypto 24/7 daily bars) — the same ANN the Phase-1
falsification and ``options_pricing`` use, so the RL net Sharpe is on the identical
scale as the documented 1.10 core.

Stages (Protocol v2):
    --stage hpo   window 0 only: HPO + train + eval + gate (Stage-1 decision artifact)
    --stage wf    all walk-forward windows (Stage-3 deploy-gating verdict)

CPU-testable, no-GPU pieces (no SB3 import at module load): the WF scheduler, the
gate logic, the manifest writer, the config→env mapping, and the **linear-core
evaluation** (drives the env with the neutral action — no model needed). Run them
with ``--linear_only`` to verify wiring + emit the per-window baselines the RL must
beat, without SB3/GPU. The SAC HPO/train is the only GPU step; SB3/Optuna are
lazy-imported.

Usage:
    # No-GPU: linear-core baselines over each WF window + full series (writes
    # linear_baseline.json) — reproduces the documented BTC 1.10 core when skip_bars=0.
    python scripts/options_vol_pipeline.py --config configs/options_vol_harvest.yaml --linear_only
    # Smoke (1 window, 2 trials, 1k steps) — end-to-end SB3 wiring check (needs SB3)
    python scripts/options_vol_pipeline.py --config configs/options_vol_harvest.yaml --smoke
    # Stage-1 HPO (window 0)  /  Stage-3 walk-forward (GPU instance w/ SB3)
    python scripts/options_vol_pipeline.py --config configs/options_vol_harvest.yaml --stage hpo
    python scripts/options_vol_pipeline.py --config configs/options_vol_harvest.yaml --stage wf
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from finrl_pro_ds.crypto.data import deribit_options_loader as dol  # noqa: E402
from finrl_pro_ds.crypto.data import options_array_builder as oab  # noqa: E402
from finrl_pro_ds.crypto.envs.options_vol_harvest_env import OptionsVolHarvestEnv  # noqa: E402

logger = logging.getLogger("options_vol_pipeline")

ANN = 365.0  # crypto 24/7 — matches options_pricing.ANN and the Phase-1 falsification

# config env: keys → OptionsVolHarvestEnv constructor kwargs (the config was authored
# for this env so the names match 1:1; ``type`` is the dispatch selector, not a kwarg).
_ENV_KWARGS = {
    "initial_capital", "base_premium_frac", "max_premium_frac", "roll_days",
    "entry_tenor_days", "option_fee_pct_underlying", "option_fee_cap_pct_premium",
    "perp_taker_fee", "option_spread_vol_pts", "allow_long_vol", "hedge_residual_gain",
    "hedge_frac_max", "max_gross_premium_frac", "max_net_vega_pct", "reward_type",
    "dsr_eta", "cvar_alpha", "cvar_penalty", "cvar_window", "turnover_penalty",
    "reward_scaling", "reward_clip_range", "random_start", "random_start_pct",
}
_ENV_OVERRIDE_KEYS = {"turnover_penalty", "cvar_penalty"}  # HPO-searched reward-shaping
# costs zeroed for the frictionless probe (g_cost_gap / AlphaSeek artifact detector)
_FRICTIONLESS_OVERRIDES = {
    "option_fee_pct_underlying": 0.0, "option_fee_cap_pct_premium": 0.0,
    "perp_taker_fee": 0.0, "option_spread_vol_pts": 0.0,
}


# --------------------------------------------------------------------------- #
# Config / gates
# --------------------------------------------------------------------------- #
def load_config(path: str) -> dict:
    """Load a YAML config (utf-8 — the configs carry box-drawing chars; cp950 default
    on Windows crashes plain reads)."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_gate_thresholds(config: dict) -> dict:
    """Read the decision gates from the gates overlay file (gates live in configs,
    never hardcoded — CLAUDE.md invariant)."""
    gates_file = config.get("ensemble", {}).get("gates_file")
    defaults = {"min_uplift": 0.10, "wf_net_sharpe_floor": 0.50,
                "max_cost_gap": 0.90, "max_worst_window_dd_pct": 25.0,
                "max_corr_to_existing_sleeves": 0.30}
    if not gates_file:
        logger.warning("no ensemble.gates_file — using gate defaults %s", defaults)
        return defaults
    p = (PROJECT_ROOT / gates_file) if not Path(gates_file).is_absolute() else Path(gates_file)
    g = yaml.safe_load(p.read_text(encoding="utf-8")).get("gates", {})
    return {
        "min_uplift": float(g.get("rl_beats_linear", {}).get("min_uplift_vs_baseline", 0.10)),
        "wf_net_sharpe_floor": float(g.get("wf_net_sharpe_floor", 0.50)),
        "max_cost_gap": float(g.get("g_cost_gap", {}).get("max_frictionless_minus_net_sharpe", 0.90)),
        "max_worst_window_dd_pct": float(g.get("tail", {}).get("max_worst_window_dd_pct", 25.0)),
        "max_corr_to_existing_sleeves":
            float(g.get("g_diversification", {}).get("max_corr_to_existing_sleeves", 0.30)),
    }


# --------------------------------------------------------------------------- #
# Data → per-asset arrays
# --------------------------------------------------------------------------- #
def load_arrays(config: dict, *, refresh: bool = False) -> dict:
    """Load Deribit panels and extract the single configured asset's 1-D arrays.

    Returns ``{spot, iv, iv_rv_spread, rv_ref, funding, timestamps, dates, asset}``.
    """
    raw = dol.load(config, refresh=refresh)
    panels = oab.build_panels(raw, config)
    assets = config.get("universe", {}).get("assets", panels.assets)
    asset = assets[0]
    if asset not in panels.assets:
        raise ValueError(f"asset {asset!r} not in panels.assets={panels.assets}")
    i = panels.assets.index(asset)
    ref_w = int(config.get("features", {}).get("iv_rv_ref_window", 30))
    rv = panels.rv_ary.get(ref_w)
    if rv is None:
        rv = next(iter(panels.rv_ary.values()))
    return {
        "spot": panels.spot_ary[:, i],
        "iv": panels.iv_ary[:, i],
        "iv_rv_spread": panels.iv_rv_spread_ary[:, i],
        "rv_ref": rv[:, i],
        "funding": panels.funding_ary[:, i],
        "timestamps": panels.timestamps,
        "dates": panels.dates,
        "asset": asset,
    }


_SLICEABLE = ("spot", "iv", "iv_rv_spread", "rv_ref", "funding", "timestamps")


def slice_arrays(full: dict, s: int, e: int) -> dict:
    """Slice the sliceable 1-D arrays to ``[s:e]`` (compute-on-full-then-slice: the
    realized-vol warmup at a window start is drawn from the immediately-preceding
    (causal, ≤t) bars — never future data, and the WF embargo separates train↔test)."""
    return {k: full[k][s:e] for k in _SLICEABLE}


# --------------------------------------------------------------------------- #
# Walk-forward scheduler (daily bars; embargo gaps = LEAK guard)
# --------------------------------------------------------------------------- #
def build_wf_schedule(dates, wf: dict) -> list[dict]:
    """Slice the daily index into [train | embargo | val | embargo | test] windows
    by bar count. Drops (and logs) any trailing partial window — never silently
    truncates coverage."""
    train, val, test = int(wf["train_bars"]), int(wf["val_bars"]), int(wf["test_bars"])
    step, embargo = int(wf["step_bars"]), int(wf.get("embargo_bars", 0))
    n = len(dates)
    windows, start, w = [], 0, 0
    last_te_e = 0
    while True:
        tr_s, tr_e = start, start + train
        va_s, va_e = tr_e + embargo, tr_e + embargo + val
        te_s, te_e = va_e + embargo, va_e + embargo + test
        if te_e > n:
            break
        last_te_e = te_e
        windows.append({
            "window": w, "tr_s": tr_s, "tr_e": tr_e, "va_s": va_s, "va_e": va_e,
            "te_s": te_s, "te_e": te_e,
            "train": [str(dates[tr_s].date()), str(dates[tr_e - 1].date())],
            "val": [str(dates[va_s].date()), str(dates[va_e - 1].date())],
            "test": [str(dates[te_s].date()), str(dates[te_e - 1].date())],
        })
        start += step
        w += 1
    logger.info("WF schedule: %d windows from %d bars (train=%d val=%d test=%d step=%d embargo=%d)",
                len(windows), n, train, val, test, step, embargo)
    if windows:
        dropped = n - last_te_e
        if dropped > 0:
            logger.info("WF: %d trailing bars after last window's test_end dropped (partial window)", dropped)
    return windows


# --------------------------------------------------------------------------- #
# Env construction + metrics + rollout (NO SB3 — usable on CPU/no-GPU)
# --------------------------------------------------------------------------- #
def make_options_env(arrays: dict, config: dict, *, overrides: dict | None = None,
                     eval_mode: bool = False) -> OptionsVolHarvestEnv:
    """Construct an :class:`OptionsVolHarvestEnv` from sliced arrays + config ``env:``."""
    env_cfg = {k: v for k, v in config.get("env", {}).items() if k in _ENV_KWARGS}
    if overrides:
        env_cfg.update({k: v for k, v in overrides.items() if k in _ENV_KWARGS})
    if "reward_clip_range" in env_cfg:
        env_cfg["reward_clip_range"] = tuple(env_cfg["reward_clip_range"])
    if eval_mode:
        env_cfg["random_start"] = False
    return OptionsVolHarvestEnv(
        spot=arrays["spot"], iv=arrays["iv"], iv_rv_spread=arrays["iv_rv_spread"],
        rv_ref=arrays["rv_ref"], funding=arrays["funding"], timestamps=arrays["timestamps"],
        **env_cfg)


# Metric helpers — byte-identical convention to options_vrp_falsification so the
# linear-core absolute reproduces results/options_vrp/verdict.json (BTC 1.10).
def _returns_from_pnl(daily_pnl: np.ndarray, eq_curve: np.ndarray) -> np.ndarray:
    prev = np.concatenate([[eq_curve[0]], eq_curve[:-1]])
    prev = np.where(prev <= 0, np.nan, prev)
    return daily_pnl / prev


def _sharpe(daily_ret: np.ndarray) -> float:
    d = daily_ret[np.isfinite(daily_ret)]
    if len(d) < 2 or d.std(ddof=1) == 0:
        return 0.0
    return float(d.mean() / d.std(ddof=1) * math.sqrt(ANN))


def _profit_factor(daily_pnl: np.ndarray) -> float:
    pos = daily_pnl[daily_pnl > 0].sum()
    neg = -daily_pnl[daily_pnl < 0].sum()
    if neg == 0:
        return float("inf") if pos > 0 else 0.0
    return float(pos / neg)


def _max_drawdown(eq: np.ndarray) -> float:
    peak = np.maximum.accumulate(eq)
    return float((1.0 - eq / peak).max())


def _drive(env: OptionsVolHarvestEnv, action_fn) -> dict:
    """Roll ``env`` to termination, action chosen by ``action_fn(obs)``. Reconstructs
    the (daily_pnl, eq_curve) in the EXACT convention of the falsification
    (daily_pnl[0] = -initial_open_cost) so the metrics are directly comparable."""
    obs, info = env.reset()
    daily_pnl = [-info["initial_open_cost"]]
    eq_curve = [env.initial_capital]
    done = False
    while not done:
        action = action_fn(obs)
        obs, _, terminated, truncated, info = env.step(action)
        daily_pnl.append(info["step_pnl"])
        eq_curve.append(info["portfolio_value"])
        done = terminated or truncated
    dp = np.asarray(daily_pnl, dtype=np.float64)
    ec = np.asarray(eq_curve, dtype=np.float64)
    ret = _returns_from_pnl(dp, ec)
    return {
        "net_sharpe": _sharpe(ret),
        "net_pf": _profit_factor(dp),
        "net_max_dd": _max_drawdown(ec),
        "total_return": float(ec[-1] / ec[0] - 1.0),
        "n_steps": int(len(dp) - 1),
    }


def evaluate_linear_core(arrays: dict, config: dict, *, overrides: dict | None = None) -> dict:
    """Frozen linear-core net metrics through the env (the rl_beats_linear baseline).

    Drives the env with the NEUTRAL action ``[neutral_conviction, 0.0]`` — identical
    conditions to the RL eval, so the RL−baseline uplift is well-posed. (Reward-shaping
    overrides don't affect step_return/step_pnl, so the core is invariant to them; we
    pass them only to keep construction byte-identical to the RL env.)"""
    env = make_options_env(arrays, config, overrides=overrides, eval_mode=True)
    neutral = np.array([env.neutral_conviction, 0.0], dtype=np.float64)
    return _drive(env, lambda obs: neutral)


def evaluate_model(model, arrays: dict, config: dict, *, overrides: dict | None = None) -> dict:
    """RL net metrics via deterministic env-native rollout (same env/convention/costs
    as ``evaluate_linear_core`` → fair gate)."""
    env = make_options_env(arrays, config, overrides=overrides, eval_mode=True)
    return _drive(env, lambda obs: model.predict(obs, deterministic=True)[0])


# --------------------------------------------------------------------------- #
# WandB (optional; liveness for monitor_fleet — avoids the false-CRIT stall)
# --------------------------------------------------------------------------- #
def _wandb_log(metrics: dict) -> None:
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics)
    except Exception:  # noqa: BLE001
        pass


def _heartbeat_callback(label: str, log_every: int = 10_000):
    """SB3 callback that WandB-logs step/SPS periodically so monitor_fleet sees the
    run is alive (the project's documented false-CRIT silent-stall guard)."""
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


# --------------------------------------------------------------------------- #
# SB3 SAC (lazy) — HPO + training
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
        return make_options_env(a, c, overrides=o)
    return DummyVecEnv([_fn for _ in range(n_envs)])


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
            # reward-shaping regularizers only — the HPO objective is the INDEPENDENT
            # net Sharpe (not the shaped reward), so this does not violate BUG-01's
            # "lock reward params" intent (cf cross_asset_pipeline, Audit PASS).
            "turnover_penalty": trial.suggest_float("turnover_penalty", 0.0005, 0.01, log=True),
            "cvar_penalty": trial.suggest_float("cvar_penalty", 0.1, 1.0),
        },
    }


def _run_hpo(train_arrays, val_arrays, config, n_trials, steps, net_arch,
             *, storage: str | None = None, study_name: str = "vrp_hpo") -> dict:
    """Optuna SAC HPO; objective = val net Sharpe (ADR-5). Returns best params.

    ``storage`` (deploy-injected sqlite URL) makes the study resumable — already
    completed trials are skipped on restart."""
    import optuna

    n_envs = config.get("training", {}).get("num_envs", 8)

    def objective(trial):
        search = _sac_search_space(trial)
        try:
            vec_env = _make_vec_env(train_arrays, config, n_envs, search["env_overrides"])
            model = _make_sac(vec_env, search["agent_params"], net_arch)
            model.learn(total_timesteps=steps,
                        callback=_heartbeat_callback(f"hpo_t{trial.number}"))
            m = evaluate_model(model, val_arrays, config, overrides=search["env_overrides"])
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
    return {
        "best_value": float(best.value) if best.value is not None else -999.0,
        "agent_params": {k: v for k, v in best.params.items() if k not in _ENV_OVERRIDE_KEYS},
        "env_overrides": {k: v for k, v in best.params.items() if k in _ENV_OVERRIDE_KEYS},
        "best_trial": best.number,
    }


# --------------------------------------------------------------------------- #
# Per-window: HPO → train → RL eval → linear-core eval → gate
# --------------------------------------------------------------------------- #
def run_window(window, data, config, n_trials, hpo_steps, train_steps,
               out_dir: Path, gate: dict, *, hpo_storage: str | None = None) -> dict:
    w_idx = window["window"]
    net_arch = config.get("agents", {}).get("sac", {}).get("network_arch", [256, 256])

    train_arrays = slice_arrays(data, window["tr_s"], window["tr_e"])
    val_arrays = slice_arrays(data, window["va_s"], window["va_e"])
    test_arrays = slice_arrays(data, window["te_s"], window["te_e"])

    logger.info("=== Window %d: HPO (%d trials × %d steps) ===", w_idx, n_trials, hpo_steps)
    hpo = _run_hpo(train_arrays, val_arrays, config, n_trials, hpo_steps, net_arch,
                   storage=hpo_storage, study_name=f"vrp_w{w_idx}")
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
    rl_test = evaluate_model(model, test_arrays, config, overrides=hpo["env_overrides"])
    core_test = evaluate_linear_core(test_arrays, config, overrides=hpo["env_overrides"])

    # g_cost_gap (AlphaSeek artifact detector): frictionless RL minus net RL Sharpe.
    # Zero the trading frictions (option fee + spread + perp taker) AND the funding
    # carry (funding=0 arrays) — matching the falsification's frictionless definition,
    # so the gap is comparable to the documented BTC frictionless-vs-net 0.76.
    fric_arrays = {**test_arrays, "funding": np.zeros_like(test_arrays["funding"])}
    fric_over = {**hpo["env_overrides"], **_FRICTIONLESS_OVERRIDES}
    rl_frictionless = evaluate_model(model, fric_arrays, config, overrides=fric_over)
    cost_gap = rl_frictionless["net_sharpe"] - rl_test["net_sharpe"]
    cost_gap_pass = cost_gap <= gate["max_cost_gap"]

    uplift = rl_test["net_sharpe"] - core_test["net_sharpe"]
    tail_pass = rl_test["net_max_dd"] * 100.0 <= gate["max_worst_window_dd_pct"]
    window_pass = (uplift >= gate["min_uplift"]) \
        and (rl_test["net_sharpe"] >= gate["wf_net_sharpe_floor"]) \
        and cost_gap_pass and tail_pass

    del model
    gc.collect()

    result = {
        "window": w_idx,
        "train": window["train"], "val": window["val"], "test": window["test"],
        "hpo_best_val_sharpe": hpo["best_value"],
        "hpo_agent_params": hpo["agent_params"],
        "hpo_env_overrides": hpo["env_overrides"],
        "rl_test": rl_test,
        "linear_core_test": core_test,
        "rl_frictionless_net_sharpe": rl_frictionless["net_sharpe"],
        "uplift_net_sharpe": float(uplift),
        "cost_gap": float(cost_gap),
        "cost_gap_pass": bool(cost_gap_pass),
        "tail_pass": bool(tail_pass),
        "gate_pass": bool(window_pass),
        "model_path": str(model_path),
    }
    (out_dir / f"w{w_idx}_result.json").write_text(json.dumps(result, indent=2))
    _wandb_log({
        f"window/{w_idx}/rl_net_sharpe": rl_test["net_sharpe"],
        f"window/{w_idx}/linear_core_net_sharpe": core_test["net_sharpe"],
        f"window/{w_idx}/uplift_net_sharpe": uplift,
        f"window/{w_idx}/cost_gap": cost_gap,
        f"window/{w_idx}/rl_net_max_dd": rl_test["net_max_dd"],
        f"window/{w_idx}/gate_pass": int(window_pass),
    })
    logger.info("  Window %d: RL=%.3f  core=%.3f  uplift=%.3f  cost_gap=%.3f  DD=%.1f%%  gate=%s",
                w_idx, rl_test["net_sharpe"], core_test["net_sharpe"], uplift, cost_gap,
                rl_test["net_max_dd"] * 100.0, "PASS" if window_pass else "fail")
    return result


# --------------------------------------------------------------------------- #
# Linear-only mode (no SB3/GPU): per-window baselines + full-series core
# --------------------------------------------------------------------------- #
def run_linear_only(config: dict, out_dir: Path, max_windows: int | None) -> dict:
    """Compute the frozen linear-core net metrics over each WF test window AND the
    full series (no SAC). Verifies the data→panels→env→metrics wiring and emits the
    per-window baselines the RL must beat — the full-series core reproduces the
    documented BTC 1.10 when features.skip_bars=0."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_arrays(config)
    full_arrays = {k: data[k] for k in _SLICEABLE}
    full_core = evaluate_linear_core(full_arrays, config)
    logger.info("Full-series linear core: net Sharpe=%.3f PF=%.3f DD=%.2f%% ret=%.1f%% (n=%d)",
                full_core["net_sharpe"], full_core["net_pf"], full_core["net_max_dd"] * 100.0,
                full_core["total_return"] * 100.0, full_core["n_steps"])

    schedule = build_wf_schedule(data["dates"], config["walk_forward"])
    if max_windows is not None:
        schedule = schedule[:max_windows]
    per_window = []
    for window in schedule:
        test_arrays = slice_arrays(data, window["te_s"], window["te_e"])
        core = evaluate_linear_core(test_arrays, config)
        per_window.append({"window": window["window"], "test": window["test"], **core})
        logger.info("  Window %d test %s..%s: core net Sharpe=%.3f PF=%.3f DD=%.2f%%",
                    window["window"], window["test"][0], window["test"][1],
                    core["net_sharpe"], core["net_pf"], core["net_max_dd"] * 100.0)

    sharpes = [w["net_sharpe"] for w in per_window]
    out = {
        "mode": "linear_only",
        "asset": data["asset"],
        "skip_bars": int(config.get("features", {}).get("skip_bars", 0)),
        "date_range": [str(data["dates"][0].date()), str(data["dates"][-1].date())],
        "full_series_core": full_core,
        "n_windows": len(per_window),
        "median_window_core_net_sharpe": float(np.median(sharpes)) if sharpes else None,
        "windows": per_window,
    }
    path = out_dir / "linear_baseline.json"
    path.write_text(json.dumps(out, indent=2,
                               default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x))
    logger.info("Linear baseline -> %s  (full-series Sharpe=%.3f, median-window Sharpe=%s)",
                path, full_core["net_sharpe"],
                f"{out['median_window_core_net_sharpe']:.3f}" if sharpes else "n/a")
    return out


# --------------------------------------------------------------------------- #
# Orchestrator + manifest
# --------------------------------------------------------------------------- #
def run_pipeline(config: dict, *, stage: str, out_dir: Path, max_windows: int | None,
                 n_trials: int, hpo_steps: int, train_steps: int,
                 force_refetch: bool = False, hpo_storage: str | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    gate = load_gate_thresholds(config)
    logger.info("Gate thresholds: %s", gate)

    data = load_arrays(config, refresh=force_refetch)
    schedule = build_wf_schedule(data["dates"], config["walk_forward"])
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
                window, data, config, n_trials, hpo_steps, train_steps,
                out_dir, gate, hpo_storage=hpo_storage))
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
    worst_dds = [r["rl_test"]["net_max_dd"] for r in ok]
    n_pass = sum(1 for r in ok if r["gate_pass"])

    # A stage that scheduled windows but completed NONE is a failure, not a PASS
    # with an empty verdict (a silent "0/0 windows pass" must not read as healthy).
    if results and not ok:
        status = "FAIL"
        reason = reason or "all_windows_failed"

    # min_windows is a Protocol-v2 WF-validity gate; the hpo stage is a single-window
    # decision artifact by design, so it is exempt.
    min_windows = int(config.get("walk_forward", {}).get("min_windows", 0))
    min_windows_met = (stage != "wf") or (len(ok) >= min_windows)

    median_uplift: float | None = None
    median_rl: float | None = None
    median_cost_gap: float | None = float(np.median(cost_gaps)) if cost_gaps else None
    worst_window_dd_pct: float | None = None
    decision = "ship_linear_core"
    if ok:
        median_uplift = float(np.median(uplifts))
        median_rl = float(np.median(rl_sharpes))
        worst_window_dd_pct = float(max(worst_dds) * 100.0) if worst_dds else None
        tail_ok = (worst_window_dd_pct is None) or (worst_window_dd_pct <= gate["max_worst_window_dd_pct"])
        cost_ok = (median_cost_gap is None) or (median_cost_gap <= gate["max_cost_gap"])
        if (median_uplift >= gate["min_uplift"] and median_rl >= gate["wf_net_sharpe_floor"]
                and tail_ok and cost_ok and min_windows_met):
            decision = "rl_beats_linear"

    manifest = {
        "stage": stage,
        "strategy": config.get("strategy", {}),
        "status": status,
        "reason": reason,
        "annualization": ANN,
        "gate_thresholds": gate,
        "gate_decision": decision,
        "n_windows": len(ok),
        "n_windows_gate_pass": n_pass,
        "min_windows_required": min_windows,
        "min_windows_met": bool(min_windows_met),
        "median_uplift_net_sharpe": median_uplift,
        "median_rl_net_sharpe": median_rl,
        "median_linear_core_net_sharpe": float(np.median(core_sharpes)) if core_sharpes else None,
        "median_cost_gap": median_cost_gap,
        "worst_window_rl_dd_pct": worst_window_dd_pct,
        # g_diversification (corr to live GMGP1/SG-1 sleeves) needs live return series —
        # a Stage-3/deploy-gate integration, not wired in this sim-only pipeline.
        # Surfaced (not silently dropped) so the deploy gate knows it is still owed.
        "diversification_gate": "DEFERRED — needs live-sleeve returns (Stage-3/deploy)",
        "windows": results,
    }
    path = out_dir / f"manifest_{stage}.json"
    path.write_text(json.dumps(manifest, indent=2,
                               default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x))
    logger.info("Manifest -> %s  (status=%s decision=%s, %d/%d windows pass)",
                path, status, decision, n_pass, len(ok))
    return manifest


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Options-VRP harvester pipeline (Protocol v2)")
    ap.add_argument("--config", default="configs/options_vol_harvest.yaml")
    ap.add_argument("--stage", choices=["hpo", "wf"], default="hpo")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--max_windows", type=int, default=None)
    ap.add_argument("--n_trials", type=int, default=None)
    ap.add_argument("--hpo_steps", type=int, default=None)
    ap.add_argument("--train_steps", type=int, default=None)
    ap.add_argument("--force_refetch", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end SB3 wiring check (1 window, 2 trials, 1k steps)")
    ap.add_argument("--linear_only", action="store_true",
                    help="no-SB3/GPU: linear-core baselines per WF window + full series")
    # Deploy compatibility (deploy_bare_metal.py injects these into the launch cmd)
    ap.add_argument("--run_name", default=None, help="WandB run name (deploy-injected)")
    ap.add_argument("--hpo_storage", default=None,
                    help="Optuna sqlite storage URL (deploy-injected; makes HPO resumable)")
    ap.add_argument("--tags", nargs="*", default=None, help="extra WandB tags (deploy-injected)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("optuna").setLevel(logging.WARNING)

    config = load_config(args.config)
    hpo_cfg = config.get("hpo", {})

    if args.linear_only:
        out_dir = Path(args.out_dir or "results/options_vrp_pipeline/linear_only")
        run_linear_only(config, out_dir, args.max_windows)
        return

    if args.smoke:
        n_trials, hpo_steps, train_steps, max_windows = 2, 1000, 2000, 1
    else:
        n_trials = args.n_trials or int(hpo_cfg.get("trials", 40))
        hpo_steps = args.hpo_steps or int(hpo_cfg.get("steps_per_trial", 500_000))
        train_steps = args.train_steps or int(config.get("training", {}).get("total_timesteps", 1_000_000))
        max_windows = args.max_windows

    out_dir = Path(args.out_dir or f"results/options_vrp_pipeline/{args.stage}")
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
                name=args.run_name or f"vrp-harvest-{args.stage}",
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
            force_refetch=args.force_refetch, hpo_storage=args.hpo_storage)
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
