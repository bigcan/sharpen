"""Factory + frozen-linear-core evaluator for the MultiAssetAllocatorEnv.

Phase 4 of the cross-sectional pivot (S553-cont-34). This is the **dispatch
concern** the resume note flagged: the cross-asset config declares an ``env:``
block whose keys do NOT all match the env constructor's kwargs (notably
``env.taker_fee`` → ``taker_fee_pct``). :func:`make_allocator_env` is the single
mapping layer (the allocator analog of ``crypto_backtest_runner.create_env``),
kept in the library (no ``scripts/`` dependency) so both the pipeline and the
tests construct the env identically.

It also owns the **frozen linear core** used by the RL-beats-linear gate
(``configs/cross_asset_momentum.gates.yaml``): :func:`evaluate_linear_core` drives
the env as a zero-RL allocator with the validated monthly-rebalanced TSMOM
conviction. Critically it uses the **env-native index convention** (the action at
step ``k`` governs the ``k→k+1`` move — exactly what ``model.predict(obs)`` does),
so the baseline and the RL are compared under identical env/convention/cost
conditions. Only the RL−baseline *difference* gates the deploy decision; the
absolute baseline (~0.5 net Sharpe through the env's fixed-notional accounting)
need not match the falsification's constant-weight 0.60 — that second-order
accounting gap is the keystone parity test's concern, not the gate's.
"""
from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import pandas as pd

from finrl_pro_ds.envs.multi_asset_allocator_env import MultiAssetAllocatorEnv

logger = logging.getLogger(__name__)

ANN = 252  # daily annualization

# config env: key  ->  MultiAssetAllocatorEnv kwarg. Where the names already
# match this is identity; the entries that differ (taker_fee→taker_fee_pct) are
# the reason this layer exists.
_ENV_KEY_MAP = {
    "initial_capital": "initial_capital",
    "taker_fee": "taker_fee_pct",            # <-- the rename
    "taker_fee_pct": "taker_fee_pct",        # tolerate the explicit form too
    "slippage_base_bps": "slippage_base_bps",
    "slippage_impact_bps": "slippage_impact_bps",
    "target_vol_asset": "target_vol_asset",
    "lev_cap": "lev_cap",
    "max_gross_exposure": "max_gross_exposure",
    "vol_floor": "vol_floor",
    "allow_short": "allow_short",
    "reward_type": "reward_type",
    "dsr_eta": "dsr_eta",
    "sortino_window": "sortino_window",
    "reward_scaling": "reward_scaling",
    "turnover_penalty": "turnover_penalty",
    "reward_clip_range": "reward_clip_range",
    "min_trade_pct": "min_trade_pct",
    "no_trade_band": "no_trade_band",
    "rebalance_interval": "rebalance_interval",
    "cost_penalty_scale": "cost_penalty_scale",
    "circuit_breaker_threshold": "circuit_breaker_threshold",
    "random_start": "random_start",
    "random_start_pct": "random_start_pct",
    "enable_trade_log": "enable_trade_log",
}

_BOOL_KW = {"allow_short", "random_start", "enable_trade_log"}
_TUPLE_KW = {"reward_clip_range"}

# v1.1 execution levers are RL trading discipline, NOT market conditions. The
# RL-beats-linear gate baseline must stay the VALIDATED linear core (monthly
# cadence, full-to-target trades) — these are forced off in evaluate_linear_core.
EXECUTION_LEVERS = {"no_trade_band", "rebalance_interval", "cost_penalty_scale"}
_EXECUTION_LEVERS_OFF = {"no_trade_band": 0.0, "rebalance_interval": 1,
                         "cost_penalty_scale": 0.0}


def make_allocator_env(
    arrays: Mapping,
    config: Mapping,
    *,
    overrides: Mapping | None = None,
    eval_mode: bool = False,
) -> MultiAssetAllocatorEnv:
    """Construct a :class:`MultiAssetAllocatorEnv` from prepared arrays + config.

    Args:
        arrays: output of ``cross_asset_loader.build_allocator_arrays`` (must carry
            ``price_ary, tech_ary, vol_ary, carry_ary, volume_ary, timestamps``).
        config: full experiment config; reads the ``env:`` block.
        overrides: per-trial env overrides (e.g. HPO ``turnover_penalty``) applied
            on top of the config ``env:`` block.
        eval_mode: when True, forces ``random_start=False`` (deterministic eval).
    """
    env_cfg = dict(config.get("env", {}))
    if overrides:
        env_cfg.update(overrides)

    kwargs: dict = {}
    for cfg_key, kw in _ENV_KEY_MAP.items():
        if cfg_key not in env_cfg:
            continue
        val = env_cfg[cfg_key]
        if kw in _BOOL_KW:
            val = bool(val)
        elif kw in _TUPLE_KW:
            val = tuple(val)
        kwargs[kw] = val  # last-writer wins (taker_fee then taker_fee_pct identity)

    if eval_mode:
        kwargs["random_start"] = False

    # `type` is the dispatch selector, not a constructor arg.
    return MultiAssetAllocatorEnv(
        price_ary=arrays["price_ary"],
        tech_ary=arrays["tech_ary"],
        vol_ary=arrays["vol_ary"],
        carry_ary=arrays["carry_ary"],
        volume_ary=arrays["volume_ary"],
        timestamps=arrays["timestamps"],
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Frozen linear core (RL-beats-linear gate baseline)
# --------------------------------------------------------------------------- #
def monthly_rebal_conviction(timestamps: np.ndarray, conviction_ary: np.ndarray) -> np.ndarray:
    """Step-function conviction that changes only on the last trading day of each
    month (the validated linear-core cadence). ``conviction_ary`` is the raw daily
    ``trend_conviction``; between rebalances the most recent month-end conviction is
    held. Causal: each month-end conviction is itself a causal signal."""
    ts = np.asarray(timestamps, dtype=np.int64)
    dates = pd.to_datetime(ts, unit="s")
    months = dates.to_period("M")
    T = len(ts)
    last_of_month = pd.Series(np.arange(T)).groupby(months.values).max().to_numpy()
    is_rebal = np.zeros(T, dtype=bool)
    is_rebal[last_of_month] = True

    held = np.zeros(conviction_ary.shape[1], dtype=np.float64)
    out = np.zeros_like(conviction_ary, dtype=np.float64)
    for t in range(T):
        if is_rebal[t]:
            held = conviction_ary[t]
        out[t] = held
    return out


def _drive(env: MultiAssetAllocatorEnv, action_at_step) -> tuple[dict, np.ndarray]:
    """Run ``env`` to termination, action chosen by ``action_at_step(step_idx, obs)``
    (env-native convention: action at step k governs the k→k+1 move). Returns
    ``(metrics, weights)``: ``metrics`` from the env's own step returns / turnover,
    and ``weights[k]`` the signed target weight held during the k→k+1 move
    (``info['position']`` after the step — post vol-scaling, availability-zeroing,
    and gross cap)."""
    obs, _ = env.reset()
    step_returns: list[float] = []
    turnovers: list[float] = []
    pvs: list[float] = [env.initial_capital]
    weights: list[np.ndarray] = []
    done = False
    while not done:
        k = env.step_idx
        action = action_at_step(k, obs)
        obs, _, terminated, truncated, info = env.step(action)
        step_returns.append(info["step_return"])
        turnovers.append(info["turnover"])
        pvs.append(info["portfolio_value"])
        weights.append(info["position"])
        done = terminated or truncated
    return _metrics(step_returns, turnovers, pvs), np.asarray(weights, dtype=np.float64)


def _metrics(step_returns: list[float], turnovers: list[float], pvs: list[float]) -> dict:
    r = np.asarray(step_returns, dtype=np.float64)
    pv = np.asarray(pvs, dtype=np.float64)
    # ddof=1 (Bessel) for both Sharpe & Sortino — consistent with crypto_perp_env /
    # env._sortino_reward. (The gate is a Sharpe *difference*, so ddof cancels there;
    # consistency matters for the reported absolute numbers.)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    sharpe = float(r.mean() / sd * np.sqrt(ANN)) if sd > 1e-12 else 0.0
    # Standard downside deviation: RMS of min(r, target=0) over ALL returns (n-1),
    # matching MultiAssetAllocatorEnv._sortino_reward — NOT the std of only the
    # negative returns (which measures their spread about their own mean).
    dd_dev = float(np.sqrt(np.sum(np.minimum(r, 0.0) ** 2) / max(len(r) - 1, 1))) if len(r) > 1 else 0.0
    sortino = float(r.mean() / dd_dev * np.sqrt(ANN)) if dd_dev > 1e-12 else 0.0
    peak = np.maximum.accumulate(pv)
    max_dd = float((1.0 - pv / np.where(peak <= 0, 1.0, peak)).max()) if len(pv) else 0.0
    years = max(len(r) / ANN, 1e-9)
    return {
        "net_sharpe": sharpe,
        "net_sortino": sortino,
        "total_return": float(pv[-1] / pv[0] - 1.0) if len(pv) > 1 else 0.0,
        "max_drawdown": max_dd,
        "turnover_ann": float(np.sum(turnovers) / years),
        "n_steps": int(len(r)),
    }


def _linear_core_drive(
    arrays: Mapping,
    config: Mapping,
    overrides: Mapping | None,
) -> tuple[dict, np.ndarray]:
    """Shared frozen-linear-core env drive behind :func:`evaluate_linear_core` (the
    gate metrics) and :func:`linear_core_weights` (the executor's weight trajectory).
    Keeping ONE drive path guarantees the executor's target weights are byte-identical
    to the gate baseline (ADR-7) — rung-1 paper-sim parity is ≈0 by construction.

    Drives the env (real config costs, ``eval_mode``) with the monthly-rebalanced
    ``conviction_ary`` under the env-native convention. Forces the v1.1 execution
    levers OFF regardless of config/overrides: the baseline is the validated monthly
    linear core, not "linear core through the RL's trading discipline" (a 5-bar
    ``rebalance_interval`` would even block the monthly cadence whenever month-end
    falls off-cadence). Requires ``arrays['conviction_ary']`` (from
    ``build_allocator_arrays``).
    """
    if "conviction_ary" not in arrays:
        raise KeyError("the linear-core drive needs arrays['conviction_ary'] "
                       "(build_allocator_arrays output)")
    conv_monthly = monthly_rebal_conviction(arrays["timestamps"], arrays["conviction_ary"])
    core_overrides = {**(overrides or {}), **_EXECUTION_LEVERS_OFF}
    env = make_allocator_env(arrays, config, overrides=core_overrides, eval_mode=True)
    return _drive(env, lambda k, obs: conv_monthly[k])


def evaluate_linear_core(
    arrays: Mapping,
    config: Mapping,
    *,
    overrides: Mapping | None = None,
) -> dict:
    """Frozen linear-core net metrics through the env (the gate baseline).

    Drives the env (real config costs, ``eval_mode``) with the monthly-rebalanced
    ``conviction_ary`` under the env-native convention — identical conditions to the
    RL eval, so the RL−baseline uplift is well-posed. Requires
    ``arrays['conviction_ary']`` (from ``build_allocator_arrays``).
    """
    metrics, _ = _linear_core_drive(arrays, config, overrides)
    return metrics


def linear_core_weights(
    arrays: Mapping,
    config: Mapping,
    *,
    overrides: Mapping | None = None,
) -> np.ndarray:
    """Per-step target-weight trajectory of the frozen linear core — the paper
    executor's weight source (ADR-7; resolves spec Open-Item-1 toward the env path).

    Returns ``w`` of shape ``(n_steps, N)`` where ``w[k]`` is the signed weight the
    core holds during the k→k+1 move (vol-scaled monthly conviction, capped). Because
    it shares :func:`_linear_core_drive` with :func:`evaluate_linear_core`, the
    executor's target weights are byte-identical to the RL-beats-linear gate baseline
    and rung-1 paper-sim parity is ≈0 by construction. ``w[-1]`` is the weight to hold
    going forward from the most recent bar (the live order-generation target).
    """
    _, weights = _linear_core_drive(arrays, config, overrides)
    return weights
