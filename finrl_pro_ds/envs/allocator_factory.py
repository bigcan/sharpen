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
    held. Causal: each month-end conviction is itself a causal signal.

    **Truncation caveat (Tier-2 audit 2026-06-14, P2-01).** ``last_of_month`` is the max
    index *within the given window*, so on a window truncated mid-month the in-progress
    FINAL bar is flagged as a month-end and ``out[-1]`` is the partial-month raw daily
    conviction. The batch drive is UNAFFECTED — its last decision is at index ``T-2``, which
    reads only confirmed interior month-ends; the final-bar flag is never consumed. But a
    forward / incremental reader MUST NOT take ``monthly_rebal_conviction(window)[-1]`` as the
    live target: use :func:`linear_core_weights` ``w[-1]`` (which lags to the last CONFIRMED
    month-end) or an independent true-month-end calendar. Reading the in-progress tail is an
    ADR-2 "daily is a different strategy" / X2-class look-ahead trap."""
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


def _drive(env: MultiAssetAllocatorEnv, action_at_step) -> tuple[dict, np.ndarray, dict]:
    """Run ``env`` to termination, action chosen by ``action_at_step(step_idx, obs)``
    (env-native convention: action at step k governs the k→k+1 move). Returns
    ``(metrics, weights, trajectory)``: ``metrics`` from the env's own step returns /
    turnover; ``weights[k]`` the signed target weight held during the k→k+1 move
    (``info['position']`` after the step — post vol-scaling, availability-zeroing,
    and gross cap); ``trajectory`` the per-step series (equity curve, step returns,
    turnover, running fees) the paper executor's parity harness consumes as the sim
    oracle (ADR-7)."""
    obs, _ = env.reset()
    step_returns: list[float] = []
    turnovers: list[float] = []
    pvs: list[float] = [env.initial_capital]
    fees: list[float] = []
    weights: list[np.ndarray] = []
    done = False
    while not done:
        k = env.step_idx
        action = action_at_step(k, obs)
        obs, _, terminated, truncated, info = env.step(action)
        step_returns.append(info["step_return"])
        turnovers.append(info["turnover"])
        pvs.append(info["portfolio_value"])
        fees.append(info["cumulative_fees"])
        weights.append(info["position"])
        done = terminated or truncated
    trajectory = {
        # equity_curve[0] == initial_capital; equity_curve[k+1] is PV after step k.
        "equity_curve": np.asarray(pvs, dtype=np.float64),            # (n_steps + 1,)
        "step_returns": np.asarray(step_returns, dtype=np.float64),   # (n_steps,)
        "turnovers": np.asarray(turnovers, dtype=np.float64),         # (n_steps,) sum|Δw| per step
        "cumulative_fees": np.asarray(fees, dtype=np.float64),        # (n_steps,) running fee+slippage
    }
    return _metrics(step_returns, turnovers, pvs), np.asarray(weights, dtype=np.float64), trajectory


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
    *,
    conv_monthly: np.ndarray | None = None,
) -> tuple[dict, np.ndarray, dict]:
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

    ``conv_monthly`` override (step-4 independent live-recompute parity, P3-01/P10-01):
    when supplied, the env is driven with this PRE-ASSEMBLED held-conviction series
    instead of the batch ``monthly_rebal_conviction(...)``. The parity harness uses it
    to feed an *independently* growing-window-recomputed conviction through the SAME env
    so ``weight_l1_drift`` becomes load-bearing. It must be shape ``(T, N)`` (indexed at
    each step ``k`` like the batch path). The default ``None`` preserves the exact gate
    baseline behaviour.
    """
    if conv_monthly is None:
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
    metrics, _, _ = _linear_core_drive(arrays, config, overrides)
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
    and rung-1 paper-sim parity is ≈0 by construction (on the accounting axis — see
    :func:`linear_core_trajectory`). ``w[-1]`` is the weight to hold going forward from the
    most recent bar (the live order-generation target) — and it is the SAFE forward target:
    the env drive's last decision (index ``T-2``) reads ``conv_monthly[T-2]``, the last
    CONFIRMED interior month-end, NOT the in-progress final bar (cf. the
    :func:`monthly_rebal_conviction` truncation caveat, P2-01). Step-4 MUST source the live
    target from here, never from ``monthly_rebal_conviction(window)[-1]``.
    """
    _, weights, _ = _linear_core_drive(arrays, config, overrides)
    return weights


def linear_core_trajectory(
    arrays: Mapping,
    config: Mapping,
    *,
    overrides: Mapping | None = None,
) -> dict:
    """Full per-step trajectory of the frozen linear core — the paper executor's
    parity SIM ORACLE (ADR-7).

    Shares :func:`_linear_core_drive` with :func:`evaluate_linear_core` and
    :func:`linear_core_weights`, so the ``weights`` and ``equity_curve`` returned here
    are byte-identical to the RL-beats-linear gate baseline. The paper executor's forward
    (SimFillEngine + paper_state) path is compared against this trajectory; rung-1 parity
    is ≈0 by construction. NOTE (Tier-2 audit 2026-06-14): that 0 is load-bearing for
    ACCOUNTING (``daily_return_te_bps`` / equity diverge if the book is wrong) but
    TAUTOLOGICAL on the weight axis (the replay consumes these very ``weights``); the
    forward-DATA path (live fetch / calendar / scheduler) is validated at step-4, not here.
    See :class:`~finrl_pro_ds.paper.parity_harness.ParityHarness` for the full scope.

    Returns dict with keys: ``weights (n_steps, N)``, ``equity_curve (n_steps + 1,)``
    (``equity_curve[0] == initial_capital``), ``step_returns (n_steps,)``,
    ``turnovers (n_steps,)``, ``cumulative_fees (n_steps,)``, ``metrics`` (the same dict
    :func:`evaluate_linear_core` returns). Requires ``arrays['conviction_ary']``.
    """
    metrics, weights, traj = _linear_core_drive(arrays, config, overrides)
    return {"weights": weights, "metrics": metrics, **traj}


def drive_with_conviction(
    arrays: Mapping,
    config: Mapping,
    conv_monthly: np.ndarray,
    *,
    overrides: Mapping | None = None,
) -> dict:
    """Drive the frozen-core env with an EXTERNALLY-assembled held-conviction series.

    The step-4 independent live-recompute parity variant (Tier-2 audit 2026-06-14,
    P3-01 / P10-01 / P8-03 / P10-08). :func:`linear_core_trajectory` (the sim oracle)
    builds its conviction in ONE batch ``monthly_rebal_conviction`` call, and the rung-1
    replay then consumes the oracle's own weights — so ``weight_l1_drift`` is 0 *by
    construction* (tautological on the weight axis). To make that drift load-bearing, the
    parity harness reconstructs the held-conviction series INDEPENDENTLY on a growing
    live-fetched window (the live scheduler's view) and drives the SAME env with it here;
    a calendar / truncation / look-ahead bug in that forward assembly then moves the
    weights and is caught by ``weight_l1_drift``.

    Shares :func:`_linear_core_drive` (identical env, costs, execution-levers-OFF,
    env-native convention) with the oracle, so a CORRECT forward assembly reproduces the
    oracle weights exactly (drift ≈ 0). ``conv_monthly`` is the already-held
    (step-function) conviction of shape ``(T, N)`` — NOT raw daily conviction; the env
    re-applies only the daily vol-scale, never a second monthly-rebalance.

    Returns the same dict shape as :func:`linear_core_trajectory`.
    """
    metrics, weights, traj = _linear_core_drive(
        arrays, config, overrides, conv_monthly=np.asarray(conv_monthly, dtype=np.float64))
    return {"weights": weights, "metrics": metrics, **traj}


# --------------------------------------------------------------------------- #
# Two-sleeve risk-parity combine (momentum + rates-carry fund-of-funds)
# --------------------------------------------------------------------------- #
# The paper executor shadows a FUND-OF-FUNDS, not a single env drive: the validated
# momentum sleeve and the validated rates-carry sleeve are each driven through the env
# independently (each gets the env's per-asset vol-targeting + gross cap on its OWN
# universe), and their post-scale weight trajectories are combined by trailing-causal
# inverse-vol RISK PARITY (S553-cont-53; research `portfolio_frontier.py` risk_parity).
#
# Why a portfolio-level combine (not a single combined-conviction drive): the research
# combine scales EACH sleeve's return to a common risk THEN sums — two independent
# vol-scalings, which a single env drive (one vol-scale on a merged conviction) cannot
# reproduce. So the combine is post-scale weight math; the env accounting is then applied
# to the combined weights via PaperState (the pinned env transcription). See
# `finrl_pro_ds/paper/two_sleeve.py`.
#
# Causality (LEAK-2): the risk-parity scalar at decision bar k uses ONLY each sleeve's
# realized step returns with index < k (returns[j] is the realized j->j+1 move, known at
# bar j+1 <= k), recomputed at month-ends and HELD between (the monthly meta-cadence the
# research used; meta-turnover ~0). Full-sample sleeve vol is NEVER used (that would be
# look-ahead) — so the combined oracle is a forward-safe book, not the ex-post research
# number, preserving the executor's parity-by-construction model.


def _monthly_held(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Forward-fill ``values`` (shape ``(K, ...)``) from the LAST decision bar of each
    (year, month) — the monthly meta-rebalance hold. ``timestamps`` is the per-row decision
    stamp (epoch s). Rows before the first month-end keep their own value (warmup)."""
    ts = np.asarray(timestamps, dtype=np.int64)
    out = np.array(values, dtype=np.float64, copy=True)
    if len(ts) == 0:
        return out
    months = pd.to_datetime(ts, unit="s").to_period("M")
    last_of_month = pd.Series(np.arange(len(ts))).groupby(months.values).max().to_numpy()
    is_me = np.zeros(len(ts), dtype=bool)
    is_me[last_of_month] = True
    held = out[0].copy() if out.ndim > 1 else out[0]
    for k in range(len(ts)):
        if is_me[k]:
            held = out[k].copy() if out.ndim > 1 else out[k]
        out[k] = held
    return out


def _trailing_ann_vol(returns: np.ndarray, *, window: int, min_periods: int, ann: int = ANN) -> np.ndarray:
    """Causal trailing annualized vol of ``returns`` (1-D, indexed by decision bar k):
    row ``k`` = std of returns ``[k-window, k-1]`` (EXCLUDES the not-yet-realized k->k+1
    move) × √ann. NaN until ``min_periods`` realized returns exist (warmup)."""
    r = pd.Series(np.asarray(returns, dtype=np.float64))
    # .shift(1): row k uses returns strictly < k (returns[k] is the k->k+1 move, unknown
    # at decision bar k). ddof=1 (Bessel), matching _metrics / the env's Sortino.
    vol = r.rolling(window, min_periods=min_periods).std(ddof=1).shift(1)
    return (vol * np.sqrt(ann)).to_numpy(np.float64)


def _trailing_ann_perf(
    returns: np.ndarray,
    *,
    window: int,
    min_periods: int,
    metric: str = "sharpe",
    ann: int = ANN,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal trailing annualized risk-adjusted performance of ``returns`` (1-D, indexed by
    decision bar k): row ``k`` uses returns ``[k-window, k-1]`` (``.shift(1)`` on BOTH the
    rolling mean and the rolling denominator — EXCLUDES the not-yet-realized k->k+1 move).

    Returns ``(perf, ann_denom)``: ``perf`` is the annualized Sharpe (``mean/std·√ann``,
    ``ddof=1``, MATH-S03) or Sortino (``mean/downside_dev·√ann``, MATH-S04 — downside
    deviation is ``sqrt(Σ min(r,0)²/(n-1))``, identical to :func:`_metrics`), and
    ``ann_denom`` the annualized denominator (vol for Sharpe, downside-dev for Sortino) the
    caller uses to mask degenerate rows (M-2). Both are NaN until ``min_periods`` realized
    returns exist (warmup). ``Sharpe = ann_ret/ann_denom`` with ``ann_ret = mean·ann``,
    ``ann_denom = denom·√ann`` (so ``ann_ret/ann_denom = mean/denom·√ann``)."""
    r = pd.Series(np.asarray(returns, dtype=np.float64))
    roll = r.rolling(window, min_periods=min_periods)
    mean = roll.mean().shift(1)
    if metric == "sharpe":
        denom = roll.std(ddof=1).shift(1)
    elif metric == "sortino":
        # downside deviation over the window: sqrt(Σ min(r,0)²/(n-1)), matching _metrics /
        # the env's Sortino (RMS of min(r,0) over n-1, NOT std of only the negatives).
        def _downside_dev(x: np.ndarray) -> float:
            n = len(x)
            return float(np.sqrt(np.sum(np.minimum(x, 0.0) ** 2) / (n - 1))) if n > 1 else np.nan
        denom = roll.apply(_downside_dev, raw=True).shift(1)
    else:
        raise ValueError(f"perf_metric must be 'sharpe' or 'sortino', got {metric!r}")
    ann_denom = denom * np.sqrt(ann)
    ann_ret = mean * ann
    # ann_ret/ann_denom = (mean·ann)/(denom·√ann) = mean/denom·√ann. Where ann_denom==0 this
    # yields inf/nan; the caller neutralizes those rows (ŝ:=0 ⇒ tilt=1) via the M-2 guard.
    with np.errstate(divide="ignore", invalid="ignore"):
        perf = ann_ret / ann_denom
    return perf.to_numpy(np.float64), ann_denom.to_numpy(np.float64)


def risk_parity_alphas(
    sleeve_returns: Mapping[str, np.ndarray],
    timestamps: np.ndarray,
    *,
    window: int = 252,
    min_periods: int = 63,
    monthly_meta: bool = True,
    target_portfolio_vol: float | None = None,
    vol_floor: float = 1e-4,
) -> dict[str, np.ndarray]:
    """Per-sleeve daily capital-allocation scalars ``α_s(k)`` (the risk-parity meta-layer).

    Two modes, identical RATIO ``α_mom:α_rat = (1/σ_mom):(1/σ_rat)``:
      - ``target_portfolio_vol is None`` (default, NO portfolio-vol overlay — consistent
        with the executor's declined-overlay decision 2026-06-14): CONVEX inverse-vol,
        ``α_s = (1/σ_s) / Σ_s'(1/σ_s')`` so ``Σ_s α_s = 1``.
      - ``target_portfolio_vol = v``: research style (``portfolio_frontier.risk_parity``),
        scale each sleeve to ``v`` then equal-weight: ``α_s = (1/N)·(v/σ_s)``.

    ``σ_s(k)`` is the trailing-``window`` causal annualized vol of sleeve ``s``'s env-render
    step returns (:func:`_trailing_ann_vol`), recomputed at month-ends and held between
    when ``monthly_meta`` (the research meta-cadence). During warmup (any sleeve σ still
    NaN, or all ≤ ``vol_floor``) the step falls back to EQUAL weights (``1/N``), so no
    look-ahead and no div-by-zero. ``timestamps`` is the per-step DECISION stamp
    (``union_timestamps[:-1]``), one per of the ``T-1`` steps.

    Returns ``{sleeve: (K,) α array}`` where ``K = len(timestamps)``.
    """
    names = list(sleeve_returns)
    K = len(np.asarray(timestamps))
    N = len(names)
    sig = {s: _trailing_ann_vol(sleeve_returns[s], window=window, min_periods=min_periods)
           for s in names}
    if monthly_meta:
        sig = {s: _monthly_held(sig[s], timestamps) for s in names}

    alphas = {s: np.full(K, 1.0 / N, dtype=np.float64) for s in names}
    for k in range(K):
        sk = np.array([sig[s][k] for s in names], dtype=np.float64)
        usable = np.isfinite(sk) & (sk > vol_floor)
        if not usable.all():
            continue  # warmup / degenerate → equal weights (already set)
        inv = 1.0 / sk
        if target_portfolio_vol is None:
            a = inv / inv.sum()                       # convex, Σα = 1
        else:
            a = (1.0 / N) * float(target_portfolio_vol) * inv  # scale-each-to-target, eq-wt
        for j, s in enumerate(names):
            alphas[s][k] = a[j]
    return alphas


def dynamic_sleeve_alphas(
    sleeve_returns: Mapping[str, np.ndarray],
    timestamps: np.ndarray,
    *,
    # --- inverse-vol prior params (identical semantics to risk_parity_alphas) ---
    window: int = 252,
    min_periods: int = 63,
    monthly_meta: bool = True,
    target_portfolio_vol: float | None = None,
    vol_floor: float = 1e-4,
    # --- NEW: rolling-performance tilt (AlphaForge mechanism) ---
    tilt_strength: float = 0.0,
    perf_window: int = 126,
    perf_min_periods: int = 63,
    perf_metric: str = "sharpe",
    tilt_clip: float = 1.5,
) -> dict[str, np.ndarray]:
    """Per-sleeve daily capital-allocation scalars ``α_s(k)`` = the convex inverse-vol prior
    **tilted** toward sleeves with stronger recent risk-adjusted performance (the AlphaForge
    rolling-performance mechanism). Component 1 of the alpha-mining loop
    (``.agent/artifacts/dynamic_sleeve_combiner_architecture.md``, C1.1).

    ``α_s(k) = (p_s·tilt_s) / Σ_s'(p_s'·tilt_s')`` with ``tilt_s = exp(λ·clip(ŝ_s, −c, +c))``:

    - ``p_s(k)`` — the CONVEX inverse-vol prior ``(1/σ_s)/Σ(1/σ_s')``, computed by the SAME
      code path as :func:`risk_parity_alphas` (``_trailing_ann_vol`` + the usable-mask
      equal-weight warmup fallback). σ uses ``window``/``min_periods``.
    - ``ŝ_s(k)`` — causal trailing-``perf_window`` annualized Sharpe (or Sortino if
      ``perf_metric='sortino'``) of sleeve ``s``'s realized step returns, ``.shift(1)`` on
      BOTH the rolling mean and the rolling denominator (LEAK-2: row ``k`` uses returns
      strictly ``< k``). ``λ = tilt_strength``, ``c = tilt_clip``.
    - **Monthly-meta (M-4):** when ``monthly_meta`` both σ AND ŝ are held to the last decision
      bar of each month (``_monthly_held``), so α rotates only at month-ends (meta-turnover
      ~0 — the cost property both executors rely on).

    **Convex-only / back-compat (M-1, ADR-C1-3):** ``λ = tilt_strength = 0`` ⇒ ``tilt_s ≡ 1``
    ⇒ ``α == p`` reproduces :func:`risk_parity_alphas` (convex branch) **bit-for-bit**
    (regression test C1-T1). The leverage-bearing scale-to-target prior has no clean tilt, so
    ``target_portfolio_vol is not None`` raises :class:`NotImplementedError` (prod uses
    ``target_portfolio_vol: null``).

    **Math-audit guards (Math report 2026-06-26, folded into the contract):**
      - **M-2 (DIV-ZERO):** when a sleeve's recent denominator ``≤ vol_floor`` (annualized) or
        it is still in perf-warmup (``< perf_min_periods`` realized returns), ``ŝ_s := 0`` ⇒
        ``tilt_s = 1`` (neutral). The prior's own σ-warmup→equal-weight fallback is reused
        unchanged (when ANY sleeve σ is unusable the step is equal-weight, ``α == p``).
      - **M-3 (EXP-OVERFLOW):** ``clip(ŝ, ±c)`` bounds the exponent; with ``validate_config``
        bounding ``tilt_strength ∈ [0, 5]`` (and ``c`` the default 1.5) ``λ·c ≤ 7.5`` (safe fp64).

    Preconditions: ``sleeve_returns[s]`` shape ``(K,)`` realized step returns (the ``j→j+1``
    return known at bar ``j+1``); ``timestamps`` shape ``(K,)`` per-step DECISION stamps
    (``union_timestamps[:-1]``, epoch s). ``0 ≤ tilt_strength``, ``tilt_clip > 0``,
    ``perf_window ≥ perf_min_periods``, ``target_portfolio_vol is None``.
    Postconditions: ``{s: (K,) α}`` with ``Σ_s α_s(k) = 1`` and ``α_s(k) > 0`` ∀s,k; during σ
    warmup (prior → equal-weight) or perf-warmup/degenerate-denom (``tilt=1``) ``α == p`` (no
    look-ahead, no div-by-zero). Returns ``{sleeve: (K,) α array}``, ``K = len(timestamps)``.
    """
    # M-1: convex-only. The scale-to-target prior is leverage-bearing and has no clean tilt.
    if target_portfolio_vol is not None:
        raise NotImplementedError(
            "dynamic_sleeve_alphas supports only the convex branch (target_portfolio_vol=None); "
            "the leverage-bearing scale-to-target prior has no clean perf-tilt (M-1). "
            "Production configs use target_portfolio_vol: null.")
    if not 0.0 <= tilt_strength <= 5.0:
        # M-3: λ·c ≤ 7.5 keeps exp() in safe fp64 range. validate_config (C1.3) also enforces
        # this, but bound it here too so the function cannot silently emit exp(inf)→NaN α
        # before that wiring exists.
        raise ValueError(f"tilt_strength (λ) must be in [0, 5], got {tilt_strength}")
    if tilt_clip <= 0:
        raise ValueError(f"tilt_clip (c) must be > 0, got {tilt_clip}")
    if perf_window < perf_min_periods:
        raise ValueError(
            f"perf_window ({perf_window}) must be >= perf_min_periods ({perf_min_periods})")

    names = list(sleeve_returns)
    K = len(np.asarray(timestamps))
    N = len(names)

    # σ_s: inverse-vol prior — SAME path as risk_parity_alphas (guarantees the λ=0 identity).
    sigma = {s: _trailing_ann_vol(sleeve_returns[s], window=window, min_periods=min_periods)
             for s in names}
    # ŝ_s: recent perf, neutralized to 0 during warmup / degenerate denom ⇒ tilt=1 (M-2).
    shat = {}
    for s in names:
        perf, denom = _trailing_ann_perf(
            sleeve_returns[s], window=perf_window, min_periods=perf_min_periods,
            metric=perf_metric)
        v = perf.copy()
        neutral = ~np.isfinite(v) | ~np.isfinite(denom) | (denom <= vol_floor)
        v[neutral] = 0.0
        shat[s] = v
    if monthly_meta:  # M-4: hold BOTH σ and ŝ at month-ends (rotate only at month-ends).
        sigma = {s: _monthly_held(sigma[s], timestamps) for s in names}
        shat = {s: _monthly_held(shat[s], timestamps) for s in names}

    alphas = {s: np.full(K, 1.0 / N, dtype=np.float64) for s in names}
    lam = float(tilt_strength)
    c = float(tilt_clip)
    for k in range(K):
        sk = np.array([sigma[s][k] for s in names], dtype=np.float64)
        usable = np.isfinite(sk) & (sk > vol_floor)
        if not usable.all():
            continue  # σ warmup / degenerate → equal weights (== prior p), no tilt
        inv = 1.0 / sk                                  # convex inverse-vol prior (unnormalized)
        shk = np.array([shat[s][k] for s in names], dtype=np.float64)
        tilt = np.exp(lam * np.clip(shk, -c, c))        # =1 where ŝ=0 (warmup/degenerate/λ=0)
        wk = inv * tilt
        a = wk / wk.sum()                               # convex, Σα = 1
        for j, s in enumerate(names):
            alphas[s][k] = a[j]
    return alphas


def combine_sleeve_weights(
    sleeve_weights: Mapping[str, np.ndarray],
    sleeve_assets: Mapping[str, list[str]],
    sleeve_alphas: Mapping[str, np.ndarray],
    union_assets: list[str],
    *,
    max_gross_exposure: float | None = None,
) -> np.ndarray:
    """Map each sleeve's ``(K, n_s)`` post-vol-scale weights into the ``(K, U)`` union
    universe (by asset name), scale by its ``α_s(k)``, and sum:
    ``w_comb[k, a] = Σ_s α_s(k) · w_s[k, a]``.

    ``max_gross_exposure`` (optional): proportional gross cap on the COMBINED book (the
    env's ``_enforce_gross_exposure`` invariant, MARGIN-CFG). A no-op for the convex
    inverse-vol combine (combined gross ≤ max sleeve gross ≤ cap by the triangle
    inequality); load-bearing only if a ``target_portfolio_vol`` overlay levers the book up.
    """
    union_assets = list(union_assets)
    idx = {a: i for i, a in enumerate(union_assets)}
    K = len(next(iter(sleeve_alphas.values())))
    U = len(union_assets)
    out = np.zeros((K, U), dtype=np.float64)
    for s, w in sleeve_weights.items():
        w = np.asarray(w, dtype=np.float64)
        cols = [idx[a] for a in sleeve_assets[s]]
        a_s = np.asarray(sleeve_alphas[s], dtype=np.float64)[:, None]
        out[:, cols] += a_s * w
    if max_gross_exposure is not None:
        gross = np.abs(out).sum(axis=1)
        over = gross > float(max_gross_exposure)
        if over.any():
            scale = np.ones(K, dtype=np.float64)
            scale[over] = float(max_gross_exposure) / gross[over]
            out *= scale[:, None]
    return out
