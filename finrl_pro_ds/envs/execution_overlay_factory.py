"""Factory + evaluate-vs-TWAP-baseline gate for the execution overlay.

Step 3 of the execution-overlay build (``.agent/artifacts/execution_overlay_architecture.md``;
ADR-3/4/8). The linear 2-sleeve book emits a **fixed** target-weight trajectory
``W_target`` (``TwoSleeveExecutor.sim_oracle`` → ``combined_w``); :class:`~finrl_pro_ds.
envs.execution_scheduler_env.ExecutionSchedulerEnv` shapes the *trade path* of each
monthly conviction shift (step 2). This module is the **dispatch + decision** layer,
the execution-overlay analog of ``allocator_factory``:

  - :func:`make_execution_env` — the single config→env mapping (the keys in the
    ``execution_overlay`` config block and the base-book ``env`` cost model do NOT all
    match the env constructor kwargs), kept in the library so both the runner and the
    tests construct the env identically. Optionally composes ``PropFirmWrapperV7`` so
    the eval-time obs shape matches the (V7-augmented) training obs (ADR-5/9).
  - :func:`drive_execution_episodes` — the episodic per-rebalance driver (ADR-2): each
    valid month-end opens a parent order worked over ``H`` bars; the action is chosen by
    ``action_at_step(step_idx, obs)`` (env-native convention, exactly what
    ``agent.predict(obs)`` consumes). Aggregates per-episode net implementation shortfall.
  - :func:`evaluate_execution_overlay` — the deploy gate (ADR-8). Builds the env from the
    FIXED target, runs (a) the trained ``agent`` and (b) a **neutral-urgency baseline**
    (``action ≡ 0`` → ``m = 1`` → the closed-loop TWAP slice) over the SAME rebalance
    events under IDENTICAL costs, and returns the net-IS uplift the gate reads. Baseline
    and overlay differ ONLY in the action source — apples-to-apples, the same discipline
    as ``allocator_factory._drive(env, action_at_step)`` / ``rl_beats_linear``.

**Sign convention (the gate-critical bit).** Implementation shortfall is a *cost* (bps of
the parent-order notional, positive = bad). The overlay should make execution *cheaper*,
so the gate reads an **uplift = saving**:

    is_uplift_bps = net_is_bps_baseline − net_is_bps_overlay        (positive ⇒ overlay
                                                                      beats TWAP by that many bps)

This is the quantity ``configs/execution_overlay.gates.yaml::execution_beats_baseline``
compares against ``min_uplift_bps`` (the gate's descriptive name ``net_is_uplift_bps`` is
this same overlay-beats-baseline saving). ``cost_stress`` scales the **impact
coefficients** (the participation-multiplied permanent slope + the reactive √-impact
coeff; the flat ``base_bps``/taker fee are not impact and stay) for the robustness variant.

WF-fold aggregation + the gate verdict are the runner's job (step 5); this returns the
single-window metrics, mirroring ``evaluate_linear_core``.

**v1 baseline = closed-loop TWAP** (the built env, step 2). The config's
``baseline_schedule``/``ac_kappa`` (Almgren–Chriss) are accepted-but-ignored here — a
documented v2 schedule (ADR-4). LEAK-2 and the execution-only invariants are the env's
(step 2); this module only routes actions and aggregates the env's own ``info``.
"""
from __future__ import annotations

import inspect
import logging
from typing import Callable, Mapping, cast

import gymnasium as gym
import numpy as np
import pandas as pd

from finrl_pro_ds.envs.execution_scheduler_env import ExecutionSchedulerEnv

logger = logging.getLogger(__name__)

# execution_overlay config-block keys that pass through to ExecutionSchedulerEnv kwargs
# by identity. NOT here (handled explicitly below): the two cost-stress-scaled impact
# coefficients (slippage_impact_bps, reactive_impact_bps) and the cost/capital params
# (taker_fee_pct, slippage_base_bps, initial_capital, max_gross_exposure) resolved via _cost().
_OVERLAY_PASSTHROUGH = (
    "horizon_bars", "urgency_min", "urgency_max", "is_reward_scale",
    "impact_penalty", "risk_penalty", "unexecuted_penalty", "asymmetric_dampen",
    "reactive_decay", "vol_window", "mom_window", "relative_equity",
    "min_parent_l1",
)
_BOOL_OVERLAY = {"relative_equity"}
# Accepted-but-ignored in v1 (the built env is closed-loop TWAP; AC is a v2 schedule, ADR-4).
_OVERLAY_IGNORED = {"baseline_schedule", "ac_kappa"}

# The participation-multiplied impact coefficients cost_stress scales (ADR-8). base_bps /
# taker fee are flat notional costs (not impact) and are left unscaled.
_IMPACT_SCALED = ("slippage_impact_bps", "reactive_impact_bps")

# Pull the env's own kwarg defaults so the factory and the env can never drift on the
# cost/impact coefficients (cost_stress is only well-posed against a known base coeff).
_ENV_DEFAULTS = {
    p.name: p.default
    for p in inspect.signature(ExecutionSchedulerEnv.__init__).parameters.values()
    if p.default is not inspect.Parameter.empty
}


def rebalance_steps_from_timestamps(timestamps: np.ndarray) -> np.ndarray:
    """Parent-order calendar as STEP INDICES: the last bar index of each (year, month).

    The execution overlay treats each **monthly conviction shift** as a parent order
    (ADR-3). This returns the index positions (not the ts values) of the true month-ends
    — the same calendar ``ParityHarness._true_month_end_ts`` / ``monthly_rebal_conviction``
    derive, expressed as ``W_target`` row indices so the env can open a parent there.
    :class:`ExecutionSchedulerEnv` filters these to the horizon-valid window ``[1, K-H]``.
    """
    ts = np.asarray(timestamps, dtype=np.int64)
    if ts.size == 0:
        return np.empty(0, dtype=np.int64)
    months = pd.to_datetime(ts, unit="s").to_period("M")
    last_idx = pd.Series(np.arange(len(ts))).groupby(months.values).max().to_numpy()
    return np.asarray(sorted({int(i) for i in last_idx}), dtype=np.int64)


def make_execution_env(
    bundle: Mapping,
    config: Mapping,
    *,
    target_weights: np.ndarray | None = None,
    overrides: Mapping | None = None,
    eval_mode: bool = True,
    apply_prop_firm: bool = False,
    cost_stress: float = 1.0,
) -> gym.Env:
    """Construct an :class:`ExecutionSchedulerEnv` (optionally V7-wrapped) from a bundle.

    Args:
        bundle: ``load_two_sleeve_data`` output; reads ``bundle["union"]`` (``price_ary,
            volume_ary, carry_ary, timestamps, assets``) for the env's market arrays.
        config: full experiment config. Overlay params come from the ``execution_overlay``
            block; the cost model + ``initial_capital`` + ``max_gross_exposure`` come from
            the base-book ``env`` block (the overlay block may override either).
        target_weights: the FIXED ``(K, U)`` target. ``None`` ⇒ computed from
            ``TwoSleeveExecutor(config).sim_oracle(bundle)["combined_w"]`` (lazy import to
            avoid an env↔paper cycle); pass it explicitly to skip the loader/sleeve drive.
        overrides: per-call overlay overrides applied on top of the ``execution_overlay`` block.
        eval_mode: ``True`` ⇒ ``random_start=False`` (deterministic; the driver sets the
            rebalance per episode via ``reset(options=...)`` regardless).
        apply_prop_firm: compose :class:`PropFirmWrapperV7` (ADR-5) reading the ``prop_firm``
            config block — so the eval obs shape matches a V7-augmented training obs.
        cost_stress: scales the impact coefficients (``slippage_impact_bps``,
            ``reactive_impact_bps``) for the gate's robustness variant (ADR-8).
    """
    union = bundle["union"]
    if target_weights is None:
        # Lazy import: two_sleeve → allocator_factory (no execution_overlay_factory dep),
        # so importing it here introduces no import cycle (env→paper direction only).
        from finrl_pro_ds.paper.two_sleeve import TwoSleeveExecutor

        _, detail = TwoSleeveExecutor(config).sim_oracle(bundle)
        target_weights = detail["combined_w"]

    env_block = dict(config.get("env", {}))
    ov_block = dict(config.get("execution_overlay", {}))
    if overrides:
        ov_block.update(overrides)

    def _cost(name: str, alt: str | None = None) -> float:
        """Read a cost/capital param: overlay block, then env block (with an ``alt`` name
        e.g. ``taker_fee`` for ``taker_fee_pct``), then the env's own default."""
        if name in ov_block:
            return float(ov_block[name])
        if name in env_block:
            return float(env_block[name])
        if alt is not None and alt in env_block:
            return float(env_block[alt])
        return float(_ENV_DEFAULTS[name])

    cs = float(cost_stress)
    if cs <= 0.0:
        raise ValueError(f"cost_stress must be > 0 (got {cost_stress})")

    kwargs: dict = {
        "initial_capital": _cost("initial_capital"),
        "max_gross_exposure": _cost("max_gross_exposure"),
        "taker_fee_pct": _cost("taker_fee_pct", alt="taker_fee"),
        "slippage_base_bps": _cost("slippage_base_bps"),
        "random_start": (not eval_mode),
    }
    for key in _OVERLAY_PASSTHROUGH:
        if key in ov_block:
            val = ov_block[key]
            kwargs[key] = bool(val) if key in _BOOL_OVERLAY else val
    # Impact coefficients: resolve (overlay → env block → env default) then scale by cost_stress.
    for key in _IMPACT_SCALED:
        base = float(ov_block.get(key, env_block.get(key, _ENV_DEFAULTS[key])))
        kwargs[key] = base * cs

    rebalance_steps = rebalance_steps_from_timestamps(union["timestamps"])

    env: gym.Env = ExecutionSchedulerEnv(
        target_weights=np.asarray(target_weights, dtype=np.float64),
        price_ary=union["price_ary"],
        volume_ary=union["volume_ary"],
        carry_ary=union["carry_ary"],
        timestamps=union["timestamps"],
        assets=list(union["assets"]),
        rebalance_steps=rebalance_steps,
        **kwargs,
    )

    if apply_prop_firm:
        env = _wrap_prop_firm(env, config)
    return env


def _wrap_prop_firm(env: gym.Env, config: Mapping) -> gym.Env:
    """Compose :class:`PropFirmWrapperV7` from the ``prop_firm`` config block (ADR-5).

    The wrapper constrains execution-induced tracking-error drawdown and augments
    ``private`` with 3 risk-budget dims; the base env reports ``info["portfolio_value"]``
    as the relative (overlay-vs-snap) equity (``relative_equity=True``) so the DD the
    wrapper sees is exactly the timing error the overlay controls.
    """
    from finrl_pro_ds.envs.prop_firm_wrapper import PropFirmWrapperV7

    pf = dict(config.get("prop_firm", {}))
    return PropFirmWrapperV7(
        env,
        max_trailing_drawdown_pct=float(pf.get("max_trailing_drawdown_pct", 0.05)),
        drawdown_penalty_start=float(pf.get("drawdown_penalty_start", 0.02)),
        drawdown_penalty_scale=float(pf.get("drawdown_penalty_scale", 5.0)),
        augment_obs=bool(pf.get("augment_obs", True)),
        static_peak=bool(pf.get("static_peak", True)),
    )


# --------------------------------------------------------------------------- #
# Action sources (baseline neutral-TWAP vs trained agent)
# --------------------------------------------------------------------------- #
def neutral_baseline_action(step_idx: int, obs: Mapping) -> np.ndarray:  # noqa: ARG001
    """The TWAP baseline action: ``a = 0 → m = 1 →`` the exact closed-loop TWAP slice.

    The deterministic, tuning-free schedule the overlay must beat (ADR-8). Ignores ``obs``
    — the baseline is open-loop equal-slicing; only the urgency *multiplier* is neutral.
    """
    return np.zeros(1, dtype=np.float32)


def _sac_action_source(
    agent, n_scales: int, *, deterministic: bool = True,
) -> Callable[[int, Mapping], np.ndarray]:
    """Wrap a trained dict-obs SAC agent as an ``action_at_step``.

    Replicates ``SACTrainer``'s summary_stats convention EXACTLY (``sac_trainer.py``):
    concatenate ``[scale_0, …, scale_{n-1}, private]`` into a flat ``(1, D)`` tensor,
    pass it as ``scale_input`` with ``private=None``, and predict (``deterministic`` ⇒ the
    mean action for the deploy/forward replay; the gate eval uses the default ``True``).
    Torch is imported lazily so the factory (and its unit tests) stay torch-free unless a
    real agent is actually driven.
    """
    import torch

    device = getattr(agent, "device", "cpu")

    def action(step_idx: int, obs: Mapping) -> np.ndarray:  # noqa: ARG001
        parts = [np.asarray(obs[f"scale_{i}"], dtype=np.float32) for i in range(n_scales)]
        parts.append(np.asarray(obs["private"], dtype=np.float32))
        flat = np.concatenate(parts).reshape(1, -1)
        scale_stack = torch.as_tensor(flat, dtype=torch.float32).to(device, non_blocking=True)
        out = agent.predict(scale_stack, None, deterministic=deterministic)
        return np.asarray(out.detach().cpu().numpy(), dtype=np.float32).reshape(-1)

    return action


def resolve_action_source(
    agent, n_scales: int, *, deterministic: bool = True,
) -> Callable[[int, Mapping], np.ndarray]:
    """Resolve ``agent`` to an ``action_at_step(step_idx, obs)`` callable.

    The canonical action-convention resolver shared by the gate eval
    (:func:`evaluate_execution_overlay`) and the executor's forward replay
    (``TwoSleeveExecutor.run_with_overlay``) so both drive a SAC policy identically.
    Accepts: ``None`` (the neutral TWAP baseline), a SAC-like agent (has ``.predict``), or
    a plain ``callable(obs) -> action`` (test stubs / fixed-urgency policies).
    """
    if agent is None:
        return neutral_baseline_action
    if hasattr(agent, "predict"):
        return _sac_action_source(agent, n_scales, deterministic=deterministic)
    if callable(agent):
        return lambda step_idx, obs: np.asarray(agent(obs), dtype=np.float32).reshape(-1)
    raise TypeError(
        "agent must be a SAC-like agent (.predict), a callable(obs)->action, or None; "
        f"got {type(agent)!r}")


# --------------------------------------------------------------------------- #
# Episodic driver + net-IS aggregation
# --------------------------------------------------------------------------- #
def drive_execution_episodes(env: gym.Env, action_at_step: Callable[[int, Mapping], np.ndarray]) -> dict:
    """Run every valid rebalance event as one episode; aggregate net implementation shortfall.

    Each episode = one parent order worked over ``H`` bars (ADR-2). The action at ``step_idx``
    is ``action_at_step(step_idx, obs)`` (env-native: the action at bar ``k`` governs the
    ``k→k+1`` execution slice). Per-step costs come from the env's own ``info`` (already in
    bps of the parent notional, independent of the reward scaling). Returns:

      ``net_is_bps`` — mean over episodes of ``Σ_h (IS_timing + impact)`` bps (the cost the
        gate minimizes); ``is_timing_bps`` / ``impact_bps`` its two components;
      ``completion_l1_max`` — worst horizon-end ``‖W_held − W_target‖₁`` (execution-only
        completion check); ``intra_horizon_drift_max`` — worst within-horizon lag vs the
        snap target; ``turnover`` — mean ``Σ|trade|``; ``per_episode`` the raw list.
    """
    base = cast(ExecutionSchedulerEnv, env.unwrapped)  # the scheduler even when V7-wrapped
    steps = [int(r) for r in base.rebalance_steps]
    episodes: list[dict] = []
    for r in steps:
        obs, _ = env.reset(options={"rebalance_step": r})
        is_bps = impact_bps = turnover = 0.0
        intra = 0.0
        info: dict = {}
        done = False
        while not done:
            k = base.step_idx
            a = np.asarray(action_at_step(k, obs), dtype=np.float32).reshape(-1)[:1]
            obs, _reward, terminated, truncated, info = env.step(a)
            is_bps += float(info["implementation_shortfall_bps"])
            impact_bps += float(info["impact_cost_bps"])
            turnover += float(info["executed_l1"])
            intra = max(intra, float(info["remaining_l1"]))  # worst mid-horizon lag
            done = terminated or truncated
        episodes.append({
            "rebalance_step": r,
            "net_is_bps": is_bps + impact_bps,
            "is_timing_bps": is_bps,
            "impact_bps": impact_bps,
            "turnover": turnover,
            "completion_l1": float(info.get("remaining_l1", 0.0)),  # residual at horizon end (~0)
            "intra_horizon_drift": intra,
        })
    return _aggregate_episodes(episodes)


def _aggregate_episodes(episodes: list[dict]) -> dict:
    if not episodes:
        return {"net_is_bps": 0.0, "is_timing_bps": 0.0, "impact_bps": 0.0,
                "turnover": 0.0, "completion_l1_max": 0.0, "intra_horizon_drift_max": 0.0,
                "n_episodes": 0, "per_episode": []}

    def col(key: str) -> np.ndarray:
        return np.array([e[key] for e in episodes], dtype=np.float64)

    return {
        "net_is_bps": float(col("net_is_bps").mean()),
        "is_timing_bps": float(col("is_timing_bps").mean()),
        "impact_bps": float(col("impact_bps").mean()),
        "turnover": float(col("turnover").mean()),
        "completion_l1_max": float(col("completion_l1").max()),
        "intra_horizon_drift_max": float(col("intra_horizon_drift").max()),
        "n_episodes": len(episodes),
        "per_episode": episodes,
    }


# --------------------------------------------------------------------------- #
# Deploy gate: overlay vs tuned-TWAP baseline (ADR-8)
# --------------------------------------------------------------------------- #
def evaluate_execution_overlay(
    bundle: Mapping,
    config: Mapping,
    agent,
    *,
    overrides: Mapping | None = None,
    cost_stress: float = 1.0,
    target_weights: np.ndarray | None = None,
    apply_prop_firm: bool | None = None,
) -> dict:
    """Net-IS uplift of the trained overlay vs the neutral-urgency TWAP baseline (the gate).

    Builds TWO identically-constructed envs from the SAME fixed target under IDENTICAL
    costs (``cost_stress`` applied to both) and drives (a) the ``agent`` and (b) the
    neutral baseline (``action ≡ 0`` → TWAP) over the same rebalance events — so the ONLY
    difference is the action source (the ``rl_beats_linear`` discipline). Returns the
    ``is_uplift_bps`` (= baseline net-IS − overlay net-IS, positive ⇒ the overlay is
    cheaper) the runner's ``execution_beats_baseline`` gate reads, plus the overlay's
    completion/drift/turnover diagnostics.

    Args:
        agent: trained execution-overlay policy (``.predict``), a ``callable(obs)->action``,
            or ``None`` (degenerates to baseline ⇒ uplift exactly 0 — the parity self-check).
        cost_stress: impact-coefficient multiplier for the robustness variant.
        target_weights: optional precomputed fixed target (skips the sleeve drive).
        apply_prop_firm: wrap both envs with V7 (so the agent's obs shape matches training).
            ``None`` ⇒ derived from ``config['prop_firm']['augment_obs']``.
    """
    if apply_prop_firm is None:
        apply_prop_firm = bool(config.get("prop_firm", {}).get("augment_obs", False))
    n_scales = int(config.get("network", {}).get("n_scales", 1))

    # Compute the FIXED target once so both envs are byte-identical inputs.
    if target_weights is None:
        from finrl_pro_ds.paper.two_sleeve import TwoSleeveExecutor

        _, detail = TwoSleeveExecutor(config).sim_oracle(bundle)
        target_weights = detail["combined_w"]

    def _build() -> gym.Env:
        return make_execution_env(
            bundle, config, target_weights=target_weights, overrides=overrides,
            eval_mode=True, apply_prop_firm=apply_prop_firm, cost_stress=cost_stress)

    baseline = drive_execution_episodes(_build(), neutral_baseline_action)
    overlay = drive_execution_episodes(_build(), resolve_action_source(agent, n_scales))

    uplift = baseline["net_is_bps"] - overlay["net_is_bps"]
    logger.info(
        "execution overlay eval: baseline net-IS %.3f bps, overlay net-IS %.3f bps, "
        "uplift %.3f bps (cost_stress=%.2f, n_episodes=%d)",
        baseline["net_is_bps"], overlay["net_is_bps"], uplift, cost_stress, overlay["n_episodes"])

    return {
        "net_is_bps_overlay": overlay["net_is_bps"],
        "net_is_bps_baseline": baseline["net_is_bps"],
        "is_uplift_bps": uplift,
        "completion_l1_max": overlay["completion_l1_max"],
        "intra_horizon_drift_max": overlay["intra_horizon_drift_max"],
        "turnover_overlay": overlay["turnover"],
        "turnover_baseline": baseline["turnover"],
        "cost_stress": float(cost_stress),
        "n_episodes": overlay["n_episodes"],
        "metrics": {"overlay": overlay, "baseline": baseline},
    }
