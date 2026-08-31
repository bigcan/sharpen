"""execution_overlay_factory — the build/drive/gate layer for the RL execution overlay.

Step 3 of the execution-overlay build (execution_overlay_architecture.md, ADR-3/4/8).
Pins the gate's apples-to-apples discipline and the uplift sign convention:
  - baseline and overlay differ ONLY in the action source (neutral agent ⇒ uplift 0);
  - is_uplift_bps = baseline net-IS − overlay net-IS (positive ⇒ overlay cheaper than TWAP);
  - a favorable (front-load-beats-TWAP) case yields a POSITIVE uplift;
  - cost_stress scales the impact coefficients (flat-price ⇒ net-IS scales linearly);
  - rebalance_steps_from_timestamps returns the month-end STEP INDICES (ADR-3 calendar);
  - make_execution_env maps the config block + V7 wrap so the eval obs matches training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sharpen.envs.execution_overlay_factory import (
    drive_execution_episodes,
    evaluate_execution_overlay,
    make_execution_env,
    neutral_baseline_action,
    rebalance_steps_from_timestamps,
)
from sharpen.envs.execution_scheduler_env import (
    MARKET_DIM,
    PRIVATE_DIM,
    ExecutionSchedulerEnv,
)

_DAY = 86_400


def _daily_ts(start: str, n: int) -> np.ndarray:
    """`n` consecutive daily epoch-second timestamps from `start` (UTC)."""
    base = int(pd.Timestamp(start, tz="UTC").timestamp())
    return base + np.arange(n, dtype=np.int64) * _DAY


def _bundle(*, price, volume, ts, assets):
    union = {
        "price_ary": np.asarray(price, dtype=np.float64),
        "volume_ary": np.asarray(volume, dtype=np.float64),
        "carry_ary": np.zeros_like(np.asarray(price, dtype=np.float64)),
        "timestamps": np.asarray(ts, dtype=np.int64),
        "assets": list(assets),
    }
    return {"union": union}


def _multi_month_bundle(U=3, n_days=100, seed=0, volume=1e9):
    """A ~3.3-month daily bundle (several interior month-ends) with random-walk prices and
    a step-function target jumping at each month-end."""
    rng = np.random.default_rng(seed)
    price = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, size=(n_days, U)), axis=0)
    ts = _daily_ts("2021-01-08", n_days)
    bundle = _bundle(price=price, volume=np.full((n_days, U), float(volume)), ts=ts,
                     assets=[f"A{i}" for i in range(U)])
    # target: hold the most recent month-end's conviction (a simple deterministic block).
    rebal = rebalance_steps_from_timestamps(ts)
    blocks = [np.array([0.3, -0.2, 0.1])[:U], np.array([0.1, 0.3, -0.2])[:U],
              np.array([-0.1, 0.1, 0.3])[:U], np.array([0.2, -0.1, 0.2])[:U]]
    K = n_days - 1
    W = np.zeros((K, U), dtype=np.float64)
    held = blocks[0]
    bi = 0
    rebal_set = set(int(r) for r in rebal)
    for k in range(K):
        if k in rebal_set:
            bi = min(bi + 1, len(blocks) - 1)
            held = blocks[bi]
        W[k] = held
    return bundle, W


def _config(**overlay):
    ov = {"horizon_bars": 3, "urgency_min": 0.0, "urgency_max": 2.0,
          "reactive_impact_bps": 0.0, "impact_penalty": 0.0}
    ov.update(overlay)
    return {
        "env": {"initial_capital": 100_000.0, "max_gross_exposure": 3.0,
                "taker_fee_pct": 0.0, "slippage_base_bps": 0.0, "slippage_impact_bps": 0.0},
        "execution_overlay": ov,
        "network": {"n_scales": 1},
    }


# --------------------------------------------------------------------------- #
# rebalance calendar (ADR-3): month-end STEP INDICES
# --------------------------------------------------------------------------- #
def test_rebalance_steps_are_month_end_indices():
    ts = _daily_ts("2021-01-15", 75)               # spans Jan, Feb, into Mar
    steps = rebalance_steps_from_timestamps(ts)
    # Independently: the max index within each (year, month).
    months = pd.to_datetime(ts, unit="s").to_period("M")
    expected = pd.Series(np.arange(len(ts))).groupby(months.values).max().to_numpy()
    np.testing.assert_array_equal(steps, np.sort(np.unique(expected)))
    # Jan-31 is the 17th day from Jan-15 (index 16); assert that concrete anchor.
    assert 16 in set(int(s) for s in steps)


def test_rebalance_steps_empty():
    assert rebalance_steps_from_timestamps(np.empty(0, dtype=np.int64)).size == 0


# --------------------------------------------------------------------------- #
# make_execution_env: config mapping + V7 composition
# --------------------------------------------------------------------------- #
def test_make_env_builds_scheduler_with_mapped_costs():
    bundle, W = _multi_month_bundle()
    cfg = _config(reactive_impact_bps=8.0)
    env = make_execution_env(bundle, cfg, target_weights=W, eval_mode=True)
    assert isinstance(env, ExecutionSchedulerEnv)
    assert env.n_assets == 3
    assert env.H == 3
    assert env.engine.reactive_impact_bps == 8.0          # overlay-block coeff mapped
    assert env.observation_space["scale_0"].shape == (MARKET_DIM,)
    assert env.observation_space["private"].shape == (PRIVATE_DIM,)
    assert env.random_start is False                      # eval_mode
    assert env.rebalance_steps.size >= 1


def test_make_env_cost_stress_scales_impact_coeff():
    bundle, W = _multi_month_bundle()
    cfg = _config(reactive_impact_bps=8.0, slippage_impact_bps=5.0)
    cfg["env"]["slippage_impact_bps"] = 5.0
    env = make_execution_env(bundle, cfg, target_weights=W, cost_stress=2.5)
    assert env.engine.reactive_impact_bps == pytest.approx(8.0 * 2.5)
    assert env.engine.slippage_impact_bps == pytest.approx(5.0 * 2.5)
    assert env.engine.slippage_base_bps == 0.0            # base (flat) NOT scaled


def test_make_env_rejects_nonpositive_cost_stress():
    bundle, W = _multi_month_bundle()
    with pytest.raises(ValueError):
        make_execution_env(bundle, _config(), target_weights=W, cost_stress=0.0)


def test_make_env_applies_prop_firm_wrapper():
    bundle, W = _multi_month_bundle()
    cfg = _config()
    cfg["prop_firm"] = {"augment_obs": True, "max_trailing_drawdown_pct": 0.05, "static_peak": True}
    env = make_execution_env(bundle, cfg, target_weights=W, apply_prop_firm=True)
    assert env.observation_space["private"].shape == (PRIVATE_DIM + 3,)
    obs, _ = env.reset(options={"rebalance_step": int(env.unwrapped.rebalance_steps[0])})
    assert obs["private"].shape == (PRIVATE_DIM + 3,)
    assert obs["scale_0"].shape == (MARKET_DIM,)


# --------------------------------------------------------------------------- #
# Gate: baseline vs overlay differ ONLY in the action source
# --------------------------------------------------------------------------- #
def test_neutral_agent_gives_zero_uplift():
    """agent=None ⇒ overlay uses the neutral baseline ⇒ identical env, identical costs,
    identical actions ⇒ is_uplift_bps == 0 exactly (the construction-identity self-check)."""
    bundle, W = _multi_month_bundle()
    res = evaluate_execution_overlay(bundle, _config(reactive_impact_bps=8.0), agent=None,
                                     target_weights=W)
    assert res["net_is_bps_overlay"] == res["net_is_bps_baseline"]
    assert res["is_uplift_bps"] == 0.0
    assert res["n_episodes"] >= 1


def test_neutral_callable_agent_also_zero_uplift():
    """A callable agent that returns the neutral action reduces to the baseline (proves the
    callable action-source path is equivalent — only the action value matters)."""
    bundle, W = _multi_month_bundle()
    res = evaluate_execution_overlay(bundle, _config(reactive_impact_bps=8.0),
                                     agent=lambda obs: np.zeros(1, dtype=np.float32),
                                     target_weights=W)
    assert res["is_uplift_bps"] == pytest.approx(0.0, abs=1e-12)


def test_evaluate_through_prop_firm_wrapper_smoke():
    """End-to-end gate eval with apply_prop_firm=True (the V7-augmented-obs eval path step 6
    will use): the driver must handle the wrapped env (obs +3, info passthrough) and return
    finite, matched-episode metrics — no shape/KeyError. A neutral agent ⇒ uplift still 0."""
    bundle, W = _multi_month_bundle()
    cfg = _config(reactive_impact_bps=8.0)
    cfg["prop_firm"] = {"augment_obs": True, "max_trailing_drawdown_pct": 0.05, "static_peak": True}
    res = evaluate_execution_overlay(
        bundle, cfg, agent=lambda obs: np.zeros(1, dtype=np.float32),
        target_weights=W, apply_prop_firm=True)
    assert res["n_episodes"] >= 1
    assert np.isfinite(res["net_is_bps_overlay"]) and np.isfinite(res["net_is_bps_baseline"])
    assert res["is_uplift_bps"] == pytest.approx(0.0, abs=1e-12)   # neutral both ⇒ apples-to-apples


def test_overlay_completes_each_parent():
    """Whatever the action source, completion is structural ⇒ completion_l1_max ≈ 0."""
    bundle, W = _multi_month_bundle()
    res = evaluate_execution_overlay(bundle, _config(),
                                     agent=lambda obs: np.array([0.5], dtype=np.float32),
                                     target_weights=W)
    assert res["completion_l1_max"] < 1e-8


# --------------------------------------------------------------------------- #
# cost_stress scales the impact-driven net-IS (flat price ⇒ linear)
# --------------------------------------------------------------------------- #
def test_cost_stress_scales_baseline_net_is_linearly():
    """Flat prices ⇒ timing IS = 0 ⇒ net-IS is pure impact. The impact coefficients scale
    linearly with cost_stress, so baseline net-IS at cost_stress=2 is ≈ 2× cost_stress=1.
    (Only *approximately* 2×: at higher cost the book loses slightly more equity each bar,
    so pv_before — and the participation-driven notional — on later bars is marginally
    lower; a real second-order feedback, ~3e-5 relative here, not a scaling bug.)"""
    U, n = 3, 100
    price = np.full((n, U), 100.0)                         # flat ⇒ zero timing shortfall
    ts = _daily_ts("2021-01-08", n)
    bundle = _bundle(price=price, volume=np.full((n, U), 2.0e6), ts=ts,
                     assets=[f"A{i}" for i in range(U)])
    rebal = rebalance_steps_from_timestamps(ts)
    K = n - 1
    W = np.zeros((K, U), dtype=np.float64)
    held = np.zeros(U)
    for k in range(K):
        if k in set(int(r) for r in rebal):
            held = np.array([0.4, -0.3, 0.2])             # a real parent to incur impact
        W[k] = held
    cfg = _config(reactive_impact_bps=8.0)

    r1 = evaluate_execution_overlay(bundle, cfg, agent=None, target_weights=W, cost_stress=1.0)
    r2 = evaluate_execution_overlay(bundle, cfg, agent=None, target_weights=W, cost_stress=2.0)
    assert r1["net_is_bps_baseline"] > 0.0                 # impact is real
    assert r2["net_is_bps_baseline"] > r1["net_is_bps_baseline"]   # strictly increases with stress
    assert r2["net_is_bps_baseline"] == pytest.approx(2.0 * r1["net_is_bps_baseline"], rel=1e-3)


# --------------------------------------------------------------------------- #
# Favorable case ⇒ positive uplift (sign convention)
# --------------------------------------------------------------------------- #
def _front_load_action(obs):
    """Max urgency (a=+1 ⇒ m=urgency_max) — front-loads the parent into the earliest bars."""
    return np.array([1.0], dtype=np.float32)


def test_front_load_beats_twap_on_rising_buy_drive_level():
    """A BUY whose price rises after arrival: front-loading fills at the cheaper early bars,
    so its cumulative IS (struck vs the fixed arrival) is below TWAP's. Tested directly on
    the driver so the sign of the aggregation is unambiguous (overlay net-IS < baseline)."""
    U, T, r0, H = 1, 16, 8, 4
    price = np.full((T, U), 100.0)
    for k in range(r0 + 1, T):                             # rise strictly AFTER arrival (price[r0])
        price[k, 0] = 100.0 * (1.0 + 0.02) ** (k - r0)
    W = np.zeros((T - 1, U))
    W[r0:] = 0.5                                           # buy 0 → 0.5 at the rebalance
    common = dict(target_weights=W, price_ary=price, volume_ary=np.full((T, U), 1e9),
                  carry_ary=np.zeros((T, U)), timestamps=np.arange(T, dtype=np.int64) * _DAY,
                  assets=["A0"], rebalance_steps=np.array([r0]), horizon_bars=H,
                  random_start=False, impact_penalty=0.0, reactive_impact_bps=0.0,
                  slippage_base_bps=0.0, slippage_impact_bps=0.0, taker_fee_pct=0.0)
    baseline = drive_execution_episodes(ExecutionSchedulerEnv(**common), neutral_baseline_action)
    overlay = drive_execution_episodes(ExecutionSchedulerEnv(**common),
                                       lambda k, obs: _front_load_action(obs))
    assert baseline["net_is_bps"] > 0.0                    # TWAP pays the rising price
    assert overlay["net_is_bps"] < baseline["net_is_bps"]  # front-load fills cheaper/earlier


def test_evaluate_overlay_positive_uplift_single_month_end():
    """End-to-end through evaluate_execution_overlay: a bundle with exactly ONE valid
    month-end (the conviction jump) + a rising post-arrival price ⇒ is_uplift_bps > 0 and
    equals baseline − overlay (the documented sign convention)."""
    U, n, H = 1, 40, 3
    ts = _daily_ts("2021-01-20", n)                        # only Jan-31 is an interior month-end
    rebal = rebalance_steps_from_timestamps(ts)
    K = n - 1
    valid = [int(r) for r in rebal if 1 <= int(r) <= K - H]
    assert len(valid) == 1, f"expected one valid month-end, got {valid}"
    r0 = valid[0]
    price = np.full((n, U), 100.0)
    for k in range(r0 + 1, n):
        price[k, 0] = 100.0 * (1.0 + 0.02) ** (k - r0)     # rising buy after arrival
    W = np.zeros((K, U), dtype=np.float64)
    W[r0:] = 0.5
    bundle = _bundle(price=price, volume=np.full((n, U), 1e9), ts=ts, assets=["A0"])
    cfg = _config(horizon_bars=H)

    res = evaluate_execution_overlay(bundle, cfg, agent=_front_load_action, target_weights=W)
    assert res["n_episodes"] == 1
    assert res["is_uplift_bps"] > 0.0
    assert res["is_uplift_bps"] == pytest.approx(
        res["net_is_bps_baseline"] - res["net_is_bps_overlay"], rel=1e-9)
