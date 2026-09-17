"""v1.1 cost-lever tests for MultiAssetAllocatorEnv (S553-cont-34 follow-up).

The Stage-3 WF proved the RL has real gross alpha (frictionless Sharpe 0.80 vs
core 0.34 across all 14 windows) but loses it all to turnover cost (median
cost_gap 0.41). These levers exist to convert gross to net:

  1. ``no_trade_band``     — hold deltas inside the band; trade to the band EDGE.
  2. ``rebalance_interval``— decision cadence in bars; off-cadence bars hold.
  3. ``cost_penalty_scale``— re-charge the bar's realized fee+slippage in reward.

Defaults (0.0 / 1 / 0.0) must keep v1 behavior byte-identical, and
``evaluate_linear_core`` must force the execution levers OFF so the gate
baseline stays the validated monthly linear core.
"""
from __future__ import annotations

import numpy as np

from sharpen.envs import allocator_factory as factory
from sharpen.envs.multi_asset_allocator_env import MultiAssetAllocatorEnv

from .conftest import build_arrays, synthetic_prices


def _env(T=60, n=2, **kw) -> MultiAssetAllocatorEnv:
    """Env with unit vol-scaling (vol == target_vol => weights == conviction),
    zero costs/dust unless overridden — isolates the lever mechanics."""
    price = synthetic_prices(T=T, n=n, seed=5)
    arrays = build_arrays(price, vol=np.full((T, n), 0.10))
    params = dict(
        target_vol_asset=0.10,
        lev_cap=2.0,
        max_gross_exposure=1e9,
        min_trade_pct=0.0,
        taker_fee_pct=0.0,
        slippage_base_bps=0.0,
        slippage_impact_bps=0.0,
        turnover_penalty=0.0,
        reward_type="simple",
    )
    params.update(kw)
    return MultiAssetAllocatorEnv(**arrays, **params)


# --------------------------------------------------------------------------- #
# 1. no_trade_band
# --------------------------------------------------------------------------- #

def test_band_holds_inside_and_trades_to_edge_outside():
    env = _env(no_trade_band=0.1)
    env.reset()
    # From flat: target 0.5, delta 0.5 > band -> trade to edge: position 0.4.
    env.step(np.array([0.5, 0.0]))
    np.testing.assert_allclose(env.positions, [0.4, 0.0], atol=1e-12)
    # Same target again: delta 0.1 <= band -> hold.
    env.step(np.array([0.5, 0.0]))
    np.testing.assert_allclose(env.positions, [0.4, 0.0], atol=1e-12)
    # Sign flip: target -0.5, delta -0.9 -> trade to edge: -0.5 + 0.1 = -0.4.
    env.step(np.array([-0.5, 0.0]))
    np.testing.assert_allclose(env.positions, [-0.4, 0.0], atol=1e-12)


def test_band_reduces_turnover_on_noisy_policy():
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, size=(40, 2))

    def total_turnover(**kw):
        env = _env(T=60, **kw)
        env.reset()
        tot = 0.0
        for a in actions:
            _, _, term, trunc, info = env.step(a)
            tot += info["turnover"]
            if term or trunc:
                break
        return tot

    assert total_turnover(no_trade_band=0.2) < total_turnover() * 0.9


def test_band_does_not_strand_position_in_dead_asset():
    """Availability-forced closes must BYPASS the band: when an asset's price goes
    invalid, the position fully closes instead of leaving a band-sized residual
    that every later (inside-band) delta would then hold forever."""
    T, n = 60, 2
    price = synthetic_prices(T=T, n=n, seed=5)
    price[10:, 0] = 0.0                              # asset 0 dies at bar 10
    arrays = build_arrays(price, vol=np.full((T, n), 0.10))
    env = MultiAssetAllocatorEnv(
        **arrays, target_vol_asset=0.10, lev_cap=2.0, max_gross_exposure=1e9,
        min_trade_pct=0.0, taker_fee_pct=0.0, slippage_base_bps=0.0,
        slippage_impact_bps=0.0, turnover_penalty=0.0, reward_type="simple",
        no_trade_band=0.3,
    )
    env.reset()
    env.step(np.array([1.0, 0.0]))                   # open: band-shrunk to 0.7
    np.testing.assert_allclose(env.positions, [0.7, 0.0], atol=1e-12)
    for _ in range(9):                               # bars 2..10 (bar 10 = dead price)
        env.step(np.array([1.0, 0.0]))
    assert env.positions[0] == 0.0, (
        f"dead-asset position not fully closed: {env.positions[0]}")


def test_band_zero_is_exact_noop():
    actions = np.random.default_rng(1).uniform(-1, 1, size=(30, 2))
    env_a, env_b = _env(), _env(no_trade_band=0.0)
    env_a.reset(), env_b.reset()
    for a in actions:
        _, ra, *_ = env_a.step(a)
        _, rb, *_ = env_b.step(a)
        assert ra == rb
        np.testing.assert_array_equal(env_a.positions, env_b.positions)


# --------------------------------------------------------------------------- #
# 2. rebalance_interval
# --------------------------------------------------------------------------- #

def test_interval_holds_between_cadence_bars():
    env = _env(rebalance_interval=5)
    env.reset()
    # Decision bar 0 (0 % 5 == 0): trade fires.
    env.step(np.array([0.5, -0.3]))
    held = env.positions.copy()
    assert np.abs(held).sum() > 0
    # Decision bars 1..4: actions are no-ops, positions held.
    for a in ([1.0, 1.0], [-1.0, 0.7], [0.0, 0.0], [0.9, -0.9]):
        _, _, _, _, info = env.step(np.array(a))
        assert info["turnover"] == 0.0
        np.testing.assert_array_equal(env.positions, held)
    # Decision bar 5: trading resumes.
    _, _, _, _, info = env.step(np.array([-0.5, 0.3]))
    assert info["turnover"] > 0


def test_interval_one_is_exact_noop():
    actions = np.random.default_rng(2).uniform(-1, 1, size=(30, 2))
    env_a, env_b = _env(), _env(rebalance_interval=1)
    env_a.reset(), env_b.reset()
    for a in actions:
        _, ra, *_ = env_a.step(a)
        _, rb, *_ = env_b.step(a)
        assert ra == rb
        np.testing.assert_array_equal(env_a.positions, env_b.positions)


# --------------------------------------------------------------------------- #
# 3. cost_penalty_scale
# --------------------------------------------------------------------------- #

def test_cost_penalty_equals_scaled_realized_cost():
    """reward(with) - reward(without) == -scale * (fee+slip) / pv_before, on the
    first trade from flat where pv_before == initial_capital."""
    fee = 0.001
    scale = 2.0
    env_a = _env(taker_fee_pct=fee)
    env_b = _env(taker_fee_pct=fee, cost_penalty_scale=scale)
    env_a.reset(), env_b.reset()
    action = np.array([1.0, -1.0])
    _, ra, *_ = env_a.step(action)
    _, rb, *_ = env_b.step(action)
    expected_cost = 2.0 * env_a.initial_capital * fee   # sum|dw|=2 notional * fee
    assert abs(env_a.cumulative_fees - expected_cost) < 1e-9
    assert abs((ra - rb) - scale * expected_cost / env_a.initial_capital) < 1e-12


def test_cost_penalty_zero_when_no_trade():
    env = _env(taker_fee_pct=0.001, cost_penalty_scale=5.0)
    env.reset()
    _, r_flat, *_ = env.step(np.array([0.0, 0.0]))   # stays flat, no cost
    env2 = _env(taker_fee_pct=0.001)
    env2.reset()
    _, r_ref, *_ = env2.step(np.array([0.0, 0.0]))
    assert r_flat == r_ref


# --------------------------------------------------------------------------- #
# 4. factory dispatch + linear-core isolation
# --------------------------------------------------------------------------- #

def _factory_arrays(T=300, N=3):
    rng = np.random.default_rng(3)
    price = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, (T, N)), axis=0)
    return {
        "price_ary": price,
        "tech_ary": rng.normal(0, 1, (T, N * 7)).astype(np.float32),
        "vol_ary": np.full((T, N), 0.10),
        "carry_ary": np.zeros((T, N)),
        "volume_ary": np.full((T, N), 1e9),
        "timestamps": (np.arange(T, dtype=np.int64) * 86400),
        "conviction_ary": np.sign(rng.normal(0, 1, (T, N))),
    }


def test_factory_maps_cost_lever_keys():
    cfg = {"env": {"no_trade_band": 0.03, "rebalance_interval": 5,
                   "cost_penalty_scale": 1.5}}
    env = factory.make_allocator_env(_factory_arrays(), cfg)
    assert env.no_trade_band == 0.03
    assert env.rebalance_interval == 5
    assert env.cost_penalty_scale == 1.5


def test_factory_lever_overrides_take_precedence():
    cfg = {"env": {"no_trade_band": 0.03}}
    env = factory.make_allocator_env(
        _factory_arrays(), cfg,
        overrides={"no_trade_band": 0.07, "rebalance_interval": 21})
    assert env.no_trade_band == 0.07
    assert env.rebalance_interval == 21


def test_linear_core_immune_to_execution_levers():
    """The gate baseline must be the VALIDATED monthly linear core: config /
    HPO-override execution levers may not leak into evaluate_linear_core."""
    arrays = _factory_arrays(T=400)
    base_cfg = {"env": {"taker_fee": 0.0002}}
    lever_cfg = {"env": {"taker_fee": 0.0002, "no_trade_band": 0.1,
                         "rebalance_interval": 5, "cost_penalty_scale": 3.0}}
    m_base = factory.evaluate_linear_core(arrays, base_cfg)
    m_lever = factory.evaluate_linear_core(
        arrays, lever_cfg,
        overrides={"no_trade_band": 0.2, "rebalance_interval": 21})
    assert m_base == m_lever


# --------------------------------------------------------------------------- #
# 5. F1 — dollar-volume slippage participation (Fable 2026-06-11)
# --------------------------------------------------------------------------- #

def _cost_env(dollar_volume: float, T=60, n=1, **kw) -> MultiAssetAllocatorEnv:
    """Single-asset env with a CONSTANT dollar-volume array and unit vol-scaling
    (weight == action). Slippage left ON; fee OFF so cumulative_fees == slippage.
    ``volume_ary`` here is dollar volume (shares × price) per the F1 contract."""
    price = synthetic_prices(T=T, n=n, seed=5)
    arrays = build_arrays(price, vol=np.full((T, n), 0.10), volume=dollar_volume)
    params = dict(
        target_vol_asset=0.10, lev_cap=2.0, max_gross_exposure=1e9,
        min_trade_pct=0.0, taker_fee_pct=0.0, turnover_penalty=0.0,
        reward_type="simple",
    )
    params.update(kw)
    return MultiAssetAllocatorEnv(**arrays, **params)


def test_slippage_participation_uses_dollar_volume():
    """Realized slippage == notional·(base + impact·notional/dollar_vol)·1e-4, i.e.
    participation = order_notional / DOLLAR volume. taker_fee=0 ⇒ cumulative_fees is
    slippage only; info['turnover'] gives the exact applied weight-delta (robust to
    the action→weight transform)."""
    V, base, impact = 1_000_000.0, 1.0, 10.0
    env = _cost_env(V, slippage_base_bps=base, slippage_impact_bps=impact)
    env.reset()
    _, _, _, _, info = env.step(np.array([0.5]))
    notional = info["turnover"] * env.initial_capital   # from flat: pv_before == initial_capital
    participation = notional / V
    expected = notional * (base + impact * participation) * 1e-4
    assert notional > 0 and participation > 1e-6, "impact term must actually be exercised"
    assert abs(env.cumulative_fees - expected) < 1e-9


def test_slippage_decreases_with_dollar_volume():
    """Impact is participation-based: deeper (larger dollar-volume) markets cost less,
    and a near-infinite-depth market charges base slippage only."""
    def slip(V):
        env = _cost_env(V, slippage_base_bps=1.0, slippage_impact_bps=50.0)
        env.reset()
        env.step(np.array([0.5]))
        return env.cumulative_fees

    s_thin, s_mid, s_deep = slip(1e5), slip(1e7), slip(1e12)
    assert s_thin > s_mid > s_deep, "slippage must fall as dollar volume rises"
    # Deep-market slippage → base only (impact participation → 0).
    env = _cost_env(1e12, slippage_base_bps=1.0, slippage_impact_bps=50.0)
    env.reset()
    _, _, _, _, info = env.step(np.array([0.5]))
    base_only = info["turnover"] * env.initial_capital * 1.0 * 1e-4
    assert abs(s_deep - base_only) / base_only < 1e-3


# --------------------------------------------------------------------------- #
# 4. evaluate_linear_core_levered — the v1.1 DIAGNOSTIC arm
# --------------------------------------------------------------------------- #
# Bidirectional by construction (the "silence is not a result" rule): a levered
# arm that silently forced its levers off would pass a one-directional
# "levers-off matches the gate baseline" check forever. So assert BOTH that it
# matches with levers off AND that it genuinely MOVES with levers on.

def test_levered_core_matches_gate_baseline_when_levers_off():
    arrays = _factory_arrays()
    cfg = {"env": {"taker_fee": 0.0002, "slippage_base_bps": 1.0,
                   "target_vol_asset": 0.10, "max_gross_exposure": 3.0}}
    off = {"no_trade_band": 0.0, "rebalance_interval": 1, "cost_penalty_scale": 0.0}
    base = factory.evaluate_linear_core(arrays, cfg, overrides=off)
    lev = factory.evaluate_linear_core_levered(arrays, cfg, overrides=off)
    assert lev == base


def test_levered_core_actually_responds_to_the_levers():
    """The other direction: with levers ON the diagnostic must diverge from the
    gate baseline (which forces them off) — lower turnover, different metrics.
    If this ever passes trivially the arm has stopped measuring anything."""
    arrays = _factory_arrays()
    cfg = {"env": {"taker_fee": 0.0002, "slippage_base_bps": 1.0,
                   "target_vol_asset": 0.10, "max_gross_exposure": 3.0}}
    on = {"no_trade_band": 0.25, "rebalance_interval": 21}
    base = factory.evaluate_linear_core(arrays, cfg, overrides=on)
    lev = factory.evaluate_linear_core_levered(arrays, cfg, overrides=on)
    assert base["turnover_ann"] > 0.0
    assert lev["turnover_ann"] < base["turnover_ann"]
    assert lev["net_sharpe"] != base["net_sharpe"]


# --------------------------------------------------------------------------- #
# 5. exposure diagnostics — telling an edge from an abstention
# --------------------------------------------------------------------------- #

def test_exposure_metrics_distinguish_flat_from_invested():
    """Bidirectional: a flat book must report zero exposure, an invested one
    must not. Without this a near-cash policy posts a healthy Sharpe and reads
    as an edge (the window-0 v1.1 signature: +0.53%/yr at 1.14 turnover)."""
    flat = factory._exposure_metrics([[0.0, 0.0]] * 5, 5)
    assert flat["exposure_frac"] == 0.0
    assert flat["gross_exposure_mean"] == 0.0

    invested = factory._exposure_metrics([[0.5, -0.25]] * 5, 5)
    assert invested["exposure_frac"] == 1.0
    assert invested["gross_exposure_mean"] == 0.75

    half = factory._exposure_metrics([[0.4, 0.0], [0.0, 0.0]] * 3, 6)
    assert half["exposure_frac"] == 0.5


def test_exposure_metrics_absent_when_weights_not_captured():
    """Callers that never collected weights get explicit None, not a fake 0.0 —
    a silent zero would read as 'flat book' and libel an active policy."""
    m = factory._metrics([0.01, -0.005], [0.1, 0.1], [100.0, 101.0, 100.5])
    assert m["gross_exposure_mean"] is None
    assert m["exposure_frac"] is None


def test_linear_core_reports_exposure_through_the_drive():
    """The real path: evaluate_linear_core must surface exposure, not None."""
    arrays = _factory_arrays()
    cfg = {"env": {"taker_fee": 0.0002, "target_vol_asset": 0.10,
                   "max_gross_exposure": 3.0}}
    m = factory.evaluate_linear_core(arrays, cfg)
    assert m["exposure_frac"] is not None and m["exposure_frac"] > 0.0
    assert m["gross_exposure_mean"] > 0.0
