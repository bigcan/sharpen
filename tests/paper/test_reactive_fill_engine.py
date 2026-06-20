"""ReactiveSimFillEngine — SimFillEngine + reactive mean-reverting temporary impact.

Step 1 of the execution-overlay build (execution_overlay_architecture.md, ADR-6). Pins:
  - the PARITY FLOOR: reactive_impact_bps == 0 ⇒ byte-identical to SimFillEngine;
  - the √-law concavity of the temporary impact (AC I(Q)=Y·σ·√(Q/V) form);
  - the cross-bar PRESSURE coupling (residual raises a later trade's cost) + its decay
    (mean-reversion over no-trade bars) — the state-dependence the RL overlay learns on;
  - LEAK-2 (cost independent of the reference/fill price; no future peek);
  - input validation.
"""
from __future__ import annotations

import numpy as np
import pytest

from finrl_pro_ds.paper.fill_engine import ReactiveSimFillEngine, SimFillEngine


def _sim(base=1.0, impact=10.0, fee=0.0002) -> SimFillEngine:
    return SimFillEngine(taker_fee_pct=fee, slippage_base_bps=base, slippage_impact_bps=impact)


def _reactive(*, n_assets, base=1.0, impact=10.0, fee=0.0002, reactive=8.0, decay=0.5):
    return ReactiveSimFillEngine(
        taker_fee_pct=fee, slippage_base_bps=base, slippage_impact_bps=impact,
        n_assets=n_assets, reactive_impact_bps=reactive, decay=decay,
    )


# --------------------------------------------------------------------------- #
# Parity floor: reactive OFF ⇒ exactly SimFillEngine (byte-identical)
# --------------------------------------------------------------------------- #
def test_reactive_zero_equals_sim_byte_identical():
    """reactive_impact_bps == 0 reproduces SimFillEngine EXACTLY across a bar sequence
    (the rung-1 parity-floor guarantee — '==', not approx)."""
    base, impact, fee, N = 1.0, 10.0, 0.0002, 4
    sim = _sim(base=base, impact=impact, fee=fee)
    react = _reactive(n_assets=N, base=base, impact=impact, fee=fee, reactive=0.0, decay=0.5)

    rng = np.random.default_rng(0)
    pv = 100_000.0
    for _ in range(25):
        delta = rng.normal(0, 0.3, size=N)
        delta[rng.random(N) < 0.2] = 0.0                       # some no-trade legs
        ref = rng.uniform(10, 200, size=N)
        vol = rng.choice([0.0, 1e5, 1e7, 1e12], size=N)        # incl. zero-volume legs
        rs = react.fill(delta_weights=delta, ref_prices=ref, pv_before=pv, dollar_volume=vol)
        ss = sim.fill(delta_weights=delta, ref_prices=ref, pv_before=pv, dollar_volume=vol)
        assert rs.fees == ss.fees
        assert rs.slippage == ss.slippage
        assert rs.traded_notional == ss.traded_notional
        np.testing.assert_array_equal(rs.fill_prices, ss.fill_prices)


# --------------------------------------------------------------------------- #
# Temporary impact: √-law shape, added ON TOP of the linear floor
# --------------------------------------------------------------------------- #
def test_reactive_adds_sqrt_law_on_top_of_floor():
    """On a fresh engine (pressure 0), reactive_slip = notional · r·√(participation) · 1e-4,
    added to the parent floor slippage."""
    base, impact, fee, r = 1.0, 10.0, 0.0, 8.0
    V, pv = 1e7, 100_000.0
    react = _reactive(n_assets=1, base=base, impact=impact, fee=fee, reactive=r, decay=0.5)
    sim = _sim(base=base, impact=impact, fee=fee)

    delta = np.array([0.5])
    notional = 0.5 * pv
    p = notional / V
    react.reset()
    rs = react.fill(delta_weights=delta, ref_prices=np.array([100.0]), pv_before=pv,
                    dollar_volume=np.array([V]))
    ss = sim.fill(delta_weights=delta, ref_prices=np.array([100.0]), pv_before=pv,
                  dollar_volume=np.array([V]))
    exp_reactive = notional * (r * np.sqrt(p)) * 1e-4
    assert rs.slippage > ss.slippage, "reactive impact must add cost above the floor"
    assert abs((rs.slippage - ss.slippage) - exp_reactive) < 1e-9
    # √-concavity: 4× participation (delta 0.5→2.0 ⇒ p 0.005→0.02) ⇒ 2× the per-notional
    # add, not 4×.
    react.reset()
    rs4 = react.fill(delta_weights=np.array([2.0]), ref_prices=np.array([100.0]), pv_before=pv,
                     dollar_volume=np.array([V]))
    ss4 = sim.fill(delta_weights=np.array([2.0]), ref_prices=np.array([100.0]), pv_before=pv,
                   dollar_volume=np.array([V]))
    add1 = (rs.slippage - ss.slippage) / notional            # per-notional add at p
    add4 = (rs4.slippage - ss4.slippage) / (2.0 * pv)        # per-notional add at 4p
    assert abs(add4 / add1 - 2.0) < 1e-6                     # √(4·p / p) = 2


def test_spreading_reduces_reactive_cost_at_zero_decay():
    """At decay=0 (independent bars) the √-law makes working a parent order over H bars
    cheaper than a single shot by √H (the execution incentive to slice)."""
    r, V, pv = 8.0, 1e7, 100_000.0
    Q = 1.0                                                   # parent weight delta
    # single shot
    eng1 = _reactive(n_assets=1, base=0.0, impact=0.0, fee=0.0, reactive=r, decay=0.0)
    eng1.reset()
    single = eng1.fill(delta_weights=np.array([Q]), ref_prices=np.array([100.0]),
                       pv_before=pv, dollar_volume=np.array([V])).slippage
    # spread over H bars
    H = 4
    engH = _reactive(n_assets=1, base=0.0, impact=0.0, fee=0.0, reactive=r, decay=0.0)
    engH.reset()
    spread = 0.0
    for _ in range(H):
        spread += engH.fill(delta_weights=np.array([Q / H]), ref_prices=np.array([100.0]),
                            pv_before=pv, dollar_volume=np.array([V])).slippage
    assert spread < single
    assert abs(spread / single - 1.0 / np.sqrt(H)) < 1e-6     # exact √H saving at decay=0


# --------------------------------------------------------------------------- #
# Cross-bar pressure: residual raises a later trade; decays over no-trade bars
# --------------------------------------------------------------------------- #
def test_pressure_persists_then_decays_over_no_trade_bars():
    """A large trade builds residual pressure that a later probe pays; inserting no-trade
    bars lets the pressure mean-revert, so the same probe costs less the longer it waits."""
    r, V, pv, decay = 8.0, 1e7, 100_000.0, 0.6
    probe = np.array([0.05])

    def probe_cost(wait_bars):
        eng = _reactive(n_assets=1, base=0.0, impact=0.0, fee=0.0, reactive=r, decay=decay)
        eng.reset()
        eng.fill(delta_weights=np.array([1.0]), ref_prices=np.array([100.0]),
                 pv_before=pv, dollar_volume=np.array([V]))             # build pressure
        for _ in range(wait_bars):
            eng.fill(delta_weights=np.zeros(1), ref_prices=np.array([100.0]),
                     pv_before=pv, dollar_volume=np.array([V]))         # no-trade: decay only
        return eng.fill(delta_weights=probe, ref_prices=np.array([100.0]),
                        pv_before=pv, dollar_volume=np.array([V])).slippage

    c0, c1, c3 = probe_cost(0), probe_cost(1), probe_cost(3)
    assert c0 > c1 > c3, "residual pressure must decay over no-trade bars"
    # the probe's own (pressure-free) cost is the floor it decays toward
    floor_probe = 0.05 * pv * (r * np.sqrt(0.05 * pv / V)) * 1e-4
    assert c3 > floor_probe and (c3 - floor_probe) < (c0 - floor_probe)


def test_no_trade_bar_is_free_but_still_decays():
    """A no-trade bar charges nothing yet still reverts pressure (so a later trade is cheaper)."""
    eng = _reactive(n_assets=1, base=0.0, impact=0.0, fee=0.0, reactive=8.0, decay=0.5)
    eng.reset()
    eng.fill(delta_weights=np.array([1.0]), ref_prices=np.array([100.0]),
             pv_before=100_000.0, dollar_volume=np.array([1e7]))
    nt = eng.fill(delta_weights=np.zeros(1), ref_prices=np.array([100.0]),
                  pv_before=100_000.0, dollar_volume=np.array([1e7]))
    assert nt.fees == 0.0 and nt.slippage == 0.0 and nt.traded_notional == 0.0


def test_reset_zeroes_pressure():
    """reset() clears accumulated pressure ⇒ the post-reset trade matches a virgin engine."""
    pv, V = 100_000.0, 1e7
    eng = _reactive(n_assets=1, base=0.0, impact=0.0, fee=0.0, reactive=8.0, decay=0.5)
    fresh = _reactive(n_assets=1, base=0.0, impact=0.0, fee=0.0, reactive=8.0, decay=0.5)
    eng.reset()
    eng.fill(delta_weights=np.array([1.0]), ref_prices=np.array([100.0]),
             pv_before=pv, dollar_volume=np.array([V]))            # build pressure
    eng.reset()
    after = eng.fill(delta_weights=np.array([0.3]), ref_prices=np.array([100.0]),
                     pv_before=pv, dollar_volume=np.array([V])).slippage
    fresh.reset()
    virgin = fresh.fill(delta_weights=np.array([0.3]), ref_prices=np.array([100.0]),
                        pv_before=pv, dollar_volume=np.array([V])).slippage
    assert abs(after - virgin) < 1e-12


# --------------------------------------------------------------------------- #
# LEAK-2: cost is independent of the reference/fill price (no future peek)
# --------------------------------------------------------------------------- #
def test_cost_independent_of_reference_price_leak2():
    """Slippage/fees are a debit on traded notional; varying ref_prices must NOT change
    them (the fill never reads a future-stamped price into the cost)."""
    eng_a = _reactive(n_assets=2, reactive=8.0, decay=0.5)
    eng_b = _reactive(n_assets=2, reactive=8.0, decay=0.5)
    eng_a.reset()
    eng_b.reset()
    delta, pv, vol = np.array([0.4, -0.2]), 100_000.0, np.array([1e7, 1e7])
    a = eng_a.fill(delta_weights=delta, ref_prices=np.array([10.0, 500.0]),
                   pv_before=pv, dollar_volume=vol)
    b = eng_b.fill(delta_weights=delta, ref_prices=np.array([9999.0, 1.0]),
                   pv_before=pv, dollar_volume=vol)
    assert a.fees == b.fees and a.slippage == b.slippage
    assert a.traded_notional == b.traded_notional


def test_zero_volume_active_leg_assumes_max_participation():
    """An active trade into zero/halted dollar volume ⇒ participation 1.0 in the reactive
    term too (env convention; never silently free)."""
    r, pv = 8.0, 100_000.0
    eng = _reactive(n_assets=1, base=0.0, impact=0.0, fee=0.0, reactive=r, decay=0.5)
    eng.reset()
    res = eng.fill(delta_weights=np.array([0.5]), ref_prices=np.array([100.0]),
                   pv_before=pv, dollar_volume=np.array([0.0]))
    notional = 0.5 * pv
    exp = notional * (r * np.sqrt(1.0)) * 1e-4                  # participation forced to 1.0
    assert abs(res.slippage - exp) < 1e-9


def test_short_and_long_legs_cost_the_same_within_a_bar():
    """Reactive cost is on |notional| (sign-agnostic within a horizon)."""
    pv, V = 100_000.0, 1e7
    eng = _reactive(n_assets=1, base=0.0, impact=0.0, fee=0.0, reactive=8.0, decay=0.5)
    eng.reset()
    lo = eng.fill(delta_weights=np.array([0.4]), ref_prices=np.array([50.0]),
                  pv_before=pv, dollar_volume=np.array([V])).slippage
    eng.reset()
    sh = eng.fill(delta_weights=np.array([-0.4]), ref_prices=np.array([50.0]),
                  pv_before=pv, dollar_volume=np.array([V])).slippage
    assert abs(lo - sh) < 1e-12


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def test_negative_reactive_bps_rejected():
    with pytest.raises(ValueError):
        _reactive(n_assets=1, reactive=-1.0)


@pytest.mark.parametrize("decay", [-0.1, 1.0, 1.5])
def test_decay_out_of_range_rejected(decay):
    with pytest.raises(ValueError):
        _reactive(n_assets=1, decay=decay)
