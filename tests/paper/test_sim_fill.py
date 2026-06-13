"""SimFillEngine — F1-correct participation-based slippage + fee model (rung 1).

Pins the fill cost to the same formula the env charges
(``MultiAssetAllocatorEnv._calc_transaction_costs_fast``); the keystone parity test
then proves end-to-end that this reproduces the env's realized cost (cost_drift_ratio
== 1.0).
"""
from __future__ import annotations

import numpy as np

from finrl_pro_ds.paper.fill_engine import SimFillEngine


def _engine(base=1.0, impact=10.0, fee=0.0002) -> SimFillEngine:
    return SimFillEngine(taker_fee_pct=fee, slippage_base_bps=base, slippage_impact_bps=impact)


def test_participation_uses_dollar_volume():
    """fee + slippage = Σnotional·fee + Σnotional·(base+impact·notional/dollar_vol)·1e-4."""
    V, base, impact, fee = 1_000_000.0, 1.0, 10.0, 0.0002
    eng = _engine(base=base, impact=impact, fee=fee)
    pv = 100_000.0
    delta = np.array([0.5])                      # one asset, weight delta 0.5
    res = eng.fill(delta_weights=delta, ref_prices=np.array([100.0]),
                   pv_before=pv, dollar_volume=np.array([V]))
    notional = 0.5 * pv
    participation = notional / V
    exp_fee = notional * fee
    exp_slip = notional * (base + impact * participation) * 1e-4
    assert participation > 1e-6, "impact term must be exercised"
    assert abs(res.fees - exp_fee) < 1e-9
    assert abs(res.slippage - exp_slip) < 1e-9
    assert abs(res.realized_cost - (exp_fee + exp_slip)) < 1e-9
    assert res.traded_notional == notional


def test_slippage_decreases_with_dollar_volume():
    eng = _engine(base=1.0, impact=50.0, fee=0.0)
    pv = 100_000.0

    def slip(V):
        return eng.fill(delta_weights=np.array([0.5]), ref_prices=np.array([100.0]),
                        pv_before=pv, dollar_volume=np.array([V])).slippage

    s_thin, s_mid, s_deep = slip(1e5), slip(1e7), slip(1e12)
    assert s_thin > s_mid > s_deep, "slippage must fall as dollar volume rises"
    # near-infinite depth ⇒ base-only slippage.
    base_only = 0.5 * pv * 1.0 * 1e-4
    assert abs(s_deep - base_only) / base_only < 1e-3


def test_zero_volume_assumes_max_impact():
    """Missing/zero dollar volume ⇒ participation ratio 1.0 (conservative env default),
    NOT silently free."""
    eng = _engine(base=1.0, impact=10.0, fee=0.0)
    pv = 100_000.0
    res = eng.fill(delta_weights=np.array([0.5]), ref_prices=np.array([100.0]),
                   pv_before=pv, dollar_volume=np.array([0.0]))
    notional = 0.5 * pv
    exp = notional * (1.0 + 10.0 * 1.0) * 1e-4       # ratio forced to 1.0
    assert abs(res.slippage - exp) < 1e-9


def test_short_path_costs_equal_long_path():
    """Cost is on |notional| — a short delta costs the same as the long of equal size."""
    eng = _engine()
    pv, V = 100_000.0, 1e7
    long = eng.fill(delta_weights=np.array([0.4]), ref_prices=np.array([50.0]),
                    pv_before=pv, dollar_volume=np.array([V]))
    short = eng.fill(delta_weights=np.array([-0.4]), ref_prices=np.array([50.0]),
                     pv_before=pv, dollar_volume=np.array([V]))
    assert abs(long.realized_cost - short.realized_cost) < 1e-12
    assert long.traded_notional == short.traded_notional


def test_no_trade_is_free_and_fills_at_reference():
    eng = _engine()
    ref = np.array([100.0, 50.0])
    res = eng.fill(delta_weights=np.zeros(2), ref_prices=ref,
                   pv_before=100_000.0, dollar_volume=np.array([1e7, 1e7]))
    assert res.fees == 0.0 and res.slippage == 0.0 and res.traded_notional == 0.0
    np.testing.assert_array_equal(res.fill_prices, ref)


def test_dust_delta_below_eps_not_traded():
    """|Δw| < 1e-8 mirrors the env's active-trade epsilon (no cost)."""
    eng = _engine()
    res = eng.fill(delta_weights=np.array([1e-12, 0.0]), ref_prices=np.array([100.0, 100.0]),
                   pv_before=100_000.0, dollar_volume=np.array([1e7, 1e7]))
    assert res.realized_cost == 0.0
