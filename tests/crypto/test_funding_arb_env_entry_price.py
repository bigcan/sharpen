"""Entry-price accounting for FundingArbEnv: an add books each leg's share-weighted VWAP.

Each leg's P&L is +/- notional * (price / entry - 1), so ``notional / entry`` is the leg's
share count, and an add must book the price at which the combined shares cost the combined
notional. The notional-weighted mean over-stated it whenever the fill prices differed
(TAILWIND Tier-2 T4-10), so each leg held fewer shares than it bought and its current
notional, which sizes funding, borrow and net delta, was under-stated.

The legs are opposite-signed, so on identical spot and perp paths their errors cancel in the
net basis P&L: a net-P&L test on such paths passes under either rule. These tests pin each
leg: its entry price, its funding or borrow charge, and the net P&L when only it moves.
"""
from __future__ import annotations

import numpy as np
import pytest

from sharpen.crypto.envs.funding_arb_env import FundingArbEnv

_COST_FREE = dict(
    spot_taker_fee_pct=0.0,
    perp_taker_fee_pct=0.0,
    slippage_base_bps=0.0,
    slippage_impact_bps=0.0,
    spot_borrow_rate_hourly=0.0,
)


def _single_pair_env(spot, perp, *, start_hour=0, funding_rate=0.0, **overrides) -> FundingArbEnv:
    """Cost-free single-pair env on fixed spot and perp paths; bar i is at UTC hour start_hour+i."""
    T = len(spot)
    return FundingArbEnv(
        spot_price_ary=np.array(spot, dtype=np.float64).reshape(-1, 1),
        perp_price_ary=np.array(perp, dtype=np.float64).reshape(-1, 1),
        funding_rate_ary=np.full((T, 1), funding_rate),
        spot_volume_ary=np.full((T, 1), 1e6),
        perp_volume_ary=np.full((T, 1), 1e6),
        tech_ary=np.zeros((T, 15), dtype=np.float32),
        timestamps=1704067200 + 3600 * (start_hour + np.arange(T, dtype=np.int64)),  # 2024-01-01
        initial_capital=1_000.0,
        **{**_COST_FREE, **overrides},
    )


def _open_add_hold(env: FundingArbEnv, side: float) -> dict:
    """$100 per leg at price[1], +$100 per leg at price[2] (sized on the equity at price[1]),
    then hold one bar. Returns the hold bar's info."""
    env.reset()
    env.step(np.array([0.1 * side]))
    env.step(np.array([0.2 * side]))
    return env.step(np.array([0.2 * side]))[4]


@pytest.mark.parametrize("side", [1.0, -1.0])
def test_add_books_the_share_weighted_vwap_on_both_legs(side):
    """Audit T4-10 hand case on each leg: $100 at $1, then $100 at $2, is 150 shares for $200,
    a $1.333 VWAP. The notional-weighted mean booked $1.50, i.e. 133 shares. Spot and perp
    move together, so the net basis P&L is zero under either rule and cannot see the entry."""
    env = _single_pair_env([1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0])
    info = _open_add_hold(env, side)
    assert env.spot_notionals[0] == pytest.approx(200.0)
    assert env.perp_notionals[0] == pytest.approx(200.0)
    assert env.spot_entry_prices[0] == pytest.approx(200.0 / 150.0, abs=1e-9)
    assert env.perp_entry_prices[0] == pytest.approx(200.0 / 150.0, abs=1e-9)
    assert info["portfolio_value"] == pytest.approx(1_000.0, abs=1e-6)  # blind: legs cancel


@pytest.mark.parametrize("side", [1.0, -1.0])
@pytest.mark.parametrize("moving_leg", ["spot", "perp"])
def test_net_basis_pnl_when_only_one_leg_moves(moving_leg, side):
    """Only one leg's price doubles, so the other leg's entry is exact under either rule and
    the net basis P&L is the moving leg's: 150 shares bought for $200 and marked at $2 is
    +/-100 (long spot, short perp for side > 0). The notional-weighted mean gave +/-66.67."""
    moving, flat = [1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0]
    spot, perp = (moving, flat) if moving_leg == "spot" else (flat, moving)
    leg_sign = side if moving_leg == "spot" else -side
    info = _open_add_hold(_single_pair_env(spot, perp), side)
    assert info["portfolio_value"] - 1_000.0 == pytest.approx(100.0 * leg_sign, abs=1e-6)


@pytest.mark.parametrize("side", [1.0, -1.0])
def test_funding_is_charged_on_the_perp_legs_bought_shares(side):
    """Funding settles on the perp leg's current notional: 150 units at $2 = $300 after the
    hand-case add, where the notional-weighted mean entry marked 133 units, $266.67. Spot and
    perp move together, so the net basis P&L cannot see this; the funding charge can."""
    rate = 1e-3
    env = _single_pair_env([1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0],
                           start_hour=5, funding_rate=rate)  # bar 3 settles at 08:00 UTC
    info = _open_add_hold(env, side)
    assert info["funding_applied"]
    # A short perp (side > 0) receives a positive rate; a long perp pays it.
    assert info["step_funding"] == pytest.approx(side * 300.0 * rate, rel=1e-6)


def test_borrow_is_charged_on_the_spot_legs_bought_shares():
    """A reverse arb borrows its short spot leg, charged on the leg's current notional: 150
    units at $2 = $300 after the hand-case add; the notional-weighted mean marked $266.67."""
    rate = 1e-4
    env = _single_pair_env([1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0],
                           spot_borrow_rate_hourly=rate)
    info = _open_add_hold(env, side=-1.0)
    assert info["step_borrow_cost"] == pytest.approx(300.0 * rate, rel=1e-6)
