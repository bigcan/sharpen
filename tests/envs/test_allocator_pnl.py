"""PnL / SHORT-ACCT tests for MultiAssetAllocatorEnv.

The fixed-entry-notional accounting is copied verbatim from CryptoPerpEnv, so these
guard the copy: shorts must lose/gain symmetrically with longs (no ``notional_debt``
inflation), and closing realizes the PnL into margin (buyback in equity).
"""
from __future__ import annotations

import numpy as np

from finrl_pro_ds.envs.multi_asset_allocator_env import MultiAssetAllocatorEnv

from .conftest import build_arrays


def _single_asset_env(prices: list[float], side_conv: float, **kw) -> MultiAssetAllocatorEnv:
    price = np.array(prices, dtype=np.float64).reshape(-1, 1)
    T = len(prices)
    vol = np.full((T, 1), 0.10)                      # scale = target/vol = 1.0
    arrays = build_arrays(price, vol=vol)
    params = dict(
        target_vol_asset=0.10, lev_cap=2.0, max_gross_exposure=5.0,
        taker_fee_pct=0.0, slippage_base_bps=0.0, slippage_impact_bps=0.0,
        min_trade_pct=0.0, turnover_penalty=0.0, reward_type="simple",
        circuit_breaker_threshold=0.0,
    )
    params.update(kw)
    return MultiAssetAllocatorEnv(**arrays, **params)


def test_no_notional_debt_attribute():
    """SHORT-ACCT: the env must not carry a notional_debt accumulator — shorts are
    accounted by fixed entry notional + buyback in equity."""
    env = _single_asset_env([100, 100, 120], -1.0)
    assert not hasattr(env, "notional_debt")


def test_long_short_pnl_symmetric_fixed_notional():
    """Identical price path: a short's PnL is exactly the negative of the long's."""
    prices = [100.0, 100.0, 120.0]                   # +20% move bar1->bar2

    long_env = _single_asset_env(prices, +1.0)
    long_env.reset()
    long_env.step(np.array([1.0]))                   # enter long at price[1]=100
    assert abs(long_env.entry_prices[0] - 100.0) < 1e-9
    _, _, _, _, info_l = long_env.step(np.array([1.0]))   # hold into price[2]=120

    short_env = _single_asset_env(prices, -1.0)
    short_env.reset()
    short_env.step(np.array([-1.0]))                 # enter short at price[1]=100
    _, _, _, _, info_s = short_env.step(np.array([-1.0]))

    # Long gains +20000 on a 100k notional; short loses the mirror image.
    assert abs(info_l["portfolio_value"] - 120_000.0) < 1e-6
    assert abs(info_s["portfolio_value"] - 80_000.0) < 1e-6
    assert abs(info_l["unrealized_pnl"] + info_s["unrealized_pnl"]) < 1e-6


def test_short_gains_when_price_falls():
    env = _single_asset_env([100.0, 100.0, 80.0], -1.0)
    env.reset()
    env.step(np.array([-1.0]))                       # short at 100
    _, _, _, _, info = env.step(np.array([-1.0]))    # price -> 80 (favorable)
    assert abs(info["portfolio_value"] - 120_000.0) < 1e-6   # +20% on the short


def test_closing_short_realizes_into_margin():
    """Closing flattens positions/notionals and books the loss into realized PnL."""
    env = _single_asset_env([100.0, 100.0, 120.0, 120.0], -1.0)
    env.reset()
    env.step(np.array([-1.0]))                        # short at 100
    env.step(np.array([-1.0]))                        # hold to 120 (unrealized -20k)
    _, _, _, _, info = env.step(np.array([0.0]))      # close at 120
    assert abs(env.positions[0]) < 1e-12
    assert abs(env.entry_notionals[0]) < 1e-12
    assert abs(env.realized_pnl + 20_000.0) < 1e-6    # -20k realized
    assert abs(info["portfolio_value"] - 80_000.0) < 1e-6
    assert abs(info["unrealized_pnl"]) < 1e-9


def test_position_flip_realizes_then_reenters():
    """Long -> short flip closes the long (realizing its PnL) and opens a fresh short."""
    env = _single_asset_env([100.0, 100.0, 110.0, 110.0], +1.0)
    env.reset()
    env.step(np.array([1.0]))                         # long at 100
    env.step(np.array([1.0]))                         # hold to 110 (+10k unrealized)
    env.step(np.array([-1.0]))                        # flip to short at 110
    assert env.positions[0] < 0
    assert abs(env.entry_prices[0] - 110.0) < 1e-9    # fresh short entry price
    assert abs(env.realized_pnl - 10_000.0) < 1e-6    # long gain realized on flip
