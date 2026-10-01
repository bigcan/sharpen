"""PnL / SHORT-ACCT tests for MultiAssetAllocatorEnv.

The fixed-entry-notional accounting is copied from CryptoPerpEnv, and an add books the
share-weighted VWAP as its entry price. These tests guard the copy: shorts must
lose/gain symmetrically with longs (no ``notional_debt`` inflation), closing realizes the
PnL into margin (buyback in equity), and the book stays share-exact across adds.
"""
from __future__ import annotations

import numpy as np

from sharpen.envs.multi_asset_allocator_env import MultiAssetAllocatorEnv

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


def test_add_at_a_higher_price_books_the_share_weighted_vwap():
    """Audit T4-10 hand case: buy $100 at $1, add $100 at $2, mark at $2. The book holds
    150 shares at a $1.333 VWAP, so equity is 1,100. The notional-weighted mean entry of
    $1.50 booked 133 shares and 1,066.67."""
    env = _single_asset_env([1.0, 1.0, 2.0, 2.0], 0.1, initial_capital=1_000.0)
    env.reset()
    env.step(np.array([0.1]))                         # $100 at price[1] = 1
    env.step(np.array([0.2]))                         # +$100 at price[2] = 2 (sized on pv at 1)
    _, _, _, _, info = env.step(np.array([0.2]))      # hold; mark at price[3] = 2
    assert abs(env.entry_prices[0] - 200.0 / 150.0) < 1e-9
    assert abs(info["portfolio_value"] - 1_100.0) < 1e-6


def _share_ledger(price: np.ndarray, W: np.ndarray, capital: float) -> np.ndarray:
    """An independent cash-and-shares book trading the env's weight path. Each change fills
    at close k+1 and is sized on the equity marked at close k (the env's ``pv_before``). An
    add buys ``|dw| * pv_before`` of shares, a reduction sells the label's fraction of the
    shares, and an open, flip or close first sells everything held."""
    n = price.shape[1]
    shares, held, cash, equity = np.zeros(n), np.zeros(n), capital, []
    for k, w in enumerate(W):
        p0, p1 = price[k], price[k + 1]
        pv_before = cash + shares @ p0
        for i in range(n):
            wo, wn = held[i], w[i]
            if abs(wn) < 1e-8 or abs(wo) < 1e-8 or np.sign(wo) != np.sign(wn):
                cash += shares[i] * p1[i]
                shares[i] = wn * pv_before / p1[i] if abs(wn) >= 1e-8 else 0.0
                cash -= shares[i] * p1[i]
            elif abs(wn) > abs(wo):
                bought = (wn - wo) * pv_before / p1[i]
                shares[i] += bought
                cash -= bought * p1[i]
            elif abs(wn) < abs(wo):
                sold = shares[i] * (abs(wo) - abs(wn)) / abs(wo)
                shares[i] -= sold
                cash += sold * p1[i]
        held = w.copy()
        equity.append(cash + shares @ p1)
    return np.array(equity)


def test_fixed_notional_book_is_share_exact():
    """Along a random path of opens, adds, reductions, flips and closes, the env's equity
    equals the independent share ledger at every step. The notional-weighted mean entry
    price parts company with it at the first add at a new price."""
    rng = np.random.default_rng(3)
    T, n = 80, 3
    price = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.02, (T, n)), axis=0)
    W = np.zeros((T - 1, n))
    for k in range(1, T - 1):
        W[k] = np.clip(W[k - 1] + rng.normal(0.0, 0.3, n) * (rng.random(n) < 0.6), -0.8, 0.8)
        W[k][rng.random(n) < 0.05] = 0.0                       # occasional closes
    env = MultiAssetAllocatorEnv(
        **build_arrays(price, vol=np.full((T, n), 0.10)), target_vol_asset=0.10, lev_cap=2.0,
        max_gross_exposure=10.0, taker_fee_pct=0.0, slippage_base_bps=0.0,
        slippage_impact_bps=0.0, min_trade_pct=0.0, turnover_penalty=0.0, reward_type="simple",
        circuit_breaker_threshold=0.0,
    )
    env.reset()
    equity = [env.step(w)[4]["portfolio_value"] for w in W]
    np.testing.assert_allclose(equity, _share_ledger(price, W, env.initial_capital), rtol=1e-9)
