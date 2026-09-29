"""Entry-price accounting for CryptoPerpEnv: an add books the share-weighted VWAP.

With fixed-entry-notional P&L, ``entry_notional / entry_price`` is the share count, so an add
must book the price at which the combined shares cost the combined notional. The
notional-weighted mean over-stated it whenever the fill prices differed (TAILWIND Tier-2
T4-10): longs were under-credited, shorts over-credited.
"""
from __future__ import annotations

import numpy as np
import pytest

from sharpen.crypto.envs.crypto_perp_env import CryptoPerpEnv


def _single_asset_env(prices: list[float]) -> CryptoPerpEnv:
    """Cost-free, funding-free single-asset env on a fixed price path."""
    T = len(prices)
    return CryptoPerpEnv(
        price_ary=np.array(prices, dtype=np.float64).reshape(-1, 1),
        tech_ary=np.zeros((T, 4), dtype=np.float32),
        funding_rate_ary=np.zeros((T, 1)),
        volume_ary=np.full((T, 1), 1e6),
        timestamps=1704067200 + 3600 * np.arange(T, dtype=np.int64),  # 2024-01-01 00:00 UTC
        initial_capital=1_000.0,
        maker_fee_pct=0.0,
        taker_fee_pct=0.0,
        slippage_base_bps=0.0,
        slippage_impact_bps=0.0,
    )


@pytest.mark.parametrize("side", [1.0, -1.0])
def test_add_at_a_higher_price_books_the_share_weighted_vwap(side):
    """Audit T4-10 hand case: buy $100 at $1, add $100 at $2, mark at $2. The book holds
    150 shares at a $1.333 VWAP, so the P&L is +100 (-100 short). The notional-weighted
    mean entry of $1.50 booked 133 shares and +66.67."""
    env = _single_asset_env([1.0, 1.0, 2.0, 2.0])
    env.reset()
    env.step(np.array([0.1 * side]))                  # $100 at price[1] = 1
    env.step(np.array([0.2 * side]))                  # +$100 at price[2] = 2 (sized on pv at 1)
    _, _, _, _, info = env.step(np.array([0.2 * side]))  # hold; mark at price[3] = 2
    assert env.entry_prices[0] == pytest.approx(200.0 / 150.0, abs=1e-9)
    assert info["portfolio_value"] - 1_000.0 == pytest.approx(100.0 * side, abs=1e-6)
