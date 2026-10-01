"""CryptoPerpEnv carries a held position through a halt, marked at its last observed price.

A bar whose price is not a valid positive number (C2: `price <= 1e-10`, NaN) is a halt. The
env used to force every halted asset's target to 0, which CLOSED a held position at the
halt's zero price: a long lost 100% of its notional and a short gained it. It now does what
CryptoPerpSwingEnv's S531 F1 does, with the mark fixed: a halted holding is carried (target
= position, no trade possible), priced at the asset's last valid close, and repriced on
reactivation. A flat asset still cannot be opened into a halt.
"""
from __future__ import annotations

import numpy as np
import pytest

from sharpen.crypto.envs.crypto_perp_env import CryptoPerpEnv

#          bar:  0      1      2      3    4    5      6
HALT_PATH = [100.0, 100.0, 110.0, 0.0, 0.0, 120.0, 120.0]


def _single_asset_env(prices: list[float], funding_rate: float = 0.0) -> CryptoPerpEnv:
    """Cost-free single-asset env on a fixed price path; bar i is at 00:00 UTC + i hours."""
    T = len(prices)
    return CryptoPerpEnv(
        price_ary=np.array(prices, dtype=np.float64).reshape(-1, 1),
        tech_ary=np.zeros((T, 4), dtype=np.float32),
        funding_rate_ary=np.full((T, 1), funding_rate),
        volume_ary=np.full((T, 1), 1e6),
        timestamps=1704067200 + 3600 * np.arange(T, dtype=np.int64),  # 2024-01-01 00:00 UTC
        initial_capital=1_000.0,
        maker_fee_pct=0.0,
        taker_fee_pct=0.0,
        slippage_base_bps=0.0,
        slippage_impact_bps=0.0,
    )


@pytest.mark.parametrize("halt_action", ["hold", "flatten"])
@pytest.mark.parametrize("side", [1.0, -1.0])
def test_halted_holding_is_carried_at_its_last_close(side, halt_action):
    """$100 at 100, marked at 110 (+/-10), then two zero-price halt bars, then 120.

    Through the halt the book keeps its position and entry and stays marked at 110, whatever
    the agent asks for: a halted market cannot be traded. On reactivation it reprices at 120
    (+/-20). The old C2 mask closed the position at price 0 on the first halt bar: 900 for a
    long, 1,100 for a short, where the book is worth 1,010 / 990.
    """
    env = _single_asset_env(HALT_PATH)
    env.reset()
    w = 0.1 * side
    env.step(np.array([w]))                                    # bar 1: $100 at 100
    _, _, _, _, info = env.step(np.array([w]))                 # bar 2: mark 110
    assert info["portfolio_value"] == pytest.approx(1_000.0 + 10.0 * side, abs=1e-6)

    during_halt = np.array([w if halt_action == "hold" else 0.0])
    for _ in range(2):                                         # bars 3-4: price 0
        _, _, _, _, info = env.step(during_halt)
        assert env.positions[0] == pytest.approx(w, abs=1e-12)
        assert env.entry_prices[0] == pytest.approx(100.0, abs=1e-9)
        assert info["portfolio_value"] == pytest.approx(1_000.0 + 10.0 * side, abs=1e-6)

    _, _, _, _, info = env.step(np.array([w]))                 # bar 5: reactivated at 120
    assert env.positions[0] == pytest.approx(w, abs=1e-12)
    assert info["portfolio_value"] == pytest.approx(1_000.0 + 20.0 * side, abs=1e-6)


def test_the_halt_bar_books_no_return():
    """The step return across a halt bar is zero, and the reactivation bar books the 110 ->
    120 move, not a -100% / +100% round trip through 0."""
    env = _single_asset_env(HALT_PATH)
    env.reset()
    env.step(np.array([0.1]))
    env.step(np.array([0.1]))
    returns = [env.step(np.array([0.1]))[4]["step_return"] for _ in range(3)]  # bars 3, 4, 5
    assert returns[0] == pytest.approx(0.0, abs=1e-12)
    assert returns[1] == pytest.approx(0.0, abs=1e-12)
    assert returns[2] == pytest.approx(10.0 / 1_010.0, rel=1e-9)


def test_a_flat_asset_cannot_be_opened_into_a_halt():
    env = _single_asset_env(HALT_PATH)
    env.reset()
    env.step(np.array([0.0]))
    env.step(np.array([0.0]))
    env.step(np.array([0.5]))                                  # bar 3: halted
    assert env.positions[0] == 0.0
    assert env.entry_prices[0] == 0.0
    assert env.entry_notionals[0] == 0.0


def test_a_carried_holding_settles_no_funding_during_the_halt():
    """Bars 7-9 are halted and bar 8 is the 08:00 UTC settlement. A carried holding is
    marked, not traded, so it settles no funding there. Bars 1-6 are 01:00-06:00, so no
    other bar settles and cumulative funding must stay exactly zero through bar 9."""
    prices = [100.0] * 7 + [0.0, 0.0, 0.0] + [100.0]
    env = _single_asset_env(prices, funding_rate=1e-3)
    env.reset()
    for _ in range(9):                                         # bars 1-9, halt at 7-9
        _, _, _, _, info = env.step(np.array([0.1]))
        assert info["cumulative_funding"] == 0.0
    assert env._funding_mask[8] and not env._asset_available[8, 0]  # the case is live
    assert env.positions[0] == pytest.approx(0.1, abs=1e-12)


def test_a_nan_price_is_a_halt_too():
    """C2 treats any price that is not > 1e-10 as unavailable, NaN included."""
    env = _single_asset_env([100.0, 100.0, 110.0, float("nan"), 120.0])
    env.reset()
    env.step(np.array([0.1]))
    env.step(np.array([0.1]))
    _, _, _, _, info = env.step(np.array([0.1]))               # bar 3: NaN
    assert env.positions[0] == pytest.approx(0.1, abs=1e-12)
    assert info["portfolio_value"] == pytest.approx(1_010.0, abs=1e-6)
    assert np.isfinite(info["portfolio_value"])
