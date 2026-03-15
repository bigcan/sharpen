"""Tests for CryptoPerpEnv long_only mode.

Covers:
1. Default action space is Box(-1, 1)
2. Long-only action space is Box(0, 1)
3. Negative actions are clipped to non-negative positions
4. Gross exposure constraint still enforced in long-only mode
"""

from __future__ import annotations

import numpy as np

from finrl_pro_ds.crypto.envs.crypto_perp_env import CryptoPerpEnv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_timestamps(n_bars: int) -> np.ndarray:
    """Create UTC epoch-second timestamps starting from 2024-01-01 00:00."""
    base = 1704067200  # 2024-01-01 00:00 UTC
    return np.arange(base, base + n_bars * 3600, 3600, dtype=np.int64)


def _make_env(n_bars: int = 200, n_assets: int = 3, long_only: bool = False) -> CryptoPerpEnv:
    """Create a minimal CryptoPerpEnv for testing."""
    timestamps = _make_timestamps(n_bars)

    # Constant prices at 100
    price_ary = np.full((n_bars, n_assets), 100.0, dtype=np.float64)
    # Slight uptrend so long positions are profitable
    for t in range(n_bars):
        price_ary[t, :] = 100.0 + 0.01 * t

    # Zero funding rates
    funding_rate_ary = np.zeros((n_bars, n_assets), dtype=np.float64)

    # Volumes
    volume_ary = np.full((n_bars, n_assets), 1e6, dtype=np.float64)

    # 6 features per asset
    tech_ary = np.random.randn(n_bars, n_assets * 6).astype(np.float32) * 0.1

    return CryptoPerpEnv(
        price_ary=price_ary,
        tech_ary=tech_ary,
        funding_rate_ary=funding_rate_ary,
        volume_ary=volume_ary,
        timestamps=timestamps,
        initial_capital=100_000.0,
        long_only=long_only,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_action_space_default():
    """Default env has action space Box(-1, 1)."""
    env = _make_env(long_only=False)
    assert env.action_space.low[0] == -1.0
    assert env.action_space.high[0] == 1.0


def test_action_space_long_only():
    """Long-only env has action space Box(0, 1)."""
    env = _make_env(long_only=True)
    assert env.action_space.low[0] == 0.0
    assert env.action_space.high[0] == 1.0


def test_long_only_clips_negative_actions():
    """Negative actions produce non-negative positions in long-only mode."""
    env = _make_env(long_only=True)
    obs, _ = env.reset()

    # Send strongly negative actions
    action = np.array([-0.5, -0.3, -0.8])
    obs, reward, terminated, truncated, info = env.step(action)

    # All positions should be non-negative (clipped before EMA)
    assert np.all(env.positions >= -1e-8), (
        f"Long-only env has negative positions: {env.positions}"
    )


def test_long_only_gross_exposure():
    """Gross exposure constraint is still enforced in long-only mode."""
    env = _make_env(long_only=True)
    obs, _ = env.reset()

    # Send actions that sum > 1.0 (should be scaled down)
    action = np.array([0.6, 0.6, 0.6])
    obs, reward, terminated, truncated, info = env.step(action)

    gross = info["gross_exposure"]
    assert gross <= env.max_gross_exposure + 1e-6, (
        f"Gross exposure {gross} exceeds max {env.max_gross_exposure}"
    )


def test_long_only_max_net_short_exposure_zero():
    """Long-only mode forces max_net_short_exposure to 0.0."""
    env = _make_env(long_only=True)
    assert env.max_net_short_exposure == 0.0


def test_long_only_multi_step():
    """Long-only env survives multiple steps with mixed actions."""
    env = _make_env(n_bars=50, long_only=True)
    obs, _ = env.reset()

    rng = np.random.default_rng(42)
    for _ in range(20):
        # Mix of positive and negative actions
        action = rng.uniform(-1.0, 1.0, size=3)
        obs, reward, terminated, truncated, info = env.step(action)
        # Positions must always be non-negative
        assert np.all(env.positions >= -1e-8), (
            f"Long-only env has negative positions: {env.positions}"
        )
        if terminated or truncated:
            break
