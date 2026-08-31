"""Reward tests for MultiAssetAllocatorEnv: DSR + turnover penalty.

The DSR path must match the shared Moody-Saffell ``DSRCalculator`` fed the same step
returns; the turnover penalty must subtract ``turnover_penalty * sum|Δw|``.
"""
from __future__ import annotations

import numpy as np

from sharpen.envs.dsr import DSRCalculator
from sharpen.envs.multi_asset_allocator_env import MultiAssetAllocatorEnv

from .conftest import build_arrays, causal_vol, synthetic_prices


def _env(price, vol=None, **kw):
    arrays = build_arrays(price, vol=vol)
    params = dict(
        target_vol_asset=0.10, lev_cap=2.0, max_gross_exposure=10.0,
        taker_fee_pct=0.0, slippage_base_bps=0.0, slippage_impact_bps=0.0,
        min_trade_pct=0.0,
    )
    params.update(kw)
    return MultiAssetAllocatorEnv(**arrays, **params)


def test_simple_reward_applies_turnover_penalty():
    """reward = step_return - turnover_penalty * sum|Δw| (simple mode)."""
    price = np.array([100.0, 100.0, 120.0], dtype=np.float64).reshape(-1, 1)
    vol = np.full((3, 1), 0.10)
    env = _env(price, vol, reward_type="simple", turnover_penalty=0.1)
    env.reset()
    # Entry step: step_return = 0 (enter at price[1], 0 unrealized), turnover = 1.0.
    _, r0, _, _, _ = env.step(np.array([1.0]))
    assert abs(r0 - (0.0 - 1.0 * 0.1)) < 1e-9
    # Hold step: turnover = 0, step_return = +0.2.
    _, r1, _, _, info = env.step(np.array([1.0]))
    assert abs(info["turnover"]) < 1e-12
    assert abs(r1 - 0.2) < 1e-9


def test_turnover_penalty_lowers_reward():
    price = np.array([100.0, 100.0, 100.0], dtype=np.float64).reshape(-1, 1)
    vol = np.full((3, 1), 0.10)
    cheap = _env(price, vol, reward_type="simple", turnover_penalty=0.0)
    pricey = _env(price, vol, reward_type="simple", turnover_penalty=0.5)
    cheap.reset()
    pricey.reset()
    _, r_cheap, _, _, _ = cheap.step(np.array([1.0]))
    _, r_pricey, _, _, _ = pricey.step(np.array([1.0]))
    assert r_pricey < r_cheap


def test_dsr_reward_matches_reference_calculator():
    """With turnover_penalty=0 and a wide clip, the env's DSR reward equals a fresh
    DSRCalculator fed the env's own step returns in order."""
    price = synthetic_prices(T=120, n=3, seed=9)
    vol = causal_vol(price)
    env = _env(price, vol, reward_type="dsr", dsr_eta=0.01,
               turnover_penalty=0.0, reward_clip_range=(-100.0, 100.0))
    env.reset()
    ref = DSRCalculator(eta=0.01, scale=1.0)

    rng = np.random.default_rng(3)
    for _ in range(80):
        a = rng.uniform(-1.0, 1.0, size=3)
        _, reward, term, trunc, info = env.step(a)
        expected = ref.compute(info["step_return"])
        assert abs(reward - expected) < 1e-9
        assert np.isfinite(reward)
        if term or trunc:
            break


def test_dsr_warmup_is_zero_then_finite():
    price = synthetic_prices(T=60, n=2, seed=2)
    vol = causal_vol(price)
    env = _env(price, vol, reward_type="dsr", turnover_penalty=0.0)
    env.reset()
    _, r0, _, _, _ = env.step(np.array([0.5, -0.5]))
    assert r0 == 0.0                                   # DSR warmup (first observation)
    for _ in range(10):
        _, r, _, _, _ = env.step(np.array([0.5, -0.5]))
        assert np.isfinite(r)
