"""Action -> weight tests for MultiAssetAllocatorEnv.

Proves the env's vol-scaling (ADR-2) is byte-faithful to the validated linear core:
feeding ``trend_conviction`` reproduces ``baseline_weight`` from the signals module.
Plus the leverage-cap, gross-cap, short, long-only, and invalid-vol paths.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sharpen.envs.multi_asset_allocator_env import MultiAssetAllocatorEnv
from sharpen.features import cross_asset_signals as cas

from .conftest import build_arrays, synthetic_prices


def _signals_wide(T: int = 420, n: int = 5, seed: int = 11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=T)
    rets = rng.normal(0.0004, 0.011, size=(T, n))
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rets, axis=0)), index=idx,
        columns=[f"A{i}" for i in range(n)],
    )
    long = cas.compute(close, asset_class={c: "all" for c in close.columns})

    def wide(col: str) -> pd.DataFrame:
        return long.pivot(index="date", columns="ticker", values=col).reindex(columns=close.columns)

    return close, wide("vol"), wide("trend_conviction"), wide("baseline_weight")


def _env(price: np.ndarray, vol: np.ndarray, **kw) -> MultiAssetAllocatorEnv:
    arrays = build_arrays(price, vol=vol)
    params = dict(
        target_vol_asset=cas.DEFAULT_TARGET_VOL_ASSET,
        lev_cap=cas.DEFAULT_LEV_CAP,
        max_gross_exposure=1e9,         # non-binding by default (per-asset comparison)
        min_trade_pct=0.0,
        slippage_base_bps=0.0,
        slippage_impact_bps=0.0,
    )
    params.update(kw)
    return MultiAssetAllocatorEnv(**arrays, **params)


# --------------------------------------------------------------------------- #
# 1. trend_conviction -> baseline_weight (the keystone identity)
# --------------------------------------------------------------------------- #

def test_conviction_reproduces_baseline_weight():
    """env._action_to_weights(trend_conviction) == signals.baseline_weight per asset,
    once the (non-binding) gross cap is removed."""
    close, vol, conv, bw = _signals_wide()
    env = _env(close.values, vol.values)

    warmup = max(cas.DEFAULT_LOOKBACKS) + cas.DEFAULT_SKIP + cas.DEFAULT_VOL_WINDOW
    checked = 0
    for step in range(warmup, len(close) - 1):
        env.step_idx = step
        conviction = np.nan_to_num(conv.iloc[step].to_numpy(), nan=0.0)
        w = env._action_to_weights(conviction)
        expected = np.nan_to_num(bw.iloc[step].to_numpy(), nan=0.0)
        np.testing.assert_allclose(w, expected, atol=1e-9,
                                   err_msg=f"weight mismatch at step {step}")
        checked += 1
    assert checked > 30, "too few non-warmup rows exercised"


# --------------------------------------------------------------------------- #
# 2. vol-scaling math on controlled vol
# --------------------------------------------------------------------------- #

def test_vol_scaling_formula_and_lev_cap():
    """w = clip(conviction * min(target/vol, lev_cap), -lev_cap, lev_cap)."""
    price = synthetic_prices(T=50, n=3, seed=1)
    T, n = price.shape
    vol = np.full((T, n), np.nan)
    vol[10:] = np.array([0.05, 0.10, 0.40])          # low / on-target / high vol
    env = _env(price, vol, target_vol_asset=0.10, lev_cap=2.0)
    env.step_idx = 20

    w = env._action_to_weights(np.array([1.0, 1.0, 1.0]))
    # scale = min(0.10/vol, 2.0) = [2.0 (capped), 1.0, 0.25]
    np.testing.assert_allclose(w, [2.0, 1.0, 0.25], atol=1e-12)

    # Conviction scales the magnitude linearly; sign preserved.
    w2 = env._action_to_weights(np.array([0.5, -1.0, -0.5]))
    np.testing.assert_allclose(w2, [1.0, -1.0, -0.125], atol=1e-12)


def test_gross_exposure_cap_binds_proportionally():
    price = synthetic_prices(T=40, n=5, seed=2)
    T, n = price.shape
    vol = np.full((T, n), np.nan)
    vol[10:] = 0.10                                   # scale = 1.0 each → raw gross = 5
    env = _env(price, vol, max_gross_exposure=2.0)
    env.step_idx = 20

    w = env._action_to_weights(np.ones(n))
    assert abs(np.abs(w).sum() - 2.0) < 1e-9         # capped to gross 2.0
    np.testing.assert_allclose(w, np.full(n, 0.4), atol=1e-12)  # proportional


def test_short_weights_allowed_by_default():
    price = synthetic_prices(T=40, n=2, seed=3)
    T, n = price.shape
    vol = np.full((T, n), np.nan)
    vol[10:] = 0.10
    env = _env(price, vol)
    env.step_idx = 20
    w = env._action_to_weights(np.array([-1.0, -0.5]))
    assert w[0] < 0 and w[1] < 0


def test_invalid_vol_forces_flat_weight():
    price = synthetic_prices(T=40, n=3, seed=4)
    T, n = price.shape
    vol = np.full((T, n), 0.10)
    vol[20, 1] = np.nan                              # NaN vol on asset 1
    vol[20, 2] = 0.0                                 # zero vol on asset 2
    env = _env(price, vol)
    env.step_idx = 20
    w = env._action_to_weights(np.ones(n))
    assert w[1] == 0.0 and w[2] == 0.0 and w[0] > 0.0


# --------------------------------------------------------------------------- #
# 3. long-only action space
# --------------------------------------------------------------------------- #

def test_long_only_clips_negative_conviction():
    price = synthetic_prices(T=40, n=3, seed=5)
    T, n = price.shape
    vol = np.full((T, n), 0.10)
    env = _env(price, vol, allow_short=False)
    assert env.action_space.low.min() == 0.0
    env.reset()
    # Negative conviction must produce non-negative positions after the step.
    env.step(np.array([-1.0, 0.5, -0.3]))
    assert (env.positions >= -1e-12).all()
    assert env.positions[1] > 0.0
