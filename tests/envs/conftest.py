"""Shared fixtures/builders for MultiAssetAllocatorEnv tests.

Deterministic synthetic data only (no Date.now / no network) so the suite is
reproducible. The keystone baseline-parity test (tests/integration/) uses the cached
real ETF prices instead.
"""
from __future__ import annotations

import numpy as np

ANN = 252


def synthetic_prices(T: int = 400, n: int = 4, seed: int = 7) -> np.ndarray:
    """Deterministic geometric-random-walk close prices, shape (T, n)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.012, size=(T, n))
    return 100.0 * np.exp(np.cumsum(rets, axis=0))


def causal_vol(price: np.ndarray, vol_window: int = 63) -> np.ndarray:
    """Annualized realized vol, strictly causal (rolling std then shift 1).

    Row t uses returns <= t-1 — the same convention as
    ``cross_asset_signals._realized_vol`` and the falsification.
    """
    T, n = price.shape
    rets = np.full((T, n), np.nan)
    rets[1:] = price[1:] / price[:-1] - 1.0
    vol = np.full((T, n), np.nan)
    for t in range(T):
        if t - 1 >= vol_window:  # need vol_window returns ending at t-1
            win = rets[t - vol_window:t]  # returns at indices [t-vol_window, t-1]
            vol[t] = win.std(axis=0, ddof=1) * np.sqrt(ANN)
    return vol


def build_arrays(
    price: np.ndarray,
    *,
    vol: np.ndarray | None = None,
    tech_dim: int = 1,
    carry: float = 0.0,
    volume: float = 1e12,
) -> dict:
    """Assemble the (T, *) arrays MultiAssetAllocatorEnv expects.

    volume defaults huge so the volume-impact slippage term ~ 0 (isolates fee).
    """
    T, n = price.shape
    if vol is None:
        vol = causal_vol(price)
    tech = np.zeros((T, n * tech_dim), dtype=np.float32)
    return {
        "price_ary": price,
        "tech_ary": tech,
        "vol_ary": vol,
        "carry_ary": np.full((T, n), carry, dtype=np.float64),
        "volume_ary": np.full((T, n), volume, dtype=np.float64),
        "timestamps": (np.arange(T, dtype=np.int64) * 86400),  # daily epoch seconds
    }
