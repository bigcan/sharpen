"""Shared fixtures for the paper-executor (rung-1) tests.

Deterministic synthetic arrays only (no network / no Date.now) so the suite is
reproducible. The keystone parity test additionally has a real-ETF variant gated on
the cached ``results/xsec_momentum`` data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
ANN = 252


@pytest.fixture(scope="session")
def cfg() -> dict:
    """The cross-asset allocator config (env block drives the fill model + caps)."""
    return yaml.safe_load((ROOT / "configs" / "cross_asset_momentum.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def gates_cfg() -> dict:
    """The pre-registered decision/paper_soak gates overlay."""
    return yaml.safe_load(
        (ROOT / "configs" / "cross_asset_momentum.gates.yaml").read_text(encoding="utf-8"))


def causal_vol(price: np.ndarray, vol_window: int = 63) -> np.ndarray:
    """Annualized realized vol, strictly causal (row t uses returns <= t-1)."""
    T, n = price.shape
    rets = np.full((T, n), np.nan)
    rets[1:] = price[1:] / price[:-1] - 1.0
    vol = np.zeros((T, n))
    for t in range(T):
        if t - 1 >= vol_window:
            vol[t] = rets[t - vol_window:t].std(axis=0, ddof=1) * np.sqrt(ANN)
    return vol


def allocator_arrays(
    *, T: int = 780, n: int = 18, seed: int = 7, volume: float = 5e8,
    assets: list[str] | None = None,
) -> dict:
    """Deterministic (T, *) arrays for the allocator env / paper replay, incl.
    ``conviction_ary`` (the frozen-core signal) and ``assets``. ``volume`` is DOLLAR
    volume (F1). Default 18 assets mirrors the validated universe size."""
    rng = np.random.default_rng(seed)
    ts = (pd.bdate_range("2018-01-01", periods=T).asi8 // 10**9).astype(np.int64)
    price = 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.012, (T, n)), axis=0)
    vol = causal_vol(price)
    return {
        "price_ary": price,
        "tech_ary": rng.normal(0, 1, (T, n * 7)).astype(np.float32),
        "vol_ary": vol,
        "carry_ary": np.zeros((T, n), dtype=np.float64),
        "volume_ary": np.full((T, n), float(volume), dtype=np.float64),
        "timestamps": ts,
        "conviction_ary": np.sign(rng.normal(0, 1, (T, n))),
        "assets": assets or [f"A{i}" for i in range(n)],
    }


@pytest.fixture
def arrays_18() -> dict:
    """18-asset synthetic arrays labeled with the real universe tickers (so SPY /
    asset-class attribution paths are exercised)."""
    universe = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD",
                "GLD", "SLV", "DBC", "USO", "DBA", "UUP", "FXE", "FXY", "FXB", "FXA"]
    return allocator_arrays(T=780, n=18, seed=11, assets=universe)
