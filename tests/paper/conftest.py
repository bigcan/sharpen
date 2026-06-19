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


# --------------------------------------------------------------------------- #
# Two-sleeve (momentum + rates-carry) fixtures
# --------------------------------------------------------------------------- #
_MOM_NAMES = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD",
              "GLD", "SLV", "DBC", "USO", "DBA", "UUP", "FXE", "FXY", "FXB", "FXA"]
_RATES_NAMES = ["SHY", "IEF", "TLT", "LQD"]
_UNION_NAMES = _MOM_NAMES + ["SHY"]


def two_sleeve_arrays(*, T: int = 780, seed: int = 13, volume: float = 5e8) -> dict:
    """Deterministic momentum(18) + rates-carry(4) + union(19) bundle sharing ONE price
    matrix — so the shared bond ETFs (IEF/TLT/LQD) are byte-identical across sleeves and
    the union (exactly the single-fetch alignment ``load_two_sleeve_data`` guarantees).
    The rates sleeve's conviction is in [-1,1] (tanh-like), the momentum sleeve's is the
    ±1 sign signal — neither is read by the linear drive's obs, only by the action."""
    rng = np.random.default_rng(seed)
    ts = (pd.bdate_range("2014-01-01", periods=T).asi8 // 10**9).astype(np.int64)
    U = len(_UNION_NAMES)
    price = 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.010, (T, U)), axis=0)
    vol = causal_vol(price)
    dvol = np.full((T, U), float(volume), dtype=np.float64)
    uidx = {a: i for i, a in enumerate(_UNION_NAMES)}

    def cols(names: list[str]) -> list[int]:
        return [uidx[a] for a in names]

    mcols, rcols = cols(_MOM_NAMES), cols(_RATES_NAMES)
    conv_rates = np.tanh(rng.normal(0, 1, (T, len(_RATES_NAMES))))
    momentum = {
        "price_ary": price[:, mcols].copy(),
        "tech_ary": rng.normal(0, 1, (T, len(_MOM_NAMES) * 7)).astype(np.float32),
        "vol_ary": vol[:, mcols].copy(),
        "carry_ary": np.zeros((T, len(_MOM_NAMES)), dtype=np.float64),
        "volume_ary": dvol[:, mcols].copy(),
        "timestamps": ts,
        "conviction_ary": np.sign(rng.normal(0, 1, (T, len(_MOM_NAMES)))),
        "assets": list(_MOM_NAMES),
    }
    rates_carry = {
        "price_ary": price[:, rcols].copy(),
        "tech_ary": conv_rates.astype(np.float32),     # (T,4) tech_dim=1 placeholder
        "vol_ary": vol[:, rcols].copy(),
        "carry_ary": np.zeros((T, len(_RATES_NAMES)), dtype=np.float64),
        "volume_ary": dvol[:, rcols].copy(),
        "timestamps": ts,
        "conviction_ary": conv_rates,
        "assets": list(_RATES_NAMES),
    }
    union = {
        "price_ary": price.copy(),
        "volume_ary": dvol.copy(),
        "carry_ary": np.zeros((T, U), dtype=np.float64),
        "timestamps": ts,
        "assets": list(_UNION_NAMES),
    }
    return {"momentum": momentum, "rates_carry": rates_carry, "union": union}


@pytest.fixture
def bundle() -> dict:
    """Synthetic two-sleeve array bundle (momentum/rates_carry/union)."""
    return two_sleeve_arrays()


@pytest.fixture(scope="session")
def paper2_cfg() -> dict:
    """The 2-sleeve paper config (19-asset union, sleeves block, risk_parity params)."""
    return yaml.safe_load(
        (ROOT / "configs" / "live_cross_asset_paper.yaml").read_text(encoding="utf-8"))
