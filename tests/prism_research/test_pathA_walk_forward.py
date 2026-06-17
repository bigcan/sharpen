"""Runtime walk-forward tripwire for the Path-A in-process PRISM provider.

This is the belt-and-suspenders check the operator flagged as the open residual in
S553-cont-48: the provider was verified walk-forward safe *by code read*, but never
runtime-tested. The definitive test: a feature at bar ``t`` must be identical whether
the provider is handed the FULL frame or a frame TRUNCATED at ``t``. If any future bar
(``> t``) changes a past feature, that is look-ahead (LEAK-2 / the Path-B GAHMM defect).

GAHMM fits with a fixed ``random_state`` (42) on ``iloc[:bar+1]``, so identical windows
give identical params -> the comparison is exact (not flaky). Chronos is deterministic
zero-shot on identical context. We unload between runs to force an independent re-fit.

Skips cleanly if the SAFFS provider is not installed on this machine.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# GAHMM-only feature columns — the look-ahead-prone half (the Path-B leak was GAHMM).
_GAHMM_COLS = [
    "gahmm_price_bear", "gahmm_price_neutral", "gahmm_price_bull",
    "gahmm_vol_low", "gahmm_vol_normal", "gahmm_vol_high", "gahmm_composite_code",
]
_CHRONOS_COLS = [
    "chronos_p10", "chronos_p30", "chronos_p50", "chronos_p70", "chronos_p90",
    "chronos_spread",
]


def _resolve_saffs() -> str | None:
    """Return a SAFFS root with the in-process provider, or None to skip."""
    cands = []
    if os.environ.get("PRISM_ROOT"):
        cands.append(Path(os.environ["PRISM_ROOT"]))
    cands.append(Path(r"C:\FinRL\SAFFS"))
    cands.append(Path.home() / "FinRL" / "SAFFS")
    for c in cands:
        if (c / "exports" / "finrl_feature_provider.py").is_file():
            return str(c)
    return None


def _synth_daily(n: int = 160, seed: int = 7) -> pd.DataFrame:
    """Deterministic synthetic daily OHLCV with a regime shift (trend -> choppy)."""
    rng = np.random.default_rng(seed)
    # First half: gentle uptrend, low vol. Second half: flat, high vol.
    drift = np.concatenate([np.full(n // 2, 0.0008), np.full(n - n // 2, -0.0002)])
    vol = np.concatenate([np.full(n // 2, 0.006), np.full(n - n // 2, 0.020)])
    rets = drift + vol * rng.standard_normal(n)
    close = 100.0 * np.exp(np.cumsum(rets))
    high = close * (1.0 + np.abs(rng.standard_normal(n)) * 0.003)
    low = close * (1.0 - np.abs(rng.standard_normal(n)) * 0.003)
    open_ = np.concatenate([[close[0]], close[:-1]])
    ts = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "timestamp": ts, "ticker": "TEST",
        "open": open_, "high": np.maximum.reduce([open_, high, close]),
        "low": np.minimum.reduce([open_, low, close]), "close": close,
        "volume": rng.uniform(1e3, 1e4, n),
    })


@pytest.mark.slow
def test_pathA_features_are_walk_forward_safe():
    saffs = _resolve_saffs()
    if saffs is None:
        pytest.skip("SAFFS in-process provider not installed on this machine")
    os.environ["PRISM_ROOT"] = saffs
    # In-process provider needs no DB; avoid SAFFS settings' DB-password guard at import.
    os.environ.setdefault("PRISM_USE_DB", "false")

    from finrl_pro_ds.crypto.features.prism_features import (
        compute_prism_features,
        unload_prism,
    )

    df = _synth_daily(n=160)
    cfg = {"chronos_device": "cpu", "gahmm_refit_every": 720, "gahmm_min_bars": 60}
    cut = 120  # bars 0..120 overlap between full and truncated runs

    def _compute(frame):
        return compute_prism_features(
            frame, ["TEST"], frame["timestamp"].iloc[0], frame["timestamp"].iloc[-1],
            config=cfg,
        ).set_index("timestamp").sort_index()

    # Full frame. The provider lazily imports GAHMM (hmmlearn) / Chronos at compute
    # time; if those deps live only in the SAFFS venv (not the active env), skip rather
    # than error — this test is meaningful only where the provider can actually run.
    unload_prism()
    try:
        full = _compute(df)
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"provider runtime deps unavailable in this interpreter: {e}")

    # Truncated frame: future bars (> cut) removed entirely.
    unload_prism()
    df_trunc = df.iloc[: cut + 1].copy()
    trunc = _compute(df_trunc)
    unload_prism()

    # Compare on overlapping bars only — must be byte-identical (no future leak).
    common = trunc.index.intersection(full.index)
    assert len(common) >= 50, f"too few overlapping bars to test ({len(common)})"

    # GAHMM is the look-ahead-prone half — assert exactly.
    for col in _GAHMM_COLS:
        a = full.loc[common, col].to_numpy()
        b = trunc.loc[common, col].to_numpy()
        assert np.allclose(a, b, atol=1e-9, equal_nan=True), (
            f"WALK-FORWARD LEAK: GAHMM '{col}' changed when future bars were appended "
            f"(max abs diff {np.nanmax(np.abs(a - b)):.3e})"
        )

    # Chronos is deterministic zero-shot; if live (non-zero), it must also match.
    chronos_live = bool(np.any(np.abs(full[_CHRONOS_COLS].to_numpy()) > 1e-9))
    if chronos_live:
        for col in _CHRONOS_COLS:
            a = full.loc[common, col].to_numpy()
            b = trunc.loc[common, col].to_numpy()
            assert np.allclose(a, b, atol=1e-6, equal_nan=True), (
                f"WALK-FORWARD LEAK: Chronos '{col}' changed with future bars "
                f"(max abs diff {np.nanmax(np.abs(a - b)):.3e})"
            )


@pytest.mark.slow
def test_pathA_negative_control_detects_injected_leak():
    """Sanity: a deliberately leaky transform (using a future bar) FAILS the same check.

    Guards against a vacuous green — proves the comparison above can actually catch a leak.
    """
    df = _synth_daily(n=160)
    cut = 120
    close = df["close"].to_numpy()

    # Causal feature: trailing mean (uses <= t). Leaky feature: centered mean (uses t+1).
    causal = pd.Series(close).rolling(5).mean().to_numpy()
    leaky = pd.Series(close).rolling(5, center=True).mean().to_numpy()

    # On the truncated frame the same indices are recomputed; the leaky (centered) feature
    # near the truncation edge differs because it needed future bars.
    causal_full, causal_trunc = causal[: cut + 1], pd.Series(close[: cut + 1]).rolling(5).mean().to_numpy()
    leaky_full, leaky_trunc = leaky[: cut + 1], pd.Series(close[: cut + 1]).rolling(5, center=True).mean().to_numpy()

    # Causal matches on the overlap; leaky diverges at the edge (would-be future peek).
    assert np.allclose(causal_full[60:cut - 2], causal_trunc[60:cut - 2], equal_nan=True)
    assert not np.allclose(leaky_full[cut - 3:cut + 1], leaky_trunc[cut - 3:cut + 1], equal_nan=True)
