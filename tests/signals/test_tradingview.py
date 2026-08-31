"""Tests for the TradingView technical-indicator signals: causality + compute + a formula."""
from __future__ import annotations

import numpy as np

from sharpen.signals import make_synthetic_panel
from sharpen.signals.eval_harness import assert_causal
from sharpen.signals.library.tradingview import SIGNALS, _roc


def test_tv_signals_causal_and_compute() -> None:
    p = make_synthetic_panel(T=300, N=30, seed=8)
    for sig in SIGNALS:
        ok, msg = assert_causal(sig, p)
        assert ok, f"{sig.spec.name}: {msg}"
        out = sig.compute(p)
        assert out.shape == (p.T, p.N)
        assert np.all(np.isfinite(out) | np.isnan(out))
        assert np.isfinite(out[100:]).mean() > 0.3, sig.spec.name


def test_tv_count_and_family() -> None:
    names = [s.spec.name for s in SIGNALS]
    assert len(names) == len(set(names)) == 12
    assert all(s.spec.family == "technical" for s in SIGNALS)
    assert all(s.spec.expected_sign in (1, -1) for s in SIGNALS)


def test_roc_formula() -> None:
    p = make_synthetic_panel(T=40, N=4, seed=9)
    expected = p.close / np.roll(p.close, 10, axis=0) - 1.0
    expected[:10] = np.nan
    assert np.allclose(_roc(p, 10), expected, equal_nan=True)
