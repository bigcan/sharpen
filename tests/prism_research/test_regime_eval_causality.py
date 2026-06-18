"""Causality tripwire for the regime-eval forward-target / trailing-signal builders.

The Path-A provider's regime columns are guarded by test_pathA_walk_forward.py. This
file guards the NEW helpers that prism_regime_eval.py adds on top of them:

  * forward realized-vol / forward return are STRICTLY FUTURE — the value at bar t
    must be invariant to any change in close_{<=t} and must respond only to close_{>t}.
  * trailing realized vol / momentum are CAUSAL — the value at bar t must be invariant
    to any change in close_{>t}.

Each direction has a positive assertion AND a negative control (an injected change in
the forbidden region MUST be ignored; an injected change in the allowed region MUST be
seen) so the test fails if look-ahead is ever reintroduced (LEAK-2 / CAUS-01..05).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prism_research.prism_regime_eval import (  # noqa: E402
    fwd_realized_vol, fwd_log_return, log_returns, trailing_realized_vol, momentum,
)

rng = np.random.default_rng(7)
CLOSE = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, size=300))
T = 150           # the bar under test
W = 25            # window for trailing-vol / momentum


def _perturb(close: np.ndarray, idx: int, factor: float = 1.05) -> np.ndarray:
    c = close.copy()
    c[idx] *= factor
    return c


# --------------------------------------------------------------------------- #
# Forward targets must use ONLY data > t
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [1, 5])
def test_forward_realized_vol_ignores_past(k):
    base = fwd_realized_vol(log_returns(CLOSE), k)
    # perturb a PAST bar (< t): target at t must NOT change.
    pert = fwd_realized_vol(log_returns(_perturb(CLOSE, T - 10)), k)
    assert np.isfinite(base[T])
    assert base[T] == pytest.approx(pert[T], abs=1e-12), "forward RV peeked at the past (leak)"


@pytest.mark.parametrize("k", [1, 5])
def test_forward_realized_vol_responds_to_future(k):
    base = fwd_realized_vol(log_returns(CLOSE), k)
    # perturb a FUTURE bar within the horizon (t+1..t+k): target at t MUST change.
    pert = fwd_realized_vol(log_returns(_perturb(CLOSE, T + 1)), k)
    assert base[T] != pytest.approx(pert[T], abs=1e-9), "forward RV blind to its own horizon"


def test_forward_return_is_strictly_t_plus_1():
    base = fwd_log_return(CLOSE)
    past = fwd_log_return(_perturb(CLOSE, T - 5))
    future = fwd_log_return(_perturb(CLOSE, T + 1))
    assert base[T] == pytest.approx(past[T], abs=1e-12), "fwd return peeked at the past"
    assert base[T] != pytest.approx(future[T], abs=1e-9), "fwd return ignored t+1"


# --------------------------------------------------------------------------- #
# Trailing signals must use ONLY data <= t
# --------------------------------------------------------------------------- #
def test_trailing_realized_vol_ignores_future():
    r = log_returns(CLOSE)
    base = trailing_realized_vol(r, W)
    pert = trailing_realized_vol(log_returns(_perturb(CLOSE, T + 5)), W)
    assert np.isfinite(base[T])
    assert base[T] == pytest.approx(pert[T], abs=1e-12), "trailing RV used a future bar (leak)"


def test_trailing_realized_vol_responds_to_recent_past():
    r = log_returns(CLOSE)
    base = trailing_realized_vol(r, W)
    pert = trailing_realized_vol(log_returns(_perturb(CLOSE, T - 1)), W)
    assert base[T] != pytest.approx(pert[T], abs=1e-9), "trailing RV blind to its own window"


def test_momentum_ignores_future():
    base = momentum(CLOSE, 20)
    pert = momentum(_perturb(CLOSE, T + 3), 20)
    assert np.isfinite(base[T])
    assert base[T] == pytest.approx(pert[T], abs=1e-12), "momentum used a future bar (leak)"
