"""crucible-v17.0 — two conventions Tier-2 capturability fixed silently, now measured.

* REBALANCE PHASE. The traded book rebalanced on rows 0, h, 2h, ... only. On Taiwan small-cap the
  ``holder_conc`` book scored frictionless Sharpe -0.045 at phase 0 and positive on 16 of 21 phases
  (deep audit 2026-09-30, "still open"). The sweep is now on the card, and PROMISING also needs the
  median phase to clear the frictionless floor (``capturability.require_phase_robust``).
* EXECUTION LAG. The book enters at the close the signal reads. A one-bar-late book is reported, and
  a large loss of Sharpe is caveated — never gated: execution, like cost, is venue-specific.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from sharpen.signals.eval_harness import _ann_sharpe, _ls_weights, tier2_capturability
from sharpen.signals.features import Panel
from tests.signals.test_capturability_gate_f3 import _card, _gates, _verdict

T, N, H = 900, 12, 5


def _panel(seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    close = np.exp(np.cumsum(0.01 * rng.standard_normal((T, N)), axis=0) + 4.0)
    dates = (np.datetime64("2014-01-02") + np.arange(T)).astype("datetime64[ns]")
    return Panel(dates, tuple(f"E{i:02d}" for i in range(N)), close, close * 1.001, close * 0.999,
                 close, np.full((T, N), 1e6), np.ones((T, N), bool), close * 1e6,
                 np.zeros(N, dtype=int), {"survivorship_free": True, "source": "synthetic"})


def _cap(panel: Panel, scores: np.ndarray, h: int = H):
    return tier2_capturability(None, panel, _gates(), neutralization=(), expected_sign=1,
                               hold_horizon=h, scores=scores)


def _ref_book(panel: Panel, scores: np.ndarray, offset: int, lag: int) -> float:
    """Independent frictionless Sharpe: weights from row t, return close[t+lag] -> close[t+lag+H]."""
    out = []
    for t in range(offset, T - H - lag, H):
        w = _ls_weights(scores[t], panel.active[t])
        r = panel.close[t + lag + H] / panel.close[t + lag] - 1.0
        out.append(float(np.nansum(w * r)))
    return _ann_sharpe(np.asarray(out), 252 / H)


def test_phase_sweep_and_lag_book_match_an_independent_construction() -> None:
    panel = _panel()
    scores = np.random.default_rng(1).standard_normal((T, N))
    cap = _cap(panel, scores)
    assert len(cap.phase_frictionless) == H == len(cap.phase_net_standard)
    assert cap.phase_frictionless[0] == cap.frictionless_sharpe      # phase 0 IS the headline book
    assert cap.phase_net_standard[0] == cap.by_cost["standard"].net_sharpe
    for o in range(H):
        assert np.isclose(cap.phase_frictionless[o], _ref_book(panel, scores, o, 0))
    assert np.isclose(cap.lag1_frictionless_sharpe, _ref_book(panel, scores, 0, 1))
    assert len(set(cap.phase_frictionless)) == H                     # the phases are different books


def test_no_phase_sweep_at_a_one_bar_hold() -> None:
    panel = _panel()
    cap = _cap(panel, np.random.default_rng(2).standard_normal((T, N)), h=1)
    assert cap.phase_frictionless == () and np.isfinite(cap.lag1_frictionless_sharpe)


def test_lag_book_loses_an_edge_that_lives_in_the_next_bar() -> None:
    """A score that knows only the NEXT bar's return: the same-close book captures it, the
    one-bar-late book cannot."""
    panel = _panel(3)
    nxt = np.vstack([panel.close[1:] / panel.close[:-1] - 1.0, np.zeros((1, N))])
    cap = _cap(panel, nxt)
    assert cap.frictionless_sharpe > 3.0
    assert cap.lag1_frictionless_sharpe < 0.5 * cap.frictionless_sharpe


def _with(card, **kw):
    return replace(card, capturability=replace(card.capturability, **kw))


def test_phase_fragile_book_is_not_promising() -> None:
    card = _with(_card(1.0, net_std=0.5), phase_frictionless=(1.0, -0.4, -0.2, -0.1, 0.3))
    out = _verdict(card)
    assert out.verdict == "LOGGED"
    assert any("PHASE-FRAGILE" in c for c in out.caveats)
    off = _gates(capturability={"require_phase_robust": False})
    assert _verdict(card, off).verdict == "PROMISING"                # the leg is what blocked it


def test_phase_robust_book_and_unmeasured_sweep_are_unchanged() -> None:
    robust = _with(_card(1.0, net_std=0.5), phase_frictionless=(1.0, 0.8, 0.9, -0.1, 0.7))
    assert _verdict(robust).verdict == "PROMISING"
    assert _verdict(_card(1.0, net_std=0.5)).verdict == "PROMISING"  # no sweep on the card


def test_phase_leg_never_rescues_a_phase_zero_failure() -> None:
    card = _with(_card(-0.045), phase_frictionless=(-0.045, 0.4, 0.5, 0.6, 0.3))
    out = _verdict(card)
    assert out.verdict == "LOGGED"
    assert any("phase-0 artifact possible" in c for c in out.caveats)


def test_execution_lag_is_a_caveat_not_a_gate() -> None:
    card = _with(_card(1.0, net_std=0.5), lag1_frictionless_sharpe=0.3,
                 lag1_net_standard_sharpe=-0.1)
    out = _verdict(card)
    assert out.verdict == "PROMISING"
    assert any("EXECUTION-LAG SENSITIVE" in c for c in out.caveats)
    kept = _with(_card(1.0, net_std=0.5), lag1_frictionless_sharpe=0.9)
    assert not any("EXECUTION-LAG" in c for c in _verdict(kept).caveats)
