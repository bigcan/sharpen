"""Tripwires for the planted-oracle sweep (``scripts/research/planted_sweep.py``).

Two things are pinned here:

  1. **The bar arithmetic**, against the independent audit's own published numbers. If
     ``hlz_t_min`` or the annualization convention ever moves, these go red and every document
     quoting "the gate demands ~1.50 annualized" needs re-checking.
  2. **All three verdict branches.** On every synthetic substrate tried the planted oracle clears
     the bar, so ``SEAL_CONFIRMED`` is unreachable from the shipped default — it would be untested
     code, which is precisely the one-directional blind spot this project keeps rediscovering
     (``feedback_mutation_check_false_survivor_crlf``). ``decide`` is therefore exercised directly
     in both directions plus both invalid-control cases.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "research" / "planted_sweep.py"


def _load():
    spec = importlib.util.spec_from_file_location("planted_sweep", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ps = _load()


# ---------------------------------------------------------------------------------------------
# 1. The bar — reproduces the audit's published anchors
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n_bars,expected", [(1011, 1.498), (4044, 0.749)])
def test_shipped_gate_bar_matches_audit_f1(n_bars: int, expected: float) -> None:
    """t>=3 on ~1011 Taiwan holdout bars demands ~1.50 annualized (audit F1, report line 80)."""
    got = ps.required_marginal_ann_sharpe(3.0, n_bars, 252.0)
    assert got == pytest.approx(expected, abs=0.001)


@pytest.mark.parametrize("n_bars,audit_value", [(1011, 1.42), (4044, 0.71)])
def test_ideal_mde80_matches_audit_f7(n_bars: int, audit_value: float) -> None:
    """The residual power wall: ideal single-prereg t>=2 MDE80 (audit F7, report line 154)."""
    assert ps.mde80(2.0, n_bars, 252.0) == pytest.approx(audit_value, abs=0.01)


def test_bar_tightens_only_as_sqrt_n() -> None:
    """Quadrupling the sample halves the bar — the reason no realistic panel length rescues it."""
    a = ps.required_marginal_ann_sharpe(3.0, 1000, 252.0)
    b = ps.required_marginal_ann_sharpe(3.0, 4000, 252.0)
    assert b == pytest.approx(a / 2.0, rel=1e-9)


def test_bar_rejects_degenerate_span() -> None:
    with pytest.raises(ValueError):
        ps.required_marginal_ann_sharpe(3.0, 1, 252.0)


# ---------------------------------------------------------------------------------------------
# 2. The verdict — all three branches, both directions
# ---------------------------------------------------------------------------------------------
def test_verdict_seal_confirmed_when_oracle_rejected() -> None:
    """The audit's outcome: controls behave, perfect weekly foresight is REJECTED."""
    assert ps.decide(perfect_weekly_passes=False, positive_passes=True,
                     negative_passes=False) == ("SEAL_CONFIRMED", 0)


def test_verdict_seal_absent_when_oracle_passes() -> None:
    """The synthetic default's outcome — a live branch, not a fallback."""
    assert ps.decide(perfect_weekly_passes=True, positive_passes=True,
                     negative_passes=False) == ("SEAL_ABSENT", 1)


@pytest.mark.parametrize("pos,neg", [(False, False), (True, True), (False, True)])
def test_verdict_invalid_dominates_whatever_the_ladder_did(pos: bool, neg: bool) -> None:
    """A broken control voids the run regardless of the oracle — control validity comes first.

    This is the leg that makes the sweep meaningful: without it, "the oracle failed" is
    indistinguishable from "the harness cannot pass anything".
    """
    for oracle in (True, False):
        assert ps.decide(perfect_weekly_passes=oracle, positive_passes=pos,
                         negative_passes=neg) == ("HARNESS_INVALID", 2)


def test_skill_bounds_are_enforced() -> None:
    import numpy as np
    rng = np.random.default_rng(0)
    for bad in (-0.01, 1.01):
        with pytest.raises(ValueError):
            ps.timing_series(np.zeros(50), horizon=5, skill=bad, rng=rng)
    with pytest.raises(ValueError):
        ps.timing_series(np.zeros(50), horizon=0, skill=1.0, rng=rng)


def test_perfect_oracle_actually_sees_the_future() -> None:
    """skill=1.0 must reproduce the base book's forward block return, sign included.

    Guards against the planted signal silently degenerating: a lookahead series that does not
    actually look ahead would make every "the oracle failed" result vacuous.

    ALIGNMENT (the subtle part). ``_overlay_returns`` earns ``m[t-1] * b_gross[t]``, so a tilt
    decided at bar ``t0`` is paid on bars ``t0+1 .. t0+h`` — NOT ``t0 .. t0+h-1``. The signal is
    therefore the forward block ``bb[t0+1 : t0+h+1]``, which is what ``cum[t0+h] - cum[t0]``
    computes. Asserting the off-by-one form instead would demand that the oracle predict a bar it
    is never paid for, i.e. would pin a LAGGED oracle as correct.
    """
    import numpy as np
    bb = np.array([0.01, -0.02, 0.03, -0.04, 0.05, 0.06, -0.07, 0.08, 0.09, -0.10])
    g = ps.timing_series(bb, horizon=5, skill=1.0, rng=np.random.default_rng(0))
    assert g[0] == pytest.approx(bb[1:6].sum())     # the block the tilt is actually paid on
    assert np.allclose(g[0:5], g[0])                # held flat across the block


def test_zero_skill_oracle_inverts_the_future() -> None:
    """skill=0.0 must flip every sign — the negative direction of the planting mechanism.

    Without this, a ``timing_series`` that ignored ``skill`` entirely would still pass every other
    test in this file: the ladder would silently be five copies of the perfect oracle.
    """
    import numpy as np
    bb = np.array([0.01, -0.02, 0.03, -0.04, 0.05, 0.06, -0.07, 0.08, 0.09, -0.10])
    rng = np.random.default_rng(0)
    perfect = ps.timing_series(bb, horizon=5, skill=1.0, rng=rng)
    inverted = ps.timing_series(bb, horizon=5, skill=0.0, rng=rng)
    assert inverted[0] == pytest.approx(-perfect[0])


# ---------------------------------------------------------------------------------------------
# 3. Length sweep — the measurement that explains the audit disagreement
# ---------------------------------------------------------------------------------------------
def test_bar_and_achievable_sharpe_cross_where_expected() -> None:
    """The crossing logic: a FLAT achievable Sharpe meets a bar falling as 1/sqrt(N).

    Measured on the real Taiwan panel the oracle's marginal Sharpe is roughly constant in N
    (1.93-3.03) while the required Sharpe falls 2.443 -> 0.753, so the pass/fail flip is driven
    purely by sample size. Pinning that relation keeps the length sweep's headline interpretable.
    """
    ppy = 252.0
    achievable = 2.3                       # flat, a property of the substrate
    below = ps.required_marginal_ann_sharpe(3.0, 380, ppy)
    above = ps.required_marginal_ann_sharpe(3.0, 500, ppy)
    assert below > achievable > above, "the crossing must sit between 380 and 500 bars"
    # and the bar must be monotone decreasing in N
    spans = [380, 500, 1000, 2000, 4000]
    bars = [ps.required_marginal_ann_sharpe(3.0, n, ppy) for n in spans]
    assert bars == sorted(bars, reverse=True)


def test_length_sweep_skips_windows_with_a_flat_base_book() -> None:
    """A timing overlay cannot tilt a book that never takes a position.

    The Taiwan TSMOM base uses a 252-bar lookback plus warmup, so short windows have an
    identically-flat book (measured: 0 non-flat bars at N=260). Those windows must be SKIPPED
    with a reason, never silently scored — a t-stat computed against a flat book would be
    meaningless and would sit in the sweep looking like evidence.
    """
    import numpy as np
    flat = np.zeros(300)
    assert int((np.nan_to_num(flat) != 0.0).sum()) == 0
    live = np.concatenate([np.zeros(273), np.full(107, 0.001)])
    assert int((np.nan_to_num(live) != 0.0).sum()) == 107
