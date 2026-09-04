"""Tripwires for the matched-null simulator (``scripts/research/null_grid_sim.py``).

The whole of finding F4 rests on one construction: a panel that differs from the real substrate in
EXACTLY one respect — no predictable structure. If the permutation is not matched, the comparison
measures a substrate difference; if it is not actually null, the "noise ceiling" is not a noise
ceiling. Both failure directions are silent, so both are pinned here.

This is the same discipline that caught the synthetic arm of ``planted_sweep``: a null that is
easier or harder than the real thing inverts the verdict without any error surfacing.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "research" / "null_grid_sim.py"


def _load():
    spec = importlib.util.spec_from_file_location("null_grid_sim", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ngs = _load()


def _panel(T: int = 600, N: int = 8, seed: int = 3):
    from sharpen.signals.features import make_synthetic_panel
    return make_synthetic_panel(T=T, N=N, seed=seed)


def _returns(close: np.ndarray) -> np.ndarray:
    return close[1:] / close[:-1] - 1.0


# ---------------------------------------------------------------------------------------------
# MATCHED: what the null must PRESERVE
# ---------------------------------------------------------------------------------------------
def test_permutation_preserves_marginal_return_distribution() -> None:
    """Each name's set of returns must survive intact — only their ORDER changes.

    If the null quietly changed return magnitudes it would be a different substrate, and the
    "noise reaches the same dSR as the record" comparison would be meaningless.
    """
    p = _panel()
    q = ngs.permute_panel(p, np.random.default_rng(0))
    r0, r1 = np.sort(_returns(p.close), axis=0), np.sort(_returns(q.close), axis=0)
    assert np.allclose(r0, r1, atol=1e-10), "sorted per-name returns must be identical"


def test_permutation_preserves_cross_sectional_covariance() -> None:
    """Contemporaneous cross-name covariance must survive — whole rows move together.

    A null that also scrambled the cross-section would hand a rank-L/S search a much easier (or
    harder) problem than the real panel, which is exactly the confound that inverted the synthetic
    arm of the planted-oracle sweep.
    """
    p = _panel()
    q = ngs.permute_panel(p, np.random.default_rng(0))
    c0, c1 = np.corrcoef(_returns(p.close).T), np.corrcoef(_returns(q.close).T)
    iu = np.triu_indices_from(c0, 1)
    assert np.allclose(c0[iu], c1[iu], atol=1e-8), "pairwise correlations must be preserved"


def test_permutation_preserves_ohlc_sanity() -> None:
    """high >= max(open, close) and low <= min(open, close), else candidates are culled upstream
    and the search is silently thinned on the null but not on the real panel."""
    q = ngs.permute_panel(_panel(), np.random.default_rng(1))
    hi, lo = np.asarray(q.high), np.asarray(q.low)
    op, cl = np.asarray(q.open), np.asarray(q.close)
    assert np.all(hi >= np.maximum(op, cl) - 1e-9)
    assert np.all(lo <= np.minimum(op, cl) + 1e-9)
    assert np.all(np.isfinite(cl)) and np.all(cl > 0)


def test_permutation_preserves_shape_and_calendar() -> None:
    p = _panel()
    q = ngs.permute_panel(p, np.random.default_rng(2))
    assert q.T == p.T and q.N == p.N
    assert np.array_equal(q.dates, p.dates)
    assert q.tickers == p.tickers


# ---------------------------------------------------------------------------------------------
# NULL: what the null must DESTROY
# ---------------------------------------------------------------------------------------------
def test_permutation_destroys_temporal_structure() -> None:
    """The point of the null: no lag-1 autocorrelation left to exploit.

    Built on a deliberately AUTOCORRELATED series so the test can fail — on an i.i.d. panel the
    "after" value would look fine no matter what the function did, which is the one-directional
    trap this file exists to avoid.
    """
    from dataclasses import replace
    p = _panel(T=1500, N=6)
    rng = np.random.default_rng(5)
    # AR(1) with strong positive persistence, compounded into prices.
    r = np.zeros((p.T - 1, p.N))
    eps = rng.standard_normal((p.T - 1, p.N)) * 0.01
    for t in range(1, p.T - 1):
        r[t] = 0.60 * r[t - 1] + eps[t]
    close = np.empty_like(np.asarray(p.close))
    close[0] = 100.0
    close[1:] = 100.0 * np.cumprod(1.0 + r, axis=0)
    ac = replace(p, close=close, open=close, high=close * 1.001, low=close * 0.999)

    def lag1(c):
        x = _returns(c)
        return float(np.mean([np.corrcoef(x[:-1, j], x[1:, j])[0, 1] for j in range(x.shape[1])]))

    before = lag1(ac.close)
    after = lag1(ngs.permute_panel(ac, np.random.default_rng(0)).close)
    assert before > 0.4, f"fixture must actually be autocorrelated (got {before:.3f})"
    assert abs(after) < 0.10, f"permutation must destroy it (got {after:.3f})"


def test_permutation_actually_reorders() -> None:
    """A no-op permutation would pass every preservation test above — pin that it moves rows."""
    p = _panel()
    q = ngs.permute_panel(p, np.random.default_rng(9))
    assert not np.allclose(np.asarray(p.close), np.asarray(q.close)), "panel must change"


def test_permutation_is_seed_deterministic() -> None:
    p = _panel()
    a = ngs.permute_panel(p, np.random.default_rng(4)).close
    b = ngs.permute_panel(p, np.random.default_rng(4)).close
    assert np.allclose(np.asarray(a), np.asarray(b))


# ---------------------------------------------------------------------------------------------
# Base-book rescaling (the matched nuisance parameter, audit F2)
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("target", [-0.25, 0.0, 0.5, 1.3])
def test_retarget_sharpe_hits_its_target_without_touching_vol(target: float) -> None:
    from sharpen.signals.eval_harness import _ann_sharpe
    rng = np.random.default_rng(11)
    r = rng.standard_normal(2000) * 0.01 + 0.0003
    out = ngs.retarget_sharpe(r, target, 252.0)
    assert _ann_sharpe(out, 252.0) == pytest.approx(target, abs=1e-6)
    assert float(out.std(ddof=1)) == pytest.approx(float(r.std(ddof=1)), rel=1e-12)


def test_seed_bank_is_the_production_one() -> None:
    """The null search must start from the SAME seeds production mines from, or the file-drawer
    it explores is not the project's file-drawer."""
    from sharpen.crucible.agentic.proposer import _CS_SEED_BANK
    seeds = ngs.seed_formulas()
    assert len(seeds) == len([i for i, _, _, _ in _CS_SEED_BANK if i != 56])
    assert all(isinstance(s, str) and s for s in seeds)


# ---------------------------------------------------------------------------------------------
# STAGGERED LISTING — the case that broke the first implementation
# ---------------------------------------------------------------------------------------------
def _staggered_panel(T: int = 900, N: int = 10):
    """A panel whose names list at different dates, like the real Taiwan ETF cross-section.

    Every fixture above starts all names at bar 0, which is why they all passed against a
    permutation that anchored each series at ``close[0]`` — a construction that silently NaN'd out
    every late lister for its whole history. Measured on the real panel that left 3 usable names
    of 10 and ZERO bars reaching ``ls_min_names``, so no candidate could form a book and the null
    ceiling was an artifact. This fixture is the missing coverage.
    """
    from dataclasses import replace
    from sharpen.signals.features import make_synthetic_panel
    p = make_synthetic_panel(T=T, N=N, seed=17)
    close = np.asarray(p.close, dtype=float).copy()
    active = np.asarray(p.active).copy()
    for j in range(N):
        start = int(j * (T // (2 * N)))          # name j lists progressively later
        close[:start, j] = np.nan
        active[:start, j] = False
    return replace(p, close=close, open=close, high=close * 1.001, low=close * 0.999,
                   active=active)


def test_permutation_keeps_every_late_listing_name() -> None:
    """A name that lists mid-panel must survive the permutation with its history intact."""
    p = _staggered_panel()
    q = ngs.permute_panel(p, np.random.default_rng(0))
    live_before = np.isfinite(np.asarray(p.close)).any(axis=0).sum()
    live_after = np.isfinite(np.asarray(q.close)).any(axis=0).sum()
    assert live_after == live_before == p.N, (
        f"permutation dropped names: {live_before} -> {live_after}")


def test_permutation_preserves_per_name_live_bar_counts() -> None:
    """Each name must keep exactly as many tradeable bars as it really had."""
    p = _staggered_panel()
    q = ngs.permute_panel(p, np.random.default_rng(1))
    before = np.isfinite(np.asarray(p.close)).sum(axis=0)
    after = np.isfinite(np.asarray(q.close)).sum(axis=0)
    assert np.array_equal(before, after), f"live-bar counts changed: {before} -> {after}"


def test_permutation_preserves_tradeable_cross_section_width() -> None:
    """The count of bars with enough active names must survive — this is the leg the search needs.

    ``_ls_weights`` returns an all-zero book below ``ls_min_names``, so a permutation that thins
    the cross-section makes every candidate flat and yields a VACUOUS 'ceiling' that reads exactly
    like a measured one.
    """
    p = _staggered_panel()
    q = ngs.permute_panel(p, np.random.default_rng(2))
    def wide(x, k=6):
        a = np.isfinite(np.asarray(x.close)) & np.asarray(x.active)
        return int((a.sum(axis=1) >= k).sum())
    assert wide(p) > 100, "fixture must have a usable cross-section to begin with"
    assert wide(q) == wide(p), f"tradeable bars changed: {wide(p)} -> {wide(q)}"


def test_permutation_still_destroys_structure_with_staggered_listing() -> None:
    """The fix must not have reintroduced temporal structure for the late listers."""
    p = _staggered_panel(T=1500, N=6)
    q = ngs.permute_panel(p, np.random.default_rng(3))
    c = np.asarray(q.close)[:, -1]
    c = c[np.isfinite(c)]
    x = c[1:] / c[:-1] - 1.0
    assert abs(float(np.corrcoef(x[:-1], x[1:])[0, 1])) < 0.12
