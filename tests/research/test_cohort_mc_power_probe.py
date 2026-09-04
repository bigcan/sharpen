"""Tripwires for the cohort MC-null power probe (``scripts/research/cohort_mc_power_probe.py``).

The probe measures a POWER figure, and a power figure is only readable if the harness can also
produce a correct NULL. The first version of `build_cell` demeaned every candidate to exactly zero
sample mean, which made all standalone Sharpes identically 0.0, degenerated the admission ranking,
and drove `t_obs` to the bottom of the comparison — p came back 1.0000 on every seed, null and
alternative alike. That would have read as a clean refutation of the published 0.25.

So the pool construction is pinned in both directions: candidate realized means must VARY (or
selection has nothing to select on), and the planted edge must actually be present.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "research" / "cohort_mc_power_probe.py"


def _load():
    spec = importlib.util.spec_from_file_location("cohort_mc_power_probe", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


probe = _load()


def _cell(n_real: int = 6, ir: float = 0.30, t: int = 2520, pool: int = 40, seed: int = 0):
    return probe.build_cell(t=t, pool_size=pool, n_real=n_real, ir=ir, base_sharpe=0.50,
                            ppy=252.0, rng=np.random.default_rng(seed))


# ---------------------------------------------------------------------------------------------
# The pool must be SELECTABLE — the defect that produced p = 1.0000 everywhere
# ---------------------------------------------------------------------------------------------
def test_candidate_realized_sharpes_vary() -> None:
    """Standalone Sharpes must have real dispersion, or the admission ranking is meaningless.

    This is the regression for the demean-to-exactly-zero bug: with every mean forced to 0 the
    Sharpes are all 0.0, `greedy_decorrelated_admission` ranks arbitrarily, and the observed
    statistic sits below every bootstrap replicate by construction.
    """
    _, pool, _, _ = _cell(n_real=0)
    srs = np.array([float(np.mean(v) / np.std(v, ddof=1)) for v in pool.values()])
    assert np.std(srs) > 1e-4, f"candidate Sharpes are degenerate (sd={np.std(srs):.2e})"
    assert len(np.unique(np.round(srs, 12))) > len(srs) // 2, "Sharpes must not be near-identical"


def test_null_pool_has_no_systematic_edge() -> None:
    """With 0 planted signals the pool's mean edge must be ~0 — otherwise the 'null' isn't one."""
    _, pool, _, real = _cell(n_real=0)
    assert real == set()
    means = np.array([float(np.mean(v)) for v in pool.values()])
    se = float(np.std(means, ddof=1) / np.sqrt(means.size))
    assert abs(float(np.mean(means))) < 4.0 * se, "null pool carries a systematic drift"


# ---------------------------------------------------------------------------------------------
# The planted edge must actually be planted
# ---------------------------------------------------------------------------------------------
def test_planted_signals_carry_the_requested_ir() -> None:
    """The real members' expected annualized Sharpe must match `ir` (within sampling noise).

    Without this, an `ir` argument that silently did nothing would leave every power measurement
    looking like a null — indistinguishable from 'the gate has no power'.
    """
    _, pool, _, real = _cell(n_real=6, ir=0.30)
    assert len(real) == 6
    got = np.mean([float(np.mean(pool[n]) / np.std(pool[n], ddof=1) * np.sqrt(252.0)) for n in real])
    assert got == pytest.approx(0.30, abs=0.35), f"planted IR not delivered (got {got:.3f})"
    noise = [n for n in pool if n not in real]
    got_noise = np.mean([float(np.mean(pool[n]) / np.std(pool[n], ddof=1) * np.sqrt(252.0))
                         for n in noise])
    assert abs(got_noise) < 0.25, f"noise members carry an edge ({got_noise:.3f})"


def test_planted_signals_are_orthogonal_to_the_base() -> None:
    base, pool, _, real = _cell(n_real=6)
    b = base["base"]
    for n in real:
        assert abs(float(np.corrcoef(b, pool[n])[0, 1])) < 0.10


def test_zero_ir_is_indistinguishable_from_noise() -> None:
    """ir=0 must produce a pool with no edge — the negative direction of the planting mechanism."""
    _, pool, _, real = _cell(n_real=6, ir=0.0)
    got = np.mean([float(np.mean(pool[n]) / np.std(pool[n], ddof=1) * np.sqrt(252.0)) for n in real])
    assert abs(got) < 0.25


def test_cell_shape_and_determinism() -> None:
    base, pool, ts, _ = _cell()
    assert len(pool) == 40 and len(base) == 1
    assert ts.shape == (2520,) and np.all(np.diff(ts) > 0)
    a = _cell(seed=3)[1]["c00"]
    b = _cell(seed=3)[1]["c00"]
    assert np.allclose(a, b)


# ---------------------------------------------------------------------------------------------
# The reported interval must be honest about how little 24 seeds pins down
# ---------------------------------------------------------------------------------------------
def test_wilson_ci_brackets_and_widens_correctly() -> None:
    """A bare pass rate over ~24 seeds is a wide estimate; the CI is what stops it being oversold."""
    lo, hi = probe.wilson_ci(3, 12)                 # the published 0.25 at its own seed count
    assert lo < 0.25 < hi
    assert hi - lo > 0.35, "12 seeds cannot pin a rate to better than ~0.35 wide"
    lo2, hi2 = probe.wilson_ci(30, 120)             # same rate, 10x the seeds
    assert (hi2 - lo2) < (hi - lo), "more seeds must narrow the interval"
    assert probe.wilson_ci(0, 0) == pytest.approx((float("nan"), float("nan")), nan_ok=True)


def test_wilson_ci_edges() -> None:
    lo, hi = probe.wilson_ci(0, 24)
    assert lo == 0.0 and 0.0 < hi < 0.20
    lo, hi = probe.wilson_ci(24, 24)
    assert hi == 1.0 and 0.80 < lo < 1.0
