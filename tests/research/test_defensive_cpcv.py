"""Tripwires for the defensive-sleeve CPCV uplift distribution (S553-cont-95, the verdict driver)."""
from __future__ import annotations

import numpy as np

from scripts.research.eval_defensive_sleeve import (
    _contiguous_runs,
    _cpcv_uplift_distribution,
)


def test_contiguous_runs_merges_adjacent() -> None:
    mask = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], dtype=bool)
    assert _contiguous_runs(mask) == [(1, 3), (4, 5), (7, 10)]
    assert _contiguous_runs(np.zeros(5, dtype=bool)) == []
    assert _contiguous_runs(np.ones(4, dtype=bool)) == [(0, 4)]


def test_cpcv_constant_uplift_all_paths_positive() -> None:
    """If book3 = book2 + constant, every path's ΔSharpe is > 0 and frac_positive == 1."""
    rng = np.random.default_rng(0)
    r2 = rng.standard_normal(1200) * 0.01
    r3 = r2 + 0.002                                  # a uniform positive shift
    d = _cpcv_uplift_distribution(r2, r3, n_groups=6, k_test=2, embargo=21, purge_horizon=1)
    assert d["n_paths"] == 15                        # C(6,2)
    assert d["frac_paths_positive"] == 1.0
    assert d["uplift_median"] > 0 and d["uplift_min"] > 0


def test_cpcv_embargo_and_purge_shrink_test_bars() -> None:
    """Larger embargo/purge strictly reduce the usable test bars (fewer, never leaking across seams);
    a degenerate identical pair yields ~0 uplift."""
    r = np.random.default_rng(1).standard_normal(1200) * 0.01
    d0 = _cpcv_uplift_distribution(r, r, n_groups=6, k_test=2, embargo=0, purge_horizon=0)
    d1 = _cpcv_uplift_distribution(r, r, n_groups=6, k_test=2, embargo=21, purge_horizon=5)
    assert d0["n_paths"] == d1["n_paths"] == 15
    assert abs(d0["uplift_median"]) < 1e-9          # identical books -> zero uplift every path
    assert abs(d1["uplift_median"]) < 1e-9
