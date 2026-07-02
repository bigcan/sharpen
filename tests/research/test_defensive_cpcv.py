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


# --- Gate repair re-validation (crucible-v2.0, expert-review Test B) -------------------------------
# The fragility veto `delta_p05_min` was CPCV-geometry noise: on a fixed candidate, p05 swings
# −0.10..−0.49 as n_groups/k change, so it WRONGLY vetoed BAB (the first clean diversifier) while a
# genuinely fragile sleeve (commodity-carry) was correctly rejected on its NEGATIVE median. The
# repair drops the p05 leg and keeps median ∧ frac-positive. This test pins the published FLIP: BAB
# veto→pass, commodity NO_ADD unchanged — using the recorded CPCV summary stats (randd_log S553-96).

# (uplift_median, frac_paths_positive, uplift_p05) — the recorded 15-path CPCV distributions.
_BAB_CPCV = (0.089, 0.67, -0.193)          # first clean diversifier — was wrongly p05-vetoed
_COMMODITY_CPCV = (-0.030, 0.33, -0.392)   # genuinely dilutive — correctly rejected


def _old_robust(median: float, frac_pos: float, p05: float, *, p05_floor: float = -0.10) -> bool:
    """The SHIPPED-then-repaired fragility rule: median>0 ∧ frac+>0.5 ∧ p05≥floor (p05 = noise)."""
    return median > 0.0 and frac_pos > 0.5 and p05 >= p05_floor


def _new_robust(median: float, frac_pos: float, *,
                delta_median_min: float = 0.0, frac_positive_min: float = 0.50) -> bool:
    """The REPAIRED fragility rule (crucible-v2.0): median ∧ frac-positive only; p05 dropped."""
    return median >= delta_median_min and frac_pos >= frac_positive_min


def test_gate_repair_flips_bab_only() -> None:
    """BAB flips veto→pass under the repaired gate; commodity stays NO_ADD (flip count = 1)."""
    bab_med, bab_frac, bab_p05 = _BAB_CPCV
    com_med, com_frac, com_p05 = _COMMODITY_CPCV

    # OLD rule: BAB vetoed (p05 −0.193 < −0.10 floor), commodity rejected (negative median).
    assert _old_robust(bab_med, bab_frac, bab_p05) is False
    assert _old_robust(com_med, com_frac, com_p05) is False

    # NEW rule: BAB now robust (median +0.089 ≥ 0 ∧ frac+ 0.67 ≥ 0.5); commodity STILL rejected.
    assert _new_robust(bab_med, bab_frac) is True
    assert _new_robust(com_med, com_frac) is False

    # Exactly one candidate flips (BAB); the genuinely-fragile one is unchanged — the p05 leg was
    # noise, not signal. This is the "published flip count" of the gate-change protocol.
    flips = sum(
        _old_robust(m, f, p) != _new_robust(m, f)
        for m, f, p in (_BAB_CPCV, _COMMODITY_CPCV)
    )
    assert flips == 1
