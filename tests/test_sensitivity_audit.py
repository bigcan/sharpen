"""Unit tests for ``finrl_pro_ds.eval.sensitivity_audit`` (v2.6 C2, S553).

Covers: pure metric helpers, grid construction, edge-stability resolver,
verdict block serializer + writer. The impure ``run_cell`` is exercised
by the C4 smoke test and operator backfill — only its invariant-assertion
preflight is tested here.

Architecture: ``.agent/artifacts/protocol_v27_a_sensitivity_audit_architecture.md``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finrl_pro_ds.eval.sensitivity_audit import (  # noqa: E402
    PF_CAP,
    PF_XCHECK_DIVERGENCE_HALT,
    PF_XCHECK_REPORT_ONLY,
    SCHEMA_VERSION,
    CellResult,
    CellSpec,
    InvariantViolation,
    PfXCheckResult,
    _assert_invariants,
    build_grid,
    build_sensitivity_audit_block,
    center_index,
    compute_pf_xcheck,
    mdd_from_pv,
    pf_from_pv,
    resolve_edge_stability,
    write_verdict_v26,
)


# ---------------------------------------------------------------------------
# pf_from_pv / mdd_from_pv
# ---------------------------------------------------------------------------


def test_pf_from_pv_flat_curve_returns_zero():
    """Constant PV (no returns) → PF=0 by convention."""
    pv = np.array([100.0] * 10)
    assert pf_from_pv(pv) == 0.0


def test_pf_from_pv_monotonic_gain_clipped_to_cap():
    """All-positive returns with no losses → clipped at PF_CAP."""
    pv = np.linspace(100.0, 200.0, 50)
    assert pf_from_pv(pv) == PF_CAP


def test_pf_from_pv_monotonic_loss_returns_zero():
    """All-negative returns with no wins → PF=0."""
    pv = np.linspace(100.0, 50.0, 50)
    assert pf_from_pv(pv) == 0.0


def test_pf_from_pv_balanced_returns_approx_one():
    """Symmetric +1% / -1% bars → PF≈1.0 (with small bias from compounding)."""
    rng = np.random.default_rng(42)
    rets = rng.choice([0.01, -0.01], size=200)
    pv = 100.0 * np.cumprod(1.0 + rets)
    pv = np.concatenate([[100.0], pv])
    pf = pf_from_pv(pv)
    assert 0.5 < pf < 2.0, f"balanced PF should be near 1.0; got {pf}"


def test_pf_from_pv_short_array_returns_zero():
    """Insufficient bars → PF=0."""
    assert pf_from_pv(np.array([100.0])) == 0.0
    assert pf_from_pv(np.array([])) == 0.0


def test_mdd_from_pv_monotonic_gain_is_zero():
    pv = np.linspace(100.0, 200.0, 50)
    assert mdd_from_pv(pv) == 0.0


def test_mdd_from_pv_50pct_drawdown():
    pv = np.array([100.0, 100.0, 50.0, 50.0])
    mdd = mdd_from_pv(pv)
    assert mdd == pytest.approx(-0.50, abs=1e-9)


def test_mdd_from_pv_peak_then_recover_records_trough():
    pv = np.array([100.0, 110.0, 80.0, 120.0])
    mdd = mdd_from_pv(pv)
    # Peak=110, trough=80 → DD = 80/110 - 1 = -0.2727...
    assert mdd == pytest.approx(80.0 / 110.0 - 1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# compute_pf_xcheck
# ---------------------------------------------------------------------------


def test_pf_xcheck_skipped_when_mid_absent():
    """N2 rename: pv_mid (the optional independent mark) None → SKIPPED."""
    pv = np.linspace(100.0, 105.0, 20)
    result = compute_pf_xcheck(pv, pv_mid=None)
    assert result.status == "SKIPPED"
    assert result.pass_
    assert result.divergence == 0.0


def test_pf_xcheck_pass_when_curves_match():
    pv = np.linspace(100.0, 105.0, 20)
    result = compute_pf_xcheck(pv, pv_mid=pv.copy())
    assert result.status == "PASS"
    assert result.pass_


def test_pf_xcheck_halt_when_divergence_exceeds_threshold():
    """Close PF = high (reference), mid PF = 0 → divergence = 100% → HALT."""
    pv_close = np.linspace(100.0, 200.0, 50)  # PF_CAP
    pv_mid = np.linspace(100.0, 50.0, 50)  # PF=0
    result = compute_pf_xcheck(pv_close, pv_mid=pv_mid)
    assert result.status == "HALT"
    assert not result.pass_
    assert result.divergence > PF_XCHECK_DIVERGENCE_HALT


def test_pf_xcheck_pass_when_divergence_below_threshold():
    """Modest sub-threshold divergence → PASS."""
    pv_close = np.array([100.0, 101.0, 100.5, 102.0, 101.5])
    pv_mid = pv_close * 1.001  # tiny perturbation
    result = compute_pf_xcheck(pv_close, pv_mid=pv_mid)
    assert result.status == "PASS"


def test_pf_xcheck_report_only_suppresses_halt():
    """ADR-N5: report_only surfaces the divergence but never HALTs."""
    pv_close = np.linspace(100.0, 200.0, 50)  # PF_CAP
    pv_mid = np.linspace(100.0, 50.0, 50)  # PF=0 → would HALT
    result = compute_pf_xcheck(pv_close, pv_mid=pv_mid, report_only=True)
    assert result.status == "PASS"
    assert result.pass_
    # Divergence is still computed and surfaced for calibration.
    assert result.divergence > PF_XCHECK_DIVERGENCE_HALT
    # The shipped default is Phase-α report-only; gates can override per
    # workstream once the threshold is locked (N2/ADR-N5).
    assert PF_XCHECK_REPORT_ONLY is True


def test_run_cell_pf_xcheck_params_default_to_module_constants():
    """N2 (ADR-N5): run_cell exposes gates-driven PF-XCHECK params that default
    to the module constants, so direct callers/tests keep the shipped Phase-α
    behavior while the orchestrator overrides them from the gates overlay."""
    import inspect

    from finrl_pro_ds.eval.sensitivity_audit import run_cell

    sig = inspect.signature(run_cell)
    assert sig.parameters["pf_xcheck_report_only"].default is PF_XCHECK_REPORT_ONLY
    assert (
        sig.parameters["pf_xcheck_divergence_halt"].default
        == PF_XCHECK_DIVERGENCE_HALT
    )


def test_pf_xcheck_denominator_is_close_marked():
    """ADR-N4: divergence normalizes by the CLOSE-marked PF, not mid."""
    pv_close = np.array([100.0, 103.0, 101.0, 104.0, 102.0])
    pv_mid = np.array([100.0, 101.0, 100.5, 102.0, 101.0])
    pf_c = pf_from_pv(pv_close)
    pf_m = pf_from_pv(pv_mid)
    result = compute_pf_xcheck(pv_close, pv_mid=pv_mid)
    assert result.pf_close == pytest.approx(pf_c, abs=1e-12)
    assert result.pf_mid == pytest.approx(pf_m, abs=1e-12)
    expected = abs(pf_c - pf_m) / max(abs(pf_c), 1e-9)
    assert result.divergence == pytest.approx(expected, abs=1e-12)


def test_pf_xcheck_halts_two_sided_when_mid_rosier():
    """Two-sided gate: a much HIGHER mid PF also HALTs (marking-sensitive edge)."""
    pv_close = np.array([100.0, 101.0, 100.0, 101.0, 100.0, 101.0])  # choppy, PF≈1
    pv_mid = np.linspace(100.0, 130.0, 6)  # monotonic up → PF_CAP
    result = compute_pf_xcheck(pv_close, pv_mid=pv_mid)
    assert result.pf_mid > result.pf_close
    assert result.status == "HALT"
    assert result.divergence > PF_XCHECK_DIVERGENCE_HALT


# ---------------------------------------------------------------------------
# CellSpec
# ---------------------------------------------------------------------------


def test_cell_spec_label_format():
    spec = CellSpec(
        deadband_threshold=0.25,
        max_leverage=1.0,
        max_leverage_mult=1.0,
        deployable=True,
    )
    assert spec.label() == "d0250_m1000"


def test_cell_spec_effective_deadband_b5_coupling():
    """B5 fix: effective_deadband = deadband_threshold * max_leverage."""
    spec = CellSpec(
        deadband_threshold=0.30,
        max_leverage=1.5,
        max_leverage_mult=1.5,
        deployable=True,
    )
    assert spec.effective_deadband == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# build_grid
# ---------------------------------------------------------------------------


def _baseline_deployed() -> dict:
    return {"deadband_threshold": 0.25, "max_leverage": 1.0}


def _baseline_gates(**overrides) -> dict:
    g = {
        "edge_stability_pf_ratio_floor": 0.70,
        "sensitivity_deadband_grid": [0.20, 0.25, 0.30],
        "sensitivity_max_leverage_mults": [0.5, 1.0, 1.5],
        "sensitivity_deployable_max_leverage_cap": 1.0,
        "sensitivity_audit_required_min_deployable_neighbors": 4,
        "sensitivity_audit_required": True,
    }
    g.update(overrides)
    return g


def test_build_grid_default_shape():
    """3x3 grid: 9 cells, row-major (deadband outer, max_lev inner)."""
    cells = build_grid(_baseline_deployed(), _baseline_gates())
    assert len(cells) == 9
    # Center cell at index 4 = (0.25, 1.0)
    assert cells[4].deadband_threshold == 0.25
    assert cells[4].max_leverage == 1.0
    # Row-major: cells[0] = (0.20, 0.5), cells[2] = (0.20, 1.5)
    assert cells[0].deadband_threshold == 0.20
    assert cells[0].max_leverage == 0.5
    assert cells[2].max_leverage == 1.5
    # cells[6] = (0.30, 0.5), cells[8] = (0.30, 1.5)
    assert cells[6].deadband_threshold == 0.30
    assert cells[8].max_leverage == 1.5


def test_build_grid_deployable_cap_excludes_high_max_lev():
    """With cap=1.0, cells with max_lev=1.5 are non-deployable."""
    cells = build_grid(_baseline_deployed(), _baseline_gates())
    high_lev = [c for c in cells if c.max_leverage > 1.0 + 1e-9]
    assert len(high_lev) == 3
    assert all(not c.deployable for c in high_lev)
    assert all("cap" in (c.deployable_reason or "") for c in high_lev)
    # 6 deployable cells
    assert sum(1 for c in cells if c.deployable) == 6


def test_build_grid_deployable_cap_relaxed_allows_all():
    """With cap=2.0, all 9 cells deployable."""
    gates = _baseline_gates(sensitivity_deployable_max_leverage_cap=2.0)
    cells = build_grid(_baseline_deployed(), gates)
    assert all(c.deployable for c in cells)


def test_build_grid_sens1_deadband_center_mismatch_raises():
    """SENS-1: grid center [0.30] does not match deployed deadband [0.25]."""
    gates = _baseline_gates(sensitivity_deadband_grid=[0.25, 0.30, 0.35])
    with pytest.raises(InvariantViolation, match="SENS-1"):
        build_grid(_baseline_deployed(), gates)


def test_build_grid_sens1_max_lev_mult_center_not_one_raises():
    gates = _baseline_gates(sensitivity_max_leverage_mults=[0.5, 1.2, 1.5])
    with pytest.raises(InvariantViolation, match="SENS-1"):
        build_grid(_baseline_deployed(), gates)


def test_build_grid_wrong_axis_length_raises():
    gates = _baseline_gates(sensitivity_deadband_grid=[0.20, 0.25])
    with pytest.raises(InvariantViolation, match="3-element"):
        build_grid(_baseline_deployed(), gates)


def test_build_grid_max_leverage_mults_scale_relative_to_deployed():
    """Deployed max_leverage=2.0 + mults=[0.5,1.0,1.5] → values [1.0,2.0,3.0]."""
    deployed = {"deadband_threshold": 0.25, "max_leverage": 2.0}
    gates = _baseline_gates(sensitivity_deployable_max_leverage_cap=2.0)
    cells = build_grid(deployed, gates)
    ml_values = sorted({c.max_leverage for c in cells})
    assert ml_values == [1.0, 2.0, 3.0]
    # cells at max_lev=3.0 are non-deployable (cap=2.0)
    non_dep = [c for c in cells if c.max_leverage > 2.0 + 1e-9]
    assert len(non_dep) == 3
    assert all(not c.deployable for c in non_dep)


# ---------------------------------------------------------------------------
# center_index
# ---------------------------------------------------------------------------


def test_center_index_returns_4_for_centered_grid():
    cells = build_grid(_baseline_deployed(), _baseline_gates())
    assert center_index(cells, _baseline_deployed()) == 4


def test_center_index_raises_when_no_match():
    cells = build_grid(_baseline_deployed(), _baseline_gates())
    wrong = {"deadband_threshold": 0.99, "max_leverage": 0.99}
    with pytest.raises(ValueError, match="No center cell"):
        center_index(cells, wrong)


# ---------------------------------------------------------------------------
# resolve_edge_stability
# ---------------------------------------------------------------------------


def _make_cell_result(
    spec: CellSpec,
    pf: float,
    *,
    halt: bool = False,
    mdd: float = -0.05,
) -> CellResult:
    return CellResult(
        spec=spec,
        pf_test=pf,
        mdd_test=mdd,
        n_bars=1000,
        pf_xcheck=PfXCheckResult(
            status="SKIPPED",
            divergence=0.0,
            pf_mid=pf,
            pf_close=pf,
        ),
        halt=halt,
        halt_reason=None,
        trajectory_paths=[],
    )


def _make_homogeneous_cells(specs: list, center_pf: float, neighbor_pf: float):
    """Build cell results: center has center_pf, all others have neighbor_pf."""
    deployed = _baseline_deployed()
    cells = []
    for s in specs:
        is_center = (
            abs(s.deadband_threshold - deployed["deadband_threshold"]) < 1e-9
            and abs(s.max_leverage - deployed["max_leverage"]) < 1e-9
        )
        cells.append(_make_cell_result(s, center_pf if is_center else neighbor_pf))
    return cells


def test_resolve_edge_stability_pass_when_ratio_above_floor():
    """Center PF=2.0, all neighbors PF=1.5 → ratio=0.75 ≥ 0.70 → PASS."""
    specs = build_grid(_baseline_deployed(), _baseline_gates())
    cells = _make_homogeneous_cells(specs, center_pf=2.0, neighbor_pf=1.5)
    verdict = resolve_edge_stability(cells, _baseline_deployed(), _baseline_gates())
    assert verdict.decision == "PASS"
    assert verdict.pass_
    assert verdict.pf_ratio == pytest.approx(0.75, abs=1e-9)


def test_resolve_edge_stability_fail_when_ratio_below_floor():
    """Center PF=2.0, worst neighbor PF=0.5 → ratio=0.25 < 0.70 → FAIL."""
    specs = build_grid(_baseline_deployed(), _baseline_gates())
    cells = []
    deployed = _baseline_deployed()
    for i, s in enumerate(specs):
        is_center = (
            abs(s.deadband_threshold - deployed["deadband_threshold"]) < 1e-9
            and abs(s.max_leverage - deployed["max_leverage"]) < 1e-9
        )
        if is_center:
            pf = 2.0
        elif i == 0:  # one bad neighbor
            pf = 0.5
        else:
            pf = 1.8
        cells.append(_make_cell_result(s, pf))
    verdict = resolve_edge_stability(cells, deployed, _baseline_gates())
    assert verdict.decision == "FAIL"
    assert not verdict.pass_
    assert verdict.pf_ratio == pytest.approx(0.25, abs=1e-9)
    # Worst cell reported
    assert verdict.pf_inner_min_cell == (
        specs[0].deadband_threshold,
        specs[0].max_leverage,
    )


def test_resolve_edge_stability_excludes_non_deployable_neighbors():
    """Non-deployable cells (max_lev > cap) excluded from ratio."""
    specs = build_grid(_baseline_deployed(), _baseline_gates())
    cells = []
    for s in specs:
        if not s.deployable:
            cells.append(_make_cell_result(s, 0.1))  # very bad, but ignored
        elif abs(s.deadband_threshold - 0.25) < 1e-9 and abs(s.max_leverage - 1.0) < 1e-9:
            cells.append(_make_cell_result(s, 2.0))  # center
        else:
            cells.append(_make_cell_result(s, 1.5))  # deployable neighbor
    verdict = resolve_edge_stability(cells, _baseline_deployed(), _baseline_gates())
    assert verdict.decision == "PASS"  # 0.1 cells excluded; 1.5/2.0 = 0.75 passes
    assert verdict.n_deployable_neighbors == 5  # 6 deployable - 1 center
    assert verdict.pf_inner_min == 1.5


def test_resolve_edge_stability_excludes_halted_neighbors():
    """Halted cells (PF-XCHECK divergence > 30%) excluded."""
    specs = build_grid(_baseline_deployed(), _baseline_gates())
    cells = []
    deployed = _baseline_deployed()
    for i, s in enumerate(specs):
        is_center = (
            abs(s.deadband_threshold - deployed["deadband_threshold"]) < 1e-9
            and abs(s.max_leverage - deployed["max_leverage"]) < 1e-9
        )
        pf = 2.0 if is_center else 1.5
        # Halt one deployable neighbor with terrible PF=0
        halt = (i == 0)
        cells.append(_make_cell_result(s, pf if not halt else 0.0, halt=halt))
    verdict = resolve_edge_stability(cells, deployed, _baseline_gates())
    assert verdict.decision == "PASS"
    assert verdict.n_halted_cells == 1


def test_resolve_edge_stability_unknown_when_center_halted():
    specs = build_grid(_baseline_deployed(), _baseline_gates())
    cells = []
    deployed = _baseline_deployed()
    for s in specs:
        is_center = (
            abs(s.deadband_threshold - deployed["deadband_threshold"]) < 1e-9
            and abs(s.max_leverage - deployed["max_leverage"]) < 1e-9
        )
        cells.append(_make_cell_result(s, 2.0 if is_center else 1.5, halt=is_center))
    verdict = resolve_edge_stability(cells, deployed, _baseline_gates())
    assert verdict.decision == "UNKNOWN_INSUFFICIENT_NEIGHBORS"


def test_resolve_edge_stability_unknown_when_too_few_neighbors():
    """Default min_neighbors=4; force only 3 eligible via halts."""
    specs = build_grid(_baseline_deployed(), _baseline_gates())
    cells = []
    deployed = _baseline_deployed()
    halt_count = 0
    for i, s in enumerate(specs):
        is_center = (
            abs(s.deadband_threshold - deployed["deadband_threshold"]) < 1e-9
            and abs(s.max_leverage - deployed["max_leverage"]) < 1e-9
        )
        if is_center:
            cells.append(_make_cell_result(s, 2.0))
            continue
        # Halt 3 deployable neighbors; 3 non-deployable already excluded
        if s.deployable and halt_count < 3:
            cells.append(_make_cell_result(s, 1.5, halt=True))
            halt_count += 1
        else:
            cells.append(_make_cell_result(s, 1.5))
    verdict = resolve_edge_stability(cells, deployed, _baseline_gates())
    # 6 deployable - 1 center - 3 halted = 2 eligible < 4 min → UNKNOWN
    assert verdict.decision == "UNKNOWN_INSUFFICIENT_NEIGHBORS"
    assert verdict.n_deployable_neighbors == 2


def test_resolve_edge_stability_unknown_when_pf_center_zero():
    """pf_center near zero → ratio undefined → UNKNOWN."""
    specs = build_grid(_baseline_deployed(), _baseline_gates())
    cells = _make_homogeneous_cells(specs, center_pf=0.0, neighbor_pf=1.5)
    verdict = resolve_edge_stability(cells, _baseline_deployed(), _baseline_gates())
    assert verdict.decision == "UNKNOWN_INSUFFICIENT_NEIGHBORS"


def test_resolve_edge_stability_unknown_when_pf_center_below_one():
    """ADR-5 (N2 audit fix): a center re-roll below breakeven (0 < pf_center < 1.0)
    is an anomaly → UNKNOWN, NOT a misleading PASS from a ratio of two losers."""
    specs = build_grid(_baseline_deployed(), _baseline_gates())
    # ratio = 0.55 / 0.60 = 0.917 would be a PASS under the old < 1e-9 guard.
    cells = _make_homogeneous_cells(specs, center_pf=0.60, neighbor_pf=0.55)
    verdict = resolve_edge_stability(cells, _baseline_deployed(), _baseline_gates())
    assert verdict.decision == "UNKNOWN_INSUFFICIENT_NEIGHBORS"
    assert not verdict.pass_


def test_resolve_edge_stability_pf_center_one_is_graded_not_unknown():
    """Boundary: pf_center == 1.0 (breakeven) is graded normally (ADR-5 is strict <)."""
    specs = build_grid(_baseline_deployed(), _baseline_gates())
    cells = _make_homogeneous_cells(specs, center_pf=1.0, neighbor_pf=0.9)
    verdict = resolve_edge_stability(cells, _baseline_deployed(), _baseline_gates())
    assert verdict.decision in ("PASS", "FAIL")  # ratio 0.9 ≥ 0.70 → PASS
    assert verdict.pf_ratio == pytest.approx(0.9, abs=1e-9)


def test_resolve_edge_stability_counts_pf_xcheck_skipped():
    """N2 audit fix: SKIPPED PF-XCHECK cells are counted and surfaced so a
    close-marked single-curve PF is never silently read as cross-checked."""
    specs = build_grid(_baseline_deployed(), _baseline_gates())
    cells = _make_homogeneous_cells(specs, center_pf=2.0, neighbor_pf=1.5)
    verdict = resolve_edge_stability(cells, _baseline_deployed(), _baseline_gates())
    # _make_cell_result builds every cell with PfXCheckResult(status="SKIPPED").
    assert verdict.n_pf_xcheck_skipped == 9
    assert verdict.decision == "PASS"  # SKIPPED is surfaced, not blocking


def test_resolve_edge_stability_respects_floor_override():
    """Stricter floor 0.85 → ratio 0.75 from previous baseline now FAILs."""
    specs = build_grid(_baseline_deployed(), _baseline_gates())
    cells = _make_homogeneous_cells(specs, center_pf=2.0, neighbor_pf=1.5)
    gates = _baseline_gates(edge_stability_pf_ratio_floor=0.85)
    verdict = resolve_edge_stability(cells, _baseline_deployed(), gates)
    assert verdict.decision == "FAIL"
    assert verdict.pf_ratio_floor == 0.85


# ---------------------------------------------------------------------------
# _assert_invariants (run_cell preflight)
# ---------------------------------------------------------------------------


def test_assert_invariants_bug03_hindsight_nonzero_raises():
    cfg = {"env": {"hindsight_weight": 0.5}}
    with pytest.raises(InvariantViolation, match="BUG-03"):
        _assert_invariants(cfg)


def test_assert_invariants_bug03_hindsight_zero_passes():
    cfg = {"env": {"hindsight_weight": 0.0}}
    _assert_invariants(cfg)  # no raise


def test_assert_invariants_leak1_norm_cutoff_overlap_raises():
    cfg = {
        "data": {
            "val_end_date": "2025-08-01",
            "test_start_date": "2025-07-01",  # overlap!
        }
    }
    with pytest.raises(InvariantViolation, match="LEAK-1"):
        _assert_invariants(cfg)


def test_assert_invariants_leak1_clean_split_passes():
    cfg = {
        "data": {
            "val_end_date": "2025-06-30",
            "test_start_date": "2025-07-01",
        }
    }
    _assert_invariants(cfg)


# ---------------------------------------------------------------------------
# Verdict block + writer
# ---------------------------------------------------------------------------


def test_build_sensitivity_audit_block_shape():
    specs = build_grid(_baseline_deployed(), _baseline_gates())
    cells = _make_homogeneous_cells(specs, center_pf=2.0, neighbor_pf=1.5)
    verdict = resolve_edge_stability(cells, _baseline_deployed(), _baseline_gates())
    block = build_sensitivity_audit_block(
        cells=cells,
        edge_stability=verdict,
        config_axes={
            "deadband_threshold": [0.20, 0.25, 0.30],
            "max_leverage_mults": [0.5, 1.0, 1.5],
            "max_leverage_cap": 1.0,
            "max_leverage_values": [0.5, 1.0, 1.5],
        },
        rule="ens_pf_weighted",
        n_seeds_per_cell=3,
        wall_time_seconds=1247.0,
        wandb_run_id="abc123",
        git_sha="deadbeef",
    )
    assert block["schema"] == "1.0"
    assert block["rule"] == "ens_pf_weighted"
    assert block["n_seeds_per_cell"] == 3
    assert len(block["cells"]) == 9
    assert block["edge_stability"]["decision"] == "PASS"
    assert block["edge_stability"]["pf_ratio"] == pytest.approx(0.75)
    assert block["thresholds_used"]["edge_stability_pf_ratio_floor"] == 0.70
    assert block["wall_time_seconds"] == 1247.0
    assert block["wandb_run_id"] == "abc123"
    assert block["git_sha"] == "deadbeef"


def test_build_sensitivity_audit_block_includes_effective_deadband_per_cell():
    """ADR-6 transparency: each cell records its effective_deadband."""
    specs = build_grid(_baseline_deployed(), _baseline_gates())
    cells = _make_homogeneous_cells(specs, center_pf=2.0, neighbor_pf=1.5)
    verdict = resolve_edge_stability(cells, _baseline_deployed(), _baseline_gates())
    block = build_sensitivity_audit_block(
        cells=cells,
        edge_stability=verdict,
        config_axes={},
        rule="solo_42",
        n_seeds_per_cell=1,
    )
    for cd in block["cells"]:
        expected = cd["deadband_threshold"] * cd["max_leverage"]
        assert cd["effective_deadband"] == pytest.approx(expected)


def test_write_verdict_v26_atomic_and_back_compat(tmp_path):
    """schema_version bumped to 2.6; v2.5 fields preserved; original untouched."""
    prior = {
        "schema_version": "2.5",
        "chosen_rule": "ens_pf_weighted",
        "decision": "PROMOTE",
        "seeds": [456, 1024, 2025],
        "test_solo_pfs": {"456": 2.55, "1024": 2.47, "2025": 2.50},
    }
    prior_path = tmp_path / "verdict.json"
    prior_path.write_text(json.dumps(prior), encoding="utf-8")

    specs = build_grid(_baseline_deployed(), _baseline_gates())
    cells = _make_homogeneous_cells(specs, center_pf=2.0, neighbor_pf=1.5)
    verdict = resolve_edge_stability(cells, _baseline_deployed(), _baseline_gates())
    block = build_sensitivity_audit_block(
        cells=cells,
        edge_stability=verdict,
        config_axes={"deadband_threshold": [0.20, 0.25, 0.30]},
        rule="ens_pf_weighted",
        n_seeds_per_cell=3,
    )

    out_path = tmp_path / "verdict_v2_6.json"
    new = write_verdict_v26(prior_path, block, _baseline_deployed(), out_path)

    assert new["schema_version"] == SCHEMA_VERSION
    assert new["chosen_rule"] == "ens_pf_weighted"  # preserved
    assert new["decision"] == "PROMOTE"
    assert new["deployed_config"]["deadband_threshold"] == 0.25
    assert new["sensitivity_audit"]["edge_stability"]["decision"] == "PASS"

    # Original untouched
    prior_now = json.loads(prior_path.read_text(encoding="utf-8"))
    assert prior_now["schema_version"] == "2.5"
    assert "sensitivity_audit" not in prior_now

    # New file is valid JSON on disk
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == "2.6"

    # No stray .tmp file (atomic rename completed)
    assert not (tmp_path / "verdict_v2_6.json.tmp").exists()
