"""N3: --parallel actually runs cells across processes and matches sequential.

Pre-N3 the --parallel flag only logged a warning and always ran sequentially.
These tests exercise the real ProcessPool path (via the CPU-trivial dry-run
cell) so it is provably executed, and pin the picklability seam: the worker
rebuilds the non-picklable aggregation closure from ``rule_name`` rather than
shipping it across the boundary.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from finrl_pro_ds.eval.sensitivity_audit import build_grid  # noqa: E402
from scripts import stage_2_5_r_sensitivity_audit as mod  # noqa: E402

DEPLOYED = {"deadband_threshold": 0.25, "max_leverage": 1.0}
GATES = {
    "sensitivity_deadband_grid": [0.20, 0.25, 0.30],
    "sensitivity_max_leverage_mults": [0.5, 1.0, 1.5],
    "sensitivity_deployable_max_leverage_cap": 1.0,
}


def _grid():
    return build_grid(DEPLOYED, GATES)


def test_dry_run_parallel_matches_sequential():
    """Real ProcessPool over the 9 dry-run cells returns the same order-preserving
    PF/MDD surface as the in-process sequential runner."""
    cells = _grid()
    base = {"dry_run": True, "deployed_config": DEPLOYED}
    seq = mod._run_cells_sequential(cells, payload_base=base)
    par = mod._run_cells_parallel(cells, payload_base=base, max_workers=3)
    assert len(par) == len(seq) == len(cells)
    assert [round(c.pf_test, 9) for c in par] == [round(c.pf_test, 9) for c in seq]
    assert [round(c.mdd_test, 9) for c in par] == [round(c.mdd_test, 9) for c in seq]
    # order preserved: cell i in both lists is the same (deadband, leverage) cell
    assert [(c.spec.deadband_threshold, c.spec.max_leverage) for c in par] == \
           [(c.spec.deadband_threshold, c.spec.max_leverage) for c in seq]


def test_parallel_single_worker_equals_sequential():
    """max_workers=1 still goes through the pool and matches sequential."""
    cells = _grid()
    base = {"dry_run": True, "deployed_config": DEPLOYED}
    seq = mod._run_cells_sequential(cells, payload_base=base)
    par = mod._run_cells_parallel(cells, payload_base=base, max_workers=1)
    assert [round(c.pf_test, 9) for c in par] == [round(c.pf_test, 9) for c in seq]


def test_cell_worker_dry_run_single():
    cells = _grid()
    payload = {"dry_run": True, "deployed_config": DEPLOYED, "spec": cells[0]}
    res = mod._cell_worker(payload)
    assert res.spec.label() == cells[0].label()
    assert res.pf_xcheck.status == "SKIPPED"


def test_resolve_rule_fn_unsupported_raises():
    with pytest.raises(ValueError):
        mod._resolve_rule_fn("not_a_rule", {})


def test_resolve_rule_fn_known_rules_return_callables():
    for name in ("ens_mean", "ens_median", "ens_agreement"):
        assert callable(mod._resolve_rule_fn(name, {}))
    assert callable(mod._resolve_rule_fn("solo_456", {}))
