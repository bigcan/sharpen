"""Tests for the X3 fixed-lot stress pass (`finrl_pro_ds/eval/fixed_lot_stress.py`).

Covers the reconstruction guard (early-term → raise) and the multi-fold
`compute_stress_subreport` gate logic (PASS / FAIL-dd / FAIL-leverage /
INCOMPLETE / SKIPPED). Protocol v2 Stage 3 stress sub-report; audit X3/P7-03/P7-07.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finrl_pro_ds.eval.fixed_lot_stress import (  # noqa: E402
    EarlyTerminatedFold,
    compute_stress_subreport,
    reconstruct_fold,
)

INIT = 100_000.0
CAP = 10.0


def _write_traj(tmp: Path, name: str, pv, position=None, term=None) -> Path:
    """Write a minimal trajectory parquet (portfolio_value [+ position, term])."""
    pv = np.asarray(pv, dtype=np.float64)
    data = {"portfolio_value": pv}
    if position is not None:
        data["position"] = np.asarray(position, dtype=np.float64)
    if term is not None:
        data["prop_firm_termination"] = list(term)
    p = tmp / f"{name}.parquet"
    pd.DataFrame(data).to_parquet(p)
    return p


# --- reconstruct_fold ------------------------------------------------------

def test_reconstruct_fold_clean(tmp_path):
    p = _write_traj(tmp_path, "f", [100000, 101000, 102000, 101000, 103000],
                    position=[0.5, 0.6, 0.7, 0.5, 0.6])
    rec = reconstruct_fold(p, INIT, None)
    assert rec["n_bars"] == 5
    assert rec["peak_abs_position"] == pytest.approx(0.7)
    # A dip from the running peak → strictly-negative fixed trailing DD.
    assert rec["fixed_trailing_mdd_peak_pct"] < 0.0
    assert rec["pf_xcheck_ok"] and rec["ret_xcheck_ok"]


def test_reconstruct_fold_early_term_raises(tmp_path):
    p = _write_traj(tmp_path, "f", [100000, 99000, 95000],
                    position=[0.5, 0.5, 0.5],
                    term=[None, None, "trailing_dd"])
    with pytest.raises(EarlyTerminatedFold):
        reconstruct_fold(p, INIT, None)


# --- compute_stress_subreport ---------------------------------------------

def test_stress_subreport_pass(tmp_path):
    folds = [
        ("fold_00", _write_traj(tmp_path, "a", [100000, 101000, 100500, 102000],
                                 position=[0.4, 0.5, 0.6, 0.5])),
        ("fold_01", _write_traj(tmp_path, "b", [100000, 100500, 101500, 101000],
                                 position=[0.3, 0.4, 0.5, 0.45])),
    ]
    r = compute_stress_subreport(folds, gates={}, initial_balance=INIT,
                                 trailing_cap_pct=CAP, graded_rule="solo_2025")
    assert r["status"] == "PASS"
    assert r["stress_pass"] is True
    assert r["n_folds_evaluated"] == 2
    assert r["stress_dd_buffer_pp"] >= 0.5
    assert r["peak_leverage"] <= 1.0


def test_stress_subreport_fail_on_deep_dd(tmp_path):
    # 10% single-bar drop → fixed trailing DD = -10% → buffer 0.0 < 0.5 → FAIL.
    folds = [("fold_00", _write_traj(tmp_path, "a", [100000, 90000],
                                     position=[0.5, 0.5]))]
    r = compute_stress_subreport(folds, gates={}, initial_balance=INIT,
                                 trailing_cap_pct=CAP, graded_rule="solo_456")
    assert r["status"] == "FAIL"
    assert r["stress_pass"] is False
    assert r["stress_dd_pass"] is False


def test_stress_subreport_fail_on_leverage(tmp_path):
    # Shallow DD but peak |position| = 1.5 > 1.0 cap → FAIL.
    folds = [("fold_00", _write_traj(tmp_path, "a", [100000, 100500, 101000],
                                     position=[0.5, 1.5, 0.8]))]
    r = compute_stress_subreport(folds, gates={"stress_leverage_max": 1.0},
                                 initial_balance=INIT, trailing_cap_pct=CAP,
                                 graded_rule="solo_456")
    assert r["status"] == "FAIL"
    assert r["stress_leverage_pass"] is False
    assert r["peak_leverage"] == pytest.approx(1.5)


def test_stress_subreport_incomplete_on_early_term(tmp_path):
    folds = [
        ("fold_00", _write_traj(tmp_path, "ok", [100000, 101000, 102000],
                                 position=[0.5, 0.5, 0.5])),
        ("fold_01", _write_traj(tmp_path, "term", [100000, 99000, 95000],
                                 position=[0.5, 0.5, 0.5],
                                 term=[None, None, "trailing_dd"])),
    ]
    r = compute_stress_subreport(folds, gates={}, initial_balance=INIT,
                                 trailing_cap_pct=CAP, graded_rule="ens_mean")
    assert r["status"] == "INCOMPLETE_NEEDS_REROLLOUT"
    assert r["stress_pass"] is False
    assert len(r["early_terminated_folds"]) == 1
    assert r["early_terminated_folds"][0]["fold"] == "fold_01"


def test_stress_subreport_skipped_when_empty(tmp_path):
    r = compute_stress_subreport([], gates={}, initial_balance=INIT,
                                 trailing_cap_pct=CAP, graded_rule="solo_456")
    assert r["status"] == "SKIPPED"
    assert r["stress_pass"] is False


def test_stress_subreport_custom_threshold(tmp_path):
    # ~1% DD fold passes default 0.5pp but fails a strict 9.5pp buffer floor.
    folds = [("fold_00", _write_traj(tmp_path, "a", [100000, 101000, 99990, 101000],
                                     position=[0.5, 0.5, 0.5, 0.5]))]
    lax = compute_stress_subreport(folds, gates={"stress_dd_buffer_pp": 0.5},
                                   initial_balance=INIT, trailing_cap_pct=CAP,
                                   graded_rule="solo_456")
    strict = compute_stress_subreport(folds, gates={"stress_dd_buffer_pp": 9.5},
                                      initial_balance=INIT, trailing_cap_pct=CAP,
                                      graded_rule="solo_456")
    assert lax["status"] == "PASS"
    assert strict["status"] == "FAIL"
