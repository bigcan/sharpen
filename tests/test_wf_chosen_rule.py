"""Audit N3: the WF gate must grade the Stage-2.5 *chosen_rule*, not the static
`gates.aggregation_rule`.

The DECAY-01 gates YAML pinned `aggregation_rule: ens_agreement` while the
Stage-2.5 val_argmax_pf verdict chose `ens_mean` — so `_evaluate_gates` graded
the wrong aggregation. These tests lock the override (and the unchanged
fallback) into `_evaluate_gates`.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from sg1_btc_velotrade_ensemble_eval import _evaluate_gates  # noqa: E402  (sibling script)

SEEDS = [456, 42, 2025]


def _gates_cfg():
    return {
        "aggregation_rule": "ens_agreement",  # the pinned/static rule
        "wf_folds": 2,
        "decision": {"all_pass_action": "approve_ensemble", "any_fail_action": "use_best_solo"},
        "gates": {
            "g1_solo_baseline": {"pf_floor": 1.1, "seeds_pass_min": 2, "min_folds_pass": 2},
            "g2_no_regression": {"ens_ratio_min": 0.9, "min_folds_pass": 2},
            "g3_uplift": {"uplift_ratio_min": 1.10},
            "g4_velotrade_compliance": {
                "trailing_dd_buffer_pp_min": 0.5, "min_folds_pass": 2,
                "daily_dd_buffer_pp_min": None, "compliance_must_pass": False,
            },
            "g5_dd_stability": {"cv_pf_max": 0.35},
        },
    }


def _fold(ens_agreement_pf, ens_mean_pf, solo_pf=1.5, buf=1.0):
    fm = {f"solo_{s}": {"pf_bar": solo_pf, "trailing_dd_buffer_pp": buf} for s in SEEDS}
    fm["ens_agreement"] = {"pf_bar": ens_agreement_pf, "trailing_dd_buffer_pp": buf}
    fm["ens_mean"] = {"pf_bar": ens_mean_pf, "trailing_dd_buffer_pp": buf}
    return fm


# ens_agreement is a regression vs best solo (1.0 < 0.9*1.5); ens_mean clears
# the 1.10 uplift (2.0 / 1.5 = 1.33). Which one is graded flips the verdict.
PER_FOLD = [_fold(1.0, 2.0), _fold(1.0, 2.0)]
STATUS = ["ok", "ok"]


def test_default_grades_static_gates_rule():
    v = _evaluate_gates(PER_FOLD, STATUS, _gates_cfg(), SEEDS)  # chosen_rule=None
    assert v["graded_rule"] == "ens_agreement"
    assert v["gates"]["G2_no_regression"]["pass"] is False
    assert v["gates"]["G3_uplift"]["pass"] is False
    assert v["overall"] == "FAIL"
    assert "warnings" not in v  # no override -> no N3 warning


def test_chosen_rule_overrides_and_flips_verdict():
    v = _evaluate_gates(PER_FOLD, STATUS, _gates_cfg(), SEEDS, chosen_rule="ens_mean")
    assert v["graded_rule"] == "ens_mean"
    assert v["gates_aggregation_rule"] == "ens_agreement"
    assert v["gates"]["G2_no_regression"]["pass"] is True
    assert v["gates"]["G3_uplift"]["pass"] is True
    assert v["overall"] == "PASS"
    assert "warnings" in v and any("aggregation_rule" in w for w in v["warnings"])


def test_chosen_equal_to_gates_rule_no_warning():
    v = _evaluate_gates(PER_FOLD, STATUS, _gates_cfg(), SEEDS, chosen_rule="ens_agreement")
    assert v["graded_rule"] == "ens_agreement"
    assert "warnings" not in v  # chosen == gates rule -> nothing to flag
