"""Step-5 tests for the RL execution-overlay runner + configs + validate_config hook.

Covers (data/training-free): the ``check_execution_overlay_gates`` presence/sanity gate and
the ``base_book`` merge in ``validate_config``; the runner's pure gate logic
(``partition_fold_uplifts`` + ``build_execution_overlay_verdict``); ``load_overlay_config``
inheritance and ``effective_gates`` two-files merge; and that the shipped
``execution_overlay_sac.yaml`` passes ``validate --stage wf``.

The full load -> train -> evaluate -> gate smoke is the CLI acceptance check
(``execution_overlay_runner.py --smoke``); it needs network/data so it is not run here.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from scripts.execution_overlay_runner import (  # noqa: E402
    build_execution_overlay_verdict,
    effective_gates,
    load_overlay_config,
    partition_fold_uplifts,
    temporal_split_bundle,
)
from scripts.validate_config import (  # noqa: E402
    ValidationResult,
    check_execution_overlay_gates,
    validate,
)

OVERLAY_CFG = ROOT / "configs" / "execution_overlay_sac.yaml"
OVERLAY_GATES = ROOT / "configs" / "execution_overlay.gates.yaml"

# Canonical, well-formed gate block (mirrors the shipped gates file).
GOOD_GATES = {
    "execution_beats_baseline": {
        "min_uplift_bps": 2.0, "cost_stress_factor": 1.5, "cost_stress_min_uplift_bps": 0.5,
        "wf_folds": 4, "robust_folds_required": 3, "else": "ship_snap_executor",
    },
    "overlay_parity": {"max_completion_l1_drift": 0.01, "max_intra_horizon_drift": 1.0},
}


def _eval(uplift, *, comp=0.0, intra=0.5, cost_stress=1.0):
    """A synthetic evaluate_execution_overlay result with the keys the verdict reads."""
    return {
        "is_uplift_bps": uplift, "completion_l1_max": comp, "intra_horizon_drift_max": intra,
        "net_is_bps_overlay": 10.0, "net_is_bps_baseline": 10.0 + uplift,
        "turnover_overlay": 1.0, "turnover_baseline": 1.0, "n_episodes": 12,
        "cost_stress": cost_stress,
    }


# --------------------------------------------------------------------------- #
# validate_config hook: presence + sanity, env-type-gated
# --------------------------------------------------------------------------- #
def test_shipped_overlay_config_validates_at_wf():
    assert validate(OVERLAY_CFG, "wf").status == "PASS"


def test_shipped_overlay_config_validates_at_oos():
    # `oos` runs only universal checks (incl. the overlay-gate hook) — base_book merge must
    # still surface env/data so the universal checks pass.
    assert validate(OVERLAY_CFG, "oos").status == "PASS"


def test_hook_is_noop_without_execution_overlay_block():
    r = ValidationResult()
    check_execution_overlay_gates({"gates": {}}, r)  # no execution_overlay key
    assert r.failures == [] and r.passed == []


def test_hook_rejects_missing_gate_block():
    r = ValidationResult()
    check_execution_overlay_gates({"execution_overlay": {"horizon_bars": 5}, "gates": {}}, r)
    assert any("execution_beats_baseline" in f for f in r.failures)


def test_hook_rejects_cost_stress_factor_not_above_one():
    bad = {**GOOD_GATES, "execution_beats_baseline":
           {**GOOD_GATES["execution_beats_baseline"], "cost_stress_factor": 1.0}}
    r = ValidationResult()
    check_execution_overlay_gates({"execution_overlay": {}, "gates": bad}, r)
    assert any("cost_stress_factor" in f for f in r.failures)


def test_hook_rejects_robust_folds_exceeding_wf_folds():
    bad = {**GOOD_GATES, "execution_beats_baseline":
           {**GOOD_GATES["execution_beats_baseline"], "robust_folds_required": 5, "wf_folds": 4}}
    r = ValidationResult()
    check_execution_overlay_gates({"execution_overlay": {}, "gates": bad}, r)
    assert any("robust_folds_required" in f for f in r.failures)


def test_hook_rejects_stress_floor_above_primary():
    bad = {**GOOD_GATES, "execution_beats_baseline":
           {**GOOD_GATES["execution_beats_baseline"], "cost_stress_min_uplift_bps": 5.0}}
    r = ValidationResult()
    check_execution_overlay_gates({"execution_overlay": {}, "gates": bad}, r)
    assert any("cost_stress_min_uplift_bps" in f for f in r.failures)


def test_hook_rejects_wrong_else_action():
    bad = {**GOOD_GATES, "execution_beats_baseline":
           {**GOOD_GATES["execution_beats_baseline"], "else": "deploy_anyway"}}
    r = ValidationResult()
    check_execution_overlay_gates({"execution_overlay": {}, "gates": bad}, r)
    assert any(".else" in f for f in r.failures)


def test_hook_accepts_good_gates():
    r = ValidationResult()
    check_execution_overlay_gates({"execution_overlay": {}, "gates": GOOD_GATES}, r)
    assert r.failures == []


# --------------------------------------------------------------------------- #
# Config inheritance + effective gates
# --------------------------------------------------------------------------- #
def test_load_overlay_config_merges_base_book():
    cfg = load_overlay_config(OVERLAY_CFG)
    # From the overlay file:
    assert cfg["execution_overlay"]["horizon_bars"] == 5
    assert cfg["prop_firm"]["augment_obs"] is True
    # Inherited from base_book (configs/live_cross_asset_paper.yaml):
    assert cfg["env"]["type"] == "multi_asset_allocator"
    assert cfg["universe"]["n_assets"] == 19
    assert cfg["data"]["source"] == "yfinance_etf"


def test_effective_gates_merges_standalone_over_inline():
    cfg = {"gates": {"execution_beats_baseline": {"min_uplift_bps": 99.0}},
           "ensemble": {"gates_file": str(OVERLAY_GATES)}}
    g = effective_gates(cfg, None)
    # Standalone file wins (its min_uplift_bps=2.0 overrides the inline 99.0).
    assert g["execution_beats_baseline"]["min_uplift_bps"] == 2.0
    assert "overlay_parity" in g


# --------------------------------------------------------------------------- #
# temporal_split_bundle — the OOS gate split (leak-free, disjoint)
# --------------------------------------------------------------------------- #
def _synthetic_bundle(T=100, U=3):
    price = np.arange(T * U, dtype=float).reshape(T, U) + 1.0
    union = {
        "price_ary": price, "volume_ary": np.ones((T, U)), "carry_ary": np.zeros((T, U)),
        "timestamps": np.arange(T, dtype=np.int64) * 86400, "assets": [f"A{i}" for i in range(U)],
    }
    # row-distinct target so value-preservation is checkable; (T-1, U).
    combined_w = (np.arange(T - 1)[:, None] * np.ones((1, U))) * 0.001
    return {"union": union, "extra": "kept"}, combined_w


def test_temporal_split_shapes_pair_with_env_contract():
    b, cw = _synthetic_bundle(100, 3)
    trb, trt, teb, tet, s = temporal_split_bundle(b, cw, 0.7)
    assert s == 70
    # env contract: target rows == price rows - 1, on BOTH windows.
    assert trt.shape == (s - 1, 3) == (trb["union"]["price_ary"].shape[0] - 1, 3)
    assert tet.shape == (100 - s - 1, 3) == (teb["union"]["price_ary"].shape[0] - 1, 3)


def test_temporal_split_is_disjoint_and_leak_free():
    b, cw = _synthetic_bundle(100, 3)
    trb, trt, teb, tet, s = temporal_split_bundle(b, cw, 0.7)
    # disjoint observation windows — the OOS eval can never see a training bar.
    assert set(trb["union"]["timestamps"].tolist()).isdisjoint(teb["union"]["timestamps"].tolist())
    assert teb["union"]["timestamps"][0] == b["union"]["timestamps"][s]
    # leak-free slice: the test target is exactly combined_w[s:] (causal rows preserved).
    assert np.allclose(tet, cw[s:])
    assert np.allclose(trt, cw[: s - 1])
    assert trb["extra"] == "kept"  # top-level keys preserved


def test_temporal_split_rejects_bad_frac():
    b, cw = _synthetic_bundle(50, 2)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            temporal_split_bundle(b, cw, bad)


def test_temporal_split_rejects_wrong_combined_w_rows():
    b, cw = _synthetic_bundle(50, 2)
    with pytest.raises(ValueError):
        temporal_split_bundle(b, cw[:-5], 0.7)


# --------------------------------------------------------------------------- #
# partition_fold_uplifts
# --------------------------------------------------------------------------- #
def _episodes(steps, net_is):
    return [{"rebalance_step": s, "net_is_bps": v} for s, v in zip(steps, net_is)]


def test_partition_fold_uplifts_pairs_by_step_and_folds():
    # baseline cost per step; overlay is cheaper by 1.0 everywhere -> every fold uplift = +1.0.
    base = _episodes([10, 20, 30, 40], [5.0, 5.0, 5.0, 5.0])
    over = _episodes([40, 30, 20, 10], [4.0, 4.0, 4.0, 4.0])  # deliberately reordered
    folds = partition_fold_uplifts(over, base, 4)
    assert folds == [1.0, 1.0, 1.0, 1.0]


def test_partition_fold_uplifts_uneven_split():
    base = _episodes([1, 2, 3, 4, 5], [5.0] * 5)
    over = _episodes([1, 2, 3, 4, 5], [4.0, 4.0, 4.0, 6.0, 6.0])  # last two are WORSE
    folds = partition_fold_uplifts(over, base, 2)  # [1,2] then [3,4,5]
    assert folds[0] == 1.0
    # fold 2 mean = mean(+1, -1, -1) = -1/3
    assert abs(folds[1] - (-1.0 / 3.0)) < 1e-9


def test_partition_fold_uplifts_empty_gives_nans():
    folds = partition_fold_uplifts([], [], 3)
    assert len(folds) == 3 and all(u != u for u in folds)  # nan != nan


# --------------------------------------------------------------------------- #
# build_execution_overlay_verdict
# --------------------------------------------------------------------------- #
def test_verdict_pass_deploys_overlay():
    v = build_execution_overlay_verdict(
        _eval(3.0, comp=0.0, intra=0.4), _eval(1.0, cost_stress=1.5),
        [1.0, 1.0, 1.0, 1.0], GOOD_GATES)
    assert v["overall_status"] == "PASS"
    assert v["decision"] == "deploy_overlay"
    assert v["gates"]["execution_beats_baseline"]["status"] == "PASS"
    assert v["gates"]["overlay_parity"]["status"] == "PASS"


def test_verdict_fail_primary_uplift_ships_snap():
    v = build_execution_overlay_verdict(
        _eval(1.0), _eval(1.0, cost_stress=1.5), [1.0, 1.0, 1.0, 1.0], GOOD_GATES)
    assert v["overall_status"] == "FAIL"
    assert v["decision"] == "ship_snap_executor"
    assert v["gates"]["execution_beats_baseline"]["checks"]["primary_uplift"] is False


def test_verdict_fail_stress_uplift():
    v = build_execution_overlay_verdict(
        _eval(3.0), _eval(0.1, cost_stress=1.5), [1.0, 1.0, 1.0, 1.0], GOOD_GATES)
    assert v["overall_status"] == "FAIL"
    assert v["gates"]["execution_beats_baseline"]["checks"]["stress_uplift"] is False


def test_verdict_fail_fold_robustness():
    # Only 2 of 4 folds positive but robust_folds_required=3.
    v = build_execution_overlay_verdict(
        _eval(3.0), _eval(1.0, cost_stress=1.5), [1.0, 1.0, -0.5, -0.5], GOOD_GATES)
    assert v["overall_status"] == "FAIL"
    assert v["gates"]["execution_beats_baseline"]["robust_folds_positive"] == 2
    assert v["gates"]["execution_beats_baseline"]["checks"]["fold_robustness"] is False


def test_verdict_completion_breach_fails_even_with_uplift():
    # Strong uplift but the overlay didn't reach the target at horizon end -> hard FAIL.
    v = build_execution_overlay_verdict(
        _eval(5.0, comp=0.5), _eval(2.0, cost_stress=1.5), [2.0] * 4, GOOD_GATES)
    assert v["overall_status"] == "FAIL"
    assert v["gates"]["overlay_parity"]["status"] == "FAIL"


def test_verdict_intra_horizon_is_sanity_warn_not_kill():
    # Completion fine, intra-horizon lag exceeds the sanity bound -> parity WARN, overall still PASS.
    v = build_execution_overlay_verdict(
        _eval(3.0, comp=0.0, intra=5.0), _eval(1.0, cost_stress=1.5), [1.0] * 4, GOOD_GATES)
    assert v["overall_status"] == "PASS"
    assert v["gates"]["overlay_parity"]["status"] == "WARN"


def test_verdict_nan_fold_counts_as_non_positive():
    v = build_execution_overlay_verdict(
        _eval(3.0), _eval(1.0, cost_stress=1.5),
        [1.0, 1.0, 1.0, float("nan")], GOOD_GATES)
    # 3 positive folds, required 3 -> still passes the fold check.
    assert v["gates"]["execution_beats_baseline"]["robust_folds_positive"] == 3
    assert v["gates"]["execution_beats_baseline"]["checks"]["fold_robustness"] is True
