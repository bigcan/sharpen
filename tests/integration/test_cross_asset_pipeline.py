"""Tests for the cross-asset pipeline CPU logic (Phase 4, S553-cont-34):
walk-forward scheduler (embargo LEAK gaps), gate-threshold loading, and the
RL-beats-linear manifest decision. No GPU/SB3 (those are lazy-imported)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import scripts.cross_asset_pipeline as pipe

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Walk-forward scheduler
# --------------------------------------------------------------------------- #
_WF = {"train_bars": 100, "val_bars": 30, "test_bars": 30, "step_bars": 30, "embargo_bars": 5}


def _dates(n):
    return pd.bdate_range("2010-01-01", periods=n)


def test_schedule_window_count_and_no_tail_overrun():
    dates = _dates(400)
    sched = pipe.build_wf_schedule(dates, _WF)
    assert len(sched) >= 1
    # No window's test_end exceeds the data.
    assert all(w["test_end"] <= dates[-1] for w in sched)


def test_schedule_embargo_gaps_present():
    """Embargo bars must separate train→val and val→test (LEAK guard)."""
    dates = _dates(400)
    sched = pipe.build_wf_schedule(dates, _WF)
    emb = _WF["embargo_bars"]
    for w in sched:
        i = {k: dates.get_loc(w[k]) for k in
             ("train_end", "val_start", "val_end", "test_start")}
        assert i["val_start"] - i["train_end"] == emb + 1
        assert i["test_start"] - i["val_end"] == emb + 1


def test_schedule_blocks_are_contiguous_and_correct_length():
    dates = _dates(400)
    sched = pipe.build_wf_schedule(dates, _WF)
    for w in sched:
        i = {k: dates.get_loc(w[k]) for k in
             ("train_start", "train_end", "val_start", "val_end", "test_start", "test_end")}
        assert i["train_end"] - i["train_start"] + 1 == _WF["train_bars"]
        assert i["val_end"] - i["val_start"] + 1 == _WF["val_bars"]
        assert i["test_end"] - i["test_start"] + 1 == _WF["test_bars"]


def test_schedule_steps_forward():
    dates = _dates(400)
    sched = pipe.build_wf_schedule(dates, _WF)
    starts = [dates.get_loc(w["train_start"]) for w in sched]
    assert starts == sorted(starts)
    if len(starts) > 1:
        assert starts[1] - starts[0] == _WF["step_bars"]


def test_schedule_empty_when_insufficient_data():
    assert pipe.build_wf_schedule(_dates(50), _WF) == []


# --------------------------------------------------------------------------- #
# Gate thresholds
# --------------------------------------------------------------------------- #
def test_load_gate_thresholds_from_overlay():
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "cross_asset_momentum.yaml").read_text(encoding="utf-8"))
    g = pipe.load_gate_thresholds(cfg)
    assert g["min_uplift"] == 0.10
    assert g["wf_net_sharpe_floor"] == 0.40
    assert g["max_cost_gap"] == 0.15


def test_load_gate_thresholds_defaults_without_file():
    g = pipe.load_gate_thresholds({})
    assert g["min_uplift"] == 0.10 and g["wf_net_sharpe_floor"] == 0.40


# --------------------------------------------------------------------------- #
# Manifest gate decision
# --------------------------------------------------------------------------- #
_GATE = {"min_uplift": 0.10, "wf_net_sharpe_floor": 0.40, "max_cost_gap": 0.15}


def _result(uplift, rl, core, gate_pass):
    return {"window": 0, "uplift_net_sharpe": uplift,
            "rl_test": {"net_sharpe": rl}, "linear_core_test": {"net_sharpe": core},
            "gate_pass": gate_pass}


def test_manifest_rl_beats_linear(tmp_path):
    results = [_result(0.20, 0.55, 0.35, True), _result(0.30, 0.60, 0.30, True)]
    m = pipe._write_manifest(tmp_path, "wf", {}, _GATE, results, status="PASS")
    assert m["gate_decision"] == "rl_beats_linear"
    assert m["n_windows_gate_pass"] == 2
    assert m["status"] == "PASS"


def test_manifest_ship_linear_when_uplift_too_small(tmp_path):
    results = [_result(0.02, 0.55, 0.53, False), _result(0.05, 0.60, 0.55, False)]
    m = pipe._write_manifest(tmp_path, "wf", {}, _GATE, results, status="PASS")
    assert m["gate_decision"] == "ship_linear_core"


def test_manifest_ship_linear_when_rl_below_floor(tmp_path):
    # Big uplift but RL Sharpe below the survival floor → still ship linear.
    results = [_result(0.20, 0.32, 0.12, False), _result(0.20, 0.35, 0.15, False)]
    m = pipe._write_manifest(tmp_path, "wf", {}, _GATE, results, status="PASS")
    assert m["gate_decision"] == "ship_linear_core"


def test_manifest_written_to_disk(tmp_path):
    results = [_result(0.20, 0.55, 0.35, True)]
    pipe._write_manifest(tmp_path, "hpo", {}, _GATE, results, status="PASS")
    assert (tmp_path / "manifest_hpo.json").exists()


def test_manifest_surfaces_cost_gap_and_deferred_diversification(tmp_path):
    """g_cost_gap is wired (median surfaced); g_diversification is explicitly
    deferred, never silently dropped."""
    r0 = {**_result(0.20, 0.55, 0.35, True), "cost_gap": 0.05}
    r1 = {**_result(0.30, 0.60, 0.30, True), "cost_gap": 0.11}
    m = pipe._write_manifest(tmp_path, "wf", {}, _GATE, [r0, r1], status="PASS")
    assert m["median_cost_gap"] == 0.08
    assert "DEFERRED" in m["diversification_gate"]


def test_manifest_ship_linear_when_cost_fragile(tmp_path):
    """Beats core on uplift + Sharpe, but median cost_gap > max_cost_gap (cost-fragile,
    the AlphaSeek failure mode) -> must NOT crown rl_beats_linear."""
    r0 = {**_result(0.20, 0.55, 0.35, False), "cost_gap": 0.50}
    r1 = {**_result(0.30, 0.60, 0.30, False), "cost_gap": 0.40}
    m = pipe._write_manifest(tmp_path, "wf", {}, _GATE, [r0, r1], status="PASS")
    assert m["gate_decision"] == "ship_linear_core"
    assert m["median_cost_gap"] == 0.45


def test_manifest_rl_beats_linear_with_costgap_in_band(tmp_path):
    r0 = {**_result(0.20, 0.55, 0.35, True), "cost_gap": 0.05}
    r1 = {**_result(0.30, 0.60, 0.30, True), "cost_gap": 0.10}
    m = pipe._write_manifest(tmp_path, "wf", {}, _GATE, [r0, r1], status="PASS")
    assert m["gate_decision"] == "rl_beats_linear"


def test_manifest_median_cost_gap_none_without_field(tmp_path):
    results = [_result(0.20, 0.55, 0.35, True)]
    m = pipe._write_manifest(tmp_path, "wf", {}, _GATE, results, status="PASS")
    assert m["median_cost_gap"] is None


# --------------------------------------------------------------------------- #
# Parallel window→GPU assignment (the WF parallelism dispatch)
# --------------------------------------------------------------------------- #
def _sched(n):
    return [{"window": i} for i in range(n)]


def test_assign_windows_round_robin_balanced():
    buckets = pipe._assign_windows_to_gpus(_sched(14), [0, 1])
    got0 = [w["window"] for w in buckets[0]]
    got1 = [w["window"] for w in buckets[1]]
    assert got0 == [0, 2, 4, 6, 8, 10, 12]
    assert got1 == [1, 3, 5, 7, 9, 11, 13]
    # every window assigned exactly once, balanced ±1
    assert sorted(got0 + got1) == list(range(14))
    assert abs(len(got0) - len(got1)) <= 1


def test_assign_windows_three_gpus():
    buckets = pipe._assign_windows_to_gpus(_sched(14), [0, 1, 2])
    allw = sorted(w["window"] for ws in buckets.values() for w in ws)
    assert allw == list(range(14))
    assert max(len(ws) for ws in buckets.values()) - min(len(ws) for ws in buckets.values()) <= 1


def test_assign_windows_single_gpu_gets_all():
    buckets = pipe._assign_windows_to_gpus(_sched(5), [0])
    assert [w["window"] for w in buckets[0]] == [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------- #
# v1.1 cost-lever HPO search space (config-gated)
# --------------------------------------------------------------------------- #
class _StubTrial:
    """Records suggest_* calls; returns lo / first choice deterministically."""

    def __init__(self):
        self.params = {}

    def suggest_float(self, name, lo, hi, log=False):
        self.params[name] = float(lo)
        return float(lo)

    def suggest_int(self, name, lo, hi, log=False):
        self.params[name] = int(lo)
        return int(lo)

    def suggest_categorical(self, name, choices):
        self.params[name] = choices[0]
        return choices[0]


def test_search_space_without_lever_bounds_is_unchanged():
    t = _StubTrial()
    out = pipe._sac_search_space(t, {"hpo": {"search": {}}})
    assert set(out["env_overrides"]) == {"turnover_penalty"}


def test_search_space_with_lever_bounds_suggests_them():
    cfg = {"hpo": {"search": {
        "no_trade_band": [0.01, 0.10],
        "rebalance_interval": [1, 5, 21],
        "cost_penalty_scale": [0.5, 5.0],
    }}}
    t = _StubTrial()
    out = pipe._sac_search_space(t, cfg)
    ov = out["env_overrides"]
    assert ov["no_trade_band"] == 0.01
    assert ov["rebalance_interval"] == 1
    assert ov["cost_penalty_scale"] == 0.5
    assert "turnover_penalty" in ov
