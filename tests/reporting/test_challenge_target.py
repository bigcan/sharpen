"""Tests for finrl_pro_ds.reporting.challenge_target."""
from __future__ import annotations

import math

import numpy as np
import pytest

from finrl_pro_ds.reporting import (
    DEFAULT_PHASE_SPECS,
    compute_challenge_target_hit_rate,
    compute_challenge_target_hit_rates,
    parse_phase_spec_arg,
)


def _ramp_pv(start: float, end: float, n: int) -> np.ndarray:
    return np.linspace(start, end, n, dtype=np.float64)


def test_full_trajectory_hit():
    pv = _ramp_pv(100_000.0, 112_000.0, 100)  # +12% by end
    out = compute_challenge_target_hit_rate(pv, target_pct=0.10)
    assert out["n_windows"] == 1
    assert out["n_hits"] == 1
    assert out["hit_rate"] == 1.0
    assert out["bars_to_target_per_window"][0] is not None
    assert out["bars_to_target_median"] == out["bars_to_target_per_window"][0]
    assert out["max_cum_return"] == pytest.approx(0.12, rel=1e-6)


def test_full_trajectory_miss():
    pv = _ramp_pv(100_000.0, 108_000.0, 100)  # only +8%
    out = compute_challenge_target_hit_rate(pv, target_pct=0.10)
    assert out["n_hits"] == 0
    assert out["hit_rate"] == 0.0
    assert out["bars_to_target_per_window"] == [None]
    assert out["bars_to_target_median"] is None
    assert out["max_cum_return"] == pytest.approx(0.08, rel=1e-6)


def test_funded_phase_inf_target():
    pv = _ramp_pv(100_000.0, 200_000.0, 100)
    out = compute_challenge_target_hit_rate(pv, target_pct=math.inf)
    assert out["n_windows"] == 0
    assert out["hit_rate"] == 0.0
    assert out["bars_to_target_median"] is None


def test_sliding_windows_partial_hit():
    # 4 non-overlapping 25-bar windows, only the second hits +5%.
    n_window = 25
    bumps = []
    for i in range(4):
        if i == 1:
            bumps.append(_ramp_pv(100.0, 106.0, n_window))   # +6% within window
        else:
            bumps.append(_ramp_pv(100.0, 102.0, n_window))   # +2% only
    pv = np.concatenate(bumps)
    out = compute_challenge_target_hit_rate(
        pv, target_pct=0.05, window_bars=n_window,
    )
    assert out["n_windows"] == 4
    assert out["n_hits"] == 1
    assert out["hit_rate"] == 0.25
    assert sum(b is not None for b in out["bars_to_target_per_window"]) == 1


def test_sliding_windows_with_stride():
    # Window 25, stride 10 → starts at 0,10,20,...,75 (window must fit)
    pv = _ramp_pv(100.0, 100.0, 100)  # flat — never hits
    out = compute_challenge_target_hit_rate(
        pv, target_pct=0.01, window_bars=25, stride_bars=10,
    )
    assert out["n_windows"] == 8       # starts at 0,10,...,70
    assert out["n_hits"] == 0


def test_window_bigger_than_trajectory_falls_back_to_full():
    pv = _ramp_pv(100.0, 110.0, 50)
    out = compute_challenge_target_hit_rate(
        pv, target_pct=0.05, window_bars=999,
    )
    assert out["n_windows"] == 1
    assert out["n_hits"] == 1


def test_empty_trajectory():
    out = compute_challenge_target_hit_rate(np.array([]), target_pct=0.10)
    assert out["n_windows"] == 0
    assert out["hit_rate"] == 0.0
    assert out["max_cum_return"] == 0.0


def test_negative_baseline_defensive():
    pv = np.array([-1.0, 1.0, 2.0])
    out = compute_challenge_target_hit_rate(pv, target_pct=0.10)
    # cum_return is zeros → no hit, no error
    assert out["n_hits"] == 0


def test_invalid_window_bars_raises():
    pv = _ramp_pv(100.0, 110.0, 50)
    with pytest.raises(ValueError):
        compute_challenge_target_hit_rate(pv, target_pct=0.05, window_bars=0)


def test_invalid_stride_raises():
    pv = _ramp_pv(100.0, 110.0, 50)
    with pytest.raises(ValueError):
        compute_challenge_target_hit_rate(
            pv, target_pct=0.05, window_bars=10, stride_bars=0,
        )


def test_multi_phase_default_specs():
    pv = _ramp_pv(100_000.0, 108_000.0, 100)  # +8%: hits step2 (5%), misses step1 (10%)
    out = compute_challenge_target_hit_rates(pv)
    assert set(out.keys()) == {"step1", "step2"}
    assert out["step1"]["n_hits"] == 0
    assert out["step2"]["n_hits"] == 1


def test_parse_phase_spec_arg_basic():
    specs = parse_phase_spec_arg("step1:0.10,step2:0.05")
    assert specs == [
        {"label": "step1", "target_pct": 0.10},
        {"label": "step2", "target_pct": 0.05},
    ]


def test_parse_phase_spec_arg_with_window():
    specs = parse_phase_spec_arg("step1:0.10:1170,step2:0.05:1170:585")
    assert specs[0] == {"label": "step1", "target_pct": 0.10, "window_bars": 1170}
    assert specs[1] == {
        "label": "step2", "target_pct": 0.05,
        "window_bars": 1170, "stride_bars": 585,
    }


def test_parse_phase_spec_arg_rejects_malformed():
    with pytest.raises(ValueError):
        parse_phase_spec_arg("step1")
    with pytest.raises(ValueError):
        parse_phase_spec_arg("a:b:c:d:e")


def test_default_phase_specs_match_canonical_targets():
    labels = [s["label"] for s in DEFAULT_PHASE_SPECS]
    assert labels == ["step1", "step2"]
    targets = {s["label"]: s["target_pct"] for s in DEFAULT_PHASE_SPECS}
    assert targets == {"step1": 0.10, "step2": 0.05}
