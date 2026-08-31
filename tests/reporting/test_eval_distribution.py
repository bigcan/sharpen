"""Tests for Protocol v2.2 §2 eval_distribution helper."""

from __future__ import annotations

import numpy as np
import pytest

from sharpen.reporting.eval_distribution import (
    DEFAULT_DEADBAND_ABS,
    DEFAULT_HIST_EDGES,
    DEFAULT_SATURATION_ABS,
    _shannon_entropy,
    compute_eval_distribution,
    summarize_scalar_actions,
)


# ---------- summarize_scalar_actions ----------------------------------------


def test_summarize_scalar_empty_returns_zeros():
    out = summarize_scalar_actions(np.array([]))
    assert out["n"] == 0
    assert out["mean"] == 0.0
    assert out["std"] == 0.0
    assert out["entropy"] == 0.0
    assert out["deadband_frac"] == 0.0
    assert sum(out["counts"]) == 0


def test_summarize_scalar_all_in_deadband():
    # All actions < 0.25 in absolute value → 100% deadband
    a = np.array([0.0, 0.1, -0.1, 0.2, -0.2])
    out = summarize_scalar_actions(a)
    assert out["deadband_frac"] == pytest.approx(1.0)
    assert out["saturation_frac"] == pytest.approx(0.0)


def test_summarize_scalar_all_saturated():
    # All |a| > 0.95
    a = np.array([0.96, -0.97, 0.99, -1.0, 1.0])
    out = summarize_scalar_actions(a)
    assert out["saturation_frac"] == pytest.approx(1.0)
    assert out["deadband_frac"] == pytest.approx(0.0)


def test_summarize_scalar_histogram_bin_count():
    # Default bins = 9 edges → 8 bins
    rng = np.random.default_rng(0)
    a = rng.uniform(-1, 1, 200)
    out = summarize_scalar_actions(a)
    assert len(out["histogram_bins"]) == 9
    assert len(out["counts"]) == 8
    assert sum(out["counts"]) == 200


def test_summarize_scalar_moments():
    a = np.array([-0.5, 0.0, 0.5])
    out = summarize_scalar_actions(a)
    assert out["mean"] == pytest.approx(0.0)
    # population std (ddof=0)
    assert out["std"] == pytest.approx(np.std(a, ddof=0))


def test_summarize_scalar_custom_thresholds():
    # Actions in [0.3, 0.4] are deadband for 0.25, but not for 0.2.
    a = np.array([0.25, 0.3, 0.35])
    assert summarize_scalar_actions(a, deadband=0.2)["deadband_frac"] == 0.0
    assert summarize_scalar_actions(a, deadband=0.4)["deadband_frac"] == pytest.approx(1.0)


# ---------- shannon entropy --------------------------------------------------


def test_shannon_entropy_uniform_eight_bins():
    # H_max = log(8) for uniform over 8 bins
    counts = np.ones(8) * 100
    h = _shannon_entropy(counts)
    assert h == pytest.approx(np.log(8))


def test_shannon_entropy_single_bin_is_zero():
    counts = np.zeros(8)
    counts[3] = 100
    assert _shannon_entropy(counts) == pytest.approx(0.0)


def test_shannon_entropy_empty_is_zero():
    assert _shannon_entropy(np.zeros(8)) == 0.0


# ---------- compute_eval_distribution (scalar) ------------------------------


def test_compute_scalar_no_bar_vol_records_unavailable():
    a = np.linspace(-1, 1, 200)
    out = compute_eval_distribution(a)
    assert out["regime_bucketing"].startswith("unavailable: bar_vol")
    assert "by_vol_quartile" not in out
    assert "regime_cutpoints" not in out


def test_compute_scalar_with_bar_vol_records_cutpoints():
    rng = np.random.default_rng(1)
    a = np.clip(rng.normal(0, 0.3, 400), -1, 1)
    bv = np.abs(rng.normal(0, 1, 400))
    out = compute_eval_distribution(
        a, bar_vol=bv,
        regime_quartiles={"vol_q1": 0.25, "vol_q2": 0.25, "vol_q3": 0.25, "vol_q4": 0.25},
    )
    # EVAL-DIST-CUTPOINTS-01: cutpoints must be persisted for live tracker.
    assert "regime_cutpoints" in out
    assert len(out["regime_cutpoints"]) == 3
    assert out["regime_cutpoints"][0] < out["regime_cutpoints"][1] < out["regime_cutpoints"][2]
    # by_vol_quartile has all four buckets, each summing to total n.
    buckets = out["by_vol_quartile"]
    assert set(buckets) == {"q1", "q2", "q3", "q4"}
    assert sum(buckets[q]["n"] for q in buckets) == out["n"]


def test_compute_scalar_quartile_missing_key_falls_back():
    a = np.linspace(-1, 1, 100)
    bv = np.abs(np.linspace(0, 1, 100))
    out = compute_eval_distribution(
        a, bar_vol=bv, regime_quartiles={"vol_q1": 0.3, "vol_q2": 0.3},
    )
    assert out["regime_bucketing"].startswith("unavailable: regime_quartiles")


def test_compute_scalar_mismatched_bar_vol_length():
    a = np.linspace(-1, 1, 100)
    bv = np.zeros(50)  # wrong size
    out = compute_eval_distribution(a, bar_vol=bv)
    assert "unavailable: bar_vol length" in out["regime_bucketing"]


def test_compute_scalar_composition_rule_attached():
    a = np.linspace(-1, 1, 50)
    out = compute_eval_distribution(a, composition_rule="ens_agreement")
    assert out["composition_rule"] == "ens_agreement"


# ---------- compute_eval_distribution (multi-dim) ---------------------------


def test_compute_multidim_by_asset_keys_preserved():
    rng = np.random.default_rng(2)
    a = np.clip(rng.normal(0, 0.3, (100, 3)), -1, 1)
    out = compute_eval_distribution(a, asset_keys=["BTC", "ETH", "SOL"])
    assert list(out["by_asset"].keys()) == ["BTC", "ETH", "SOL"]
    for k in ("BTC", "ETH", "SOL"):
        assert out["by_asset"][k]["n"] == 100


def test_compute_multidim_default_asset_names():
    a = np.zeros((50, 4))
    out = compute_eval_distribution(a)
    assert list(out["by_asset"].keys()) == ["asset_0", "asset_1", "asset_2", "asset_3"]


def test_compute_multidim_mismatched_asset_keys_raises():
    a = np.zeros((50, 3))
    with pytest.raises(ValueError, match="asset_keys length"):
        compute_eval_distribution(a, asset_keys=["BTC", "ETH"])


def test_compute_3d_rejected():
    a = np.zeros((10, 10, 10))
    with pytest.raises(ValueError, match="must be 1-D or 2-D"):
        compute_eval_distribution(a)


# ---------- defaults match spec ---------------------------------------------


def test_default_constants_match_v22_spec():
    # v2.2 §8.2 reference values.
    assert DEFAULT_DEADBAND_ABS == 0.25
    assert DEFAULT_SATURATION_ABS == 0.95
    assert DEFAULT_HIST_EDGES[0] == -1.0
    assert DEFAULT_HIST_EDGES[-1] == 1.0
    assert len(DEFAULT_HIST_EDGES) == 9  # 8 bins


# ---------- Fix 2 (S538-cont) NaN sentinel filtering -----------------------


def test_summarize_scalar_filters_nan_from_deadband():
    # 100 actions: 30 NaN (no consensus), 50 in deadband, 20 directional.
    # deadband_frac is over consensus bars only: 50/70.
    # no_consensus_frac is over all bars: 30/100 = 0.30.
    a = np.concatenate([
        np.full(30, np.nan),
        np.full(50, 0.05),
        np.full(20, 0.5),
    ])
    out = summarize_scalar_actions(a)
    assert out["no_consensus_frac"] == pytest.approx(0.30)
    assert out["deadband_frac"] == pytest.approx(50 / 70)
    assert out["n"] == 70  # consensus-bar count
    assert out["n_total_bars"] == 100


def test_summarize_scalar_all_nan_returns_zero_metrics():
    # All-NaN input → no consensus bars; metrics are zero, no_consensus=1.0.
    a = np.full(50, np.nan)
    out = summarize_scalar_actions(a)
    assert out["no_consensus_frac"] == pytest.approx(1.0)
    assert out["n"] == 0
    assert out["n_total_bars"] == 50
    assert out["deadband_frac"] == 0.0


def test_summarize_scalar_no_nan_consensus_frac_zero():
    # Existing baselines (pre-Fix-2) had no NaN at all. Verify the new
    # field defaults sanely: no_consensus_frac == 0.0.
    a = np.array([-0.5, 0.0, 0.5])
    out = summarize_scalar_actions(a)
    assert out["no_consensus_frac"] == 0.0
    assert out["n"] == 3
    assert out["n_total_bars"] == 3
