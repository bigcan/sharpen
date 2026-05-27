"""Tests for ``LiveObsBuilder.feature_variance_status()`` — Step 1 of the
ActionDriftTracker v2.6 amendment (S551-cont-4).

These tests bypass bootstrap and directly populate the rolling state so the
helper paths can be exercised in isolation. Parity / bootstrap coverage lives
in ``test_live_obs_parity.py`` / ``test_live_obs_bootstrap_calendar.py``.
"""

from collections import deque

import numpy as np
import pytest

from finrl_pro_ds.crypto.live.live_obs_builder import (
    FeatureVarianceStatus,
    LiveObsBuilder,
)


def _builder(scales: list[int] = (3, 15, 60)) -> LiveObsBuilder:
    """LiveObsBuilder configured but not bootstrapped — we'll inject state directly."""
    return LiveObsBuilder(
        scales=list(scales),
        window_size=30,
        norm_span=120,
        n_features=8,
        bootstrap_bars=1000,
        drift_detection=True,
        drift_window=100,
    )


def _seed_history(
    builder: LiveObsBuilder,
    *,
    scale: int,
    median_var: float,
    current_var: float,
    n_samples: int = 20,
) -> None:
    """Seed _variance_history + _scale_features so _compute_variance_ratio returns
    a known ratio = current_var / median_var (within float rounding).

    We deposit ``n_samples`` history entries all equal to ``median_var`` so the
    rolling median is exactly ``median_var``, then set the latest feature row
    such that ``np.var(row) == current_var``.
    """
    builder._variance_history[scale] = deque(
        [median_var] * n_samples, maxlen=builder._drift_window
    )
    # Construct an 8-cell row whose np.var equals current_var.
    # np.var(x) = mean((x-mean)^2). For an alternating-sign row of magnitude a,
    # mean=0 and every element contributes a^2, so var = a^2 → pick a=sqrt(var).
    if current_var <= 0:
        row = np.zeros(8, dtype=np.float64)
    else:
        a = float(np.sqrt(current_var))
        row = np.array([a, -a, a, -a, a, -a, a, -a], dtype=np.float64)
    builder._scale_features[scale] = np.array([row])


def test_feature_variance_status_returns_ok_on_normal_buffer():
    """Ratios in the normal band → state=OK."""
    b = _builder()
    for s in b.scales:
        _seed_history(b, scale=s, median_var=1.0, current_var=1.0)
    fvs = b.feature_variance_status()
    assert isinstance(fvs, FeatureVarianceStatus)
    assert fvs.state == "OK"
    assert fvs.base_scale == 3
    assert fvs.base_scale_ratio == pytest.approx(1.0, rel=1e-6)
    assert set(fvs.per_scale_ratio.keys()) == set(b.scales)


def test_feature_variance_status_returns_flat_on_constant_features():
    """Ratio < flat_threshold (0.01) on the base scale → state=FLAT."""
    b = _builder()
    # Base scale flat (ratio 0.001 < 0.01); other scales healthy
    _seed_history(b, scale=3, median_var=1.0, current_var=0.001)
    _seed_history(b, scale=15, median_var=1.0, current_var=1.0)
    _seed_history(b, scale=60, median_var=1.0, current_var=1.0)
    fvs = b.feature_variance_status(scale="base")
    assert fvs.state == "FLAT"
    assert fvs.base_scale_ratio == pytest.approx(0.001, rel=1e-6)


def test_feature_variance_status_returns_explode_on_jump():
    """Ratio > explode_threshold (100) on base scale → state=EXPLODE."""
    b = _builder()
    _seed_history(b, scale=3, median_var=1.0, current_var=200.0)
    _seed_history(b, scale=15, median_var=1.0, current_var=1.0)
    _seed_history(b, scale=60, median_var=1.0, current_var=1.0)
    fvs = b.feature_variance_status(scale="base")
    assert fvs.state == "EXPLODE"
    assert fvs.base_scale_ratio == pytest.approx(200.0, rel=1e-6)


def test_feature_variance_status_insufficient_samples_is_ok():
    """<10 history samples → ratio=None → state=OK (conservative, no veto on missing data)."""
    b = _builder()
    # Seed only 5 samples on each scale (below the 10-minimum)
    for s in b.scales:
        b._variance_history[s] = deque([1.0] * 5, maxlen=b._drift_window)
        b._scale_features[s] = np.array([[0.001, -0.001, 0, 0, 0, 0, 0, 0]])
    fvs = b.feature_variance_status()
    assert fvs.state == "OK"
    assert fvs.base_scale_ratio is None
    assert all(v is None for v in fvs.per_scale_ratio.values())


def test_feature_variance_status_per_scale_resolution():
    """``scale=int`` and ``scale='all'`` produce different states when only one
    non-base scale is FLAT."""
    b = _builder()
    # Base + 60min healthy; 15min FLAT
    _seed_history(b, scale=3, median_var=1.0, current_var=1.0)
    _seed_history(b, scale=15, median_var=1.0, current_var=0.001)
    _seed_history(b, scale=60, median_var=1.0, current_var=1.0)

    assert b.feature_variance_status(scale="base").state == "OK"     # base healthy
    assert b.feature_variance_status(scale=15).state == "FLAT"        # the flat one
    assert b.feature_variance_status(scale=60).state == "OK"
    assert b.feature_variance_status(scale="all").state == "FLAT"     # any-flat


def test_feature_variance_status_all_picks_explode_over_ok_but_not_over_flat():
    """``scale='all'`` priority: FLAT > EXPLODE > OK."""
    b = _builder()
    _seed_history(b, scale=3, median_var=1.0, current_var=1.0)
    _seed_history(b, scale=15, median_var=1.0, current_var=200.0)     # EXPLODE
    _seed_history(b, scale=60, median_var=1.0, current_var=0.001)     # FLAT
    fvs = b.feature_variance_status(scale="all")
    assert fvs.state == "FLAT"  # FLAT wins because it's the harder failure to recover from


def test_feature_variance_status_disabled_returns_ok():
    """drift_detection=False → all ratios None → state=OK."""
    b = LiveObsBuilder(
        scales=[3, 15, 60],
        drift_detection=False,
    )
    _seed_history(b, scale=3, median_var=1.0, current_var=0.001)  # would be FLAT if enabled
    fvs = b.feature_variance_status()
    assert fvs.state == "OK"
    assert fvs.base_scale_ratio is None


def test_feature_variance_status_invalid_scale_int_raises():
    """``scale=99`` (not in configured scales) → ValueError."""
    b = _builder()
    for s in b.scales:
        _seed_history(b, scale=s, median_var=1.0, current_var=1.0)
    with pytest.raises(ValueError, match="not in configured"):
        b.feature_variance_status(scale=99)


def test_feature_variance_status_invalid_scale_type_raises():
    """``scale=2.5`` (float, not int/str) → ValueError."""
    b = _builder()
    for s in b.scales:
        _seed_history(b, scale=s, median_var=1.0, current_var=1.0)
    with pytest.raises(ValueError, match="must be 'base', 'all', or int"):
        b.feature_variance_status(scale=2.5)


def test_compute_variance_ratio_degenerate_median_returns_none():
    """Rolling median < 1e-15 → None (matches _check_drift guard)."""
    b = _builder()
    b._variance_history[3] = deque([0.0] * 20, maxlen=b._drift_window)
    b._scale_features[3] = np.array([[1.0, -1.0, 0, 0, 0, 0, 0, 0]])
    assert b._compute_variance_ratio(3) is None


def test_check_drift_behavior_unchanged_after_refactor():
    """Regression: _check_drift still flags FLAT/EXPLODE and updates _drift_detected.

    This guards the refactor — extracting _ratio_to_state must NOT change the
    semantics of _check_drift's existing detection path.
    """
    b = _builder()
    # Seed sufficient history; latest feature row is FLAT vs rolling median.
    for s in b.scales:
        _seed_history(b, scale=s, median_var=1.0, current_var=0.001, n_samples=50)
    detected = b._check_drift()
    assert detected is True
    assert b.drift_detected is True
