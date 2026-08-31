"""LiveObsBuilder bootstrap calendar-window expansion (S546 GAP-B).

Verifies that the asset_class-aware expansion factor lands the right number
of calendar minutes for crypto vs weekend-closed CFD/futures markets.  See
docs/research/sg1_xauusd_sim_to_live_gap_audit.md §3 + §4.2.
"""

from __future__ import annotations

import logging

import pytest

from sharpen.crypto.live.live_obs_builder import LiveObsBuilder


def _builder(asset_class: str, bootstrap_bars: int = 30_000) -> LiveObsBuilder:
    return LiveObsBuilder(
        scales=[3, 15, 60],
        window_size=30,
        norm_span=120,
        n_features=8,
        bootstrap_bars=bootstrap_bars,
        drift_detection=False,
        asset_class=asset_class,
    )


class TestCalendarExpansion:
    def test_crypto_no_expansion(self):
        b = _builder("crypto")
        assert b.calendar_expansion_factor() == pytest.approx(1.0)
        # No expansion → calendar_minutes ≈ bootstrap_bars + 60.
        assert b.compute_bootstrap_calendar_minutes() == 30_060

    @pytest.mark.parametrize("asset_class", ["cfd_gold", "cfd_forex", "cme_futures"])
    def test_weekend_closed_expansion(self, asset_class: str):
        b = _builder(asset_class)
        assert b.calendar_expansion_factor() == pytest.approx(1.45)
        # 30_060 * 1.45 = 43_587 minutes ≈ 30.27 calendar days, enough for
        # 30_000 active bars on a 5-day-trading-week instrument.
        assert b.compute_bootstrap_calendar_minutes() == 43_587

    def test_unknown_asset_class_falls_back_to_one(self, caplog):
        # The warning about unknown asset classes is emitted at construction;
        # the factor still falls back to 1.0 so we don't silently over-fetch
        # for a 24/7-like market.
        with caplog.at_level(logging.WARNING):
            b = _builder("commodities_synthetic")
        assert b.calendar_expansion_factor() == pytest.approx(1.0)
        assert b.compute_bootstrap_calendar_minutes() == 30_060
        assert any(
            "unknown asset_class" in record.message for record in caplog.records
        )

    def test_expansion_scales_with_bootstrap_bars(self):
        b = _builder("cfd_gold", bootstrap_bars=10_000)
        # (10_000 + 60) * 1.45 = 14_587.0 → int() = 14_587
        assert b.compute_bootstrap_calendar_minutes() == 14_587

    def test_audit_sg1_xauusd_target_lands(self):
        # S546 audit §3.2: a 30_000-bar bootstrap on Gold CFDs returned only
        # 20_468 active 1-min bars under the naïve calendar window
        # (30_060 min ≈ 20.875 calendar days, ≥3 full weekends lost).
        # With the 1.45× expansion we now request ~30.3 calendar days,
        # which fits the audit's projection of "guaranteed 100%
        # normalization convergence on the 60-min scale".
        b = _builder("cfd_gold", bootstrap_bars=30_000)
        calendar_min = b.compute_bootstrap_calendar_minutes()
        calendar_days = calendar_min / (60 * 24)
        # Allow ±1 day around the §4.2 estimate of 31.25 calendar days.
        assert calendar_days == pytest.approx(30.27, abs=1.0)
