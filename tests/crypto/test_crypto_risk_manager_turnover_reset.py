"""Tests for CryptoRiskManager bar-interval-aware daily-turnover reset.

Pre-S498 the reset window was hardcoded to 24 bars. On sub-hour bars combined
with signal-gate + deadband the "daily" window stretched across multiple
calendar days (S495-cont sg1-xauusd incident: 9 check() calls in 21h paper).

Post-S498 the window is `round(24*60 / bar_interval_minutes)` so:
  60 →  24 (1H, legacy default — backwards compat)
  15 →  96
   3 → 480
   1 →1440

Backwards-compat hinges on the default `bar_interval_minutes=60`.
"""
from __future__ import annotations

import numpy as np
import pytest

from finrl_pro_ds.crypto.mlops.crypto_risk_manager import (
    CryptoRiskConfig,
    CryptoRiskManager,
)


def _make_rm(bar_interval_minutes: int | None = None) -> CryptoRiskManager:
    kwargs = dict(
        enabled=True,
        max_drawdown_pct=0.99,
        max_position_pct=1.0,
        max_net_short_exposure=-1.0,
        min_effective_bets=1.0,
        max_gross_exposure=10.0,
        daily_turnover_limit=4.0,
        funding_rate_alert=999.0,
        min_margin_reserve_pct=0.0,
    )
    if bar_interval_minutes is not None:
        kwargs["bar_interval_minutes"] = bar_interval_minutes
    rm = CryptoRiskManager(CryptoRiskConfig(**kwargs))
    rm.reset(initial_capital=100_000.0)
    return rm


def _step(rm: CryptoRiskManager, action: float = 0.0) -> tuple[np.ndarray, list[str]]:
    """Single-asset step at PV=100K, no funding, current pos = 0."""
    a = np.array([action], dtype=np.float64)
    pos = np.array([0.0], dtype=np.float64)
    fund = np.array([0.0], dtype=np.float64)
    return rm.check(a, 100_000.0, 100_000.0, pos, fund)


@pytest.mark.parametrize(
    "bar_interval, expected_reset_bars",
    [(60, 24), (15, 96), (3, 480), (1, 1440)],
)
def test_reset_bars_scales_with_bar_interval(bar_interval, expected_reset_bars):
    rm = _make_rm(bar_interval_minutes=bar_interval)
    assert rm._reset_bars == expected_reset_bars


def test_default_bar_interval_preserves_legacy_24_bar_window():
    """No bar_interval_minutes set → defaults to 60 → reset every 24 bars (pre-S498)."""
    rm = _make_rm()  # no override
    assert rm.config.bar_interval_minutes == 60
    assert rm._reset_bars == 24


def test_15min_resets_at_96_bars_not_24():
    """The S495-cont signature: 15-min bars must NOT reset at 24."""
    rm = _make_rm(bar_interval_minutes=15)
    # Drive 24 bars — old code would have reset; new code must not.
    for _ in range(24):
        _step(rm)
    assert rm.state.bars_since_day_start == 24
    # Step 25: still no reset (was 25 with old code).
    _step(rm)
    assert rm.state.bars_since_day_start == 25
    # Burn through to 96 — that's the reset point.
    for _ in range(96 - 25):
        _step(rm)
    assert rm.state.bars_since_day_start == 96
    # Next step reset+increment → 1.
    _step(rm)
    assert rm.state.bars_since_day_start == 1


def test_3min_resets_at_480_bars():
    """SG-1 case: 3-min bars → reset window 480 bars (~24h wall-clock)."""
    rm = _make_rm(bar_interval_minutes=3)
    for _ in range(480):
        _step(rm)
    assert rm.state.bars_since_day_start == 480
    _step(rm)
    assert rm.state.bars_since_day_start == 1


def test_turnover_accumulator_actually_resets():
    """The reset must clear daily_turnover_accumulated (not just the bar counter)."""
    rm = _make_rm(bar_interval_minutes=60)  # 24-bar window
    # Build up some turnover
    for i in range(5):
        a = np.array([0.5 if i % 2 == 0 else -0.5], dtype=np.float64)
        pos = np.array([-0.5 if i % 2 == 0 else 0.5], dtype=np.float64)
        rm.check(a, 100_000.0, 100_000.0, pos, np.array([0.0]))
    pre_reset = rm.state.daily_turnover_accumulated
    assert pre_reset > 0
    # Spin to the reset boundary
    for _ in range(24 - 5):
        _step(rm)
    assert rm.state.bars_since_day_start == 24
    # Next call should reset+increment
    _step(rm)
    assert rm.state.bars_since_day_start == 1
    # Reset cleared accumulator (the trivial _step has zero delta so accumulator stays 0)
    assert rm.state.daily_turnover_accumulated == 0.0


def test_misconfigured_bar_interval_zero_does_not_div_zero():
    """Defensive: bar_interval=0 must not crash; floor at 1 → reset_bars=1440."""
    rm = _make_rm(bar_interval_minutes=0)
    assert rm._reset_bars == 1440  # 24 * 60 / max(1, 0)


def test_misconfigured_bar_interval_negative_floors_to_one():
    rm = _make_rm(bar_interval_minutes=-5)
    assert rm._reset_bars == 1440
