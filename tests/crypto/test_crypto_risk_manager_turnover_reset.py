"""Tests for CryptoRiskManager daily-turnover reset behavior.

Two layers covered here:

1. **Option A (S498 — call-count fallback).** Pre-S498 the reset window was
   hardcoded to 24 bars. On sub-hour bars combined with signal-gate + deadband
   the "daily" window stretched across multiple calendar days (S495-cont
   sg1-xauusd incident: 9 check() calls in 21h paper). S498 made the window
   `round(24*60 / bar_interval_minutes)` so:
     60 →  24 (1H, legacy default — backwards compat)
     15 →  96
      3 → 480
      1 →1440
   This path runs whenever ``bar_time`` is NOT supplied (e.g., unit tests,
   research harnesses).

2. **Option B (S506 — UTC-midnight wall-clock reset).** When the engine
   passes ``bar_time``, the reset fires on the first call after a UTC date
   boundary crosses. This is gating-resistant (signal_gate / deadband can
   stretch the call rate without stretching the reset window) and aligns
   with prop-firm "daily loss" semantics already used by
   ``eod_trailing_drawdown``.

Backwards-compat hinges on:
  - The default ``bar_interval_minutes=60`` for Option A.
  - Falling back to Option A bit-identically when ``bar_time`` is omitted.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from finrl_pro_ds.crypto.mlops.crypto_risk_manager import (
    CryptoRiskConfig,
    CryptoRiskManager,
)


def _make_rm(
    bar_interval_minutes: int | None = None,
    soft_throttle_start: float | None = None,
) -> CryptoRiskManager:
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
    if soft_throttle_start is not None:
        kwargs["soft_throttle_start"] = soft_throttle_start
    rm = CryptoRiskManager(CryptoRiskConfig(**kwargs))
    rm.reset(initial_capital=100_000.0)
    return rm


def _step(rm: CryptoRiskManager, action: float = 0.0) -> tuple[np.ndarray, list[str]]:
    """Single-asset step at PV=100K, no funding, current pos = 0."""
    a = np.array([action], dtype=np.float64)
    pos = np.array([0.0], dtype=np.float64)
    fund = np.array([0.0], dtype=np.float64)
    return rm.check(a, 100_000.0, 100_000.0, pos, fund)


def _step_at(
    rm: CryptoRiskManager,
    action: float,
    bar_time,
    current_pos: float = 0.0,
) -> tuple[np.ndarray, list[str]]:
    """Single-asset step at PV=100K, no funding, with explicit ``bar_time``.

    Mirrors how ``live_engine._trading_step_inner`` calls ``check()`` after
    S506 wiring: the engine forwards its ``bar_time: datetime`` into the
    risk manager so daily-turnover reset can pivot on UTC date.
    """
    a = np.array([action], dtype=np.float64)
    pos = np.array([current_pos], dtype=np.float64)
    fund = np.array([0.0], dtype=np.float64)
    return rm.check(a, 100_000.0, 100_000.0, pos, fund, bar_time=bar_time)


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


# -----------------------------------------------------------------------------
# Option B (S506) — UTC-midnight-anchored reset when ``bar_time`` is supplied.
# -----------------------------------------------------------------------------


def test_no_bar_time_falls_back_to_call_count_reset():
    """Regression: when ``bar_time`` is omitted entirely, the manager must use
    the S498 Option-A bar-interval-aware call-count reset bit-identically.

    This test exists alongside the existing Option-A coverage above to make
    the *contract* (no bar_time → Option A) explicit, since callers like the
    paper-trade harness or ad-hoc REPL sessions may not have a wall clock.
    """
    rm = _make_rm(bar_interval_minutes=15)  # _reset_bars=96
    # Drive 95 zero-delta calls — no reset yet.
    for _ in range(95):
        _step(rm)
    assert rm.state.bars_since_day_start == 95
    assert rm.state.last_turnover_reset_date == ""  # Option B never engaged
    # 96th call still no reset (Option A resets BEFORE the 97th increment).
    _step(rm)
    assert rm.state.bars_since_day_start == 96
    # 97th call: Option A reset fires, counter goes to 1.
    _step(rm)
    assert rm.state.bars_since_day_start == 1
    assert rm.state.last_turnover_reset_date == ""  # still never engaged


def test_utc_midnight_resets_turnover():
    """Happy path: cross UTC midnight → accumulator clears on next call."""
    rm = _make_rm(bar_interval_minutes=15)
    pre_midnight = datetime(2026, 4, 29, 23, 55, tzinfo=timezone.utc)
    post_midnight = datetime(2026, 4, 30, 0, 1, tzinfo=timezone.utc)

    # Pre-midnight: accumulate some turnover (action=0.5 on flat pos → delta=0.5).
    _step_at(rm, 0.5, pre_midnight)
    assert rm.state.daily_turnover_accumulated == pytest.approx(0.5)
    assert rm.state.last_turnover_reset_date == "20260429"

    # Post-midnight: reset fires, accumulator starts fresh from this call.
    _step_at(rm, 0.5, post_midnight)
    assert rm.state.daily_turnover_accumulated == pytest.approx(0.5)
    assert rm.state.last_turnover_reset_date == "20260430"
    assert rm.state.bars_since_day_start == 1


def test_pre_midnight_no_reset():
    """Many calls within the same UTC date must not trigger a reset."""
    rm = _make_rm(bar_interval_minutes=15)
    base = datetime(2026, 4, 29, 0, 1, tzinfo=timezone.utc)
    for i in range(50):
        bt = base + timedelta(minutes=15 * i)  # 00:01..12:36, all 2026-04-29
        _step_at(rm, 0.05, bt)
    # 50 calls × 0.05 delta = 2.5 (under the 4.0 cap; no scaling).
    assert rm.state.daily_turnover_accumulated == pytest.approx(2.5)
    assert rm.state.last_turnover_reset_date == "20260429"
    assert rm.state.bars_since_day_start == 50


def test_multi_day_jump_resets_once():
    """A gap in trading (no calls for several days) followed by a single call
    on day N+3 must reset exactly once and leave the accumulator clean.
    """
    rm = _make_rm(bar_interval_minutes=60)
    day_n = datetime(2026, 4, 26, 12, 0, tzinfo=timezone.utc)
    day_n_plus_3 = datetime(2026, 4, 29, 12, 0, tzinfo=timezone.utc)

    _step_at(rm, 0.7, day_n)
    assert rm.state.last_turnover_reset_date == "20260426"
    assert rm.state.daily_turnover_accumulated == pytest.approx(0.7)

    _step_at(rm, 0.3, day_n_plus_3)
    assert rm.state.last_turnover_reset_date == "20260429"
    # Accumulator cleared then accrued the day-N+3 delta only.
    assert rm.state.daily_turnover_accumulated == pytest.approx(0.3)
    assert rm.state.bars_since_day_start == 1


# -----------------------------------------------------------------------------
# Reproducer tests for the two production incidents that drove this fix.
# -----------------------------------------------------------------------------


def test_sg1_xauusd_3min_signal_gated_no_freeze():
    """S495-cont reproducer: SG-1 XAUUSD 3-min ens_agreement signal-gated.

    Production trace: ~9 ``check()`` calls in 21h paper, accumulating ~3.79
    turnover. With Option-A's 480-call window the accumulator never resets,
    so the 10th call clipped against the 4.0 cap and the strategy froze
    flat for 7+ hours.

    Post-fix expectation: 9 same-UTC-day calls accumulate without violation
    (3.78 < 4.0); the 10th call on the next UTC date resets the accumulator
    cleanly and only carries its own delta.
    """
    rm = _make_rm(bar_interval_minutes=3)  # _reset_bars=480 in fallback

    # 9 calls on 2026-04-22 spread across ~21h, each delta ≈ 0.42.
    base = datetime(2026, 4, 22, 0, 30, tzinfo=timezone.utc)
    for i in range(9):
        bt = base + timedelta(hours=2.5 * i)  # 00:30..20:30 — all same UTC day
        _, viol = _step_at(rm, 0.42, bt)
        assert all("DAILY_TURNOVER" not in v for v in viol), (
            f"call {i} unexpectedly clipped on turnover: {viol}"
        )
    # Accumulated below the cap.
    assert rm.state.daily_turnover_accumulated == pytest.approx(0.42 * 9, abs=1e-6)
    assert rm.state.daily_turnover_accumulated < 4.0
    assert rm.state.last_turnover_reset_date == "20260422"

    # 10th call after UTC midnight → reset fires.
    next_day = datetime(2026, 4, 23, 0, 30, tzinfo=timezone.utc)
    _, viol = _step_at(rm, 0.42, next_day)
    assert all("DAILY_TURNOVER" not in v for v in viol)
    assert rm.state.last_turnover_reset_date == "20260423"
    assert rm.state.daily_turnover_accumulated == pytest.approx(0.42)


def test_gmgp1_btc_15min_deadband_no_freeze():
    """S506 reproducer: gmgp1-btc 15-min, deadband=0.25, signal_gate disabled.

    Production trace: continuous DAILY_TURNOVER violations 4.35–5.21 over 5+
    hours because deadband swallowed enough bars that the 96-call Option-A
    window stretched well past the calendar day.

    Post-fix expectation: ~3h spanning UTC midnight, 6 deadband-survivor
    calls — each delta=1.0 — cap budget is 4.0/day. Pre-midnight side
    accumulates to 3.0; the post-midnight reset means the post-midnight side
    also caps at 3.0. Total 6 calls but zero DAILY_TURNOVER violations.
    """
    rm = _make_rm(bar_interval_minutes=15)

    # 3 pre-midnight calls 21:30 / 22:30 / 23:30 on 2026-04-28.
    pre_times = [
        datetime(2026, 4, 28, 21, 30, tzinfo=timezone.utc),
        datetime(2026, 4, 28, 22, 30, tzinfo=timezone.utc),
        datetime(2026, 4, 28, 23, 30, tzinfo=timezone.utc),
    ]
    for bt in pre_times:
        _, viol = _step_at(rm, 1.0, bt)
        assert all("DAILY_TURNOVER" not in v for v in viol), (
            f"pre-midnight call {bt} unexpectedly clipped: {viol}"
        )
    assert rm.state.daily_turnover_accumulated == pytest.approx(3.0)
    assert rm.state.last_turnover_reset_date == "20260428"

    # 3 post-midnight calls 00:00 / 00:15 / 00:30 on 2026-04-29.
    # 00:00 must reset the accumulator before accruing its own delta.
    post_times = [
        datetime(2026, 4, 29, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 4, 29, 0, 15, tzinfo=timezone.utc),
        datetime(2026, 4, 29, 0, 30, tzinfo=timezone.utc),
    ]
    for bt in post_times:
        _, viol = _step_at(rm, 1.0, bt)
        assert all("DAILY_TURNOVER" not in v for v in viol), (
            f"post-midnight call {bt} unexpectedly clipped: {viol}"
        )
    assert rm.state.last_turnover_reset_date == "20260429"
    # Per-day max accumulator stayed at 3.0 (under the 4.0 cap).
    assert rm.state.daily_turnover_accumulated == pytest.approx(3.0)


# -----------------------------------------------------------------------------
# Edge cases for the bar_time → UTC date coercion.
# -----------------------------------------------------------------------------


def test_dst_unaffected_via_utc_normalization():
    """DST flips in non-UTC timezones must not affect reset semantics — the
    reset key is the UTC date string, and the helper coerces tz-aware inputs
    via ``astimezone(timezone.utc)`` before extracting the date.
    """
    rm = _make_rm(bar_interval_minutes=15)
    eastern = timezone(timedelta(hours=-5))  # constant offset (no DST modeling)

    # Both Eastern timestamps fall on 2026-03-08 in UTC.
    bar1 = datetime(2026, 3, 7, 23, 30, tzinfo=eastern)  # = 2026-03-08 04:30 UTC
    bar2 = datetime(2026, 3, 8, 0, 30, tzinfo=eastern)   # = 2026-03-08 05:30 UTC

    _step_at(rm, 0.5, bar1)
    pre = rm.state.daily_turnover_accumulated
    assert rm.state.last_turnover_reset_date == "20260308"

    _step_at(rm, 0.5, bar2)
    # No reset (still same UTC day), accumulator continues climbing.
    assert rm.state.daily_turnover_accumulated > pre
    assert rm.state.last_turnover_reset_date == "20260308"


def test_leap_second_does_not_double_reset():
    """Python's ``datetime`` does not model leap seconds (cannot construct
    ``2016-12-31T23:59:60Z``), so leap-second handling is a documented
    no-op. We assert that a regular timestamp inside the surrounding minute
    behaves normally.
    """
    rm = _make_rm()
    bar = datetime(2016, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    _step_at(rm, 0.5, bar)
    assert rm.state.last_turnover_reset_date == "20161231"
    next_bar = datetime(2017, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
    _step_at(rm, 0.5, next_bar)
    assert rm.state.last_turnover_reset_date == "20170101"


def test_bar_time_as_int_epoch_works():
    """The engine sometimes passes ``int(timestamp.timestamp())``. The same
    isinstance check used by the EOD trailing-DD path must apply here.
    """
    rm = _make_rm()
    epoch = int(datetime(2026, 4, 29, 12, 0, tzinfo=timezone.utc).timestamp())
    _step_at(rm, 0.5, epoch)
    assert rm.state.last_turnover_reset_date == "20260429"


def test_bar_time_as_float_epoch_works():
    """Float epoch (sub-second precision) must also coerce cleanly."""
    rm = _make_rm()
    epoch = datetime(2026, 4, 29, 12, 0, 0, 500_000, tzinfo=timezone.utc).timestamp()
    _step_at(rm, 0.5, epoch)
    assert rm.state.last_turnover_reset_date == "20260429"


def test_bar_time_as_naive_datetime_treated_as_utc():
    """Defensive: if a caller hands in a tz-naive datetime, treat as UTC
    (do not crash, do not silently use system local time).
    """
    rm = _make_rm()
    naive = datetime(2026, 4, 29, 12, 0)  # no tzinfo
    _step_at(rm, 0.5, naive)
    assert rm.state.last_turnover_reset_date == "20260429"


# -----------------------------------------------------------------------------
# Soft-throttle (S535 ADR-1) — piecewise-linear taper before hard 100% cap.
#
# The legacy hard wall (soft_throttle_start=1.0) is the dataclass default and
# is exercised by every test above; these tests cover the opt-in path where
# the throttle band is enabled.
# -----------------------------------------------------------------------------


def _step_with_pos(
    rm: CryptoRiskManager, action: float, current_pos: float = 0.0,
) -> tuple[np.ndarray, list[str]]:
    """Single-asset step with explicit current position (no bar_time)."""
    a = np.array([action], dtype=np.float64)
    pos = np.array([current_pos], dtype=np.float64)
    fund = np.array([0.0], dtype=np.float64)
    return rm.check(a, 100_000.0, 100_000.0, pos, fund)


def test_soft_throttle_below_band_no_clip():
    """Below `soft_throttle_start`: action passes through unchanged."""
    rm = _make_rm(soft_throttle_start=0.8)  # limit=4.0 → band entry at 3.2
    # Pre-load accumulator to 2.0 (50% budget — well below 80% band).
    rm.state.daily_turnover_accumulated = 2.0
    out, viol = _step_with_pos(rm, 0.5)
    assert out[0] == pytest.approx(0.5)  # full pass-through
    assert all("DAILY_TURNOVER" not in v for v in viol)
    assert rm.state.daily_turnover_accumulated == pytest.approx(2.5)


def test_soft_throttle_midband_half_scale():
    """At midpoint of [0.8, 1.0] band (budget=0.9): scale ≈ 0.5."""
    rm = _make_rm(soft_throttle_start=0.8)  # limit=4.0
    # Pre-load accumulator to 3.6 (90% budget — midway through the band).
    # Linear taper: scale = (1.0 - 0.9) / (1.0 - 0.8) = 0.5.
    rm.state.daily_turnover_accumulated = 3.6
    out, viol = _step_with_pos(rm, 0.5)
    # Output should be midway between current pos (0.0) and target (0.5).
    assert out[0] == pytest.approx(0.25)
    assert any("DAILY_TURNOVER" in v for v in viol)
    # Accumulator advanced by the *post-throttle* delta = 0.25.
    assert rm.state.daily_turnover_accumulated == pytest.approx(3.85)


def test_soft_throttle_full_block_at_exhaustion():
    """At 100% budget (or beyond): scale = 0, output = current position."""
    rm = _make_rm(soft_throttle_start=0.8)  # limit=4.0
    rm.state.daily_turnover_accumulated = 4.0  # exactly exhausted
    out, viol = _step_with_pos(rm, 0.5, current_pos=0.1)
    assert out[0] == pytest.approx(0.1)  # held at current pos
    assert any("DAILY_TURNOVER" in v for v in viol)
    assert rm.state.daily_turnover_accumulated == pytest.approx(4.0)


def test_soft_throttle_disabled_with_start_at_one_preserves_legacy():
    """`soft_throttle_start=1.0` (default) bit-identical to legacy hard wall:
    full pass-through for any pre-budget < 100%, then sharp clip when
    pre+delta crosses 100% (clipped to remaining budget).
    """
    rm = _make_rm(soft_throttle_start=1.0)  # explicit default
    rm.state.daily_turnover_accumulated = 3.7  # 92.5% — would be in throttle band if start=0.8
    out, viol = _step_with_pos(rm, 0.5)
    # Legacy: full pass-through because budget_pre < 1.0 and pre+delta=4.2 > 4.0
    # → hard cap clamps delta from 0.5 → 0.3 (remaining = 0.3).
    assert out[0] == pytest.approx(0.3)
    assert any("DAILY_TURNOVER" in v for v in viol)
    assert rm.state.daily_turnover_accumulated == pytest.approx(4.0)


def test_soft_throttle_below_band_but_overshoot_clamped_by_hard_cap():
    """Operator-amendment clamp (S535): a single large delta that originates
    well below `soft_throttle_start` but would push post-budget over 100%
    must be clipped by the hard cap to land at exactly 100%, not allowed to
    overshoot just because soft_scale=1.0.

    Realistic scenario: SAC flips from full short (-1.0) to full long (+1.0)
    in a single bar — delta=2.0. With pre_accumulated=3.0 of a 4.0 budget,
    budget_pre=0.75 is below the band entry 0.8, so the soft taper does not
    fire. Without the operator clamp the post-budget would land at 5.0
    (overshoot by 25%). The hard cap clamps final delta to remaining=1.0, so
    the executed flip is half the policy intent (final position = 0.0) and
    the accumulator pins at exactly the limit.
    """
    rm = _make_rm(soft_throttle_start=0.8)  # band: 3.2..4.0
    rm.state.daily_turnover_accumulated = 3.0  # 75% — below band entry 0.8 (3.2)
    out, viol = _step_with_pos(rm, action=1.0, current_pos=-1.0)  # delta=2.0
    # soft_scale=1.0 (pre 75% < 80%); post would be 5.0; hard cap clamps to 1.0.
    # scale = 1.0/2.0 = 0.5; modified = -1 + (1 - (-1)) * 0.5 = 0.0.
    assert out[0] == pytest.approx(0.0)
    assert any("DAILY_TURNOVER" in v for v in viol)
    assert rm.state.daily_turnover_accumulated == pytest.approx(4.0)
