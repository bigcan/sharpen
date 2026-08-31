"""Intra-bar DD projection regression suite (S501 → S548-cont).

Covers `LiveTradingEngine._check_intrabar_dd` early-return guards and the
trip path. The 2026-04-27 S501 incident persisted a misleading
`pos=0.0000` in `risk_state.json` — caused by reading
`self._current_position` AFTER `_emergency_flatten()` rather than at
trip time. The pos-snapshot fix (S548-cont) plus FIND-01
(`max_daily_loss_pct <= 0` early-return, S498-cont 2026-04-26) and the
flat-position guard (`|pos| < 1e-9`, original S427) cover the bug
surface. This suite pins those guarantees so future refactors can't
regress them silently.

Test matrix:
- Early returns:
  - `safety.intrabar_dd_projection: false` disables
  - `safety.max_daily_loss_pct <= 0` disables (Velotrade contract)
  - flat position (`|pos| < 1e-9`) skips
  - non-positive `daily_start_value` skips
  - non-positive HL or close skips (obs_builder warmup)
- Trip path:
  - LONG position breaches via low-excursion projection
  - SHORT position breaches via high-excursion projection
  - `risk_state.json` detail records `pos_at_trip` (pre-flatten), not 0
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


def _build_minimal_config(
    *,
    max_daily_loss_pct: float = 0.04,  # FTMO-style; intrabar check armed
    intrabar_enabled: bool = True,
    state_dir: Path | None = None,
) -> dict:
    halt_path = (
        str(state_dir / "risk_state.json") if state_dir is not None
        else "/tmp/finrl_test_intrabar_dd_risk_state_doesnotexist.json"
    )
    return {
        "exchange": {"asset": "BTC"},
        "features": {"scales": [15, 60, 240], "obs_mode": "flat"},
        "trading": {"deadband_threshold": 0.05, "initial_balance": 100_000.0},
        "agent": {"device": "cpu"},
        "dry_run": True,
        "safety": {
            "kill_file": "/tmp/finrl_test_intrabar_dd_kill_doesnotexist",
            "emergency_flatten_on_error": False,
            "max_daily_loss_pct": max_daily_loss_pct,
            "intrabar_dd_projection": intrabar_enabled,
            "halt_state_file": halt_path,
        },
        "risk": {"enabled": False},
        "bar_clock": {"base_interval_minutes": 15},
    }


def _build_engine(
    config: dict,
    monkeypatch: pytest.MonkeyPatch,
    *,
    pos: float = 0.0,
    pv: float = 100_000.0,
    daily_start: float = 100_000.0,
    hl: tuple[float, float] = (101.0, 99.0),
    close: float = 100.0,
):
    """Wire engine with deps short-circuited; obs HL/close configurable."""
    from sharpen.crypto.live.live_engine import LiveTradingEngine

    monkeypatch.setenv("STRATEGY_NAME", "test-engine-intrabar-dd")
    monkeypatch.setenv("METRICS_PORT", "0")

    obs_builder = MagicMock()
    obs_builder.get_current_close = MagicMock(return_value=close)
    obs_builder.get_current_hl = MagicMock(return_value=hl)

    engine = LiveTradingEngine(
        agent=MagicMock(),
        broker=MagicMock(),
        obs_builder=obs_builder,
        risk_manager=MagicMock(),
        bar_clock=MagicMock(),
        loader=MagicMock(),
        config=config,
    )

    engine._current_position = pos
    engine._portfolio_value = pv
    engine._daily_start_value = daily_start
    engine._should_stop = False
    engine._strategy_name = "test-engine-intrabar-dd"
    return engine


def _bar_time() -> datetime:
    return datetime(2026, 5, 22, 12, 30, tzinfo=timezone.utc)


# -------------------------------------------------------------------
# Early-return guards
# -------------------------------------------------------------------


def test_disabled_flag_skips_check(monkeypatch):
    """`safety.intrabar_dd_projection: false` is the master kill-switch."""
    config = _build_minimal_config(intrabar_enabled=False)
    engine = _build_engine(config, monkeypatch, pos=-0.5)
    flatten_mock = AsyncMock()
    monkeypatch.setattr(engine, "_emergency_flatten", flatten_mock)

    asyncio.run(engine._check_intrabar_dd(_bar_time()))

    flatten_mock.assert_not_awaited()
    assert engine._should_stop is False


def test_max_daily_loss_zero_disables(monkeypatch):
    """FIND-01 (S498-cont): Velotrade-style configs with no daily rule are
    silent on the intrabar projection. Without this guard, any tiny
    negative projected_return trips on the first bar that holds a
    position (the original 2026-04-26 sg1-btc symptom)."""
    config = _build_minimal_config(max_daily_loss_pct=0.0)
    engine = _build_engine(config, monkeypatch, pos=-0.5)
    flatten_mock = AsyncMock()
    monkeypatch.setattr(engine, "_emergency_flatten", flatten_mock)

    asyncio.run(engine._check_intrabar_dd(_bar_time()))

    flatten_mock.assert_not_awaited()
    assert engine._should_stop is False


def test_flat_position_skips(monkeypatch):
    """Original S427 guard: a flat position has no adverse excursion to
    project. The daily-loss check (separate code path) owns the
    'we already lost too much sitting flat' case."""
    config = _build_minimal_config()
    engine = _build_engine(config, monkeypatch, pos=0.0)
    flatten_mock = AsyncMock()
    monkeypatch.setattr(engine, "_emergency_flatten", flatten_mock)

    asyncio.run(engine._check_intrabar_dd(_bar_time()))

    flatten_mock.assert_not_awaited()
    assert engine._should_stop is False


def test_non_positive_daily_start_skips(monkeypatch):
    """Division-by-zero / nonsense-baseline guard."""
    config = _build_minimal_config()
    engine = _build_engine(config, monkeypatch, pos=-0.5, daily_start=0.0)
    flatten_mock = AsyncMock()
    monkeypatch.setattr(engine, "_emergency_flatten", flatten_mock)

    asyncio.run(engine._check_intrabar_dd(_bar_time()))

    flatten_mock.assert_not_awaited()


def test_obs_builder_warmup_skips(monkeypatch):
    """get_current_hl returns (0, 0) before the first bar lands; the
    intrabar check must short-circuit instead of dividing by 0."""
    config = _build_minimal_config()
    engine = _build_engine(
        config, monkeypatch, pos=-0.5,
        hl=(0.0, 0.0), close=0.0,
    )
    flatten_mock = AsyncMock()
    monkeypatch.setattr(engine, "_emergency_flatten", flatten_mock)

    asyncio.run(engine._check_intrabar_dd(_bar_time()))

    flatten_mock.assert_not_awaited()


# -------------------------------------------------------------------
# Trip path
# -------------------------------------------------------------------


def test_long_breach_trips_via_low_excursion(monkeypatch, tmp_path):
    """LONG pos with a low far enough below close to project a daily
    loss > max_daily_loss_pct must flatten and halt."""
    config = _build_minimal_config(state_dir=tmp_path)
    # pos=+0.5, pv=100k, daily_start=100k, max_loss=4%
    # adverse_pnl = 0.5 * 100k * (low - close) / close
    # need adverse_pnl ≤ -4000 → (low - 100) / 100 ≤ -0.08 → low ≤ 92
    engine = _build_engine(
        config, monkeypatch, pos=0.5,
        hl=(105.0, 90.0), close=100.0,
    )
    flatten_mock = AsyncMock()
    monkeypatch.setattr(engine, "_emergency_flatten", flatten_mock)

    asyncio.run(engine._check_intrabar_dd(_bar_time()))

    flatten_mock.assert_awaited_once()
    assert engine._should_stop is True


def test_short_breach_trips_via_high_excursion(monkeypatch, tmp_path):
    """SHORT pos with a high far enough above close to project a daily
    loss > max_daily_loss_pct must flatten and halt."""
    config = _build_minimal_config(state_dir=tmp_path)
    # pos=-0.5, adverse_pnl = -0.5 * 100k * (high - 100) / 100
    # need ≤ -4000 → (high - 100) / 100 ≥ 0.08 → high ≥ 108
    engine = _build_engine(
        config, monkeypatch, pos=-0.5,
        hl=(110.0, 99.0), close=100.0,
    )
    flatten_mock = AsyncMock()
    monkeypatch.setattr(engine, "_emergency_flatten", flatten_mock)

    asyncio.run(engine._check_intrabar_dd(_bar_time()))

    flatten_mock.assert_awaited_once()
    assert engine._should_stop is True


# -------------------------------------------------------------------
# S548-cont halt-detail snapshot contract
# -------------------------------------------------------------------


def test_halt_detail_records_pre_flatten_position(monkeypatch, tmp_path):
    """S501 root cause: detail showed `pos=0.0000` because the f-string
    sampled `self._current_position` AFTER `_emergency_flatten()` zeroed
    it. Post S548-cont fix: detail records `pos_at_trip` from a
    snapshot taken BEFORE the flatten."""
    config = _build_minimal_config(state_dir=tmp_path)
    engine = _build_engine(
        config, monkeypatch, pos=-0.5,
        hl=(110.0, 99.0), close=100.0,
    )

    async def _flatten_zeros_position():
        engine._current_position = 0.0

    monkeypatch.setattr(
        engine, "_emergency_flatten",
        AsyncMock(side_effect=_flatten_zeros_position),
    )

    asyncio.run(engine._check_intrabar_dd(_bar_time()))

    halt_path = tmp_path / "risk_state.json"
    assert halt_path.exists(), "halt state file must be written on trip"
    payload = json.loads(halt_path.read_text())

    assert payload["reason"] == "intrabar_dd_projection"
    assert "pos_at_trip=-0.5000" in payload["detail"], (
        f"detail must record pre-flatten position; got {payload['detail']!r} "
        "(regression: post-flatten sampling would show pos_at_trip=0.0000)"
    )
    # Post-flatten internal state confirms the flatten DID run, so the
    # pre-flatten snapshot is the only place the trip-time pos survives.
    assert engine._current_position == 0.0
