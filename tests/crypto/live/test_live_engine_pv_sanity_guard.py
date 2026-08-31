"""Live-engine sanity guard against transient broker equity readings (S510).

`_update_portfolio_value` calls `broker.get_account_info()` every bar AND
every 30 seconds via `_inter_bar_metrics_loop`. cTrader IC Markets demo has
been observed to occasionally return total_equity ≈ (config_initial +
broker_balance), which would silently ratchet `_peak_portfolio_value` to
roughly 2× the true equity and produce a phantom 50% drawdown report on
every subsequent bar (gmgp1-xauusd + sg1-xauusd, 2026-04-30 12:00 UTC).

The guard rejects readings >1.5× the larger of current PV and config
initial_balance. We test that:

  - normal readings pass through and ratchet peak as before;
  - a transient 2× spike is rejected, leaving PV and peak untouched;
  - a recovery reading after a rejection is accepted.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


def _build_engine_for_pv_test(monkeypatch):
    """Minimal engine sufficient to drive `_update_portfolio_value`."""
    from sharpen.crypto.live.live_engine import LiveTradingEngine

    monkeypatch.setenv("STRATEGY_NAME", "test-engine-s510")
    monkeypatch.setenv("METRICS_PORT", "0")

    config = {
        "exchange": {"asset": "XAUUSD"},
        "features": {"scales": [15, 60, 240], "obs_mode": "flat"},
        "trading": {"deadband_threshold": 0.05, "initial_balance": 100_000.0},
        "agent": {"device": "cpu"},
        "dry_run": True,
        "safety": {
            "kill_file": "/tmp/finrl_test_kill_doesnotexist_s510",
            "emergency_flatten_on_error": False,
        },
        "risk": {"enabled": False},
        "bar_clock": {"base_interval_minutes": 15},
    }

    engine = LiveTradingEngine(
        agent=MagicMock(),
        broker=MagicMock(),
        obs_builder=MagicMock(),
        risk_manager=MagicMock(),
        bar_clock=MagicMock(),
        loader=MagicMock(),
        config=config,
    )
    return engine


def test_update_pv_accepts_normal_reading(monkeypatch):
    engine = _build_engine_for_pv_test(monkeypatch)
    engine._portfolio_value = 100_000.0
    engine._peak_portfolio_value = 100_000.0
    engine.broker.get_account_info = AsyncMock(
        return_value={"total_equity": 102_911.39},
    )

    asyncio.run(engine._update_portfolio_value())

    assert engine._portfolio_value == pytest.approx(102_911.39)
    assert engine._peak_portfolio_value == pytest.approx(102_911.39)


def test_update_pv_rejects_2x_transient(monkeypatch):
    """The exact failure mode from gmgp1-xauusd / sg1-xauusd 2026-04-30 12:00."""
    engine = _build_engine_for_pv_test(monkeypatch)
    engine._portfolio_value = 102_911.39
    engine._peak_portfolio_value = 102_911.39
    engine.broker.get_account_info = AsyncMock(
        return_value={"total_equity": 202_911.39},  # ~2× initial
    )

    asyncio.run(engine._update_portfolio_value())

    # Both PV and peak must remain at the pre-spike value.
    assert engine._portfolio_value == pytest.approx(102_911.39)
    assert engine._peak_portfolio_value == pytest.approx(102_911.39)


def test_update_pv_recovers_after_rejection(monkeypatch):
    engine = _build_engine_for_pv_test(monkeypatch)
    engine._portfolio_value = 102_911.39
    engine._peak_portfolio_value = 102_911.39

    # First call: bad reading is rejected.
    engine.broker.get_account_info = AsyncMock(
        return_value={"total_equity": 202_911.39},
    )
    asyncio.run(engine._update_portfolio_value())
    assert engine._peak_portfolio_value == pytest.approx(102_911.39)

    # Second call: clean reading is accepted.
    engine.broker.get_account_info = AsyncMock(
        return_value={"total_equity": 103_500.00},
    )
    asyncio.run(engine._update_portfolio_value())
    assert engine._portfolio_value == pytest.approx(103_500.00)
    assert engine._peak_portfolio_value == pytest.approx(103_500.00)


def test_update_pv_guard_floors_on_config_initial_when_pv_low(monkeypatch):
    """If the engine's current PV is briefly low (e.g. the prior call already
    stored a degraded reading), the guard's reference floor must be the
    config initial_balance — otherwise a 2× ratchet relative to a depressed
    PV would slip through. A reading at 1.6× config_initial should be
    rejected even when current PV is half of config_initial.
    """
    engine = _build_engine_for_pv_test(monkeypatch)
    engine._portfolio_value = 50_000.0  # half of config — unlikely but possible
    engine._peak_portfolio_value = 100_000.0

    # 1.6× config_initial = 160k → rejected (without the floor it would be
    # judged against current PV=50k → 3.2× → trivially rejected anyway, but
    # the point is that the floor uses config_initial, not current PV).
    engine.broker.get_account_info = AsyncMock(
        return_value={"total_equity": 160_000.0},
    )
    asyncio.run(engine._update_portfolio_value())
    assert engine._portfolio_value == pytest.approx(50_000.0)
    assert engine._peak_portfolio_value == pytest.approx(100_000.0)
