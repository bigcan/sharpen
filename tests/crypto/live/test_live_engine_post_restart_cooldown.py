"""Post-restart cooldown wiring for SG-1-BTC (S542).

`decision_sg1_btc_skip_first_3_trades_s538.md` mandates a count-based
post-restart cooldown that skips the first N would-be trades to dodge
the OOD execution skew observed on sg1-btc's first-3 live trades
(perfect-fill +$12 vs live -$45 over 5 legs).

Contract under test:
  - `risk.post_restart_cooldown_bars: N` arms the cooldown at engine
    __init__; counter resets on every container start.
  - Counter decrements ONLY when an actual trade is about to fire
    (passed deadband + risk-manager re-deadband). Bar-side skips
    (signal_gate, deadband, warmup) do NOT decrement — the cooldown
    counts *trades*, not *bars*.
  - When counter > 0, the engine holds the current position and the
    skipped trade does NOT consume daily-turnover budget.
  - Counter is monotone non-increasing; once it hits zero it stays zero.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest


def _build_minimal_config(post_restart_cooldown: int = 0) -> dict:
    """Same wiring as test_live_engine_turnover_reset, plus the new key."""
    return {
        "exchange": {"asset": "BTC"},
        "features": {"scales": [15, 60, 240], "obs_mode": "flat"},
        "trading": {"deadband_threshold": 0.05, "initial_balance": 100_000.0},
        "agent": {"device": "cpu"},
        "dry_run": True,
        "safety": {
            "kill_file": "/tmp/finrl_test_kill_doesnotexist_s542",
            "emergency_flatten_on_error": False,
        },
        "risk": {
            "enabled": False,
            "post_restart_cooldown_bars": post_restart_cooldown,
        },
        "bar_clock": {"base_interval_minutes": 15},
    }


def _build_engine(config: dict, risk_manager, monkeypatch):
    """Wire the engine with all non-target deps short-circuited."""
    from finrl_pro_ds.crypto.live.live_engine import LiveTradingEngine

    monkeypatch.setenv("STRATEGY_NAME", "test-engine-s542-cooldown")
    monkeypatch.setenv("METRICS_PORT", "0")

    obs_builder = MagicMock()
    obs_builder.get_current_close = MagicMock(return_value=100.0)
    obs_builder.get_observation = MagicMock(return_value=np.zeros(10))
    obs_builder.update = MagicMock()

    engine = LiveTradingEngine(
        agent=MagicMock(),
        broker=MagicMock(),
        obs_builder=obs_builder,
        risk_manager=risk_manager,
        bar_clock=MagicMock(),
        loader=MagicMock(),
        config=config,
    )

    monkeypatch.setattr(engine, "_maybe_roll_contract", AsyncMock(return_value=None))
    monkeypatch.setattr(engine, "_update_portfolio_value", AsyncMock(return_value=None))
    monkeypatch.setattr(engine, "_reconcile_all", AsyncMock(return_value=None))
    monkeypatch.setattr(engine, "_check_daily_loss", AsyncMock(return_value=None))
    monkeypatch.setattr(engine, "_check_weekend_flatten", AsyncMock(return_value=False))
    monkeypatch.setattr(engine, "_update_funding_rate", AsyncMock(return_value=None))
    monkeypatch.setattr(
        engine, "_fetch_new_bars", AsyncMock(return_value=[{"close": 100.0}]),
    )
    monkeypatch.setattr(engine, "_check_intrabar_dd", AsyncMock(return_value=None))

    monkeypatch.setattr(engine, "_check_signal_gate", lambda: True)
    monkeypatch.setattr(engine, "_check_funding_rate_gate", lambda: True)
    monkeypatch.setattr(engine, "_check_obs_sanity", lambda obs: True)
    monkeypatch.setattr(engine, "_predict", lambda obs: 0.5)

    engine._warmup_bars_remaining = 0
    engine._drift_tracker = None
    engine._agreement_decay_tracker = None
    engine._prism_overlay = None
    engine._challenge_state_machine = None
    engine._current_position = 0.0
    engine._prev_close = 100.0
    engine._should_stop = False
    engine._portfolio_value = 100_000.0
    engine._peak_portfolio_value = 100_000.0
    engine._total_bars = 0
    engine._total_trades = 0
    engine._total_fees = 0.0

    return engine


def _passthrough_risk_manager():
    """Risk manager that passes every action through unmodified."""
    rm = MagicMock()
    rm.check = MagicMock(return_value=(np.array([0.5]), []))
    rm.rollback_last_turnover = MagicMock()
    return rm


def test_cooldown_arms_from_config(monkeypatch):
    """`risk.post_restart_cooldown_bars: 3` initializes counter to 3."""
    config = _build_minimal_config(post_restart_cooldown=3)
    engine = _build_engine(config, _passthrough_risk_manager(), monkeypatch)
    assert engine._post_restart_cooldown_initial == 3
    assert engine._post_restart_cooldown_remaining == 3


def test_cooldown_default_disabled(monkeypatch):
    """Default (key absent / 0) keeps cooldown disabled."""
    config = _build_minimal_config(post_restart_cooldown=0)
    engine = _build_engine(config, _passthrough_risk_manager(), monkeypatch)
    assert engine._post_restart_cooldown_remaining == 0


def test_cooldown_decrements_only_on_would_be_trades(monkeypatch):
    """Trade-side skip path: counter ticks once per would-be trade and the
    broker is never called while the cooldown is active.

    Three bars: each one passes signal/deadband/risk → would normally execute.
    With cooldown=3 ALL three should be held; counter ends at 0; broker
    `execute_position_change` (dry_run path: position update) never advances.
    """
    config = _build_minimal_config(post_restart_cooldown=3)
    risk_manager = _passthrough_risk_manager()
    engine = _build_engine(config, risk_manager, monkeypatch)

    log_calls: list = []
    monkeypatch.setattr(
        engine, "_log_step",
        lambda *a, **k: log_calls.append({"args": a, "kwargs": k}),
    )

    times = [
        datetime(2026, 4, 29, 12, 30, tzinfo=timezone.utc),
        datetime(2026, 4, 29, 12, 45, tzinfo=timezone.utc),
        datetime(2026, 4, 29, 13, 0, tzinfo=timezone.utc),
    ]
    for t in times:
        asyncio.run(engine._trading_step_inner(t))

    assert engine._post_restart_cooldown_remaining == 0, (
        "counter should reach 0 after 3 would-be trades"
    )
    # Dry-run path advances _current_position on actual trades; cooldown
    # holds it at 0.0 the whole time.
    assert engine._current_position == 0.0, (
        f"position must be held during cooldown, got {engine._current_position}"
    )
    # Each call must roll back the turnover consumed by risk_manager.check.
    assert risk_manager.rollback_last_turnover.call_count == 3
    # All 3 log calls should be the cooldown skip.
    for entry in log_calls:
        assert entry["kwargs"].get("skip_reason") == "post_restart_cooldown", (
            f"expected skip_reason=post_restart_cooldown, got "
            f"{entry['kwargs'].get('skip_reason')}"
        )


def test_cooldown_does_not_decrement_on_deadband(monkeypatch):
    """Deadband skip is NOT a would-be trade → counter must NOT tick.

    Set deadband=0.6 so the agent's target 0.5 never clears the gate; engine
    short-circuits at the first deadband check. Counter should be untouched.
    """
    config = _build_minimal_config(post_restart_cooldown=3)
    config["trading"]["deadband_threshold"] = 0.6
    risk_manager = _passthrough_risk_manager()
    engine = _build_engine(config, risk_manager, monkeypatch)
    # Refresh the engine's cached threshold (set in __init__).
    engine._deadband_threshold = 0.6

    bar_time = datetime(2026, 4, 29, 12, 30, tzinfo=timezone.utc)
    asyncio.run(engine._trading_step_inner(bar_time))

    assert engine._post_restart_cooldown_remaining == 3, (
        "deadband skip must NOT decrement the cooldown"
    )
    # risk_manager.check is the first place a real trade gets vetted; on
    # deadband-only paths it must not be called.
    risk_manager.check.assert_not_called()


def test_cooldown_does_not_decrement_on_signal_gate(monkeypatch):
    """Signal-gate closed: not a would-be trade, counter must NOT tick."""
    config = _build_minimal_config(post_restart_cooldown=3)
    risk_manager = _passthrough_risk_manager()
    engine = _build_engine(config, risk_manager, monkeypatch)
    # Slam the gate shut.
    monkeypatch.setattr(engine, "_check_signal_gate", lambda: False)

    bar_time = datetime(2026, 4, 29, 12, 30, tzinfo=timezone.utc)
    asyncio.run(engine._trading_step_inner(bar_time))

    assert engine._post_restart_cooldown_remaining == 3
    risk_manager.check.assert_not_called()


def test_cooldown_releases_after_n_trades(monkeypatch):
    """After exactly N=2 would-be trades the cooldown opens; the 3rd bar
    must execute (dry_run path: _current_position advances to target)."""
    config = _build_minimal_config(post_restart_cooldown=2)
    risk_manager = _passthrough_risk_manager()
    engine = _build_engine(config, risk_manager, monkeypatch)

    times = [
        datetime(2026, 4, 29, 12, 30, tzinfo=timezone.utc),
        datetime(2026, 4, 29, 12, 45, tzinfo=timezone.utc),
        datetime(2026, 4, 29, 13, 0, tzinfo=timezone.utc),
    ]
    for t in times:
        asyncio.run(engine._trading_step_inner(t))

    assert engine._post_restart_cooldown_remaining == 0
    # 3rd bar (cooldown=0) takes the dry_run trade path → position advances.
    assert engine._current_position == pytest.approx(0.5), (
        f"3rd bar should execute (dry_run path advances position to target); "
        f"got {engine._current_position}"
    )


def test_cooldown_counter_does_not_go_negative(monkeypatch):
    """Cooldown=0 → counter stays at 0 across many bars; never goes negative."""
    config = _build_minimal_config(post_restart_cooldown=0)
    risk_manager = _passthrough_risk_manager()
    engine = _build_engine(config, risk_manager, monkeypatch)

    for hour in range(5):
        t = datetime(2026, 4, 29, 12 + hour, 0, tzinfo=timezone.utc)
        asyncio.run(engine._trading_step_inner(t))
        # Reset position so each bar produces a fresh delta.
        engine._current_position = 0.0

    assert engine._post_restart_cooldown_remaining == 0
