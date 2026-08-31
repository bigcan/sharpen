"""Live-engine bar_time wiring for daily-turnover reset (S506 Option B).

Confirms that ``LiveTradingEngine._trading_step_inner`` forwards its
``bar_time`` parameter into ``self.risk_manager.check(...)``. Without this
kwarg the manager falls back to the S498 Option A bar-interval-aware
call-count reset, which gets stretched across multiple calendar days under
upstream gating (signal_gate / funding_rate_gate / deadband). That stretch
is the production reproducer behind both the sg1-xauusd 7+h freeze and the
gmgp1-btc 5+h DAILY_TURNOVER violation storm.

We do NOT re-test reset semantics here — those are covered exhaustively in
``tests/crypto/test_crypto_risk_manager_turnover_reset.py``. These tests
cover only the engine→manager wiring contract.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from sharpen.crypto.mlops.crypto_risk_manager import (
    CryptoRiskConfig,
    CryptoRiskManager,
)


def _build_minimal_config() -> dict:
    """Minimal config sufficient for ``LiveTradingEngine`` __init__.

    ``dry_run=True`` short-circuits the broker call after the risk check, so
    we don't need to mock ``broker.execute_position_change`` and friends.
    """
    return {
        "exchange": {"asset": "BTC"},
        "features": {"scales": [15, 60, 240], "obs_mode": "flat"},
        "trading": {"deadband_threshold": 0.05, "initial_balance": 100_000.0},
        "agent": {"device": "cpu"},
        "dry_run": True,
        "safety": {
            "kill_file": "/tmp/finrl_test_kill_doesnotexist_s506",
            "emergency_flatten_on_error": False,
        },
        "risk": {"enabled": False},  # not used; we inject the manager directly
        "bar_clock": {"base_interval_minutes": 15},
    }


def _build_engine(config: dict, risk_manager, monkeypatch):
    """Construct a ``LiveTradingEngine`` with all non-target dependencies
    short-circuited. Yields an engine where the only meaningful behavior is
    the path from ``_trading_step_inner(bar_time)`` to ``risk_manager.check``.
    """
    from sharpen.crypto.live.live_engine import LiveTradingEngine

    monkeypatch.setenv("STRATEGY_NAME", "test-engine-s506")
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

    # Async helpers — no-op so flow reaches risk_manager.check.
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

    # Sync gates — all open.
    monkeypatch.setattr(engine, "_check_signal_gate", lambda: True)
    monkeypatch.setattr(engine, "_check_funding_rate_gate", lambda: True)
    monkeypatch.setattr(engine, "_check_obs_sanity", lambda obs: True)
    # Deterministic agent prediction; > deadband (0.05) so risk.check fires.
    monkeypatch.setattr(engine, "_predict", lambda obs: 0.5)
    # _log_step touches wandb / metrics state we don't care about here.
    monkeypatch.setattr(engine, "_log_step", lambda *a, **k: None)

    # Trading state defaults.
    engine._warmup_bars_remaining = 0
    engine._drift_tracker = None
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


def test_engine_passes_bar_time_to_risk_manager(monkeypatch):
    """Spy: assert ``risk_manager.check`` was called with the engine's
    ``bar_time`` kwarg.

    A regression here means somebody removed the ``bar_time=bar_time`` line
    from ``live_engine.py`` and Option B silently fell back to Option A —
    this is exactly the production failure mode the spec was written to
    close, so the wiring deserves a behavioral guard.
    """
    config = _build_minimal_config()
    risk_manager = MagicMock()
    risk_manager.check = MagicMock(return_value=(np.array([0.5]), []))
    risk_manager.rollback_last_turnover = MagicMock()

    engine = _build_engine(config, risk_manager, monkeypatch)

    bar_time = datetime(2026, 4, 29, 12, 30, tzinfo=timezone.utc)
    asyncio.run(engine._trading_step_inner(bar_time))

    assert risk_manager.check.call_count == 1, (
        f"expected exactly one risk_manager.check call, "
        f"got {risk_manager.check.call_count}"
    )
    kwargs = risk_manager.check.call_args.kwargs
    assert "bar_time" in kwargs, (
        f"risk_manager.check called without bar_time kwarg "
        f"(Option A fallback); kwargs={list(kwargs)}"
    )
    assert kwargs["bar_time"] == bar_time, (
        f"bar_time kwarg mismatch: expected {bar_time}, got {kwargs['bar_time']}"
    )


def test_engine_full_loop_resets_at_utc_midnight(monkeypatch):
    """Drive ``_trading_step_inner`` twice across a UTC midnight boundary
    with a real ``CryptoRiskManager``. The second call must reset the
    daily-turnover accumulator before accruing its own delta.

    This is the closest-to-production behavioral test of the bug fix:
    it exercises the engine wiring AND the manager's reset dispatch in
    one path, and would have failed pre-fix.
    """
    config = _build_minimal_config()

    risk_manager = CryptoRiskManager(CryptoRiskConfig(
        enabled=True,
        max_drawdown_pct=0.99,
        max_position_pct=1.0,
        max_net_short_exposure=-1.0,
        min_effective_bets=1.0,
        max_gross_exposure=10.0,
        daily_turnover_limit=4.0,
        funding_rate_alert=999.0,
        min_margin_reserve_pct=0.0,
        bar_interval_minutes=15,
    ))
    risk_manager.reset(initial_capital=100_000.0)

    engine = _build_engine(config, risk_manager, monkeypatch)

    # Pre-midnight bar (delta = 0.5 against starting position 0.0).
    pre_midnight = datetime(2026, 4, 29, 23, 45, tzinfo=timezone.utc)
    asyncio.run(engine._trading_step_inner(pre_midnight))
    assert risk_manager.state.daily_turnover_accumulated == pytest.approx(0.5)
    assert risk_manager.state.last_turnover_reset_date == "20260429"

    # Reset position so the next bar's delta is meaningful (dry_run path
    # advances _current_position to target after each "trade").
    engine._current_position = 0.0

    # Post-midnight bar: must clear the accumulator before accruing 0.5.
    post_midnight = datetime(2026, 4, 30, 0, 15, tzinfo=timezone.utc)
    asyncio.run(engine._trading_step_inner(post_midnight))
    assert risk_manager.state.last_turnover_reset_date == "20260430"
    assert risk_manager.state.daily_turnover_accumulated == pytest.approx(0.5)


def test_engine_emits_policy_target_position_to_log_step(monkeypatch):
    """S535 ADR-4: ``_log_step`` must receive ``policy_target_position`` as a
    kwarg from every post-predict call site. This is the wiring contract
    that lets WandB / drift detector see the gap between the agent's raw
    intent and the post-risk executed action.

    Pre-fix: the drift detector and WandB only saw ``target_position``,
    which was overwritten by the risk manager's clip; soft-throttle / hard-cap
    rewrites were invisible. Adding this kwarg makes the intent visible.
    """
    config = _build_minimal_config()
    risk_manager = MagicMock()
    risk_manager.check = MagicMock(return_value=(np.array([0.5]), []))
    risk_manager.rollback_last_turnover = MagicMock()

    engine = _build_engine(config, risk_manager, monkeypatch)

    log_calls: list = []
    monkeypatch.setattr(
        engine, "_log_step",
        lambda *a, **k: log_calls.append({"args": a, "kwargs": k}),
    )

    bar_time = datetime(2026, 4, 29, 12, 30, tzinfo=timezone.utc)
    asyncio.run(engine._trading_step_inner(bar_time))

    assert len(log_calls) == 1, f"expected one _log_step call, got {len(log_calls)}"
    kwargs = log_calls[0]["kwargs"]
    assert "policy_target_position" in kwargs, (
        f"_log_step called without policy_target_position kwarg "
        f"(observability regression); kwargs={list(kwargs)}"
    )
    # Engine's _predict was monkeypatched to return 0.5 in _build_engine.
    assert kwargs["policy_target_position"] == pytest.approx(0.5), (
        f"policy_target_position mismatch: expected 0.5, "
        f"got {kwargs['policy_target_position']}"
    )
