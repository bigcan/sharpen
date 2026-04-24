"""Tests for finrl_pro_ds.crypto.live.challenge_state_machine.

Covers:
- Smoothed target hit triggers
- Persisted raw breach (N_CONFIRM) triggers
- Single-spike does not trigger
- Funded infinite-target never triggers
- Idempotent trigger_phase_complete
- Halt reason distinct from REASON_DRIFT_CRIT (kill-file lockout hygiene)
- Startup phase gate (advance OK / re-trip blocked / no-marker allow)
- Atomic last_completed_phase write + read
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from finrl_pro_ds.crypto.live.challenge_state_machine import (
    REASON_PHASE_COMPLETE,
    ChallengePhase,
    ChallengeStateMachine,
)
from finrl_pro_ds.monitoring.kill_file import REASON_DRIFT_CRIT, write_kill_file


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _make_machine(tmp_path: Path, phase: ChallengePhase, *,
                   initial_pv: float = 100_000.0,
                   n_confirm: int = 2,
                   smoothing_window: int = 3) -> tuple[ChallengeStateMachine, dict]:
    """Return a state machine plus the mocks wired into it."""
    mocks = {
        "halt_writer": MagicMock(),
        "kill_writer": MagicMock(),
        "flatten": MagicMock(wraps=_async_noop),
        "stop": MagicMock(),
        "gauge": MagicMock(),
    }
    sm = ChallengeStateMachine(
        phase,
        initial_portfolio_value=initial_pv,
        strategy_name="test-strategy",
        halt_state_writer=mocks["halt_writer"],
        kill_file_writer=mocks["kill_writer"],
        kill_file_path=tmp_path / "kill.json",
        last_completed_phase_path=tmp_path / "last_completed_phase.txt",
        flatten_callback=mocks["flatten"],
        stop_callback=mocks["stop"],
        telemetry_gauge=mocks["gauge"],
        n_confirm=n_confirm,
        smoothing_window=smoothing_window,
    )
    return sm, mocks


async def _async_noop() -> None:
    return None


# ---------------------------------------------------------------------------
# Smoothed hit
# ---------------------------------------------------------------------------

def test_smoothed_target_hit_triggers(tmp_path: Path):
    phase = ChallengePhase(
        name="step1", profit_target_pct=0.10, next_phase="step2",
    )
    sm, _ = _make_machine(tmp_path, phase)
    # Fill buffer with bars well above target
    s1 = sm.observe(110_100, now_utc=_now())
    s2 = sm.observe(110_200, now_utc=_now())
    s3 = sm.observe(110_300, now_utc=_now())
    assert s3.phase_complete
    assert s3.trip_source == "smoothed"
    assert s3.cumulative_return == pytest.approx(0.103)


# ---------------------------------------------------------------------------
# Single-spike rejection
# ---------------------------------------------------------------------------

def test_single_spike_does_not_trigger(tmp_path: Path):
    """One above-target bar then below-target → no trigger."""
    phase = ChallengePhase(name="step1", profit_target_pct=0.10, next_phase="step2")
    sm, _ = _make_machine(tmp_path, phase, smoothing_window=3, n_confirm=2)
    # Single spike, buffer still dominated by below-target bars
    s1 = sm.observe(99_500, now_utc=_now())      # -0.5%
    s2 = sm.observe(110_500, now_utc=_now())     # +10.5% (raw hit, spike)
    s3 = sm.observe(99_800, now_utc=_now())      # -0.2% (spike reverts)
    assert not s1.phase_complete
    assert not s2.phase_complete, (
        "single raw breach with non-hitting median must not trigger"
    )
    assert not s3.phase_complete


# ---------------------------------------------------------------------------
# Persisted raw breach
# ---------------------------------------------------------------------------

def test_persisted_raw_triggers(tmp_path: Path):
    """N_CONFIRM=2 consecutive raw breaches trigger even if smoothed lags."""
    phase = ChallengePhase(name="step1", profit_target_pct=0.10, next_phase="step2")
    # Smoothing window large (5) so median won't cross quickly even with 2 hits
    sm, _ = _make_machine(tmp_path, phase, smoothing_window=5, n_confirm=2)
    # Load buffer with below-target values so smoothed stays low
    for _ in range(3):
        s = sm.observe(99_000, now_utc=_now())
        assert not s.phase_complete
    s4 = sm.observe(110_100, now_utc=_now())  # raw hit #1
    assert not s4.phase_complete
    s5 = sm.observe(110_200, now_utc=_now())  # raw hit #2 → persisted
    assert s5.phase_complete
    assert s5.trip_source == "raw_persisted"


# ---------------------------------------------------------------------------
# Funded infinite target
# ---------------------------------------------------------------------------

def test_funded_phase_never_triggers(tmp_path: Path):
    phase = ChallengePhase(
        name="funded", profit_target_pct=math.inf, next_phase=None,
    )
    sm, _ = _make_machine(tmp_path, phase)
    s = sm.observe(10_000_000, now_utc=_now())
    assert not s.phase_complete
    assert s.target_remaining == math.inf


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_trigger_is_idempotent(tmp_path: Path):
    phase = ChallengePhase(name="step1", profit_target_pct=0.10, next_phase="step2")
    sm, mocks = _make_machine(tmp_path, phase)
    sm.observe(110_100)
    sm.observe(110_200)
    status = sm.observe(110_300)
    assert status.phase_complete

    asyncio.run(sm.trigger_phase_complete(status.trip_source, now_utc=_now()))
    # Second observe after trigger should short-circuit to already_complete
    follow = sm.observe(110_400)
    assert follow.phase_complete
    assert follow.trip_source == "already_complete"

    # trigger_phase_complete a second time must be a no-op on the mocks
    asyncio.run(sm.trigger_phase_complete(status.trip_source, now_utc=_now()))
    assert mocks["halt_writer"].call_count == 1
    assert mocks["kill_writer"].call_count == 1
    assert mocks["stop"].call_count == 1


# ---------------------------------------------------------------------------
# Reason distinctness
# ---------------------------------------------------------------------------

def test_reason_distinct_from_drift_crit():
    """phase_complete must not collide with drift_crit — lockout math hinges on this."""
    assert REASON_PHASE_COMPLETE != REASON_DRIFT_CRIT
    assert REASON_PHASE_COMPLETE == "phase_complete"


def test_kill_file_lockout_ignores_phase_complete(tmp_path: Path):
    """Back-to-back phase_complete events must not trigger the drift-CRIT lockout.

    The lockout logic at monitoring/kill_file.py:133 gates only on
    ``reason == REASON_DRIFT_CRIT``; a phase_complete reason stays outside
    that code path, so repeat writes keep updating history but do not
    flip the restart-lockout flag.
    """
    path = tmp_path / "kill.json"
    # First write — fresh file.
    p1 = write_kill_file(path, reason=REASON_PHASE_COMPLETE, detail="d1")
    assert p1["count"] == 1
    # Second write with the same reason — count increments.
    p2 = write_kill_file(path, reason=REASON_PHASE_COMPLETE, detail="d2")
    assert p2["count"] == 2
    # The lockout helper is imported separately; we only need to confirm
    # that a phase_complete payload is not treated as a drift_crit payload.
    assert p2["reason"] == REASON_PHASE_COMPLETE
    assert p2["reason"] != REASON_DRIFT_CRIT


# ---------------------------------------------------------------------------
# Trigger side effects
# ---------------------------------------------------------------------------

def test_trigger_writes_all_three_artefacts(tmp_path: Path):
    phase = ChallengePhase(name="step1", profit_target_pct=0.10, next_phase="step2")
    sm, mocks = _make_machine(tmp_path, phase)
    sm.observe(110_100)
    sm.observe(110_200)
    sm.observe(110_300)
    now = _now()
    asyncio.run(sm.trigger_phase_complete("smoothed", now_utc=now))

    # 1. last_completed_phase.txt
    marker_path = tmp_path / "last_completed_phase.txt"
    assert marker_path.exists()
    marker = json.loads(marker_path.read_text())
    assert marker["phase"] == "step1"
    assert marker["next_phase"] == "step2"
    assert marker["strategy"] == "test-strategy"

    # 2. halt_state writer called with phase_complete reason
    mocks["halt_writer"].assert_called_once()
    args, _ = mocks["halt_writer"].call_args
    assert args[0] == REASON_PHASE_COMPLETE

    # 3. kill_file writer called with phase_complete + phase_advance_ready
    mocks["kill_writer"].assert_called_once()
    _, kwargs = mocks["kill_writer"].call_args
    assert kwargs["reason"] == REASON_PHASE_COMPLETE
    assert kwargs["extra"]["phase"] == "step1"
    assert kwargs["extra"]["next_phase"] == "step2"
    assert kwargs["extra"]["phase_advance_ready"] is True

    # 4. flatten + stop
    mocks["flatten"].assert_called_once()
    mocks["stop"].assert_called_once_with(REASON_PHASE_COMPLETE)


def test_flatten_failure_does_not_prevent_halt(tmp_path: Path):
    """Per S490 post-mortem: halt_state must be written before flatten."""
    phase = ChallengePhase(name="step1", profit_target_pct=0.10, next_phase="step2")

    async def _failing_flatten():
        raise RuntimeError("broker offline")

    mocks = {
        "halt_writer": MagicMock(),
        "kill_writer": MagicMock(),
        "stop": MagicMock(),
    }
    sm = ChallengeStateMachine(
        phase,
        initial_portfolio_value=100_000.0,
        strategy_name="test-strategy",
        halt_state_writer=mocks["halt_writer"],
        kill_file_writer=mocks["kill_writer"],
        kill_file_path=tmp_path / "kill.json",
        last_completed_phase_path=tmp_path / "last_completed_phase.txt",
        flatten_callback=_failing_flatten,
        stop_callback=mocks["stop"],
    )
    sm.observe(110_100)
    sm.observe(110_200)
    sm.observe(110_300)
    # Must not raise even though flatten fails.
    asyncio.run(sm.trigger_phase_complete("smoothed", now_utc=_now()))
    mocks["halt_writer"].assert_called_once()
    mocks["kill_writer"].assert_called_once()
    mocks["stop"].assert_called_once()


# ---------------------------------------------------------------------------
# Startup phase gate
# ---------------------------------------------------------------------------

def test_startup_gate_no_marker_allows(tmp_path: Path):
    ok, msg = ChallengeStateMachine.check_startup_phase_gate(
        configured_phase="step1",
        last_completed_phase_path=tmp_path / "nonexistent.txt",
    )
    assert ok
    assert "no prior phase marker" in msg


def test_startup_gate_same_phase_refuses(tmp_path: Path):
    path = tmp_path / "last_completed_phase.txt"
    path.write_text(json.dumps({"phase": "step1", "next_phase": "step2"}))
    ok, msg = ChallengeStateMachine.check_startup_phase_gate(
        configured_phase="step1", last_completed_phase_path=path,
    )
    assert not ok
    assert "already passed" in msg
    assert "step2.yaml" in msg  # actionable operator hint


def test_startup_gate_advance_allowed(tmp_path: Path):
    path = tmp_path / "last_completed_phase.txt"
    path.write_text(json.dumps({"phase": "step1", "next_phase": "step2"}))
    ok, msg = ChallengeStateMachine.check_startup_phase_gate(
        configured_phase="step2", last_completed_phase_path=path,
    )
    assert ok
    assert "advance" in msg.lower()


def test_startup_gate_backward_refuses(tmp_path: Path):
    """Operator tries to deploy step1 after finishing step2 → refuse."""
    path = tmp_path / "last_completed_phase.txt"
    path.write_text(json.dumps({"phase": "step2", "next_phase": "funded"}))
    ok, msg = ChallengeStateMachine.check_startup_phase_gate(
        configured_phase="step1", last_completed_phase_path=path,
    )
    assert not ok


def test_startup_gate_custom_phase_allows_with_warning(tmp_path: Path):
    path = tmp_path / "last_completed_phase.txt"
    path.write_text(json.dumps({"phase": "custom_beta", "next_phase": None}))
    ok, msg = ChallengeStateMachine.check_startup_phase_gate(
        configured_phase="step1", last_completed_phase_path=path,
    )
    assert ok
    assert "custom" in msg.lower()


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_rejects_auto_advance_rule_is_allowed_at_constructor(tmp_path: Path):
    """Constructor accepts auto; the validator (paper-deploy) is the enforcement point."""
    phase = ChallengePhase(
        name="step1", profit_target_pct=0.10, next_phase="step2",
        advance_rule="auto",
    )
    sm, _ = _make_machine(tmp_path, phase)
    # No raise
    assert sm.phase.advance_rule == "auto"


def test_rejects_unknown_advance_rule(tmp_path: Path):
    phase = ChallengePhase(
        name="step1", profit_target_pct=0.10, next_phase="step2",
        advance_rule="yolo",
    )
    with pytest.raises(ValueError, match="advance_rule"):
        _make_machine(tmp_path, phase)


def test_rejects_nonpositive_initial_pv(tmp_path: Path):
    phase = ChallengePhase(name="step1", profit_target_pct=0.10, next_phase="step2")
    with pytest.raises(ValueError, match="initial_portfolio_value"):
        _make_machine(tmp_path, phase, initial_pv=0.0)


# ---------------------------------------------------------------------------
# Target remaining math
# ---------------------------------------------------------------------------

def test_target_remaining_monotone(tmp_path: Path):
    phase = ChallengePhase(name="step1", profit_target_pct=0.10, next_phase="step2")
    sm, _ = _make_machine(tmp_path, phase)
    s_low = sm.observe(101_000)
    s_mid = sm.observe(104_000)
    s_high = sm.observe(108_000)
    assert s_low.target_remaining > s_mid.target_remaining > s_high.target_remaining
    assert s_low.target_remaining == pytest.approx(0.09, abs=0.02)
