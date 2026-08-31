"""Regression tests for the S502 ambiguous-execution reconcile path.

After a ccxt order error whose outcome is unknown (e.g. binance ``-1007
"Send status unknown; execution status unknown"``), the order may have
filled at the exchange while the engine assumes failure. The fix forces
an immediate broker reconcile so the agent doesn't act on stale internal
state on the next bar.

Incident: gmgp1-btc 2026-04-27 08:45 UTC. ``-1007`` timeout at 08:00 left
the engine with internal=-0.7276 while the exchange position had flipped
to +0.5101; periodic reconcile (every 4 bars) didn't catch it until 08:45,
yielding divergence 1.24.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from sharpen.crypto.live.live_engine import LiveTradingEngine


def _make_engine(reconcile_interval: int = 4, total_bars: int = 17):
    """Minimal engine instance for testing the post-error reconcile path."""
    eng = object.__new__(LiveTradingEngine)
    eng._reconcile_interval = reconcile_interval
    eng._last_reconcile_bar = 16  # last reconcile was 1 bar ago
    eng._total_bars = total_bars

    calls: list[datetime] = []

    async def _fake_reconcile(bar_time: datetime) -> None:
        calls.append(bar_time)
        # Mimic real ``_reconcile_all`` updating its own gate.
        eng._last_reconcile_bar = eng._total_bars

    eng._reconcile_all = _fake_reconcile
    return eng, calls


# ---------------------------------------------------------------------------
# Detection — pure classifier
# ---------------------------------------------------------------------------

def test_detect_dash_1007_binance_message():
    """The exact ccxt message that fired the S502 incident."""
    msg = (
        'binance {"code":-1007,"msg":"Timeout waiting for response from '
        'backend server. Send status unknown; execution status unknown."}'
    )
    assert LiveTradingEngine._is_ambiguous_execution_error(msg) is True


def test_detect_send_status_unknown_alone():
    assert LiveTradingEngine._is_ambiguous_execution_error(
        "Send status unknown",
    ) is True


def test_detect_execution_status_unknown_alone():
    assert LiveTradingEngine._is_ambiguous_execution_error(
        "execution status unknown",
    ) is True


def test_detect_case_insensitive():
    assert LiveTradingEngine._is_ambiguous_execution_error(
        "SEND STATUS UNKNOWN",
    ) is True


def test_does_not_detect_margin_insufficient():
    """``-2019 Margin is insufficient`` is a real error, not ambiguous."""
    msg = 'binance {"code":-2019,"msg":"Margin is insufficient."}'
    assert LiveTradingEngine._is_ambiguous_execution_error(msg) is False


def test_does_not_detect_unrelated_errors():
    for msg in (
        "ConnectionError: HTTPSConnectionPool",
        'binance {"code":-1021,"msg":"Timestamp for this request is outside of the recvWindow."}',
        "ValueError: invalid order size",
        "",
    ):
        assert LiveTradingEngine._is_ambiguous_execution_error(msg) is False


def test_substring_overlap_is_acceptable():
    """A free-form message containing -1007 (e.g. PnL like '-1007.50') would
    match. This is acceptable because the consequence is one extra reconcile
    call (idempotent broker query). Documenting the trade-off here so future
    refactors don't tighten the matcher without considering the cost.
    """
    assert LiveTradingEngine._is_ambiguous_execution_error(
        "current pnl: -1007.50 USDT",
    ) is True


# S527-cont: Bybit demo's ``fetch_order`` can return Python ``None`` instead
# of raising. The broker swallows it, fires a Market fallback that double-
# fills, and the engine surfaces the fault as a downstream format-string
# crash with no recognizable broker error string. The classifier must also
# treat those crashes as ambiguous so reconcile fires within 1 bar.

def test_detect_unsupported_format_string():
    """``f"{None:.6f}"`` raises this when ``OrderResult.filled_quantity`` is None."""
    assert LiveTradingEngine._is_ambiguous_execution_error(
        "unsupported format string passed to NoneType.__format__",
    ) is True


def test_detect_could_not_fetch_order_status():
    """Bybit demo: ``_execute_order`` cancel-path warns this when fetch_order
    returns None — propagated up as the exception message in some paths."""
    assert LiveTradingEngine._is_ambiguous_execution_error(
        "Could not fetch order status for abc-123, assuming unfilled",
    ) is True


def test_detect_nonetype_attribute_error():
    """``order["status"]`` on None raises ``TypeError: 'NoneType' object is
    not subscriptable``; an attribute access raises a similar ``NoneType``
    AttributeError. Both must classify as ambiguous."""
    assert LiveTradingEngine._is_ambiguous_execution_error(
        "AttributeError: 'NoneType' object has no attribute 'status'",
    ) is True
    assert LiveTradingEngine._is_ambiguous_execution_error(
        "TypeError: 'NoneType' object is not subscriptable",
    ) is True


# ---------------------------------------------------------------------------
# Trigger — async helper
# ---------------------------------------------------------------------------

def test_ambiguous_error_triggers_immediate_reconcile():
    """The S502 fix: ambiguous error → immediate broker reconcile."""
    eng, calls = _make_engine(reconcile_interval=4, total_bars=17)
    bar_time = datetime(2026, 4, 27, 8, 0, tzinfo=timezone.utc)

    msg = (
        'binance {"code":-1007,"msg":"Timeout waiting for response from '
        'backend server. Send status unknown; execution status unknown."}'
    )

    triggered = asyncio.run(
        eng._force_reconcile_if_ambiguous(msg, bar_time),
    )
    assert triggered is True
    assert len(calls) == 1
    assert calls[0] == bar_time


def test_non_ambiguous_error_skips_reconcile():
    """``-2019 Margin is insufficient`` must NOT trigger forced reconcile."""
    eng, calls = _make_engine()
    bar_time = datetime(2026, 4, 27, 8, 30, tzinfo=timezone.utc)

    triggered = asyncio.run(
        eng._force_reconcile_if_ambiguous(
            'binance {"code":-2019,"msg":"Margin is insufficient."}',
            bar_time,
        ),
    )
    assert triggered is False
    assert calls == []


def test_reset_math_forces_reconcile_gate_open():
    """``_last_reconcile_bar`` reset must put ``bars_since >= interval``.

    ``_reconcile_all`` would skip if ``bars_since < _reconcile_interval``.
    Verifies the off-by-one is correct.
    """
    eng, _calls = _make_engine(reconcile_interval=4, total_bars=17)
    bar_time = datetime(2026, 4, 27, 8, 0, tzinfo=timezone.utc)

    asyncio.run(eng._force_reconcile_if_ambiguous("Send status unknown", bar_time))

    # After the fake reconcile runs, _last_reconcile_bar should equal _total_bars.
    # If our reset math were wrong (e.g. forgot to subtract interval), the
    # fake reconcile would never have been invoked and _last_reconcile_bar
    # would remain at 16.
    assert eng._last_reconcile_bar == eng._total_bars  # == 17


def test_reconcile_failure_does_not_propagate():
    """Forced reconcile failures must be logged, not raised — caller is in
    the order-execution exception handler and a propagating exception would
    skip the emergency-flatten path that follows.
    """
    eng, _calls = _make_engine()
    bar_time = datetime(2026, 4, 27, 8, 0, tzinfo=timezone.utc)

    async def _raising_reconcile(_bar_time):
        raise RuntimeError("broker connection lost")

    eng._reconcile_all = _raising_reconcile

    # Must complete without raising
    triggered = asyncio.run(
        eng._force_reconcile_if_ambiguous("-1007", bar_time),
    )
    assert triggered is True


def test_returns_false_for_empty_string():
    """Defensive: empty error string (unlikely but possible) returns False."""
    eng, calls = _make_engine()
    triggered = asyncio.run(
        eng._force_reconcile_if_ambiguous(
            "", datetime(2026, 4, 27, tzinfo=timezone.utc),
        ),
    )
    assert triggered is False
    assert calls == []
