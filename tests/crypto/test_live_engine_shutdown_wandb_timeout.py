"""Shutdown timeout test for LiveTradingEngine._shutdown wandb path.

Regression for S489 (2026-04-21): `wandb.finish()` without a timeout kept
Python PID 1 alive for 2h+ after `=== TRADING STOPPED ===`, defeating
docker's `restart: unless-stopped` policy on gmgp1-gold. Fix runs
`wandb.finish()` in a daemon thread with a bounded join.

Tests assert that even when `wandb.finish()` hangs or raises, `_shutdown`
still completes within the configured bound.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import types

from sharpen.crypto.live.live_engine import LiveTradingEngine


def _make_engine_for_shutdown() -> LiveTradingEngine:
    """Minimal engine stub — only fields `_shutdown` touches are populated."""
    eng = object.__new__(LiveTradingEngine)

    eng.config = {"safety": {"flatten_on_stop": False}}
    eng._asset = "MGC"
    eng._current_position = 0.0
    eng._total_bars = 0
    eng._total_trades = 0
    eng._total_fees = 0.0
    eng._portfolio_value = 100_000.0
    eng._wandb_run = object()  # Truthy sentinel triggers wandb.finish path

    class _Broker:
        async def get_single_position(self, _asset):
            return 0.0

        async def close(self):
            pass

    class _Loader:
        _exchange = None

    eng.broker = _Broker()
    eng.loader = _Loader()
    return eng


def _install_fake_wandb(finish_fn) -> None:
    """Register a module named `wandb` whose `finish` is the given callable."""
    fake = types.ModuleType("wandb")
    fake.finish = finish_fn
    sys.modules["wandb"] = fake


def _compress_thread_join(cap_seconds: float = 1.0):
    """Cap `Thread.join` timeouts invoked inside `_shutdown` to keep tests fast.

    The real engine uses 30s; compressing to ~1s keeps the test under 3s.
    Returns a cleanup callable.
    """
    original_join = threading.Thread.join

    def _fast_join(self, timeout=None):
        if timeout is not None and timeout > cap_seconds:
            timeout = cap_seconds
        return original_join(self, timeout)

    threading.Thread.join = _fast_join
    return lambda: setattr(threading.Thread, "join", original_join)


def test_shutdown_completes_when_wandb_finish_hangs():
    """If wandb.finish() hangs, _shutdown must still return within the bound."""
    hang_evt = threading.Event()

    def _hanging_finish():
        hang_evt.wait(timeout=30.0)  # Much longer than join cap

    _install_fake_wandb(_hanging_finish)
    restore_join = _compress_thread_join(cap_seconds=1.0)
    try:
        eng = _make_engine_for_shutdown()
        start = time.monotonic()
        asyncio.run(eng._shutdown())
        elapsed = time.monotonic() - start

        assert elapsed < 3.0, (
            f"_shutdown() took {elapsed:.2f}s — wandb.finish() bound not enforced"
        )
    finally:
        restore_join()
        hang_evt.set()  # Release the daemon thread
        sys.modules.pop("wandb", None)


def test_shutdown_completes_when_wandb_finish_raises():
    """wandb.finish() raising an exception must not abort shutdown."""
    def _raising_finish():
        raise RuntimeError("wandb internal error")

    _install_fake_wandb(_raising_finish)
    try:
        eng = _make_engine_for_shutdown()
        asyncio.run(eng._shutdown())  # Must not raise
    finally:
        sys.modules.pop("wandb", None)
