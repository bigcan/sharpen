"""Regression tests for the S490/S491/FIND-02 halt-persistence paths.

Covers three safety knobs that previously crash-looped the engine:

    1. S490 balance-mismatch startup guard — now writes halt_state and exits
       cleanly (FIND-02), so the pre-broker halt gate sleeps to midnight on
       restart instead of re-hitting the same mismatch under
       ``restart: unless-stopped``.

    2. S491 position_mismatch during reconciliation — persists halt_state at
       ``_reconcile_all`` so Docker restart can't thrash broker connects.

    3. FIND-01 daily-loss ``limit <= 0`` disables the check (Velotrade
       2-step configs previously tripped on the first red bar).

The engine is instantiated via ``object.__new__`` bypassing ``__init__``;
only the fields exercised by each path are set. Side-effect methods are
stubbed so assertions can target pure logic.
"""

from __future__ import annotations

import asyncio
import collections
from datetime import datetime, timezone
from pathlib import Path

from finrl_pro_ds.crypto.live.live_engine import LiveTradingEngine


# ---------------------------------------------------------------------------
# FIND-01: daily-loss disabled when limit <= 0
# ---------------------------------------------------------------------------

def _make_daily_loss_engine(limit: float) -> LiveTradingEngine:
    eng = object.__new__(LiveTradingEngine)
    eng._portfolio_value = 10_000.0
    eng._daily_start_value = 10_000.0
    eng._last_daily_reset_date = None
    eng._max_daily_loss_pct = limit
    eng._pv_buffer = collections.deque(maxlen=3)
    eng._consecutive_raw_breach_count = 0
    eng._trip_fired = False
    eng._halt_detail = None

    async def _fake_flatten():
        eng._trip_fired = True

    def _fake_write_halt_state(reason, detail, now_utc):
        eng._halt_detail = (reason, detail)

    def _fake_request_stop(reason):
        pass

    eng._emergency_flatten = _fake_flatten
    eng._write_halt_state = _fake_write_halt_state
    eng._request_stop = _fake_request_stop
    return eng


def test_daily_loss_disabled_at_zero():
    """FIND-01: limit=0 must treat the check as disabled, not trip on any loss."""
    async def run():
        eng = _make_daily_loss_engine(limit=0.0)
        t = datetime(2026, 4, 22, 10, 0, tzinfo=timezone.utc)

        # A steady drawdown that would trip at any positive limit must pass
        # through silently when the check is disabled.
        for i, pv in enumerate([10_000.0, 9_500.0, 9_000.0, 8_000.0, 5_000.0]):
            eng._portfolio_value = pv
            await eng._check_daily_loss(t.replace(minute=15 * i % 60, hour=10 + i // 4))

        assert eng._trip_fired is False
        assert eng._halt_detail is None
    asyncio.run(run())


def test_daily_loss_disabled_negative():
    """Negative limit (a config typo) must also disable the check, not trip."""
    async def run():
        eng = _make_daily_loss_engine(limit=-0.05)
        t = datetime(2026, 4, 22, 10, 0, tzinfo=timezone.utc)

        eng._portfolio_value = 9_000.0  # -10%
        await eng._check_daily_loss(t)
        assert eng._trip_fired is False
    asyncio.run(run())


def test_daily_loss_disabled_still_rotates_daily_anchor():
    """Disabled check must still roll the daily anchor on UTC day change so
    reporting metrics stay correct."""
    async def run():
        eng = _make_daily_loss_engine(limit=0.0)
        day1 = datetime(2026, 4, 22, 23, 45, tzinfo=timezone.utc)
        day2 = datetime(2026, 4, 23, 0, 0, tzinfo=timezone.utc)

        eng._portfolio_value = 10_000.0
        await eng._check_daily_loss(day1)
        assert eng._daily_start_value == 10_000.0

        eng._portfolio_value = 9_200.0
        await eng._check_daily_loss(day2)
        assert eng._daily_start_value == 9_200.0
        assert eng._last_daily_reset_date == day2.date()
    asyncio.run(run())


# ---------------------------------------------------------------------------
# S491: position_mismatch persists halt_state before _request_stop
# ---------------------------------------------------------------------------

class _StubBroker:
    def __init__(self, exchange_pos: float, equity: float):
        self._exchange_pos = exchange_pos
        self._equity = equity

    async def get_single_position(self, asset):
        return self._exchange_pos

    async def get_account_info(self):
        return {"total_equity": self._equity}


class _StubMetrics:
    def __init__(self):
        self.calls = []

    def update_reconciliation(self, **kwargs):
        self.calls.append(kwargs)


def _make_reconcile_engine(internal_pos: float, exchange_pos: float) -> LiveTradingEngine:
    eng = object.__new__(LiveTradingEngine)
    eng._asset = "XAUUSD"
    eng._current_position = internal_pos
    eng._portfolio_value = 100_000.0
    eng._reconciliation_halt_pct = 0.15
    eng._reconciliation_warn_pct = 0.05
    eng._pv_divergence_warn_pct = 0.03
    eng._reconcile_interval = 1
    eng._last_reconcile_bar = 0
    eng._total_bars = 1
    eng._should_stop = False
    eng.broker = _StubBroker(exchange_pos=exchange_pos, equity=100_000.0)
    eng._metrics = _StubMetrics()
    eng._halt_writes = []
    eng._stop_reasons = []

    def _fake_check_broker_alive():
        return True

    def _fake_write_halt_state(reason, detail, now_utc):
        eng._halt_writes.append((reason, detail))

    def _fake_request_stop(reason):
        eng._stop_reasons.append(reason)
        eng._should_stop = True

    eng._check_broker_alive = _fake_check_broker_alive
    eng._write_halt_state = _fake_write_halt_state
    eng._request_stop = _fake_request_stop
    return eng


def test_position_mismatch_persists_halt_state_before_request_stop():
    """S491 regression: reconciliation divergence > halt_pct must call
    _write_halt_state BEFORE _request_stop so a Docker restart enters the
    sleep-to-midnight loop instead of thrashing broker reconnects."""
    async def run():
        eng = _make_reconcile_engine(internal_pos=0.94, exchange_pos=0.09)
        bar_time = datetime(2026, 4, 22, 2, 0, tzinfo=timezone.utc)

        await eng._reconcile_all(bar_time)

        assert len(eng._halt_writes) == 1
        reason, detail = eng._halt_writes[0]
        assert reason == "position_mismatch"
        assert "internal=" in detail and "exchange=" in detail
        assert "divergence=" in detail
        assert eng._stop_reasons == ["position_mismatch"]
    asyncio.run(run())


def test_position_warn_zone_does_not_halt():
    """Warn-level divergence (5% < d < 15%) trusts exchange — no halt."""
    async def run():
        eng = _make_reconcile_engine(internal_pos=0.50, exchange_pos=0.43)  # 7%
        bar_time = datetime(2026, 4, 22, 2, 0, tzinfo=timezone.utc)

        await eng._reconcile_all(bar_time)

        assert eng._halt_writes == []
        assert eng._stop_reasons == []
        # Internal position gets pulled to exchange value on warn zone.
        assert abs(eng._current_position - 0.43) < 1e-9
    asyncio.run(run())


# ---------------------------------------------------------------------------
# S490 + FIND-02: balance_mismatch must persist halt_state (not raise)
# ---------------------------------------------------------------------------

def test_write_halt_state_balance_mismatch_contract(tmp_path: Path):
    """FIND-02 fix: balance_mismatch writes a halt_state with halted_until at
    next UTC midnight so the pre-broker halt gate sleeps on restart instead
    of crashing the container again. This asserts the contract that
    _write_halt_state records the reason + halted_until correctly."""
    eng = object.__new__(LiveTradingEngine)
    eng._halt_state_file = tmp_path / "risk_state.json"
    eng._strategy_name = "test-strategy"
    eng._daily_start_value = 100_000.0
    eng._portfolio_value = 30_000.0

    now = datetime(2026, 4, 22, 2, 0, tzinfo=timezone.utc)
    eng._write_halt_state(
        reason="balance_mismatch",
        detail="broker_equity=30000 config_ib=100000 drift=0.7000",
        now_utc=now,
    )

    import json
    data = json.loads(eng._halt_state_file.read_text(encoding="utf-8"))
    assert data["reason"] == "balance_mismatch"
    assert data["halted_until"].startswith("2026-04-23T00:00:00")
    assert "drift=0.7000" in data["detail"]


# ---------------------------------------------------------------------------
# FIND-01 parity (S498-cont): intrabar_dd projection honors limit <= 0
# ---------------------------------------------------------------------------

def _make_intrabar_engine(limit: float):
    """Engine stub for _check_intrabar_dd. Carries a tiny stub obs_builder
    that returns adverse OHLC the projection would otherwise trip on."""
    eng = object.__new__(LiveTradingEngine)
    eng._intrabar_dd_enabled = True
    eng._max_daily_loss_pct = limit
    eng._current_position = -0.27  # short, like sg1-btc bar 1
    eng._portfolio_value = 4_999.26
    eng._daily_start_value = 4_999.26
    eng._trip_fired = False
    eng._halt_detail = None

    class _StubObs:
        def get_current_hl(self):
            return 77502.30, 77502.30  # high == low — bar that yields ~0 adverse
        def get_current_close(self):
            return 77502.30
    eng.obs_builder = _StubObs()

    async def _fake_flatten():
        eng._trip_fired = True
    def _fake_write_halt_state(reason, detail, now_utc):
        eng._halt_detail = (reason, detail)
    def _fake_request_stop(reason):
        pass
    eng._emergency_flatten = _fake_flatten
    eng._write_halt_state = _fake_write_halt_state
    eng._request_stop = _fake_request_stop
    return eng


def test_intrabar_dd_disabled_at_zero():
    """FIND-01 parity: max_daily_loss_pct=0 must disable intrabar projection.

    Regression: SG-1-BTC paper deploy 2026-04-26 halted on bar 1 because
    `projected_return < -0.0` evaluates True for any tiny negative value.
    """
    async def run():
        eng = _make_intrabar_engine(limit=0.0)
        t = datetime(2026, 4, 26, 0, 6, tzinfo=timezone.utc)
        await eng._check_intrabar_dd(t)
        assert eng._trip_fired is False
        assert eng._halt_detail is None
    asyncio.run(run())


def test_intrabar_dd_disabled_negative():
    """Negative limit (config typo) must also disable, mirroring closing check."""
    async def run():
        eng = _make_intrabar_engine(limit=-0.05)
        t = datetime(2026, 4, 26, 0, 6, tzinfo=timezone.utc)
        await eng._check_intrabar_dd(t)
        assert eng._trip_fired is False
    asyncio.run(run())


def test_intrabar_dd_still_trips_when_enabled():
    """Sanity: real adverse excursion still trips when limit is positive."""
    async def run():
        eng = _make_intrabar_engine(limit=0.05)
        # Force an adverse short by widening high vs close so projection
        # crosses the 5% daily-loss limit. Short pos -1.0 of pv against a
        # 6% adverse high → projected_return ≈ -6% < -5%.
        eng._current_position = -1.0
        class _BadObs:
            def get_current_hl(self):
                return 82152.4, 77502.30  # high 6% above close
            def get_current_close(self):
                return 77502.30
        eng.obs_builder = _BadObs()
        t = datetime(2026, 4, 26, 0, 6, tzinfo=timezone.utc)
        await eng._check_intrabar_dd(t)
        assert eng._trip_fired is True
        assert eng._halt_detail[0] == "intrabar_dd_projection"
    asyncio.run(run())
