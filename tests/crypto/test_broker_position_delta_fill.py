"""Phase 2 tests: position-delta fill detection in broker cancel path.

S527-cont. Bybit demo's ``fetch_order`` returns Python ``None`` after a
cancel even when the limit fully filled at the exchange, which makes the
broker's fetch_order-based fill detection think nothing filled. The
fallback then fires a Market order for the same quantity, doubling the
intended position. Phase 2 trusts position deltas instead.

Reference: project_bybit_phase2_handoff.md
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from finrl_pro_ds.crypto.execution.exchange_perp_broker import (
    ExchangePerpBroker,
    OrderResult,
)


def _make_broker() -> ExchangePerpBroker:
    """Bare broker with a Mock exchange — no real network."""
    broker = ExchangePerpBroker(
        exchange="bybit",
        testnet=False,
        api_key="test-key",
        api_secret="test-secret",
    )
    broker._exchange = MagicMock()
    broker._exchange.fetch_balance = AsyncMock(
        return_value={"total": {"USDT": 50_000.0}},
    )
    broker._exchange.fetch_order = AsyncMock(return_value=None)
    return broker


# ---------------------------------------------------------------------------
# Unit tests on _detect_fill_after_cancel
# ---------------------------------------------------------------------------


def test_position_delta_detects_filled_limit_when_fetch_order_none():
    """fetch_order is None but pos delta = +0.9 weight on $50k @ $80k mid →
    already_filled ≈ 0.5625 BTC. fetch_order should NOT be consulted on the
    success path."""
    broker = _make_broker()
    broker.get_single_position = AsyncMock(return_value=0.9)  # post-cancel

    filled = asyncio.run(
        broker._detect_fill_after_cancel(
            order_id="abc-123", symbol="BTC/USDT:USDT", asset="BTC",
            side="buy", mid_price=80_000.0, pos_before_weight=0.0,
        ),
    )

    expected = abs(0.9 - 0.0) * 50_000.0 / 80_000.0  # 0.5625
    assert abs(filled - expected) < 1e-9
    broker._exchange.fetch_order.assert_not_called()


def test_position_delta_detects_partial_when_fetch_order_none():
    """Pos delta is 40% of an intended +1.0 weight delta — already_filled
    reflects the actual fill, leaving the residual for the market fallback."""
    broker = _make_broker()
    broker.get_single_position = AsyncMock(return_value=0.4)  # post-cancel

    filled = asyncio.run(
        broker._detect_fill_after_cancel(
            order_id="abc-123", symbol="BTC/USDT:USDT", asset="BTC",
            side="buy", mid_price=80_000.0, pos_before_weight=0.0,
        ),
    )

    expected = abs(0.4 - 0.0) * 50_000.0 / 80_000.0  # 0.25
    assert abs(filled - expected) < 1e-9


def test_position_delta_falls_back_to_fetch_order_on_snapshot_failure():
    """If get_single_position raises on the post-cancel snapshot, fall back
    to fetch_order (original behaviour). Returns whatever fetch_order
    reports — a populated dict here, to verify the fallback path actually
    runs."""
    broker = _make_broker()
    broker.get_single_position = AsyncMock(side_effect=RuntimeError("ws disconnect"))
    broker._exchange.fetch_order = AsyncMock(return_value={"filled": 0.123})

    filled = asyncio.run(
        broker._detect_fill_after_cancel(
            order_id="abc-123", symbol="BTC/USDT:USDT", asset="BTC",
            side="buy", mid_price=80_000.0, pos_before_weight=0.0,
        ),
    )

    assert filled == 0.123
    broker._exchange.fetch_order.assert_awaited_once()


def test_position_delta_falls_back_when_balance_returns_zero():
    """fetch_balance race / empty response → total_equity=0 → fall back
    to fetch_order. Phase 1 reconcile is the safety net for over-fills."""
    broker = _make_broker()
    broker.get_single_position = AsyncMock(return_value=0.5)
    broker._exchange.fetch_balance = AsyncMock(return_value={})
    broker._exchange.fetch_order = AsyncMock(return_value=None)  # Bybit demo

    filled = asyncio.run(
        broker._detect_fill_after_cancel(
            order_id="abc-123", symbol="BTC/USDT:USDT", asset="BTC",
            side="buy", mid_price=80_000.0, pos_before_weight=0.0,
        ),
    )

    # fetch_order returned None → 0.0; the over-fill case is impossible
    # here because already_filled is undercounted, not overcounted.
    assert filled == 0.0
    broker._exchange.fetch_order.assert_awaited_once()


def test_position_delta_zero_for_sell_when_pos_unchanged():
    """A sell whose position is unchanged → zero fill. Market fallback
    will then trade the full sell quantity."""
    broker = _make_broker()
    broker.get_single_position = AsyncMock(return_value=0.5)  # unchanged

    filled = asyncio.run(
        broker._detect_fill_after_cancel(
            order_id="abc-123", symbol="BTC/USDT:USDT", asset="BTC",
            side="sell", mid_price=80_000.0, pos_before_weight=0.5,
        ),
    )

    assert filled == 0.0


def test_position_delta_zero_when_direction_opposite():
    """If the post-cancel position moved opposite to the order side
    (e.g. another path flattened), don't credit the delta to the
    cancelled limit. Treat as zero fill so the market fallback can fire
    and Phase 1 reconcile catches any over-fill within 1 bar."""
    broker = _make_broker()
    broker.get_single_position = AsyncMock(return_value=-0.2)

    filled = asyncio.run(
        broker._detect_fill_after_cancel(
            order_id="abc-123", symbol="BTC/USDT:USDT", asset="BTC",
            side="buy", mid_price=80_000.0, pos_before_weight=0.0,
        ),
    )

    assert filled == 0.0


# ---------------------------------------------------------------------------
# Integration: _execute_order limit branch end-to-end
# ---------------------------------------------------------------------------


def test_no_double_trade_on_bybit_demo_simulation():
    """End-to-end: simulate Bybit demo where the limit fills at the exchange,
    _wait_for_fill returns None (status was never observed 'closed'), and
    fetch_order returns None on the cancel-path query. Phase 2 must detect
    the fill via position delta and skip the market fallback. Without
    Phase 2, a market order with the full quantity would have fired (the
    pre-Phase-2 double-trade)."""
    broker = _make_broker()

    # Order setup: BUY 0.5 BTC at $80k, $50k account.
    broker._exchange.fetch_ticker = AsyncMock(
        return_value={"bid": 79_999, "ask": 80_001, "last": 80_000},
    )
    broker._exchange.amount_to_precision = MagicMock(side_effect=lambda s, q: f"{q:.3f}")
    broker._exchange.price_to_precision = MagicMock(side_effect=lambda s, p: f"{p:.2f}")
    broker._exchange.market = MagicMock(
        return_value={"limits": {"amount": {"min": 0.001}}},
    )

    create_order_calls: list[dict] = []

    async def _create_order(**kwargs):
        create_order_calls.append(kwargs)
        return {"id": f"order-{len(create_order_calls)}", "status": "open"}

    broker._exchange.create_order = AsyncMock(side_effect=_create_order)
    broker._exchange.cancel_order = AsyncMock(return_value=None)
    broker._exchange.fetch_order = AsyncMock(return_value=None)

    # _wait_for_fill returns None (Bybit demo can't observe 'closed').
    async def _stub_wait_for_fill(*_a, **_kw):
        return None
    broker._wait_for_fill = _stub_wait_for_fill

    # Position snapshots: 0.0 before, 0.8 after (the limit fully filled).
    broker.get_single_position = AsyncMock(side_effect=[0.0, 0.8])

    result = asyncio.run(broker._execute_order(
        asset="BTC", symbol="BTC/USDT:USDT", side="buy",
        notional=40_000.0,  # 0.5 BTC at $80k
    ))

    # Exactly one create_order — the limit. No market fallback.
    assert len(create_order_calls) == 1
    assert create_order_calls[0]["type"] == "limit"
    assert isinstance(result, OrderResult)
    # already_filled ≈ 0.8 * 50000 / 80000 = 0.5 BTC → remaining_qty ≈ 0
    # → partial-fill return path.
    assert result.status == "partial"
    expected_filled = 0.8 * 50_000.0 / 80_000.0
    assert abs(result.filled_quantity - expected_filled) < 1e-6
