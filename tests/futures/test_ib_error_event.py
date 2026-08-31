"""Unit tests for IBFuturesBroker errorEvent handling (S488 fix).

Verifies that IB order-rejection errorEvents (codes 200-204) are captured
and surfaced as IBOrderRejected exceptions from _place_order, so the
engine counts them toward consecutive_execution_errors instead of
silently returning status="failed".
"""

import pytest

from sharpen.futures.execution.ib_futures_broker import (
    IB_HMDS_CODES,
    IB_ORDER_REJECTION_CODES,
    IBFuturesBroker,
    IBOrderRejected,
)


def _new_broker() -> IBFuturesBroker:
    return IBFuturesBroker()


def test_rejection_codes_cover_200_to_204():
    assert IB_ORDER_REJECTION_CODES == frozenset({200, 201, 202, 203, 204})


def test_hmds_codes_cover_321_322():
    assert IB_HMDS_CODES == frozenset({321, 322})


def test_on_ib_error_stores_order_rejection_by_req_id():
    broker = _new_broker()
    broker._on_ib_error(reqId=42, errorCode=201, errorString="Margin check failed")
    assert broker._ib_order_rejections[42] == (201, "Margin check failed")


def test_on_ib_error_ignores_negative_req_id():
    broker = _new_broker()
    broker._on_ib_error(reqId=-1, errorCode=201, errorString="Server not connected")
    assert broker._ib_order_rejections == {}


def test_on_ib_error_does_not_store_hmds_codes():
    broker = _new_broker()
    broker._on_ib_error(reqId=7, errorCode=322, errorString="HMDS no data")
    assert broker._ib_order_rejections == {}


def test_on_ib_error_does_not_store_unrelated_codes():
    broker = _new_broker()
    broker._on_ib_error(reqId=7, errorCode=2104, errorString="Farm connection OK")
    assert broker._ib_order_rejections == {}


class _FakeOrder:
    def __init__(self, order_id: int):
        self.orderId = order_id


class _FakeTrade:
    def __init__(self, order_id: int):
        self.order = _FakeOrder(order_id)


def test_raise_if_rejected_raises_on_pending_rejection():
    broker = _new_broker()
    broker._on_ib_error(reqId=99, errorCode=201, errorString="Margin")
    with pytest.raises(IBOrderRejected) as exc_info:
        broker._raise_if_rejected(_FakeTrade(99), filled=False)
    assert exc_info.value.error_code == 201
    assert exc_info.value.order_id == 99
    # Rejection drained so subsequent orders aren't poisoned
    assert 99 not in broker._ib_order_rejections


def test_raise_if_rejected_no_op_when_no_rejection():
    broker = _new_broker()
    broker._raise_if_rejected(_FakeTrade(99), filled=False)  # should not raise


def test_raise_if_rejected_swallows_when_fill_succeeded():
    """If a rejection event arrived but the fill still went through,
    treat the error as a warning (edge-case race) and drain it."""
    broker = _new_broker()
    broker._on_ib_error(reqId=99, errorCode=201, errorString="Late warning")
    broker._raise_if_rejected(_FakeTrade(99), filled=True)  # must not raise
    assert 99 not in broker._ib_order_rejections
