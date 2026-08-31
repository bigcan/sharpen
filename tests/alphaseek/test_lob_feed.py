"""Tests for BybitLOBFeed — snapshot format, stale detection, error handling."""

import time

import pytest

from sharpen.alphaseek.lob_feed import _FEATURE_DEPTH, BybitLOBFeed


class TestOBToSnapshot:
    """Test orderbook → snapshot conversion (no exchange needed)."""

    def test_snapshot_keys(self):
        feed = BybitLOBFeed()
        ob = {
            "bids": [[100.0, 1.5], [99.99, 2.0], [99.98, 3.0], [99.97, 1.0], [99.96, 0.5]],
            "asks": [[100.01, 2.5], [100.02, 1.0], [100.03, 0.8], [100.04, 1.2], [100.05, 0.3]],
            "timestamp": 1700000000000,
        }
        snap = feed._ob_to_snapshot(ob)

        assert "best_bid_price" in snap
        assert "best_ask_price" in snap
        assert "best_bid_qty" in snap
        assert "best_ask_qty" in snap
        assert "spread" in snap
        assert "bid_prices_5" in snap
        assert "bid_qtys_5" in snap
        assert "ask_prices_5" in snap
        assert "ask_qtys_5" in snap
        assert "timestamp_ms" in snap

    def test_snapshot_values(self):
        feed = BybitLOBFeed()
        ob = {
            "bids": [[100.0, 1.5], [99.99, 2.0], [99.98, 3.0], [99.97, 1.0], [99.96, 0.5]],
            "asks": [[100.01, 2.5], [100.02, 1.0], [100.03, 0.8], [100.04, 1.2], [100.05, 0.3]],
            "timestamp": 1700000000000,
        }
        snap = feed._ob_to_snapshot(ob)

        assert snap["best_bid_price"] == 100.0
        assert snap["best_ask_price"] == 100.01
        assert snap["best_bid_qty"] == 1.5
        assert snap["best_ask_qty"] == 2.5
        assert abs(snap["spread"] - 0.01) < 1e-10
        assert len(snap["bid_prices_5"]) == _FEATURE_DEPTH
        assert len(snap["ask_qtys_5"]) == _FEATURE_DEPTH
        assert snap["bid_prices_5"][0] == 100.0
        assert snap["ask_prices_5"][-1] == 100.05

    def test_empty_orderbook(self):
        feed = BybitLOBFeed()
        ob = {"bids": [], "asks": [], "timestamp": 0}
        snap = feed._ob_to_snapshot(ob)
        assert snap["best_bid_price"] == 0.0
        assert snap["best_ask_price"] == 0.0
        assert snap["spread"] == 0.0
        assert len(snap["bid_prices_5"]) == 0

    def test_shallow_orderbook(self):
        """Orderbook with fewer than 5 levels."""
        feed = BybitLOBFeed()
        ob = {
            "bids": [[100.0, 1.0], [99.99, 2.0]],
            "asks": [[100.01, 1.5]],
            "timestamp": 0,
        }
        snap = feed._ob_to_snapshot(ob)
        assert len(snap["bid_prices_5"]) == 2
        assert len(snap["ask_prices_5"]) == 1


class TestBybitLOBFeedState:
    def test_not_connected_by_default(self):
        feed = BybitLOBFeed()
        assert not feed.is_connected

    def test_stale_before_update(self):
        feed = BybitLOBFeed()
        assert feed.is_stale

    def test_stale_threshold(self):
        feed = BybitLOBFeed(stale_threshold_s=5.0)
        feed._last_update_ts = time.monotonic()
        assert not feed.is_stale

    def test_last_update_age(self):
        feed = BybitLOBFeed()
        assert feed.last_update_age_s == float("inf")
        feed._last_update_ts = time.monotonic() - 3.0
        assert abs(feed.last_update_age_s - 3.0) < 0.5


class TestBybitLOBFeedNotConnected:
    def test_get_latest_raises_if_not_connected(self):
        import asyncio

        feed = BybitLOBFeed()
        with pytest.raises(RuntimeError, match="not connected"):
            asyncio.get_event_loop().run_until_complete(feed.get_latest_snapshot())
