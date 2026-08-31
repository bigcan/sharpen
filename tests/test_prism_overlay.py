"""Tests for PRISM L2 position sizing overlay."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

# The overlay's client is an optional dependency: `prism_client` ships INSIDE the
# live-trading image (`docker/live/prism_sdk/`, an installable package) and is not
# part of the workstation env. Every test here patches "prism_client.PRISMClient",
# which resolves the module at call time -> 12 hard ModuleNotFoundError failures
# rather than a signal about this code. Skip cleanly when the SDK is absent so the
# tests still run wherever it IS installed (`pip install -e docker/live/prism_sdk`).
# PRISM itself is falsified (S413+, `prism.enabled: false` everywhere); this keeps
# the retired-but-deployed overlay testable without a red local suite.
pytest.importorskip("prism_client", reason="prism_client SDK not installed (docker/live/prism_sdk)")


def _call(overlay):
    """Drive the async ``get_position_multiplier`` to completion.

    The overlay method is ``async`` (FIX PRS-01 runs the blocking PRISM client
    call in a thread executor). These tests are synchronous, so each call runs in
    a fresh event loop; the overlay's cache/backoff state lives on the instance
    and persists across loops, so that behaviour is preserved.
    """
    return asyncio.run(overlay.get_position_multiplier())


@pytest.fixture
def base_config():
    return {
        "base_url": "http://prism-api:8001",
        "ticker": "BTC-USD",
        "timeframe": "daily",
        "timeout": 5,
        "cache_ttl": 2,
        "crisis_flatten": True,
        "multipliers": {
            "LOW_VOL": 1.3,
            "NORMAL_VOL": 1.0,
            "HIGH_VOL": 0.3,
        },
    }


def _make_regime(
    price_regime="BULLISH",
    vol_regime="NORMAL_VOL",
    composite_code=7,
    composite_label="Normal rally",
    confidence=0.65,
):
    """Create a mock RegimeResult."""
    mock = MagicMock()
    mock.price_regime = price_regime
    mock.vol_regime = vol_regime
    mock.composite_code = composite_code
    mock.composite_label = composite_label
    mock.confidence = confidence
    mock.is_crisis = composite_code == 2
    mock.position_multiplier = {"LOW_VOL": 1.3, "NORMAL_VOL": 1.0, "HIGH_VOL": 0.3}.get(vol_regime, 1.0)
    return mock


class TestPRISMOverlay:
    """Test PRISMOverlay L2 position sizing."""

    @patch("prism_client.PRISMClient")
    def test_normal_vol_multiplier_1x(self, mock_client_cls, base_config):
        from sharpen.crypto.live.prism_overlay import PRISMOverlay

        mock_client = MagicMock()
        mock_client.get_regime.return_value = _make_regime(vol_regime="NORMAL_VOL")
        mock_client_cls.return_value = mock_client

        overlay = PRISMOverlay(base_config)
        multiplier, info = _call(overlay)

        assert multiplier == 1.0
        assert info["vol_regime"] == "NORMAL_VOL"
        assert info["fallback"] is False

    @patch("prism_client.PRISMClient")
    def test_low_vol_multiplier_1_3x(self, mock_client_cls, base_config):
        from sharpen.crypto.live.prism_overlay import PRISMOverlay

        mock_client = MagicMock()
        mock_client.get_regime.return_value = _make_regime(vol_regime="LOW_VOL", composite_code=6)
        mock_client_cls.return_value = mock_client

        overlay = PRISMOverlay(base_config)
        multiplier, info = _call(overlay)

        assert multiplier == 1.3
        assert info["vol_regime"] == "LOW_VOL"

    @patch("prism_client.PRISMClient")
    def test_high_vol_multiplier_0_3x(self, mock_client_cls, base_config):
        from sharpen.crypto.live.prism_overlay import PRISMOverlay

        mock_client = MagicMock()
        mock_client.get_regime.return_value = _make_regime(vol_regime="HIGH_VOL", composite_code=5)
        mock_client_cls.return_value = mock_client

        overlay = PRISMOverlay(base_config)
        multiplier, info = _call(overlay)

        assert multiplier == 0.3
        assert info["vol_regime"] == "HIGH_VOL"

    @patch("prism_client.PRISMClient")
    def test_crisis_flatten(self, mock_client_cls, base_config):
        from sharpen.crypto.live.prism_overlay import PRISMOverlay

        mock_client = MagicMock()
        mock_client.get_regime.return_value = _make_regime(
            price_regime="BEARISH",
            vol_regime="HIGH_VOL",
            composite_code=2,
            composite_label="Crash/Panic",
        )
        mock_client_cls.return_value = mock_client

        overlay = PRISMOverlay(base_config)
        multiplier, info = _call(overlay)

        assert multiplier == 0.0
        assert info["composite_code"] == 2

    @patch("prism_client.PRISMClient")
    def test_crisis_flatten_disabled(self, mock_client_cls, base_config):
        from sharpen.crypto.live.prism_overlay import PRISMOverlay

        base_config["crisis_flatten"] = False
        mock_client = MagicMock()
        mock_client.get_regime.return_value = _make_regime(
            price_regime="BEARISH",
            vol_regime="HIGH_VOL",
            composite_code=2,
        )
        mock_client_cls.return_value = mock_client

        overlay = PRISMOverlay(base_config)
        multiplier, info = _call(overlay)

        # Without crisis flatten, HIGH_VOL multiplier applies
        assert multiplier == 0.3

    @patch("prism_client.PRISMClient")
    def test_api_error_fallback(self, mock_client_cls, base_config):
        from sharpen.crypto.live.prism_overlay import PRISMOverlay

        mock_client = MagicMock()
        mock_client.get_regime.side_effect = ConnectionError("API unreachable")
        mock_client_cls.return_value = mock_client

        overlay = PRISMOverlay(base_config)
        multiplier, info = _call(overlay)

        assert multiplier == 1.0
        assert info["fallback"] is True
        assert "error" in info

    @patch("prism_client.PRISMClient")
    def test_timeout_error_fallback(self, mock_client_cls, base_config):
        from sharpen.crypto.live.prism_overlay import PRISMOverlay

        import requests
        mock_client = MagicMock()
        mock_client.get_regime.side_effect = requests.Timeout("Request timed out")
        mock_client_cls.return_value = mock_client

        overlay = PRISMOverlay(base_config)
        multiplier, info = _call(overlay)

        assert multiplier == 1.0
        assert info["fallback"] is True

    @patch("prism_client.PRISMClient")
    def test_cache_hit(self, mock_client_cls, base_config):
        from sharpen.crypto.live.prism_overlay import PRISMOverlay

        mock_client = MagicMock()
        mock_client.get_regime.return_value = _make_regime(vol_regime="LOW_VOL", composite_code=6)
        mock_client_cls.return_value = mock_client

        overlay = PRISMOverlay(base_config)

        # First call hits API
        m1, _ = _call(overlay)
        # Second call should use cache
        m2, _ = _call(overlay)

        assert m1 == m2 == 1.3
        assert mock_client.get_regime.call_count == 1  # Only 1 API call

    @patch("prism_client.PRISMClient")
    def test_cache_expiry(self, mock_client_cls, base_config):
        from sharpen.crypto.live.prism_overlay import PRISMOverlay

        base_config["cache_ttl"] = 0  # Immediate expiry

        mock_client = MagicMock()
        mock_client.get_regime.return_value = _make_regime(vol_regime="NORMAL_VOL")
        mock_client_cls.return_value = mock_client

        overlay = PRISMOverlay(base_config)
        _call(overlay)
        _call(overlay)

        assert mock_client.get_regime.call_count == 2

    @patch("prism_client.PRISMClient")
    def test_custom_multipliers(self, mock_client_cls, base_config):
        from sharpen.crypto.live.prism_overlay import PRISMOverlay

        base_config["multipliers"]["HIGH_VOL"] = 0.5

        mock_client = MagicMock()
        mock_client.get_regime.return_value = _make_regime(vol_regime="HIGH_VOL", composite_code=5)
        mock_client_cls.return_value = mock_client

        overlay = PRISMOverlay(base_config)
        multiplier, _ = _call(overlay)

        assert multiplier == 0.5

    @patch("prism_client.PRISMClient")
    def test_stats_tracking(self, mock_client_cls, base_config):
        from sharpen.crypto.live.prism_overlay import PRISMOverlay

        # cache_ttl=0 so a *successful* result is never reused, but an error
        # opens a 5-minute fallback backoff during which the cached fallback is
        # served WITHOUT a new API call (FIX PRS-03 + PRISM-01). So the 3rd call
        # below does not reach the client.
        base_config["cache_ttl"] = 0

        mock_client = MagicMock()
        mock_client.get_regime.side_effect = [
            _make_regime(),
            ConnectionError("fail"),
        ]
        mock_client_cls.return_value = mock_client

        overlay = PRISMOverlay(base_config)
        _call(overlay)  # success         -> API call 1
        _call(overlay)  # error           -> API call 2, opens 5-min backoff
        _call(overlay)  # within backoff  -> cached fallback, no API call

        stats = overlay.get_stats()
        assert stats["total_calls"] == 2
        assert stats["total_errors"] == 1
        assert stats["total_fallbacks"] == 1
        assert mock_client.get_regime.call_count == 2

    @patch("prism_client.PRISMClient")
    def test_regime_info_fields(self, mock_client_cls, base_config):
        from sharpen.crypto.live.prism_overlay import PRISMOverlay

        mock_client = MagicMock()
        mock_client.get_regime.return_value = _make_regime(
            price_regime="BULLISH",
            vol_regime="LOW_VOL",
            composite_code=6,
            composite_label="Steady trend",
            confidence=0.72,
        )
        mock_client_cls.return_value = mock_client

        overlay = PRISMOverlay(base_config)
        _, info = _call(overlay)

        assert info["price_regime"] == "BULLISH"
        assert info["vol_regime"] == "LOW_VOL"
        assert info["composite_code"] == 6
        assert info["composite_label"] == "Steady trend"
        assert info["confidence"] == 0.72
        assert info["multiplier"] == 1.3
        assert "latency_ms" in info
