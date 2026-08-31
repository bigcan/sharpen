"""PRISM L2 Position Sizing Overlay — regime-aware position scaling.

Queries the PRISM API for vol-regime detection and scales the agent's
target position accordingly.  Falls back to multiplier=1.0 on any error.

Usage::

    overlay = PRISMOverlay(config["prism"])
    multiplier, info = overlay.get_position_multiplier()
    target_position *= multiplier
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MULTIPLIERS = {
    "LOW_VOL": 1.3,
    "NORMAL_VOL": 1.0,
    "HIGH_VOL": 0.3,
}


class PRISMOverlay:
    """L2 position sizing overlay using PRISM regime detection.

    Args:
        config: ``prism:`` section from the live trading YAML.
            Required keys: ``base_url``, ``ticker``.
            Optional: ``api_key``, ``timeout``, ``cache_ttl``,
            ``crisis_flatten``, ``multipliers``, ``timeframe``.
    """

    def __init__(self, config: dict[str, Any]):
        from prism_client import PRISMClient

        self._enabled = bool(config.get("enabled", False))
        self._base_url = config.get("base_url", "http://prism-api:8001")
        self._ticker = config["ticker"]
        self._timeframe = config.get("timeframe", "daily")
        self._timeout = config.get("timeout", 5)
        self._cache_ttl = config.get("cache_ttl", 60)
        self._crisis_flatten = config.get("crisis_flatten", True)

        # Allow per-strategy multiplier overrides
        mult_cfg = config.get("multipliers", {})
        self._multipliers = {
            k: mult_cfg.get(k, v) for k, v in _DEFAULT_MULTIPLIERS.items()
        }

        self._client = PRISMClient(
            base_url=self._base_url,
            api_key=config.get("api_key") or None,
            timeout=self._timeout,
        )

        # Cache
        self._cached_result: dict[str, Any] | None = None
        self._cache_ts: float = 0.0

        # Stats
        self._total_calls = 0
        self._total_errors = 0
        self._total_fallbacks = 0

        logger.info(
            f"PRISMOverlay initialized: {self._base_url}, "
            f"ticker={self._ticker}, timeframe={self._timeframe}, "
            f"cache_ttl={self._cache_ttl}s, crisis_flatten={self._crisis_flatten}",
        )

    @property
    def enabled(self) -> bool:
        """Reflects the ``prism.enabled`` config flag (default False).

        The live runner already gates construction on this flag, so any
        constructed overlay is normally enabled; the property exists so that
        callers reading it get the truthful config value rather than a hardcoded
        constant.
        """
        return self._enabled

    async def get_position_multiplier(self) -> tuple[float, dict[str, Any]]:
        """Get vol-regime position multiplier.

        FIX PRS-01: Made async — the underlying PRISMClient uses synchronous
        requests which would block the event loop. Now runs the blocking call
        in a thread executor.

        Returns:
            (multiplier, regime_info) where multiplier is in [0.0, 1.3]
            and regime_info is a dict with regime details for logging.
            On error: (1.0, {"fallback": True}).
        """
        # Check cache (FIX PRISM-01: use _effective_cache_ttl which respects error backoff)
        now = time.monotonic()
        effective_ttl = getattr(self, "_effective_cache_ttl", self._cache_ttl)
        if self._cached_result is not None and (now - self._cache_ts) < effective_ttl:
            self._total_cache_hits = getattr(self, "_total_cache_hits", 0) + 1
            return self._cached_result["multiplier"], self._cached_result["info"]

        self._total_calls += 1
        t0 = time.monotonic()

        try:
            # FIX PRS-01: Run blocking HTTP call in thread executor
            loop = asyncio.get_running_loop()
            regime = await loop.run_in_executor(
                None,
                lambda: self._client.get_regime(
                    ticker=self._ticker,
                    timeframe=self._timeframe,
                ),
            )
            latency = time.monotonic() - t0

            # Crisis mode: BEARISH + HIGH_VOL (code 2) → flatten
            if self._crisis_flatten and regime.is_crisis:
                multiplier = 0.0
                logger.warning(
                    f"PRISM CRISIS detected ({regime.composite_label}) — "
                    f"flattening position",
                )
            else:
                multiplier = self._multipliers.get(regime.vol_regime, 1.0)

            info: dict[str, Any] = {
                "price_regime": regime.price_regime,
                "vol_regime": regime.vol_regime,
                "composite_code": regime.composite_code,
                "composite_label": regime.composite_label,
                "confidence": regime.confidence,
                "multiplier": multiplier,
                "latency_ms": latency * 1000,
                "fallback": False,
            }

            # Update cache + reset to normal TTL on success
            self._cached_result = {"multiplier": multiplier, "info": info}
            self._cache_ts = now
            self._effective_cache_ttl = self._cache_ttl  # FIX PRISM-01: restore normal TTL

            return multiplier, info

        except Exception as e:
            self._total_errors += 1
            self._total_fallbacks += 1
            latency = time.monotonic() - t0
            logger.warning(
                f"PRISM API error ({latency:.1f}s): {e} — using fallback multiplier=1.0",
            )
            # FIX PRS-03 + PRISM-01: Cache the fallback with a 5-minute error TTL.
            # Previously set _cache_ttl_override which was never read by the cache
            # check. Now uses _effective_cache_ttl which IS read.
            fallback_info: dict[str, Any] = {"fallback": True, "error": str(e)}
            self._cached_result = {"multiplier": 1.0, "info": fallback_info}
            self._cache_ts = time.monotonic()
            self._effective_cache_ttl = 300  # 5 min error backoff
            return 1.0, fallback_info

    def get_stats(self) -> dict[str, Any]:
        """Return overlay usage statistics."""
        # FIX PRS-02: Correct cache hit rate formula
        cache_hits = getattr(self, "_total_cache_hits", 0)
        total = cache_hits + self._total_calls
        return {
            "total_calls": self._total_calls,
            "total_errors": self._total_errors,
            "total_fallbacks": self._total_fallbacks,
            "cache_hit_rate": cache_hits / max(total, 1),
        }
