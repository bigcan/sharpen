"""PRISM API Client — typed Python interface for the PRISM REST API.

Usage:
    from prism_client import PRISMClient

    client = PRISMClient("http://localhost:8001", api_key="your-key")
    regime = client.get_regime("BTC-USD")
    forecast = client.get_forecast("BTC-USD", prices=[...], horizon=10)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import requests

from prism_client.types import (
    ForecastResult,
    HealthStatus,
    RefitResult,
    RegimeHistoryEntry,
    RegimeResult,
)

logger = logging.getLogger(__name__)


class PRISMClient:
    """Synchronous Python client for the PRISM API.

    Args:
        base_url: PRISM API base URL (e.g. ``"http://localhost:8001"``).
        api_key: Optional API key for authentication. If the server has
            ``PRISM_API_KEY`` set, this must match.
        timeout: Request timeout in seconds (default 30).
        session: Optional ``requests.Session`` for connection pooling.

    Example::

        client = PRISMClient("http://localhost:8001", api_key="my-key")

        # Check health
        health = client.health()
        assert health.chronos_loaded

        # Get regime
        regime = client.get_regime("BTC-USD")
        print(regime.composite_label)
        print(regime.position_multiplier)

        # Get forecast
        forecast = client.get_forecast("BTC-USD", prices=prices, horizon=10)
        print(f"Expected return: {forecast.expected_return:.4f}")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        api_key: str | None = None,
        timeout: int = 30,
        session: requests.Session | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = session or requests.Session()

        if api_key:
            self._session.headers["X-API-Key"] = api_key

    def health(self) -> HealthStatus:
        """Check PRISM API health.

        Returns:
            HealthStatus with ``status``, ``chronos_loaded``, ``hmm_service``.

        Raises:
            requests.HTTPError: If the server is unreachable.
        """
        resp = self._get("/health")
        return HealthStatus(
            status=resp["status"],
            chronos_loaded=resp.get("chronos_loaded", False),
            hmm_service=str(resp.get("hmm_service", "unknown")),
        )

    def get_forecast(
        self,
        ticker: str,
        prices: List[float],
        horizon: int = 10,
        dates: List[str] | None = None,
    ) -> ForecastResult:
        """Get a Chronos-2 probabilistic forecast.

        Args:
            ticker: Asset symbol (e.g. ``"BTC-USD"``).
            prices: Historical price series (minimum 30 data points).
            horizon: Number of forecast steps (default 10).
            dates: Optional date strings matching ``prices``.

        Returns:
            ForecastResult with quantile log-returns, samples, and metadata.

        Raises:
            requests.HTTPError: On 4xx/5xx responses.
            ValueError: If fewer than 30 prices provided.
        """
        if len(prices) < 30:
            raise ValueError(f"Need at least 30 prices, got {len(prices)}")

        payload: Dict = {
            "ticker": ticker,
            "prices": prices,
            "horizon": horizon,
        }
        if dates:
            payload["dates"] = dates

        resp = self._post("/predict", payload)
        fc = resp["forecast"]

        return ForecastResult(
            ticker=resp["ticker"],
            timestamp=resp["timestamp"],
            p10=fc.get("ensemble_p10", []),
            p30=fc.get("ensemble_p30", []),
            p50=fc.get("ensemble_p50", fc.get("aggregated_forecast", [])),
            p70=fc.get("ensemble_p70", []),
            p90=fc.get("ensemble_p90", []),
            quantile_spread=fc.get("quantile_spread", 0.0),
            samples=fc.get("aggregated_samples"),
            is_fallback=fc.get("is_fallback", False),
            regime_context=fc.get("regime_context"),
            calibration_diagnostics=fc.get("calibration_diagnostics"),
        )

    def get_regime(
        self,
        ticker: str,
        timeframe: str = "daily",
    ) -> RegimeResult:
        """Get the current dual regime prediction.

        Args:
            ticker: Asset symbol (e.g. ``"BTC-USD"``).
            timeframe: ``"daily"``, ``"1min"``, or ``"5min"``.

        Returns:
            RegimeResult with price/vol regimes, probabilities, and composite code.
        """
        resp = self._post("/regime/predict", {
            "ticker": ticker,
            "timeframe": timeframe,
        })
        return RegimeResult(
            ticker=resp["ticker"],
            timeframe=resp["timeframe"],
            timestamp=resp["timestamp"],
            price_regime=resp["price_regime"],
            vol_regime=resp["vol_regime"],
            composite_code=resp["composite_code"],
            composite_label=resp["composite_label"],
            price_probabilities=resp["price_probabilities"],
            vol_probabilities=resp["vol_probabilities"],
            confidence=resp["confidence"],
            model_version=resp.get("model_version"),
        )

    def refit(
        self,
        ticker: str,
        timeframe: str = "daily",
        model_type: str = "both",
        warm_start: bool = True,
        lookback_days: int | None = None,
    ) -> RefitResult:
        """Trigger HMM model retraining.

        Args:
            ticker: Asset symbol.
            timeframe: ``"daily"`` or intraday.
            model_type: ``"price"``, ``"vol"``, or ``"both"``.
            warm_start: Use previous model as initialization.
            lookback_days: Training data lookback (default 730).

        Returns:
            RefitResult with new model version IDs.
        """
        payload: Dict = {
            "ticker": ticker,
            "timeframe": timeframe,
            "model_type": model_type,
            "warm_start": warm_start,
        }
        if lookback_days is not None:
            payload["lookback_days"] = lookback_days

        resp = self._post("/regime/refit", payload)
        return RefitResult(
            status=resp["status"],
            versions=resp.get("versions", {}),
        )

    def get_regime_history(
        self,
        ticker: str,
        timeframe: str = "daily",
        start: str | None = None,
        end: str | None = None,
        limit: int = 100,
    ) -> List[RegimeHistoryEntry]:
        """Get historical regime predictions.

        Args:
            ticker: Asset symbol.
            timeframe: ``"daily"`` or intraday.
            start: Start date ``"YYYY-MM-DD"`` (optional).
            end: End date ``"YYYY-MM-DD"`` (optional).
            limit: Maximum records to return (default 100).

        Returns:
            List of RegimeHistoryEntry records.
        """
        params: Dict = {"ticker": ticker, "timeframe": timeframe, "limit": limit}
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        resp = self._get("/regime/history", params=params)
        return [
            RegimeHistoryEntry(
                timestamp=h["timestamp"],
                price_regime=h["price_regime"],
                vol_regime=h["vol_regime"],
                composite_code=h["composite_code"],
                price_probs=h.get("price_probs", {}),
                vol_probs=h.get("vol_probs", {}),
                confidence=h.get("confidence", 0.0),
            )
            for h in resp.get("history", [])
        ]

    # --- Internal helpers ---

    def _get(self, path: str, params: Dict | None = None) -> Dict:
        url = f"{self.base_url}{path}"
        resp = self._session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, json_data: Dict) -> Dict:
        url = f"{self.base_url}{path}"
        resp = self._session.post(url, json=json_data, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
