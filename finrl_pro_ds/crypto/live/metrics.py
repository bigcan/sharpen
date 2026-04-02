"""Prometheus metrics for live trading.

Runs a lightweight HTTP server on a daemon thread, completely decoupled
from the async trading loop.  ``Gauge.set()`` is thread-safe (atomic
C-level operation), so the main loop can call ``update()`` without locks.

Usage::

    metrics = TradingMetrics(port=9101, strategy_name="gmgp1-gold")
    metrics.start()  # daemon thread — dies with process

    # Called from async trading loop (thread-safe):
    metrics.update(
        position=0.35,
        portfolio_value=10250.50,
        drawdown_pct=0.012,
        ...
    )
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False


class TradingMetrics:
    """Thread-safe Prometheus metrics for live trading.

    If ``prometheus_client`` is not installed or ``port`` is 0,
    all methods become no-ops — trading is never affected.
    """

    def __init__(self, port: int = 0, strategy_name: str = "unknown"):
        self._port = port
        self._enabled = port > 0 and _HAS_PROMETHEUS
        self._started = False

        if not self._enabled:
            return

        labels = {"strategy": strategy_name}

        # Gauges (current value, can go up or down)
        self._position = Gauge(
            "finrl_position",
            "Current position fraction [-1, 1]",
            ["strategy"],
        )
        self._portfolio_value = Gauge(
            "finrl_portfolio_value",
            "Portfolio value in USD",
            ["strategy"],
        )
        self._drawdown_pct = Gauge(
            "finrl_drawdown_pct",
            "Current drawdown from peak",
            ["strategy"],
        )
        self._daily_loss_pct = Gauge(
            "finrl_daily_loss_pct",
            "Daily PnL percentage",
            ["strategy"],
        )
        self._broker_connected = Gauge(
            "finrl_broker_connected",
            "Broker connection status (1=connected, 0=disconnected)",
            ["strategy"],
        )
        self._bar_count = Gauge(
            "finrl_bar_count",
            "Total bars processed",
            ["strategy"],
        )
        self._consecutive_errors = Gauge(
            "finrl_consecutive_errors",
            "Consecutive error count",
            ["strategy"],
        )
        self._last_bar_timestamp = Gauge(
            "finrl_last_bar_timestamp",
            "Unix timestamp of last processed bar",
            ["strategy"],
        )
        self._funding_rate = Gauge(
            "finrl_funding_rate",
            "Current funding rate",
            ["strategy"],
        )

        # Counters (monotonically increasing)
        self._total_trades = Counter(
            "finrl_total_trades",
            "Total trades executed",
            ["strategy"],
        )
        self._total_fees = Counter(
            "finrl_total_fees",
            "Total fees paid in USD",
            ["strategy"],
        )

        # PRISM regime overlay metrics
        self._prism_multiplier = Gauge(
            "prism_position_multiplier",
            "PRISM vol-regime position multiplier",
            ["strategy"],
        )
        self._prism_composite_code = Gauge(
            "prism_composite_code",
            "PRISM composite regime code (0-8, -1=fallback)",
            ["strategy"],
        )
        self._prism_api_latency = Histogram(
            "prism_api_latency_seconds",
            "PRISM API call latency",
            ["strategy"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0),
        )
        self._prism_api_errors = Counter(
            "prism_api_errors_total",
            "PRISM API error count",
            ["strategy"],
        )
        self._prism_fallback_active = Gauge(
            "prism_fallback_active",
            "PRISM fallback mode (1=active, 0=normal)",
            ["strategy"],
        )

        self._labels = labels
        self._prev_trades: int = 0
        self._prev_fees: float = 0.0

    def start(self) -> None:
        """Start Prometheus HTTP server on a daemon thread."""
        if not self._enabled or self._started:
            return

        try:
            # start_http_server launches a daemon thread internally
            start_http_server(self._port)
            self._started = True
            logger.info(f"Prometheus metrics server started on :{self._port}")

            # Initialize counter labels so they appear in /metrics with 0
            # (Counters are invisible until .inc() is called otherwise)
            s = self._labels.get("strategy", "unknown")
            self._total_trades.labels(strategy=s)
            self._total_fees.labels(strategy=s)
            self._prism_api_errors.labels(strategy=s)
        except Exception as e:
            logger.warning(f"Prometheus server failed to start: {e}")
            self._enabled = False

    def update(
        self,
        *,
        position: float = 0.0,
        portfolio_value: float = 0.0,
        drawdown_pct: float = 0.0,
        daily_loss_pct: float = 0.0,
        broker_connected: bool = True,
        bar_count: int = 0,
        total_trades: int = 0,
        total_fees: float = 0.0,
        consecutive_errors: int = 0,
        last_bar_timestamp: float = 0.0,
        funding_rate: float = 0.0,
    ) -> None:
        """Update all metric values. Thread-safe."""
        if not self._enabled:
            return

        s = self._labels["strategy"]

        self._position.labels(strategy=s).set(position)
        self._portfolio_value.labels(strategy=s).set(portfolio_value)
        self._drawdown_pct.labels(strategy=s).set(drawdown_pct)
        self._daily_loss_pct.labels(strategy=s).set(daily_loss_pct)
        self._broker_connected.labels(strategy=s).set(
            1.0 if broker_connected else 0.0,
        )
        self._bar_count.labels(strategy=s).set(bar_count)
        self._consecutive_errors.labels(strategy=s).set(consecutive_errors)
        self._last_bar_timestamp.labels(strategy=s).set(last_bar_timestamp)
        self._funding_rate.labels(strategy=s).set(funding_rate)

        # Counters: compute increment from previous value
        trade_delta = total_trades - self._prev_trades
        if trade_delta > 0:
            self._total_trades.labels(strategy=s).inc(trade_delta)
            self._prev_trades = total_trades

        fee_delta = total_fees - self._prev_fees
        if fee_delta > 0:
            self._total_fees.labels(strategy=s).inc(fee_delta)
            self._prev_fees = total_fees

    def update_prism(
        self,
        *,
        multiplier: float = 1.0,
        composite_code: int = -1,
        latency_ms: float = 0.0,
        is_fallback: bool = False,
        is_error: bool = False,
    ) -> None:
        """Update PRISM regime overlay metrics. Thread-safe."""
        if not self._enabled:
            return

        s = self._labels["strategy"]

        self._prism_multiplier.labels(strategy=s).set(multiplier)
        self._prism_composite_code.labels(strategy=s).set(composite_code)
        self._prism_fallback_active.labels(strategy=s).set(
            1.0 if is_fallback else 0.0,
        )

        if latency_ms > 0:
            self._prism_api_latency.labels(strategy=s).observe(latency_ms / 1000.0)

        if is_error:
            self._prism_api_errors.labels(strategy=s).inc()
