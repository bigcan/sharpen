"""AlphaSeek monitoring — per-tick WandB metrics, aggregates, and alerts.

Provides real-time monitoring for the AlphaSeek live trading engine:
- Per-tick metrics: position, action, Q-values, confidence, price, PnL
- Periodic aggregates (60s): trades/min, hold rate, rolling Sharpe
- Alerts: position stuck, Q-divergence, latency spike, connection drop
- Health endpoint for fleet monitoring integration
"""

import logging
import time
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class AlphaSeekMonitor:
    """Monitoring and alerting for AlphaSeek live trading.

    Usage:
        monitor = AlphaSeekMonitor(wandb_enabled=True)
        monitor.log_tick(tick_data)  # every 2 seconds
        # Automatically logs aggregates every 60 seconds
    """

    def __init__(
        self,
        wandb_enabled: bool = True,
        aggregate_interval_s: float = 60.0,
        alert_stuck_ticks: int = 150,  # 5 minutes at 2s
    ):
        self.wandb_enabled = wandb_enabled and HAS_WANDB
        self.aggregate_interval_s = aggregate_interval_s
        self.alert_stuck_ticks = alert_stuck_ticks

        # Rolling windows for aggregates
        self._returns: deque[float] = deque(maxlen=1800)  # 1 hour at 2s
        self._latencies: deque[float] = deque(maxlen=300)  # 10 minutes
        self._actions: deque[int] = deque(maxlen=1800)
        self._spreads: deque[float] = deque(maxlen=1800)

        # Counters
        self._tick_count: int = 0
        self._trade_count: int = 0
        self._last_aggregate_time: float = time.time()
        self._last_trade_tick: int = 0
        self._start_time: float = time.time()
        self._cumulative_pnl: float = 0.0
        self._last_portfolio_value: float = 0.0

        # Alert state
        self._consecutive_same_position: int = 0
        self._last_position: int = 0
        self._connection_drops: int = 0

    def log_tick(
        self,
        tick: int,
        position: int,
        action_int: int,
        mid_price: float,
        spread: float,
        portfolio_value: float,
        drawdown: float,
        holding: int,
        q_means: Optional[list[float]] = None,
        confidence: float = 0.0,
        agents_agree: bool = True,
        latency_ms: float = 0.0,
        risk_violations: Optional[list[str]] = None,
        trade_executed: bool = False,
    ):
        """Log per-tick metrics."""
        self._tick_count = tick
        self._latencies.append(latency_ms)
        self._actions.append(action_int)
        self._spreads.append(spread)

        # PnL tracking
        if self._last_portfolio_value > 0:
            tick_return = (portfolio_value - self._last_portfolio_value) / self._last_portfolio_value
            self._returns.append(tick_return)
            self._cumulative_pnl = portfolio_value - self._last_portfolio_value
        self._last_portfolio_value = portfolio_value

        if trade_executed:
            self._trade_count += 1
            self._last_trade_tick = tick

        # --- WandB per-tick logging ---
        if self.wandb_enabled and wandb.run is not None:
            metrics = {
                "tick/position": position,
                "tick/action": action_int,
                "tick/mid_price": mid_price,
                "tick/spread": spread,
                "tick/portfolio_value": portfolio_value,
                "tick/drawdown": drawdown,
                "tick/holding": holding,
                "tick/confidence": confidence,
                "tick/latency_ms": latency_ms,
                "tick/agents_agree": int(agents_agree),
                "tick/cumulative_pnl": self._cumulative_pnl,
            }
            if q_means:
                for i, qm in enumerate(q_means):
                    metrics[f"tick/q_mean_agent_{i}"] = qm
            if risk_violations:
                metrics["tick/risk_violations"] = len(risk_violations)

            wandb.log(metrics, step=tick, commit=True)

        # --- Alerts ---
        self._check_alerts(position, confidence, agents_agree, latency_ms)

        # --- Periodic aggregates ---
        now = time.time()
        if now - self._last_aggregate_time >= self.aggregate_interval_s:
            self._log_aggregates()
            self._last_aggregate_time = now

    def _check_alerts(
        self,
        position: int,
        confidence: float,
        agents_agree: bool,
        latency_ms: float,
    ):
        """Check for alert conditions."""
        # Position stuck
        if position == self._last_position and position != 0:
            self._consecutive_same_position += 1
        else:
            self._consecutive_same_position = 0
        self._last_position = position

        if self._consecutive_same_position >= self.alert_stuck_ticks:
            logger.warning(
                "ALERT: Position stuck at %d for %d ticks (%.0fs)",
                position, self._consecutive_same_position,
                self._consecutive_same_position * 2,
            )
            if self.wandb_enabled and wandb.run is not None:
                wandb.alert(
                    title="Position Stuck",
                    text=f"Position {position} unchanged for {self._consecutive_same_position} ticks",
                    level=wandb.AlertLevel.WARN,
                )

        # Q-value divergence (agents strongly disagree)
        if not agents_agree and confidence > 0.01:
            logger.warning("ALERT: Agent disagreement with high confidence (%.4f)", confidence)

        # Latency spike
        if latency_ms > 1000:
            logger.warning("ALERT: Extreme latency: %.1fms", latency_ms)

    def _log_aggregates(self):
        """Log periodic aggregate metrics."""
        if not self._actions:
            return

        actions_arr = np.array(list(self._actions))
        n_ticks = len(actions_arr)
        n_trades = int(np.count_nonzero(actions_arr))
        hold_rate = 1.0 - (n_trades / max(n_ticks, 1))

        # Trades per minute
        elapsed_s = max(time.time() - self._start_time, 1.0)
        trades_per_min = self._trade_count / (elapsed_s / 60.0)

        # Rolling Sharpe (if enough returns)
        rolling_sharpe = 0.0
        if len(self._returns) >= 30:
            returns_arr = np.array(list(self._returns))
            std = returns_arr.std()
            if std > 0:
                rolling_sharpe = returns_arr.mean() / std

        # Latency stats
        latency_arr = np.array(list(self._latencies)) if self._latencies else np.array([0.0])
        avg_latency = latency_arr.mean()
        p95_latency = np.percentile(latency_arr, 95) if len(latency_arr) > 1 else 0.0

        # Spread stats
        avg_spread = np.mean(list(self._spreads)) if self._spreads else 0.0

        logger.info(
            "AGGREGATE [tick=%d]: trades/min=%.1f, hold_rate=%.2f, "
            "sharpe_rolling=%.4f, avg_latency=%.1fms, p95_latency=%.1fms, "
            "avg_spread=%.6f, total_trades=%d",
            self._tick_count, trades_per_min, hold_rate,
            rolling_sharpe, avg_latency, p95_latency,
            avg_spread, self._trade_count,
        )

        if self.wandb_enabled and wandb.run is not None:
            wandb.log({
                "agg/trades_per_minute": trades_per_min,
                "agg/hold_rate": hold_rate,
                "agg/sharpe_rolling": rolling_sharpe,
                "agg/avg_latency_ms": avg_latency,
                "agg/p95_latency_ms": p95_latency,
                "agg/avg_spread": avg_spread,
                "agg/total_trades": self._trade_count,
                "agg/total_ticks": self._tick_count,
                "agg/uptime_hours": elapsed_s / 3600,
            }, step=self._tick_count, commit=True)

    def log_connection_drop(self):
        """Log a WebSocket connection drop."""
        self._connection_drops += 1
        logger.warning("Connection drop #%d", self._connection_drops)
        if self.wandb_enabled and wandb.run is not None:
            wandb.alert(
                title="Connection Drop",
                text=f"WebSocket connection dropped (total: {self._connection_drops})",
                level=wandb.AlertLevel.WARN,
            )

    def get_health(self) -> dict:
        """Return health status for fleet monitoring."""
        elapsed_s = max(time.time() - self._start_time, 1.0)
        latency_arr = np.array(list(self._latencies)) if self._latencies else np.array([0.0])

        return {
            "status": "running",
            "tick_count": self._tick_count,
            "trade_count": self._trade_count,
            "uptime_hours": round(elapsed_s / 3600, 2),
            "trades_per_minute": round(self._trade_count / (elapsed_s / 60.0), 1),
            "avg_latency_ms": round(float(latency_arr.mean()), 1),
            "p95_latency_ms": round(float(np.percentile(latency_arr, 95)), 1) if len(latency_arr) > 1 else 0.0,
            "connection_drops": self._connection_drops,
            "position_stuck_ticks": self._consecutive_same_position,
            "portfolio_value": self._last_portfolio_value,
        }
