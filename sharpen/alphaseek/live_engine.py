"""AlphaSeek live trading engine — orchestrates all components for live BTC trading.

Follows the LiveTradingEngine pattern from sharpen/crypto/live/live_engine.py,
adapted for 2-second discrete (DQN ensemble) trading on BTC perpetual futures.

Main loop per tick:
    1. TickClock.wait_for_next_tick()
    2. StateBuilder.get_state(position, holding)
    3. Sanity check (NaN/Inf)
    4. Ensemble.predict(state)
    5. PositionManager.apply_action(action_int, mid_price)
    6. RiskManager.check(action, position, mid_price, spread, latency)
    7. Broker.execute_position_change() if trade_delta != 0
    8. Position reconciliation
    9. WandB + Monitor logging
"""

import asyncio
import logging
import os
import signal
import time
from typing import Optional, Protocol

import torch

from .ensemble import AlphaSeekEnsemble
from .monitoring import AlphaSeekMonitor
from .position_manager import DiscretePositionManager
from .risk import AlphaSeekRiskConfig, AlphaSeekRiskManager
from .tick_clock import TickClock

logger = logging.getLogger(__name__)


class BrokerProtocol(Protocol):
    """Minimal broker interface for AlphaSeek."""

    async def execute_position_change(
        self, asset: str, current_position: float,
        target_position: float, portfolio_value: float,
    ) -> object: ...

    async def get_single_position(self, asset: str) -> float: ...

    async def get_account_info(self) -> dict: ...

    async def emergency_flatten(self, assets: list[str]) -> object: ...


class StateBuilderProtocol(Protocol):
    """Minimal state builder interface (v3: 12-dim state with execution dims)."""

    def get_state(
        self,
        position: int,
        holding: int,
        pending_limit_active: int = 0,
        bars_since_last_trade: int = 0,
    ) -> torch.Tensor: ...

    @property
    def is_ready(self) -> bool: ...

    def get_current_mid_price(self) -> float: ...

    def get_current_spread(self) -> float: ...


class AlphaSeekLiveEngine:
    """Main orchestrator for AlphaSeek live BTC trading.

    Wires together: TickClock, StateBuilder, Ensemble, PositionManager,
    RiskManager, Broker, and Monitor.

    Usage:
        engine = AlphaSeekLiveEngine(
            ensemble=ensemble,
            state_builder=state_builder,
            broker=broker,
            config=config,
        )
        await engine.start()
    """

    def __init__(
        self,
        ensemble: AlphaSeekEnsemble,
        state_builder: StateBuilderProtocol,
        broker: BrokerProtocol,
        config: dict,
        tick_clock: Optional[TickClock] = None,
        position_manager: Optional[DiscretePositionManager] = None,
        risk_manager: Optional[AlphaSeekRiskManager] = None,
        monitor: Optional[AlphaSeekMonitor] = None,
    ):
        self.ensemble = ensemble
        self.state_builder = state_builder
        self.broker = broker
        self.config = config

        # Trading config
        trading_cfg = config.get("trading", {})
        self.asset = config.get("exchange", {}).get("asset", "BTC")
        self.position_notional = trading_cfg.get("position_notional", 10000.0)
        self.symbol = f"{self.asset}/USDT"

        # Components
        tick_cfg = config.get("tick_clock", {})
        self.tick_clock = tick_clock or TickClock(
            interval_seconds=tick_cfg.get("interval_seconds", 2.0),
            max_late_seconds=tick_cfg.get("max_late_seconds", 1.0),
        )
        self.position_manager = position_manager or DiscretePositionManager(
            max_position=trading_cfg.get("max_position", 1),
            max_holding=trading_cfg.get("max_holding", 1800),
            stop_loss_thresh=trading_cfg.get("stop_loss_thresh", 1e-4),
        )

        risk_cfg = config.get("risk", {})
        self.risk_manager = risk_manager or AlphaSeekRiskManager(
            AlphaSeekRiskConfig(
                enabled=risk_cfg.get("enabled", True),
                max_drawdown_pct=risk_cfg.get("max_drawdown_pct", 0.05),
                max_daily_loss_pct=risk_cfg.get("max_daily_loss_pct", 0.02),
                circuit_breaker_cooldown_ticks=risk_cfg.get("circuit_breaker_cooldown_ticks", 300),
                daily_turnover_limit=risk_cfg.get("daily_turnover_limit", 500),
                flash_crash_pct=risk_cfg.get("flash_crash_pct", 0.02),
                max_spread_multiplier=risk_cfg.get("max_spread_multiplier", 5.0),
                max_latency_ms=risk_cfg.get("max_latency_ms", 500.0),
            ),
        )

        wandb_cfg = config.get("wandb", {})
        self.monitor = monitor or AlphaSeekMonitor(
            wandb_enabled=wandb_cfg.get("enabled", True),
        )

        # Safety
        safety_cfg = config.get("safety", {})
        self.kill_file = safety_cfg.get("kill_file", "/tmp/finrl_alphaseek_kill")
        self.emergency_flatten_on_error = safety_cfg.get("emergency_flatten_on_error", True)
        self.max_consecutive_errors = safety_cfg.get("max_consecutive_errors", 5)
        self.dry_run = config.get("dry_run", False)

        # State
        self._running = False
        self._tick_count = 0
        self._consecutive_errors = 0
        self._portfolio_value = trading_cfg.get("initial_balance", 10000.0)

    async def start(self):
        """Main trading loop."""
        logger.info("=" * 60)
        logger.info("AlphaSeek Live Engine starting")
        logger.info("  Asset: %s", self.asset)
        logger.info("  Position notional: $%.2f", self.position_notional)
        logger.info("  Tick interval: %.1fs", self.tick_clock.interval_seconds)
        logger.info("  Dry run: %s", self.dry_run)
        logger.info("  Kill file: %s", self.kill_file)
        logger.info("=" * 60)

        # Register signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_shutdown)

        # Initialize
        self.risk_manager.reset(self._portfolio_value)
        self._running = True

        # Wait for state builder to be ready
        if not self.state_builder.is_ready:
            logger.info("Waiting for state builder to bootstrap...")
            while not self.state_builder.is_ready and self._running:
                await asyncio.sleep(1.0)
            if not self._running:
                return

        logger.info("State builder ready. Starting trading loop.")

        try:
            while self._running:
                await self._trading_tick()
        except Exception as e:
            logger.exception("Fatal error in trading loop: %s", e)
            if self.emergency_flatten_on_error:
                await self._emergency_flatten()
        finally:
            logger.info(
                "Engine stopped after %d ticks, %d trades",
                self._tick_count, self.position_manager.total_trades,
            )

    async def _trading_tick(self):
        """Execute one trading tick."""
        # --- 0. Check kill file ---
        if os.path.exists(self.kill_file):
            logger.warning("Kill file detected: %s. Shutting down.", self.kill_file)
            await self._graceful_shutdown()
            return

        # --- 1. Wait for next tick ---
        await self.tick_clock.wait_for_next_tick()
        tick_start = time.perf_counter()
        self._tick_count += 1

        try:
            # --- 2. Get state ---
            state = self.state_builder.get_state(
                self.position_manager.position,
                self.position_manager.holding,
            )
            mid_price = self.state_builder.get_current_mid_price()
            spread = self.state_builder.get_current_spread()

            # --- 3. Sanity check ---
            if not self._check_state_sanity(state):
                logger.warning("Tick %d: State sanity check failed, skipping", self._tick_count)
                return

            # --- 4. Ensemble inference ---
            action_int, metadata = self.ensemble.predict(state)

            # --- 5. Position manager ---
            pos_result = self.position_manager.apply_action(action_int, mid_price)

            # --- 6. Risk check ---
            latency_ms = (time.perf_counter() - tick_start) * 1000
            risk_action, violations = self.risk_manager.check(
                action_int=pos_result.trade_delta,
                position=self.position_manager.position,
                mid_price=mid_price,
                spread=spread,
                tick_latency_ms=latency_ms,
            )

            if violations:
                logger.info(
                    "Tick %d: Risk violations: %s (action: %d -> %d)",
                    self._tick_count, violations, pos_result.trade_delta, risk_action,
                )
                # Override action if risk manager changed it
                if risk_action != pos_result.trade_delta:
                    pos_result = self.position_manager.apply_action(risk_action, mid_price)

            # --- 7. Execute trade ---
            trade_executed = False
            if pos_result.trade_delta != 0 and not self.dry_run:
                try:
                    await self._execute_trade(pos_result.trade_delta, mid_price)
                    trade_executed = True
                except Exception as e:
                    logger.error("Trade execution failed: %s", e)
                    self._consecutive_errors += 1
                    if self._consecutive_errors >= self.max_consecutive_errors:
                        logger.error("Max consecutive errors reached. Shutting down.")
                        await self._emergency_flatten()
                        return
            elif pos_result.trade_delta != 0 and self.dry_run:
                trade_executed = True  # count dry-run trades for monitoring
                logger.debug(
                    "DRY RUN: tick=%d, delta=%d, price=%.2f",
                    self._tick_count, pos_result.trade_delta, mid_price,
                )

            if trade_executed:
                self._consecutive_errors = 0

            # --- 8. Update portfolio value ---
            self._update_portfolio_value(mid_price)
            self.risk_manager.update_portfolio_value(self._portfolio_value)

            # --- 9. Monitor ---
            total_latency_ms = (time.perf_counter() - tick_start) * 1000
            self.monitor.log_tick(
                tick=self._tick_count,
                position=self.position_manager.position,
                action_int=action_int,
                mid_price=mid_price,
                spread=spread,
                portfolio_value=self._portfolio_value,
                drawdown=self.risk_manager.drawdown,
                holding=self.position_manager.holding,
                q_means=metadata.get("q_means"),
                confidence=metadata.get("confidence", 0.0),
                agents_agree=metadata.get("agents_agree", True),
                latency_ms=total_latency_ms,
                risk_violations=violations,
                trade_executed=trade_executed,
            )

        except Exception as e:
            logger.error("Error in tick %d: %s", self._tick_count, e)
            self._consecutive_errors += 1
            if self._consecutive_errors >= self.max_consecutive_errors:
                logger.error("Max consecutive errors reached.")
                if self.emergency_flatten_on_error:
                    await self._emergency_flatten()
                self._running = False

    async def _execute_trade(self, trade_delta: int, mid_price: float):
        """Execute a position change via the broker."""
        current_pos_frac = float(self.position_manager.position - trade_delta) / self.position_manager.max_position
        target_pos_frac = float(self.position_manager.position) / self.position_manager.max_position

        result = await self.broker.execute_position_change(
            asset=self.asset,
            current_position=current_pos_frac,
            target_position=target_pos_frac,
            portfolio_value=self.position_notional,
        )
        logger.info(
            "Trade executed: delta=%d, price=%.2f, result=%s",
            trade_delta, mid_price, result,
        )

    def _update_portfolio_value(self, mid_price: float):
        """Update portfolio value based on current position and price."""
        # Simple mark-to-market: cash + position * notional * price_change
        # In production, this should come from broker.get_account_info()
        pass  # Will be filled with broker query in integration

    def _check_state_sanity(self, state: torch.Tensor) -> bool:
        """Check for NaN/Inf in state tensor."""
        if torch.isnan(state).any():
            logger.error("NaN detected in state tensor")
            return False
        if torch.isinf(state).any():
            logger.error("Inf detected in state tensor")
            return False
        return True

    async def _emergency_flatten(self):
        """Emergency close all positions."""
        logger.warning("EMERGENCY FLATTEN initiated")
        try:
            if not self.dry_run:
                await self.broker.emergency_flatten([self.asset])
            self.position_manager.force_flatten(0.0)
            logger.info("Emergency flatten complete")
        except Exception as e:
            logger.error("Emergency flatten failed: %s", e)
        self._running = False

    async def _graceful_shutdown(self):
        """Graceful shutdown: close positions, stop loop."""
        logger.info("Graceful shutdown initiated")
        if self.position_manager.position != 0:
            logger.info("Closing position before shutdown...")
            try:
                if not self.dry_run:
                    mid_price = self.state_builder.get_current_mid_price()
                    result = self.position_manager.force_flatten(mid_price)
                    await self._execute_trade(result.trade_delta, mid_price)
            except Exception as e:
                logger.error("Position close failed during shutdown: %s", e)
        self._running = False

    def _handle_shutdown(self):
        """Signal handler for SIGINT/SIGTERM."""
        logger.info("Shutdown signal received")
        self._running = False
