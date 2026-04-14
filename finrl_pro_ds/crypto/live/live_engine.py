"""Live Trading Engine — Orchestrates agent inference, execution, and risk management.

Main loop per bar:
    1. Wait for bar close (BarClock)
    2. Fetch latest 1-min bars from exchange
    3. Update observation builder
    4. Build observation + sanity check
    5. Agent inference (deterministic)
    6. Deadband filter
    7. Risk manager check
    8. Execute position change via broker
    9. Reconcile broker position vs internal state
    10. Log to WandB

Safety:
    - Kill file check every bar
    - Max daily loss hard stop
    - Emergency flatten on unrecoverable error
    - Position reconciliation (warn >5%, halt >15%)
    - Observation NaN/Inf detection → skip bar
    - SIGINT/SIGTERM graceful shutdown with position reconciliation
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import logging
import os
import signal
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class LiveTradingEngine:
    """Orchestrates live trading for a trained SAC agent.

    Wires together:
        - BarClock: timing
        - CryptoLoader: live OHLCV fetching
        - LiveObsBuilder: observation construction
        - SACAgent: policy inference
        - ExchangePerpBroker: order execution
        - CryptoRiskManager: risk controls
        - WandB: trade monitoring

    Usage:
        engine = LiveTradingEngine(agent, broker, obs_builder, ...)
        await engine.start()  # Runs until killed or drawdown hit
    """

    def __init__(
        self,
        agent,
        broker,
        obs_builder,
        risk_manager,
        bar_clock,
        loader,
        config: dict,
    ):
        self.agent = agent
        self.broker = broker
        self.obs_builder = obs_builder
        self.risk_manager = risk_manager
        self.bar_clock = bar_clock
        self.loader = loader
        self.config = config

        # Trading state
        self._asset = config.get("exchange", {}).get("asset", "BTC")
        self._n_scales = len(config.get("features", {}).get("scales", [15, 60, 240]))
        self._obs_mode = config.get("features", {}).get("obs_mode", "window")
        self._deadband_threshold = config.get("trading", {}).get("deadband_threshold", 0.25)
        self._device = torch.device(config.get("agent", {}).get("device", "cpu"))
        self._dry_run = config.get("dry_run", False)

        # Position tracking
        self._current_position = 0.0
        self._prev_close = 0.0
        self._portfolio_value = config.get("trading", {}).get("initial_balance", 10000.0)
        self._initial_portfolio_value = self._portfolio_value
        self._peak_portfolio_value = self._portfolio_value
        self._current_funding_rate = 0.0

        # FIX AUD-H04: Track daily loss by UTC date, not bar count
        self._daily_start_value = self._portfolio_value
        self._last_daily_reset_date: Optional[datetime] = None
        self._max_daily_loss_pct = config.get("safety", {}).get("max_daily_loss_pct", 0.05)

        # FIX AUD-H07: Periodic funding rate fetch interval (bars between fetches)
        # FIX AUD-FR-02: Scale interval by bar size (target ~8h between fetches)
        bar_minutes = config.get("bar_clock", {}).get("base_interval_minutes", 15)
        self._funding_rate_fetch_interval = max(1, 480 // bar_minutes)  # 480min = 8h
        self._bars_since_funding_fetch = 0

        # Safety
        self._kill_file = Path(config.get("safety", {}).get(
            "kill_file", "/tmp/finrl_live_kill",
        ))
        self._emergency_flatten_on_error = config.get("safety", {}).get(
            "emergency_flatten_on_error", True,
        )
        # Session 426: require N consecutive execution errors before emergency
        # flatten. Transient IB HMDS stalls (Error 162/322) produced a single
        # market-price timeout that killed the container (exit=0) mid-run.
        self._execution_error_threshold = int(config.get("safety", {}).get(
            "execution_error_threshold", 3,
        ))
        self._consecutive_execution_errors = 0
        self._reconciliation_warn_pct = config.get("safety", {}).get(
            "reconciliation_warn_pct", 0.05,
        )
        self._reconciliation_halt_pct = config.get("safety", {}).get(
            "reconciliation_halt_pct", 0.15,
        )
        self._pv_divergence_warn_pct = config.get("safety", {}).get(
            "pv_divergence_warn_pct", 0.03,
        )
        # XVal: reconcile every N bars (default 4 = hourly for 15-min bars)
        self._reconcile_interval = config.get("safety", {}).get(
            "reconcile_interval", 4,
        )
        self._last_reconcile_bar: int = 0

        # Position persistence file (crash recovery)
        self._position_file = Path(config.get("safety", {}).get(
            "position_file", "/tmp/finrl_last_position.json",
        ))

        # Persistent risk-halt state (S427 revive-after-halt fix).
        # When daily-loss / MAX_DD trips, we write halted_until so that
        # docker `restart: unless-stopped` can't revive a halted engine
        # before UTC day rollover. Engine gates on this file at startup.
        self._halt_state_file = Path(config.get("safety", {}).get(
            "halt_state_file", "/app/state/risk_state.json",
        ))

        # Intra-bar DD projection (S427 follow-up): project worst-case PV
        # using the bar's adverse extreme (low for long, high for short).
        # Disabled by setting safety.intrabar_dd_projection=false.
        self._intrabar_dd_enabled = bool(config.get("safety", {}).get(
            "intrabar_dd_projection", True,
        ))

        # Daily loss smoothing: median of last 3 PV readings prevents false
        # triggers from broker API hiccups returning anomalous portfolio values
        self._pv_buffer: collections.deque[float] = collections.deque(maxlen=3)

        # Control
        self._should_stop = False
        self._wandb_run = None

        # Session 426: per-day contract roll check (futures brokers only).
        # Stores the UTC date of the last check so we run it once per day.
        self._last_roll_check_date: date | None = None

        # FIX AUD-ENG-05: Track last 1-min timestamp for dedup in fetch
        self._last_1min_ts_ms: int = 0
        # FIX LIVE-02: Prevent PV race between _inter_bar_metrics_loop and _trading_step
        self._trading_step_active: bool = False
        # XVal: Track last bar timestamp for inter-bar metrics (fixes timestamp bug)
        self._last_bar_timestamp: float = 0.0

        # Stats
        self._total_bars = 0
        self._total_trades = 0
        self._total_fees = 0.0
        self._consecutive_errors = 0

        # Health status file for Docker HEALTHCHECK
        monitoring_cfg = config.get("monitoring", {})
        self._health_file = Path(monitoring_cfg.get(
            "health_file", "/tmp/health_status.json",
        ))
        self._strategy_name = (
            config.get("strategy_name")
            or os.environ.get("STRATEGY_NAME", "unknown")
        )

        # Prometheus metrics (started in start(), no-op if port=0 or missing lib)
        from finrl_pro_ds.crypto.live.metrics import TradingMetrics
        metrics_port = monitoring_cfg.get("metrics_port") or int(
            os.environ.get("METRICS_PORT", "0"),
        )
        self._metrics = TradingMetrics(
            port=metrics_port,
            strategy_name=self._strategy_name,
        )

        # PRISM L2 regime overlay (optional, attached by runner script)
        self._prism_overlay = None

        # Weekend flatten (CFD markets only): flatten positions before Friday close
        wf_cfg = config.get("trading", {}).get("weekend_flatten", {})
        self._weekend_flatten_enabled = wf_cfg.get("enabled", False)
        self._weekend_flatten_minutes = float(wf_cfg.get("minutes_before_close", 30))
        self._weekend_flatten_done = False  # Reset each week at Sunday open

        # Funding rate EMA gate (Option C): flatten when funding < borrow cost
        fr_gate_cfg = config.get("funding_rate_gate", {})
        self._fr_gate_enabled = fr_gate_cfg.get("enabled", False)
        if self._fr_gate_enabled:
            self._fr_gate_ema_span = int(fr_gate_cfg.get("ema_span", 168))
            self._fr_gate_borrow_cost_hourly = float(
                fr_gate_cfg.get("borrow_cost_hourly", 8.33e-6),
            )
            # Annualized threshold: hourly_rate * 8760
            self._fr_gate_threshold = self._fr_gate_borrow_cost_hourly * 8760
            self._fr_gate_ema_value: float = 0.0
            self._fr_gate_ema_alpha = 2.0 / (self._fr_gate_ema_span + 1)
            self._fr_gate_n_updates = 0
            self._fr_gate_min_warmup = int(fr_gate_cfg.get("min_warmup", 24))
            self._fr_gate_flatten_on_close = bool(fr_gate_cfg.get("flatten_on_close", True))
            self._fr_gate_total_skipped = 0
            logger.info(
                f"Funding rate gate enabled: EMA span={self._fr_gate_ema_span}h, "
                f"threshold={self._fr_gate_threshold * 100:.1f}% ann "
                f"(borrow={self._fr_gate_borrow_cost_hourly:.2e}/h)",
            )

        # Signal gate (SG-1): skip low-signal bars, mirroring training wrapper
        gate_cfg = config.get("signal_gate", {})
        self._signal_gate_enabled = gate_cfg.get("enabled", False)
        if self._signal_gate_enabled:
            self._gate_mode = gate_cfg.get("gate_mode", "composite")
            self._gate_atr_threshold = float(gate_cfg.get("atr_threshold", 0.3))
            self._gate_parkinson_threshold = float(gate_cfg.get("parkinson_threshold", 0.02))
            self._gate_volume_threshold = float(gate_cfg.get("volume_threshold", 0.5))
            self._gate_return_threshold = float(gate_cfg.get("return_threshold", 0.002))
            self._gate_max_hold_bars = int(gate_cfg.get("max_hold_bars", 20))
            self._gate_always_on_first = bool(gate_cfg.get("gate_always_on_first", True))
            self._gate_base_scale = config.get("features", {}).get("scales", [3])[0]
            self._gate_consecutive_holds = 0
            self._gate_total_skipped = 0
            self._gate_is_first_bar = True

        # Feature warmup: skip trading for N bars after bootstrap if EMA not converged
        self._warmup_bars = config.get("features", {}).get("warmup_bars", 0)
        self._warmup_bars_remaining = 0

    # -------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------
    async def start(self) -> None:
        """Main trading loop. Runs until killed, drawdown hit, or error."""

        # Register signal handlers for graceful shutdown
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                asyncio.get_event_loop().add_signal_handler(
                    sig, lambda s=sig: self._request_stop(f"signal_{s.name}"),
                )
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                signal.signal(sig, lambda s, f: self._request_stop("signal"))

        # Connect broker
        await self.broker.connect()
        logger.info(f"Broker connected: {self.broker.exchange_id} "
                     f"({'TESTNET' if self.broker.testnet else 'MAINNET'})")

        # Write health file early so Docker doesn't kill us during bootstrap
        self._write_bootstrap_health("bootstrapping")

        # Bootstrap observation builder if not already done
        if not self.obs_builder.is_ready:
            await self._bootstrap_with_retry()
            self._write_bootstrap_health("bootstrap_done")

        # Check feature warmup quality after bootstrap
        warmup_quality = self.obs_builder.get_warmup_quality()
        min_quality = min(warmup_quality.values()) if warmup_quality else 1.0
        if min_quality < 1.0:
            # Auto-calculate warmup bars if not explicitly configured
            if self._warmup_bars > 0:
                self._warmup_bars_remaining = self._warmup_bars
            else:
                # Default: skip 10 bars (~2.5h at 15-min) to stabilize
                self._warmup_bars_remaining = 10
            logger.warning(
                f"Feature warmup incomplete (min quality {min_quality:.0%}). "
                f"Skipping first {self._warmup_bars_remaining} trading bars. "
                f"Quality per scale: {warmup_quality}",
            )
        elif self._warmup_bars > 0:
            self._warmup_bars_remaining = self._warmup_bars
            logger.info(
                f"Feature warmup OK. Explicit warmup_bars={self._warmup_bars} configured.",
            )
        else:
            logger.info(f"Feature warmup OK. Quality: {warmup_quality}")

        # FIX BUG-15: Fetch portfolio value BEFORE syncing position so that
        # broker._portfolio_value is set (otherwise get_single_position()
        # returns 0.0 because _contracts_to_position divides by zero PV).
        await self._update_portfolio_value()

        # FIX AUD-M01: Re-calibrate peak/initial from broker equity, not config.
        # Config initial_balance may differ from actual account balance,
        # causing false drawdown triggers on startup.
        self._initial_portfolio_value = self._portfolio_value
        self._peak_portfolio_value = self._portfolio_value
        self._daily_start_value = self._portfolio_value

        # Sync position from exchange
        await self._sync_position()

        # FIX CT-05: On cTrader hedging-mode accounts, check for orphaned
        # positions that get_single_position() hides behind a net-zero sum.
        await self._check_orphaned_positions()

        # Check persisted position file for crash-recovery mismatch
        self._check_persisted_position()

        # Initialize prev_close
        self._prev_close = self.obs_builder.get_current_close()

        # FIX AUD-C06: Reset risk manager with initial portfolio value
        self.risk_manager.reset(self._portfolio_value)

        # S427: gate on persistent halt state before entering main loop.
        # If a prior process tripped daily-loss / MAX_DD, stay halted
        # until halted_until so `restart: unless-stopped` can't revive us.
        if await self._check_persistent_halt():
            return

        # Initialize WandB
        self._init_wandb()

        # Start Prometheus metrics server (no-op if port=0)
        self._metrics.start()

        mode_str = "DRY RUN" if self._dry_run else "LIVE"
        logger.info(
            f"=== {mode_str} TRADING STARTED ===\n"
            f"  Asset: {self._asset}\n"
            f"  Exchange: {self.broker.exchange_id}\n"
            f"  Testnet: {self.broker.testnet}\n"
            f"  Initial balance: ${self._portfolio_value:,.2f}\n"
            f"  Position: {self._current_position:.4f}\n"
            f"  Deadband: {self._deadband_threshold}\n"
            f"  Scales: {self.obs_builder.scales}\n"
            f"  Bar interval: {self.bar_clock.interval}min\n"
            f"  Signal gate: {'enabled (' + self._gate_mode + ')' if self._signal_gate_enabled else 'disabled'}",
        )

        # Final bootstrap health write — engine is fully initialized, waiting
        # for first bar. Keeps health file fresh so Docker doesn't kill us
        # during the up-to-15-min wait.
        self._write_bootstrap_health("waiting_for_first_bar")

        # Wire heartbeat callback on CMEBarClock so health file stays fresh
        # during market-closed sleeps (prevents false UNHEALTHY alerts).
        if hasattr(self.bar_clock, "_heartbeat_callback"):
            self.bar_clock._heartbeat_callback = self._write_bootstrap_health

        # Background task: update Prometheus metrics between bars so Grafana
        # shows real-time PV changes (IB sends portfolio updates every ~3 min).
        metrics_task = asyncio.create_task(self._inter_bar_metrics_loop())

        try:
            while not self._should_stop:
                try:
                    bar_time = await self.bar_clock.wait_for_next_bar()

                    # Ensure broker connection is alive before trading step
                    if not await self._ensure_broker_connected():
                        self._consecutive_errors += 1
                        if self._consecutive_errors >= 5:
                            raise RuntimeError("5 consecutive reconnect failures — stopping")
                        continue

                    await self._trading_step(bar_time)
                    # FIX AUD-ENG-05: Reset consecutive errors on successful step
                    self._consecutive_errors = 0
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    if isinstance(e, RuntimeError) and "5 consecutive" in str(e):
                        raise # Re-raise my own error to bypass the catch-all
                    self._consecutive_errors += 1
                    logger.error(f"Trading step error: {e}", exc_info=True)
                    if self._consecutive_errors >= 5:
                        raise RuntimeError("5 consecutive errors — stopping")
        finally:
            metrics_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await metrics_task
            await self._shutdown()

    async def _trading_step(self, bar_time: datetime) -> None:
        """Execute a single trading iteration."""
        self._total_bars += 1
        self._trading_step_active = True  # FIX LIVE-02: guard against PV race
        try:
            await self._trading_step_inner(bar_time)
        finally:
            self._trading_step_active = False

    async def _trading_step_inner(self, bar_time: datetime) -> None:
        """Inner trading step logic (wrapped by _trading_step for PV race guard)."""
        # --- Safety: kill file check ---
        if self._kill_file.exists():
            logger.warning(f"Kill file detected: {self._kill_file}")
            self._request_stop("kill_file")
            return

        # Session 426: once-per-day futures contract roll check.
        await self._maybe_roll_contract(bar_time)
        if self._should_stop:
            return

        # FIX AUD-H05: Update portfolio value at start of every step (not just traded bars)
        await self._update_portfolio_value()

        # XVal Layer 1: Periodic broker reconciliation on ALL bars (not just trade bars)
        await self._reconcile_all(bar_time)

        # FIX CT-10: Bail out immediately if reconciliation requested a halt.
        # Previously the trading step continued to execute trades after a
        # position-mismatch HALT, worsening the orphan state.
        if self._should_stop:
            return

        # FIX AUD-H11: Check daily loss on EVERY bar (not just traded bars).
        # Previously only ran at step 11 after trade execution, so holding bars
        # could breach the daily loss limit without detection.
        await self._check_daily_loss(bar_time)

        # --- Weekend flatten (CFD only): close positions before Friday close ---
        if await self._check_weekend_flatten(bar_time):
            return

        # FIX AUD-H07: Periodically fetch funding rate
        self._bars_since_funding_fetch += 1
        if self._bars_since_funding_fetch >= self._funding_rate_fetch_interval:
            await self._update_funding_rate()
            self._bars_since_funding_fetch = 0

        # --- 1. Fetch latest bars ---
        new_bars = await self._fetch_new_bars(bar_time)
        if new_bars is None or len(new_bars) == 0:
            logger.warning(f"No new bars at {bar_time}, skipping")
            return

        # --- 2. Update observation builder ---
        self.obs_builder.update(new_bars)

        # S427: intra-bar DD projection against the bar just closed.
        # Closing-price DD checks miss max adverse excursion within the bar,
        # which is the realistic worst case a broker / prop firm monitors.
        await self._check_intrabar_dd(bar_time)
        if self._should_stop:
            return

        # --- 2b. Feature warmup skip ---
        if self._warmup_bars_remaining > 0:
            self._warmup_bars_remaining -= 1
            current_close = self.obs_builder.get_current_close()
            self._prev_close = current_close
            logger.info(
                f"[WARMUP] Bar {self._total_bars}: features warming up, "
                f"holding position. {self._warmup_bars_remaining} bars remaining.",
            )
            self._log_step(
                bar_time, self._current_position,
                traded=False, skip_reason="feature_warmup",
            )
            return

        # --- 2c. Signal gate check (SG-1: skip low-signal bars) ---
        if not self._check_signal_gate():
            current_close = self.obs_builder.get_current_close()
            self._prev_close = current_close
            self._log_step(
                bar_time, self._current_position,
                traded=False, skip_reason="signal_gate_closed",
            )
            return

        # --- 2d. Funding rate EMA gate (Option C) ---
        if not self._check_funding_rate_gate():
            current_close = self.obs_builder.get_current_close()
            self._prev_close = current_close
            # If gate closed and we hold a position, flatten it
            if self._fr_gate_flatten_on_close and abs(self._current_position) > 0.01:
                logger.info(
                    f"Funding gate closed with position {self._current_position:.4f} "
                    f"— flattening",
                )
                await self._emergency_flatten()
                # FIX LIVE-04: Re-sync risk manager after emergency flatten so
                # its internal position/turnover state matches reality.
                self.risk_manager.reset(self._portfolio_value)
                self._log_step(
                    bar_time, 0.0,
                    traded=True, skip_reason="funding_gate_flatten",
                )
                return
            self._log_step(
                bar_time, self._current_position,
                traded=False, skip_reason="funding_gate_closed",
            )
            return

        # --- 3. Build observation ---
        current_close = self.obs_builder.get_current_close()
        # FIX AUD-C04: Pass timestamp=None so obs_builder uses bar timestamp
        obs = self.obs_builder.get_observation(
            current_position=self._current_position,
            prev_close=self._prev_close,
            current_close=current_close,
        )

        # --- 4. Sanity check ---
        if not self._check_obs_sanity(obs):
            logger.warning("Observation sanity check failed — holding position")
            self._prev_close = current_close
            self._log_step(bar_time, self._current_position, traded=False, skip_reason="obs_sanity")
            return

        # --- 5. Agent inference ---
        target_position = self._predict(obs)

        # --- 5b. PRISM L2 regime overlay ---
        regime_info: dict = {}
        if self._prism_overlay is not None:
            multiplier, regime_info = await self._prism_overlay.get_position_multiplier()
            if multiplier != 1.0:
                pre_prism = target_position
                target_position = float(np.clip(target_position * multiplier, -1.0, 1.0))
                logger.info(
                    f"PRISM overlay: {pre_prism:.4f} * {multiplier:.2f} = "
                    f"{target_position:.4f} ({regime_info.get('composite_label', 'n/a')})",
                )

        # --- 6. Deadband filter ---
        delta = target_position - self._current_position
        if abs(delta) < self._deadband_threshold:
            self._prev_close = current_close
            self._log_step(bar_time, target_position, traded=False, skip_reason="deadband", regime_info=regime_info)
            return

        # --- 7. Risk manager check ---
        action_array = np.array([target_position])
        checked_action, violations = self.risk_manager.check(
            action=action_array,
            portfolio_value=self._portfolio_value,
            margin_balance=self._portfolio_value * 0.95,  # Conservative estimate
            positions=np.array([self._current_position]),
            funding_rates=np.array([self._current_funding_rate]),
        )
        target_position = float(checked_action[0])

        if violations:
            logger.info(f"Risk violations: {violations}")
            # FIX RSK-02: Target-action=0 is not enough when DD/circuit breach
            # fires — broker order may fail, deadband may block, or position
            # may persist through cooldown accumulating more loss. On any
            # drawdown / circuit-breaker violation, force-close via broker and
            # stop the engine immediately. FTMO-safe.
            if any(
                v.startswith(("MAX_DRAWDOWN", "CIRCUIT_BREAKER", "FLASH_CRASH"))
                for v in violations
            ):
                logger.critical(
                    f"RISK HALT: violations={violations} — force-flattening "
                    f"position {self._current_position:.4f} and stopping.",
                )
                try:
                    await self._emergency_flatten()
                finally:
                    self._write_halt_state(
                        reason="risk_halt",
                        detail=",".join(violations),
                        now_utc=datetime.now(timezone.utc),
                    )
                    self._request_stop("risk_halt")
                return

        # Re-check deadband after risk adjustment
        delta = target_position - self._current_position
        if abs(delta) < self._deadband_threshold:
            # Trade won't execute — rollback turnover budget consumed by check()
            self.risk_manager.rollback_last_turnover()
            self._prev_close = current_close
            self._log_step(bar_time, target_position, traded=False, skip_reason="risk_deadband", regime_info=regime_info)
            return

        # --- 8. Execute trade ---
        if self._dry_run:
            logger.info(
                f"[DRY RUN] Would trade: {self._current_position:.4f} → "
                f"{target_position:.4f} (delta={delta:.4f})",
            )
            self._current_position = target_position
            self._prev_close = current_close
            self._log_step(bar_time, target_position, traded=True, regime_info=regime_info)
            return

        order = None
        try:
            order = await self.broker.execute_position_change(
                asset=self._asset,
                current_position=self._current_position,
                target_position=target_position,
                portfolio_value=self._portfolio_value,
            )

            if order.status == "filled":
                # BUG-16: Discrete contract rounding (esp. when portfolio <
                # contract notional) means the actual exchange position fraction
                # can differ significantly from the agent's target.  Sync from
                # broker so _current_position matches reality and reconciliation
                # won't false-trigger.
                try:
                    exchange_pos = await self.broker.get_single_position(self._asset)
                except Exception as e:
                    logger.warning(f"Post-fill position query failed: {e} — using target")
                    exchange_pos = target_position
                if abs(exchange_pos - target_position) > 0.01:
                    logger.info(
                        f"Position quantized: target={target_position:.4f} → "
                        f"actual={exchange_pos:.4f} (contract rounding)",
                    )
                self._current_position = exchange_pos
                self._total_trades += 1
                self._total_fees += order.fee
                logger.info(
                    f"Trade executed: {order.side} {order.filled_quantity:.6f} "
                    f"@ {order.avg_fill_price:.2f}, fee={order.fee:.4f} USDT",
                )
            elif order.status == "partial":
                # FIX AUD-H06 + LIVE-01: Don't assume target on partial fill.
                # Sync actual position from exchange immediately so
                # _current_position reflects reality. Without this, the next
                # bar computes delta from stale position → double execution.
                self._total_trades += 1
                self._total_fees += order.fee
                try:
                    exchange_pos = await self.broker.get_single_position(self._asset)
                    self._current_position = exchange_pos
                except Exception as e:
                    logger.warning(f"Post-partial-fill position query failed: {e}")
                logger.warning(
                    f"Partial fill: {order.filled_quantity:.6f} of "
                    f"{order.quantity:.6f} — synced position to {self._current_position:.4f}",
                )
            else:
                logger.warning(f"Trade failed: {order.status} — {order.error}")
                # FIX S307: Skipped/failed orders are not actual trades
                # BUG-18: Rollback turnover so failed trades don't consume
                # the risk budget and block subsequent bars.
                self.risk_manager.rollback_last_turnover()
                self._prev_close = current_close
                # F-03: broker returned without raising — reset execution-error
                # counter so a prior transient HMDS error doesn't persist.
                self._consecutive_execution_errors = 0
                self._log_step(bar_time, target_position, traded=False, skip_reason="broker_skipped", regime_info=regime_info)
                return

        except Exception as e:
            self._consecutive_execution_errors += 1
            logger.error(
                f"Order execution error ({self._consecutive_execution_errors}/"
                f"{self._execution_error_threshold}): {e}",
            )
            # BUG-18: Rollback turnover on execution exception too
            self.risk_manager.rollback_last_turnover()
            # FIX AUD-L05: Update prev_close even on execution error
            self._prev_close = current_close
            if (
                self._emergency_flatten_on_error
                and self._consecutive_execution_errors >= self._execution_error_threshold
            ):
                logger.warning(
                    "Emergency flatten triggered: "
                    f"{self._consecutive_execution_errors} consecutive execution errors",
                )
                await self._emergency_flatten()
                self._request_stop("execution_error")
            return

        # Successful execution: reset consecutive-error counter.
        self._consecutive_execution_errors = 0

        # --- 8b. Persist position to file (crash recovery) ---
        self._write_position_file()

        # --- 9. (Reconciliation moved to _reconcile_all() at top of step) ---

        # --- 10. Update state ---
        self._prev_close = current_close

        # (Daily loss check moved to start of _trading_step — runs on ALL bars)

        # --- 11. Log ---
        self._log_step(bar_time, target_position, traded=True, order=order, regime_info=regime_info)

    # -------------------------------------------------------------------
    # Agent inference
    # -------------------------------------------------------------------
    def _predict(self, obs: dict) -> float:
        """Run SAC agent inference. Returns target position in [-1, 1]."""
        if self._obs_mode == "summary_stats":
            # Flat mode: concatenate all scale summary stats + private → (1, D)
            parts = [obs[f"scale_{i}"] for i in range(self._n_scales)]
            parts.append(obs["private"])
            flat = np.concatenate(parts)
            flat_tensor = torch.as_tensor(
                flat, dtype=torch.float32,
            ).unsqueeze(0).to(self._device, non_blocking=True)

            with torch.no_grad():
                action = self.agent.predict(
                    flat_tensor, None, deterministic=True,
                )
        else:
            scale_np = np.stack(
                [obs[f"scale_{i}"] for i in range(self._n_scales)], axis=0,
            )
            scale_tensor = torch.as_tensor(
                scale_np, dtype=torch.float32,
            ).unsqueeze(0).to(self._device, non_blocking=True)

            private_tensor = torch.as_tensor(
                obs["private"], dtype=torch.float32,
            ).unsqueeze(0).to(self._device, non_blocking=True)

            with torch.no_grad():
                action = self.agent.predict(
                    scale_tensor, private_tensor, deterministic=True,
                )

        return float(np.clip(action[0, 0].cpu().item(), -1.0, 1.0))

    def _check_signal_gate(self) -> bool:
        """Check if current bar passes signal gate. Returns True if gate open.

        Mirrors SignalGatedWrapper._gate_open() from training. Feature layout:
            idx 0: log_return, 1: atr_norm, 2: parkinson_vol, 7: volume_z
        """
        if not self._signal_gate_enabled:
            return True

        # First bar after startup always passes (matches training wrapper)
        if self._gate_is_first_bar and self._gate_always_on_first:
            self._gate_is_first_bar = False
            return True

        # Safety cap: force open after max consecutive holds
        if self._gate_consecutive_holds >= self._gate_max_hold_bars:
            self._gate_consecutive_holds = 0
            return True

        features = self.obs_builder._scale_features.get(self._gate_base_scale)
        if features is None or len(features) == 0:
            return True  # Passthrough if no features

        latest = features[-1]
        atr_norm = float(latest[1])
        parkinson = float(latest[2])
        volume_z = float(latest[7]) if len(latest) > 7 else 0.0

        if self._gate_mode == "return":
            gate_open = abs(float(latest[0])) > self._gate_return_threshold
        elif self._gate_mode == "atr":
            gate_open = atr_norm > self._gate_atr_threshold
        elif self._gate_mode == "parkinson":
            gate_open = parkinson > self._gate_parkinson_threshold
        elif self._gate_mode == "volume":
            gate_open = abs(volume_z) > self._gate_volume_threshold
        else:
            # Composite: ANY signal exceeding threshold opens the gate
            gate_open = (
                atr_norm > self._gate_atr_threshold
                or parkinson > self._gate_parkinson_threshold
                or abs(volume_z) > self._gate_volume_threshold
            )

        if gate_open:
            self._gate_consecutive_holds = 0
        else:
            self._gate_consecutive_holds += 1
            self._gate_total_skipped += 1

        return gate_open

    # -------------------------------------------------------------------
    # Bootstrap
    # -------------------------------------------------------------------
    async def _bootstrap_with_retry(
        self,
        max_retries: int = 5,
        base_delay: float = 60.0,
    ) -> None:
        """Bootstrap obs builder with exponential-backoff retry.

        HMDS can fail during CME maintenance windows or IB pacing hits.
        Instead of crashing, retry with increasing delays so the engine
        can start once data becomes available again.
        """
        for attempt in range(1, max_retries + 1):
            try:
                # FIX AUD: Start a background task to keep health file fresh
                # during long bootstraps (fetching 100K bars takes time).
                health_task = asyncio.create_task(self._health_heartbeat_loop())
                try:
                    await self.obs_builder.bootstrap(self.loader, self._asset)
                finally:
                    health_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await health_task
                return
            except Exception as e:
                if attempt == max_retries:
                    logger.critical(
                        f"Bootstrap failed after {max_retries} attempts: {e}"
                    )
                    raise
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"Bootstrap attempt {attempt}/{max_retries} failed: {e}. "
                    f"Retrying in {delay:.0f}s..."
                )
                await asyncio.sleep(delay)

    async def _health_heartbeat_loop(self) -> None:
        """Keep health file fresh during long blocking operations."""
        while True:
            self._write_bootstrap_health("bootstrapping")
            await asyncio.sleep(60)  # Refresh every minute

    # -------------------------------------------------------------------
    # Data fetching
    # -------------------------------------------------------------------
    async def _fetch_new_bars(self, bar_time: datetime) -> Optional[list[dict]]:
        """Fetch 1-min bars since last update.

        FIX AUD-C01: Uses CryptoLoader's public fetch_ohlcv() API instead of
        accessing private _get_exchange()/_to_symbol() methods.
        """
        try:
            latest_ts = self.obs_builder.get_latest_timestamp()
            if latest_ts is None:
                return None

            # Compute time range for fetch
            since_dt = latest_ts.to_pydatetime() + timedelta(minutes=1)
            end_dt = bar_time + timedelta(minutes=1)

            # Use loader's public API
            df = await self.loader.fetch_ohlcv(
                assets=[self._asset],
                start=since_dt.isoformat(),
                end=end_dt.isoformat(),
                timeframe="1m",
            )

            if df is None or len(df) == 0:
                return None

            # Filter to single asset and convert to list of dicts
            if 'ticker' in df.columns:
                df = df[df['ticker'] == self._asset].drop(columns=['ticker'])

            bars = []
            for _, row in df.iterrows():
                bars.append({
                    "timestamp": row["timestamp"],
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                })

            return bars

        except Exception as e:
            logger.error(f"Failed to fetch bars: {e}")
            return None

    # -------------------------------------------------------------------
    # Position management
    # -------------------------------------------------------------------
    async def _ensure_broker_connected(self) -> bool:
        """Check broker connection and reconnect if dropped.

        IB Gateway can silently drop connections during long idle waits
        between bars. cTrader TCP connections can also drop silently.
        This detects the drop and reconnects BEFORE the trading step,
        preventing cascading errors.

        FIX CT-03: Added cTrader connection check. Previously only checked
        IB, causing dropped cTrader connections to fail on the first API call
        and skip the entire bar.
        """
        try:
            # --- IB broker check ---
            ib = getattr(self.broker, "_ib", None)
            if ib is not None:
                if ib.isConnected():
                    return True
                logger.warning("IB connection lost — attempting reconnect")
                await self.broker.connect()
                # Re-wire loader references that point to the old IB session
                if hasattr(self.loader, "_ib"):
                    self.loader._ib = self.broker._ib
                if hasattr(self.loader, "_contract_manager"):
                    self.loader._contract_manager = self.broker._contract_manager
                return True

            # --- cTrader broker check ---
            ct_connected = getattr(self.broker, "_connected", None)
            if ct_connected is not None:
                if ct_connected:
                    return True
                logger.warning("cTrader connection lost — attempting reconnect")
                reconnect_fn = getattr(self.broker, "_reconnect", None)
                if reconnect_fn is not None:
                    success = await reconnect_fn()
                    if success:
                        # Re-wire loader's shared client reference
                        if hasattr(self.loader, "_client"):
                            self.loader._client = self.broker._client
                        return True
                    return False
                # Fallback: full reconnect via connect()
                await self.broker.connect()
                return getattr(self.broker, "_connected", False)

            # CCXT/HTTP brokers are stateless — always connected
            return True
        except Exception as e:
            logger.error(f"Broker reconnect failed: {e}")
            return False

    async def _sync_position(self) -> None:
        """Sync internal position from exchange on startup."""
        try:
            pos = await self.broker.get_single_position(self._asset)
            self._current_position = pos
            logger.info(f"Position synced from exchange: {pos:.4f}")
        except Exception as e:
            logger.warning(f"Could not sync position: {e}. Starting at 0.0")
            self._current_position = 0.0

    async def _check_orphaned_positions(self) -> None:
        """FIX CT-05: Detect orphaned positions on hedging-mode accounts.

        On cTrader hedging accounts, ``get_single_position()`` returns the
        NET of all positions — which can be 0.0 even with multiple opposing
        positions open.  This method calls ``check_orphaned_positions()``
        (if the broker supports it) to enumerate individual positions and
        warn about orphans at startup.

        Does NOT auto-close: orphan cleanup requires human review because
        closing the wrong side could realize a large loss.  The CRITICAL
        log ensures the operator is alerted immediately.
        """
        check_fn = getattr(self.broker, "check_orphaned_positions", None)
        if check_fn is None:
            # Broker doesn't support individual position enumeration
            # (e.g., CCXT, IB) — skip silently.
            return

        try:
            positions = await check_fn()
            n_positions = len(positions)

            if n_positions > 1:
                # Multiple individual positions — likely orphans from
                # hedging-mode NewOrderReq creating opposing positions.
                net_position = self._current_position
                logger.critical(
                    "CT-05: %d individual positions detected at startup "
                    "but net position = %.4f. This indicates orphaned "
                    "positions on a hedging-mode account. Positions: %s. "
                    "ACTION REQUIRED: Review and close orphans manually "
                    "or via emergency_flatten().",
                    n_positions,
                    net_position,
                    positions,
                )
            elif n_positions == 1 and abs(self._current_position) < 1e-6:
                # Single position exists but engine thinks it's flat —
                # net calculation may have zeroed it incorrectly.
                logger.warning(
                    "CT-05: 1 position exists on exchange but synced "
                    "position = %.4f. Position details: %s",
                    self._current_position,
                    positions[0],
                )
            elif n_positions == 0 and abs(self._current_position) > 1e-6:
                logger.warning(
                    "CT-05: No positions on exchange but synced "
                    "position = %.4f. Resetting to flat.",
                    self._current_position,
                )
                self._current_position = 0.0

        except Exception as e:
            logger.warning(
                "CT-05: Could not check for orphaned positions: %s", e
            )

    async def _reconcile_all(self, bar_time: datetime) -> None:
        """XVal Layer 1: Periodic broker position + PV cross-validation.

        Runs every ``_reconcile_interval`` bars.  On trade bars the post-fill
        broker sync (lines 510/534) already corrects position immediately, so
        this catches drift on *hold/deadband* bars where no trade executes.

        Emits Prometheus gauges for external cross-validation (Layer 2).
        """
        # Fast path: skip unless reconcile interval reached
        bars_since = self._total_bars - self._last_reconcile_bar
        if bars_since < self._reconcile_interval:
            return

        # Skip if broker is disconnected (would fail anyway)
        if not self._check_broker_alive():
            return

        self._last_reconcile_bar = self._total_bars

        try:
            exchange_pos = await self.broker.get_single_position(self._asset)
            pos_divergence = abs(exchange_pos - self._current_position)

            # PV cross-check
            info = await self.broker.get_account_info()
            broker_pv = info.get("total_equity", 0.0)
            pv_divergence_pct = (
                abs(broker_pv - self._portfolio_value) / self._portfolio_value
                if self._portfolio_value > 0
                else 0.0
            )

            # Emit Prometheus gauges (always, even if OK)
            self._metrics.update_reconciliation(
                broker_position=exchange_pos,
                position_divergence=pos_divergence,
                pv_divergence_pct=pv_divergence_pct,
                reconcile_timestamp=time.time(),
            )

            # Position divergence thresholds
            if pos_divergence > self._reconciliation_halt_pct:
                logger.critical(
                    f"POSITION MISMATCH: internal={self._current_position:.4f}, "
                    f"exchange={exchange_pos:.4f}, divergence={pos_divergence:.4f} "
                    f"(>{self._reconciliation_halt_pct:.0%}). HALTING.",
                )
                self._request_stop("position_mismatch")
            elif pos_divergence > self._reconciliation_warn_pct:
                logger.warning(
                    f"Position divergence: internal={self._current_position:.4f}, "
                    f"exchange={exchange_pos:.4f} — trusting exchange",
                )
                self._current_position = exchange_pos

            # PV divergence warning (informational, no auto-correction)
            if pv_divergence_pct > self._pv_divergence_warn_pct:
                logger.warning(
                    f"PV divergence: cached=${self._portfolio_value:,.2f}, "
                    f"broker=${broker_pv:,.2f} ({pv_divergence_pct:.1%})",
                )

        except Exception as e:
            logger.warning(f"Reconciliation failed: {e}")

    async def _update_portfolio_value(self) -> None:
        """Update portfolio value from exchange.

        FIX LIVE-03: Track consecutive zero-equity readings. If the broker
        returns 0 repeatedly, it's a real problem (not a glitch). Log critical
        after 3 consecutive zeros so it doesn't go unnoticed.
        """
        try:
            info = await self.broker.get_account_info()
            equity = info.get("total_equity", 0)
            if equity > 0:
                self._portfolio_value = equity
                self._peak_portfolio_value = max(self._peak_portfolio_value, equity)
                self._consecutive_zero_equity = 0
            else:
                self._consecutive_zero_equity = getattr(self, "_consecutive_zero_equity", 0) + 1
                if self._consecutive_zero_equity >= 3:
                    logger.critical(
                        f"Broker returned zero equity {self._consecutive_zero_equity} "
                        f"consecutive times — PV frozen at {self._portfolio_value:.2f}. "
                        f"Possible API issue or margin call.",
                    )
        except Exception as e:
            logger.warning(f"Portfolio value update failed: {e}")

    async def _inter_bar_metrics_loop(self) -> None:
        """Background task: refresh Prometheus metrics every 30s between bars.

        Fetches the latest portfolio value from the broker so Grafana shows
        real-time PV/drawdown changes instead of stale values from the last
        bar close.
        """
        while True:
            await asyncio.sleep(30)
            try:
                # FIX LIVE-02: Skip PV update if trading step is active to
                # prevent the metrics loop from mutating _portfolio_value
                # mid-calculation (used for risk checks + trade sizing).
                if not self._trading_step_active:
                    await self._update_portfolio_value()
                drawdown = 0.0
                if self._peak_portfolio_value > 0:
                    drawdown = 1.0 - self._portfolio_value / self._peak_portfolio_value
                daily_loss_pct = (
                    (self._portfolio_value - self._daily_start_value)
                    / self._daily_start_value
                    if self._daily_start_value > 0
                    else 0.0
                )
                self._metrics.update(
                    position=self._current_position,
                    portfolio_value=self._portfolio_value,
                    drawdown_pct=drawdown,
                    daily_loss_pct=daily_loss_pct,
                    broker_connected=self._check_broker_alive(),
                    bar_count=self._total_bars,
                    total_trades=self._total_trades,
                    total_fees=self._total_fees,
                    consecutive_errors=self._consecutive_errors,
                    last_bar_timestamp=self._last_bar_timestamp or time.time(),
                    funding_rate=self._current_funding_rate,
                )
            except Exception as e:
                # FIX ENG-02: Log instead of silently swallowing — systematic
                # failures (e.g., bad metric name) would otherwise go unnoticed
                logger.debug(f"Inter-bar metrics update failed: {e}")

    async def _update_funding_rate(self) -> None:
        """FIX AUD-H07: Fetch current funding rate for risk manager."""
        try:
            rates = await self.broker.get_funding_rates([self._asset])
            self._current_funding_rate = rates.get(self._asset, 0.0)
            logger.debug(f"Funding rate updated: {self._current_funding_rate:.6f}")
            # Update funding rate EMA for gate + metrics
            if self._fr_gate_enabled:
                self._update_funding_ema(self._current_funding_rate)
                self._metrics.update_funding_gate(
                    ema_value=self._fr_gate_ema_value,
                    gate_open=self._fr_gate_ema_value > self._fr_gate_threshold,
                )
        except Exception as e:
            logger.debug(f"Funding rate fetch failed: {e}")

    def _update_funding_ema(self, raw_rate: float) -> None:
        """Update EMA of annualized funding rate for the gate."""
        ann_rate = raw_rate * 3 * 365  # per-8h → annualized
        if self._fr_gate_n_updates == 0:
            self._fr_gate_ema_value = ann_rate
        else:
            alpha = self._fr_gate_ema_alpha
            self._fr_gate_ema_value = alpha * ann_rate + (1 - alpha) * self._fr_gate_ema_value
        self._fr_gate_n_updates += 1

    def _check_funding_rate_gate(self) -> bool:
        """Check if funding rate EMA is above borrow cost threshold.

        Returns True if trading should proceed, False if gate is closed.
        """
        if not self._fr_gate_enabled:
            return True

        # Warmup: allow trading until we have enough EMA samples
        if self._fr_gate_n_updates < self._fr_gate_min_warmup:
            return True

        is_open = self._fr_gate_ema_value > self._fr_gate_threshold
        if not is_open:
            self._fr_gate_total_skipped += 1
            if self._fr_gate_total_skipped % 10 == 1:
                logger.info(
                    f"Funding gate CLOSED: EMA={self._fr_gate_ema_value * 100:.1f}% "
                    f"< threshold={self._fr_gate_threshold * 100:.1f}% "
                    f"(skipped {self._fr_gate_total_skipped} bars)",
                )
        return is_open

    async def _maybe_roll_contract(self, bar_time: datetime) -> None:
        """Check once per UTC day whether the futures front-month needs to roll.

        Only runs when the broker exposes a ``_contract_manager`` (futures).
        If a roll is needed, flatten any open position first (required by
        ``FuturesContractManager.roll_contract``), then roll. If the flatten
        partially fails, request stop rather than trading across two contracts.
        """
        cm = getattr(self.broker, "_contract_manager", None)
        if cm is None:
            return
        today = bar_time.astimezone(timezone.utc).date()
        if self._last_roll_check_date == today:
            return
        self._last_roll_check_date = today
        try:
            if not cm.check_roll_needed(as_of=today):
                return
        except Exception as e:
            logger.warning(f"Roll check failed: {e}")
            return

        old_symbol = getattr(cm.contract, "local_symbol", "?")
        logger.warning(f"Contract roll required for {old_symbol} — flattening first")
        if abs(self._current_position) > 1e-6:
            try:
                result = await self.broker.emergency_flatten([self._asset])
                if result.n_failed != 0:
                    logger.critical(
                        "Roll flatten partially failed — refusing to roll, "
                        "requesting stop to avoid cross-contract exposure",
                    )
                    self._request_stop("roll_flatten_failed")
                    return
                self._current_position = 0.0
            except Exception as e:
                logger.critical(f"Roll flatten raised: {e} — requesting stop")
                self._request_stop("roll_flatten_error")
                return
        try:
            new_info = await cm.roll_contract()
            logger.warning(
                f"Rolled futures contract: {old_symbol} -> {new_info.local_symbol}",
            )
        except Exception as e:
            logger.critical(f"Roll failed: {e} — requesting stop")
            self._request_stop("roll_failed")

    async def _emergency_flatten(self) -> None:
        """Close all positions via market orders.

        FIX AUD-C05: Only set position to 0 if flatten actually succeeded.
        """
        try:
            result = await self.broker.emergency_flatten([self._asset])
            if result.n_failed == 0:
                self._current_position = 0.0
                logger.info("Emergency flatten succeeded — position zeroed")
            else:
                logger.critical(
                    f"EMERGENCY FLATTEN PARTIAL: {result.n_failed} orders failed. "
                    f"Position may still be open on exchange!",
                )
        except Exception as e:
            logger.critical(f"EMERGENCY FLATTEN FAILED: {e}. Position may be open!")

    # -------------------------------------------------------------------
    # Safety checks
    # -------------------------------------------------------------------
    def _check_obs_sanity(self, obs: dict) -> bool:
        """Check observation for NaN/Inf values."""
        for key, arr in obs.items():
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                logger.warning(f"Observation '{key}' contains NaN/Inf")
                return False
        return True

    async def _check_daily_loss(self, bar_time: datetime) -> None:
        """Check if daily loss limit exceeded using median-smoothed PV.

        FIX AUD-H04: Reset at UTC midnight instead of bar count.

        Uses median of the last 3 portfolio value readings to prevent false
        triggers from broker API hiccups returning anomalous values. A single
        bad reading cannot trip the limit on its own.
        """
        current_date = bar_time.date()

        if self._last_daily_reset_date is None or current_date != self._last_daily_reset_date:
            self._daily_start_value = self._portfolio_value
            self._last_daily_reset_date = current_date
            # Reset PV buffer on new day to avoid stale cross-day values
            self._pv_buffer.clear()

        # Append current reading to smoothing buffer
        self._pv_buffer.append(self._portfolio_value)

        if self._daily_start_value > 0:
            # Use median of buffered readings for robustness
            smoothed_pv = float(np.median(list(self._pv_buffer)))
            smoothed_return = (smoothed_pv - self._daily_start_value) / self._daily_start_value

            # Also compute single-reading return for comparison
            raw_return = (self._portfolio_value - self._daily_start_value) / self._daily_start_value

            if smoothed_return < -self._max_daily_loss_pct:
                logger.critical(
                    f"DAILY LOSS LIMIT: smoothed={smoothed_return:.2%} "
                    f"(raw={raw_return:.2%}) < -{self._max_daily_loss_pct:.0%}. "
                    f"Flattening and stopping trading.",
                )
                # FIX RSK-02: Flatten before stop — main loop exits on
                # should_stop without closing open positions otherwise.
                try:
                    await self._emergency_flatten()
                finally:
                    self._write_halt_state(
                        reason="daily_loss_limit",
                        detail=f"smoothed_return={smoothed_return:.4f}",
                        now_utc=datetime.now(timezone.utc),
                    )
                    self._request_stop("daily_loss_limit")
            elif raw_return < -self._max_daily_loss_pct:
                # Single reading breached but median didn't — likely API hiccup
                logger.debug(
                    f"Daily loss: raw={raw_return:.2%} breached limit but "
                    f"smoothed={smoothed_return:.2%} did not "
                    f"(buffer={[round(v, 2) for v in self._pv_buffer]}). "
                    f"Suppressing false trigger.",
                )

    # -------------------------------------------------------------------
    # Weekend flatten (CFD markets)
    # -------------------------------------------------------------------
    async def _check_weekend_flatten(self, bar_time: datetime) -> bool:
        """Flatten positions before Friday market close to avoid weekend gap risk.

        Returns True if the bar was handled (flattened or held flat), meaning
        the caller should skip normal trading logic for this bar.

        Resets the flatten flag on non-Friday bars so the strategy resumes
        normally when the market reopens on Sunday.
        """
        if not self._weekend_flatten_enabled:
            return False

        # Reset the weekly flag on non-Friday bars (market reopened)
        if bar_time.weekday() != 4:
            self._weekend_flatten_done = False
            return False

        # Friday: check if we're within the flatten window
        if hasattr(self.bar_clock, "minutes_to_friday_close"):
            remaining = self.bar_clock.minutes_to_friday_close(bar_time)
            if remaining is None:
                return False  # Not Friday per the clock (shouldn't happen)
        else:
            # Not a CFD bar clock — skip
            return False

        if remaining > self._weekend_flatten_minutes:
            return False  # Still far from close, trade normally

        # Within flatten window
        if not self._weekend_flatten_done and abs(self._current_position) > 0.01:
            logger.warning(
                f"WEEKEND FLATTEN: {remaining:.0f} min to Friday close. "
                f"Flattening position {self._current_position:.4f} to avoid "
                f"weekend gap risk.",
            )
            await self._emergency_flatten()
            self.risk_manager.reset(self._portfolio_value)
            self._weekend_flatten_done = True
            self._log_step(
                bar_time, 0.0,
                traded=True, skip_reason="weekend_flatten",
            )
        elif not self._weekend_flatten_done:
            logger.info(
                f"Weekend flatten window active ({remaining:.0f} min to close) "
                f"but already flat.",
            )
            self._weekend_flatten_done = True
        # else: already flattened this week, just hold

        return True  # Skip normal trading — we're in pre-close hold

    # -------------------------------------------------------------------
    # Position persistence (crash recovery)
    # -------------------------------------------------------------------
    def _write_position_file(self) -> None:
        """Atomically write current position to disk for crash recovery.

        Uses write-to-tmpfile + os.rename for atomicity. Non-critical:
        if this fails, trading continues unaffected.
        """
        data = {
            "position": round(self._current_position, 6),
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "strategy": self._strategy_name,
            "broker_position": round(self._current_position, 6),
        }
        try:
            parent = self._position_file.parent
            parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(parent), suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f)
                os.replace(tmp_path, str(self._position_file))
            except BaseException:
                # Clean up tmpfile on any error
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
                raise
        except Exception as e:
            logger.debug(f"Position file write failed: {e}")

    def _check_persisted_position(self) -> None:
        """On startup, compare persisted position file with broker position.

        Logs WARNING if they differ by more than 5%, indicating a potential
        crash-recovery mismatch that needs manual attention.
        """
        if not self._position_file.exists():
            logger.info("No persisted position file found — clean start")
            return

        try:
            data = json.loads(self._position_file.read_text(encoding="utf-8"))
            persisted_pos = float(data.get("position", 0.0))
            broker_pos = self._current_position  # Already synced from exchange

            discrepancy = abs(persisted_pos - broker_pos)
            if discrepancy > 0.05:
                logger.warning(
                    f"POSITION MISMATCH on startup: persisted={persisted_pos:.4f} "
                    f"(from {data.get('timestamp', 'unknown')}), "
                    f"broker={broker_pos:.4f}, "
                    f"discrepancy={discrepancy:.4f} (>{0.05:.0%}). "
                    f"Trusting broker position. Investigate if unexpected.",
                )
            else:
                logger.info(
                    f"Persisted position matches broker: "
                    f"persisted={persisted_pos:.4f}, broker={broker_pos:.4f}",
                )
        except Exception as e:
            logger.warning(f"Could not read persisted position file: {e}")

    def _request_stop(self, reason: str) -> None:
        """Request graceful shutdown."""
        logger.info(f"Stop requested: {reason}")
        self._should_stop = True

    # -------------------------------------------------------------------
    # S427: Persistent halt state + intra-bar DD projection
    # -------------------------------------------------------------------
    @staticmethod
    def _next_utc_midnight(now_utc: datetime) -> datetime:
        return (now_utc + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )

    def _write_halt_state(
        self, reason: str, detail: str, now_utc: datetime,
    ) -> None:
        """Persist halt metadata so restarts stay halted until UTC day rollover.

        On permission errors (misconfigured container with non-writable
        state dir), fall back to /tmp/risk_state.json and rewrite
        ``self._halt_state_file`` so the persistent-halt gate on the next
        startup reads from the same location. A disarmed gate would let
        Docker ``restart: unless-stopped`` revive a halted engine into
        another breach — log CRITICAL so the regression is visible.
        """
        halted_until = self._next_utc_midnight(now_utc)
        payload = {
            "halted_until": halted_until.isoformat(),
            "halted_at": now_utc.isoformat(),
            "reason": reason,
            "detail": detail,
            "strategy": self._strategy_name,
            "daily_start_value": self._daily_start_value,
            "portfolio_value": self._portfolio_value,
        }

        candidates = [self._halt_state_file]
        fallback = Path("/tmp/risk_state.json")
        if self._halt_state_file != fallback:
            candidates.append(fallback)

        for idx, target in enumerate(candidates):
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(payload, indent=2))
                tmp.replace(target)
                if idx > 0:
                    logger.critical(
                        f"HALT STATE FALLBACK: {self._halt_state_file} "
                        f"not writable; persisted to {target}. "
                        f"Fix container state-dir permissions.",
                    )
                    self._halt_state_file = target
                logger.critical(
                    f"HALT STATE WRITTEN: {target} "
                    f"halted_until={halted_until.isoformat()} reason={reason}",
                )
                return
            except Exception as e:
                logger.error(
                    f"Failed to write halt state to {target}: {e}",
                    exc_info=True,
                )

        logger.critical(
            "HALT STATE NOT PERSISTED: all write targets failed. "
            "Docker may revive a halted engine — manual intervention required.",
        )

    def _clear_halt_state(self) -> None:
        try:
            if self._halt_state_file.exists():
                self._halt_state_file.unlink()
                logger.info(f"Cleared halt state: {self._halt_state_file}")
        except Exception as e:
            logger.warning(f"Failed to clear halt state: {e}")

    async def _check_persistent_halt(self) -> bool:
        """Gate startup on a prior halt. Returns True if start() should exit."""
        if not self._halt_state_file.exists():
            return False
        try:
            data = json.loads(self._halt_state_file.read_text())
            halted_until = datetime.fromisoformat(data["halted_until"])
            if halted_until.tzinfo is None:
                halted_until = halted_until.replace(tzinfo=timezone.utc)
        except Exception as e:
            logger.error(
                f"Halt state file unreadable ({e}); removing and continuing.",
            )
            with contextlib.suppress(Exception):
                self._halt_state_file.unlink()
            return False

        now = datetime.now(timezone.utc)
        if halted_until <= now:
            logger.info(
                f"Halt expired at {halted_until.isoformat()} "
                f"(reason={data.get('reason')}). Clearing state and starting.",
            )
            self._clear_halt_state()
            return False

        sleep_seconds = (halted_until - now).total_seconds()
        logger.critical(
            f"PERSISTENT HALT ACTIVE: reason={data.get('reason')} "
            f"detail={data.get('detail')} halted_at={data.get('halted_at')} "
            f"halted_until={halted_until.isoformat()} "
            f"sleeping {sleep_seconds:.0f}s before release.",
        )

        # Heartbeat every 30s so healthcheck.sh doesn't flag us UNHEALTHY
        # and docker doesn't kill+restart the container in a tight loop.
        self._write_bootstrap_health("risk_halt_sleeping")
        heartbeat_interval = 30.0
        remaining = sleep_seconds
        while remaining > 0 and not self._should_stop:
            chunk = min(heartbeat_interval, remaining)
            try:
                await asyncio.sleep(chunk)
            except asyncio.CancelledError:
                logger.info("Halt sleep cancelled.")
                return True
            remaining -= chunk
            self._write_bootstrap_health("risk_halt_sleeping")

        if self._should_stop:
            return True

        self._clear_halt_state()
        logger.info("Persistent halt released — resuming normal startup.")
        return False

    async def _check_intrabar_dd(self, bar_time: datetime) -> None:
        """Project worst-case intra-bar PV against daily-loss limit.

        Uses the latest base-scale bar's adverse extreme (low for long,
        high for short) to estimate the maximum adverse excursion that
        occurred within the bar. If projected daily return would breach
        the limit, flatten and halt (persistently).
        """
        if not self._intrabar_dd_enabled:
            return
        if abs(self._current_position) < 1e-9:
            return
        if self._daily_start_value <= 0:
            return

        high, low = self.obs_builder.get_current_hl()
        close = self.obs_builder.get_current_close()
        if close <= 0 or high <= 0 or low <= 0:
            return

        # Position is a unit-less target in [-1, 1] representing fraction of
        # portfolio_value deployed as notional. Worst-case intra-bar PnL:
        #   long  → price visits `low`   → pnl = pos * pv * (low  - close) / close
        #   short → price visits `high`  → pnl = pos * pv * (high - close) / close
        adverse_price = low if self._current_position > 0 else high
        adverse_pnl = (
            self._current_position * self._portfolio_value
            * (adverse_price - close) / close
        )
        worst_pv = self._portfolio_value + adverse_pnl
        projected_return = (worst_pv - self._daily_start_value) / self._daily_start_value

        if projected_return < -self._max_daily_loss_pct:
            logger.critical(
                f"INTRA-BAR DD PROJECTION: pos={self._current_position:+.4f} "
                f"close={close:.4f} adverse={adverse_price:.4f} "
                f"worst_pv={worst_pv:.2f} daily_start={self._daily_start_value:.2f} "
                f"projected_return={projected_return:.2%} < -{self._max_daily_loss_pct:.0%}. "
                f"Flattening and halting.",
            )
            try:
                await self._emergency_flatten()
            finally:
                self._write_halt_state(
                    reason="intrabar_dd_projection",
                    detail=(
                        f"projected_return={projected_return:.4f} "
                        f"pos={self._current_position:.4f} "
                        f"adverse={adverse_price:.4f} close={close:.4f}"
                    ),
                    now_utc=datetime.now(timezone.utc),
                )
                self._request_stop("intrabar_dd_projection")

    # -------------------------------------------------------------------
    # WandB logging
    # -------------------------------------------------------------------
    def _init_wandb(self) -> None:
        """Initialize WandB run for live monitoring.

        Uses a deterministic run ID so restarts resume the same WandB run
        instead of creating duplicates. One run per strategy.
        """
        wandb_cfg = self.config.get("wandb", {})
        if not wandb_cfg.get("project"):
            return

        try:
            import wandb

            # Deterministic ID: one WandB run per (asset, exchange, mode) tuple
            run_id = wandb_cfg.get("run_id") or self._make_wandb_run_id()

            self._wandb_run = wandb.init(
                project=wandb_cfg.get("project", "FinRL-Pro-DS"),
                entity=wandb_cfg.get("entity"),
                tags=wandb_cfg.get("tags", ["live"]),
                config=self.config,
                name=f"live_{self._asset}_{self.broker.exchange_id}",
                id=run_id,
                resume="allow",
            )
            logger.info(
                f"WandB run initialized: {self._wandb_run.url} "
                f"(id={run_id}, resume=allow)",
            )
        except Exception as e:
            logger.warning(f"WandB init failed: {e}")

    def _make_wandb_run_id(self) -> str:
        """Generate deterministic WandB run ID from strategy config.

        Format: ``live-{asset}-{exchange}-{mode}``
        e.g. ``live-gc-ib-paper`` or ``live-btc-bybit-live``
        """
        asset = self._asset.lower()
        exchange = self.broker.exchange_id.lower()
        mode = "paper" if getattr(self.broker, "testnet", True) else "live"
        return f"live-{asset}-{exchange}-{mode}"

    def _log_step(
        self,
        bar_time: datetime,
        target_position: float,
        traded: bool,
        skip_reason: str = "",
        order=None,
        regime_info: dict | None = None,
    ) -> None:
        """Log step metrics to WandB and logger."""
        drawdown = 0.0
        if self._peak_portfolio_value > 0:
            drawdown = 1.0 - self._portfolio_value / self._peak_portfolio_value

        metrics = {
            "bar": self._total_bars,
            "position": self._current_position,
            "target_position": target_position,
            "portfolio_value": self._portfolio_value,
            "drawdown_pct": drawdown,
            "total_trades": self._total_trades,
            "total_fees": self._total_fees,
            "traded": int(traded),
            "funding_rate": self._current_funding_rate,
        }

        if order is not None:
            metrics["order_fee"] = order.fee
            metrics["order_fill_price"] = order.avg_fill_price

        # PRISM regime info (if available)
        if regime_info and not regime_info.get("fallback", False):
            metrics["prism_composite_code"] = regime_info.get("composite_code", -1)
            metrics["prism_vol_regime"] = regime_info.get("vol_regime", "unknown")
            metrics["prism_price_regime"] = regime_info.get("price_regime", "unknown")
            metrics["prism_multiplier"] = regime_info.get("multiplier", 1.0)
            metrics["prism_confidence"] = regime_info.get("confidence", 0.0)

        if self._wandb_run is not None:
            try:
                import wandb
                wandb.log(metrics, commit=True)
            except Exception as e:
                logger.debug(f"WandB log failed: {e}")

        # Shared state for health file + Prometheus (compute once)
        broker_connected = self._check_broker_alive()
        daily_loss_pct = (
            (self._portfolio_value - self._daily_start_value)
            / self._daily_start_value
            if self._daily_start_value > 0
            else 0.0
        )

        # Health status file for Docker HEALTHCHECK
        self._write_health_status(
            bar_time, drawdown, daily_loss_pct, broker_connected,
        )

        # Prometheus metrics (thread-safe, no-op if disabled)
        self._last_bar_timestamp = bar_time.timestamp()
        self._metrics.update(
            position=self._current_position,
            portfolio_value=self._portfolio_value,
            drawdown_pct=drawdown,
            daily_loss_pct=daily_loss_pct,
            broker_connected=broker_connected,
            bar_count=self._total_bars,
            total_trades=self._total_trades,
            total_fees=self._total_fees,
            consecutive_errors=self._consecutive_errors,
            last_bar_timestamp=self._last_bar_timestamp,
            funding_rate=self._current_funding_rate,
        )

        # PRISM Prometheus metrics (no-op if disabled)
        if regime_info:
            self._metrics.update_prism(
                multiplier=regime_info.get("multiplier", 1.0),
                composite_code=regime_info.get("composite_code", -1),
                latency_ms=regime_info.get("latency_ms", 0.0),
                is_fallback=regime_info.get("fallback", False),
                is_error="error" in regime_info,
            )

        # Periodic console log
        if self._total_bars % 4 == 0 or traded:
            action_str = "TRADE" if traded else f"HOLD({skip_reason})"
            logger.info(
                f"[Bar {self._total_bars}] {bar_time.strftime('%H:%M')} UTC | "
                f"{action_str} | pos={self._current_position:.3f} | "
                f"PV=${self._portfolio_value:,.2f} | DD={drawdown:.2%}",
            )

    # -------------------------------------------------------------------
    # Health status file
    # -------------------------------------------------------------------
    def _check_broker_alive(self) -> bool:
        """Check broker connection status without async call.

        IB brokers have a persistent TCP connection that can drop.
        cTrader brokers have a Twisted TCP connection with _connected flag.
        CCXT/HTTP brokers are stateless — always returns True.

        FIX CT-02: Added cTrader connection check. Previously always returned
        True for cTrader, causing health JSON + Prometheus to show false-positive
        broker_connected=True when the TCP connection was actually down.
        """
        # IB broker check
        ib = getattr(self.broker, "_ib", None)
        if ib is not None:
            return ib.isConnected()
        # cTrader broker check
        ct_connected = getattr(self.broker, "_connected", None)
        if ct_connected is not None:
            return bool(ct_connected)
        return True

    def _write_bootstrap_health(self, phase: str = "bootstrapping") -> None:
        """Write health file during startup so Docker doesn't kill us.

        Called before/after bootstrap and after init — keeps the health
        file fresh while we wait (up to 15 min) for the first bar.
        """
        status = {
            "timestamp": time.time(),
            "bar_count": 0,
            "last_bar_time": datetime.now(tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ",
            ),
            "position": 0.0,
            "portfolio_value": 0.0,
            "drawdown_pct": 0.0,
            "daily_loss_pct": 0.0,
            "broker_connected": True,
            "consecutive_errors": 0,
            "total_trades": 0,
            "should_stop": False,
            "strategy_name": self._strategy_name,
            "phase": phase,
        }
        try:
            tmp = self._health_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(status), encoding="utf-8")
            tmp.replace(self._health_file)
        except Exception as e:
            logger.debug(f"Bootstrap health write failed: {e}")

    def _write_health_status(
        self,
        bar_time: datetime,
        drawdown: float,
        daily_loss_pct: float,
        broker_connected: bool,
    ) -> None:
        """Write health status JSON for Docker HEALTHCHECK.

        Non-critical: if this fails, trading continues unaffected.
        The healthcheck.sh script reads this file and checks:
          - staleness (file age > MAX_STALE_SECONDS)
          - broker_connected, consecutive_errors, should_stop
        """
        status = {
            "timestamp": time.time(),
            "bar_count": self._total_bars,
            "last_bar_time": bar_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "position": round(self._current_position, 6),
            "portfolio_value": round(self._portfolio_value, 2),
            "drawdown_pct": round(drawdown, 6),
            "daily_loss_pct": round(daily_loss_pct, 6),
            "broker_connected": broker_connected,
            "consecutive_errors": self._consecutive_errors,
            "consecutive_execution_errors": self._consecutive_execution_errors,
            "total_trades": self._total_trades,
            "should_stop": self._should_stop,
            "strategy_name": self._strategy_name,
        }

        try:
            tmp = self._health_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(status), encoding="utf-8")
            tmp.replace(self._health_file)
        except Exception as e:
            logger.debug(f"Health status write failed: {e}")

    # -------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------
    async def _shutdown(self) -> None:
        """Graceful shutdown: reconcile, log, close WandB, close broker.

        FIX AUD-C02: Reconcile position on shutdown to detect orphaned positions.
        FIX AUD-M04: Close loader exchange connection too.
        FIX LIVE-08: Flatten open positions on safety stops (daily loss, consecutive
        errors) to prevent orphaned exposure. Configurable via safety.flatten_on_stop.
        """
        flatten_on_stop = self.config.get("safety", {}).get("flatten_on_stop", True)

        # Final position reconciliation — flatten if configured
        try:
            exchange_pos = await self.broker.get_single_position(self._asset)
            if abs(exchange_pos) > 0.01:
                if flatten_on_stop:
                    logger.warning(
                        f"SHUTDOWN: Flattening open position {exchange_pos:.4f} "
                        f"(safety.flatten_on_stop=true)",
                    )
                    await self._emergency_flatten()
                else:
                    logger.warning(
                        f"SHUTDOWN: POSITION STILL OPEN on exchange: {exchange_pos:.4f}. "
                        f"Position will be ORPHANED. Flatten manually or restart.",
                    )
            else:
                logger.info(f"Shutdown: exchange position is flat ({exchange_pos:.4f})")
            if abs(exchange_pos - self._current_position) > 0.01:
                logger.warning(
                    f"SHUTDOWN: Position discrepancy: internal={self._current_position:.4f}, "
                    f"exchange={exchange_pos:.4f}",
                )
        except Exception as e:
            logger.warning(f"Shutdown position check failed: {e}")

        logger.info(
            f"=== TRADING STOPPED ===\n"
            f"  Bars: {self._total_bars}\n"
            f"  Trades: {self._total_trades}\n"
            f"  Fees: ${self._total_fees:.4f}\n"
            f"  Final PV: ${self._portfolio_value:,.2f}\n"
            f"  Position: {self._current_position:.4f}",
        )

        if self._wandb_run is not None:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

        try:
            await self.broker.close()
        except Exception:
            pass

        # FIX AUD-M04: Close loader's internal exchange connection
        try:
            if hasattr(self.loader, '_exchange') and self.loader._exchange is not None:
                await self.loader._exchange.close()
        except Exception:
            pass
