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
from datetime import date, datetime, timedelta, timezone  # noqa: F401 (date used in annotations)
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from sharpen.monitoring import (
    ActionDriftTracker,
    AgreementDecayTracker,
    CONSENSUS_RULES,
    CostDriftStatus,
    CostDriftTracker,
    REASON_AGREEMENT_DECAY_CRIT,
    REASON_DRIFT_CRIT,
    read_kill_file,
    should_lockout,
    write_kill_file,
)
from sharpen.live.challenge_state_machine import (
    ChallengePhase,
    ChallengeStateMachine,
)

logger = logging.getLogger(__name__)


def _init_drift_tracker(config: dict) -> Optional[ActionDriftTracker]:
    """Build a Protocol v2.2 §8.2 ActionDriftTracker from engine config.

    Returns None (tracker disabled) when:
      * `drift.enabled` is false/missing, OR
      * baseline_path is missing / unreadable / has no eval_distribution.
    Baseline resolution:
      * `ensemble_report.json` → reads `ensemble_eval_distribution`
      * `seed_report.json` with `drift.baseline_seed` → reads
         `eval_distribution_by_seed.<seed>`
    The tracker itself is permissive: an unreadable baseline returns
    status=LOG_ONLY, which lets live monitoring run in observability-only
    mode until T5 backfill writes proper artifacts.
    """
    drift_cfg = config.get("drift") or {}
    if not drift_cfg.get("enabled", False):
        return None

    baseline_path = drift_cfg.get("baseline_path")
    baseline: Optional[dict] = None
    if baseline_path:
        try:
            with open(baseline_path, encoding="utf-8") as f:
                payload = json.load(f)
            if "ensemble_eval_distribution" in payload:
                baseline = payload["ensemble_eval_distribution"]
            elif "eval_distribution_by_seed" in payload:
                seed_key = str(drift_cfg.get("baseline_seed", ""))
                by_seed = payload["eval_distribution_by_seed"]
                if seed_key and seed_key in by_seed:
                    baseline = by_seed[seed_key]
                elif by_seed:
                    # Fall back to the first seed with a stable sort so repeat
                    # restarts pick the same baseline.
                    first = sorted(by_seed.keys())[0]
                    baseline = by_seed[first]
                    logger.warning(
                        f"drift.baseline_seed unset; using seed {first} from "
                        f"{baseline_path}",
                    )
            elif "eval_distribution" in payload:
                baseline = payload["eval_distribution"]
            else:
                logger.warning(
                    f"drift baseline at {baseline_path} has no recognized "
                    f"eval_distribution block — running in LOG_ONLY",
                )
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                f"drift baseline load failed ({e}) — running in LOG_ONLY",
            )

    safe_cfg = config.get("safe_mode") or {}
    # Scalar V7 deadband matches the env's deadband_threshold so the live
    # tracker and training baseline bucket the same way.
    deadband_abs = float(config.get("trading", {}).get(
        "deadband_threshold",
        config.get("env", {}).get("deadband_threshold", 0.25),
    ))
    # Prefer drift.* keys (v2.2 spec); fall back to top-level `gates.drift.*`
    # if callers pass them through that path.
    gates_drift = (config.get("gates") or {}).get("drift") or {}
    gates_safe = (config.get("gates") or {}).get("safe_mode") or {}

    def _pick(key: str, default: float) -> float:
        for src in (drift_cfg, safe_cfg, gates_drift, gates_safe):
            if key in src:
                return float(src[key])
        return float(default)

    cutpoints = drift_cfg.get("regime_cutpoints") or gates_drift.get("regime_cutpoints")
    # v2.6 (S551-cont-4): feature_variance_veto.max_veto_frac flows to the
    # tracker so it can run the ADR-5 escalation. The veto's enabled/scale
    # toggles live on the engine (read separately) — the tracker doesn't need
    # to know whether the engine WILL pass feature_state; an unused max_veto_frac
    # just sits idle (default 0.50, ratio stays at 0 when no FLAT bars seen).
    veto_block = (gates_drift.get("feature_variance_veto") or {})
    max_veto_frac = float(veto_block.get("max_veto_frac", 0.50))
    tracker = ActionDriftTracker(
        baseline=baseline,
        window_bars=int(_pick("window_bars", 1000)),
        min_bars_before_check=int(_pick("min_bars_before_check", 500)),
        deadband_warn=_pick("deadband_frac_warn", 0.15),
        deadband_crit=_pick("deadband_frac_crit", 0.30),
        saturation_warn=_pick("saturation_frac_warn", 0.15),
        saturation_crit=_pick("saturation_frac_crit", 0.30),
        action_kl_warn=_pick("action_kl_warn", 0.5),
        action_kl_crit=_pick("action_kl_crit", 1.0),
        deadband_abs=deadband_abs,
        regime_cutpoints=cutpoints,
        vol_estimator_bars=int(_pick("vol_estimator_bars", 20)),
        max_veto_frac=max_veto_frac,
    )
    logger.info(
        f"ActionDriftTracker active: baseline={'YES' if baseline else 'LOG_ONLY'} "
        f"window={tracker.window_bars} warmup={tracker.min_bars_before_check}",
    )
    return tracker


def _init_agreement_decay_tracker(
    config: dict,
    agent,
) -> Optional[AgreementDecayTracker]:
    """Build a Protocol v2.3 §8.2-extension AgreementDecayTracker.

    Returns None (tracker disabled) when:
      * the live agent isn't a consensus-rule ensemble (`ens_agreement` /
        `ens_majority`), OR
      * `drift.enabled` is false (we piggy-back on the same enable flag —
        gating the silent-death detector while running drift in LOG_ONLY
        is incoherent), OR
      * the baseline file is missing / has no `ensemble_eval_distribution`.

    Baseline is the post-aggregation `deadband_frac` from
    `ensemble_report.json → ensemble_eval_distribution`. The same
    `drift.baseline_path` that feeds ActionDriftTracker is reused so the
    operator declares ONE artifact path.
    """
    rule = getattr(agent, "aggregation_rule", None)
    if rule not in CONSENSUS_RULES:
        return None

    drift_cfg = config.get("drift") or {}
    if not drift_cfg.get("enabled", False):
        return None

    baseline_flat: Optional[float] = None
    baseline_path = drift_cfg.get("baseline_path")
    if baseline_path:
        try:
            with open(baseline_path, encoding="utf-8") as f:
                payload = json.load(f)
            ens = payload.get("ensemble_eval_distribution")
            if isinstance(ens, dict) and "deadband_frac" in ens:
                # F2-AUD-02 guard (S548-cont): pre-Fix-2 baselines folded
                # no-consensus zeros into deadband_frac (aggregator returned
                # 0 on consensus failure); post-Fix-2 baselines exclude NaN
                # bars from numerator+denominator. Pairing an old baseline
                # with a consensus-rule live deploy would compute a biased
                # `flat_frac_delta`. Detect via the `no_consensus_frac` field
                # — its presence is the implicit Fix-2 marker. Degrade to
                # LOG_ONLY (preserves WandB telemetry; suppresses false
                # WARN/CRIT) and require operator re-bake under Fix 2.
                if "no_consensus_frac" not in ens:
                    logger.warning(
                        f"agreement-decay baseline at {baseline_path} "
                        f"lacks `no_consensus_frac` — pre-Fix-2 schema. "
                        f"Pairing with rule={rule!r} would compute biased "
                        f"flat_frac_delta (baseline includes consensus-failure "
                        f"bars in deadband_frac denominator; live tracker "
                        f"excludes NaN). Running in LOG_ONLY — re-bake "
                        f"baseline under Fix 2 to re-enable WARN/CRIT "
                        f"(F2-AUD-02 closure, S548-cont).",
                    )
                else:
                    baseline_flat = float(ens["deadband_frac"])
            else:
                logger.warning(
                    f"agreement-decay baseline at {baseline_path} has no "
                    f"ensemble_eval_distribution.deadband_frac — running in "
                    f"LOG_ONLY",
                )
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(
                f"agreement-decay baseline load failed ({e}) — running in "
                f"LOG_ONLY",
            )

    gates_drift = (config.get("gates") or {}).get("drift") or {}

    def _pick(key: str, default: float) -> float:
        for src in (drift_cfg, gates_drift):
            if key in src:
                return float(src[key])
        return float(default)

    deadband = float(
        getattr(agent, "deadband", None)
        or config.get("trading", {}).get("deadband_threshold")
        or config.get("env", {}).get("deadband_threshold", 0.25),
    )

    window_bars = int(_pick("agreement_flat_window_bars", 2000))
    tracker = AgreementDecayTracker(
        baseline_flat_frac=baseline_flat,
        rule=rule,
        deadband=deadband,
        window_bars=window_bars,
        min_bars_before_check=int(
            _pick("agreement_flat_min_bars_before_check", window_bars // 2),
        ),
        warn_delta=_pick("agreement_flat_delta_warn", 0.20),
        crit_delta=_pick("agreement_flat_delta_crit", 0.40),
        no_consensus_warn=_pick("agreement_no_consensus_warn", 0.30),
        no_consensus_crit=_pick("agreement_no_consensus_crit", 0.60),
    )
    logger.info(
        f"AgreementDecayTracker active: rule={rule} "
        f"baseline={'YES' if baseline_flat is not None else 'LOG_ONLY'} "
        f"window={tracker.window_bars} warmup={tracker.min_bars_before_check} "
        f"flat_warn={tracker.warn_delta} flat_crit={tracker.crit_delta} "
        f"no_consensus_warn={tracker.no_consensus_warn} "
        f"no_consensus_crit={tracker.no_consensus_crit}",
    )
    return tracker


def _init_cost_drift_tracker(config: dict) -> Optional[CostDriftTracker]:
    """Build a Protocol v2 §4.5 trigger #6 CostDriftTracker from engine config.

    Returns None when no `gates.retrain` block is declared (no cost-drift
    thresholds to gate on). Thresholds come from `gates.retrain`:
    `cost_drift_ratio` (default 1.20) and `cost_drift_window_trades`
    (default 100) — read from the LIVE config's inline gates block, since
    the engine reads `config["gates"]` at runtime (S551-cont-9 lesson; the
    canonical `<ws>_ensemble.gates.yaml` drives Stage-3 WF gating only).

    The denominator is the one-way per-trade cost the policy was *configured*
    under: `trading.taker_fee` (+ `trading.slippage_base_bps`/1e4 if present),
    falling back to the `env.*` mirror. A missing / non-positive assumption
    leaves the tracker in LOG_ONLY (realized cost still recorded; never FIRED).
    Cost drift is a retrain trigger, NOT a §8.3 halt — see CostDriftTracker.
    """
    retrain_cfg = (config.get("gates") or {}).get("retrain") or {}
    if not retrain_cfg:
        return None

    trading = config.get("trading") or {}
    env = config.get("env") or {}
    taker = trading.get("taker_fee", env.get("taker_fee"))
    slip_bps = trading.get("slippage_base_bps", env.get("slippage_base_bps", 0.0))
    config_cost_frac: Optional[float] = None
    if taker is not None:
        try:
            config_cost_frac = float(taker) + float(slip_bps or 0.0) / 1e4
        except (TypeError, ValueError):
            config_cost_frac = None

    tracker = CostDriftTracker(
        config_cost_frac,
        cost_drift_ratio=float(retrain_cfg.get("cost_drift_ratio", 1.20)),
        window_trades=int(retrain_cfg.get("cost_drift_window_trades", 100)),
    )
    logger.info(
        "CostDriftTracker active: config_cost_frac=%s ratio=%s window=%s",
        f"{tracker.config_cost_frac:.6f}" if tracker.config_cost_frac else "LOG_ONLY",
        tracker.cost_drift_ratio,
        tracker.window_trades,
    )
    return tracker


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

        # FE-02 fail-closed guard (2026-07-08 FE audit): the live path is only
        # correct for max_leverage == 1.0. Training scales actions by
        # env.max_leverage and reports private-state position / max_leverage
        # (B1/B5); live _predict() clips to [-1, 1] unscaled and position
        # sizing is fraction-of-PV. Shipping a leverage-trained bundle through
        # this engine would silently under-size every trade and feed the agent
        # out-of-distribution private dims. Refuse to start rather than trade
        # wrong; lifting this requires wiring max_leverage through _predict,
        # position accounting, AND LiveObsBuilder(max_leverage=...).
        _cfg_max_leverage = float(
            config.get("env", {}).get(
                "max_leverage", config.get("trading", {}).get("max_leverage", 1.0),
            ),
        )
        if abs(_cfg_max_leverage - 1.0) > 1e-9:
            raise RuntimeError(
                f"max_leverage={_cfg_max_leverage} is not supported by the live "
                f"engine (only 1.0): action scaling, position accounting and "
                f"private-state normalization are not leverage-wired (FE-02).",
            )

        # Position tracking
        self._current_position = 0.0
        self._prev_close = 0.0
        self._portfolio_value = config.get("trading", {}).get("initial_balance", 10000.0)
        # S490: preserve config-declared balance for the startup mismatch guard.
        # _current_position is stored as a fraction with _portfolio_value in the
        # denominator, so a silent broker-vs-config drift corrupts position tracking.
        self._config_initial_balance = self._portfolio_value
        self._initial_portfolio_value = self._portfolio_value
        self._peak_portfolio_value = self._portfolio_value
        self._current_funding_rate = 0.0

        # FIX AUD-H04: Track daily loss by UTC date, not bar count
        self._daily_start_value = self._portfolio_value
        self._last_daily_reset_date: Optional[date] = None
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
        # v2.2 §8.3 override path: operator file that lifts repeat-CRIT lockout.
        # Distinct from the kill_file itself; operator must also clear the
        # kill_file to re-enable the strategy after a lockout.
        self._kill_file_override = Path(config.get("safety", {}).get(
            "kill_file_override", f"{self._kill_file}.override",
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

        # Step 0b (E1 architect pass): live-replay fixture capture. When
        # `capture_ccxt_raw_responses` is true, the engine opens a JSONL writer
        # at startup and registers a capture callback on the broker; raw
        # CCXT fetch_positions / fetch_balance responses are appended for the
        # next `capture_max_bars` trading steps, then capture closes itself.
        # Default false; deploy with true after the 2026-04-27 00:00 UTC un-halt
        # to collect the Tier-3 live-replay fixture for the E1 refactor.
        self._capture_enabled = bool(config.get("safety", {}).get(
            "capture_ccxt_raw_responses", False,
        ))
        self._capture_max_bars = int(config.get("safety", {}).get(
            "capture_max_bars", 96,
        ))
        self._capture_dir = Path(config.get("safety", {}).get(
            "capture_dir", "/app/state",
        ))
        self._capture_writer = None
        self._capture_path: Optional[Path] = None
        self._capture_bars_remaining = 0

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
        # S448 Defect 3 fix: consecutive raw-breach counter. Trip when raw
        # breach persists ≥2 bars even if median hasn't caught up — kills the
        # 2-bar lag that let XAUUSD run to -11.74% DD on 2026-04-13 while
        # smoothed sat at -9.37%.
        self._consecutive_raw_breach_count: int = 0

        # Control
        self._should_stop = False
        self._wandb_run = None

        # Session 426: per-day contract roll check (futures brokers only).
        # Stores the UTC date of the last check so we run it once per day.
        self._last_roll_check_date: date | None = None

        # AUD-ENG-05 note: fetch-overlap dedup now lives in
        # LiveObsBuilder.update() (timestamp drop_duplicates, keep='last' —
        # FE-01, 2026-07-08 FE audit). The former _last_1min_ts_ms tracker
        # here was declared but never wired.
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
        from sharpen.crypto.live.metrics import TradingMetrics
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

        # S542 / decision_sg1_btc_skip_first_3_trades_s538: post-restart cooldown.
        # Skip the first N would-be trades after a fresh container start to avoid
        # the OOD execution skew seen on sg1-btc's first-3 live trades (perfect-fill
        # +$12 vs live -$45). Counter resets on each engine __init__ (i.e., every
        # restart) and decrements ONLY when an actual trade would otherwise fire
        # (passed deadband + risk-manager re-deadband). signal_gate / funding_gate /
        # warmup / deadband skips do NOT decrement — the cooldown counts trades,
        # not bars. Default 0 = disabled (every strategy except sg1-btc).
        self._post_restart_cooldown_initial = int(
            (config.get("risk") or {}).get("post_restart_cooldown_bars", 0),
        )
        self._post_restart_cooldown_remaining = self._post_restart_cooldown_initial
        if self._post_restart_cooldown_remaining > 0:
            logger.info(
                f"[Post-restart cooldown] Armed: will skip first "
                f"{self._post_restart_cooldown_remaining} would-be trades after this "
                f"restart (count resets on every container start).",
            )

        # Protocol v2.2 §8.2 action-drift tracker. Engine-side concerns:
        #   - observe(target_position, current_close) after agent.predict()
        #   - WARN: disable new entries (handled in _apply_drift_status); holds stay
        #   - CRIT: write kill_file (§8.3 T2) + FTMO force-close; watchdog lockout
        # Safe defaults: disabled unless `drift.enabled: true` and baseline resolvable.
        self._drift_tracker = _init_drift_tracker(config)
        self._drift_warn_active = False
        self._drift_last_status = None

        # v2.6 (S551-cont-4): feature-variance veto config — flows from
        # gates.drift.feature_variance_veto.{enabled,scale}. Engine-side
        # because we read `LiveObsBuilder.feature_variance_status()` per bar.
        # Defaults: enabled=True (every prop-firm workstream gets it once they
        # bump max_veto_frac into their gates.yaml), scale="base" (ADR-4).
        _veto_cfg = (
            (config.get("gates") or {}).get("drift", {}).get("feature_variance_veto", {})
        ) or {}
        self._feature_variance_veto_enabled = bool(_veto_cfg.get("enabled", True))
        _scale = _veto_cfg.get("scale", "base")
        if isinstance(_scale, str) and _scale not in ("base", "all"):
            raise ValueError(
                f"gates.drift.feature_variance_veto.scale must be 'base', 'all', "
                f"or an int in features.scales; got {_scale!r}"
            )
        self._feature_variance_veto_scale = _scale

        # Protocol v2.3 §8.2-extension agreement-decay tracker. Active only
        # for ensemble agents whose aggregation_rule is consensus-based
        # (`ens_agreement` / `ens_majority`). Silent-death detector: when
        # seeds diverge under regime shift the consensus filter stays flat
        # → no losses (PF gates don't fire) → capital utilization → 0.
        # Same WARN-blocks-new-entries / CRIT-flatten plumbing as drift,
        # but kill_file reason is `agreement_decay_crit` so retrospective
        # analysis (§4.5 Stage 2.5-R trigger #4) can attribute correctly.
        self._agreement_decay_tracker = _init_agreement_decay_tracker(
            config, agent,
        )
        self._agreement_decay_warn_active = False
        self._agreement_decay_last_status = None

        # Protocol v2 §4.5 trigger #6 cost-drift tracker. Rolling realized
        # one-way cost (fee + slippage vs the decision-bar close) measured
        # against the configured cost assumption. A FIRED status is a
        # Stage 2.5-R retrain signal — logged once per transition + surfaced
        # to WandB (drift/cost_*) — NOT a §8.3 halt: it never flattens or
        # writes the kill_file (execution being dearer than assumed is a
        # model-staleness signal, not an unsafe-to-trade one).
        self._cost_drift_tracker = _init_cost_drift_tracker(config)
        self._cost_drift_fired = False
        self._cost_drift_last_report = None

        # S495-cont prop-firm decoupling: challenge-phase state machine.
        # Only instantiated when config.challenge.enabled=true (live deploy
        # configs). Training / HPO / backtest configs have no challenge:
        # block at all, so this stays None and the per-bar hook in
        # _trading_step is a no-op.
        self._challenge_state_machine: Optional[ChallengeStateMachine] = None
        self._last_completed_phase_file = Path(config.get("safety", {}).get(
            "last_completed_phase_file", "/app/state/last_completed_phase.txt",
        ))
        challenge_cfg = config.get("challenge", {}) or {}
        if challenge_cfg.get("enabled", False):
            target = challenge_cfg.get("profit_target_pct")
            if target is None:
                target = float("inf")
            phase = ChallengePhase(
                name=str(challenge_cfg.get("phase", "custom")),
                profit_target_pct=float(target),
                next_phase=challenge_cfg.get("next_phase"),
                advance_rule=str(challenge_cfg.get("advance_rule", "manual_ack")),
            )
            self._challenge_state_machine = ChallengeStateMachine(
                phase,
                initial_portfolio_value=self._initial_portfolio_value,
                strategy_name=self._strategy_name,
                halt_state_writer=self._write_halt_state,
                kill_file_writer=write_kill_file,
                kill_file_path=self._kill_file,
                last_completed_phase_path=self._last_completed_phase_file,
                flatten_callback=self._emergency_flatten,
                stop_callback=self._request_stop,
                telemetry_gauge=self._metrics.update_generic
                    if hasattr(self._metrics, "update_generic") else None,
                n_confirm=int(challenge_cfg.get("n_confirm", 2)),
                smoothing_window=int(challenge_cfg.get("smoothing_window", 3)),
            )
            logger.info(
                f"Challenge state machine active: phase={phase.name}, "
                f"target={phase.profit_target_pct}, next={phase.next_phase}, "
                f"advance_rule={phase.advance_rule}, "
                f"initial_pv={self._initial_portfolio_value:.2f}",
            )

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

        # v2.2 §8.3 startup gate: if a prior process wrote a kill_file (drift
        # CRIT or operator halt), refuse to start until the operator clears
        # it. Repeat-CRIT lockout requires `kill_file.override` alongside
        # clearing the kill_file. Runs BEFORE broker.connect() so we don't
        # burn a cTrader token re-auth while locked out.
        if self._check_kill_file_startup_gate():
            return

        # S495-cont prop-firm decoupling startup gate: if a prior process
        # completed this phase (last_completed_phase.txt written by the
        # ChallengeStateMachine), refuse to restart on the SAME phase
        # config. ADR-2 operator-error guard: the operator must both clear
        # the kill_file AND swap to the next-phase overlay. This gate
        # catches the half-migration case where the kill_file was cleared
        # but the config still points at the completed phase.
        if self._check_challenge_phase_startup_gate():
            return

        # Audit FIND-02 / S491: pre-broker halt gate. If a prior process
        # persisted a halt (daily_loss / position_mismatch / balance_mismatch /
        # risk_halt / intrabar_dd), sleep until halted_until BEFORE paying the
        # cost of broker.connect() + bootstrap. Avoids death-looping broker
        # auth on a stale token while we wait for UTC day rollover.
        self._write_bootstrap_health("pre_broker_halt_gate")
        if await self._check_persistent_halt():
            return

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

        # S490 + FIND-02: balance-mismatch startup guard. The engine stores
        # _current_position as lots*lot_size*price / _portfolio_value — a
        # mid-session PV change silently invalidates cached positions and trips
        # the reconciler. On detection, persist halt_state and exit cleanly so
        # the pre-broker halt gate sleeps until UTC day rollover on next
        # restart, instead of raising and death-looping under Docker
        # restart-policy. Operator can opt out via safety.accept_balance_mismatch.
        safety_cfg = self.config.get("safety", {})
        mismatch_tolerance = safety_cfg.get("balance_mismatch_tolerance_pct", 0.10)
        accept_mismatch = safety_cfg.get("accept_balance_mismatch", False)
        config_ib = self._config_initial_balance
        if config_ib > 0:
            drift = abs(self._portfolio_value - config_ib) / config_ib
            if drift > mismatch_tolerance:
                msg = (
                    f"BALANCE MISMATCH: broker equity ${self._portfolio_value:,.2f} "
                    f"vs config.initial_balance ${config_ib:,.2f} "
                    f"(drift {drift:.1%} > tolerance {mismatch_tolerance:.1%}). "
                    f"Position tracking uses a PV-denominated fraction; starting "
                    f"with a mismatch will corrupt reconciliation. Update "
                    f"trading.initial_balance to match the broker and restart, or "
                    f"set safety.accept_balance_mismatch=true to bypass."
                )
                if accept_mismatch:
                    logger.warning(
                        msg + " (BYPASSED via safety.accept_balance_mismatch=true)",
                    )
                else:
                    logger.critical(msg)
                    self._write_halt_state(
                        reason="balance_mismatch",
                        detail=(
                            f"broker_equity={self._portfolio_value:.2f} "
                            f"config_ib={config_ib:.2f} "
                            f"drift={drift:.4f} "
                            f"tolerance={mismatch_tolerance:.4f}"
                        ),
                        now_utc=datetime.now(timezone.utc),
                    )
                    self._request_stop("balance_mismatch")
                    # Clean shutdown so broker TCP + loader close gracefully.
                    # Pre-broker halt gate will fire on next restart.
                    await self._shutdown()
                    return

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
        # Post-FIND-02 this is a defensive secondary check; the pre-broker
        # halt gate catches virtually every case.
        if await self._check_persistent_halt():
            # FIND-05: broker + loader are wired by now — close cleanly on
            # mid-sleep SIGTERM so we don't leak a TCP connection.
            await self._shutdown()
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

        # Step 0b: open capture writer + register broker callback if enabled.
        if self._capture_enabled:
            self._open_capture_writer()

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
            # Step 0b: bound capture window by bar count.
            if self._capture_writer is not None and self._capture_bars_remaining > 0:
                self._capture_bars_remaining -= 1
                if self._capture_bars_remaining == 0:
                    self._close_capture_writer(reason="max_bars_reached")

    async def _trading_step_inner(self, bar_time: datetime) -> None:
        """Inner trading step logic (wrapped by _trading_step for PV race guard)."""
        # --- Safety: kill file check ---
        # v2.2 §8.3: if the file is a drift-CRIT JSON, log the reason/count so
        # the watchdog stream tells humans what happened. Legacy empty file
        # still stops the engine with reason="kill_file" (backward-compat).
        if self._kill_file.exists():
            payload = read_kill_file(self._kill_file)
            reason = (payload or {}).get("reason", "legacy")
            detail = (payload or {}).get("detail", "")
            count = (payload or {}).get("count", 1)
            logger.warning(
                f"Kill file detected: {self._kill_file} "
                f"(reason={reason} count={count} detail={detail})",
            )
            self._request_stop(f"kill_file:{reason}")
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
        if self._should_stop:
            return

        # S495-cont prop-firm decoupling: observe PV against challenge
        # target, trigger phase_complete on a smoothed + N_CONFIRM-persisted
        # breach. Placed here — between _check_daily_loss and the remaining
        # per-bar logic — because:
        #   - PV was just refreshed at start of _trading_step via
        #     _update_portfolio_value (line ~626).
        #   - If the daily-loss check already halted (should_stop=True),
        #     we bail out above and skip the challenge check.
        #   - The trigger's flatten + halt_state + kill_file writes are
        #     idempotent with daily-loss's own halt write ordering, and
        #     both use _write_halt_state so the same gate semantics apply.
        if self._challenge_state_machine is not None:
            status = self._challenge_state_machine.observe(
                self._portfolio_value, now_utc=bar_time,
            )
            if status.phase_complete and status.trip_source != "already_complete":
                await self._request_phase_complete(status, bar_time)
                return

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
        # S535 ADR-4: snapshot the raw policy intent before any overlay (PRISM)
        # or risk-manager rewrite. Threaded into _log_step calls so the gap
        # between policy intent and executed action is observable in WandB.
        policy_target_position = float(target_position)

        # --- 5z. No-consensus NaN sentinel (Fix 2, S538-cont, 2026-05-08) ---
        # EnsembleAgent._agreement returns NaN when no >=2 directional
        # majority. Pre-Fix-2 it returned 0.0 which engine-side conflated
        # with "deliberate flat = liquidate" (sg1-btc dispersion-collapse
        # incident, drift_crit at warmup +6.62% DD). Treat NaN as "hold
        # prior position" — feed NaN to the trackers (they classify it as
        # no-consensus, NOT deadband) but skip every downstream trade-side
        # path so current_position is preserved.
        if np.isnan(policy_target_position):
            if self._drift_tracker is not None:
                try:
                    self._drift_tracker.observe(
                        target_position, bar_close=current_close,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"drift tracker error on NaN bar (non-fatal): {e}")
            if self._agreement_decay_tracker is not None:
                try:
                    self._agreement_decay_tracker.observe(target_position)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        f"agreement-decay tracker error on NaN bar (non-fatal): {e}",
                    )
            self._prev_close = current_close
            self._log_step(
                bar_time, self._current_position, traded=False,
                skip_reason="no_consensus",
                policy_target_position=policy_target_position,
            )
            return

        # --- 5a. Action-drift tracking (Protocol v2.2 §8.2 + v2.6 ADR-1) ---
        # observe() records the *policy output* before any overlays / deadband
        # / risk clipping, so the live distribution matches the stage-2/2.5
        # eval baseline (which was also the raw policy output).
        #
        # v2.6 (S551-cont-4): feature_state lets the tracker exclude bars where
        # the engineered features were pathologically flat — the policy COULD
        # not have responded, so counting that bar as "policy went deadband"
        # is a false positive (Mode B in
        # decision_drift_two_failure_modes_s551_cont_3). Status returned is
        # VETOED for those bars — silent pass-through in _apply_drift_status.
        if self._drift_tracker is not None:
            try:
                feature_state: Optional[str] = None
                if self._feature_variance_veto_enabled and hasattr(
                    self.obs_builder, "feature_variance_status"
                ):
                    feature_state = self.obs_builder.feature_variance_status(
                        scale=self._feature_variance_veto_scale,
                    ).state
                report = self._drift_tracker.observe(
                    target_position, bar_close=current_close,
                    feature_state=feature_state,
                )
                await self._apply_drift_status(report, bar_time)
                # If CRIT forced a stop, the kill_file writer + flatten ran
                # via T2 wiring; bail out of this step without executing trades.
                if self._should_stop:
                    return
                # WARN: disable new entries / flips / size-ups, but ALLOW
                # moves toward zero so the agent can still close exposure.
                # The XAUUSD S491 crash storm post-mortem flagged the inverse
                # failure mode ("halt while long = can't reduce risk").
                if self._drift_warn_active and self._blocked_by_drift_warn(target_position):
                    self._log_step(
                        bar_time, self._current_position,
                        traded=False, skip_reason="drift_warn_no_new_entries",
                        policy_target_position=policy_target_position,
                    )
                    self._prev_close = current_close
                    return
            except Exception as e:  # noqa: BLE001
                logger.warning(f"drift tracker error (non-fatal): {e}")

        # --- 5a-bis. Agreement-decay tracking (Protocol v2.3 §8.2 ext) ---
        # Same observe-then-dispatch pattern as the action-drift block above;
        # the WARN gate reuses _blocked_by_drift_warn (no-new-entries) since
        # the failure mode is identical from the engine's POV.
        if self._agreement_decay_tracker is not None:
            try:
                ad_report = self._agreement_decay_tracker.observe(
                    target_position,
                )
                await self._apply_agreement_decay_status(ad_report, bar_time)
                if self._should_stop:
                    return
                if (
                    self._agreement_decay_warn_active
                    and self._blocked_by_drift_warn(target_position)
                ):
                    self._log_step(
                        bar_time, self._current_position,
                        traded=False,
                        skip_reason="agreement_decay_warn_no_new_entries",
                        policy_target_position=policy_target_position,
                    )
                    self._prev_close = current_close
                    return
            except Exception as e:  # noqa: BLE001
                logger.warning(f"agreement-decay tracker error (non-fatal): {e}")

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
            self._log_step(
                bar_time, target_position, traded=False, skip_reason="deadband",
                regime_info=regime_info,
                policy_target_position=policy_target_position,
            )
            return

        # --- 7. Risk manager check ---
        action_array = np.array([target_position])
        checked_action, violations = self.risk_manager.check(
            action=action_array,
            portfolio_value=self._portfolio_value,
            margin_balance=self._portfolio_value * 0.95,  # Conservative estimate
            positions=np.array([self._current_position]),
            funding_rates=np.array([self._current_funding_rate]),
            # S506 Option B: enables UTC-midnight-anchored daily-turnover reset.
            # bar_time is in scope from _trading_step_inner signature.
            bar_time=bar_time,
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
            self._log_step(
                bar_time, target_position, traded=False, skip_reason="risk_deadband",
                regime_info=regime_info,
                policy_target_position=policy_target_position,
            )
            return

        # --- 7b. Post-restart cooldown (S542) -------------------------------
        # By here a real trade is about to fire: passed signal/funding gates,
        # passed both deadband checks, risk manager cleared. Decrement the
        # counter and hold the current position. This implements the
        # `risk.post_restart_cooldown_bars` knob from
        # decision_sg1_btc_skip_first_3_trades_s538.md — count *would-be
        # trades*, not bars, so signal_gate / deadband / funding_gate skips
        # do not consume the cooldown.
        if self._post_restart_cooldown_remaining > 0:
            self._post_restart_cooldown_remaining -= 1
            # Rollback turnover budget consumed by risk_manager.check above so
            # the skipped trade doesn't burn the daily quota.
            self.risk_manager.rollback_last_turnover()
            self._prev_close = current_close
            logger.info(
                f"[Post-restart cooldown] Bar {self._total_bars}: holding "
                f"position {self._current_position:.4f}, "
                f"{self._post_restart_cooldown_remaining} cooldown trades "
                f"remaining.",
            )
            self._log_step(
                bar_time, self._current_position,
                traded=False, skip_reason="post_restart_cooldown",
                regime_info=regime_info,
                policy_target_position=policy_target_position,
            )
            return

        # --- 8. Execute trade ---
        if self._dry_run:
            logger.info(
                f"[DRY RUN] Would trade: {self._current_position:.4f} → "
                f"{target_position:.4f} (delta={delta:.4f})",
            )
            self._current_position = target_position
            self._prev_close = current_close
            self._log_step(
                bar_time, target_position, traded=True, regime_info=regime_info,
                policy_target_position=policy_target_position,
            )
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
                self._observe_cost_drift(order, current_close)
                # S527-cont: Bybit demo can return None for filled/avg/fee
                # via ccxt; format with `or 0` so logging never crashes the
                # success path (which would mask the fill as an exec error).
                _fq = order.filled_quantity if order.filled_quantity is not None else 0.0
                _avg = order.avg_fill_price if order.avg_fill_price is not None else 0.0
                _fee = order.fee if order.fee is not None else 0.0
                logger.info(
                    f"Trade executed: {order.side} {_fq:.6f} "
                    f"@ {_avg:.2f}, fee={_fee:.4f} USDT",
                )
            elif order.status == "partial":
                # FIX AUD-H06 + LIVE-01: Don't assume target on partial fill.
                # Sync actual position from exchange immediately so
                # _current_position reflects reality. Without this, the next
                # bar computes delta from stale position → double execution.
                self._total_trades += 1
                self._total_fees += order.fee
                self._observe_cost_drift(order, current_close)
                try:
                    exchange_pos = await self.broker.get_single_position(self._asset)
                    self._current_position = exchange_pos
                except Exception as e:
                    logger.warning(f"Post-partial-fill position query failed: {e}")
                _fq = order.filled_quantity if order.filled_quantity is not None else 0.0
                _qty = order.quantity if order.quantity is not None else 0.0
                logger.warning(
                    f"Partial fill: {_fq:.6f} of "
                    f"{_qty:.6f} — synced position to {self._current_position:.4f}",
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
                self._log_step(
                    bar_time, target_position, traded=False, skip_reason="broker_skipped",
                    regime_info=regime_info,
                    policy_target_position=policy_target_position,
                )
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

            # FIX S502: Ambiguous-execution outcome — order may have filled at
            # the exchange while the engine assumes failure. Force an immediate
            # broker reconcile so the agent doesn't act on stale internal state
            # on the next bar. Without this, periodic reconcile (every
            # _reconcile_interval bars) may not detect the divergence until the
            # position has drifted past the halt threshold (S502 gmgp1-btc
            # 08:45 incident: -1007 timeout at 08:00 → opposite-sign divergence
            # 1.24 by 08:45 reconcile).
            await self._force_reconcile_if_ambiguous(str(e), bar_time)

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
        self._log_step(
            bar_time, target_position, traded=True, order=order,
            regime_info=regime_info,
            policy_target_position=policy_target_position,
        )

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

        Parity note (GATE-CAUSAL-01, S553-cont-170): ``features[-1]`` is the last
        CLOSED bar, which is all live can ever see. Training now gates on the same
        bar (``handler._ptr - 1``); before that fix the wrapper gated on the
        not-yet-closed ``_ptr``, so this docstring's "mirrors" claim was false and
        the two paths traded different bar sets. Do not "align" training back onto
        the unclosed bar to close a parity gap -- that direction reintroduces the
        leak.
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

            # --- _connected-attribute broker check (cTrader, OANDA) ---
            ct_connected = getattr(self.broker, "_connected", None)
            if ct_connected is not None:
                if ct_connected:
                    return True
                logger.warning(
                    "%s connection lost — attempting reconnect",
                    type(self.broker).__name__,
                )
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

    # S502: ccxt error patterns where the order may have filled despite
    # raising. These require an immediate broker reconcile to resync the
    # internal position cache before the agent acts on stale state.
    # S527-cont: Bybit demo `fetch_order` can return Python `None` instead
    # of raising; the broker swallows this, fires a Market fallback that
    # double-fills, and the engine surfaces the fault as a downstream
    # format-string crash with no recognizable broker error string. Treat
    # those crashes as ambiguous so reconcile fires within 1 bar.
    _AMBIGUOUS_EXECUTION_MARKERS = (
        "-1007",                      # binance: timeout, send status unknown
        "send status unknown",
        "execution status unknown",
        "unsupported format string",  # bybit demo: NoneType.__format__ crash
        "could not fetch order status",  # bybit demo: silent None from fetch_order
        "nonetype",                   # bybit demo: 'NoneType' attr/subscript errors
    )

    @classmethod
    def _is_ambiguous_execution_error(cls, err_str: str) -> bool:
        """Return True if the ccxt error string indicates an ambiguous fill."""
        err_lower = err_str.lower()
        return any(m in err_lower for m in cls._AMBIGUOUS_EXECUTION_MARKERS)

    async def _force_reconcile_if_ambiguous(
        self, err_str: str, bar_time: datetime,
    ) -> bool:
        """Trigger immediate ``_reconcile_all`` if the error is ambiguous.

        Returns True if reconcile was attempted (regardless of outcome).
        Caller is the order-execution exception handler in
        ``_trading_step_inner``; reconcile must not propagate exceptions.
        """
        if not self._is_ambiguous_execution_error(err_str):
            return False
        logger.warning(
            "Ambiguous execution outcome detected — forcing immediate "
            "broker reconcile (order may have filled despite error).",
        )
        # Reset reconcile gate so this call is not skipped by the interval
        # check. ``_reconcile_all`` itself updates ``_last_reconcile_bar`` to
        # ``_total_bars`` once it begins work.
        self._last_reconcile_bar = (
            self._total_bars - self._reconcile_interval
        )
        try:
            await self._reconcile_all(bar_time)
        except Exception as rec_err:
            logger.error(
                f"Forced reconcile after ambiguous execution failed: "
                f"{rec_err} — next bar will retry via interval gate.",
            )
        return True

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
                # S491: persist halt so Docker restart enters safe sleep-to-midnight
                # loop instead of thrashing through broker reconnect on every cycle
                # (see project_xauusd_crash_storm_s491 — 4h gmgp1-xauusd outage).
                self._write_halt_state(
                    reason="position_mismatch",
                    detail=(
                        f"internal={self._current_position:.4f} "
                        f"exchange={exchange_pos:.4f} "
                        f"divergence={pos_divergence:.4f} "
                        f"threshold={self._reconciliation_halt_pct:.4f}"
                    ),
                    now_utc=datetime.now(timezone.utc),
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

        S510: sanity guard against transient broker readings that have been
        observed to come back at ~2× the real equity (cTrader IC Markets demo
        — symptom: peak ratchets to a value never seen in any logged bar, then
        every subsequent bar reports a phantom 50% drawdown). Reject readings
        >1.5× the larger of current PV and config initial_balance — bar-to-bar
        equity moves on prop-firm paper accounts are bounded by leverage ×
        intra-bar volatility and never come close to 50%. A real deposit is
        an operator event that warrants a config bump, not a silent ratchet.
        """
        try:
            info = await self.broker.get_account_info()
            equity = info.get("total_equity", 0)
            if equity > 0:
                ref = max(self._portfolio_value, self._config_initial_balance)
                if ref > 0 and equity > 1.5 * ref:
                    # cTrader exposes balance/unrealized_pnl/n_positions/
                    # mid_price; other brokers default to None and the log
                    # collapses to "n/a". Splits a balance spike (protobuf
                    # mis-pairing) from a unrealized_pnl spike (stale mid +
                    # broker-side orphan position).
                    bal = info.get("balance")
                    upnl = info.get("unrealized_pnl")
                    n_pos = info.get("n_positions")
                    mid = info.get("mid_price")
                    logger.warning(
                        f"S510: discarding suspicious broker equity "
                        f"${equity:,.2f} (>1.5× ref ${ref:,.2f}); "
                        f"keeping PV=${self._portfolio_value:,.2f} and "
                        f"peak=${self._peak_portfolio_value:,.2f}; "
                        f"breakdown: balance={bal}, unrealized_pnl={upnl}, "
                        f"n_positions={n_pos}, mid_price={mid}",
                    )
                    return
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

        Median of last 3 PV readings prevents API-hiccup false triggers.
        S448 Defect 3 fix: also trip when the raw reading has breached the
        limit on ≥2 consecutive bars — median lags raw by up to 2 bars on
        fast adverse moves, and that lag let XAUUSD run past FTMO's 10% DD
        line on 2026-04-13 (raw -11.74% final vs smoothed -9.37% at trip).
        Two-consecutive-bar requirement preserves single-spike robustness.

        Audit FIND-01: ``max_daily_loss_pct <= 0`` disables the check. A
        literal 0.0 used to trip ``raw_return < -0.0`` on any losing bar
        (Velotrade 2-step configs tripped on the first red bar). Prop-firm
        challenges without a daily-loss rule (Velotrade 2-step) can now set
        this to 0.0 as an explicit "disabled" flag.
        """
        current_date = bar_time.date()

        # FIND-01: non-positive limit disables the check. Still rotate the
        # daily anchor so reporting remains correct across UTC day rollovers.
        if self._max_daily_loss_pct <= 0:
            if (
                self._last_daily_reset_date is None
                or current_date != self._last_daily_reset_date
            ):
                self._daily_start_value = self._portfolio_value
                self._last_daily_reset_date = current_date
                self._pv_buffer.clear()
                self._consecutive_raw_breach_count = 0
            return

        if self._last_daily_reset_date is None or current_date != self._last_daily_reset_date:
            self._daily_start_value = self._portfolio_value
            self._last_daily_reset_date = current_date
            # Reset PV buffer + breach counter on new day
            self._pv_buffer.clear()
            self._consecutive_raw_breach_count = 0

        # Append current reading to smoothing buffer
        self._pv_buffer.append(self._portfolio_value)

        if self._daily_start_value > 0:
            # Use median of buffered readings for robustness
            smoothed_pv = float(np.median(list(self._pv_buffer)))
            smoothed_return = (smoothed_pv - self._daily_start_value) / self._daily_start_value

            # Also compute single-reading return for comparison
            raw_return = (self._portfolio_value - self._daily_start_value) / self._daily_start_value
            limit = self._max_daily_loss_pct

            raw_breach = raw_return < -limit
            smoothed_breach = smoothed_return < -limit

            if raw_breach:
                self._consecutive_raw_breach_count += 1
            else:
                self._consecutive_raw_breach_count = 0

            persisted_raw = self._consecutive_raw_breach_count >= 2

            if smoothed_breach or persisted_raw:
                worst_return = min(raw_return, smoothed_return)
                trip_source = "smoothed" if smoothed_breach else "raw_persisted"
                logger.critical(
                    f"DAILY LOSS LIMIT ({trip_source}): "
                    f"smoothed={smoothed_return:.2%} raw={raw_return:.2%} "
                    f"consec_raw={self._consecutive_raw_breach_count} "
                    f"< -{limit:.0%}. Flattening and stopping trading.",
                )
                # FIX RSK-02: Flatten before stop — main loop exits on
                # should_stop without closing open positions otherwise.
                try:
                    await self._emergency_flatten()
                finally:
                    self._write_halt_state(
                        reason="daily_loss_limit",
                        detail=(
                            f"trip={trip_source} "
                            f"worst_return={worst_return:.4f} "
                            f"smoothed={smoothed_return:.4f} "
                            f"raw={raw_return:.4f} "
                            f"consec_raw={self._consecutive_raw_breach_count}"
                        ),
                        now_utc=datetime.now(timezone.utc),
                    )
                    self._request_stop("daily_loss_limit")
            elif raw_breach:
                # First-bar raw breach, median hasn't caught up yet — likely
                # API hiccup. Waiting one more bar to confirm.
                logger.debug(
                    f"Daily loss: raw={raw_return:.2%} breached limit but "
                    f"smoothed={smoothed_return:.2%} did not "
                    f"(consec_raw={self._consecutive_raw_breach_count}, "
                    f"buffer={[round(v, 2) for v in self._pv_buffer]}). "
                    f"Awaiting confirmation.",
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

        FIND-01 parity (S498-cont, 2026-04-26): ``max_daily_loss_pct <= 0``
        disables the check, mirroring the closing-price daily-loss guard.
        Without this, ``projected_return < -0.0`` trips on any tiny negative
        excursion and persistently halts Velotrade-style deploys (no daily
        rule) on the first bar that holds a position. SG-1-BTC paper deploy
        2026-04-26 hit this on bar 1.
        """
        if not self._intrabar_dd_enabled:
            return
        if self._max_daily_loss_pct <= 0:
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
            # Snapshot pos BEFORE _emergency_flatten() zeros it — otherwise
            # the persisted halt detail reports pos=0 and post-mortem reads
            # like a flat-position false positive (S501).
            pos_at_trip = self._current_position
            logger.critical(
                f"INTRA-BAR DD PROJECTION: pos={pos_at_trip:+.4f} "
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
                        f"pos_at_trip={pos_at_trip:.4f} "
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

    def _blocked_by_drift_warn(self, target_position: float) -> bool:
        """Return True if WARN state should block a pending position move.

        WARN "disable new entries, hold existing" (v2.2 §8.3): permit any
        trade that REDUCES exposure (moves |target| toward 0 without flipping
        sign) and block everything else (new entries from flat, flips,
        size-ups). A small tolerance absorbs SAC's stochastic float noise
        so "same position" isn't spuriously classified as size-up.
        """
        cur = float(self._current_position)
        tgt = float(target_position)
        tol = 1e-4
        # Flat → any non-flat target is a new entry: BLOCK.
        if abs(cur) <= tol:
            return abs(tgt) > tol
        # Sign flip (long→short or short→long) is strictly a new entry on
        # the opposite side after closing: BLOCK.
        if (cur > 0 and tgt < -tol) or (cur < 0 and tgt > tol):
            return True
        # Same direction: ALLOW if |tgt| <= |cur| + tol (reduce or hold);
        # BLOCK if |tgt| > |cur| + tol (size-up adds new exposure).
        return abs(tgt) > abs(cur) + tol

    def _check_kill_file_startup_gate(self) -> bool:
        """Return True if engine must refuse to start (kill_file present).

        Lockout semantics per Protocol v2.2 §8.3 (see
        `sharpen.monitoring.kill_file.should_lockout`). We log the
        reason and exit via `_request_stop` so the main loop falls through
        without connecting to the broker. Any existing kill_file halts
        startup — operator must manually delete kill_file to re-enable.
        For repeat-CRIT branch (drift_crit, count ≥ 2 within 24h) the
        operator must ALSO write `kill_file.override` before deleting
        kill_file, per scoped-override semantic.
        """
        payload = read_kill_file(self._kill_file)
        if payload is None:
            return False
        _, reason = should_lockout(payload, self._kill_file_override)
        logger.critical(f"STARTUP REFUSED — {reason}")
        self._request_stop(f"kill_file_lockout:{payload.get('reason')}")
        return True

    def _check_challenge_phase_startup_gate(self) -> bool:
        """Return True if engine must refuse to start (wrong-phase config).

        S495-cont (ADR-2 prop-firm decoupling). Reads
        ``last_completed_phase.txt`` and refuses startup when:
        - The file marks a phase equal to or ordered-after the
          configured phase. Operator must swap to the next-phase overlay
          (e.g. ``configs/deploy/ftmo/step2.yaml``) before restart.

        No-op when no challenge state machine is active (training /
        backtest configs have no ``challenge:`` block).
        """
        if self._challenge_state_machine is None:
            return False
        configured = self._challenge_state_machine.phase.name
        ok, msg = ChallengeStateMachine.check_startup_phase_gate(
            configured_phase=configured,
            last_completed_phase_path=self._last_completed_phase_file,
        )
        if not ok:
            logger.critical(f"STARTUP REFUSED — challenge phase gate: {msg}")
            self._request_stop(f"challenge_phase_gate:{configured}")
            return True
        logger.info(f"Challenge phase startup gate: {msg}")
        return False

    async def _request_phase_complete(self, status, bar_time: datetime) -> None:
        """S495-cont prop-firm decoupling phase_complete path.

        Mirrors :meth:`_request_drift_crit` shape: writes persistent
        markers, flattens, requests stop. Delegates the actual work to
        the :class:`ChallengeStateMachine` which owns the idempotency
        guard and atomic-write semantics.
        """
        if self._challenge_state_machine is None:
            return
        try:
            await self._challenge_state_machine.trigger_phase_complete(
                status.trip_source, now_utc=bar_time,
            )
        except Exception as e:  # noqa: BLE001
            logger.critical(
                f"[challenge] trigger_phase_complete raised ({e}) — "
                f"halt_state + kill_file may be partial; operator must "
                f"inspect {self._halt_state_file} and {self._kill_file}",
            )

    async def _request_drift_crit(self, report, bar_time: datetime) -> None:
        """Protocol v2.2 §8.3 CRIT path.

        1. Write/increment drift_crit kill_file JSON (count enables repeat-CRIT
           lockout on the next restart).
        2. Await the existing FTMO-style emergency_flatten so open positions
           are genuinely closed before stop is signaled (the "halt without
           flatten is the bleed window" failure mode — S491 post-mortem).
        3. Request stop with reason=drift_crit → container exits non-zero →
           watchdog refuses auto-restart while kill_file is present.
        """
        try:
            payload = write_kill_file(
                self._kill_file,
                reason=REASON_DRIFT_CRIT,
                detail=report.reason,
                extra={
                    "bar_time": bar_time.isoformat() if bar_time else None,
                    "kl": report.kl,
                    "deadband_frac_delta": report.deadband_frac_delta,
                    "saturation_frac_delta": report.saturation_frac_delta,
                    "bucket": report.bucket,
                    "n_bars": report.n_bars,
                },
            )
            logger.critical(
                f"[drift] CRIT → kill_file written "
                f"(count={payload.get('count', 1)} at {self._kill_file})",
            )
        except OSError as e:
            logger.critical(f"[drift] CRIT but kill_file write failed: {e}")

        try:
            await self._emergency_flatten()
        except Exception as e:  # noqa: BLE001
            logger.critical(
                f"[drift] CRIT emergency_flatten raised ({e}) — "
                f"position may still be open on exchange",
            )
        self._request_stop("drift_crit")

    def _observe_cost_drift(self, order, decision_price: float) -> None:
        """Feed one filled/partial trade to the §4.5 #6 cost-drift tracker.

        Cost drift is a RETRAIN trigger, not a §8.3 halt: a FIRED status is
        logged once per OK→FIRED transition (and surfaced to WandB via
        ``_log_step``), but never flattens positions or writes the kill_file.
        A clear (FIRED→OK) is logged at INFO. No-op when the tracker is
        disabled (no ``gates.retrain`` block). Robust to the Bybit-demo
        ``None`` fee/fill/qty case — the tracker skips such fills.
        """
        if self._cost_drift_tracker is None or order is None:
            return
        report = self._cost_drift_tracker.observe(
            fee=getattr(order, "fee", None),
            avg_fill_price=getattr(order, "avg_fill_price", None),
            decision_price=decision_price,
            filled_quantity=getattr(order, "filled_quantity", None),
        )
        self._cost_drift_last_report = report
        if report.status == CostDriftStatus.FIRED:
            if not self._cost_drift_fired:
                logger.warning(f"[cost_drift] {report.reason}")
                self._cost_drift_fired = True
        elif report.status == CostDriftStatus.OK and self._cost_drift_fired:
            logger.info(
                f"[cost_drift] cleared at bar {self._total_bars} — "
                f"cost_ratio={report.cost_ratio:.3f} back within "
                f"{report.cost_drift_ratio}",
            )
            self._cost_drift_fired = False

    async def _apply_drift_status(self, report, bar_time: datetime) -> None:
        """Dispatch a Protocol v2.2 §8.2 DriftReport into engine side-effects.

        * WARN  → set self._drift_warn_active so the step loop blocks new
                  entries / flips / size-ups via `_blocked_by_drift_warn`.
                  Rearmed downward on next OK.
        * CRIT  → await self._request_drift_crit() which writes the kill_file
                  JSON, flattens open positions, and requests stop.
        All status transitions are logged at INFO+ and attached to WandB.
        """
        from sharpen.monitoring import DriftStatus

        status = report.status
        prev = self._drift_last_status
        self._drift_last_status = status

        if self._wandb_run is not None:
            try:
                import wandb
                wandb.log(
                    {f"drift/{k}": v for k, v in report.to_dict().items()
                     if v is not None and not isinstance(v, str)},
                    commit=False,
                )
            except Exception:  # noqa: BLE001
                pass

        if status == DriftStatus.CRIT:
            if prev != DriftStatus.CRIT:
                logger.critical(
                    f"[drift] CRIT at bar {self._total_bars}: {report.reason} "
                    f"(bucket={report.bucket}, n={report.n_bars})",
                )
            await self._request_drift_crit(report, bar_time)
        elif status == DriftStatus.WARN:
            if not self._drift_warn_active or prev != DriftStatus.WARN:
                logger.warning(
                    f"[drift] WARN at bar {self._total_bars}: {report.reason} "
                    f"(bucket={report.bucket}, n={report.n_bars})",
                )
            self._drift_warn_active = True
        elif status == DriftStatus.OK:
            if self._drift_warn_active:
                logger.info(
                    f"[drift] OK at bar {self._total_bars} — clearing WARN",
                )
            self._drift_warn_active = False
        elif status == DriftStatus.VETOED:
            # v2.6 (S551-cont-4): bar excluded from drift accumulation because
            # the observation features were flat. NOT a halt signal — engine
            # continues normal trading. _drift_warn_active is NOT toggled
            # (WARN-clear semantics preserved). WandB already saw the veto
            # via the to_dict() log above (drift/flat_veto_frac, drift/feature_state).
            pass
        # WARMUP / LOG_ONLY: silent pass-through

    async def _request_agreement_decay_crit(
        self, report, bar_time: datetime,
    ) -> None:
        """Protocol v2.3 §8.2-ext CRIT path (parallel to _request_drift_crit).

        Distinct kill_file reason (`agreement_decay_crit`) so retrospective
        analysis can attribute the §4.5 Stage 2.5-R retrain trigger #4
        correctly. Same flatten + stop semantics — silent capital
        starvation is just as harmful as a true distribution drift.
        """
        try:
            payload = write_kill_file(
                self._kill_file,
                reason=REASON_AGREEMENT_DECAY_CRIT,
                detail=report.reason,
                extra={
                    "bar_time": bar_time.isoformat() if bar_time else None,
                    "rule": report.rule,
                    "flat_bar_frac_live": report.flat_bar_frac_live,
                    "flat_bar_frac_baseline": report.flat_bar_frac_baseline,
                    "flat_bar_frac_delta": report.flat_bar_frac_delta,
                    # F2-AUD-01 closure (S548-cont): surface no_consensus_frac
                    # so retrospective analysis can disambiguate the two
                    # CRIT signals (flat-drift vs no-consensus rate).
                    "no_consensus_frac": report.no_consensus_frac,
                    "n_bars": report.n_bars,
                    "trigger": "agreement_decay",  # §4.5 Stage 2.5-R trigger #4
                },
            )
            logger.critical(
                f"[agreement-decay] CRIT → kill_file written "
                f"(count={payload.get('count', 1)} at {self._kill_file})",
            )
        except OSError as e:
            logger.critical(
                f"[agreement-decay] CRIT but kill_file write failed: {e}",
            )

        try:
            await self._emergency_flatten()
        except Exception as e:  # noqa: BLE001
            logger.critical(
                f"[agreement-decay] CRIT emergency_flatten raised ({e}) — "
                f"position may still be open on exchange",
            )
        self._request_stop("agreement_decay_crit")

    async def _apply_agreement_decay_status(
        self, report, bar_time: datetime,
    ) -> None:
        """Dispatch a Protocol v2.3 §8.2-ext AgreementDecayReport.

        Mirrors :meth:`_apply_drift_status`: WARN flips
        ``_agreement_decay_warn_active`` (engine then refuses new entries
        / flips / size-ups via ``_blocked_by_drift_warn`` — the no-new-
        entries semantics are identical), CRIT routes through
        :meth:`_request_agreement_decay_crit`. WARMUP / LOG_ONLY pass
        through silently except for WandB telemetry.
        """
        from sharpen.monitoring import AgreementDecayStatus

        status = report.status
        prev = self._agreement_decay_last_status
        self._agreement_decay_last_status = status

        if self._wandb_run is not None:
            try:
                import wandb
                wandb.log(
                    {f"agreement_decay/{k}": v for k, v in report.to_dict().items()
                     if v is not None and not isinstance(v, str)},
                    commit=False,
                )
            except Exception:  # noqa: BLE001
                pass

        if status == AgreementDecayStatus.CRIT:
            if prev != AgreementDecayStatus.CRIT:
                logger.critical(
                    f"[agreement-decay] CRIT at bar {self._total_bars}: "
                    f"{report.reason} (rule={report.rule}, n={report.n_bars})",
                )
            await self._request_agreement_decay_crit(report, bar_time)
        elif status == AgreementDecayStatus.WARN:
            if not self._agreement_decay_warn_active or prev != AgreementDecayStatus.WARN:
                logger.warning(
                    f"[agreement-decay] WARN at bar {self._total_bars}: "
                    f"{report.reason} (rule={report.rule}, n={report.n_bars})",
                )
            self._agreement_decay_warn_active = True
        elif status == AgreementDecayStatus.OK:
            if self._agreement_decay_warn_active:
                logger.info(
                    f"[agreement-decay] OK at bar {self._total_bars} — clearing WARN",
                )
            self._agreement_decay_warn_active = False
        # WARMUP / LOG_ONLY: silent pass-through (telemetry only)

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
        policy_target_position: float | None = None,
    ) -> None:
        """Log step metrics to WandB and logger.

        S535 ADR-4 (observability): when ``policy_target_position`` is supplied
        (set right after ``self._predict(obs)`` and threaded through the
        pipeline), emit it alongside the existing ``target_position`` so the
        gap between policy intent and executed action is visible in WandB.
        The risk manager's soft-throttle / hard-cap can silently rewrite the
        action — this metric makes that override observable for the drift
        detector and downstream review (currently they only see the post-clip
        ``target_position``).
        """
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

        # S535 ADR-4: policy-intent observability. Emitted only when supplied
        # by the caller (post-predict paths); pre-predict skip paths (signal
        # gate, funding-rate gate, obs sanity) leave this None.
        if policy_target_position is not None:
            metrics["policy_target_position"] = policy_target_position
            risk_clip_delta = policy_target_position - target_position
            metrics["risk_clip_delta"] = risk_clip_delta
            denom = max(abs(policy_target_position), 0.01)
            metrics["risk_clip_fraction"] = risk_clip_delta / denom
            metrics["risk_clip_active"] = int(abs(risk_clip_delta) > 0.01)

        if order is not None:
            metrics["order_fee"] = order.fee
            metrics["order_fill_price"] = order.avg_fill_price

        # Protocol v2 §4.5 #6 cost-drift telemetry (rolling realized vs config
        # cost). Emitted every bar so the ratio trend is visible even between
        # fills; None-valued stats (warmup / log-only) are filtered like the
        # §8.2 drift report. `drift/cost_fired` is the retrain-trigger flag —
        # informational only (no halt; see _observe_cost_drift).
        if self._cost_drift_tracker is not None:
            _cd = self._cost_drift_tracker.snapshot()
            metrics["drift/cost_fired"] = int(_cd["status"] == CostDriftStatus.FIRED)
            metrics["drift/cost_n_trades"] = _cd["n_trades"]
            if _cd["cost_ratio"] is not None:
                metrics["drift/cost_ratio"] = _cd["cost_ratio"]
            if _cd["realized_cost_frac_mean"] is not None:
                metrics["drift/cost_realized_frac_mean"] = _cd["realized_cost_frac_mean"]

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
    # Step 0b: live-replay fixture capture (E1 architect pass)
    # -------------------------------------------------------------------
    def _open_capture_writer(self) -> None:
        """Open a JSONL capture file and register the broker callback.

        Any failure here is logged and disables capture; trading is unaffected.
        """
        try:
            self._capture_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self._capture_path = self._capture_dir / (
                f"ccxt_capture_{self._strategy_name}_{ts}.jsonl"
            )
            self._capture_writer = self._capture_path.open("a", buffering=1, encoding="utf-8")
            self._capture_bars_remaining = self._capture_max_bars
            if hasattr(self.broker, "set_capture_callback"):
                self.broker.set_capture_callback(self._capture_emit)
                logger.info(
                    f"CCXT capture mode ENABLED: path={self._capture_path} "
                    f"max_bars={self._capture_max_bars}",
                )
            else:
                logger.warning(
                    "capture_ccxt_raw_responses=true but broker has no "
                    "set_capture_callback(); capture will be no-op.",
                )
                self._close_capture_writer(reason="broker_unsupported")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Capture writer open failed (non-fatal): {e}")
            self._capture_writer = None

    def _capture_emit(self, event_name: str, raw_response) -> None:
        """Broker callback: append one JSONL record per fetch.

        Errors are swallowed: capture must never disrupt trading.
        """
        if self._capture_writer is None:
            return
        try:
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "bar": self._total_bars,
                "event": event_name,
                "strategy": self._strategy_name,
                "raw": raw_response,
                "engine_position": self._current_position,
                "engine_pv": self._portfolio_value,
            }
            self._capture_writer.write(json.dumps(record, default=str) + "\n")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Capture write failed (non-fatal): {e}")

    def _close_capture_writer(self, reason: str) -> None:
        """Close the capture writer and unregister the broker callback."""
        if self._capture_writer is not None:
            try:
                self._capture_writer.close()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Capture writer close raised (non-fatal): {e}")
            self._capture_writer = None
        if hasattr(self.broker, "set_capture_callback"):
            try:
                self.broker.set_capture_callback(None)
            except Exception:  # noqa: BLE001
                pass
        logger.info(
            f"CCXT capture mode CLOSED: reason={reason} path={self._capture_path}",
        )

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

        # Final position reconciliation — flatten if configured.
        # S535-cont-2: skip when broker is known-disconnected. ib_futures_broker.get_single_position
        # falls back to clipped raw-contract count when _portfolio_value <= 0 (BUG-15 fallback at
        # ib_futures_broker.py:381-386); during a dead IB connection ib.positions() returns []
        # so n_contracts=0 → exchange_pos=0.0, producing a misleading "Position discrepancy:
        # internal=0.47 exchange=0.00" log every time the engine self-stops on
        # 5-consecutive-reconnect-failures (gmgp1-gold daily IB outage 23:30-00:45 UTC).
        # The cached _current_position is correct; the broker query is unreliable.
        if not self._check_broker_alive():
            logger.warning(
                f"SHUTDOWN: broker disconnected — skipping final position verify; "
                f"last known internal position fraction={self._current_position:.4f} "
                f"(use broker UI / next-restart sync for ground truth).",
            )
        else:
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

        # Step 0b: ensure capture writer is flushed and closed on shutdown.
        # Guard with getattr: shutdown must complete even on a partially
        # constructed engine (e.g. __init__ failed before line 437 set the
        # attribute), so PID 1 always exits and docker's restart policy fires.
        if getattr(self, "_capture_writer", None) is not None:
            self._close_capture_writer(reason="shutdown")

        if self._wandb_run is not None:
            # Bounded wandb.finish() — unbounded hang here kept PID 1 alive
            # for 2h+ in S489 (gmgp1-gold), defeating docker restart policy.
            # Use a daemon thread (NOT asyncio.to_thread) so asyncio.run()
            # cleanup doesn't join a stuck worker.
            import threading as _threading
            finish_exc: list[BaseException] = []

            def _finish_worker() -> None:
                try:
                    import wandb
                    wandb.finish()
                except BaseException as e:  # noqa: BLE001
                    finish_exc.append(e)

            t = _threading.Thread(target=_finish_worker, daemon=True, name="wandb-finish")
            t.start()
            t.join(timeout=30.0)
            if t.is_alive():
                logger.warning(
                    "wandb.finish() exceeded 30s timeout — proceeding with shutdown",
                )
            elif finish_exc:
                logger.debug(f"wandb.finish failed: {finish_exc[0]}")

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
