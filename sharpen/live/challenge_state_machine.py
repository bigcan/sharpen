"""Challenge state machine — live-only profit-target tracking for prop firms.

Post the S495-cont prop-firm decoupling (see
``.agent/artifacts/prop_firm_decoupling_architecture.md``), profit-target
handling lives in the live engine rather than the training wrapper. This
module:

1. Tracks cumulative return against a per-phase profit target (step1:
   10%, step2: 5%, funded: disabled).
2. Uses median-of-N smoothing plus an ``n_confirm`` consecutive-raw-breach
   requirement to reject single-bar quote spikes — mirroring the S448
   pattern used by ``_check_daily_loss`` (``live_engine.py:1565-1582``).
3. On a confirmed target hit, writes a persistent
   ``last_completed_phase.txt`` marker, writes ``halt_state`` + ``kill_file``
   with ``reason="phase_complete"``, triggers ``broker.flatten_all()``, and
   requests engine stop. Idempotent — second and further ``observe()``
   calls after a trigger are no-ops.
4. Exposes ``read_last_completed_phase`` so the engine startup gate can
   refuse to run on the same phase twice (ADR-2 operator-error guard).

The ``reason="phase_complete"`` string is distinct from
``REASON_DRIFT_CRIT`` so it does not count toward the repeat-CRIT lockout
math in ``sharpen/monitoring/kill_file.py`` (line 133).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


# Public constant mirroring REASON_DRIFT_CRIT et al. in monitoring/kill_file.
# Distinct string guarantees the repeat-CRIT lockout at
# monitoring/kill_file.py:133 does not treat a phase_complete event as a
# drift event.
REASON_PHASE_COMPLETE = "phase_complete"


# Phase ordering for the startup guard. Higher rank must follow lower rank;
# re-running the same phase is rejected.
PHASE_ORDER = {"step1": 0, "step2": 1, "funded": 2}


@dataclass(frozen=True)
class ChallengePhase:
    """One prop-firm phase: target, successor, advance policy."""
    name: str                        # "step1" | "step2" | "funded" | "custom"
    profit_target_pct: float         # 0.10 for step1, 0.05 for step2, math.inf for funded
    next_phase: Optional[str]        # "step2" after "step1", etc.; None if terminal
    advance_rule: str = "manual_ack"  # "manual_ack" (default) | "auto" (rejected at paper-deploy)


@dataclass
class ChallengeStatus:
    """Return value of ``ChallengeStateMachine.observe()``."""
    cumulative_return: float
    smoothed_return: float
    target_remaining: float
    phase_complete: bool
    trip_source: str                 # "smoothed" | "raw_persisted" | "none" | "already_complete"
    reason: str                      # "in_progress" | "phase_complete" | "below_target"


class ChallengeStateMachine:
    """Smoothed, debounced profit-target tracker for a single live strategy.

    Parameters
    ----------
    phase : ChallengePhase
        The phase being evaluated. Use ``profit_target_pct=math.inf`` for
        the funded phase (never triggers).
    initial_portfolio_value : float
        Baseline for cumulative-return arithmetic. Typically the engine's
        ``_initial_portfolio_value`` at boot.
    strategy_name : str
        For log + halt_state labelling.
    halt_state_writer : callable
        Bound ``engine._write_halt_state(reason, detail, now_utc)`` method.
    kill_file_writer : callable
        Usually ``sharpen.monitoring.kill_file.write_kill_file``.
    kill_file_path : Path
        Target path for the kill_file write.
    last_completed_phase_path : Path
        Path to the persistent ``last_completed_phase.txt`` marker. Same
        directory as ``risk_state.json`` typically; must be writable and
        survive container restart.
    flatten_callback : async callable → None
        Bound ``broker.flatten_all()`` (or ``engine._emergency_flatten``).
    stop_callback : callable(str)
        Bound ``engine._request_stop(reason)``.
    telemetry_gauge : optional callable(str, float)
        ``metrics.update_generic(name, value)`` style. Called with gauge
        names ``challenge_cumulative_return``,
        ``challenge_target_remaining``, and ``challenge_phase_complete``.
    n_confirm : int
        Number of consecutive raw breaches required to trip when the
        smoothed median has not yet caught up. Default 2 (mirrors S448).
    smoothing_window : int
        Maxlen of the PV median buffer. Default 3 (mirrors
        ``_check_daily_loss``).
    """

    def __init__(
        self,
        phase: ChallengePhase,
        initial_portfolio_value: float,
        strategy_name: str,
        *,
        halt_state_writer: Callable[[str, str, datetime], None],
        kill_file_writer: Callable[..., dict],
        kill_file_path: Path,
        last_completed_phase_path: Path,
        flatten_callback: Callable[[], Awaitable[None]],
        stop_callback: Callable[[str], None],
        telemetry_gauge: Optional[Callable[[str, float], None]] = None,
        n_confirm: int = 2,
        smoothing_window: int = 3,
    ) -> None:
        if phase.advance_rule not in ("manual_ack", "auto"):
            raise ValueError(
                f"advance_rule must be 'manual_ack' or 'auto', got {phase.advance_rule!r}"
            )
        if smoothing_window < 1:
            raise ValueError(f"smoothing_window must be >= 1, got {smoothing_window}")
        if n_confirm < 1:
            raise ValueError(f"n_confirm must be >= 1, got {n_confirm}")
        if initial_portfolio_value <= 0:
            raise ValueError(
                f"initial_portfolio_value must be > 0, got {initial_portfolio_value}"
            )

        self.phase = phase
        self.initial_pv = float(initial_portfolio_value)
        self.strategy_name = strategy_name
        self._halt_state_writer = halt_state_writer
        self._kill_file_writer = kill_file_writer
        self._kill_file_path = Path(kill_file_path)
        self._last_completed_phase_path = Path(last_completed_phase_path)
        self._flatten_callback = flatten_callback
        self._stop_callback = stop_callback
        self._telemetry_gauge = telemetry_gauge

        self.n_confirm = int(n_confirm)
        self.smoothing_window = int(smoothing_window)

        self._pv_buffer: deque[float] = deque(maxlen=self.smoothing_window)
        self._consecutive_raw_breach = 0
        self._phase_complete = False
        self._trigger_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(
        self, portfolio_value: float, *, now_utc: Optional[datetime] = None,
    ) -> ChallengeStatus:
        """Consume a new PV reading; return the resulting status.

        Does NOT itself trigger the phase-complete side effects — it
        returns a :class:`ChallengeStatus` with ``phase_complete=True``
        when a trip is detected, and the caller (live engine) is
        expected to ``await trigger_phase_complete(status.trip_source)``.
        This split keeps ``observe()`` synchronous so it can be called
        from inside a ``_trading_step`` without forcing every engine
        method through the event loop.
        """
        cumulative_return = (portfolio_value - self.initial_pv) / self.initial_pv

        if self._phase_complete:
            return ChallengeStatus(
                cumulative_return=cumulative_return,
                smoothed_return=cumulative_return,
                target_remaining=0.0,
                phase_complete=True,
                trip_source="already_complete",
                reason=REASON_PHASE_COMPLETE,
            )

        if not math.isfinite(self.phase.profit_target_pct):
            # Funded phase — target disabled forever.
            self._emit_gauges(cumulative_return, cumulative_return, math.inf)
            return ChallengeStatus(
                cumulative_return=cumulative_return,
                smoothed_return=cumulative_return,
                target_remaining=math.inf,
                phase_complete=False,
                trip_source="none",
                reason="in_progress",
            )

        self._pv_buffer.append(float(portfolio_value))
        median_pv = _median(self._pv_buffer)
        smoothed_return = (median_pv - self.initial_pv) / self.initial_pv

        raw_hit = cumulative_return >= self.phase.profit_target_pct
        smoothed_hit = smoothed_return >= self.phase.profit_target_pct

        if raw_hit:
            self._consecutive_raw_breach += 1
        else:
            self._consecutive_raw_breach = 0

        persisted_raw = self._consecutive_raw_breach >= self.n_confirm

        if smoothed_hit or persisted_raw:
            trip_source = "smoothed" if smoothed_hit else "raw_persisted"
            target_remaining = max(
                0.0, self.phase.profit_target_pct - smoothed_return
            )
            self._emit_gauges(cumulative_return, smoothed_return, target_remaining)
            return ChallengeStatus(
                cumulative_return=cumulative_return,
                smoothed_return=smoothed_return,
                target_remaining=target_remaining,
                phase_complete=True,
                trip_source=trip_source,
                reason=REASON_PHASE_COMPLETE,
            )

        target_remaining = max(0.0, self.phase.profit_target_pct - smoothed_return)
        self._emit_gauges(cumulative_return, smoothed_return, target_remaining)
        return ChallengeStatus(
            cumulative_return=cumulative_return,
            smoothed_return=smoothed_return,
            target_remaining=target_remaining,
            phase_complete=False,
            trip_source="none",
            reason=(
                "below_target" if smoothed_return < self.phase.profit_target_pct
                else "in_progress"
            ),
        )

    # ------------------------------------------------------------------
    # Side-effect trigger
    # ------------------------------------------------------------------

    async def trigger_phase_complete(
        self, trip_source: str, *, now_utc: datetime,
    ) -> None:
        """Execute the phase-complete side effects. Idempotent.

        Order:
        1. Persist ``last_completed_phase.txt`` (atomic write) — done FIRST
           so a crash between here and flatten still leaves the engine
           unable to re-trip on restart.
        2. Write ``halt_state`` with ``reason="phase_complete"``.
        3. Write ``kill_file`` with ``reason="phase_complete"`` +
           ``extra={"phase", "next_phase", "phase_advance_ready": True}``.
        4. Await ``flatten_callback()`` — best-effort; a failure here is
           logged critical but does not prevent the halt (per S490 post-
           mortem, halt_state must be written BEFORE flatten so a flatten
           failure still leaves the engine halted on restart).
        5. Emit final telemetry gauge.
        6. ``stop_callback("phase_complete")``.
        """
        async with self._trigger_lock:
            if self._phase_complete:
                return
            self._phase_complete = True

        detail = (
            f"target={self.phase.profit_target_pct:.4f}; trip={trip_source}; "
            f"advance={self.phase.advance_rule}"
        )

        try:
            self._write_last_completed_phase()
        except OSError as e:
            logger.critical(
                f"[challenge] phase_complete BUT last_completed_phase write "
                f"failed ({e}) — engine may re-trip on restart; operator "
                f"must manually create {self._last_completed_phase_path}",
            )

        try:
            self._halt_state_writer(REASON_PHASE_COMPLETE, detail, now_utc)
        except Exception as e:  # noqa: BLE001
            logger.critical(f"[challenge] halt_state write failed: {e}")

        try:
            self._kill_file_writer(
                self._kill_file_path,
                reason=REASON_PHASE_COMPLETE,
                detail=detail,
                extra={
                    "phase": self.phase.name,
                    "next_phase": self.phase.next_phase,
                    "phase_advance_ready": True,
                    "strategy": self.strategy_name,
                },
            )
            logger.critical(
                f"[challenge] phase_complete kill_file written "
                f"at {self._kill_file_path}",
            )
        except OSError as e:
            logger.critical(f"[challenge] kill_file write failed: {e}")

        try:
            await self._flatten_callback()
        except Exception as e:  # noqa: BLE001
            logger.critical(
                f"[challenge] phase_complete but flatten_all raised ({e}) — "
                f"position may still be open on exchange; halt_state + "
                f"kill_file are already written so restart will stay halted",
            )

        if self._telemetry_gauge is not None:
            try:
                self._telemetry_gauge("challenge_phase_complete", 1.0)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"[challenge] telemetry emit failed: {e}")

        self._stop_callback(REASON_PHASE_COMPLETE)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _write_last_completed_phase(self) -> None:
        """Atomic write via tempfile + rename."""
        target = self._last_completed_phase_path
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        payload = {
            "phase": self.phase.name,
            "next_phase": self.phase.next_phase,
            "strategy": self.strategy_name,
        }
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(target)
        logger.critical(
            f"[challenge] last_completed_phase written: "
            f"{target} (phase={self.phase.name}, next={self.phase.next_phase})"
        )

    @staticmethod
    def read_last_completed_phase(path: Path) -> Optional[dict[str, Any]]:
        """Return the contents of ``last_completed_phase.txt``, or None."""
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                f"[challenge] last_completed_phase read failed at {path}: {e}"
            )
            return None

    # ------------------------------------------------------------------
    # Startup gate
    # ------------------------------------------------------------------

    @classmethod
    def check_startup_phase_gate(
        cls,
        *,
        configured_phase: str,
        last_completed_phase_path: Path,
    ) -> tuple[bool, str]:
        """Evaluate whether the engine may start on ``configured_phase``.

        Returns ``(ok, message)``. ``ok=True`` means the engine should
        proceed; ``ok=False`` means it must abort with ``message``.

        Policy (ADR-2 operator-error guard):
        - No prior marker → allow (first deploy).
        - Marker phase == configured phase → REFUSE (re-trip loop).
        - Marker phase ordered before configured phase → allow (advance).
        - Marker phase ordered after or equal to configured phase → REFUSE.
        - Unknown phase name in marker or config → WARN + allow (custom
          phases are operator-defined; cannot enforce ordering).
        """
        marker = cls.read_last_completed_phase(last_completed_phase_path)
        if marker is None:
            return True, "no prior phase marker"

        last_phase = marker.get("phase")
        if not isinstance(last_phase, str):
            return True, f"marker has no phase field ({marker!r}); allowing"

        if last_phase == configured_phase:
            return False, (
                f"last_completed_phase={last_phase!r} equals configured "
                f"phase={configured_phase!r}. The challenge was already "
                f"passed on this phase. Update configs/deploy/<firm>/ overlay "
                f"to the next phase (e.g. step2.yaml) and delete "
                f"{last_completed_phase_path} before restarting."
            )

        last_rank = PHASE_ORDER.get(last_phase)
        cur_rank = PHASE_ORDER.get(configured_phase)
        if last_rank is None or cur_rank is None:
            return True, (
                f"custom phase in ordering check (last={last_phase}, "
                f"configured={configured_phase}); skipping strict gate"
            )

        if cur_rank > last_rank:
            return True, (
                f"advance from {last_phase} to {configured_phase} approved"
            )

        return False, (
            f"configured_phase={configured_phase!r} (rank={cur_rank}) is "
            f"not strictly after last_completed_phase={last_phase!r} "
            f"(rank={last_rank}). Refusing startup — update the overlay."
        )

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def _emit_gauges(
        self, cumulative: float, smoothed: float, remaining: float,
    ) -> None:
        if self._telemetry_gauge is None:
            return
        try:
            self._telemetry_gauge("challenge_cumulative_return", cumulative)
            self._telemetry_gauge("challenge_smoothed_return", smoothed)
            # math.inf survives JSON but not Prometheus; clamp.
            clamped = 1e9 if not math.isfinite(remaining) else remaining
            self._telemetry_gauge("challenge_target_remaining", clamped)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[challenge] telemetry emit failed: {e}")


# ---------------------------------------------------------------------------
# Free helpers
# ---------------------------------------------------------------------------

def _median(values) -> float:
    """Pure-python median over a small iterable. Returns 0.0 when empty."""
    arr = sorted(float(v) for v in values)
    n = len(arr)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return arr[mid]
    return 0.5 * (arr[mid - 1] + arr[mid])
