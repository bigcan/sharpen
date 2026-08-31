"""Protocol v2.3 §8.2 (extension) live agreement-decay tracker.

Silent-death detector for ensemble-deployed strategies that use a consensus
aggregation rule (`ens_agreement` / `ens_majority`). Under regime shift the
per-seed actors can disagree; the consensus filter then returns flat → no
losses (so PF gates don't fire) but capital utilization → 0. The
ActionDriftTracker (§8.2) catches *distribution* drift but cannot
distinguish "agent learned to hold" from "agent silently lost consensus,"
so this tracker watches the post-aggregation flat-bar fraction directly.

Baseline = `ensemble_eval_distribution.deadband_frac` from
`ensemble_report.json` (the post-aggregation flat-bar fraction on the
test window for the chosen rule). The tracker fires on two independent
signals (max severity wins):

  * **Flat-bar drift** — |live − baseline| > `agreement_flat_delta_*`
    sustained for `window_bars`. Catches dispersion-collapse where
    seeds disagree more in regime and the consensus aggregator
    inflates the flat-bar fraction. (Pre-Fix-2 contract; the only
    signal that existed before S538-cont.)
  * **No-consensus rate** — fraction of bars where the aggregator
    emitted NaN > `no_consensus_*`. Catches the F2-AUD-01 silent-death
    case where 100% of bars hit no-consensus — the strategy stops
    trading but the flat-bar fraction is undefined (no consensus bars
    to compute on), so the flat-delta signal stays LOG_ONLY.

CRIT routes through §8.3 flatten + kill_file + watchdog lockout (engine
side) and also fires the §4.5 Stage 2.5-R retrain trigger #4. No-op for
non-consensus rules (`ens_mean`, `ens_median`, `ens_pf_weighted`).
"""

from __future__ import annotations

import collections
import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


CONSENSUS_RULES: tuple[str, ...] = ("ens_agreement", "ens_majority")


class AgreementDecayStatus:
    """String constants — JSON-friendly, parallel to DriftStatus."""

    OK = "OK"
    WARMUP = "WARMUP"
    LOG_ONLY = "LOG_ONLY"
    WARN = "WARN"
    CRIT = "CRIT"


@dataclass
class AgreementDecayReport:
    status: str
    reason: str
    n_bars: int
    flat_bar_frac_live: Optional[float]
    flat_bar_frac_baseline: Optional[float]
    flat_bar_frac_delta: Optional[float]
    rule: str
    # Fix 2 (S538-cont, 2026-05-08): fraction of bars where the aggregator
    # emitted NaN (no >=2 directional consensus). Replaces the old conflation
    # of "flat" with "no consensus". flat_bar_frac_live now measures
    # naturally-flat bars only (within consensus bars).
    no_consensus_frac: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "n_bars": self.n_bars,
            "flat_bar_frac_live": self.flat_bar_frac_live,
            "flat_bar_frac_baseline": self.flat_bar_frac_baseline,
            "flat_bar_frac_delta": self.flat_bar_frac_delta,
            "rule": self.rule,
            "no_consensus_frac": self.no_consensus_frac,
        }


class AgreementDecayTracker:
    """Per-strategy rolling flat-bar-fraction drift tracker.

    Args:
        baseline_flat_frac: post-aggregation deadband_frac from
            `ensemble_report.json → ensemble_eval_distribution.deadband_frac`.
            None disables gating (LOG_ONLY).
        rule: aggregation rule the live agent uses. Tracker is a no-op
            (CONSENSUS_RULES.SKIP) for non-consensus rules; callers should
            avoid constructing it in those cases, but the rule is retained
            here so a misconfigured runner emits a clear LOG_ONLY rather
            than firing a false CRIT.
        deadband: |action| < deadband counts as flat. Should match the
            EnsembleAgent.deadband used at inference (which itself comes
            from the resolved bundle config). Post-Fix-2 the
            `ens_agreement` aggregator returns NaN when consensus fails
            (no-consensus bars are excluded from numerator and denominator
            of the flat-bar fraction); see `update()` for the NaN-skip
            contract.
        window_bars: rolling window size (default 2000 per spec).
        min_bars_before_check: warmup floor; gating is suppressed below
            this count (default = window_bars / 2).
        warn_delta / crit_delta: |live − baseline| thresholds for the
            flat-bar drift signal.
        no_consensus_warn / no_consensus_crit: fraction-of-window-bars
            thresholds for the no-consensus rate signal (Fix 2
            S538-cont + F2-AUD-01 closure S548-cont). Defaults 0.30 /
            0.60: sg1-btc fold_07 baseline saw ~14% no-consensus rate;
            sg1-xauusd ~36%; 60% means majority of bars have no
            directional consensus → strategy is effectively idle.
    """

    def __init__(
        self,
        baseline_flat_frac: Optional[float],
        *,
        rule: str,
        deadband: float = 0.25,
        window_bars: int = 2000,
        min_bars_before_check: Optional[int] = None,
        warn_delta: float = 0.20,
        crit_delta: float = 0.40,
        no_consensus_warn: float = 0.30,
        no_consensus_crit: float = 0.60,
    ):
        if window_bars < 100:
            raise ValueError("window_bars must be >= 100")
        if warn_delta <= 0 or crit_delta <= 0:
            raise ValueError("warn_delta / crit_delta must be > 0")
        if warn_delta >= crit_delta:
            raise ValueError("warn_delta must be < crit_delta")
        if deadband <= 0:
            raise ValueError("deadband must be > 0")
        if not 0.0 < no_consensus_warn < 1.0:
            raise ValueError("no_consensus_warn must be in (0, 1)")
        if not 0.0 < no_consensus_crit <= 1.0:
            raise ValueError("no_consensus_crit must be in (0, 1]")
        if no_consensus_warn >= no_consensus_crit:
            raise ValueError("no_consensus_warn must be < no_consensus_crit")

        self.baseline_flat_frac = (
            float(baseline_flat_frac) if baseline_flat_frac is not None else None
        )
        self.rule = str(rule)
        self.deadband = float(deadband)
        self.window_bars = int(window_bars)
        self.min_bars_before_check = (
            int(min_bars_before_check)
            if min_bars_before_check is not None
            else self.window_bars // 2
        )
        if self.min_bars_before_check > self.window_bars:
            raise ValueError("min_bars_before_check must be <= window_bars")
        self.warn_delta = float(warn_delta)
        self.crit_delta = float(crit_delta)
        self.no_consensus_warn = float(no_consensus_warn)
        self.no_consensus_crit = float(no_consensus_crit)

        self._is_consensus = self.rule in CONSENSUS_RULES
        self._flat_flags: collections.deque[bool] = collections.deque(
            maxlen=self.window_bars,
        )
        # Fix 2: a parallel deque tracking NaN sentinels (no consensus) so
        # the live flat_bar_frac measurement is on consensus bars only.
        self._no_consensus_flags: collections.deque[bool] = collections.deque(
            maxlen=self.window_bars,
        )

        if not self._is_consensus:
            logger.warning(
                "AgreementDecayTracker constructed for non-consensus rule "
                "%r — emitting LOG_ONLY (caller should skip construction "
                "to avoid baseline-mismatch confusion)",
                self.rule,
            )
        elif self.baseline_flat_frac is None:
            logger.warning(
                "AgreementDecayTracker has no baseline flat_bar_frac — "
                "running in LOG_ONLY (live deltas will be recorded but "
                "no WARN/CRIT will fire)",
            )

    # --- public API -----------------------------------------------------

    def observe(self, aggregated_action: float) -> AgreementDecayReport:
        """Append one bar's post-aggregation action. Returns drift status.

        Pass the **post-aggregation** scalar action (the EnsembleAgent's
        output before the engine-side deadband / risk-manager clipping).

        Fix 2 (S538-cont, 2026-05-08): the aggregator now emits NaN on
        no >=2 directional consensus. NaN bars are recorded in
        ``_no_consensus_flags`` and excluded from ``flat_bar_frac_live``
        (which measures naturally-flat bars within consensus bars only).
        Pre-Fix-2 the aggregator returned 0 on consensus failure and the
        live flat fraction conflated "consensus said hold" with "consensus
        failed silently" — the sg1-btc dispersion-collapse bug.
        """
        try:
            agg = float(aggregated_action)
        except (TypeError, ValueError):
            agg = float("nan")
        is_no_consensus = math.isnan(agg)
        is_flat = (not is_no_consensus) and abs(agg) < self.deadband
        self._flat_flags.append(is_flat)
        self._no_consensus_flags.append(is_no_consensus)
        return self._evaluate()

    def snapshot(self) -> dict:
        """Current window stats without mutating state. For WandB logging."""
        return self._evaluate().to_dict()

    # --- internals ------------------------------------------------------

    # Severity ordering used by the dual-signal evaluator. LOG_ONLY is the
    # weakest "we computed something but cannot gate on it" state, so
    # anything from WARN upward overrides it.
    _SEVERITY_ORDER: tuple[str, ...] = (
        AgreementDecayStatus.OK,
        AgreementDecayStatus.LOG_ONLY,
        AgreementDecayStatus.WARN,
        AgreementDecayStatus.CRIT,
    )

    @classmethod
    def _max_severity(cls, a: str, b: str) -> str:
        return a if cls._SEVERITY_ORDER.index(a) >= cls._SEVERITY_ORDER.index(b) else b

    def _evaluate(self) -> AgreementDecayReport:
        n = len(self._flat_flags)

        if not self._is_consensus:
            return AgreementDecayReport(
                status=AgreementDecayStatus.LOG_ONLY,
                reason=f"non-consensus rule {self.rule!r}: tracker is no-op",
                n_bars=n,
                flat_bar_frac_live=None,
                flat_bar_frac_baseline=self.baseline_flat_frac,
                flat_bar_frac_delta=None,
                rule=self.rule,
                no_consensus_frac=None,
            )

        if n < self.min_bars_before_check:
            return AgreementDecayReport(
                status=AgreementDecayStatus.WARMUP,
                reason=(
                    f"warmup: {n}/{self.min_bars_before_check} bars "
                    f"before first check"
                ),
                n_bars=n,
                flat_bar_frac_live=None,
                flat_bar_frac_baseline=self.baseline_flat_frac,
                flat_bar_frac_delta=None,
                rule=self.rule,
                no_consensus_frac=None,
            )

        # Fix 2: live flat-bar fraction excludes no-consensus bars, since
        # those are now "unobserved" rather than "flat". no_consensus_frac
        # is reported separately AND gated on its own thresholds —
        # F2-AUD-01 closure (S548-cont) replaces the old "all-NaN → silent
        # LOG_ONLY" behavior with explicit WARN/CRIT firing.
        n_no_consensus = sum(self._no_consensus_flags)
        n_consensus = n - n_no_consensus
        no_consensus_frac = n_no_consensus / n if n else 0.0

        # ---------- Signal 1: no-consensus rate ----------
        # Pure rate threshold — no baseline needed. Fires regardless of
        # whether the flat-fraction signal is computable.
        if no_consensus_frac > self.no_consensus_crit:
            nc_status = AgreementDecayStatus.CRIT
            nc_reason = (
                f"CRIT: no_consensus_frac={no_consensus_frac:.3f} "
                f"(crit={self.no_consensus_crit}) — consensus failed on "
                f"majority of window; strategy is effectively idle"
            )
        elif no_consensus_frac > self.no_consensus_warn:
            nc_status = AgreementDecayStatus.WARN
            nc_reason = (
                f"WARN: no_consensus_frac={no_consensus_frac:.3f} "
                f"(warn={self.no_consensus_warn})"
            )
        else:
            nc_status = AgreementDecayStatus.OK
            nc_reason = ""

        # ---------- Signal 2: flat-bar drift vs baseline ----------
        if n_consensus == 0:
            # No consensus bars → flat-frac signal undefined. Defer to
            # the no-consensus rate signal alone (which is guaranteed
            # CRIT here since no_consensus_frac == 1.0 > no_consensus_crit
            # for any sane threshold).
            fd_status = AgreementDecayStatus.LOG_ONLY
            fd_reason = (
                f"all {n} bars in window were no-consensus (NaN); "
                f"flat-fraction signal not computable"
            )
            live = None
            delta = None
        else:
            live = sum(self._flat_flags) / n_consensus
            if self.baseline_flat_frac is None:
                fd_status = AgreementDecayStatus.LOG_ONLY
                fd_reason = "no baseline flat_bar_frac available"
                delta = None
            else:
                delta = abs(live - self.baseline_flat_frac)
                if delta > self.crit_delta:
                    fd_status = AgreementDecayStatus.CRIT
                    fd_reason = (
                        f"CRIT: flat_frac_delta={delta:.3f} "
                        f"(crit={self.crit_delta}) — consensus has decayed; "
                        f"capital utilization at risk"
                    )
                elif delta > self.warn_delta:
                    fd_status = AgreementDecayStatus.WARN
                    fd_reason = (
                        f"WARN: flat_frac_delta={delta:.3f} "
                        f"(warn={self.warn_delta})"
                    )
                else:
                    fd_status = AgreementDecayStatus.OK
                    fd_reason = f"within thresholds (delta={delta:.3f})"

        # ---------- Resolve max severity + reason ----------
        final = self._max_severity(fd_status, nc_status)
        if final == fd_status == nc_status:
            reason = fd_reason if fd_reason else nc_reason
        elif final == fd_status:
            reason = fd_reason
        else:
            reason = nc_reason
        # If both signals are non-OK at the same severity, surface both.
        if (
            fd_status != AgreementDecayStatus.OK
            and nc_status != AgreementDecayStatus.OK
            and fd_status == nc_status
            and fd_reason and nc_reason
        ):
            reason = f"{fd_reason} | {nc_reason}"

        return AgreementDecayReport(
            status=final,
            reason=reason,
            n_bars=n,
            flat_bar_frac_live=live,
            flat_bar_frac_baseline=self.baseline_flat_frac,
            flat_bar_frac_delta=delta,
            rule=self.rule,
            no_consensus_frac=no_consensus_frac,
        )
