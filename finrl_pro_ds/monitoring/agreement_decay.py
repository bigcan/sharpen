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
test window for the chosen rule). The tracker fires:

  * WARN when |live − baseline| > `agreement_flat_delta_warn` sustained
    for `window_bars`
  * CRIT when |live − baseline| > `agreement_flat_delta_crit` sustained
    for `window_bars`

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
            from the resolved bundle config). For `ens_agreement`, when
            consensus fails the aggregator returns exactly 0, so any
            sensible deadband (>0) classifies the bar as flat.
        window_bars: rolling window size (default 2000 per spec).
        min_bars_before_check: warmup floor; gating is suppressed below
            this count (default = window_bars / 2).
        warn_delta / crit_delta: |live − baseline| thresholds.
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
    ):
        if window_bars < 100:
            raise ValueError("window_bars must be >= 100")
        if warn_delta <= 0 or crit_delta <= 0:
            raise ValueError("warn_delta / crit_delta must be > 0")
        if warn_delta >= crit_delta:
            raise ValueError("warn_delta must be < crit_delta")
        if deadband <= 0:
            raise ValueError("deadband must be > 0")

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
        # is reported separately (and AgreementDecayTracker fires CRIT on
        # sustained no-consensus regimes via its own thresholds — TODO 5.5).
        n_no_consensus = sum(self._no_consensus_flags)
        n_consensus = n - n_no_consensus
        no_consensus_frac = n_no_consensus / n if n else 0.0
        if n_consensus == 0:
            return AgreementDecayReport(
                status=AgreementDecayStatus.LOG_ONLY,
                reason=(
                    f"all {n} bars in window were no-consensus (NaN); "
                    f"no flat-fraction signal computable"
                ),
                n_bars=n,
                flat_bar_frac_live=None,
                flat_bar_frac_baseline=self.baseline_flat_frac,
                flat_bar_frac_delta=None,
                rule=self.rule,
                no_consensus_frac=no_consensus_frac,
            )
        live = sum(self._flat_flags) / n_consensus

        if self.baseline_flat_frac is None:
            return AgreementDecayReport(
                status=AgreementDecayStatus.LOG_ONLY,
                reason="no baseline flat_bar_frac available",
                n_bars=n,
                flat_bar_frac_live=live,
                flat_bar_frac_baseline=None,
                flat_bar_frac_delta=None,
                rule=self.rule,
                no_consensus_frac=no_consensus_frac,
            )

        delta = abs(live - self.baseline_flat_frac)

        if delta > self.crit_delta:
            return AgreementDecayReport(
                status=AgreementDecayStatus.CRIT,
                reason=(
                    f"CRIT: flat_frac_delta={delta:.3f} "
                    f"(crit={self.crit_delta}) — consensus has decayed; "
                    f"capital utilization at risk"
                ),
                n_bars=n,
                flat_bar_frac_live=live,
                flat_bar_frac_baseline=self.baseline_flat_frac,
                flat_bar_frac_delta=delta,
                rule=self.rule,
                no_consensus_frac=no_consensus_frac,
            )
        if delta > self.warn_delta:
            return AgreementDecayReport(
                status=AgreementDecayStatus.WARN,
                reason=(
                    f"WARN: flat_frac_delta={delta:.3f} "
                    f"(warn={self.warn_delta})"
                ),
                n_bars=n,
                flat_bar_frac_live=live,
                flat_bar_frac_baseline=self.baseline_flat_frac,
                flat_bar_frac_delta=delta,
                rule=self.rule,
                no_consensus_frac=no_consensus_frac,
            )
        return AgreementDecayReport(
            status=AgreementDecayStatus.OK,
            reason=f"within thresholds (delta={delta:.3f})",
            n_bars=n,
            flat_bar_frac_live=live,
            flat_bar_frac_baseline=self.baseline_flat_frac,
            flat_bar_frac_delta=delta,
            rule=self.rule,
            no_consensus_frac=no_consensus_frac,
        )
