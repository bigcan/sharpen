"""Protocol v2.2 §8.2 live action-distribution drift tracker.

An RL agent in unseen conditions often does not fail loudly — it can quietly
collapse to a single safe action (for V7 scalar envs, the deadband-flat case).
Feature drift (§8.1) in isolation misses this because observations can remain
in-distribution while the *policy's response* has shifted. This tracker
computes the distance between a rolling live-action window and an
eval-time baseline, regime-conditioned on realized volatility.

Scalar action spaces: `deadband_frac_delta`, `saturation_frac_delta`.
Multi-dim (`Box(-1,1,(K,))`): per-asset marginal KL divergence — joint KL at
20-asset (CryptoPerp / Funding-Arb) is intractable and the marginal is the
informative view per §8.2.

The tracker ONLY computes and emits a status; it does not take engine
actions. WARN (disable new entries) and CRIT (kill_file + flatten) are the
engine/watchdog's responsibilities (§8.3).
"""

from __future__ import annotations

import collections
import logging
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---- Status ----------------------------------------------------------------

class DriftStatus:
    """String constants for drift status — kept plain for JSON-friendly logs."""
    OK = "OK"
    WARMUP = "WARMUP"
    LOG_ONLY = "LOG_ONLY"   # no baseline → record but don't fire
    WARN = "WARN"
    CRIT = "CRIT"
    # v2.6 (S551-cont-4): bar excluded from drift accumulation because the
    # observation features were flat (LiveObsBuilder.feature_variance_status()
    # returned "FLAT"). Engine continues normal trading — VETOED is NOT a halt
    # signal — it only protects the histogram from degenerate inputs.
    VETOED = "VETOED"


@dataclass
class DriftReport:
    status: str
    reason: str
    n_bars: int
    bucket: Optional[str]
    deadband_frac_live: Optional[float]
    deadband_frac_baseline: Optional[float]
    deadband_frac_delta: Optional[float]
    saturation_frac_live: Optional[float]
    saturation_frac_baseline: Optional[float]
    saturation_frac_delta: Optional[float]
    kl: Optional[float]
    # Fix 2 (S538-cont, 2026-05-08): fraction of bars where the aggregator
    # emitted NaN (no consensus). NaN bars are excluded from the deadband /
    # saturation denominator since they are "unobserved" rather than "flat".
    no_consensus_frac: Optional[float] = None
    # v2.6 (S551-cont-4): rolling fraction of FLAT-feature bars in the window
    # and the snapshot of feature_state passed to observe() for this bar.
    flat_veto_frac: Optional[float] = None
    feature_state: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "status": self.status, "reason": self.reason, "n_bars": self.n_bars,
            "bucket": self.bucket,
            "deadband_frac_live": self.deadband_frac_live,
            "deadband_frac_baseline": self.deadband_frac_baseline,
            "deadband_frac_delta": self.deadband_frac_delta,
            "saturation_frac_live": self.saturation_frac_live,
            "saturation_frac_baseline": self.saturation_frac_baseline,
            "saturation_frac_delta": self.saturation_frac_delta,
            "kl": self.kl,
            "no_consensus_frac": self.no_consensus_frac,
            "flat_veto_frac": self.flat_veto_frac,
            "feature_state": self.feature_state,
        }


# ---- KL helper -------------------------------------------------------------

def _marginal_kl_from_hists(
    live_counts: np.ndarray,
    baseline_counts: np.ndarray,
    smoothing: float = 1e-6,
) -> float:
    """KL(live || baseline) on matching histogram bins.

    `smoothing` is added to both count vectors before normalization to handle
    zero-support bins (KL is infinite for `p_baseline[i] = 0 < p_live[i]`).
    The smoothing is symmetric so identical histograms give KL=0.
    """
    p = np.asarray(live_counts, dtype=np.float64) + smoothing
    q = np.asarray(baseline_counts, dtype=np.float64) + smoothing
    p = p / p.sum()
    q = q / q.sum()
    return float((p * np.log(p / q)).sum())


# ---- Tracker ---------------------------------------------------------------

class ActionDriftTracker:
    """Per-strategy rolling action-distribution drift tracker.

    Baseline schema matches the Protocol v2.2 §2 `eval_distribution` block
    emitted by the Stage 2 / 2.5 report writers (see
    `sharpen.reporting.eval_distribution`). The tracker reads:

      * scalar: `deadband_frac`, `saturation_frac`, and optionally
        `by_vol_quartile.{q1..q4}.{deadband_frac,saturation_frac}` plus
        `histogram_bins` + `counts` (used when a KL fallback is requested).
      * multi-dim: `by_asset.<k>.{histogram_bins,counts}` — KL is computed
        per-asset marginal.

    `regime_cutpoints` (monotonic sequence of 3 floats) is optional. When
    present, the tracker buckets incoming bars by the live rolling realized
    vol → quartile, and compares against `by_vol_quartile.q<N>`. Without
    cutpoints the tracker falls back to the global baseline block and logs
    a one-time WARN that regime bucketing is disabled.

    `window_bars`, `min_bars_before_check`, and all threshold numbers come
    from `configs/<workstream>.gates.yaml` under `drift:` / `safe_mode:`.
    The tracker is not responsible for loading them — callers pass resolved
    values.
    """

    def __init__(
        self,
        baseline: Optional[dict],
        *,
        window_bars: int = 1000,
        min_bars_before_check: int = 500,
        deadband_warn: float = 0.15,
        deadband_crit: float = 0.30,
        saturation_warn: float = 0.15,
        saturation_crit: float = 0.30,
        action_kl_warn: float = 0.5,
        action_kl_crit: float = 1.0,
        deadband_abs: float = 0.25,
        saturation_abs: float = 0.95,
        regime_cutpoints: Optional[Sequence[float]] = None,
        vol_estimator_bars: int = 20,
        hist_edges: Optional[Sequence[float]] = None,
        asset_keys: Optional[Sequence[str]] = None,
        max_veto_frac: float = 0.50,
    ):
        if window_bars < 50:
            raise ValueError("window_bars must be >= 50")
        if min_bars_before_check > window_bars:
            raise ValueError("min_bars_before_check must be <= window_bars")
        self.baseline = baseline
        self.window_bars = int(window_bars)
        self.min_bars_before_check = int(min_bars_before_check)
        self.deadband_warn = float(deadband_warn)
        self.deadband_crit = float(deadband_crit)
        self.saturation_warn = float(saturation_warn)
        self.saturation_crit = float(saturation_crit)
        self.action_kl_warn = float(action_kl_warn)
        self.action_kl_crit = float(action_kl_crit)
        self.deadband_abs = float(deadband_abs)
        self.saturation_abs = float(saturation_abs)
        self.vol_estimator_bars = int(vol_estimator_bars)
        if not 0.0 < float(max_veto_frac) < 1.0:
            raise ValueError(
                f"max_veto_frac must be in (0, 1); got {max_veto_frac!r}"
            )
        self.max_veto_frac = float(max_veto_frac)

        # Multi-dim detection
        self._is_scalar: Optional[bool] = None
        self._n_dim: Optional[int] = None

        # Rolling buffers
        self._actions: collections.deque[np.ndarray] = collections.deque(
            maxlen=self.window_bars,
        )
        self._close_returns: collections.deque[float] = collections.deque(
            maxlen=self.window_bars,
        )
        self._prev_close: Optional[float] = None

        # v2.6 (S551-cont-4): rolling per-bar feature_state for the
        # max_veto_frac ADR-5 escalation. Tracks ALL bars (including FLAT
        # vetoes that are excluded from _actions), so the ratio reflects the
        # share of the last `window_bars` bars that were vetoed.
        self._recent_feature_states: collections.deque[str] = collections.deque(
            maxlen=self.window_bars,
        )
        # Monotonic counter — diagnostic only (lifetime veto count).
        self._flat_veto_count: int = 0

        # Histogram bin config — prefer baseline's bins so live/baseline align
        if hist_edges is not None:
            self.hist_edges: tuple[float, ...] = tuple(hist_edges)
        elif baseline is not None and "histogram_bins" in baseline:
            self.hist_edges = tuple(baseline["histogram_bins"])
        else:
            self.hist_edges = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)

        # Regime cutpoints (3 floats → 4 quartiles). When None, regime
        # bucketing is disabled and comparison uses the top-level baseline.
        # Prefer explicit cutpoints, then fall back to baseline-embedded
        # cutpoints (EVAL-DIST-CUTPOINTS-01 fix).
        if regime_cutpoints is None and baseline is not None:
            baseline_cuts = baseline.get("regime_cutpoints")
            if baseline_cuts is not None:
                regime_cutpoints = baseline_cuts
        if regime_cutpoints is not None and len(regime_cutpoints) != 3:
            raise ValueError("regime_cutpoints must have length 3 (q25,q50,q75)")
        self.regime_cutpoints = (
            tuple(regime_cutpoints) if regime_cutpoints is not None else None
        )

        # Expected per-asset ordering for multi-dim baselines. When provided
        # and the baseline's by_asset keys are in a DIFFERENT order, raise on
        # first observe() to catch silent mis-mapping of per-asset marginals.
        self.expected_asset_keys: Optional[tuple[str, ...]] = (
            tuple(asset_keys) if asset_keys is not None else None
        )
        if (
            self.expected_asset_keys is not None
            and baseline is not None
            and isinstance(baseline.get("by_asset"), dict)
        ):
            baseline_keys = tuple(baseline["by_asset"].keys())
            if baseline_keys != self.expected_asset_keys:
                raise ValueError(
                    f"baseline by_asset keys {baseline_keys} do not match "
                    f"expected asset_keys {self.expected_asset_keys} — this "
                    f"would silently mis-map per-asset marginals. Fix the "
                    f"baseline ordering or the live config's asset_keys."
                )

        self._regime_warning_logged = False
        if baseline is None:
            logger.warning(
                "ActionDriftTracker initialized without baseline — live-only "
                "mode (status=LOG_ONLY for all checks)",
            )
        elif self.regime_cutpoints is None and _has_vq(baseline):
            logger.warning(
                "ActionDriftTracker: baseline has by_vol_quartile but no "
                "regime_cutpoints supplied — comparing against global baseline",
            )

    # --- public API -----------------------------------------------------

    def observe(
        self,
        action: float | Sequence[float] | np.ndarray,
        bar_close: Optional[float],
        *,
        feature_state: Optional[str] = None,
    ) -> DriftReport:
        """Append one bar. Returns the post-append drift status.

        Args:
            action: scalar or K-vector matching the dim seen on first call.
            bar_close: current bar's close — used for realized-vol bucketing.
                `None` falls back to the global baseline for this call.
            feature_state: v2.6 — one of {"OK","FLAT","EXPLODE"} or ``None``
                (≡ "OK"). When ``"FLAT"``, the bar is excluded from the action
                histogram (Mode B mitigation: flat features ≠ informative
                policy output). `bar_close` IS still appended to the vol
                buffer because flat features do not imply flat prices. The
                returned report has ``status=VETOED``.

        Invariants:
            * The dim-change guard fires on every call, including VETOED bars.
            * Every call advances ``_recent_feature_states`` so the rolling
              ``flat_veto_frac`` reflects the last ``window_bars`` calls.
        """
        fs = "OK" if feature_state is None else feature_state
        if fs not in ("OK", "FLAT", "EXPLODE"):
            raise ValueError(
                f"feature_state must be None, 'OK', 'FLAT', or 'EXPLODE'; "
                f"got {feature_state!r}"
            )

        a = np.atleast_1d(np.asarray(action, dtype=np.float64)).ravel()
        if self._n_dim is None:
            self._n_dim = int(a.size)
            self._is_scalar = self._n_dim == 1
        elif a.size != self._n_dim:
            raise ValueError(
                f"action dim changed: first={self._n_dim} now={a.size}",
            )

        # Vol bucketing advances on every observation — flat features can
        # coincide with very active prices (a tight-range chop is flat in
        # engineered features but the close still moves).
        if bar_close is not None:
            if self._prev_close is not None and self._prev_close > 0 and bar_close > 0:
                try:
                    r = math.log(bar_close / self._prev_close)
                    self._close_returns.append(r)
                except (ValueError, ZeroDivisionError):
                    pass
            self._prev_close = float(bar_close)

        self._recent_feature_states.append(fs)

        if fs == "FLAT":
            self._flat_veto_count += 1
            return self._veto_report()

        self._actions.append(a)
        return self._evaluate(feature_state=fs)

    def snapshot(self) -> dict:
        """Current window stats without mutating state. For WandB logging."""
        return self._evaluate().to_dict()

    def _flat_veto_frac(self) -> float:
        """Rolling fraction of ``_recent_feature_states`` equal to ``"FLAT"``.

        Denominator is ``len(self._recent_feature_states)`` (capped at
        ``window_bars``), NOT ``len(_actions) + _flat_veto_count`` — using the
        deque keeps the ratio honest as old vetoes roll off, which is the only
        interpretation that makes ADR-5's "if 80% of bars in a window are
        vetoed" semantics work for a rolling-window tracker.
        """
        if not self._recent_feature_states:
            return 0.0
        states = list(self._recent_feature_states)
        return sum(1 for s in states if s == "FLAT") / len(states)

    def _veto_report(self) -> DriftReport:
        """DriftReport returned for FLAT bars. Engine treats VETOED as a silent
        pass-through (see `LiveTradingEngine._apply_drift_status`)."""
        return DriftReport(
            status=DriftStatus.VETOED,
            reason="feature_variance_flat: bar skipped from drift accumulation",
            n_bars=len(self._actions),
            bucket=None,
            deadband_frac_live=None, deadband_frac_baseline=None,
            deadband_frac_delta=None,
            saturation_frac_live=None, saturation_frac_baseline=None,
            saturation_frac_delta=None,
            kl=None, no_consensus_frac=None,
            flat_veto_frac=self._flat_veto_frac(),
            feature_state="FLAT",
        )

    # --- internals ------------------------------------------------------

    def _evaluate(self, *, feature_state: Optional[str] = None) -> DriftReport:
        n = len(self._actions)
        if n < self.min_bars_before_check:
            report = DriftReport(
                status=DriftStatus.WARMUP,
                reason=(
                    f"warmup: {n}/{self.min_bars_before_check} bars "
                    f"before first check"
                ),
                n_bars=n, bucket=None,
                deadband_frac_live=None, deadband_frac_baseline=None,
                deadband_frac_delta=None,
                saturation_frac_live=None, saturation_frac_baseline=None,
                saturation_frac_delta=None, kl=None,
                no_consensus_frac=None,
            )
        elif self.baseline is None:
            report = self._log_only_report(n)
        else:
            bucket_key = self._current_bucket()
            baseline_block = self._resolve_baseline_block(bucket_key)
            if self._is_scalar:
                report = self._evaluate_scalar(n, bucket_key, baseline_block)
            else:
                report = self._evaluate_multidim(n, bucket_key, baseline_block)

        # v2.6 (S551-cont-4): stamp feature_state + flat_veto_frac on every
        # report so WandB sees the veto cadence regardless of branch.
        report.feature_state = feature_state
        report.flat_veto_frac = self._flat_veto_frac()

        # ADR-5 escalation: too much veto saturation is itself a signal that
        # the bundle is stuck in degenerate regime — escalate OK → WARN so the
        # engine's existing drift_warn_no_new_entries hold-gate fires. Only
        # escalates OK; existing WARN/CRIT/LOG_ONLY/WARMUP/VETOED are left
        # alone so we never SOFTEN a stronger signal.
        if (
            report.status == DriftStatus.OK
            and report.flat_veto_frac is not None
            and report.flat_veto_frac > self.max_veto_frac
        ):
            report.status = DriftStatus.WARN
            report.reason = (
                f"feature_variance_veto_exceeded: "
                f"flat_veto_frac={report.flat_veto_frac:.3f} > "
                f"max={self.max_veto_frac:.3f}. "
                f"Recommend operator run scripts/recal_drift_baseline.py "
                f"(Mode A baseline staleness possible). See "
                f"decision_drift_two_failure_modes_s551_cont_3."
            )
        return report

    def _current_bucket(self) -> Optional[str]:
        if self.regime_cutpoints is None:
            return None
        if len(self._close_returns) < self.vol_estimator_bars:
            return None
        recent = np.asarray(
            list(self._close_returns)[-self.vol_estimator_bars:],
            dtype=np.float64,
        )
        live_vol = float(recent.std(ddof=0))
        if not math.isfinite(live_vol):
            return None
        c1, c2, c3 = self.regime_cutpoints
        if live_vol < c1:
            return "q1"
        if live_vol < c2:
            return "q2"
        if live_vol < c3:
            return "q3"
        return "q4"

    def _resolve_baseline_block(self, bucket_key: Optional[str]) -> Optional[dict]:
        if self.baseline is None:
            return None
        vq = self.baseline.get("by_vol_quartile")
        if bucket_key is not None and vq is not None and bucket_key in vq:
            return vq[bucket_key]
        return self.baseline

    def _log_only_report(self, n: int) -> DriftReport:
        live_actions = np.asarray(self._actions, dtype=np.float64)
        no_consensus_frac: Optional[float] = None
        if self._is_scalar:
            live = live_actions[:, 0]
            # Fix 2: filter NaN from deadband/saturation; report NaN share separately.
            nan_mask = np.isnan(live)
            no_consensus_frac = float(nan_mask.mean()) if n else 0.0
            live_clean = live[~nan_mask]
            if live_clean.size:
                live_dead = float((np.abs(live_clean) < self.deadband_abs).mean())
                live_sat = float((np.abs(live_clean) > self.saturation_abs).mean())
            else:
                live_dead = None
                live_sat = None
        else:
            live_dead = None
            live_sat = None
        return DriftReport(
            status=DriftStatus.LOG_ONLY,
            reason="no baseline eval_distribution available",
            n_bars=n, bucket=None,
            deadband_frac_live=live_dead, deadband_frac_baseline=None,
            deadband_frac_delta=None,
            saturation_frac_live=live_sat, saturation_frac_baseline=None,
            saturation_frac_delta=None, kl=None,
            no_consensus_frac=no_consensus_frac,
        )

    def _evaluate_scalar(
        self,
        n: int,
        bucket_key: Optional[str],
        baseline_block: Optional[dict],
    ) -> DriftReport:
        live = np.asarray(self._actions, dtype=np.float64)[:, 0]
        # Fix 2 (S538-cont): NaN sentinel = "no consensus, no actionable
        # signal". Excluded from deadband/saturation denominators; reported
        # separately as no_consensus_frac. The pre-Fix-2 zeros_like behavior
        # silently inflated deadband_frac and masked the dispersion-collapse
        # mechanism that drove the sg1-btc 6.62% DD incident.
        nan_mask = np.isnan(live)
        no_consensus_frac = float(nan_mask.mean()) if n else 0.0
        live_clean = live[~nan_mask]
        if live_clean.size:
            live_dead = float((np.abs(live_clean) < self.deadband_abs).mean())
            live_sat = float((np.abs(live_clean) > self.saturation_abs).mean())
        else:
            live_dead = None
            live_sat = None

        if baseline_block is None:
            return DriftReport(
                status=DriftStatus.LOG_ONLY,
                reason="baseline resolution returned None",
                n_bars=n, bucket=bucket_key,
                deadband_frac_live=live_dead, deadband_frac_baseline=None,
                deadband_frac_delta=None,
                saturation_frac_live=live_sat, saturation_frac_baseline=None,
                saturation_frac_delta=None, kl=None,
                no_consensus_frac=no_consensus_frac,
            )

        # All bars are no-consensus → cannot compare distributions
        # meaningfully; emit a degenerate report instead of NaN-vs-baseline.
        if live_dead is None:
            return DriftReport(
                status=DriftStatus.LOG_ONLY,
                reason="all bars in window were no-consensus (NaN)",
                n_bars=n, bucket=bucket_key,
                deadband_frac_live=None, deadband_frac_baseline=None,
                deadband_frac_delta=None,
                saturation_frac_live=None, saturation_frac_baseline=None,
                saturation_frac_delta=None, kl=None,
                no_consensus_frac=no_consensus_frac,
            )

        base_dead = float(baseline_block.get("deadband_frac", 0.0))
        base_sat = float(baseline_block.get("saturation_frac", 0.0))
        dead_delta = abs(live_dead - base_dead)
        sat_delta = abs(live_sat - base_sat)

        status = DriftStatus.OK
        reason = "within thresholds"
        if dead_delta > self.deadband_crit or sat_delta > self.saturation_crit:
            status = DriftStatus.CRIT
            reason = (
                f"CRIT: dead_delta={dead_delta:.3f} "
                f"(crit={self.deadband_crit}) "
                f"sat_delta={sat_delta:.3f} (crit={self.saturation_crit})"
            )
        elif dead_delta > self.deadband_warn or sat_delta > self.saturation_warn:
            status = DriftStatus.WARN
            reason = (
                f"WARN: dead_delta={dead_delta:.3f} "
                f"(warn={self.deadband_warn}) "
                f"sat_delta={sat_delta:.3f} (warn={self.saturation_warn})"
            )

        return DriftReport(
            status=status, reason=reason, n_bars=n, bucket=bucket_key,
            deadband_frac_live=live_dead, deadband_frac_baseline=base_dead,
            deadband_frac_delta=dead_delta,
            saturation_frac_live=live_sat, saturation_frac_baseline=base_sat,
            saturation_frac_delta=sat_delta, kl=None,
            no_consensus_frac=no_consensus_frac,
        )

    def _evaluate_multidim(
        self,
        n: int,
        bucket_key: Optional[str],
        baseline_block: Optional[dict],
    ) -> DriftReport:
        live = np.asarray(self._actions, dtype=np.float64)
        by_asset = baseline_block.get("by_asset") if baseline_block else None
        if by_asset is None:
            return DriftReport(
                status=DriftStatus.LOG_ONLY,
                reason="multi-dim baseline missing by_asset block",
                n_bars=n, bucket=bucket_key,
                deadband_frac_live=None, deadband_frac_baseline=None,
                deadband_frac_delta=None,
                saturation_frac_live=None, saturation_frac_baseline=None,
                saturation_frac_delta=None, kl=None,
                no_consensus_frac=None,
            )
        max_kl = 0.0
        bins = list(self.hist_edges)
        assert self._n_dim is not None, "observe() must set _n_dim before eval"
        # Keys in by_asset are in insertion order; if expected_asset_keys was
        # supplied, __init__ already validated that ordering matches. Without
        # that check the live action vector could silently mis-map to the
        # wrong asset's marginal.
        for i, (_asset_key, asset_baseline) in enumerate(by_asset.items()):
            if i >= self._n_dim:
                break
            asset_live = live[:, i]
            # Fix 2: filter NaN before histogram so multi-dim ensembles
            # (none today, defensive) get apples-to-apples KL.
            asset_live = asset_live[~np.isnan(asset_live)]
            if asset_live.size == 0:
                continue
            live_counts, _ = np.histogram(asset_live, bins=bins)
            base_counts = np.asarray(asset_baseline.get("counts", []), dtype=np.float64)
            if base_counts.size != live_counts.size:
                continue
            kl = _marginal_kl_from_hists(live_counts, base_counts)
            if kl > max_kl:
                max_kl = kl

        status = DriftStatus.OK
        reason = f"max per-asset marginal KL = {max_kl:.4f}"
        if max_kl > self.action_kl_crit:
            status = DriftStatus.CRIT
            reason = f"CRIT: max KL {max_kl:.4f} > crit {self.action_kl_crit}"
        elif max_kl > self.action_kl_warn:
            status = DriftStatus.WARN
            reason = f"WARN: max KL {max_kl:.4f} > warn {self.action_kl_warn}"

        # Multi-dim per-asset NaN frac (informational only — gating uses KL).
        nan_per_asset = float(np.isnan(live).any(axis=1).mean()) if live.size else 0.0
        return DriftReport(
            status=status, reason=reason, n_bars=n, bucket=bucket_key,
            deadband_frac_live=None, deadband_frac_baseline=None,
            deadband_frac_delta=None,
            saturation_frac_live=None, saturation_frac_baseline=None,
            saturation_frac_delta=None, kl=max_kl,
            no_consensus_frac=nan_per_asset,
        )


def _has_vq(baseline: dict) -> bool:
    return isinstance(baseline.get("by_vol_quartile"), dict)
