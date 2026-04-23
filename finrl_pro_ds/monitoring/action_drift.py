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
    `finrl_pro_ds.reporting.eval_distribution`). The tracker reads:

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

        # Histogram bin config — prefer baseline's bins so live/baseline align
        if hist_edges is not None:
            self.hist_edges: tuple[float, ...] = tuple(hist_edges)
        elif baseline is not None and "histogram_bins" in baseline:
            self.hist_edges = tuple(baseline["histogram_bins"])
        else:
            self.hist_edges = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)

        # Regime cutpoints (3 floats → 4 quartiles). When None, regime
        # bucketing is disabled and comparison uses the top-level baseline.
        if regime_cutpoints is not None and len(regime_cutpoints) != 3:
            raise ValueError("regime_cutpoints must have length 3 (q25,q50,q75)")
        self.regime_cutpoints = (
            tuple(regime_cutpoints) if regime_cutpoints is not None else None
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
    ) -> DriftReport:
        """Append one bar. Returns the post-append drift status.

        `bar_close` is the market close price for the current bar — used to
        estimate realized vol for regime bucketing. If `None`, bucketing
        falls back to global baseline for this call.
        """
        a = np.atleast_1d(np.asarray(action, dtype=np.float64)).ravel()
        if self._n_dim is None:
            self._n_dim = int(a.size)
            self._is_scalar = self._n_dim == 1
        elif a.size != self._n_dim:
            raise ValueError(
                f"action dim changed: first={self._n_dim} now={a.size}",
            )

        self._actions.append(a)

        if bar_close is not None:
            if self._prev_close is not None and self._prev_close > 0 and bar_close > 0:
                try:
                    r = math.log(bar_close / self._prev_close)
                    self._close_returns.append(r)
                except (ValueError, ZeroDivisionError):
                    pass
            self._prev_close = float(bar_close)

        return self._evaluate()

    def snapshot(self) -> dict:
        """Current window stats without mutating state. For WandB logging."""
        return self._evaluate().to_dict()

    # --- internals ------------------------------------------------------

    def _evaluate(self) -> DriftReport:
        n = len(self._actions)
        if n < self.min_bars_before_check:
            return DriftReport(
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
            )

        if self.baseline is None:
            return self._log_only_report(n)

        bucket_key = self._current_bucket()
        baseline_block = self._resolve_baseline_block(bucket_key)

        if self._is_scalar:
            return self._evaluate_scalar(n, bucket_key, baseline_block)
        return self._evaluate_multidim(n, bucket_key, baseline_block)

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
        if self._is_scalar:
            live = live_actions[:, 0]
            live_dead = float((np.abs(live) < self.deadband_abs).mean())
            live_sat = float((np.abs(live) > self.saturation_abs).mean())
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
        )

    def _evaluate_scalar(
        self,
        n: int,
        bucket_key: Optional[str],
        baseline_block: Optional[dict],
    ) -> DriftReport:
        live = np.asarray(self._actions, dtype=np.float64)[:, 0]
        live_dead = float((np.abs(live) < self.deadband_abs).mean())
        live_sat = float((np.abs(live) > self.saturation_abs).mean())

        if baseline_block is None:
            return DriftReport(
                status=DriftStatus.LOG_ONLY,
                reason="baseline resolution returned None",
                n_bars=n, bucket=bucket_key,
                deadband_frac_live=live_dead, deadband_frac_baseline=None,
                deadband_frac_delta=None,
                saturation_frac_live=live_sat, saturation_frac_baseline=None,
                saturation_frac_delta=None, kl=None,
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
            )
        max_kl = 0.0
        bins = list(self.hist_edges)
        # Keys in by_asset are in insertion order; assume training ordered
        # is the same as live action ordering. This is the same assumption
        # the rest of the multi-asset pipeline makes (see CryptoPerpEnv).
        for i, (asset_key, asset_baseline) in enumerate(by_asset.items()):
            if i >= self._n_dim:
                break
            asset_live = live[:, i]
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

        return DriftReport(
            status=status, reason=reason, n_bars=n, bucket=bucket_key,
            deadband_frac_live=None, deadband_frac_baseline=None,
            deadband_frac_delta=None,
            saturation_frac_live=None, saturation_frac_baseline=None,
            saturation_frac_delta=None, kl=max_kl,
        )


def _has_vq(baseline: dict) -> bool:
    return isinstance(baseline.get("by_vol_quartile"), dict)
