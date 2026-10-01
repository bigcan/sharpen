"""Forward-incubation evidence — the CR-8 lockbox measurement (spec §6.2).

The lockbox judges a PROMISING survivor on data that **did not exist when its hypothesis was
written**: it accrues evidence ONLY on bars timestamped strictly after ``proposal_ts``. This module
owns the pure measurement — given a candidate formula and the (forward-extended) substrate panel, it
computes the candidate's **forward marginal-contribution Sharpe**. The lockbox state machine
(:mod:`.lockbox`) owns enrollment, the pre-registered criterion, and the verdict.

Two design calls, both to keep the moat honest:

* **Marginal, not standalone.** The funnel selects a candidate on its *marginal* uplift to the C1
  combined book (``b_aug − b_base``; GP4-03 HLZ is on that stream, never standalone strength). So the
  forward gate measures the SAME quantity forward — an honest diversifier whose own Sharpe is modest
  but whose incremental book return is real is judged on its incremental return, exactly as in-sample.
  We reuse the funnel's own book builders (``_candidate_returns`` / ``_overlay_returns`` /
  ``_combined_book``) rather than reimplement them, so the forward number is the same statistic the
  funnel computed, just on unseen data — no divergent (leak-prone) second implementation.

* **Warmup may read the past; evidence may not.** The signal / combiner weights at a forward bar ``t``
  legitimately use history ≤ t (that is what a live trader sees in real time) — this is feature
  warmup, not look-ahead. What is forbidden is *counting P&L* from bars ≤ ``proposal_ts`` as forward
  evidence. So the candidate book and the combined book are built causally over the FULL panel (warmup
  free to use pre-proposal history), and only the realized marginal-return SERIES is sliced to bars
  strictly after ``proposal_ts`` (LEAK-2 pattern: warmup carry allowed, evidence window sliced). A
  negative tripwire (``tests/crucible/test_lockbox.py``) corrupts the pre-proposal bars and asserts
  the forward Sharpe is unchanged — a pre-proposal bar must never leak into the forward number.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ...signals.eval_harness import _ann_sharpe
from ...signals.features import Panel
from ...signals.generation.evolve import _candidate_returns, _overlay_returns
from ...signals.generation.fitness import (
    _CAND,
    FitnessConfig,
    _combined_book,
    _combined_book_with_components,
    augmented_book,
)

if TYPE_CHECKING:
    from ...signals.generation.base_sleeves import SleeveComponents


#: The lockbox decision rules. ``fixed`` is the pre-v16 rule, kept so an entry enrolled under it is
#: judged by it (CR-2: the criterion is pinned at enrollment); ``sprt`` is the crucible-v16.0 rule.
INCUBATION_TESTS = ("fixed", "sprt")


@dataclass(frozen=True, slots=True)
class IncubationCriterion:
    """The pre-registered forward-incubation criterion (CR-2/CR-8). Loaded from
    ``configs/crucible_lockbox.gates.yaml`` and copied VERBATIM into each lockbox entry at
    enrollment — pinned per candidate, never edited afterward.

    ``test="fixed"`` (pre-v16, the library default so old constructions keep their meaning):

    * ``min_forward_bars`` — the fixed evaluation HORIZON (post-proposal bars) before ANY verdict is
      rendered. Outcome-independent, so the single evaluation at the horizon is a fixed-horizon test,
      not optional stopping / peeking.
    * ``min_forward_sharpe`` — the forward marginal-contribution annualized Sharpe the candidate must
      clear AT the horizon to become eligible for the human Tier-2 gate. Below it -> REJECTED.

    ``test="sprt"`` (crucible-v16.0, the shipped gates). The fixed rule had two defects. Its statistic,
    the Sharpe of ``b_aug − b_base``, is the Tier-C-sealed substitution residual (``w·(r_c − b_base)``):
    a genuine variance-reducing diversifier has a NEGATIVE one, so the lockbox would reject exactly the
    candidates the corrected contract promotes. And one 63-bar look at a 0.30 floor is a coin flip:
    P(clear) 0.44 at zero edge vs 0.64 at IR 1.0. The v16 rule measures the forward Sharpe DIFFERENCE
    (the statistic the corrected contract certifies) and decides it with Wald's sequential probability
    ratio test on ``block_bars`` blocks:

    * ``delta_sr_h1`` — the alternative: the annualized ΔSR a candidate worth a human's time delivers
      forward. H0 is ΔSR = 0.
    * ``alpha`` / ``beta`` — P(CLEARED | H0) and P(REJECTED | H1). Boundaries ln((1−β)/α) and
      ln(β/(1−α)); no verdict before ``min_forward_bars``.
    * ``max_forward_bars`` — the cap: still undecided there ⇒ INCONCLUSIVE (terminal, not eligible).
    * ``calib_bars`` — the pre-proposal window that pins each book's vol and the block noise scale.
    """

    min_forward_bars: int
    min_forward_sharpe: float
    test: str = "fixed"
    max_forward_bars: int = 756
    block_bars: int = 21
    calib_bars: int = 504
    delta_sr_h1: float = 0.30
    alpha: float = 0.10
    beta: float = 0.20

    def __post_init__(self) -> None:
        if int(self.min_forward_bars) < 2:
            raise ValueError("incubation.min_forward_bars must be >= 2")
        if not np.isfinite(self.min_forward_sharpe):
            raise ValueError("incubation.min_forward_sharpe must be finite")
        if self.test not in INCUBATION_TESTS:
            raise ValueError(f"incubation.test must be one of {INCUBATION_TESTS}; got {self.test!r}")
        if self.test == "sprt":
            if not (0.0 < self.alpha < 1.0 and 0.0 < self.beta < 1.0 and self.alpha + self.beta < 1.0):
                raise ValueError("incubation.alpha / beta must lie in (0, 1) with alpha + beta < 1")
            if int(self.block_bars) < 2 or int(self.calib_bars) < 2 * int(self.block_bars):
                raise ValueError("incubation.block_bars must be >= 2 and calib_bars >= 2 * block_bars")
            if int(self.max_forward_bars) < int(self.min_forward_bars):
                raise ValueError("incubation.max_forward_bars must be >= min_forward_bars")
            if not (np.isfinite(self.delta_sr_h1) and self.delta_sr_h1 > 0.0):
                raise ValueError("incubation.delta_sr_h1 must be a finite positive ΔSR")

    @property
    def log_upper(self) -> float:
        """SPRT acceptance boundary for H1 (CLEARED): ln((1 − β) / α)."""
        return math.log((1.0 - self.beta) / self.alpha)

    @property
    def log_lower(self) -> float:
        """SPRT acceptance boundary for H0 (REJECTED): ln(β / (1 − α))."""
        return math.log(self.beta / (1.0 - self.alpha))


@dataclass(frozen=True, slots=True)
class ForwardEvidence:
    """Accrued forward evidence for one candidate at one incubation pass (all on post-proposal bars)."""

    n_forward_bars: int
    forward_sharpe: float
    forward_start_ts: str | None
    forward_end_ts: str | None
    # crucible-v16.0 SPRT evidence (NaN / 0 when the criterion is "fixed" or not yet calibratable)
    forward_delta_sr: float = float("nan")    # annualized SR(b_aug) − SR(b_base) on the forward bars
    sprt_llr: float = float("nan")            # Wald log-likelihood ratio over complete forward blocks
    n_blocks: int = 0


# --- incubation criterion loader (no hardcoded gates — CLAUDE.md) ---------------------------------
_INCUBATION_DEFAULTS: dict = {"min_forward_bars": 63, "min_forward_sharpe": 0.30, "test": "fixed"}
_SPRT_KEYS = ("max_forward_bars", "block_bars", "calib_bars", "delta_sr_h1", "alpha", "beta")


def load_incubation_criterion(gates_path: str | Path) -> IncubationCriterion:
    """Load the pre-registered incubation criterion from a gates YAML's ``incubation:`` block.

    Deliberately a SEPARATE file from ``signal_eval.gates.yaml``: the lockbox is a downstream forward
    gate, not part of the funnel's verdict, so its thresholds must not perturb the frozen funnel
    ``gates_hash`` (the moat). Missing block -> the documented defaults (back-compat)."""
    import yaml

    with open(gates_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    g = {**_INCUBATION_DEFAULTS, **dict(cfg.get("incubation", {}))}
    extra = {k: g[k] for k in _SPRT_KEYS if k in g}
    return IncubationCriterion(min_forward_bars=int(g["min_forward_bars"]),
                               min_forward_sharpe=float(g["min_forward_sharpe"]),
                               test=str(g["test"]),
                               **{k: (float(v) if k in ("delta_sr_h1", "alpha", "beta") else int(v))
                                  for k, v in extra.items()})


# --- forward-window slicing (the CR-8 keystone) ---------------------------------------------------
def _epoch_seconds_scalar(ts_iso: str) -> float:
    """A proposal timestamp (ISO string) -> POSIX seconds, normalized to UTC. tz-aware inputs are
    converted to UTC; tz-naive inputs are treated as UTC (never local-time, which would jitter the
    lockbox boundary by the machine's offset)."""
    t = pd.Timestamp(ts_iso)
    if t.tz is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return float(np.datetime64(t.to_datetime64(), "ns").astype("int64")) / 1e9


def _epoch_seconds_array(ts: np.ndarray) -> np.ndarray:
    """The panel timestamp axis -> POSIX seconds. Accepts a ``datetime64`` axis (panel dates) or a
    numeric epoch-seconds axis (the substrate ``timestamps`` convention)."""
    ts = np.asarray(ts)
    if np.issubdtype(ts.dtype, np.datetime64):
        return ts.astype("datetime64[ns]").astype("int64").astype(np.float64) / 1e9
    return ts.astype(np.float64)


def forward_mask(timestamps: np.ndarray, proposal_ts: str) -> np.ndarray:
    """Boolean mask of bars STRICTLY after ``proposal_ts`` — the CR-8 forward window.

    Strict ``>`` is intentional: the return realized at bar ``t`` spans ``t -> t+1``, so requiring
    ``timestamps[t] > proposal_ts`` guarantees that return is entirely post-proposal (a bar exactly
    at the proposal instant is excluded — its move began at proposal time)."""
    return _epoch_seconds_array(timestamps) > _epoch_seconds_scalar(proposal_ts)


def _iso(epoch_seconds: float) -> str:
    return pd.Timestamp(epoch_seconds, unit="s", tz="UTC").isoformat()


def _candidate_book(formula: str, candidate_type: str, panel: Panel,
                    base_returns: dict[str, np.ndarray], timestamps: np.ndarray,
                    cfg: FitnessConfig, *, hold_horizon: int, cost_bps: float,
                    ls_min_names: int,
                    base_components: "dict[str, SleeveComponents] | None" = None,
                    ) -> np.ndarray | None:
    """The candidate's net-of-cost return stream on the FULL panel, via the funnel's OWN builders
    (never reimplemented). Overlay tilts the combined base book; cross-sectional is the rank-L/S
    sleeve. Returns None on a degenerate genome (all-NaN / constant), matching ``evolve``.

    ``base_components`` (F14): overlay cost is charged against the base book's true gross / embedded
    cost when supplied; else the unit-gross fallback (exact for synthetic/proxy books)."""
    if candidate_type == "overlay":
        base_net = {str(k): np.asarray(v, dtype=np.float64) for k, v in base_returns.items()}
        if base_components is not None:
            base_book, bg, bc, ge = _combined_book_with_components(
                base_net, base_components, timestamps, cfg)
        else:
            base_book, bg, bc, ge = _combined_book(base_net, timestamps, cfg), None, None, None
        cr = _overlay_returns(formula, panel, base_book, cost_bps=cost_bps,
                              base_gross=bg, base_cost=bc, gross_exposure=ge)
    elif candidate_type == "cross_sectional":
        cr = _candidate_returns(formula, panel, hold_horizon=hold_horizon,
                                cost_bps=cost_bps, min_names=ls_min_names)
    else:
        raise ValueError(f"candidate_type must be 'overlay' or 'cross_sectional'; got {candidate_type!r}")
    return None if cr is None else cr[0]


def _sprt_evidence(base: dict[str, np.ndarray], cand: np.ndarray, timestamps: np.ndarray,
                   cfg: FitnessConfig, fwd: np.ndarray, crit: IncubationCriterion) -> dict:
    """The v16 forward evidence: ΔSR and Wald's log-likelihood ratio (see :class:`IncubationCriterion`).

    Books come from :func:`fitness.augmented_book` — the candidate joins only where the combiner can size
    it, the same book the corrected contract scores. The per-bar difference of VOL-STANDARDIZED returns
    ``d = b_aug/σ_aug − b_base/σ_base`` has mean SR_aug − SR_base (per bar), and because the two books
    share most of their holdings its noise is small (the Memmel pairing that powers the JKM test). The
    vols and the block noise scale ``s`` are pinned from the last ``calib_bars`` PRE-proposal bars — data
    that existed at enrollment, so every SPRT increment is a function of forward data alone. Block sums
    of ``d`` over ``block_bars`` absorb the autocorrelation of held books; with ``y_b = D_b / s`` and
    ``θ = (delta_sr_h1 / √ppy)·B / s``, the LLR is ``Σ_b (θ·y_b − θ²/2)``. Returns ``{}`` when the
    pre-proposal window is too short to calibrate (the entry keeps incubating)."""
    b_base, b_aug, _ = augmented_book(base, cand, timestamps, cfg)
    ok = np.isfinite(b_base) & np.isfinite(b_aug)
    blk = int(crit.block_bars)
    pre_idx = np.flatnonzero(ok & ~fwd)[-int(crit.calib_bars):]
    n_cal = pre_idx.size // blk
    if n_cal < 2:
        return {}
    sa, sb = float(np.std(b_aug[pre_idx], ddof=1)), float(np.std(b_base[pre_idx], ddof=1))
    if not (sa > 0.0 and sb > 0.0):
        return {}
    d = b_aug / sa - b_base / sb
    cal = d[pre_idx[pre_idx.size - n_cal * blk:]].reshape(n_cal, blk).sum(axis=1)
    s = float(np.std(cal, ddof=1))
    if not (s > 0.0 and np.isfinite(s)):
        return {}
    post_idx = np.flatnonzero(ok & fwd)
    n_blk = post_idx.size // blk
    fb = d[post_idx[: n_blk * blk]].reshape(n_blk, blk).sum(axis=1) if n_blk else np.zeros(0)
    theta = crit.delta_sr_h1 / math.sqrt(cfg.periods_per_year) * blk / s
    llr = float(np.sum(theta * (fb / s) - 0.5 * theta * theta))
    dsr = (_ann_sharpe(b_aug[post_idx], cfg.periods_per_year)
           - _ann_sharpe(b_base[post_idx], cfg.periods_per_year)) if post_idx.size >= 2 else math.nan
    return {"forward_delta_sr": float(dsr), "sprt_llr": llr, "n_blocks": int(n_blk)}


def forward_evidence(*, formula: str, candidate_type: str, panel: Panel,
                     base_returns: dict[str, np.ndarray], timestamps: np.ndarray,
                     proposal_ts: str, cfg: FitnessConfig, hold_horizon: int, cost_bps: float,
                     ls_min_names: int,
                     base_components: "dict[str, SleeveComponents] | None" = None,
                     criterion: IncubationCriterion | None = None,
                     ) -> ForwardEvidence | None:
    """Forward marginal-contribution evidence for a candidate (spec §6.2).

    Builds the candidate book and the C1 combined books CAUSALLY over the full panel (warmup may read
    pre-proposal history — that is legitimate real-time warmup), forms the marginal stream
    ``b_aug − b_base``, then slices it to bars STRICTLY after ``proposal_ts`` and returns the forward
    window's size + annualized Sharpe. Returns None if the candidate is degenerate or no forward bar
    exists yet (the panel has not extended past the proposal).

    With an ``sprt`` ``criterion`` (crucible-v16.0) the evidence also carries the forward ΔSR and the
    SPRT log-likelihood ratio the entry is judged on (:func:`_sprt_evidence`); the legacy marginal
    Sharpe is still reported, unchanged."""
    cand_ret = _candidate_book(formula, candidate_type, panel, base_returns, timestamps, cfg,
                               hold_horizon=hold_horizon, cost_bps=cost_bps, ls_min_names=ls_min_names,
                               base_components=base_components)
    if cand_ret is None:
        return None

    base = {str(k): np.asarray(v, dtype=np.float64) for k, v in base_returns.items()}
    aug = {**base, _CAND: np.asarray(cand_ret, dtype=np.float64)}
    marg = _combined_book(aug, timestamps, cfg) - _combined_book(base, timestamps, cfg)

    ts_sec = _epoch_seconds_array(timestamps)
    fwd = ts_sec > _epoch_seconds_scalar(proposal_ts)
    n = min(fwd.size, marg.size)                      # defensive length align (all should equal T)
    fwd, marg, ts_sec = fwd[:n], marg[:n], ts_sec[:n]
    fwd_finite = fwd & np.isfinite(marg)
    n_fwd = int(fwd_finite.sum())
    if n_fwd == 0:
        return ForwardEvidence(0, float("nan"), None, None)

    fwd_marg = marg[fwd_finite]
    sharpe = _ann_sharpe(fwd_marg, cfg.periods_per_year) if n_fwd >= 2 else float("nan")
    fwd_secs = ts_sec[fwd_finite]
    extra = (_sprt_evidence({k: v[:n] for k, v in base.items()},
                            np.asarray(cand_ret, dtype=np.float64)[:n], np.asarray(timestamps)[:n],
                            cfg, fwd, criterion)
             if criterion is not None and criterion.test == "sprt" else {})
    return ForwardEvidence(n_forward_bars=n_fwd, forward_sharpe=float(sharpe),
                           forward_start_ts=_iso(float(fwd_secs.min())),
                           forward_end_ts=_iso(float(fwd_secs.max())), **extra)
