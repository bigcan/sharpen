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

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ...signals.eval_harness import _ann_sharpe
from ...signals.features import Panel
from ...signals.generation.evolve import _candidate_returns, _overlay_returns
from ...signals.generation.fitness import _CAND, FitnessConfig, _combined_book


@dataclass(frozen=True, slots=True)
class IncubationCriterion:
    """The pre-registered forward-incubation criterion (CR-2/CR-8). Loaded from
    ``configs/crucible_lockbox.gates.yaml`` and copied VERBATIM into each lockbox entry at
    enrollment — pinned per candidate, never edited afterward.

    * ``min_forward_bars`` — the fixed evaluation HORIZON (post-proposal bars) before ANY verdict is
      rendered. Outcome-independent, so the single evaluation at the horizon is a fixed-horizon test,
      not optional stopping / peeking.
    * ``min_forward_sharpe`` — the forward marginal-contribution annualized Sharpe the candidate must
      clear AT the horizon to become eligible for the human Tier-2 gate. Below it -> REJECTED.
    """

    min_forward_bars: int
    min_forward_sharpe: float

    def __post_init__(self) -> None:
        if int(self.min_forward_bars) < 2:
            raise ValueError("incubation.min_forward_bars must be >= 2")
        if not np.isfinite(self.min_forward_sharpe):
            raise ValueError("incubation.min_forward_sharpe must be finite")


@dataclass(frozen=True, slots=True)
class ForwardEvidence:
    """Accrued forward evidence for one candidate at one incubation pass (all on post-proposal bars)."""

    n_forward_bars: int
    forward_sharpe: float
    forward_start_ts: str | None
    forward_end_ts: str | None


# --- incubation criterion loader (no hardcoded gates — CLAUDE.md) ---------------------------------
_INCUBATION_DEFAULTS: dict = {"min_forward_bars": 63, "min_forward_sharpe": 0.30}


def load_incubation_criterion(gates_path: str | Path) -> IncubationCriterion:
    """Load the pre-registered incubation criterion from a gates YAML's ``incubation:`` block.

    Deliberately a SEPARATE file from ``signal_eval.gates.yaml``: the lockbox is a downstream forward
    gate, not part of the funnel's verdict, so its thresholds must not perturb the frozen funnel
    ``gates_hash`` (the moat). Missing block -> the documented defaults (back-compat)."""
    import yaml

    with open(gates_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    g = {**_INCUBATION_DEFAULTS, **dict(cfg.get("incubation", {}))}
    return IncubationCriterion(min_forward_bars=int(g["min_forward_bars"]),
                               min_forward_sharpe=float(g["min_forward_sharpe"]))


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
                    ls_min_names: int) -> np.ndarray | None:
    """The candidate's net-of-cost return stream on the FULL panel, via the funnel's OWN builders
    (never reimplemented). Overlay tilts the combined base book; cross-sectional is the rank-L/S
    sleeve. Returns None on a degenerate genome (all-NaN / constant), matching ``evolve``."""
    if candidate_type == "overlay":
        base_book = _combined_book(
            {str(k): np.asarray(v, dtype=np.float64) for k, v in base_returns.items()},
            timestamps, cfg)
        cr = _overlay_returns(formula, panel, base_book, cost_bps=cost_bps)
    elif candidate_type == "cross_sectional":
        cr = _candidate_returns(formula, panel, hold_horizon=hold_horizon,
                                cost_bps=cost_bps, min_names=ls_min_names)
    else:
        raise ValueError(f"candidate_type must be 'overlay' or 'cross_sectional'; got {candidate_type!r}")
    return None if cr is None else cr[0]


def forward_evidence(*, formula: str, candidate_type: str, panel: Panel,
                     base_returns: dict[str, np.ndarray], timestamps: np.ndarray,
                     proposal_ts: str, cfg: FitnessConfig, hold_horizon: int, cost_bps: float,
                     ls_min_names: int) -> ForwardEvidence | None:
    """Forward marginal-contribution evidence for a candidate (spec §6.2).

    Builds the candidate book and the C1 combined books CAUSALLY over the full panel (warmup may read
    pre-proposal history — that is legitimate real-time warmup), forms the marginal stream
    ``b_aug − b_base``, then slices it to bars STRICTLY after ``proposal_ts`` and returns the forward
    window's size + annualized Sharpe. Returns None if the candidate is degenerate or no forward bar
    exists yet (the panel has not extended past the proposal)."""
    cand_ret = _candidate_book(formula, candidate_type, panel, base_returns, timestamps, cfg,
                               hold_horizon=hold_horizon, cost_bps=cost_bps, ls_min_names=ls_min_names)
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
    return ForwardEvidence(n_forward_bars=n_fwd, forward_sharpe=float(sharpe),
                           forward_start_ts=_iso(float(fwd_secs.min())),
                           forward_end_ts=_iso(float(fwd_secs.max())))
