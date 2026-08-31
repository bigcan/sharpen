"""Cohort MC null — novel statistic #2: the selection-aware Monte-Carlo null (the BINDING gate).

Design + math audit: ``docs/research/crucible_mc_null_spec.md`` (Doc 2 of 3). This is the
authoritative cohort verdict; the analytic ``SR*_cohort`` in :mod:`cohort` is only a cheap
pre-filter. A cohort recorded PROMISING must clear ``p ≤ α_cohort`` HERE **and** an embargoed
holdout (Doc 2 §4).

H₀ (Doc 2 §1): the cohort book has no genuine edge — its Sharpe is entirely attributable to
**selection** (which m of N) plus **combination** (√m variance reduction), given the candidates'
realized vol, autocorrelation, and cross-candidate covariance. This is the panel analogue of the
White (2000) / Hansen (2005) Reality Check, extended from "best single of N" to "best
select-and-combine of m-of-N".

The three ⛔ MATH-AUDIT BLOCKERs (each empirically demonstrated; each pinned as a regression test):
  1. **Resample the BASE sleeves too, jointly** (same block indices). Holding ``b_base`` fixed
     destroys candidate↔base covariance → size 8.3% > α. Demean ONLY the candidate columns.
  2. **The replicate carries the ORIGINAL monotone timestamps positionally.** The combiner is
     time-order-dependent (trailing vol/corr, month-end rotation via ``_monthly_held``); a scrambled
     axis produces silent garbage rotation. Only returns are randomized; the time axis is fixed.
  3. **The statistic is within-replicate ΔSR** ``SR_pp(base∪cohort) − SR_pp(base)``, NOT the
     marginal-stream Sharpe (which is ≤0 for genuine variance-reduction diversifiers → 6–9% power).

Determinism (Doc 2 §6): the bootstrap RNG is seeded from ``seed`` (the caller derives it from
``gates_hash ‖ pool_content_hash ‖ run_id``); no ``Math.random`` — ``crucible reproduce`` must
re-derive a byte-identical p-value. No threshold is hardcoded — ``α_cohort``, ``B``, ℓ and
``min_cohort_size`` arrive via :class:`~cohort.CohortConfig` / the gates block.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .cohort import (
    CohortConfig,
    _with_redundancy,
    greedy_decorrelated_admission,
    standalone_sharpe_scores,
)
from .fitness import FitnessConfig, _combined_book, _per_period_sharpe

logger = logging.getLogger(__name__)

_NEG_INF = float("-inf")


@dataclass(frozen=True, slots=True)
class McNullResult:
    """The MC-null verdict artifact."""

    t_obs: float                 # observed within-panel ΔSR
    p_value: float               # add-one one-sided p = (1 + #{T_b ≥ T_obs}) / (B + 1)
    n_reps: int                  # B
    n_valid_reps: int            # reps that admitted ≥ min_cohort_size (others contribute T_b=-inf)
    block_length: int            # ℓ used
    passes_mc: bool              # p ≤ alpha_cohort
    members_obs: tuple[str, ...]  # the observed admitted cohort (for the holdout guard)


# --------------------------------------------------------------------------------------------
# Stationary block bootstrap (Politis & Romano 1994) — joint over [base ‖ candidates]
# --------------------------------------------------------------------------------------------
def stationary_bootstrap_indices(t: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """A length-``t`` index vector drawn by the stationary bootstrap: geometric block lengths with
    mean ``block_length``, circular (wrap-around) so every position is equiprobable. The SAME index
    vector is applied to base + candidate columns jointly (BLOCKER 1), so panel covariance and each
    stream's within-block autocorrelation survive."""
    if t <= 0:
        return np.empty(0, dtype=np.int64)
    p = 1.0 / max(1, int(block_length))
    idx = np.empty(t, dtype=np.int64)
    pos = int(rng.integers(0, t))
    for k in range(t):
        idx[k] = pos
        if rng.random() < p:          # start a new block
            pos = int(rng.integers(0, t))
        else:                         # extend the current block (circular)
            pos = (pos + 1) % t
    return idx


def auto_block_length(columns: np.ndarray, *, fallback: int = 21) -> int:
    """A lightweight lag-1-AR block-length heuristic (median across columns), a pragmatic stand-in
    for Politis–White (2004). ℓ ≈ round( (1+ρ1)/(1−ρ1) )·baseline, clamped to [3, t//4]. The binding
    gate pins ℓ from config; this is used only when config asks for auto. The ℓ-sensitivity test
    proves calibration is stable across ℓ ∈ {5, 21, 63}, so the exact estimate is not load-bearing.
    """
    x = np.asarray(columns, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    t = x.shape[0]
    if t < 8:
        return max(1, min(fallback, t))
    ls: list[float] = []
    for j in range(x.shape[1]):
        col = x[:, j]
        m = np.isfinite(col)
        c = col[m]
        if c.size < 8 or c.std() <= 0:
            continue
        r1 = float(np.corrcoef(c[:-1], c[1:])[0, 1])
        r1 = max(-0.99, min(0.99, r1 if np.isfinite(r1) else 0.0))
        ls.append((1.0 + r1) / (1.0 - r1))
    if not ls:
        return max(1, min(fallback, t))
    ell = int(round(float(np.median(ls)) * 3.0))
    return max(3, min(ell, t // 4 if t >= 12 else t))


# --------------------------------------------------------------------------------------------
# The within-replicate statistic (identical functional on observed + null panels)
# --------------------------------------------------------------------------------------------
def _cohort_delta_sr(
    base: Mapping[str, np.ndarray],
    cand: Mapping[str, np.ndarray],
    timestamps: np.ndarray,
    ccfg: CohortConfig,
    fcfg: FitnessConfig,
    cohort_cfg: FitnessConfig,
) -> tuple[float, tuple[str, ...]]:
    """T = SR_pp(combine(base ∪ admit(cand))) − SR_pp(combine(base)) on ONE panel. Re-runs admission
    AND the combiner (never frozen — BLOCKER 2 anti-pattern). Returns (T, admitted members); T is
    -inf when fewer than ``min_cohort_size`` are admitted (Doc 2 §3.2)."""
    scores = standalone_sharpe_scores(cand)
    members = greedy_decorrelated_admission(
        cand, scores, max_pairwise_corr=ccfg.max_pairwise_corr, max_cohort_size=ccfg.max_cohort_size)
    b_base = _combined_book(dict(base), timestamps, fcfg)
    sr_base = _per_period_sharpe(b_base[np.isfinite(b_base)])
    if len(members) < ccfg.min_cohort_size or not np.isfinite(sr_base):
        return _NEG_INF, members
    aug = {**dict(base), **{m: np.asarray(cand[m], dtype=np.float64) for m in members}}
    b_aug = _combined_book(aug, timestamps, cohort_cfg)
    sr_aug = _per_period_sharpe(b_aug[np.isfinite(b_aug)])
    if not np.isfinite(sr_aug):
        return _NEG_INF, members
    return float(sr_aug - sr_base), members


# --------------------------------------------------------------------------------------------
# The MC null gate
# --------------------------------------------------------------------------------------------
def mc_null_pvalue(
    pool_returns: Mapping[str, np.ndarray],
    base_returns: Mapping[str, np.ndarray],
    timestamps: np.ndarray,
    ccfg: CohortConfig,
    fcfg: FitnessConfig,
    *,
    n_reps: int,
    alpha_cohort: float,
    seed: int,
    block_length: int | None = None,
    resample_base: bool = True,
) -> McNullResult:
    """Selection-aware MC null p-value for a cohort (Doc 2 §3). ``resample_base`` MUST be True for a
    valid gate — the ``False`` path exists ONLY for the BLOCKER-1 regression test (it reproduces the
    8.3% anti-conservative size and must stay caught). ``seed`` makes the bootstrap deterministic.

    Cost is O(B · (admission + 2·combiner)) — no per-candidate CPCV (the standalone-Sharpe ranking,
    Doc 2 §5(b)). Run the analytic ``SR*_cohort`` pre-filter first so this only fires on survivors.
    """
    ts = np.asarray(timestamps)
    base = {str(k): np.asarray(v, dtype=np.float64) for k, v in base_returns.items()}
    cand = {str(k): np.asarray(v, dtype=np.float64) for k, v in pool_returns.items()}
    names = list(cand)
    t = ts.size
    cohort_cfg = _with_redundancy(fcfg, ccfg.combiner_redundancy_strength)

    # ---- observed T on the REAL (non-demeaned, non-resampled) panel with its true timestamps ----
    t_obs, members_obs = _cohort_delta_sr(base, cand, ts, ccfg, fcfg, cohort_cfg)

    # ---- impose H0: demean CANDIDATES ONLY (base keeps its real drift — BLOCKER 1) ----
    cand_dm = {k: v - np.nanmean(v) for k, v in cand.items()}

    ell = int(block_length) if block_length else auto_block_length(
        np.column_stack([base[s] for s in base] + [cand_dm[c] for c in names]) if names else
        np.column_stack([base[s] for s in base]))

    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    n_ge = 0
    n_valid = 0
    for _b in range(int(n_reps)):
        idx = stationary_bootstrap_indices(t, ell, rng)
        if resample_base:
            base_b = {s: base[s][idx] for s in base}          # SAME indices (joint) — BLOCKER 1
        else:
            base_b = base                                     # anti-conservative variant (test only)
        cand_b = {c: cand_dm[c][idx] for c in names}
        # timestamps stay the ORIGINAL monotone axis, applied positionally — BLOCKER 2
        t_b, _ = _cohort_delta_sr(base_b, cand_b, ts, ccfg, fcfg, cohort_cfg)
        if t_b > _NEG_INF:
            n_valid += 1
        if t_b >= t_obs:                                      # ≥ tie-handling (Doc 2 §3.3)
            n_ge += 1
    p = (1.0 + n_ge) / (int(n_reps) + 1.0)
    passes = bool(np.isfinite(t_obs) and p <= alpha_cohort and len(members_obs) >= ccfg.min_cohort_size)
    return McNullResult(
        t_obs=float(t_obs), p_value=float(p), n_reps=int(n_reps), n_valid_reps=int(n_valid),
        block_length=int(ell), passes_mc=passes, members_obs=members_obs)
