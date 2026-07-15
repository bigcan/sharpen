"""Cohort evaluator ORCHESTRATION (Phase 4 integration) — the single library entry point that runs
the full weak-signal cohort gate end-to-end: assemble the overlay candidate pool → analytic
``SR*_cohort`` pre-filter (Doc 1) → selection-aware MC null (Doc 2 §3, binding) → embargoed holdout
guard (Doc 2 §4). It owns NO new statistic — it wires the Math-verified pieces in
:mod:`cohort` / :mod:`cohort_mc` in the pre-registered order (Doc 2 §5) and returns one
:class:`CohortVerdict` artifact.

Design: ``.agent/artifacts/crucible_cohort_integration_architecture.md`` (Step 1). This module is
PURE (signals-only — no crucible imports) so it stays free of the orchestrator import cycle; the P2
loop / P3 orchestrator call :func:`evaluate_cohort` and turn its verdict into a card + FDR test +
manifest provenance (Steps 4–5). Nothing here promotes past PROMISING (CLAUDE.md: Tier-2 for capital).

Split contract (matches ``evolve``): the pool is computed ONCE on the full panel (each overlay's
causal expanding z-score makes a full-panel value at bar ``t`` identical to a train-only value there,
LEAK-1/LEAK-2), then sliced — the analytic + MC run on the **train/selection span** ``[0, n_train)``
(``n_train = cut − embargo``); the holdout guard runs on the **embargoed tail** ``[cut, T)``, its
members + combiner weights frozen from the train pipeline (Doc 2 §4). Units: the holdout ΔSR is
**annualized** (``_ann_sharpe``) so it is compared against ``min_book_uplift`` in the SAME units the
funnel + :func:`evaluate_cohort_analytic` use (Doc 2 §4 unit-pin).

Determinism (Doc 2 §6): the MC seed is derived by the caller via :func:`derive_cohort_seed` from
``funnel_gates_hash ‖ cohort_gates_hash ‖ pool_content_hash ‖ run_id``; ``crucible reproduce`` must
re-derive a byte-identical p-value. No threshold is hardcoded — every number arrives via
:class:`CohortConfig` / ``mc_kwargs`` (built from ``configs/crucible_cohort.gates.yaml`` by the runner).
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np
import pandas as pd

from finrl_pro_ds.envs.allocator_factory import dynamic_sleeve_alphas

from ..eval_harness import _ann_sharpe
from ..features import Panel
from .cohort import (
    CohortConfig,
    CohortEvidence,
    _with_redundancy,
    evaluate_cohort_analytic,
)
from .cohort_mc import McNullResult, mc_null_pvalue
from .evolve import _overlay_returns
from .fitness import FitnessConfig, _combined_book, _combined_book_with_components

if TYPE_CHECKING:
    from .base_sleeves import SleeveComponents

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CohortVerdict:
    """The cohort gate artifact (one per evaluated tick). ``verdict == "PROMISING"`` iff the analytic
    floor PASSED, the MC null cleared (``p ≤ alpha_cohort``), AND the embargoed holdout ΔSR cleared
    ``min_book_uplift`` — all three (Doc 2 §4: MC and holdout fail differently, so both are required).
    A ``"LOGGED"`` verdict records the evidence at whichever stage it stopped."""

    members: tuple[str, ...]
    n_members: int
    n_candidates_seen: int
    n_culled: int
    mean_pairwise_corr: float
    sr_star_cohort: float
    dsr_cohort_book: float
    passes_analytic_floor: bool
    mc_p_value: float
    mc_t_obs: float
    mc_n_reps: int
    mc_n_valid_reps: int
    mc_block_length: int
    passes_mc: bool
    holdout_delta_sr: float
    holdout_passes: bool
    pool_content_hash: str
    verdict: str                       # "PROMISING" | "LOGGED"


# --------------------------------------------------------------------------------------------
# Determinism helpers (Doc 2 §6) — the caller derives the MC seed from these
# --------------------------------------------------------------------------------------------
def pool_content_hash(overlay_formulas: Mapping[str, str]) -> str:
    """12-hex SHA-256 over the sorted ``(name, formula)`` pairs of the pool (mirrors
    ``spec.content_hash``). Deterministic ⇒ the derived MC seed is reproducible; a changed pool ⇒ a
    changed hash ⇒ a different (visible) seed."""
    items = sorted((str(k), str(v)) for k, v in overlay_formulas.items())
    payload = "\n".join(f"{k}\t{v}" for k, v in items)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def derive_cohort_seed(
    funnel_gates_hash: str, cohort_gates_hash: str, pool_hash: str, run_id: str
) -> int:
    """``int_hash(funnel_gates_hash ‖ cohort_gates_hash ‖ pool_content_hash ‖ run_id)`` (Doc 2 §6,
    extended for the two-file gates split, ADR-1). Folds BOTH gate hashes so a cohort-gate edit
    re-derives a different p-value while the frozen funnel hash stays out of the funnel's own path."""
    payload = f"{funnel_gates_hash}|{cohort_gates_hash}|{pool_hash}|{run_id}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:15], 16)


# --------------------------------------------------------------------------------------------
# Pool assembly (promoted from the diversity driver — the SAME return-stream path)
# --------------------------------------------------------------------------------------------
def assemble_overlay_pool(
    panel: Panel,
    base_book: np.ndarray,
    overlay_formulas: Mapping[str, str],
    *,
    cost_bps: float,
    base_gross: "np.ndarray | None" = None,
    base_cost: "np.ndarray | None" = None,
    gross_exposure: "np.ndarray | None" = None,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Recompute each overlay formula's return stream via :func:`evolve._overlay_returns` on ``panel``
    against the combined ``base_book`` — exactly the funnel's OVERLAY scoring path, so the cohort
    admits from precisely the streams the pool-diversity instrument measured. Returns
    ``(returns, culled)``; ``culled`` are formulas :func:`_overlay_returns` rejected as degenerate
    (all-NaN / globally constant → None). Source/asset-class labelling is a diversity-REPORT concern
    and stays caller-side (it would pull a crucible import into this pure signals module).

    ``culled`` is the pre-registered "hard-infeasible / leak-culled" EXCLUSION (Doc 1 Algorithm step 1;
    Doc 2 §3.1): the deflation N is "all scored candidates with a **finite, non-degenerate** return
    stream" == ``len(returns)``, NOT ``len(returns) + len(culled)``. ``n_culled`` is recorded on the
    verdict for provenance only; it must NEVER feed ``n_candidates_seen``/``SR*_cohort`` (that would
    over-count N, break the m=1 BLdP calibration anchor, and desync from the MC null which re-admits
    exactly these ``len(returns)`` columns). LOCKED BY
    ``test_cohort_eval.py::test_n_candidates_seen_excludes_culled_not_all_scored``."""
    returns: dict[str, np.ndarray] = {}
    culled: list[str] = []
    for name, formula in overlay_formulas.items():
        out = _overlay_returns(str(formula), panel, base_book, cost_bps=cost_bps,
                               base_gross=base_gross, base_cost=base_cost,
                               gross_exposure=gross_exposure)   # F14 overlay-cost fix (unit if None)
        if out is None:
            culled.append(str(name))
            continue
        returns[str(name)] = out[0]
    return returns, culled


# --------------------------------------------------------------------------------------------
# Embargoed holdout guard (Doc 2 §4) — freeze members + last month-end α, apply to the tail
# --------------------------------------------------------------------------------------------
def _alpha_paths(returns: Mapping[str, np.ndarray], timestamps: np.ndarray,
                 cfg: FitnessConfig) -> dict[str, np.ndarray]:
    """The C1 combiner's per-sleeve α paths (same call ``_combined_book`` wraps, but keeping the
    weights instead of collapsing to the book) so we can freeze the last month-end vector."""
    return dynamic_sleeve_alphas(
        returns, timestamps, window=cfg.combiner_window,
        min_periods=cfg.combiner_min_periods, monthly_meta=cfg.combiner_monthly_meta,
        target_portfolio_vol=None, tilt_strength=cfg.tilt_strength,
        perf_window=cfg.perf_window, perf_min_periods=cfg.perf_min_periods,
        tilt_clip=cfg.tilt_clip, redundancy_strength=cfg.combiner_redundancy_strength)


def _frozen_weight_book(returns_ho: Mapping[str, np.ndarray],
                        frozen_alpha: Mapping[str, float]) -> np.ndarray:
    """``b(t) = Σ_s α_frozen[s] · r_s(t)`` on the holdout rows with a CONSTANT (frozen) weight vector
    — no re-estimation (Doc 2 §4). NaN-aware (a NaN sleeve return contributes 0 that bar)."""
    names = list(returns_ho)
    a = np.array([float(frozen_alpha[s]) for s in names], dtype=np.float64)              # (S,)
    r = np.stack([np.asarray(returns_ho[s], dtype=np.float64) for s in names], axis=1)   # (T, S)
    return np.nansum(a[None, :] * r, axis=1)


def _last_confirmed_month_end_idx(ts_train: np.ndarray) -> int:
    """Index into ``ts_train`` of the LAST CONFIRMED month-end held under ``monthly_meta`` — the bar
    whose combiner weights Doc 2 §4 requires the holdout guard to freeze. Derived from ``ts_train``
    ALONE via the SAME convention the combiner's ``_monthly_held`` uses (pandas period ``'M'``,
    groupby-max), so it lands on a bar the combiner actually rotated on.

    The combiner flags each month's max-index bar as a rotation point; on a span truncated MID-month
    the in-progress FINAL bar is falsely one of them (the P2-01 truncation caveat in
    ``allocator_factory.monthly_rebal_conviction``). We accept the final bar as a confirmed month-end
    ONLY when it is the true last calendar day of its month (``day == days_in_month`` — a property of
    that timestamp alone, no look-ahead); otherwise we drop it and take the previous month's rotation
    bar. Falls back to ``T-1`` (byte-identical to the old ``[-1]``) when no confirmed interior
    month-end exists in the span (a single partial month), so short spans are unaffected. Causal:
    reads only ``ts_train`` (train-span timestamps)."""
    ts = np.asarray(ts_train, dtype=np.int64)
    T = int(ts.size)
    if T == 0:
        return 0
    dt = pd.to_datetime(ts, unit="s")
    months = dt.to_period("M")
    last_of_month = pd.Series(np.arange(T)).groupby(months.values).max().to_numpy()
    final = dt[-1]
    final_is_true_month_end = bool(final.day == final.days_in_month)
    confirmed = [int(i) for i in last_of_month if (i < T - 1) or final_is_true_month_end]
    if not confirmed:
        return T - 1                      # single partial month → old [-1] behaviour (safe fallback)
    return max(confirmed)


def _cohort_holdout_guard(
    members: tuple[str, ...],
    base_train: Mapping[str, np.ndarray],
    pool_train: Mapping[str, np.ndarray],
    ts_train: np.ndarray,
    base_ho: Mapping[str, np.ndarray],
    pool_ho: Mapping[str, np.ndarray],
    ccfg: CohortConfig,
    fcfg: FitnessConfig,
) -> tuple[float, bool]:
    """Doc 2 §4 guard: freeze the observed cohort's members AND combiner weights from the TRAIN
    pipeline, apply them (no re-estimation) to the embargoed holdout tail, and require the
    **annualized** ΔSR there ``≥ min_book_uplift``. Two frozen weight vectors — combine(base∪members)
    (cohort book, redundancy ON) and combine(base) (base book) — both frozen at the LAST CONFIRMED
    month-end within ``ts_train`` (:func:`_last_confirmed_month_end_idx`; the in-progress final bar is
    a partial-month α on a mid-month-truncated span — the P2-01 truncation caveat in
    ``allocator_factory.monthly_rebal_conviction`` — so freezing ``[-1]`` would book the holdout under
    a partial-month weight; byte-identical to ``[-1]`` when the span ends on a true month-end).
    Returns ``(annualized_delta_sr, passes)``; ``(nan, False)`` if either book Sharpe is undefined."""
    member_train = {m: np.asarray(pool_train[m], dtype=np.float64) for m in members}
    member_ho = {m: np.asarray(pool_ho[m], dtype=np.float64) for m in members}
    aug_cfg = _with_redundancy(fcfg, ccfg.combiner_redundancy_strength)   # cohort book: redundancy ON

    a_aug = _alpha_paths({**dict(base_train), **member_train}, ts_train, aug_cfg)
    a_base = _alpha_paths(dict(base_train), ts_train, fcfg)
    me = _last_confirmed_month_end_idx(ts_train)   # last CONFIRMED month-end (P2-01: NOT the tail)
    frozen_aug = {s: float(np.asarray(a_aug[s], dtype=np.float64)[me]) for s in a_aug}
    frozen_base = {s: float(np.asarray(a_base[s], dtype=np.float64)[me]) for s in a_base}

    book_aug = _frozen_weight_book({**dict(base_ho), **member_ho}, frozen_aug)
    book_base = _frozen_weight_book(dict(base_ho), frozen_base)
    # ANNUALIZED to match min_book_uplift's units (== funnel min_combination_uplift, _cpcv_delta_sr).
    sr_aug = _ann_sharpe(book_aug, fcfg.periods_per_year)
    sr_base = _ann_sharpe(book_base, fcfg.periods_per_year)
    if not (np.isfinite(sr_aug) and np.isfinite(sr_base)):
        return float("nan"), False
    delta = float(sr_aug - sr_base)
    return delta, bool(delta >= ccfg.min_book_uplift)


# --------------------------------------------------------------------------------------------
# The single orchestration entry point
# --------------------------------------------------------------------------------------------
def _verdict(ev: CohortEvidence, mc: McNullResult | None, holdout: tuple[float, bool],
             pool_hash: str, n_culled: int, verdict: str) -> CohortVerdict:
    return CohortVerdict(
        members=(mc.members_obs if mc is not None else ev.members),
        n_members=(len(mc.members_obs) if mc is not None else ev.n_members),
        n_candidates_seen=ev.n_candidates_seen, n_culled=n_culled,
        mean_pairwise_corr=ev.mean_pairwise_corr, sr_star_cohort=ev.sr_star_cohort,
        dsr_cohort_book=ev.dsr_cohort_book, passes_analytic_floor=ev.passes_analytic_floor,
        mc_p_value=(mc.p_value if mc is not None else float("nan")),
        mc_t_obs=(mc.t_obs if mc is not None else float("nan")),
        mc_n_reps=(mc.n_reps if mc is not None else 0),
        mc_n_valid_reps=(mc.n_valid_reps if mc is not None else 0),
        mc_block_length=(mc.block_length if mc is not None else 0),
        passes_mc=(mc.passes_mc if mc is not None else False),
        holdout_delta_sr=holdout[0], holdout_passes=holdout[1],
        pool_content_hash=pool_hash, verdict=verdict)


def evaluate_cohort(
    panel: Panel,
    base_returns: Mapping[str, np.ndarray],
    timestamps: np.ndarray,
    overlay_formulas: Mapping[str, str],
    ccfg: CohortConfig,
    fcfg: FitnessConfig,
    *,
    mc_kwargs: Mapping,
    cost_bps: float,
    holdout_frac: float,
    holdout_embargo: int,
    seed: int,
    base_components: "Mapping[str, SleeveComponents] | None" = None,
) -> CohortVerdict | None:
    """Run the full cohort gate. Returns ``None`` when no cohort can form (pool < ``min_cohort_size``
    de-correlated admits, or a degenerate split). Otherwise a :class:`CohortVerdict`; PROMISING iff
    analytic-floor PASS ∧ MC ``p ≤ alpha_cohort`` ∧ holdout ΔSR ``≥ min_book_uplift``.

    Evaluation order (Doc 2 §5) short-circuits so the expensive MC never fires on a doomed cohort:
    analytic pre-filter → (pass) MC null → (pass) holdout guard. ``overlay_formulas`` maps candidate
    id → DSL formula (the tick's pre-registered OVERLAY specs). ``mc_kwargs`` = ``{n_reps,
    alpha_cohort, block_length}``. ``seed`` is the caller's :func:`derive_cohort_seed` value."""
    ts_full = np.asarray(timestamps)
    base_full = {str(k): np.asarray(v, dtype=np.float64) for k, v in base_returns.items()}
    # F14: with base_components, charge the overlay tilt against the base book's TRUE gross / embedded
    # cost; without, the unit-gross fallback (byte-identical to the pre-fix cohort path).
    bg_full: np.ndarray | None
    bc_full: np.ndarray | None
    ge_full: np.ndarray | None
    if base_components is not None:
        base_book_full, bg_full, bc_full, ge_full = _combined_book_with_components(
            base_full, base_components, ts_full, fcfg)
    else:
        base_book_full = _combined_book(base_full, ts_full, fcfg)
        bg_full = bc_full = ge_full = None
    pool_full, culled = assemble_overlay_pool(
        panel, base_book_full, overlay_formulas, cost_bps=cost_bps,
        base_gross=bg_full, base_cost=bc_full, gross_exposure=ge_full)
    n_culled = len(culled)

    T = int(panel.T)
    cut = int(T * (1.0 - float(holdout_frac)))
    n_train = max(1, cut - int(holdout_embargo))
    if n_train < 2 or cut >= T:
        logger.info("cohort: degenerate split (T=%d n_train=%d cut=%d) — no verdict", T, n_train, cut)
        return None

    pool_train = {k: v[:n_train] for k, v in pool_full.items()}
    base_train = {k: v[:n_train] for k, v in base_full.items()}
    ts_train = ts_full[:n_train]
    pool_ho = {k: v[cut:] for k, v in pool_full.items()}
    base_ho = {k: v[cut:] for k, v in base_full.items()}

    # 1. analytic SR*_cohort pre-filter on the train/selection span (Doc 1) — cheap reject.
    ev = evaluate_cohort_analytic(pool_train, base_train, ts_train, ccfg, fcfg)
    if ev is None:
        logger.info("cohort: pool cannot form >= min_cohort_size on train span — no verdict")
        return None
    pch = pool_content_hash(overlay_formulas)
    if not ev.passes_analytic_floor:
        logger.info("cohort: analytic floor NOT cleared (dsr=%.3f) — LOGGED, MC skipped",
                    ev.dsr_cohort_book)
        return _verdict(ev, None, (float("nan"), False), pch, n_culled, "LOGGED")

    # 2. selection-aware MC null on survivors (Doc 2 §3) — the BINDING statistical gate.
    bl = mc_kwargs.get("block_length")
    mc = mc_null_pvalue(
        pool_train, base_train, ts_train, ccfg, fcfg,
        n_reps=int(mc_kwargs["n_reps"]), alpha_cohort=float(mc_kwargs["alpha_cohort"]),
        seed=int(seed), block_length=(int(bl) if bl else None))
    if not mc.passes_mc:
        logger.info("cohort: MC null NOT cleared (p=%.4f > alpha) — LOGGED, holdout skipped", mc.p_value)
        return _verdict(ev, mc, (float("nan"), False), pch, n_culled, "LOGGED")

    # 3. embargoed holdout guard (Doc 2 §4) — persistence, orthogonal to selection.
    hd, hp = _cohort_holdout_guard(
        mc.members_obs, base_train, pool_train, ts_train, base_ho, pool_ho, ccfg, fcfg)
    verdict = "PROMISING" if hp else "LOGGED"
    logger.info("cohort: MC PASS (p=%.4f) + holdout ΔSR=%.3f (floor %.3f) -> %s",
                mc.p_value, hd, ccfg.min_book_uplift, verdict)
    return _verdict(ev, mc, (hd, hp), pch, n_culled, verdict)
