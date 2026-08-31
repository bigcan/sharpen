"""Cohort evaluator ORCHESTRATION (Phase 4 integration) — the single library entry point that runs
the full weak-signal cohort gate end-to-end: assemble the candidate pool → analytic
``SR*_cohort`` pre-filter (Doc 1) → selection-aware MC null (Doc 2 §3, binding) → embargoed holdout
guard (Doc 2 §4). It owns NO new statistic — it wires the Math-verified pieces in
:mod:`cohort` / :mod:`cohort_mc` in the pre-registered order (Doc 2 §5) and returns one
:class:`CohortVerdict` artifact.

MIXED-TYPE POOLS (crucible-v12.1). The pool was OVERLAY-only through v12.0 (ADR-3 deferred
cross-sectional admission to "a future extension, only if measured to help"). It no longer is:
:func:`assemble_candidate_pool` dispatches per candidate on its ``candidate_type``, so a cohort can
be built from cross-sectional rank-L/S sleeves, overlays, or both. Everything downstream is already
type-agnostic — admission, the MC null and the holdout guard consume ``(T,)`` return streams and
never look at how a stream was produced — so this is a pool-assembly change, not a statistics change.

WHY it matters (2026-08-09 root cause, ``docs/research/crucible_zero_alpha_root_cause_2026-08-09.md``):
``IR_cohort = δ·√K·hit_rate``, and the two halves of the machine were mispaired. Cross-sectional
candidates carry the panel's breadth (``n_eff`` 42.1 on ``us_equity``) and so have the larger
per-member δ, but they were routed to the per-candidate gate, which has ~0% power at any plausible
alpha. Overlays — one scalar per day, δ small by construction, and structurally correlated with the
base book they tilt — were routed to the cohort MC null, the ONE gate with measured power (25% @
IR 0.30 / 50% @ IR 0.50 at a 15% hit rate). Admitting cross-sectional candidates here pairs the
high-δ hypotheses with the powered gate for the first time. Note what does NOT change: the cohort
statistic is still ``T`` observations of a book ΔSR — a candidate's ``T×N`` panel buys a cleaner
per-day stream (larger δ), not more rows for the null.

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

from sharpen.envs.allocator_factory import dynamic_sleeve_alphas

from ..eval_harness import _ann_sharpe
from ..features import Panel
from .cohort import (
    CohortConfig,
    CohortEvidence,
    _with_redundancy,
    evaluate_cohort_analytic,
)
from .cohort_mc import McNullResult, mc_null_pvalue
from .evolve import _candidate_returns, _overlay_returns
from .fitness import FitnessConfig, _combined_book, _combined_book_with_components

if TYPE_CHECKING:
    from .base_sleeves import SleeveComponents

logger = logging.getLogger(__name__)

# The two candidate types the funnel mines (``evolve.evolve`` rejects anything else). ``_OVERLAY`` is
# the pool's back-compat default so a caller that passes no type map gets the pre-v12.1 behaviour.
_OVERLAY = "overlay"
_CROSS_SECTIONAL = "cross_sectional"


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
    # v12.1 pool composition — WHAT the gate adjudicated, travelling with the verdict rather than
    # living in a log line (the `n_holdout_tested` lesson: a null whose denominator is unreadable
    # costs sessions). `n_pool_cross_sectional` counts the assembled pool; `n_members_cross_sectional`
    # counts how many SURVIVED de-correlated admission. 0/0 == an overlay-only cohort.
    n_pool_cross_sectional: int = 0
    n_members_cross_sectional: int = 0


# --------------------------------------------------------------------------------------------
# Determinism helpers (Doc 2 §6) — the caller derives the MC seed from these
# --------------------------------------------------------------------------------------------
def pool_content_hash(formulas: Mapping[str, str],
                      candidate_types: "Mapping[str, str] | None" = None) -> str:
    """12-hex SHA-256 over the sorted ``(name, formula[, candidate_type])`` triples of the pool
    (mirrors ``spec.content_hash``). Deterministic ⇒ the derived MC seed is reproducible; a changed
    pool ⇒ a changed hash ⇒ a different (visible) seed.

    The type is folded into a member's payload line ONLY when it is not ``"overlay"`` (v12.1). An
    all-overlay pool therefore hashes byte-identically to every pre-v12.1 pool, so the two cohort
    cards already on disk keep their ``pool_content_hash`` and their MC seed — the mixed-pool
    capability cannot silently re-seed a recorded verdict (CRU-1)."""
    ct = dict(candidate_types or {})
    items = sorted((str(k), str(v), str(ct.get(str(k), _OVERLAY))) for k, v in formulas.items())
    payload = "\n".join(f"{k}\t{v}" if t == _OVERLAY else f"{k}\t{v}\t{t}" for k, v, t in items)
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
    return assemble_candidate_pool(
        panel, base_book, overlay_formulas, cost_bps=cost_bps, base_gross=base_gross,
        base_cost=base_cost, gross_exposure=gross_exposure)


def assemble_candidate_pool(
    panel: Panel,
    base_book: np.ndarray,
    formulas: Mapping[str, str],
    *,
    cost_bps: float,
    candidate_types: "Mapping[str, str] | None" = None,
    hold_horizon: int = 21,
    ls_min_names: int = 6,
    fcfg: "FitnessConfig | None" = None,
    base_gross: "np.ndarray | None" = None,
    base_cost: "np.ndarray | None" = None,
    gross_exposure: "np.ndarray | None" = None,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Recompute each candidate's return stream through **its own funnel scoring path**, dispatched on
    ``candidate_types[name]`` (crucible-v12.1). Missing/None ⇒ ``"overlay"``, so an existing caller is
    byte-identical to :func:`assemble_overlay_pool`.

    * ``"overlay"`` → :func:`evolve._overlay_returns` against the combined ``base_book`` (the timing
      tilt; F14 cost decomposition via ``base_gross``/``base_cost``/``gross_exposure``).
    * ``"cross_sectional"`` → :func:`evolve._candidate_returns`: the dollar-neutral rank-L/S sleeve
      return, rebalanced every ``hold_horizon`` bars, net of turnover·``cost_bps``, ``ls_min_names``
      minimum names. This is the SAME call ``evolve``'s ``_returns_for`` makes for a cross-sectional
      genome, so the cohort admits exactly the stream the per-candidate gate scored — the two gates
      differ in how they adjudicate a stream, never in what the stream is.

    ``base_book`` is ignored for cross-sectional members (an L/S sleeve is a standalone book, not a
    tilt of the base) — which is precisely why they enter the combiner near-orthogonal to the base
    while an overlay is structurally correlated with it.

    Causality: both paths are computed ONCE on the full panel and sliced by the caller, matching
    ``evolve``'s own holdout convention (``evolve.py`` "scoring on the FULL panel so trailing-window
    operators warm up from the (causal, past) train history"). A cross-sectional stream at bar ``t``
    reads only rows ``≤ t`` (DSL ts-operators are backward-looking; ``_ls_weights`` is per-row; the
    rebalance phase is anchored at index 0, identical under truncation), so slicing a full-panel
    stream to ``[0, n_train)`` is bit-identical to computing it on the truncated panel (LEAK-2).

    FEASIBILITY (v13.0). With ``fcfg`` supplied, a member whose annualized turnover exceeds
    ``fcfg.turnover_soft_cap * 2`` is CULLED — byte-for-byte the same hard-infeasibility rule
    :func:`evolve.score` applies (``evolve.py``: "hard-infeasible (turnover/size)"). Without it the
    cohort adjudicated candidates the funnel itself refuses to trade: the 2026-08-11 ``us_equity``
    run put 92 cross-sectional members in the pool of which **91 were hard-infeasible on turnover**,
    and 9 of the 12 ADMITTED members came from that pool. Netting turnover cost into each stream is
    not a substitute — it makes an infeasible candidate merely unattractive, so a cohort could still
    be certified PROMISING out of members no one can trade. That is the v11.0 capturability defect
    exactly: a feasibility fact computed and then consulted by nothing. ``fcfg=None`` preserves the
    v12.1 behaviour for callers that have no FitnessConfig.

    Returns ``(returns, culled)``; see :func:`assemble_overlay_pool` for the pre-registered rule that
    ``culled`` is an EXCLUSION and must never inflate ``n_candidates_seen``."""
    types = dict(candidate_types or {})
    returns: dict[str, np.ndarray] = {}
    culled: list[str] = []
    max_turnover = None if fcfg is None else float(fcfg.turnover_soft_cap) * 2.0
    for name, formula in formulas.items():
        ct = str(types.get(str(name), _OVERLAY))
        if ct not in (_CROSS_SECTIONAL, _OVERLAY):
            raise ValueError(
                f"candidate_type must be 'cross_sectional' or 'overlay'; got {ct!r} for {name!r}")
        # A member that cannot be SCORED is culled, never fatal — same discipline `evolve.score`
        # applies ("eval raised: ... -> _INFEASIBLE"). This is load-bearing for the ledger-sourced
        # pool (v13.0): it replays pre-registrations from EARLIER ticks, and the alt-data bridge
        # de-duplicates its slot set per tick, so a historical overlay can reference a feature slot
        # today's panel no longer carries (measured 2026-08-11: `unknown variable:
        # cot:gold_noncomm_net` took down a whole tick AFTER the cohort had scored 100+ members).
        # An unscoreable member is missing evidence, not a reason to discard every other member's.
        try:
            if ct == _CROSS_SECTIONAL:
                out = _candidate_returns(str(formula), panel, hold_horizon=int(hold_horizon),
                                         cost_bps=cost_bps, min_names=int(ls_min_names))
            else:
                out = _overlay_returns(str(formula), panel, base_book, cost_bps=cost_bps,
                                       base_gross=base_gross, base_cost=base_cost,
                                       gross_exposure=gross_exposure)   # F14 (unit-gross if None)
        except Exception as exc:                          # noqa: BLE001 - cull, never kill the tick
            logger.info("cohort pool: %s CULLED (scoring raised %r)", name, exc)
            culled.append(str(name))
            continue
        if out is None:
            culled.append(str(name))
            continue
        if max_turnover is not None and float(out[1]) > max_turnover:
            logger.info("cohort pool: %s CULLED hard-infeasible (turnover %.1f > %.1f/yr)",
                        name, float(out[1]), max_turnover)
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
             pool_hash: str, n_culled: int, verdict: str,
             xsec_pool: "frozenset[str]" = frozenset()) -> CohortVerdict:
    members = (mc.members_obs if mc is not None else ev.members)
    return CohortVerdict(
        members=members,
        n_members=(len(mc.members_obs) if mc is not None else ev.n_members),
        n_pool_cross_sectional=len(xsec_pool),
        n_members_cross_sectional=sum(1 for m in members if m in xsec_pool),
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
    formulas: Mapping[str, str],
    ccfg: CohortConfig,
    fcfg: FitnessConfig,
    *,
    mc_kwargs: Mapping,
    cost_bps: float,
    holdout_frac: float,
    holdout_embargo: int,
    seed: int,
    base_components: "Mapping[str, SleeveComponents] | None" = None,
    candidate_types: "Mapping[str, str] | None" = None,
    hold_horizon: int = 21,
    ls_min_names: int = 6,
) -> CohortVerdict | None:
    """Run the full cohort gate. Returns ``None`` when no cohort can form (pool < ``min_cohort_size``
    de-correlated admits, or a degenerate split). Otherwise a :class:`CohortVerdict`; PROMISING iff
    analytic-floor PASS ∧ MC ``p ≤ alpha_cohort`` ∧ holdout ΔSR ``≥ min_book_uplift``.

    Evaluation order (Doc 2 §5) short-circuits so the expensive MC never fires on a doomed cohort:
    analytic pre-filter → (pass) MC null → (pass) holdout guard. ``formulas`` maps candidate
    id → DSL formula (named ``overlay_formulas`` pre-v12.1, when the pool could only be overlays);
    ``candidate_types`` maps the same ids to ``"overlay"`` (default, and the whole
    pool pre-v12.1) or ``"cross_sectional"``, which selects that member's scoring path in
    :func:`assemble_candidate_pool`. ``hold_horizon`` / ``ls_min_names`` are the cross-sectional
    sleeve's rebalance period and minimum-names floor (the caller passes its ``evolve_kwargs`` values
    so the cohort's streams match the mine's). ``mc_kwargs`` = ``{n_reps, alpha_cohort,
    block_length}``. ``seed`` is the caller's :func:`derive_cohort_seed` value."""
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
    pool_full, culled = assemble_candidate_pool(
        panel, base_book_full, formulas, cost_bps=cost_bps,
        candidate_types=candidate_types, hold_horizon=hold_horizon, ls_min_names=ls_min_names,
        fcfg=(fcfg if ccfg.enforce_funnel_feasibility else None),
        base_gross=bg_full, base_cost=bc_full, gross_exposure=ge_full)
    n_culled = len(culled)
    # Composition of the SURVIVING pool (culled members are not in it and must not be counted).
    xsec_pool = frozenset(k for k in pool_full
                          if str((candidate_types or {}).get(k, _OVERLAY)) == _CROSS_SECTIONAL)

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
    pch = pool_content_hash(formulas, candidate_types)
    if not ev.passes_analytic_floor:
        if not ccfg.analytic_floor_advisory:
            logger.info("cohort: analytic floor NOT cleared (dsr=%.3f) — LOGGED, MC skipped",
                        ev.dsr_cohort_book)
            return _verdict(ev, None, (float("nan"), False), pch, n_culled, "LOGGED", xsec_pool)
        # ADVISORY (default): record the miss and continue to the BINDING MC null. The floor reuses
        # promising_dsr / cohort_hlz_t_min, which pass 0/170 across the lifetime record, so gating on
        # it made the only high-power gate in the system unreachable — the third instance of "a cheap
        # pre-filter stricter than the gate it protects" (2026-08-09 root-cause report §5b Finding 5).
        # Nothing is loosened: the MC null and the embargoed holdout below are unchanged.
        logger.info("cohort: analytic floor NOT cleared (dsr=%.3f) — ADVISORY, continuing to MC null",
                    ev.dsr_cohort_book)

    # 2. selection-aware MC null on survivors (Doc 2 §3) — the BINDING statistical gate.
    bl = mc_kwargs.get("block_length")
    mc = mc_null_pvalue(
        pool_train, base_train, ts_train, ccfg, fcfg,
        n_reps=int(mc_kwargs["n_reps"]), alpha_cohort=float(mc_kwargs["alpha_cohort"]),
        seed=int(seed), block_length=(int(bl) if bl else None))
    if not mc.passes_mc:
        logger.info("cohort: MC null NOT cleared (p=%.4f > alpha) — LOGGED, holdout skipped", mc.p_value)
        return _verdict(ev, mc, (float("nan"), False), pch, n_culled, "LOGGED", xsec_pool)

    # 3. embargoed holdout guard (Doc 2 §4) — persistence, orthogonal to selection.
    hd, hp = _cohort_holdout_guard(
        mc.members_obs, base_train, pool_train, ts_train, base_ho, pool_ho, ccfg, fcfg)
    verdict = "PROMISING" if hp else "LOGGED"
    logger.info("cohort: MC PASS (p=%.4f) + holdout ΔSR=%.3f (floor %.3f) -> %s",
                mc.p_value, hd, ccfg.min_book_uplift, verdict)
    return _verdict(ev, mc, (hd, hp), pch, n_culled, verdict, xsec_pool)
