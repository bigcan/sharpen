"""The generation loop (C3.4) — evolve DSL alphas against the combined-book fitness.

Warm-starts from seed formulas, evaluates each genome's marginal contribution to the C1
combined book (C3.3 fitness) on the FERTILE cross-asset panel, and evolves via type-safe
mutation/crossover (C3.1). The honest-by-construction controls (ADR-C3-3):

  * **file-drawer N** — every genome ever evaluated (incl. culled/infeasible) counts toward
    ``gen_n_total`` → ``gen_n_eff`` fed to the deflation (the search IS the multiple comparison);
  * **held-out tail** — the embargoed last ``holdout_frac`` of the timeline is never seen during
    evolution; PROMISING survivors are re-scored there before they are reported.

Deterministic given ``rng_seed``. Advisory only — the result tops out at PROMISING and a deploy
read of any survivor requires a Tier-2 deep lifecycle audit.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np

from ..eval_harness import _ls_weights
from ..features import Panel
from .dsl_signal import eval_on_panel
from .fitness import FitnessConfig, FitnessResult, _combined_book, combination_fitness
from .grammar import INPUTS, available_terminals, crossover, mutate, node_count, parse, to_formula

log = logging.getLogger("alpha_evolve")
_INFEASIBLE = float("-inf")


@dataclass(frozen=True, slots=True)
class Candidate:
    formula: str
    fitness: float
    result: FitnessResult | None     # None when culled before fitness (leak / hard-infeasible)
    reason: str = ""                 # cull reason, if any


@dataclass(frozen=True, slots=True)
class GenerationReport:
    hall_of_fame: list[Candidate]
    gen_n_total: int                 # file-drawer N: every DISTINCT genome scored (dedup'd), incl. culled
    gen_n_eff: float                 # effective trial count fed to the deflation (== gen_n_total)
    holdout_validation: list[dict]   # PROMISING survivors re-scored on the embargoed tail
    promising: list[Candidate] = field(default_factory=list)
    pbo: dict | None = None          # CSCV Probability of Backtest Overfitting (advisory, GP7-03)


def _candidate_returns(formula: str, panel: Panel, *, hold_horizon: int, cost_bps: float,
                       min_names: int) -> tuple[np.ndarray, float] | None:
    """Daily-marked dollar-neutral rank-L/S sleeve return for ``formula`` on ``panel``, NET of
    turnover·bps, plus annualized turnover. Weights are rebalanced every ``hold_horizon`` days
    and HELD (marked daily with the 1-day forward return) — the live-book convention, low
    turnover. Returns None if the score is everywhere NaN (degenerate genome)."""
    scores = eval_on_panel(formula, panel)
    if not np.isfinite(scores).any():
        return None
    fwd1 = panel.forward_returns(1)
    T = panel.T
    rets = np.full(T, np.nan)
    turns: list[float] = []
    w = np.zeros(panel.N)
    for t in range(T - 1):
        if t % hold_horizon == 0:                       # rebalance day → new target weights
            w_new = _ls_weights(scores[t], panel.active[t], min_names=min_names)
            turns.append(float(np.abs(w_new - w).sum()))
            w = w_new
        rets[t] = float(np.nansum(w * fwd1[t]) - cost_bps * (turns[-1] if t % hold_horizon == 0
                                                             and turns else 0.0))
    ppy = 252.0 / hold_horizon
    turnover_ann = float(np.mean(turns) * ppy) if turns else 0.0
    return rets, turnover_ann


_OVERLAY_MIN_OBS = 20            # expanding-window warmup before the overlay tilt is trusted (neutral before)


def _overlay_returns(formula: str, panel: Panel, base_book: np.ndarray, *,
                     cost_bps: float) -> tuple[np.ndarray, float] | None:
    """CR-9 OVERLAY candidate returns: use ``formula`` (a timing signal, typically referencing a
    non-OHLCV feature slot) as a TIME-varying multiplier on the existing combined ``base_book``.

    A cross-sectional rank-L/S book on a broadcast (constant-across-N) series is identically zero
    (``_ls_weights`` centered-rank of a constant row sums to zero). The overlay instead:

      1. ``scores = eval_on_panel(formula, panel)`` → (T,N);
      2. collapse the cross-section to a per-day timing scalar ``g[t] = nanmean_n(scores[t])``
         (for a broadcast series this is just the series value); all-NaN → None (cull);
      3. standardize ``g`` with a CAUSAL expanding-window z-score — at bar ``t`` the mean/std use
         ONLY ``g[:t+1]`` (LEAK-1 safe: no train/holdout-boundary leak; passes the Tier-0
         truncation-equivalence tripwire that a whole-sample z-score fails). Neutral (``m=0``)
         until ``_OVERLAY_MIN_OBS`` finite observations have accrued;
      4. bounded tilt ``m[t] = tanh(z_g[t]) ∈ [-1,1]``;
      5. overlay marginal return ``cand[t] = m[t-1] * base_book[t]`` — LAGGED one bar (strict
         causality, LEAK-2), NET of turnover cost ``cost_bps * |Δm|``.

    Returns ``(cand, turnover_ann)`` or None if the timing series is degenerate (all-NaN/constant).
    """
    scores = eval_on_panel(formula, panel)
    if not np.isfinite(scores).any():
        return None
    with warnings.catch_warnings():                          # all-NaN row → NaN g[t] (handled below)
        warnings.simplefilter("ignore", RuntimeWarning)
        g = np.nanmean(scores, axis=1)                       # (T,) per-day timing scalar
    finite = np.isfinite(g)
    if not finite.any():
        return None
    gf = g[finite]
    if not np.isfinite(gf.std()) or gf.std() <= 0.0:         # globally constant → cull (no signal)
        return None
    # Causal expanding-window z-score: bar t uses ONLY g[:t+1]. cumsum over the identical prefix is
    # bit-reproducible, so compute(truncated(t))[t] == compute(panel)[t] (Tier-0 tripwire holds) and
    # the holdout tilt is never standardized with holdout-inclusive stats (LEAK-1).
    g0 = np.where(finite, g, 0.0)
    cnt = np.cumsum(finite.astype(np.float64))               # finite-obs count through t
    safe_cnt = np.maximum(cnt, 1.0)
    cmean = np.cumsum(g0) / safe_cnt
    cvar = np.maximum(np.cumsum(g0 * g0) / safe_cnt - cmean * cmean, 0.0)
    cstd = np.sqrt(cvar)
    ready = finite & (cnt >= _OVERLAY_MIN_OBS) & (cstd > 0.0)   # neutral until warmed up
    z = np.where(ready, (g - cmean) / np.where(cstd > 0.0, cstd, 1.0), 0.0)
    m = np.tanh(z)                                           # bounded tilt in [-1, 1]
    bb = np.asarray(base_book, dtype=np.float64)
    T = bb.shape[0]
    m_lag = np.empty(T, dtype=np.float64)                    # m[t-1], strictly causal
    m_lag[0] = 0.0
    m_lag[1:] = m[:T - 1]
    dm = np.abs(np.diff(m_lag, prepend=0.0))                 # per-bar turnover of the tilt
    cand = m_lag * bb - cost_bps * dm                        # net of turnover cost
    turnover_ann = float(np.nanmean(dm) * 252.0)             # per-bar tilt turnover → annualized (daily)
    if not np.isfinite(cand).any():
        return None
    return cand, turnover_ann


def _split(panel: Panel, holdout_frac: float, embargo: int) -> tuple[Panel, Panel]:
    """(train, holdout) split by row; the train end is embargoed away from the holdout start."""
    cut = int(panel.T * (1.0 - holdout_frac))
    train = panel.truncated(max(1, cut - embargo) - 1)
    # holdout keeps rows [cut, T): rebuild a Panel slice (truncated only trims the tail).
    # feature_slots MUST be sliced on axis 0 too (CR-9) or an overlay terminal would NaN-out /
    # misalign on the holdout panel; _sliced_slots handles both (T,) and (T,N) shapes.
    from dataclasses import replace
    s = slice(cut, panel.T)
    hold = replace(panel, dates=panel.dates[s], open=panel.open[s], high=panel.high[s],
                   low=panel.low[s], close=panel.close[s], volume=panel.volume[s],
                   active=panel.active[s], adv_usd=panel.adv_usd[s],
                   feature_slots=panel._sliced_slots(s))
    return train, hold


def evolve(
    seed_formulas: list[str],
    panel: Panel,
    base_returns: dict[str, np.ndarray],
    timestamps: np.ndarray,
    cfg: FitnessConfig,
    *,
    rng_seed: int = 7,
    pop_size: int = 60,
    n_generations: int = 10,
    hold_horizon: int = 21,
    cost_bps: float = 0.0010,
    ls_min_names: int = 6,
    holdout_frac: float = 0.25,
    holdout_embargo: int = 21,
    elite_frac: float = 0.3,
    pbo_max_strategies: int = 128,
    pbo_n_splits: int = 10,
    candidate_type: str = "cross_sectional",
) -> GenerationReport:
    """Evolve DSL alphas on the TRAIN split; re-validate PROMISING survivors on the held-out
    tail. ``base_returns``/``timestamps`` align to the train split's rows.

    ``candidate_type`` (CR-9): ``"cross_sectional"`` (default) is the OHLCV rank-L/S alpha path —
    byte-identical to pre-P1a. ``"overlay"`` scores each genome as a TIMING signal on the existing
    combined base book (``_overlay_returns``): a non-OHLCV broadcast series that a rank() path would
    zero out becomes a non-constant-in-time exposure multiplier, so its marginal ΔSR through
    ``combination_fitness`` is well-posed. Both types are scored by the SAME marginal-contribution
    fitness (no gate change → frozen crucible-v2.0 gates_hash untouched).

    Anti-mirage controls: the deflation N (``gen_n_eff``) is the file-drawer count of every genome
    ever scored; the DSR dispersion is the cross-search **population pool** of augmented-book
    Sharpes (M1/GP4-02) fed at the binding held-out gate; and an advisory **CSCV PBO** (GP7-03) is
    computed over the first ``pbo_max_strategies`` candidates' return series (memory-bounded sample)
    and reported — P(the in-sample-best candidate underperforms OOS), the best-of-N overfit metric
    the deflated Sharpe does not estimate. ``pbo_max_strategies=0`` disables it."""
    if candidate_type not in ("cross_sectional", "overlay"):
        raise ValueError(f"candidate_type must be 'cross_sectional' or 'overlay'; got {candidate_type!r}")
    rng = np.random.default_rng(rng_seed)
    train, hold = _split(panel, holdout_frac, holdout_embargo)
    n_train = train.T
    base_tr = {k: np.asarray(v)[:n_train] for k, v in base_returns.items()}
    ts_tr = np.asarray(timestamps)[:n_train]
    # CR-9 terminal registry: the value-leaf set the generator may draw from. The cross_sectional path
    # draws INPUTS ONLY (the OHLCV-derived leaves); ONLY the overlay path also gets the panel's feature
    # slots so its genomes can reference the non-OHLCV series. (C2-06 fix: drawing feature slots into
    # cross_sectional made broadcast terminals rank() to constant/dead genomes that inflated gen_n and
    # polluted the DSR dispersion pool — and contradicted this very "INPUTS-only" docstring. On an
    # OHLCV-only panel available_terminals(panel) == INPUTS, so this is a NO-OP there; it changes the
    # draw sequence only on a panel that carries feature slots. [crucible-v2.9 MINOR])
    is_overlay = candidate_type == "overlay"
    inputs = available_terminals(panel) if is_overlay else INPUTS
    # OVERLAY dispatch: the combined base book (C1) is the multiplier target, computed ONCE per
    # split. On the train split it is over the train rows; the holdout path rebuilds it on full rows.
    base_book_tr = _combined_book(base_tr, ts_tr, cfg) if is_overlay else None

    def _returns_for(formula: str, pnl: Panel, base_book: "np.ndarray | None"
                     ) -> tuple[np.ndarray, float] | None:
        if is_overlay:
            return _overlay_returns(formula, pnl, base_book, cost_bps=cost_bps)   # type: ignore[arg-type]
        return _candidate_returns(formula, pnl, hold_horizon=hold_horizon,
                                  cost_bps=cost_bps, min_names=ls_min_names)

    pop = [parse(f) for f in seed_formulas]
    if not pop:
        raise ValueError("seed_formulas must be non-empty (warm-start)")
    gen_n_total = 0
    scored: dict[str, Candidate] = {}        # formula -> best Candidate seen (dedup)
    # GP4-02/M1: the cross-search DISPERSION pool — every scored genome's augmented-book per-period
    # Sharpe. Fed as the DSR ``trial_sharpes`` at the BINDING held-out gate so the deflation
    # benchmark reflects the search's own spread, not one book's within-CPCV paths.
    sharpe_pool: list[float] = []
    # GP7-03: memory-bounded sample of candidate return series for the CSCV PBO diagnostic.
    cand_return_bank: list[np.ndarray] = []

    def score(formula: str, gen_n_eff: float) -> Candidate:
        nonlocal gen_n_total
        # file-drawer N: counts every DISTINCT genome (score() is called once per formula via the
        # `f not in scored` dedup in the loop) — a re-derived duplicate is the SAME hypothesis, not a
        # new trial, so distinct is the correct multiplicity count (GP5-01: doc/code reconciled).
        gen_n_total += 1
        try:
            cr = _returns_for(formula, train, base_book_tr)
        except Exception as exc:                          # noqa: BLE001 - cull, don't crash a run
            return Candidate(formula, _INFEASIBLE, None, f"eval raised: {exc!r}")
        if cr is None:
            return Candidate(formula, _INFEASIBLE, None, "degenerate score (all-NaN)")
        cand_ret, turnover_ann = cr
        if 0 < pbo_max_strategies and len(cand_return_bank) < pbo_max_strategies:
            cand_return_bank.append(np.asarray(cand_ret, dtype=np.float64))   # PBO sample (GP7-03)
        n_nodes = node_count(parse(formula))
        if turnover_ann > cfg.turnover_soft_cap * 2.0 or n_nodes > cfg.max_ast_nodes:
            return Candidate(formula, _INFEASIBLE, None, "hard-infeasible (turnover/size)")
        try:                                              # F3: fitness must not crash the run
            res = combination_fitness(cand_ret, base_tr, ts_tr, cfg, gen_n_eff=gen_n_eff,
                                      turnover_ann=turnover_ann, n_nodes=n_nodes)
        except Exception as exc:                          # noqa: BLE001 - cull, don't crash a run
            log.warning("genome culled — combination_fitness raised %r (formula=%.90s)", exc, formula)
            return Candidate(formula, _INFEASIBLE, None, f"fitness raised: {exc!r}")
        if np.isfinite(res.aug_book_sharpe_pp):
            sharpe_pool.append(float(res.aug_book_sharpe_pp))
        return Candidate(formula, res.fitness, res)

    for _gen in range(n_generations):
        gen_n_eff = float(max(2, len(scored) + len(pop)))    # running file-drawer N
        for node in pop:
            f = to_formula(node)
            if f not in scored:
                scored[f] = score(f, gen_n_eff)
        ranked = sorted(scored.values(), key=lambda c: c.fitness, reverse=True)
        n_elite = max(2, int(pop_size * elite_frac))
        elites = [parse(c.formula) for c in ranked[:n_elite] if np.isfinite(c.fitness)]
        if not elites:
            elites = pop[:n_elite]
        nxt = list(elites)
        while len(nxt) < pop_size:
            a = elites[rng.integers(len(elites))]
            if rng.random() < 0.5 and len(elites) > 1:
                b = elites[rng.integers(len(elites))]
                nxt.append(crossover(a, b, rng, max_nodes=cfg.max_ast_nodes, inputs=inputs))
            else:
                nxt.append(mutate(a, rng, max_nodes=cfg.max_ast_nodes, inputs=inputs))
        pop = nxt

    final_n_eff = float(max(2, gen_n_total))             # the FULL file-drawer N (every genome)
    ranked = sorted(scored.values(), key=lambda c: c.fitness, reverse=True)
    # cheap pre-filter on the (under-counted) running-N train gate; the BINDING decision is the
    # held-out gate at the full file-drawer N below (F1: the train gate alone under-deflates).
    train_passers = [c for c in ranked if c.result is not None and c.result.passes_gate]

    # Held-out re-validation at the FULL file-drawer N, scoring on the FULL panel so trailing-
    # window operators warm up from the (causal, past) train history (F2: a fresh hold slice
    # would NaN out every long-window alpha). Only candidates that clear THIS gate are PROMISING.
    n_hold = hold.T
    base_ho = {k: np.asarray(v)[panel.T - n_hold:] for k, v in base_returns.items()}
    ts_ho = np.asarray(timestamps)[panel.T - n_hold:]
    # OVERLAY (CR-9): the base book on the FULL timeline is the multiplier target for the full-panel
    # re-score; the holdout rows are sliced off it below, matching the cross_sectional warm-up path.
    base_full = ({k: np.asarray(v) for k, v in base_returns.items()})
    base_book_full = _combined_book(base_full, np.asarray(timestamps), cfg) if is_overlay else None
    holdout_validation: list[dict] = []
    promising: list[Candidate] = []
    for c in train_passers:
        try:
            full = _returns_for(c.formula, panel, base_book_full)
        except Exception:                                 # noqa: BLE001
            full = None
        if full is None:
            holdout_validation.append({"formula": c.formula, "holdout": "degenerate"})
            continue
        cand_ho = full[0][panel.T - n_hold:]              # holdout rows, warmed up from train
        # BINDING gate: deflate the held-out book Sharpe against the FULL search dispersion pool
        # (GP4-02/M1) at the full file-drawer count.
        try:                                              # F3: fitness must not crash the run
            hv = combination_fitness(cand_ho, base_ho, ts_ho, cfg, gen_n_eff=final_n_eff,
                                     turnover_ann=full[1], n_nodes=node_count(parse(c.formula)),
                                     trial_sharpe_pool=sharpe_pool)
        except Exception:                                 # noqa: BLE001
            holdout_validation.append({"formula": c.formula, "holdout": "fitness raised"})
            continue
        holdout_validation.append({
            "formula": c.formula, "train_delta": c.result.delta_sr_oos,   # type: ignore[union-attr]
            "holdout_delta": hv.delta_sr_oos, "holdout_passes": hv.passes_gate})
        if hv.passes_gate:
            promising.append(c)

    # GP7-03: advisory CSCV PBO over the bounded candidate sample (the best-of-N overfit metric
    # the deflated Sharpe doesn't estimate). Advisory only — reported, not a hard gate.
    pbo = None
    if pbo_max_strategies and len(cand_return_bank) >= 2:
        from finrl_pro_ds.crypto.eval.statistics import probability_of_backtest_overfitting
        try:
            pbo = probability_of_backtest_overfitting(
                np.column_stack(cand_return_bank), n_splits=pbo_n_splits)
        except Exception as exc:                          # noqa: BLE001 - advisory, never fatal
            log.warning("PBO computation skipped: %r", exc)

    return GenerationReport(
        hall_of_fame=ranked[:10],
        gen_n_total=gen_n_total, gen_n_eff=final_n_eff,
        holdout_validation=holdout_validation, promising=promising, pbo=pbo)
