"""Phase 4 Step 1 — the cohort-evaluator ORCHESTRATION library (`cohort_eval.py`).

Scope of THIS file: the wiring correctness (pool assembly parity, split/short-circuit order,
determinism, the Doc 2 §4 holdout-guard math). The STATISTICAL size/power guarantees are not
duplicated here — they split across two files (Doc 2 §7): the MC-null calibration ≈ α, the
BLOCKER-1/3 regressions, and the MC-path production-config noise rejection live in
``test_generation_cohort_mc.py``; the ANALYTIC ``SR*_cohort`` m=9/N=100 noise tripwire lives in
``test_generation_cohort.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finrl_pro_ds.signals.features import Panel
from finrl_pro_ds.signals.generation.cohort import CohortConfig
from finrl_pro_ds.signals.generation.cohort_eval import (
    CohortVerdict,
    _cohort_holdout_guard,
    _frozen_weight_book,
    assemble_overlay_pool,
    derive_cohort_seed,
    evaluate_cohort,
    pool_content_hash,
)
from finrl_pro_ds.signals.generation.evolve import _overlay_returns
from finrl_pro_ds.signals.generation.fitness import FitnessConfig, _combined_book

T, N = 600, 12
_CFG = FitnessConfig()          # defaults — the funnel's real combiner/CPCV knobs
_CCFG = CohortConfig(
    max_cohort_size=12, min_cohort_size=3, max_pairwise_corr=0.35,
    promising_dsr=0.90, cohort_hlz_t_min=3.0, min_book_uplift=0.10,
    combiner_redundancy_strength=0.5)
_MC_KWARGS = {"n_reps": 120, "alpha_cohort": 0.05, "block_length": 21}


def _noise_slots(n_slots: int, seed: int) -> dict[str, np.ndarray]:
    """`n_slots` mutually-INDEPENDENT (T,) macro-style feature series (random-walk levels), so their
    overlay tilts are genuinely de-correlated — the pool the diversity run showed alt-data breadth
    supplies."""
    rng = np.random.default_rng(seed)
    return {f"fred:X{i:02d}": np.cumsum(0.05 * rng.standard_normal(T)).astype(np.float64)
            for i in range(n_slots)}


def _panel(slots: dict[str, np.ndarray], seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    close = np.exp(np.cumsum(0.01 * rng.standard_normal((T, N)), axis=0) + 4.0)
    open_ = close * (1 + 0.001 * rng.standard_normal((T, N)))
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    vol = rng.uniform(1e6, 1e8, (T, N))
    dates = (np.datetime64("2012-01-02") + np.arange(T) * np.timedelta64(1, "D")
             ).astype("datetime64[ns]")
    return Panel(dates, tuple(f"E{i:02d}" for i in range(N)), open_, high, low, close, vol,
                 np.ones((T, N), bool), close * vol, rng.integers(0, 4, size=N),
                 {"survivorship_free": True, "source": "synthetic"}, feature_slots=slots)


def _base_and_ts(seed: int = 1) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    base = {"tsmom": (0.0004 + 0.008 * rng.standard_normal(T)).astype(np.float64),
            "rates_carry": (0.0003 + 0.007 * rng.standard_normal(T)).astype(np.float64)}
    idx = pd.date_range("2012-01-02", periods=T, freq="B")
    return base, idx.view("int64").astype(np.float64) / 1e9


def _overlays(slots: dict[str, np.ndarray]) -> dict[str, str]:
    """One `level` overlay per slot (formula == the terminal name)."""
    return {f"ov-{t.replace(':', '-')}": t for t in slots}


# --------------------------------------------------------------- determinism helpers ----

def test_pool_content_hash_stable_and_sensitive() -> None:
    a = {"ov-x": "fred:X00", "ov-y": "fred:X01"}
    assert pool_content_hash(a) == pool_content_hash(dict(reversed(list(a.items()))))  # order-free
    assert pool_content_hash(a) != pool_content_hash({"ov-x": "fred:X00", "ov-y": "fred:X02"})


def test_derive_cohort_seed_deterministic_and_sensitive() -> None:
    s1 = derive_cohort_seed("funnelaaa", "cohortbbb", "poolccc", "run-1")
    assert s1 == derive_cohort_seed("funnelaaa", "cohortbbb", "poolccc", "run-1")
    assert s1 != derive_cohort_seed("funnelaaa", "cohortbbb", "poolccc", "run-2")   # run_id folds in
    assert s1 != derive_cohort_seed("funnelaaa", "cohortZZZ", "poolccc", "run-1")   # cohort gate folds in


# --------------------------------------------------------------- pool assembly (DRY) ----

def test_assemble_overlay_pool_matches_direct_overlay_returns() -> None:
    slots = _noise_slots(4, seed=7)
    panel = _panel(slots)
    base, ts = _base_and_ts()
    base_book = _combined_book(base, ts, _CFG)
    overlays = _overlays(slots)

    returns, culled = assemble_overlay_pool(panel, base_book, overlays, cost_bps=0.0010)
    assert culled == [] and set(returns) == set(overlays)
    for name, formula in overlays.items():
        direct = _overlay_returns(formula, panel, base_book, cost_bps=0.0010)
        assert direct is not None
        np.testing.assert_array_equal(returns[name], direct[0])   # identical stream (no re-derivation drift)


# --------------------------------------------------------------- frozen-weight book -----

def test_frozen_weight_book_is_exact_linear_combo() -> None:
    ho = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([10.0, 20.0, 30.0])}
    book = _frozen_weight_book(ho, {"a": 2.0, "b": 1.0})
    np.testing.assert_allclose(book, [12.0, 24.0, 36.0])


def test_frozen_weight_book_nan_contributes_zero() -> None:
    ho = {"a": np.array([1.0, np.nan, 3.0]), "b": np.array([10.0, 20.0, 30.0])}
    book = _frozen_weight_book(ho, {"a": 2.0, "b": 1.0})
    np.testing.assert_allclose(book, [12.0, 20.0, 36.0])          # NaN 'a' at t=1 → only b contributes


# --------------------------------------------------------------- holdout guard ----------

def test_holdout_guard_threshold_and_units() -> None:
    """The guard passes iff the ANNUALIZED holdout ΔSR ≥ min_book_uplift; verify the returned delta
    drives the boolean at exactly that floor (so the unit is annualized, matching the funnel)."""
    slots = _noise_slots(4, seed=11)
    base, ts = _base_and_ts()
    panel = _panel(slots)
    base_book = _combined_book(base, ts, _CFG)
    pool, _ = assemble_overlay_pool(panel, base_book, _overlays(slots), cost_bps=0.0010)

    cut, emb = int(T * 0.75), 21
    n_train = cut - emb
    base_tr = {k: v[:n_train] for k, v in base.items()}
    base_ho = {k: v[cut:] for k, v in base.items()}
    pool_tr = {k: v[:n_train] for k, v in pool.items()}
    pool_ho = {k: v[cut:] for k, v in pool.items()}
    members = tuple(list(pool)[:3])

    delta, passes = _cohort_holdout_guard(
        members, base_tr, pool_tr, ts[:n_train], base_ho, pool_ho, _CCFG, _CFG)
    assert np.isfinite(delta)
    assert passes == bool(delta >= _CCFG.min_book_uplift)         # threshold logic is exactly the floor


def test_holdout_guard_delta_is_annualized_not_per_period() -> None:
    """NEGATIVE TRIPWIRE for the units invariant (Doc 2 §4 unit-pin): the guard's ΔSR MUST be
    ANNUALIZED (matching min_book_uplift == funnel min_combination_uplift, which _cpcv_delta_sr
    annualizes). A member with a materially positive per-period Sharpe is added to a ~zero-Sharpe base;
    a guard that (wrongly) used per-period Sharpe would report a ΔSR ~sqrt(252)× smaller. Reverting
    _ann_sharpe → _per_period_sharpe inside _cohort_holdout_guard MUST fail this."""
    import pytest

    from finrl_pro_ds.signals.eval_harness import _ann_sharpe
    from finrl_pro_ds.signals.generation.cohort import _with_redundancy
    from finrl_pro_ds.signals.generation.cohort_eval import (
        _alpha_paths,
        _frozen_weight_book,
        _last_confirmed_month_end_idx,
    )

    rng = np.random.default_rng(7)
    base_full = {"b": (0.0 + 0.010 * rng.standard_normal(T)).astype(np.float64)}     # ~zero Sharpe
    member_full = {"m": (0.0012 + 0.002 * rng.standard_normal(T)).astype(np.float64)}  # strong +Sharpe
    _, ts = _base_and_ts()
    cut, emb = int(T * 0.75), 21
    n_train = cut - emb
    base_tr = {"b": base_full["b"][:n_train]}
    base_ho = {"b": base_full["b"][cut:]}
    pool_tr = {"m": member_full["m"][:n_train]}
    pool_ho = {"m": member_full["m"][cut:]}

    delta, passes = _cohort_holdout_guard(
        ("m",), base_tr, pool_tr, ts[:n_train], base_ho, pool_ho, _CCFG, _CFG)

    # independent recomputation with the ANNUALIZED convention the guard must use.
    a_aug = _alpha_paths({**base_tr, **pool_tr}, ts[:n_train],
                         _with_redundancy(_CFG, _CCFG.combiner_redundancy_strength))
    a_base = _alpha_paths(dict(base_tr), ts[:n_train], _CFG)
    me = _last_confirmed_month_end_idx(ts[:n_train])   # freeze at the SAME index the guard uses (P2-01)
    faug = {s: float(np.asarray(a_aug[s])[me]) for s in a_aug}
    fbase = {s: float(np.asarray(a_base[s])[me]) for s in a_base}
    b_aug = _frozen_weight_book({**base_ho, **pool_ho}, faug)
    b_base = _frozen_weight_book(dict(base_ho), fbase)
    expected_ann = _ann_sharpe(b_aug, _CFG.periods_per_year) - _ann_sharpe(b_base, _CFG.periods_per_year)
    per_period = _per_period_sharpe_local(b_aug) - _per_period_sharpe_local(b_base)

    assert delta == pytest.approx(expected_ann)                  # guard uses _ann_sharpe (annualized)
    assert abs(delta) > 5 * abs(per_period)                      # annualized ≫ per-period (√252 factor)
    assert passes is True                                        # a strong +Sharpe member clears 0.10 annualized


def _per_period_sharpe_local(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    sd = float(x.std(ddof=1))
    return float(x.mean() / sd) if (x.size >= 2 and sd > 0) else float("nan")


# --------------------------------------------------------------- evaluate_cohort --------

def test_evaluate_cohort_small_pool_returns_none() -> None:
    """A pool with fewer de-correlated admits than min_cohort_size (here 2 independent slots < 3)
    yields NO cohort verdict — the analytic pre-filter returns None and evaluate_cohort short-circuits."""
    slots = _noise_slots(2, seed=5)
    panel = _panel(slots)
    base, ts = _base_and_ts()
    v = evaluate_cohort(panel, base, ts, _overlays(slots), _CCFG, _CFG,
                        mc_kwargs=_MC_KWARGS, cost_bps=0.0010, holdout_frac=0.25,
                        holdout_embargo=21, seed=123)
    assert v is None


def test_evaluate_cohort_deterministic() -> None:
    """Same inputs + same seed → byte-identical verdict (the crucible-reproduce invariant, Doc 2 §6)."""
    slots = _noise_slots(6, seed=9)
    panel = _panel(slots)
    base, ts = _base_and_ts()
    overlays = _overlays(slots)
    import dataclasses
    kw = dict(mc_kwargs=_MC_KWARGS, cost_bps=0.0010, holdout_frac=0.25, holdout_embargo=21, seed=777)
    v1 = evaluate_cohort(panel, base, ts, overlays, _CCFG, _CFG, **kw)
    v2 = evaluate_cohort(panel, base, ts, overlays, _CCFG, _CFG, **kw)
    assert isinstance(v1, CohortVerdict)
    # NaN-aware structural equality (dataclass == fails on NaN fields when a stage short-circuits);
    # assert_equal treats nan == nan, so this is a true byte-for-byte reproducibility check.
    np.testing.assert_equal(dataclasses.asdict(v1), dataclasses.asdict(v2))
    if np.isfinite(v1.mc_p_value):
        assert 0.0 <= v1.mc_p_value <= 1.0


def test_evaluate_cohort_pure_noise_not_promising() -> None:
    """Fixed-seed null-safety smoke: a pool of pure-noise overlays on a noise book must NOT be
    PROMISING (MC and/or holdout reject). The distributional size guarantee (≈ α) is proven with
    ≥200 meta-reps in test_generation_cohort_mc.py; here we assert the deterministic wired outcome."""
    slots = _noise_slots(6, seed=2)
    panel = _panel(slots)
    base, ts = _base_and_ts(seed=3)
    v = evaluate_cohort(panel, base, ts, _overlays(slots), _CCFG, _CFG,
                        mc_kwargs=_MC_KWARGS, cost_bps=0.0010, holdout_frac=0.25,
                        holdout_embargo=21, seed=4242)
    assert v is None or v.verdict != "PROMISING"


def test_evaluate_cohort_short_circuits_on_analytic_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the analytic floor fails, the (expensive) MC null is never called — Doc 2 §5 order.
    Seed-independent: the analytic pre-filter is stubbed to return evidence with the floor NOT
    cleared, so the assertion is about the ORCHESTRATION order, not a particular pool's statistics."""
    import finrl_pro_ds.signals.generation.cohort_eval as ce
    from finrl_pro_ds.signals.generation.cohort import CohortEvidence

    slots = _noise_slots(6, seed=13)
    panel = _panel(slots)
    base, ts = _base_and_ts()
    called = {"mc": False}

    def _spy_mc(*a, **k):
        called["mc"] = True
        raise AssertionError("MC must not run when the analytic floor fails")

    stub_ev = CohortEvidence(
        members=("ov-a", "ov-b", "ov-c"), n_members=3, n_candidates_seen=6,
        delta_sr_oos=0.0, dsr_cohort_book=0.5, cohort_hlz_t=1.0, mean_pairwise_corr=0.1,
        sr_star_cohort=0.2, passes_analytic_floor=False)     # floor NOT cleared
    monkeypatch.setattr(ce, "evaluate_cohort_analytic", lambda *a, **k: stub_ev)
    monkeypatch.setattr(ce, "mc_null_pvalue", _spy_mc)

    v = evaluate_cohort(panel, base, ts, _overlays(slots), _CCFG, _CFG,
                        mc_kwargs=_MC_KWARGS, cost_bps=0.0010, holdout_frac=0.25,
                        holdout_embargo=21, seed=1)
    assert called["mc"] is False
    assert v is not None and v.verdict == "LOGGED" and v.passes_analytic_floor is False


# --------------------------------------------------------------- CAUS-05 whole-path causality ----

def _poison_panel_tail(panel: Panel, cut: int, spike: float = 1.0e4) -> Panel:
    """Return a copy of ``panel`` whose every feature slot has rows ``>= cut`` (the embargoed
    holdout tail) overwritten with a large constant spike. OHLCV is untouched (the overlays here
    are pure feature-slot ``level`` terminals; a future-poison that a causal path ignores must not
    move any past pool value). ``(T,)`` and ``(T,N)`` slots both slice on axis 0."""
    import dataclasses
    poisoned = {}
    for k, v in panel.feature_slots.items():
        arr = np.array(v, dtype=np.float64, copy=True)
        arr[cut:] = spike
        poisoned[k] = arr
    return dataclasses.replace(panel, feature_slots=poisoned)


def test_assemble_then_slice_train_span_immune_to_future_poison() -> None:
    """CAUS-05 NEGATIVE TRIPWIRE (whole-path composite causality). Assemble the pool on the full
    panel and slice [:n_train]; then POISON the holdout tail (rows >= cut) of every feature slot
    with a huge spike, reassemble on the poisoned panel, and slice [:n_train]. The train-span pool
    returns MUST be byte-identical — a future value must never change a past pool value.

    MUTATION THIS BITES: replace the CAUSAL expanding-window z-score in
    ``evolve._overlay_returns`` (cmean=cumsum(g0)/cnt etc., bar t uses only g[:t+1]) with a GLOBAL
    full-sample normalization, e.g. z=(g-np.nanmean(g))/np.nanstd(g); m=np.tanh(z). Under that
    mutation the tail spike shifts nanmean(g)/nanstd(g), so EVERY z[t] for t<n_train changes ->
    m[t] -> cand[t]=m[t-1]*base_book[t] changes on the train span; assert_array_equal FAILS
    (empirically 428/429 = 99.8% mismatch)."""
    slots = _noise_slots(6, seed=21)
    panel = _panel(slots)
    base, ts = _base_and_ts()
    base_book = _combined_book(base, ts, _CFG)          # base_book is NOT panel-derived -> identical both sides
    overlays = _overlays(slots)

    T_ = panel.T
    cut = int(T_ * (1.0 - 0.25))                        # same split arithmetic evaluate_cohort uses
    n_train = max(1, cut - 21)
    assert 2 <= n_train < cut < T_

    clean_pool, clean_culled = assemble_overlay_pool(panel, base_book, overlays, cost_bps=0.0010)
    poisoned_panel = _poison_panel_tail(panel, cut)
    poisoned_pool, poisoned_culled = assemble_overlay_pool(
        poisoned_panel, base_book, overlays, cost_bps=0.0010)

    assert clean_culled == [] and poisoned_culled == []      # both pools fully formed (no accidental cull)
    assert set(clean_pool) == set(poisoned_pool) == set(overlays)

    # sanity: the poison MUST actually change the HOLDOUT span (else the test would be vacuous).
    tail_moved = any(
        not np.array_equal(clean_pool[k][cut:], poisoned_pool[k][cut:], equal_nan=True)
        for k in overlays)
    assert tail_moved, "future poison did not perturb the holdout span — test would be vacuous"

    for name in overlays:
        np.testing.assert_array_equal(
            clean_pool[name][:n_train], poisoned_pool[name][:n_train],
            err_msg=f"future poison leaked into train span of overlay {name!r} "
                    f"(non-causal transform in _overlay_returns?)")


def test_assemble_then_slice_equals_truncate_then_assemble() -> None:
    """CAUS-05 symmetric leg: assemble-then-slice == truncate-then-assemble. Computing the pool on
    the FULL panel and slicing [:n_train] must equal computing it on a panel PHYSICALLY TRUNCATED
    to the first n_train rows.

    MUTATION THIS BITES: the SAME global/full-sample normalization. On the full panel the z-score
    denominator uses all T rows; on the truncated panel it uses only n_train rows -> different
    m[t] -> different cand[t] -> equality FAILS. The causal expanding code makes the truncated
    cumsum a bit-identical prefix of the full one, so equality holds."""
    slots = _noise_slots(6, seed=22)
    panel = _panel(slots)
    base, ts = _base_and_ts()
    base_book = _combined_book(base, ts, _CFG)
    overlays = _overlays(slots)

    T_ = panel.T
    cut = int(T_ * (1.0 - 0.25))
    n_train = max(1, cut - 21)
    assert 2 <= n_train < T_

    full_pool, _ = assemble_overlay_pool(panel, base_book, overlays, cost_bps=0.0010)
    trunc_panel = panel.truncated(n_train - 1)          # keeps rows [0..n_train-1] -> T == n_train
    assert trunc_panel.T == n_train
    trunc_pool, _ = assemble_overlay_pool(
        trunc_panel, base_book[:n_train], overlays, cost_bps=0.0010)

    assert set(trunc_pool) == set(overlays)
    for name in overlays:
        np.testing.assert_array_equal(
            full_pool[name][:n_train], trunc_pool[name],
            err_msg=f"assemble-then-slice != truncate-then-assemble for overlay {name!r} "
                    f"(non-causal transform in the overlay feature path?)")


# --------------------------------------------------------------- N-CANDIDATES deflation-N -----

def test_n_candidates_seen_excludes_culled_not_all_scored() -> None:
    """NEGATIVE TRIPWIRE for the deflation-N semantics (Doc 1 Algorithm step 1; Doc 2 §2 MEDIUM/pool
    + §3.1): N == the pool of candidates with a FINITE, NON-DEGENERATE return stream (incl. LOGGED),
    and degenerate (all-NaN / globally-constant) overlays are the pre-registered "hard-infeasible /
    leak-culled" EXCLUSION — they raise ``n_culled`` but must NOT enter ``n_candidates_seen`` (and
    hence must not raise ``SR*_cohort``).

    Construction: the SAME 6 independent noise overlays evaluated twice — once alone, once with ONE
    extra CONSTANT feature slot whose overlay ``_overlay_returns`` culls (zero-variance timing → None).
    The constant slot changes NOTHING the deflation sees:
      * ``n_candidates_seen`` is 6 in BOTH cases (the culled overlay is not a scored candidate);
      * ``n_culled`` is 0 vs 1 (provenance only);
      * ``sr_star_cohort`` is byte-identical (N unchanged ⇒ order-statistic benchmark unchanged).

    MUTATION THIS FAILS ON: threading the orchestrator's ``n_culled`` into ``n_candidates_seen``
    (equivalently reverting the analytic ``n = len(pool)`` to count culled/degenerate streams) — the
    completeness-critic's proposed "include culled" change. That lifts the culled variant's N from
    6 → 7; ``expected_topm_order_stat_sum`` is strictly increasing in N, so ``sr_star_cohort`` jumps
    (0.034575 → 0.040247 at these moments) and the ``n_candidates_seen`` and/or ``sr_star_cohort``
    equalities below flip to failing. (Empirically verified: threading n_culled into _verdict flips
    line `assert v_culled.n_candidates_seen == 6` to `assert 7 == 6`.)
    """
    slots = _noise_slots(6, seed=9)
    overlays_clean = _overlays(slots)

    slots_with_const = dict(slots)
    slots_with_const["fred:CONST"] = np.full(T, 3.14, dtype=np.float64)   # zero-variance ⇒ culled
    overlays_culled = _overlays(slots_with_const)

    base, ts = _base_and_ts()
    kw = dict(mc_kwargs=_MC_KWARGS, cost_bps=0.0010, holdout_frac=0.25, holdout_embargo=21, seed=777)

    v_clean = evaluate_cohort(_panel(slots), base, ts, overlays_clean, _CCFG, _CFG, **kw)
    v_culled = evaluate_cohort(_panel(slots_with_const), base, ts, overlays_culled, _CCFG, _CFG, **kw)

    assert isinstance(v_clean, CohortVerdict) and isinstance(v_culled, CohortVerdict)
    # the constant overlay is culled, not scored
    assert v_clean.n_culled == 0 and v_culled.n_culled == 1
    # ...and the deflation N is the non-degenerate pool in BOTH — the culled candidate is NOT in N
    assert v_clean.n_candidates_seen == 6
    assert v_culled.n_candidates_seen == 6            # FAILS at 7 under n = len(pool) + n_culled
    # ...so the order-statistic deflation benchmark is unchanged by the culled candidate
    assert v_culled.sr_star_cohort == v_clean.sr_star_cohort   # FAILS (0.034575 → 0.040247) under the mutation


# --------------------------------------------------------------- HOLDOUT partial-month freeze ----

def _daily_ts(start: str, periods: int) -> np.ndarray:
    """Daily (calendar) decision stamps (epoch s) — a DAILY grid so a span can end MID calendar month
    (business-day grids can end on the last *business* day short of the calendar month-end)."""
    idx = pd.date_range(start, periods=periods, freq="D")
    return idx.view("int64").astype(np.float64) / 1e9


def _confirmed_month_end_idx_ref(ts: np.ndarray) -> int:
    """Independent reference for the last confirmed month-end index (does NOT call the SUT helper)."""
    ts = np.asarray(ts, dtype=np.int64)
    Tn = ts.size
    dt = pd.to_datetime(ts, unit="s")
    lom = pd.Series(np.arange(Tn)).groupby(dt.to_period("M").values).max().to_numpy()
    final = dt[-1]
    final_true = bool(final.day == final.days_in_month)
    conf = [int(i) for i in lom if (i < Tn - 1) or final_true]
    return max(conf) if conf else Tn - 1


def test_holdout_guard_freezes_last_confirmed_month_end() -> None:
    """NEGATIVE TRIPWIRE (Doc 2 §4 frozen-weight contract / P2-01). On a train span that ends
    MID-month, the guard MUST freeze the combiner weights at the LAST CONFIRMED month-end, NOT the
    partial-month final train bar. Reverting the freeze index from `_last_confirmed_month_end_idx`
    back to `[-1]` MUST fail this: the guard would book the holdout under partial-month alpha, so its
    returned delta would equal the alpha[-1] book delta, not the alpha[last-ME] book delta (the two
    frozen vectors are asserted to differ, so the assertion is non-vacuous)."""
    from finrl_pro_ds.signals.eval_harness import _ann_sharpe
    from finrl_pro_ds.signals.generation.cohort import _with_redundancy
    from finrl_pro_ds.signals.generation.cohort_eval import _alpha_paths, _frozen_weight_book

    n = 430
    ts_full = _daily_ts("2012-01-02", n)
    cut = int(n * 0.75)          # 322
    emb = 21
    n_train = cut - emb          # 301
    ts_train = ts_full[:n_train]
    me = _confirmed_month_end_idx_ref(ts_train)
    assert me != n_train - 1, "test setup must end mid-month for the tripwire to bite"

    rng = np.random.default_rng(20260704)
    base_full = {"tsmom": (0.0004 + 0.008 * rng.standard_normal(n)).astype(np.float64),
                 "rates_carry": (0.0003 + 0.007 * rng.standard_normal(n)).astype(np.float64)}
    member_full = {"m0": (0.0006 + 0.009 * rng.standard_normal(n)).astype(np.float64),
                   "m1": (0.0002 + 0.011 * rng.standard_normal(n)).astype(np.float64)}
    base_tr = {k: v[:n_train] for k, v in base_full.items()}
    base_ho = {k: v[cut:] for k, v in base_full.items()}
    pool_tr = {k: v[:n_train] for k, v in member_full.items()}
    pool_ho = {k: v[cut:] for k, v in member_full.items()}
    members = ("m0", "m1")

    delta, _ = _cohort_holdout_guard(
        members, base_tr, pool_tr, ts_train, base_ho, pool_ho, _CCFG, _CFG)

    a_aug = _alpha_paths({**base_tr, **pool_tr}, ts_train,
                         _with_redundancy(_CFG, _CCFG.combiner_redundancy_strength))
    a_base = _alpha_paths(dict(base_tr), ts_train, _CFG)
    frozen_aug_me = {s: float(np.asarray(a_aug[s])[me]) for s in a_aug}
    frozen_base_me = {s: float(np.asarray(a_base[s])[me]) for s in a_base}
    frozen_aug_tail = {s: float(np.asarray(a_aug[s])[-1]) for s in a_aug}
    frozen_base_tail = {s: float(np.asarray(a_base[s])[-1]) for s in a_base}

    assert any(frozen_aug_me[s] != frozen_aug_tail[s] for s in a_aug), \
        "alpha at last-confirmed-ME must differ from partial-final-bar alpha (else the tripwire is vacuous)"

    book_aug_me = _frozen_weight_book({**base_ho, **pool_ho}, frozen_aug_me)
    book_base_me = _frozen_weight_book(dict(base_ho), frozen_base_me)
    delta_me = _ann_sharpe(book_aug_me, _CFG.periods_per_year) - _ann_sharpe(book_base_me, _CFG.periods_per_year)

    book_aug_tail = _frozen_weight_book({**base_ho, **pool_ho}, frozen_aug_tail)
    book_base_tail = _frozen_weight_book(dict(base_ho), frozen_base_tail)
    delta_tail = _ann_sharpe(book_aug_tail, _CFG.periods_per_year) - _ann_sharpe(book_base_tail, _CFG.periods_per_year)

    assert delta == pytest.approx(delta_me)
    assert delta != pytest.approx(delta_tail)


def test_holdout_guard_month_end_span_is_noop() -> None:
    """CONTROL (byte-identical requirement / ADR-1): when the train span ends ON a true calendar
    month-end, freezing at the last confirmed month-end IS the final bar, so the fix is a NO-OP —
    it returns exactly what the old `[-1]` freeze returned. Guards against the fix perturbing the
    happy path."""
    from finrl_pro_ds.signals.eval_harness import _ann_sharpe
    from finrl_pro_ds.signals.generation.cohort import _with_redundancy
    from finrl_pro_ds.signals.generation.cohort_eval import _alpha_paths, _frozen_weight_book

    ts_full = _daily_ts("2012-01-02", 430)             # ends 2013-03-06
    idx = pd.date_range("2012-01-02", periods=430, freq="D")
    n_train = int(np.where(idx.date == pd.Timestamp("2012-12-31").date())[0][0]) + 1
    cut = n_train + 21
    ts_train = ts_full[:n_train]
    assert _confirmed_month_end_idx_ref(ts_train) == n_train - 1, "train span must end ON a month-end"

    rng = np.random.default_rng(11)
    base_full = {"tsmom": (0.0004 + 0.008 * rng.standard_normal(430)).astype(np.float64),
                 "rates_carry": (0.0003 + 0.007 * rng.standard_normal(430)).astype(np.float64)}
    member_full = {"m0": (0.0006 + 0.009 * rng.standard_normal(430)).astype(np.float64),
                   "m1": (0.0002 + 0.011 * rng.standard_normal(430)).astype(np.float64)}
    base_tr = {k: v[:n_train] for k, v in base_full.items()}
    base_ho = {k: v[cut:] for k, v in base_full.items()}
    pool_tr = {k: v[:n_train] for k, v in member_full.items()}
    pool_ho = {k: v[cut:] for k, v in member_full.items()}

    delta, _ = _cohort_holdout_guard(
        ("m0", "m1"), base_tr, pool_tr, ts_train, base_ho, pool_ho, _CCFG, _CFG)

    a_aug = _alpha_paths({**base_tr, **pool_tr}, ts_train,
                         _with_redundancy(_CFG, _CCFG.combiner_redundancy_strength))
    a_base = _alpha_paths(dict(base_tr), ts_train, _CFG)
    faug = {s: float(np.asarray(a_aug[s])[-1]) for s in a_aug}
    fbase = {s: float(np.asarray(a_base[s])[-1]) for s in a_base}
    b_aug = _frozen_weight_book({**base_ho, **pool_ho}, faug)
    b_base = _frozen_weight_book(dict(base_ho), fbase)
    delta_tail = _ann_sharpe(b_aug, _CFG.periods_per_year) - _ann_sharpe(b_base, _CFG.periods_per_year)

    assert delta == pytest.approx(delta_tail)          # month-end span → fix is a byte-for-byte no-op
