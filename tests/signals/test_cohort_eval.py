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
    from finrl_pro_ds.signals.generation.cohort_eval import _alpha_paths, _frozen_weight_book

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
    faug = {s: float(np.asarray(a_aug[s])[-1]) for s in a_aug}
    fbase = {s: float(np.asarray(a_base[s])[-1]) for s in a_base}
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
