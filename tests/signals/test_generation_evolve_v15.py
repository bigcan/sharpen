"""crucible-v15.0 tripwires for the mining engine (``signals/generation/evolve.py`` + ``fitness.py``).

Deep audit 2026-09-30. Each test pins a defect found by the audit and FAILS on the pre-v15 code:

  * a pre-registered seed beyond the GP's search bounds (> max_ast_nodes, or turnover > 2x soft cap)
    was culled before fitness, never reached the holdout, and was still charged a LORD++ test — 40 of
    the 100 published WQ101 formulas exceed 24 nodes, including curated seed #9;
  * raw (non-canonical) seed strings silently emptied the ``prereg_only`` eligible set;
  * ``n_holdout_tested`` counted candidates whose holdout scoring was degenerate or raised;
  * a NaN corrected statistic was recorded as a holdout rejection;
  * the market-beta exposure leg read the PREVIOUS bar's market move against the NEXT bar's book
    return, so it measured ~0 on a pure-beta book and could never fire.

Plus bit-identity pins for the byte-identical speedups (vectorized ``_candidate_returns``, the cached
base book, the single full-panel evaluation reused by the causality probe).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from sharpen.crucible.corrected_contract import CorrectedConfig, _market_beta  # noqa: E402
from sharpen.signals.eval_harness import _ls_weights, assert_causal  # noqa: E402
from sharpen.signals.features import Panel  # noqa: E402
from sharpen.signals.generation import evolve as ev  # noqa: E402
from sharpen.signals.generation.dsl_signal import DslSignal, eval_on_panel  # noqa: E402
from sharpen.signals.generation.fitness import FitnessConfig, combination_fitness  # noqa: E402
from sharpen.signals.generation.grammar import node_count, parse, to_formula  # noqa: E402
from sharpen.signals.library._alpha_formulas import FORMULAS  # noqa: E402
from research.crucible_calibration import (  # noqa: E402
    _panel_ts,
    _planted_base_sleeves,
    _planted_panel,
    load_calib,
)

_CORRECTED_GATES = ROOT / "configs" / "crucible_corrected_contract.gates.yaml"
_CALIB_GATES = ROOT / "configs" / "crucible_calibration.gates.yaml"
_FUNNEL_GATES = ROOT / "configs" / "signal_eval.gates.yaml"


@pytest.fixture(scope="module")
def calib():
    return load_calib(_CALIB_GATES, _FUNNEL_GATES)


@pytest.fixture(scope="module")
def corr() -> CorrectedConfig:
    return CorrectedConfig.from_yaml(_CORRECTED_GATES)


def _substrate(t: int = 900, n: int = 10, seed: int = 4242):
    panel, s = _planted_panel(t, n, seed=seed)
    return panel, _planted_base_sleeves(s, beta=0.0, seed=seed), _panel_ts(panel)


def _run_xs(calib, corr, seeds, **over):
    panel, base, ts = _substrate()
    ek = dict(calib.ek)
    ek.update(pop_size=8, n_generations=1, ls_min_names=4, rng_seed=11)
    ek.update(over)
    return evolve_xs(seeds, panel, base, ts, calib, corr, ek)


def evolve_xs(seeds, panel, base, ts, calib, corr, ek):
    return ev.evolve(seeds, panel, base, ts, calib.fit_cfg, candidate_type="cross_sectional",
                     contract=ev.CONTRACT_CORRECTED, corrected_cfg=corr, **ek)


# ------------------------------------------------------------------ search bounds vs pre-registration
def test_wq101_node_census_is_what_the_audit_measured() -> None:
    """The premise: 40 of the 100 published formulas exceed the shipped 24-node search bound, and
    curated seed #9 is one of them. If the grammar or the bank changes this test is the place to re-read
    the argument, not a reason to delete the exemption."""
    sizes = {i: node_count(parse(f)) for i, f in FORMULAS.items() if i != 56}
    assert sum(1 for n in sizes.values() if n > 24) == 40
    assert sizes[9] > 24


def test_prereg_beyond_node_cap_reaches_the_holdout_but_is_never_bred(calib, corr) -> None:
    big = to_formula(parse(FORMULAS[9]))                     # 28 nodes, curated seed #9
    small = to_formula(parse(FORMULAS[4]))
    assert node_count(parse(big)) > calib.fit_cfg.max_ast_nodes
    rep = _run_xs(calib, corr, [big, small])
    decided = {hv["formula"] for hv in rep.holdout_validation if "holdout_passes" in hv}
    assert big in decided, "a pre-registered seed must be TESTED, not culled by a search bound"
    assert small in decided
    cand = next(c for c in rep.hall_of_fame if c.formula == big)
    assert cand.result is not None and cand.fitness == float("-inf")   # scored, but never an elite
    assert rep.n_holdout_tested == 2 and rep.not_tested == []


def test_prereg_beyond_turnover_bound_reaches_the_holdout(calib, corr) -> None:
    """A daily-rebalanced reversal seed runs far above 2x the turnover soft cap (24/yr) — the H=2
    us_equity seeds ran 88-168/yr and were culled in production. Its cost is already in its NET
    return stream, so under the corrected contract it is tested."""
    f = to_formula(parse("rank(delta(close, 1))"))
    rep = _run_xs(calib, corr, [f], hold_horizon=1)
    hv = [h for h in rep.holdout_validation if h["formula"] == f]
    assert hv and "holdout_passes" in hv[0]


def test_offspring_still_respect_the_search_bounds(calib, corr, monkeypatch) -> None:
    """The exemption is for PRE-REGISTRATIONS only: a bred genome over the bound is still culled."""
    panel, base, ts = _substrate()
    ek = dict(calib.ek)
    ek.update(pop_size=8, n_generations=1, ls_min_names=4, rng_seed=11)
    big = to_formula(parse(FORMULAS[9]))
    rep = ev.evolve([big], panel, base, ts, calib.fit_cfg, candidate_type="cross_sectional",
                    contract=ev.CONTRACT_SHIPPED, **ek)
    cand = next(c for c in rep.hall_of_fame if c.formula == big)
    assert cand.result is None and "hard-infeasible" in cand.reason   # shipped contract: unchanged


def test_raw_seed_strings_match_the_preregistration(calib, corr) -> None:
    """Verbatim WQ101 strings are not canonical (``-1 * x`` emits as ``((-1) * x)``); the eligible set
    used to be matched on the RAW strings and came back empty."""
    raw = [FORMULAS[3], FORMULAS[4]]
    assert all(to_formula(parse(f)) != f for f in raw)
    rep = _run_xs(calib, corr, raw)
    assert rep.n_holdout_tested == 2


def test_denominator_counts_decisions_and_names_what_was_not_tested(calib, corr) -> None:
    dead = to_formula(parse("log(((-1) * close))"))          # all-NaN score → degenerate, never decided
    live = to_formula(parse(FORMULAS[4]))
    rep = _run_xs(calib, corr, [dead, live])
    assert rep.n_holdout_tested == 1
    assert [d["formula"] for d in rep.not_tested] == [dead]
    assert "degenerate" in rep.not_tested[0]["reason"]


def test_nan_statistic_is_not_a_holdout_decision(calib, corr, monkeypatch) -> None:
    import sharpen.crucible.corrected_contract as cc_mod

    real = cc_mod.corrected_contract_fitness

    def nan_stat(*a, **k):
        r = real(*a, **k)
        from dataclasses import replace
        return replace(r, corrected_t=float("nan"), passes_corrected=False, t_pass=False)

    monkeypatch.setattr(cc_mod, "corrected_contract_fitness", nan_stat)
    live = to_formula(parse(FORMULAS[4]))
    rep = _run_xs(calib, corr, [live])
    assert rep.n_holdout_tested == 0
    assert rep.not_tested and "degenerate statistic" in rep.not_tested[0]["reason"]
    assert all("holdout_passes" not in hv for hv in rep.holdout_validation)


_ZERO = "(close * 0)"        # the shape a crossover leaves when it splices a zero into a coefficient slot


def test_candidate_never_in_the_book_is_not_a_holdout_decision(calib, corr) -> None:
    """A stream the combiner cannot size on any holdout bar leaves the augmented book EQUAL to the base
    book: z = 0 exactly, p = 0.5 — recorded (and charged) as a holdout rejection before v15.0."""
    dead = to_formula(parse(_ZERO))
    live = to_formula(parse(FORMULAS[4]))
    rep = _run_xs(calib, corr, [dead, live])
    assert rep.n_holdout_tested == 1
    assert [d["formula"] for d in rep.not_tested] == [dead]
    assert "never sized" in rep.not_tested[0]["reason"]
    assert not any("holdout_passes" in hv for hv in rep.holdout_validation if hv["formula"] == dead)


def test_degenerate_genome_is_scored_but_never_bred(calib, corr) -> None:
    """Its train "uplift" is the combiner's equal-weight fallback, not the genome (+0.33 on this
    substrate), so it used to rank among the elites; it keeps its result but never breeds."""
    dead = to_formula(parse(_ZERO))
    rep = _run_xs(calib, corr, [dead, to_formula(parse(FORMULAS[4]))])
    cand = next(c for c in rep.hall_of_fame if c.formula == dead)
    assert cand.result is not None and not cand.result.not_degenerate
    assert cand.fitness == float("-inf")


# ------------------------------------------------------------------ exposure (market-beta) alignment
def test_market_series_is_forward_aligned_with_book_returns() -> None:
    panel, _, _ = _substrate(t=700, n=12, seed=99)
    mkt = ev._panel_market_returns(panel)
    fwd = panel.forward_returns(1)
    ew = np.nanmean(np.where(panel.active, fwd, np.nan), axis=1)
    finite = np.isfinite(ew)
    assert np.allclose(mkt[finite], ew[finite])
    # a book that IS half the market plus independent noise must measure beta ~0.5 (the pre-fix
    # lagged series measured ~0 and the 0.30 us_equity leg could never fire)
    rng = np.random.default_rng(3)
    book = 0.5 * ew + 0.002 * rng.standard_normal(ew.size)
    assert _market_beta(book, mkt) == pytest.approx(0.5, abs=0.08)


# ------------------------------------------------------------------ byte-identical speedups
def _ref_candidate_returns(formula, panel, *, hold_horizon, cost_bps, min_names):
    """VERBATIM copy of the pre-v15 per-day loop."""
    scores = eval_on_panel(formula, panel)
    if not np.isfinite(scores).any():
        return None
    fwd1 = panel.forward_returns(1)
    T = panel.T
    rets = np.full(T, np.nan)
    turns: list[float] = []
    w = np.zeros(panel.N)
    for t in range(T - 1):
        if t % hold_horizon == 0:
            w_new = _ls_weights(scores[t], panel.active[t], min_names=min_names)
            turns.append(float(np.abs(w_new - w).sum()))
            w = w_new
        rets[t] = float(np.nansum(w * fwd1[t]) - cost_bps * (turns[-1] if t % hold_horizon == 0
                                                             and turns else 0.0))
    ppy = 252.0 / hold_horizon
    turnover_ann = float(np.mean(turns) * ppy) if turns else 0.0
    return rets, turnover_ann


def _ragged_panel(seed: int) -> Panel:
    rng = np.random.default_rng(seed)
    t, n = int(rng.integers(30, 400)), int(rng.integers(3, 40))
    close = np.exp(np.cumsum(0.01 * rng.standard_normal((t, n)), axis=0) + 4.0)
    close[rng.random((t, n)) < 0.03] = np.nan
    active = rng.random((t, n)) > 0.1
    dates = (np.datetime64("2015-01-02") + np.arange(t) * np.timedelta64(1, "D")).astype("datetime64[ns]")
    return Panel(dates, tuple(f"R{i}" for i in range(n)), close, close * 1.01, close * 0.99, close,
                 rng.uniform(1e5, 1e7, (t, n)), active, close * 1e6, rng.integers(0, 3, size=n),
                 {"survivorship_free": True, "source": "synthetic"})


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("hold", [1, 2, 5, 21])
def test_vectorized_candidate_returns_bit_identical(seed, hold) -> None:
    panel = _ragged_panel(seed)
    for f in ("rank(delta(close, 5))", "(-1 * correlation(rank(open), rank(volume), 10))"):
        got = ev._candidate_returns(f, panel, hold_horizon=hold, cost_bps=0.0007, min_names=3)
        ref = _ref_candidate_returns(f, panel, hold_horizon=hold, cost_bps=0.0007, min_names=3)
        assert (got is None) == (ref is None)
        if got is not None:
            assert np.array_equal(got[0], ref[0], equal_nan=True) and got[1] == ref[1]


def test_base_book_memo_is_an_identity() -> None:
    rng = np.random.default_rng(5)
    t = 600
    base = {"a": 0.01 * rng.standard_normal(t), "b": 0.02 * rng.standard_normal(t)}
    ts = (np.datetime64("2012-01-02", "s") + np.arange(t) * np.timedelta64(86400, "s")
          ).astype(np.int64).astype(np.float64)
    cand = 0.015 * rng.standard_normal(t)
    cfg = FitnessConfig()
    from sharpen.signals.generation.fitness import _combined_book
    memo = _combined_book(base, ts, cfg)
    a = combination_fitness(cand, base, ts, cfg, gen_n_eff=10, turnover_ann=3.0, n_nodes=5)
    b = combination_fitness(cand, base, ts, cfg, gen_n_eff=10, turnover_ann=3.0, n_nodes=5,
                            base_book=memo)
    assert repr(a) == repr(b)


def test_assert_causal_with_precomputed_full_is_the_same_check() -> None:
    panel, _, _ = _substrate(t=400, n=8, seed=7)
    f = "rank(delta(close, 5))"
    full = eval_on_panel(f, panel)
    assert assert_causal(DslSignal(f), panel, full=full) == assert_causal(DslSignal(f), panel)

    class Leaky:                                             # reads tomorrow's close
        def compute(self, p):
            c = np.asarray(p.close, dtype=np.float64)
            out = np.full_like(c, np.nan)
            out[:-1] = c[1:]
            return out

    leaky = Leaky()
    ok, _ = assert_causal(leaky, panel, full=leaky.compute(panel))
    assert not ok


# ------------------------------------------------------------------ offspring search skip (v15.0)
def test_skipping_the_offspring_search_leaves_every_prereg_decision_identical(calib, corr) -> None:
    """Under corrected + prereg_only an offspring can never be promoted, so the search is skipped by
    default (auto). The pre-registered seeds' holdout decisions — statistic, p-value, legs — must be
    bit-identical to a run that did search."""
    seeds = [to_formula(parse(FORMULAS[i])) for i in (3, 4, 6, 12)]
    searched = _run_xs(calib, corr, seeds, n_generations=3, pop_size=12, search_offspring=True)
    auto = _run_xs(calib, corr, seeds, n_generations=3, pop_size=12)
    assert searched.offspring_searched and not auto.offspring_searched
    assert {c.formula for c in auto.hall_of_fame} <= set(seeds)          # file drawer = the seeds
    assert auto.gen_n_total == len(seeds) < searched.gen_n_total
    key = lambda r: sorted((hv["formula"], repr(hv)) for hv in r.holdout_validation  # noqa: E731
                           if hv["formula"] in seeds)
    assert key(auto) == key(searched)
    assert auto.n_holdout_tested == searched.n_holdout_tested == len(seeds)


def test_shipped_contract_still_searches_by_default(calib) -> None:
    panel, base, ts = _substrate()
    ek = dict(calib.ek)
    ek.update(pop_size=8, n_generations=2, ls_min_names=4, rng_seed=11)
    rep = ev.evolve([to_formula(parse(FORMULAS[4]))], panel, base, ts, calib.fit_cfg,
                    candidate_type="cross_sectional", contract=ev.CONTRACT_SHIPPED, **ek)
    assert rep.offspring_searched and rep.gen_n_total > 1
