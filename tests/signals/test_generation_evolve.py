"""C3.4 tripwires — the evolutionary generation loop.

The load-bearing test is the NOISE CALIBRATION: evolving on a pure-noise cross-asset panel must
surface ZERO promising survivors — the harness must not hallucinate alpha from noise (the
cont-73 lesson, mechanized). Plus the file-drawer-N count (every genome, incl. culled),
determinism, and an end-to-end smoke.

Design: .agent/artifacts/alpha_generation_component3_architecture.md §C3.4
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sharpen.signals.features import Panel
from sharpen.signals.generation.evolve import evolve
from sharpen.signals.generation.fitness import FitnessConfig
from sharpen.signals.library._alpha_formulas import FORMULAS

T, N = 320, 12
_SEEDS = [FORMULAS[1], FORMULAS[3], FORMULAS[6], FORMULAS[12]]
_CFG = FitnessConfig(embargo=10)


def _noise_panel(seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    close = np.exp(np.cumsum(0.01 * rng.standard_normal((T, N)), axis=0) + 4.0)
    open_ = close * (1 + 0.001 * rng.standard_normal((T, N)))
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    vol = rng.uniform(1e6, 1e8, (T, N))
    dates = (np.datetime64("2014-01-02") + np.arange(T) * np.timedelta64(1, "D")
             ).astype("datetime64[ns]")
    return Panel(dates, tuple(f"E{i:02d}" for i in range(N)), open_, high, low, close, vol,
                 np.ones((T, N), bool), close * vol, rng.integers(0, 4, size=N),
                 {"survivorship_free": True, "source": "synthetic"})


def _base_and_ts(seed: int = 1) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    base = {"tsmom": 0.0004 + 0.010 * rng.standard_normal(T),
            "rates_carry": 0.0003 + 0.008 * rng.standard_normal(T)}
    idx = pd.date_range("2014-01-02", periods=T, freq="B")
    return base, idx.view("int64").astype(np.float64) / 1e9


def _run(seed: int = 0):
    panel = _noise_panel(seed)
    base, ts = _base_and_ts()
    return evolve(_SEEDS, panel, base, ts, _CFG, rng_seed=11, pop_size=16, n_generations=2,
                  hold_horizon=21, ls_min_names=6)


def test_noise_panel_yields_zero_promising() -> None:
    rep = _run()
    assert rep.promising == []                         # must NOT manufacture alpha from noise


def test_file_drawer_n_counts_every_genome() -> None:
    rep = _run()
    # every genome ever scored (incl. culled/infeasible) counts toward the deflation N
    assert rep.gen_n_total >= len(_SEEDS)
    assert rep.gen_n_eff == float(max(2, rep.gen_n_total))


def test_end_to_end_smoke_returns_ranked_hall_of_fame() -> None:
    rep = _run()
    assert 1 <= len(rep.hall_of_fame) <= 10
    fits = [c.fitness for c in rep.hall_of_fame]
    assert fits == sorted(fits, reverse=True)          # ranked best-first
    assert all(isinstance(c.formula, str) for c in rep.hall_of_fame)


def test_deterministic_under_seed() -> None:
    a = _run()
    b = _run()
    assert [c.formula for c in a.hall_of_fame] == [c.formula for c in b.hall_of_fame]
    assert a.gen_n_total == b.gen_n_total


def test_seed_formulas_have_no_uneval_token() -> None:
    """GP5-04: every warm-start seed parses + round-trips cleanly — none carries a token outside the
    grammar's eval inputs (the `cap` class). Seed 56 (uses `cap`) is SKIP'd and not in SEED_NUMS, and
    grammar.INPUTS excludes `cap`, so no genome can introduce it; this locks that."""
    from sharpen.signals.generation.grammar import parse, to_formula
    from sharpen.signals.library._alpha_formulas import FORMULAS
    from sharpen.signals.library.alphas101 import SKIP
    seed_nums = (1, 3, 4, 6, 9, 12, 14, 19, 33, 53)      # mirrors generate_alphas.SEED_NUMS
    for n in seed_nums:
        assert n not in SKIP
        assert "cap" not in FORMULAS[n]
        assert isinstance(to_formula(parse(FORMULAS[n])), str)   # parses + round-trips


def test_pbo_reported_advisory(monkeypatch) -> None:
    """GP7-03: evolve emits an advisory CSCV PBO over the candidate sample."""
    rep = _run()
    assert rep.pbo is None or (0.0 <= rep.pbo["pbo"] <= 1.0 and rep.pbo["n_strategies"] >= 2)


def test_fitness_exception_culls_genome_not_crash_run(monkeypatch) -> None:
    """F3 (regression): a genome that makes combination_fitness RAISE is culled (counts toward the
    file-drawer N, fitness=-inf), it does NOT crash the whole evolve run. This guards the first
    full --mode real run, where a real-data genome surfaced an exception the synthetic calibration
    never produced — combination_fitness is called outside the per-genome try in score()."""
    import sharpen.signals.generation.evolve as ev

    def _boom(*_a, **_k):
        raise ValueError("synthetic fitness blow-up")

    monkeypatch.setattr(ev, "combination_fitness", _boom)
    rep = _run()                                          # must return, not raise
    assert isinstance(rep.hall_of_fame, list)
    assert rep.promising == []                            # all culled → nothing promising
    assert rep.gen_n_total >= len(_SEEDS)                 # culled genomes still counted (file-drawer N)


def test_holdout_validation_present_for_any_promising() -> None:
    rep = _run()
    # holdout covers train-gate pre-passers; PROMISING ⊆ those (requires the full-N holdout
    # gate too). On noise both are empty.
    assert isinstance(rep.holdout_validation, list)
    assert len(rep.holdout_validation) >= len(rep.promising)


def test_cross_sectional_ignores_feature_slots() -> None:
    """NOW-11A (C2-06): cross_sectional draws INPUTS only, so a panel's feature slots do NOT enter its
    search — the hall of fame is IDENTICAL with and without the slot, and no genome references it. (An
    OHLCV-only panel is unaffected; the change bites only where feature slots were leaking in.)"""
    import dataclasses

    base, ts = _base_and_ts()
    panel = _noise_panel(0)
    slot = np.sin(2 * np.pi * np.arange(T) / 50.0).astype(np.float64)
    panel_with = dataclasses.replace(panel, feature_slots={"macro:x": slot})
    kw = dict(candidate_type="cross_sectional", rng_seed=11, pop_size=16, n_generations=2,
              hold_horizon=21, ls_min_names=6)
    r_without = evolve(_SEEDS, panel, base, ts, _CFG, **kw)
    r_with = evolve(_SEEDS, panel_with, base, ts, _CFG, **kw)
    assert [c.formula for c in r_with.hall_of_fame] == [c.formula for c in r_without.hall_of_fame]
    assert not any("macro:x" in c.formula for c in r_with.hall_of_fame)
