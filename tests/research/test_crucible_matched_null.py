"""Tests for the Crucible F5 matched-null harness (scripts/research/crucible_matched_null.py, cont-139).

The figure's claim: on PURE NOISE, the shipped evolve+gate produces a Hall-of-Fame whose MAX ΔSR /
marginal_t reach the real record's maxima — while NOTHING survives the binding gate (0 PROMISING).
These tests pin (a) that a noise search yields a positive HoF ceiling with zero survivors at a small
budget, and (b) the record-vs-ceiling comparison logic, deterministically (no dependence on the
stochastic full-budget numbers, which are the artifact's job, not the test's).
"""
from __future__ import annotations

import numpy as np

import scripts.research.crucible_matched_null as mn
from finrl_pro_ds.signals.generation.fitness import FitnessConfig


def test_zero_sharpe_base_is_near_zero_sharpe() -> None:
    """The base book is ~0 Sharpe (the un-sign-sealed regime) — else marginal_t would be sign-sealed
    negative (F1/F3) and the positive-t noise ceiling would not appear."""
    base = mn._zero_sharpe_base(3000, seed=1)
    for v in base.values():
        ann_sharpe = float(np.mean(v) / np.std(v) * np.sqrt(252))
        assert abs(ann_sharpe) < 0.6, ann_sharpe          # ~0 Sharpe (sampling noise on a 0-mean series)


def test_noise_search_gives_a_positive_ceiling_and_zero_survivors() -> None:
    """A small-budget noise search still yields a POSITIVE HoF-max ΔSR (the ceiling exists on pure
    noise) and finite marginal_t, while 0 candidates survive the binding gate (PROMISING) — the core
    of the matched-null argument, at a budget cheap enough for CI."""
    cfg = FitnessConfig(embargo=10)
    s = mn.run_matched_null(cfg, t=700, n=10, pop_size=20, n_generations=3, n_seeds=2, out_path=None)
    assert s["n_seeds_done"] == 2
    assert {"seed", "gen_n_total", "hof_max_delta_sr", "hof_max_marginal_t", "n_promising",
            "elapsed_s"} <= set(s["rows"][0])
    assert s["noise_ceiling"]["max_delta_sr"] > 0.3           # noise best-of-N is a positive ΔSR
    assert np.isfinite(s["noise_ceiling"]["max_marginal_t"])
    assert s["noise_ceiling"]["total_promising"] == 0         # nothing survives the binding gate


def test_summary_record_comparison_logic() -> None:
    """The record-vs-ceiling booleans are computed correctly in BOTH directions (deterministic, no
    evolve): a high ceiling puts the real record at/below it; a low ceiling does not."""
    hi = [{"seed": 0, "gen_n_total": 100, "hof_max_delta_sr": 0.95, "hof_max_marginal_t": 2.90,
           "n_promising": 0, "elapsed_s": 1.0},
          {"seed": 1, "gen_n_total": 100, "hof_max_delta_sr": 0.80, "hof_max_marginal_t": 2.50,
           "n_promising": 0, "elapsed_s": 1.0}]
    s = mn._summary(hi, {"pop_size": 200})
    assert s["noise_ceiling"]["max_delta_sr"] == 0.95
    assert s["noise_ceiling"]["max_marginal_t"] == 2.90
    assert s["record_at_or_below_ceiling"]["delta_sr"] is True     # 0.99 <= 0.95 + 0.05
    assert s["record_at_or_below_ceiling"]["marginal_t"] is True   # 2.12 <= 2.90

    lo = [{"seed": 0, "gen_n_total": 10, "hof_max_delta_sr": 0.30, "hof_max_marginal_t": 1.00,
           "n_promising": 0, "elapsed_s": 1.0}]
    s2 = mn._summary(lo, {"pop_size": 20})
    assert s2["record_at_or_below_ceiling"]["delta_sr"] is False   # 0.99 > 0.30 + 0.05
    assert s2["record_at_or_below_ceiling"]["marginal_t"] is False  # 2.12 > 1.00
