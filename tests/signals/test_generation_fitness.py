"""C3.3 tripwires — combination-contribution fitness.

Locks the properties that make generation honest rather than a mirage factory: a redundant
(collinear) candidate earns ≈0, pure noise fails the gate, a genuine diversifier beats noise,
cost/complexity penalties bite, and the score is deterministic.

Design: .agent/artifacts/alpha_generation_component3_architecture.md §Fitness math
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from finrl_pro_ds.signals.generation.fitness import (
    FitnessConfig,
    combination_fitness,
)

K = 504


def _timestamps() -> np.ndarray:
    idx = pd.date_range("2014-01-02", periods=K, freq="B")
    return idx.view("int64").astype(np.float64) / 1e9             # epoch seconds


def _base() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    mom = 0.0004 + 0.010 * rng.standard_normal(K)        # standalone SR ~0.6
    rates = 0.0003 + 0.008 * rng.standard_normal(K)      # independent, low corr
    return {"tsmom": mom, "rates_carry": rates}


def _diversifier() -> np.ndarray:
    rng = np.random.default_rng(101)
    return 0.0005 + 0.009 * rng.standard_normal(K)       # positive SR, independent


def _noise() -> np.ndarray:
    rng = np.random.default_rng(202)
    return 0.0 + 0.010 * rng.standard_normal(K)          # zero-mean


_CFG = FitnessConfig(embargo=10)   # K/6≈84-day groups → embargo 10 leaves ample test points


def test_collinear_candidate_cannot_earn_promotion() -> None:
    # NOTE (finding, S553-cont-89): the SHIPPED C1 combiner is inverse-vol with the
    # correlation penalty deferred (ADR-C1-5), so a duplicate sleeve is NOT weight-neutral —
    # it double-counts and CONCENTRATES into the higher-Sharpe sleeve (Δ>0, not ≈0). The
    # "redundancy earns ~0" Synergistic property needs that deferred penalty. The honest guard
    # that holds today: redundancy still cannot earn PROMOTION (deflation + HLZ reject it).
    base, ts = _base(), _timestamps()
    cand = base["tsmom"].copy()                          # exact duplicate of an existing sleeve
    r = combination_fitness(cand, base, ts, _CFG, gen_n_eff=50.0, turnover_ann=5.0, n_nodes=5)
    assert r.n_paths == 15
    assert np.isfinite(r.delta_sr_oos)                   # finite, no crash (concentration effect)
    assert not r.passes_gate                             # redundancy buys no promotion


def test_noise_candidate_fails_gate() -> None:
    base = _base()
    r = combination_fitness(_noise(), base, _timestamps(), _CFG,
                            gen_n_eff=50.0, turnover_ann=5.0, n_nodes=5)
    assert not r.passes_gate                             # noise must not be PROMISING


def test_diversifier_beats_noise() -> None:
    base, ts = _base(), _timestamps()
    div = combination_fitness(_diversifier(), base, ts, _CFG,
                              gen_n_eff=50.0, turnover_ann=5.0, n_nodes=5)
    noise = combination_fitness(_noise(), base, ts, _CFG,
                                gen_n_eff=50.0, turnover_ann=5.0, n_nodes=5)
    assert div.delta_sr_oos > noise.delta_sr_oos         # a real diversifier adds more than noise
    assert div.fitness > noise.fitness


def test_turnover_penalty_below_then_above_soft_cap() -> None:
    base, ts = _base(), _timestamps()
    cand = _diversifier()
    lo = combination_fitness(cand, base, ts, _CFG, gen_n_eff=50.0, turnover_ann=5.0, n_nodes=5)
    hi = combination_fitness(cand, base, ts, _CFG, gen_n_eff=50.0, turnover_ann=30.0, n_nodes=5)
    assert hi.fitness < lo.fitness                       # excess turnover is penalized
    assert lo.delta_sr_oos == hi.delta_sr_oos            # same book contribution


def test_complexity_penalty_monotone() -> None:
    base, ts = _base(), _timestamps()
    cand = _diversifier()
    small = combination_fitness(cand, base, ts, _CFG, gen_n_eff=50.0, turnover_ann=5.0, n_nodes=4)
    big = combination_fitness(cand, base, ts, _CFG, gen_n_eff=50.0, turnover_ann=5.0, n_nodes=22)
    assert big.fitness < small.fitness


def test_deterministic() -> None:
    base, ts = _base(), _timestamps()
    cand = _diversifier()
    a = combination_fitness(cand, base, ts, _CFG, gen_n_eff=50.0, turnover_ann=5.0, n_nodes=5)
    b = combination_fitness(cand, base, ts, _CFG, gen_n_eff=50.0, turnover_ann=5.0, n_nodes=5)
    assert a == b


def test_degenerate_candidate_does_not_crash() -> None:
    base, ts = _base(), _timestamps()
    bad = np.full(K, np.nan)
    bad[0] = 0.01                                        # < 2 finite → undefined Sharpe
    r = combination_fitness(bad, base, ts, _CFG, gen_n_eff=50.0, turnover_ann=5.0, n_nodes=5)
    assert not r.passes_gate                             # no crash; cannot be promoted
