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
    # The combiner itself is still inverse-vol with the ADR-C1-5 penalty deferred (so a duplicate
    # sleeve CONCENTRATES, Δ>0 not ≈0), but the fitness now carries an explicit collinearity hurdle
    # (GP4-01): a candidate whose max |corr| to a base sleeve exceeds max_base_corr is rejected, so
    # a redundant rediscovery cannot earn PROMOTION by concentration.
    base, ts = _base(), _timestamps()
    cand = base["tsmom"].copy()                          # exact duplicate of an existing sleeve
    r = combination_fitness(cand, base, ts, _CFG, gen_n_eff=50.0, turnover_ann=5.0, n_nodes=5)
    assert r.n_paths == 15
    assert np.isfinite(r.delta_sr_oos)                   # finite, no crash (concentration effect)
    assert r.max_base_corr_obs > _CFG.max_base_corr      # collinearity detected
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


# --------------------------------------------------------------------------- #
# POSITIVE CONTROL (GP4-04): the gate MUST be able to fire. Without this, "0 PROMISING" is
# unfalsifiable — a gate that returns empty for everything passes every negative test. A genuine,
# strong, LOW-CORRELATION diversifier must clear the SHIPPED gate; a collinear or noise candidate
# in the SAME realistic regime must not. This is the discrimination proof the Tier-2 demanded.
_T_LONG = 3000


def _long_ts() -> np.ndarray:
    idx = pd.date_range("2008-01-02", periods=_T_LONG, freq="B")
    return idx.view("int64").astype(np.float64) / 1e9


def _long_base() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    tsmom = 0.30 / np.sqrt(252) * 0.010 + 0.010 * rng.standard_normal(_T_LONG)   # ann SR ~0.30
    rates = 0.45 / np.sqrt(252) * 0.008 + 0.008 * rng.standard_normal(_T_LONG)   # ann SR ~0.45
    return {"tsmom": tsmom, "rates_carry": rates}


def _shipped_cfg() -> FitnessConfig:
    from pathlib import Path

    from finrl_pro_ds.signals.generation.config import load_generation_config
    root = Path(__file__).resolve().parents[2]
    cfg, _ = load_generation_config(root / "configs" / "signal_eval.gates.yaml")
    return cfg


def _search_pool() -> list[float]:
    # A realistic file-drawer population of augmented-book per-period Sharpes: most scored genomes
    # are weak, so the book stays near the base level (~0.02 per-period ≈ ann 0.32) with a modest
    # spread — the dispersion a best-of-N winner must beat.
    rng = np.random.default_rng(11)
    return (0.020 + 0.006 * rng.standard_normal(200)).tolist()


def test_genuine_diversifier_passes_gate_positive_control() -> None:
    cfg, ts, base = _shipped_cfg(), _long_ts(), _long_base()
    rng = np.random.default_rng(303)
    # strong, INDEPENDENT (low-corr) diversifier — ann SR ~5, daily vol like the bases
    cand = 5.0 / np.sqrt(252) * 0.009 + 0.009 * rng.standard_normal(_T_LONG)
    r = combination_fitness(cand, base, ts, cfg, gen_n_eff=4495.0, turnover_ann=4.0, n_nodes=8,
                            trial_sharpe_pool=_search_pool())
    assert r.max_base_corr_obs <= cfg.max_base_corr      # genuinely diversifying
    assert r.cand_hlz_pass                               # marginal contribution significant
    assert r.dsr_aug >= cfg.promising_dsr                # clears deflation vs the search pool
    assert r.delta_sr_oos >= cfg.min_combination_uplift  # economically meaningful uplift
    assert r.passes_gate                                 # THE gate CAN fire (not dead)


def test_span_collinear_rejected_where_max_single_corr_would_miss() -> None:
    """Negative tripwire for the base-SPAN multiple-correlation upgrade (GP4-01 / Math MATH-CORR-01):
    a candidate that is a mix of 3 near-orthogonal sleeves is collinear with the SPAN (R²≈1) yet has
    only ~1/√3≈0.58 corr to each SINGLE sleeve — a max-over-single hurdle would NOT fire, but the
    span guard does. Fails if the span guard is reverted to max-single."""
    cfg, ts = _shipped_cfg(), _long_ts()
    rng = np.random.default_rng(55)
    a = 0.010 * rng.standard_normal(_T_LONG)
    b = 0.010 * rng.standard_normal(_T_LONG)
    c = 0.010 * rng.standard_normal(_T_LONG)
    base = {"s_a": 0.0003 + a, "s_b": 0.0003 + b, "s_c": 0.0003 + c}
    cand = a + b + c                                     # exactly in the 3-sleeve span
    r = combination_fitness(cand, base, ts, cfg, gen_n_eff=4495.0, turnover_ann=4.0,
                            n_nodes=8, trial_sharpe_pool=_search_pool())
    max_single = max(abs(np.corrcoef(cand, base[k])[0, 1]) for k in base)
    assert max_single < cfg.max_base_corr               # a max-single hurdle would MISS this
    assert r.max_base_corr_obs > cfg.max_base_corr       # the span multiple-corr CATCHES it
    assert not r.passes_gate


def test_p05_fragility_veto_wired() -> None:
    """GP4-06: the 5th-percentile path ΔSR (delta_sr_p05) is wired into the gate. A candidate that
    PASSES with the shipped floor is VETOED once delta_p05_min is raised above its own p05 —
    confirming the fragility veto is load-bearing, not decorative."""
    cfg, ts, base = _shipped_cfg(), _long_ts(), _long_base()
    rng = np.random.default_rng(303)
    cand = 5.0 / np.sqrt(252) * 0.009 + 0.009 * rng.standard_normal(_T_LONG)
    r = combination_fitness(cand, base, ts, cfg, gen_n_eff=4495.0, turnover_ann=4.0, n_nodes=8,
                            trial_sharpe_pool=_search_pool())
    assert r.passes_gate and np.isfinite(r.delta_sr_p05)
    from dataclasses import replace
    strict = replace(cfg, delta_p05_min=r.delta_sr_p05 + 1.0)   # floor now above the candidate's p05
    r2 = combination_fitness(cand, base, ts, strict, gen_n_eff=4495.0, turnover_ann=4.0, n_nodes=8,
                             trial_sharpe_pool=_search_pool())
    assert not r2.passes_gate                                   # vetoed purely on fragility


def test_dsr_monotone_in_trials_and_observed() -> None:
    """GP7-08: the deflated Sharpe is monotone — stricter with more trials, looser with a higher
    observed Sharpe. Locks the DSR primitive's two key directions (a sign/inequality flip fails)."""
    from finrl_pro_ds.crypto.eval.statistics import deflated_sharpe_ratio
    pool = (0.02 + 0.01 * np.random.default_rng(1).standard_normal(300)).tolist()
    kw = dict(n_obs=1000, skew=0.0, excess_kurt=0.0, periods_per_year=1)
    d_few = deflated_sharpe_ratio(0.06, pool, n_trials=10, **kw)
    d_many = deflated_sharpe_ratio(0.06, pool, n_trials=2000, **kw)
    assert d_few["dsr"] >= d_many["dsr"]                        # more trials → stricter
    d_lo = deflated_sharpe_ratio(0.04, pool, n_trials=100, **kw)
    d_hi = deflated_sharpe_ratio(0.10, pool, n_trials=100, **kw)
    assert d_hi["dsr"] >= d_lo["dsr"]                           # higher observed → higher DSR


def test_dsr_aug_tightens_with_more_trials() -> None:
    """GP5-02 N-bite tripwire at the generation wiring: the SAME candidate/book deflates HARDER
    (lower dsr_aug) at a larger gen_n_eff. Fails if the file-drawer-N → deflation link is inverted
    or hardcoded."""
    cfg, ts, base = _shipped_cfg(), _long_ts(), _long_base()
    rng = np.random.default_rng(303)
    cand = 5.0 / np.sqrt(252) * 0.009 + 0.009 * rng.standard_normal(_T_LONG)
    pool = _search_pool()
    lo = combination_fitness(cand, base, ts, cfg, gen_n_eff=5.0, turnover_ann=4.0, n_nodes=8,
                             trial_sharpe_pool=pool)
    hi = combination_fitness(cand, base, ts, cfg, gen_n_eff=5000.0, turnover_ann=4.0, n_nodes=8,
                             trial_sharpe_pool=pool)
    assert lo.dsr_aug >= hi.dsr_aug                       # more trials → stricter deflation


def test_cpcv_right_seam_purged() -> None:
    """GP4-05: the right-seam purge (purge_horizon=1) drops boundary indices whose forward return
    reaches into a train group. Each purged path is a strict subset of the un-purged path, and at
    least one index is removed overall. Fails if the right purge is reverted to none."""
    from finrl_pro_ds.signals.generation.fitness import _cpcv_index_paths
    k_len, n_groups, k_test, embargo = 120, 6, 2, 5
    no_purge = _cpcv_index_paths(k_len, n_groups, k_test, embargo, 0)
    purged = _cpcv_index_paths(k_len, n_groups, k_test, embargo, 1)
    assert len(no_purge) == len(purged)                        # same combinatorial path count
    assert sum(len(a) - len(b) for a, b in zip(no_purge, purged)) > 0   # right-seam indices removed
    for a, b in zip(no_purge, purged):
        assert set(int(x) for x in b).issubset(set(int(x) for x in a))  # purge only ever removes


def test_gate_discriminates_collinear_and_noise_in_same_regime() -> None:
    cfg, ts, base = _shipped_cfg(), _long_ts(), _long_base()
    pool = _search_pool()
    # (a) a STRONG candidate collinear with tsmom (corr≈1) — must be rejected by the corr hurdle
    collinear = base["tsmom"] * 3.0
    rc = combination_fitness(collinear, base, ts, cfg, gen_n_eff=4495.0, turnover_ann=4.0,
                             n_nodes=8, trial_sharpe_pool=pool)
    assert rc.max_base_corr_obs > cfg.max_base_corr and not rc.passes_gate
    # (b) pure noise — no marginal contribution → rejected
    rng = np.random.default_rng(404)
    noise = 0.009 * rng.standard_normal(_T_LONG)
    rn = combination_fitness(noise, base, ts, cfg, gen_n_eff=4495.0, turnover_ann=4.0,
                             n_nodes=8, trial_sharpe_pool=pool)
    assert not rn.passes_gate
