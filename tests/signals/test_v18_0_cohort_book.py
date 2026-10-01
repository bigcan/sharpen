"""The cohort path's book, after crucible-v18.0.

v17.0 replaced the raw combiner on the cohort path with a spliced book (``fitness.joined_book``:
the base book on bars where a member could not be sized). v18.0 withdrew it from the analytic floor
and the MC null: on the real us_equity pool with every member's edge destroyed, the spliced book's
null test rejected 60 of 60 no-edge panels at alpha 0.05, the raw combiner's 1 of 60 (see the note in
``cohort_mc._cohort_delta_sr``). What remains from v17.0 is the holdout guard's handling of a member
the combiner cannot size at the freeze bar, which involves no splice."""
from __future__ import annotations

import numpy as np

from sharpen.signals.generation import fitness
from sharpen.signals.generation.cohort import CohortConfig
from sharpen.signals.generation.cohort_eval import _cohort_holdout_guard
from sharpen.signals.generation.cohort_mc import _cohort_delta_sr
from sharpen.signals.generation.fitness import FitnessConfig, _combined_book, _per_period_sharpe

K = 1500


def _ts(n: int = K) -> np.ndarray:
    return np.int64(1_400_000_000) + np.arange(n, dtype=np.int64) * 86_400


def _base(rng) -> dict[str, np.ndarray]:
    return {"tsmom": 0.0004 + 0.012 * rng.standard_normal(K),
            "carry": 0.0002 + 0.003 * rng.standard_normal(K)}


def _ccfg(**kw) -> CohortConfig:
    base = dict(max_cohort_size=8, min_cohort_size=1, max_pairwise_corr=0.95,
                promising_dsr=0.90, cohort_hlz_t_min=3.0, min_book_uplift=0.10,
                combiner_redundancy_strength=0.0)
    base.update(kw)
    return CohortConfig(**base)


def test_the_spliced_book_is_gone() -> None:
    assert not hasattr(fitness, "joined_book")


def test_cohort_statistic_is_the_raw_combiner_even_with_a_dead_stretch() -> None:
    """The MC statistic books base + members through the raw combiner on EVERY bar, including bars
    where a member is dead. A splice there (v17.0) broke the observed/null symmetry."""
    rng = np.random.default_rng(2)
    base, cfg, ts = _base(rng), FitnessConfig(), _ts()
    live = 0.006 * rng.standard_normal(K)
    dead = 0.006 * rng.standard_normal(K)
    dead[500:1100] = 0.0
    cand = {"live": live, "dead": dead}
    t, members = _cohort_delta_sr(base, cand, ts, _ccfg(), cfg, cfg)
    aug = _combined_book({**base, **{m: cand[m] for m in members}}, ts, cfg)
    b_base = _combined_book(base, ts, cfg)
    expected = (_per_period_sharpe(aug[np.isfinite(aug)])
                - _per_period_sharpe(b_base[np.isfinite(b_base)]))
    assert set(members) == {"live", "dead"} and t == expected


def test_holdout_guard_freezes_an_unsized_member_at_zero_weight() -> None:
    """A member the combiner cannot size at the freeze bar gets frozen weight 0, so the guard's ΔSR
    equals the ΔSR of the cohort without it. It used to trip the equal-weight fallback at that bar and
    book the whole holdout under equal weights."""
    rng = np.random.default_rng(5)
    n_tr, n_ho = 1100, 400
    base = _base(rng)
    live = 0.0005 + 0.006 * rng.standard_normal(K)
    dead = 0.006 * rng.standard_normal(K)
    dead[n_tr - 400:n_tr] = 0.0                                    # unsizeable at the train end
    cfg, ccfg, ts = FitnessConfig(), _ccfg(), _ts()
    tr = lambda d: {k: v[:n_tr] for k, v in d.items()}             # noqa: E731
    ho = lambda d: {k: v[K - n_ho:] for k, v in d.items()}         # noqa: E731
    both = {"live": live, "dead": dead}
    only = {"live": live}
    d_both, _ = _cohort_holdout_guard(("live", "dead"), tr(base), tr(both), ts[:n_tr],
                                      ho(base), ho(both), ccfg, cfg)
    d_only, _ = _cohort_holdout_guard(("live",), tr(base), tr(only), ts[:n_tr],
                                      ho(base), ho(only), ccfg, cfg)
    assert np.isfinite(d_both) and d_both == d_only
