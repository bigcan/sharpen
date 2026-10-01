"""crucible-v17.0 — the cohort books admit a sleeve only where the combiner can size it
(``fitness.joined_book``). ``combination_fitness`` keeps the raw combiner on purpose (see its note).

The v15.0 audit closed the degenerate channel on the per-candidate holdout gate and left it open on
the cohort path (``cohort.py``, ``cohort_mc.py``, ``cohort_eval.py``): one
unusable sleeve makes the raw combiner weight EVERY sleeve equally, so the "augmented" book measured
an EW-vs-inverse-vol reweighting of the others. Each test here fails on the raw combiner."""
from __future__ import annotations

import numpy as np

from sharpen.signals.generation.cohort import CohortConfig
from sharpen.signals.generation.cohort_eval import _cohort_holdout_guard
from sharpen.signals.generation.cohort_mc import _cohort_delta_sr
from sharpen.signals.generation.fitness import (
    FitnessConfig,
    _candidate_usable,
    _combined_book,
    augmented_book,
    joined_book,
)

K = 1500


def _ts(n: int = K) -> np.ndarray:
    return np.int64(1_400_000_000) + np.arange(n, dtype=np.int64) * 86_400


def _base(rng) -> dict[str, np.ndarray]:
    # Unequal vols, so equal weight and inverse-vol weight are different books.
    return {"tsmom": 0.0004 + 0.012 * rng.standard_normal(K),
            "carry": 0.0002 + 0.003 * rng.standard_normal(K)}


def _ccfg(**kw) -> CohortConfig:
    base = dict(max_cohort_size=8, min_cohort_size=1, max_pairwise_corr=0.95,
                promising_dsr=0.90, cohort_hlz_t_min=3.0, min_book_uplift=0.10,
                combiner_redundancy_strength=0.0)
    base.update(kw)
    return CohortConfig(**base)


def test_joined_book_equals_raw_combiner_where_every_member_is_usable() -> None:
    rng = np.random.default_rng(0)
    base, cfg, ts = _base(rng), FitnessConfig(), _ts()
    members = {"a": 0.006 * rng.standard_normal(K), "b": 0.009 * rng.standard_normal(K)}
    usable = np.all([_candidate_usable(v, ts, cfg) for v in members.values()], axis=0)
    assert usable.sum() > K // 2 and (~usable).sum() > 0          # warm-up exists, then all usable
    raw = _combined_book({**base, **members}, ts, cfg)
    got = joined_book(base, members, ts, cfg)
    assert np.array_equal(got[usable], raw[usable])               # bit-identical where usable
    assert np.array_equal(got[~usable], _combined_book(base, ts, cfg)[~usable])


def test_joined_book_with_one_member_is_augmented_book() -> None:
    rng = np.random.default_rng(1)
    base, cfg, ts = _base(rng), FitnessConfig(), _ts()
    cand = 0.007 * rng.standard_normal(K)
    cand[600:900] = 0.0                                            # a dead stretch mid-sample
    _, b_aug, _ = augmented_book(base, cand, ts, cfg)
    assert np.array_equal(joined_book(base, {"_candidate": cand}, ts, cfg), b_aug)


def test_a_dead_member_drops_out_instead_of_equal_weighting_the_book() -> None:
    """A member with a zero-return stretch leaves the book for that stretch; the other member stays
    sized. Under the raw combiner those bars were equal weight across base AND both members."""
    rng = np.random.default_rng(2)
    base, cfg, ts = _base(rng), FitnessConfig(), _ts()
    live = 0.006 * rng.standard_normal(K)
    dead = 0.006 * rng.standard_normal(K)
    dead[500:1100] = 0.0
    dead_off = ~_candidate_usable(dead, ts, cfg) & _candidate_usable(live, ts, cfg)
    assert dead_off.sum() > 200
    got = joined_book(base, {"live": live, "dead": dead}, ts, cfg)
    without = _combined_book({**base, "live": live}, ts, cfg)
    raw = _combined_book({**base, "live": live, "dead": dead}, ts, cfg)
    assert np.array_equal(got[dead_off], without[dead_off])
    assert not np.allclose(raw[dead_off], without[dead_off])       # the channel the raw book carried


def test_an_all_zero_cohort_adds_exactly_nothing() -> None:
    rng = np.random.default_rng(3)
    base, cfg, ts = _base(rng), FitnessConfig(), _ts()
    zero = {"z": np.zeros(K)}
    t, members = _cohort_delta_sr(base, zero, ts, _ccfg(), cfg, cfg)
    assert members == ("z",)
    assert t == 0.0
    # ...and the raw combiner did not say zero: equal-weight(base, z) vs inverse-vol(base).
    b_base = _combined_book(base, ts, cfg)
    assert not np.allclose(_combined_book({**base, **zero}, ts, cfg), b_base)


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
