"""crucible-v15.0 tripwires for the corrected contract's degenerate-candidate channels (deep audit
2026-09-30). Every test FAILS on the pre-v15 scorer.

The combiner falls back to EQUAL weight for every sleeve on any bar where one sleeve's trailing vol is
unusable. With the candidate in the sleeve set, a candidate that is unusable on some bars (warm-up, a
zero-return stretch, constant, all-NaN) turned the augmented book into equal-weight(base) there while
the base book stayed inverse-vol — so the "marginal" statistic scored the base book's weighting, not the
candidate. Measured through the shipped scorer: a constant positive stream (cash, not alpha) reached
z = 2.7-15.6 and passed outright; an all-zero stream passed 0.3-2.3% of the time.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from sharpen.crucible.corrected_contract import (
    CorrectedConfig,
    corrected_contract_fitness,
    fresh_lord_level,
)
from sharpen.envs.allocator_factory import dynamic_sleeve_alphas
from sharpen.signals.generation import fitness as fit
from sharpen.signals.generation.fitness import FitnessConfig, _combined_book, augmented_book

ROOT = Path(__file__).resolve().parents[2]
_CC = CorrectedConfig.from_yaml(ROOT / "configs" / "crucible_corrected_contract.gates.yaml")
_CFG = FitnessConfig()
T = 1200
TS = (np.datetime64("2012-01-02", "s") + np.arange(T) * np.timedelta64(86400, "s")
      ).astype(np.int64).astype(np.float64)


def _base(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {"a": 0.0008 + 0.010 * rng.standard_normal(T), "b": 0.0002 + 0.004 * rng.standard_normal(T)}


def test_vol_floor_mirror_matches_the_combiner_default() -> None:
    floor = inspect.signature(dynamic_sleeve_alphas).parameters["vol_floor"].default
    assert fit._COMBINER_VOL_FLOOR == floor


@pytest.mark.parametrize("seed", range(4))
def test_constant_positive_stream_is_not_alpha(seed) -> None:
    cand = np.full(T, 0.001)
    r = corrected_contract_fitness(cand, _base(seed), TS, _CFG, _CC, lord_level=fresh_lord_level(_CC))
    assert not r.passes_corrected
    assert r.corrected_t == 0.0 and r.cand_usable_frac == 0.0     # never joined the book
    assert not r.degenerate_pass


def test_all_zero_candidate_leaves_the_book_untouched() -> None:
    base = _base(1)
    b_base, b_aug, usable = augmented_book(base, np.zeros(T), TS, _CFG)
    assert not usable.any()
    assert np.array_equal(b_aug, b_base)
    r = corrected_contract_fitness(np.zeros(T), base, TS, _CFG, _CC, lord_level=0.05)
    assert not r.passes_corrected and not r.degenerate_pass


def test_candidate_joins_only_where_usable_and_is_bit_identical_there() -> None:
    rng = np.random.default_rng(3)
    base = _base(2)
    cand = 0.008 * rng.standard_normal(T)
    cand[400:700] = 0.0                                       # a zero-return stretch → unusable σ
    b_base, b_aug, usable = augmented_book(base, cand, TS, _CFG)
    legacy = _combined_book({**base, fit._CAND: cand}, TS, _CFG)
    assert usable.any() and (~usable).any()
    assert np.array_equal(b_aug[~usable], b_base[~usable])      # candidate absent where unusable
    assert np.array_equal(b_aug[usable], legacy[usable])        # otherwise the combiner, verbatim


def test_eval_from_scores_only_the_evaluation_rows() -> None:
    rng = np.random.default_rng(4)
    base = _base(3)
    cand = 0.008 * rng.standard_normal(T)
    k = 900
    r = corrected_contract_fitness(cand, base, TS, _CFG, _CC, lord_level=0.02, eval_from=k)
    assert r.n_bars <= T - k
    # the train rows only SIZE the book (trailing vol); scrambling their ORDER within the window used
    # for sizing leaves the evaluated statistic finite and on the evaluation rows only
    r0 = corrected_contract_fitness(cand[k:], {n: v[k:] for n, v in base.items()}, TS[k:], _CFG, _CC,
                                    lord_level=0.02)
    assert r0.n_bars <= T - k and np.isfinite(r.corrected_t) and np.isfinite(r0.corrected_t)


def test_eval_from_zero_is_the_plain_call() -> None:
    rng = np.random.default_rng(5)
    base = _base(4)
    cand = 0.008 * rng.standard_normal(T)
    a = corrected_contract_fitness(cand, base, TS, _CFG, _CC, lord_level=0.02)
    b = corrected_contract_fitness(cand, base, TS, _CFG, _CC, lord_level=0.02, eval_from=0)
    assert repr(a) == repr(b)
