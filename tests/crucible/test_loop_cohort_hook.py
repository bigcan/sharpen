"""Phase 4 Step 4 — the cohort hook in run_hypothesis_loop.

Load-bearing: (1) DISABLED (default) is a no-op AND leaves the manifest byte-identical to the
pre-cohort path (extra stays {}); (2) ENABLED runs the gate on the tick's overlay pool, attaches a
CohortCard + manifest provenance, is deterministic, and is null-safe on noise (not PROMISING).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sharpen.crucible import TrialLedger
from sharpen.crucible.agentic import (
    HypothesisAuthor,
    LibrarySeedProposer,
    run_hypothesis_loop,
)
from sharpen.signals.features import Panel
from sharpen.signals.generation.cohort import CohortConfig
from sharpen.signals.generation.fitness import FitnessConfig

T, N = 480, 12
_CFG = FitnessConfig(embargo=10)
_TS = "2026-07-03T00:00:00+00:00"
_EK = dict(rng_seed=7, pop_size=16, n_generations=2, hold_horizon=21, cost_bps=0.001,
           ls_min_names=6, holdout_frac=0.25, holdout_embargo=21, elite_frac=0.3)
_CCFG = CohortConfig(max_cohort_size=12, min_cohort_size=3, max_pairwise_corr=0.35,
                     promising_dsr=0.90, cohort_hlz_t_min=3.0, min_book_uplift=0.10,
                     combiner_redundancy_strength=0.5)
_MC = {"enabled": True, "n_reps": 80, "alpha_cohort": 0.05, "block_length": 21}


def _panel() -> Panel:
    """Noise OHLCV panel carrying 5 mutually-INDEPENDENT macro feature slots (so the overlay pool has
    genuine de-correlated breadth and a cohort can actually form)."""
    rng = np.random.default_rng(0)
    close = np.exp(np.cumsum(0.01 * rng.standard_normal((T, N)), axis=0) + 4.0)
    open_ = close * (1 + 0.001 * rng.standard_normal((T, N)))
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    vol = rng.uniform(1e6, 1e8, (T, N))
    dates = (np.datetime64("2014-01-02") + np.arange(T) * np.timedelta64(1, "D")
             ).astype("datetime64[ns]")
    slots = {f"macro:X{i:02d}": np.cumsum(0.05 * rng.standard_normal(T)).astype(np.float64)
             for i in range(5)}
    return Panel(dates, tuple(f"E{i:02d}" for i in range(N)), open_, high, low, close, vol,
                 np.ones((T, N), bool), close * vol, rng.integers(0, 4, size=N),
                 {"survivorship_free": True, "source": "synthetic"}, feature_slots=slots)


def _base_ts() -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(1)
    base = {"tsmom": (0.0004 + 0.006 * rng.standard_normal(T)).astype(np.float64),
            "rates_carry": (0.0003 + 0.008 * rng.standard_normal(T)).astype(np.float64)}
    ts = pd.date_range("2014-01-02", periods=T, freq="B").view("int64").astype(np.float64) / 1e9
    return base, ts


def _run(tmp_path, db, **cohort_kw):
    ledger = TrialLedger(tmp_path / db)
    author = HypothesisAuthor(LibrarySeedProposer(), ledger)
    panel = _panel()
    base, ts = _base_ts()
    return run_hypothesis_loop(
        panel=panel, base_returns=base, timestamps=ts, cfg=_CFG, evolve_kwargs=dict(_EK),
        author=author, run_id="hyp-cohort", crucible_version="crucible-v2.7", gates_hash="ffff",
        proposal_ts=_TS, data_snapshot_hash="snap", **cohort_kw)


def test_cohort_disabled_is_noop_and_manifest_byte_identical(tmp_path) -> None:
    """No cohort params vs an explicitly-disabled cohort config → both no-op, and BOTH produce a
    manifest byte-identical to the pre-cohort loop (extra == {})."""
    r_absent = _run(tmp_path, "a.db")
    r_off = _run(tmp_path, "b.db", cohort_cfg=_CCFG,
                 cohort_mc_kwargs={**_MC, "enabled": False}, cohort_gates_hash="cccc")
    assert r_absent.cohort_cards == [] and r_off.cohort_cards == []
    assert r_absent.manifest.extra == {} and r_off.manifest.extra == {}
    assert r_absent.manifest.content_hash() == r_off.manifest.content_hash()   # byte-identical


def test_cohort_enabled_attaches_card_and_manifest_provenance(tmp_path) -> None:
    r = _run(tmp_path, "c.db", cohort_cfg=_CCFG, cohort_mc_kwargs=dict(_MC),
             cohort_gates_hash="dd563b4b4f7c")
    assert len(r.cohort_cards) == 1                          # a cohort formed from the 5-slot pool
    card = r.cohort_cards[0]
    assert card.n_members >= _CCFG.min_cohort_size
    assert card.funnel_gates_hash == "ffff" and card.cohort_gates_hash == "dd563b4b4f7c"
    assert card.incubation_status == "PENDING_P4" and card.eligible_for_human_gate is False
    # cohort provenance pinned into the reproduce contract (typed manifest fields, not `extra`)
    assert r.manifest.extra == {}
    assert r.manifest.cohort_gates_hash == "dd563b4b4f7c"
    assert r.manifest.cohort_verdicts == {card.cohort_hash: card.verdict}
    assert r.manifest.cohort_card_hashes == {card.cohort_hash: card.content_hash()}


def test_cohort_enabled_is_deterministic_and_null_safe(tmp_path) -> None:
    r1 = _run(tmp_path, "d1.db", cohort_cfg=_CCFG, cohort_mc_kwargs=dict(_MC), cohort_gates_hash="cc")
    r2 = _run(tmp_path, "d2.db", cohort_cfg=_CCFG, cohort_mc_kwargs=dict(_MC), cohort_gates_hash="cc")
    # NaN-aware structural equality (a LOGGED card carries NaN MC/holdout fields when a stage
    # short-circuits; dataclass == fails on NaN, but the run IS byte-deterministic — Doc 2 §6).
    np.testing.assert_equal(r1.cohort_cards[0].to_json(), r2.cohort_cards[0].to_json())
    assert r1.cohort_cards[0].verdict != "PROMISING"          # null-safety on a pure-noise pool
