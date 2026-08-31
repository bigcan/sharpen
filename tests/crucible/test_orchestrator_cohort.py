"""Phase 4 Step 5 — the cohort fold in run_orchestrator_tick: outcome fold, ADR-4 FDR charge,
card writes, manifest provenance. Disabled ⇒ byte-identical pre-cohort behavior (no +1 FDR test).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sharpen.crucible import (
    OrchestratorStore,
    Substrate,
    TrialLedger,
    run_orchestrator_tick,
)
from sharpen.crucible.orchestrator.substrate import PreparedSubstrate
from sharpen.signals.features import Panel
from sharpen.signals.generation.cohort import CohortConfig
from sharpen.signals.generation.fitness import FitnessConfig

GATES = "configs/signal_eval.gates.yaml"
T, N = 480, 12
_EK = dict(rng_seed=7, pop_size=16, n_generations=2, hold_horizon=21, cost_bps=0.001,
           ls_min_names=6, holdout_frac=0.25, holdout_embargo=21, elite_frac=0.3)
_CCFG = CohortConfig(max_cohort_size=12, min_cohort_size=3, max_pairwise_corr=0.35,
                     promising_dsr=0.90, cohort_hlz_t_min=3.0, min_book_uplift=0.10,
                     combiner_redundancy_strength=0.5)
_MC = {"enabled": True, "n_reps": 60, "alpha_cohort": 0.05, "block_length": 21}


def _panel() -> Panel:
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


def _make_substrate(tmp_path, name: str, *, cohort: bool) -> Substrate:
    def prepare() -> PreparedSubstrate:
        panel = _panel()
        base, ts = _base_ts()
        return PreparedSubstrate(panel=panel, base_returns=base, timestamps=ts,
                                 asset_classes=("macro",), snapshot_hash="snap-cohort")
    ledger = TrialLedger(tmp_path / f"{name}_ledger.db")
    kw = {}
    if cohort:
        kw = dict(cohort_cfg=_CCFG, cohort_mc_kwargs=dict(_MC), cohort_gates_hash="dd563b4b4f7c")
    return Substrate(substrate_id=name, prepare=prepare, ledger=ledger,
                     cfg=FitnessConfig(embargo=10), evolve_kwargs=dict(_EK), **kw)


@pytest.mark.slow
def test_cohort_tick_folds_outcome_charges_fdr_and_writes_card(tmp_path) -> None:
    store = OrchestratorStore(tmp_path / "orch.db")
    sub = _make_substrate(tmp_path, "syn_on", cohort=True)
    res = run_orchestrator_tick(substrates=[sub], store=store, gates_path=GATES,
                                tick_ts="2026-07-04T00:00:00+00:00", out_dir=tmp_path / "out")
    o = res.outcomes[0]
    assert o.mined and len(o.cohort_cards) == 1              # a cohort formed from the 5-slot pool
    # ADR-4: exactly ONE extra online-FDR test beyond the per-candidate charges (the cohort test).
    assert o.fdr_num_tests == o.n_preregistered + 1
    assert o.n_cohort_promising == sum(1 for c in o.cohort_cards if c.verdict == "PROMISING")
    # manifest provenance (pinned typed fields) + card written to disk beside the discovery cards.
    card = o.cohort_cards[0]
    assert o.result.manifest.cohort_gates_hash == "dd563b4b4f7c"
    assert o.result.manifest.cohort_verdicts == {card.cohort_hash: card.verdict}
    assert o.result.manifest.cohort_card_hashes == {card.cohort_hash: card.content_hash()}
    written = tmp_path / "out" / "syn_on" / "2026-07-04T00_00_00_00_00" / "cards" / f"cohort_{card.cohort_hash}.json"
    assert written.exists()


@pytest.mark.slow
def test_cohort_disabled_charges_no_extra_fdr_test(tmp_path) -> None:
    """ADR-4 isolation: with no cohort config the tick charges exactly the per-candidate tests — the
    +1 cohort test appears ONLY when a cohort is evaluated (byte-identical pre-cohort FDR accounting)."""
    store = OrchestratorStore(tmp_path / "orch.db")
    sub = _make_substrate(tmp_path, "syn_off", cohort=False)
    res = run_orchestrator_tick(substrates=[sub], store=store, gates_path=GATES,
                                tick_ts="2026-07-04T00:00:00+00:00")
    o = res.outcomes[0]
    assert o.mined and o.cohort_cards == [] and o.n_cohort_promising == 0
    assert o.fdr_num_tests == o.n_preregistered          # no +1 cohort test
