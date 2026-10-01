"""crucible-v15.0 robustness tripwires (deep audit 2026-09-30). Each fails on the pre-v15 code."""
from __future__ import annotations

import logging

from sharpen.crucible.agentic.jev_ranker import RankedProposer
from sharpen.crucible.agentic.llm_proposer import LlmProposer
from sharpen.crucible.agentic.proposer import HypothesisProposal, ProposalContext
from sharpen.crucible.manifest import RunManifest
from sharpen.crucible.orchestrator.fdr import OnlineFDR
from sharpen.crucible.orchestrator.substrate import OrchestratorStore
from sharpen.crucible.reproduce import compare_manifests


def test_reproduce_pins_the_decision_function() -> None:
    a = RunManifest(run_id="r", contract="corrected", corrected_gates_hash="aaa")
    for drift in (dict(contract="shipped"), dict(corrected_gates_hash="bbb"),
                  dict(search_memory_gates_hash="ccc")):
        b = RunManifest(run_id="r", **{"contract": "corrected", "corrected_gates_hash": "aaa", **drift})
        rep = compare_manifests(a, b)
        assert not rep.ok and not rep.pins_match, drift


def test_persisted_fdr_account_drift_is_reported(tmp_path, caplog) -> None:
    store = OrchestratorStore(tmp_path / "o.db")
    store.save_fdr("s", OnlineFDR(alpha=0.10))
    with caplog.at_level(logging.WARNING, logger="crucible.orchestrator"):
        acct = store.load_fdr("s", alpha=0.05)
    assert acct.alpha == 0.10                                   # the stream is kept …
    assert any("differs from the configured" in r.message for r in caplog.records)   # … but flagged


def test_llm_proposer_survives_a_non_list_payload() -> None:
    def transport(url, headers, body):
        return {"content": [{"type": "tool_use", "name": "propose_hypotheses",
                             "input": {"proposals": {"not": "a list"}}}], "usage": {}}
    assert LlmProposer(api_key="k", transport=transport).propose(ProposalContext()) == []

    def transport2(url, headers, body):
        return {"content": [{"type": "tool_use", "name": "propose_hypotheses",
                             "input": {"proposals": [42, "x"]}}], "usage": {}}
    assert LlmProposer(api_key="k", transport=transport2).propose(ProposalContext()) == []


class _Inner:
    model_id = "inner"

    def propose(self, context):
        return [HypothesisProposal("a", "h", "101alpha", 1, "cross_sectional", "rank(close)")]


class _DeadRanker:
    model_id = "jev-dead"
    last_scores: dict = {}

    def rank(self, pool, context):
        self.last_scores = {}
        return list(pool)                                        # identity fallback


def test_jev_identity_fallback_is_visible_in_the_provenance() -> None:
    rp = RankedProposer(_Inner(), _DeadRanker())                 # type: ignore[arg-type]
    rp.propose(ProposalContext(max_proposals=4))
    assert rp.model_id.endswith("+identity-fallback")


def test_cohort_gate_failure_does_not_abort_the_tick(tmp_path, monkeypatch) -> None:
    import numpy as np
    import pandas as pd

    from sharpen.crucible.agentic import loop as loop_mod
    from sharpen.crucible.agentic.hypothesis import HypothesisAuthor
    from sharpen.crucible.ledger import TrialLedger
    from sharpen.signals.features import Panel
    from sharpen.signals.generation.cohort import CohortConfig
    from sharpen.signals.generation.fitness import FitnessConfig

    def boom(**kw):
        raise RuntimeError("cohort exploded")

    monkeypatch.setattr(loop_mod, "_evaluate_cohort_gate", boom)
    t, n = 400, 8
    rng = np.random.default_rng(0)
    close = np.exp(np.cumsum(0.01 * rng.standard_normal((t, n)), axis=0) + 4.0)
    dates = (np.datetime64("2014-01-02") + np.arange(t) * np.timedelta64(1, "D")).astype("datetime64[ns]")
    panel = Panel(dates, tuple(f"E{i}" for i in range(n)), close, close, close, close,
                  rng.uniform(1e6, 1e8, (t, n)), np.ones((t, n), bool), close * 1e6,
                  rng.integers(0, 3, size=n), {}, feature_slots={"macro:x": rng.standard_normal(t)})
    base = {"a": 0.0004 + 0.006 * rng.standard_normal(t), "b": 0.0003 + 0.008 * rng.standard_normal(t)}
    ts = pd.date_range("2014-01-02", periods=t, freq="B").view("int64").astype(np.float64) / 1e9

    class _P:
        model_id = "p"

        def propose(self, ctx):
            return [HypothesisProposal("o", "h", "altdata", 1, "overlay", "macro:x")]

    author = HypothesisAuthor(_P(), TrialLedger(tmp_path / "l.db"))
    cc = CohortConfig(max_cohort_size=12, min_cohort_size=2, max_pairwise_corr=0.35,
                      promising_dsr=0.9, cohort_hlz_t_min=3.0, min_book_uplift=0.1)
    res = loop_mod.run_hypothesis_loop(
        panel=panel, base_returns=base, timestamps=ts, cfg=FitnessConfig(embargo=10),
        evolve_kwargs=dict(rng_seed=1, pop_size=4, n_generations=1, hold_horizon=21,
                           cost_bps=0.001, ls_min_names=4, holdout_frac=0.25, holdout_embargo=21),
        author=author, run_id="tick-s-1", crucible_version="t", gates_hash="g", proposal_ts="1",
        cohort_cfg=cc, cohort_mc_kwargs={"enabled": True})
    assert res.cohort_cards == []                               # failed cohort → no card, no abort
