"""crucible-v15.0 tripwires for the hypothesis-supply layer (deep audit 2026-09-30).

  * The library proposer truncated at ``max_proposals`` BEFORE the Author deduplicated, so with the
    extended WQ101 bank every tick after the first re-emitted the same 32 spent formulas and the other 77
    were unreachable.
  * Statistical duplicates (``rank(x)`` vs ``x`` vs ``scale(x)`` vs ``3*x`` …) produce the identical book
    but were separate hypotheses, each ledgered and LORD++-charged.
  * Per-name (T,N) slots only ever received OVERLAY templates, which average the cross-section away —
    under prereg_only a per-name cross-sectional signal could never be promoted.
"""
from __future__ import annotations

import sqlite3

from sharpen.crucible.agentic.hypothesis import HypothesisAuthor
from sharpen.crucible.agentic.llm_proposer import _render_context
from sharpen.crucible.agentic.proposer import (
    HypothesisProposal,
    LibrarySeedProposer,
    ProposalContext,
)
from sharpen.crucible.ledger import TrialLedger, TrialRecord
from sharpen.crucible.search_memory import candidate_hash, statistical_hash

_TS = "2026-09-30T00:00:00+00:00"


class _Emit:
    model_id = "test-emit"

    def __init__(self, props):
        self._p = props

    def propose(self, context):
        return list(self._p)


def test_extended_bank_advances_past_registered_formulas(tmp_path) -> None:
    ledger = TrialLedger(tmp_path / "l.db")
    prop = LibrarySeedProposer(include_overlay=False, extended_cs_bank=True)
    author = HypothesisAuthor(prop, ledger, max_proposals=32)
    first = author.propose(author.build_context(("close",)), proposal_ts=_TS)
    assert len(first) == 32
    for s in first:                                          # scored → a verdict → dedup keys
        ledger.record(TrialRecord(candidate_hash=s.candidate_hash, crucible_version="t",
                                  candidate_type="cross_sectional", formula=s.formula,
                                  verdict="SCORED_NOT_SELECTED"))
    second = author.propose(author.build_context(("close",)), proposal_ts=_TS)
    assert len(second) == 32                                 # pre-v15: 0 — the cap came before dedup
    assert not ({s.candidate_hash for s in first} & {s.candidate_hash for s in second})


def test_fresh_ledger_batch_is_unchanged_by_the_prefilter() -> None:
    ctx = ProposalContext(available_terminals=("close", "macro:x"), max_proposals=64)
    a = LibrarySeedProposer(extended_cs_bank=True).propose(ctx)
    assert len(a) == 64 and a[0].name == "cs-ret-reversal"


def test_statistical_duplicates_are_one_hypothesis(tmp_path) -> None:
    ledger = TrialLedger(tmp_path / "l.db")
    ledger.record(TrialRecord(candidate_hash=candidate_hash("adv60"), crucible_version="t",
                              candidate_type="cross_sectional", formula="adv60", verdict="LOGGED"))
    props = [HypothesisProposal(n, "h", "101alpha", 1, "cross_sectional", f) for n, f in (
        ("r", "rank(adv60)"), ("s", "rank(scale(adv60))"), ("k", "3 * adv60"),
        ("mirror", "-1 * rank(adv60)"), ("mirror2", "(-2) * adv60"))]
    author = HypothesisAuthor(_Emit(props), ledger)
    specs = author.propose(author.build_context(("adv60",)), proposal_ts=_TS)
    assert [s.spec.name for s in specs] == ["mirror"]       # the mirror is a DIFFERENT hypothesis, once


def test_overlay_statistical_key_respects_the_timing_transform() -> None:
    assert statistical_hash("(2 * macro:x)", "overlay") == statistical_hash("macro:x", "overlay")
    assert statistical_hash("rank(macro:x)", "overlay") != statistical_hash("macro:x", "overlay")
    assert statistical_hash("macro:x", "overlay") != statistical_hash("macro:x", "cross_sectional")


def test_stat_hash_backfill_migrates_an_old_ledger(tmp_path) -> None:
    db = tmp_path / "old.db"
    TrialLedger(db).record(TrialRecord(candidate_hash="h1", crucible_version="t",
                                       candidate_type="cross_sectional", formula="rank(adv60)",
                                       verdict="LOGGED"))
    con = sqlite3.connect(db)                                # simulate a pre-v15 ledger
    con.executescript("DROP INDEX ix_trial_stat; DROP VIEW ledger_agent_view; "
                      "ALTER TABLE trial_ledger DROP COLUMN stat_hash;")
    con.commit()
    con.close()
    led = TrialLedger(db)
    row = led._conn.execute("SELECT stat_hash FROM trial_ledger WHERE candidate_hash='h1'").fetchone()
    assert row[0] == statistical_hash("adv60", "cross_sectional")
    assert statistical_hash("adv60") in led.agent_view()["stat_hashes"]


def test_per_name_slots_get_cross_sectional_templates() -> None:
    ctx = ProposalContext(available_terminals=("close", "twse:flow", "macro:x"),
                          per_name_slots=("twse:flow",), max_proposals=64)
    props = LibrarySeedProposer().propose(ctx)
    xs = [p for p in props if p.candidate_type == "cross_sectional" and "twse:flow" in p.formula]
    assert {p.formula for p in xs} == {"rank(twse:flow)", "rank(delta(twse:flow, 20))",
                                       "rank(decay_linear(twse:flow, 10))"}
    assert any(p.candidate_type == "overlay" and "twse:flow" in p.formula for p in props)
    none = LibrarySeedProposer().propose(ProposalContext(
        available_terminals=("close", "twse:flow"), max_proposals=64))
    assert not any(p.candidate_type == "cross_sectional" and "twse:flow" in p.formula for p in none)


def test_llm_render_names_per_name_slots_and_is_unchanged_without_them() -> None:
    base = ProposalContext(available_terminals=("close", "fred:DGS10"))
    assert "Available non-OHLCV feature slots (overlay-eligible only): fred:DGS10" in _render_context(base)
    rich = ProposalContext(available_terminals=("close", "fred:DGS10", "twse:flow"),
                           per_name_slots=("twse:flow",))
    msg = _render_context(rich)
    assert "PER-NAME feature slots" in msg and "twse:flow" in msg.split("PER-NAME")[1]
    assert "BROADCAST feature slots (one value per bar; overlay-eligible only): fred:DGS10" in msg
