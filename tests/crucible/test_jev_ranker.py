"""JevRanker / RankedProposer — fixture-driven, no network or key ever touched.

Mirrors ``test_llm_proposer.py``: every test injects ``transport``. Two families matter most here and
neither exists in the LLM-proposer suite, because the ranker adds a surface that module does not have:

  * the STRATIFICATION tripwires, which fail if a ranking is ever allowed to turn the
    ``max_proposals`` cap into a candidate-type or source FILTER (the two bugs ``_interleave`` and
    ``_round_robin_by_source`` were written to fix), and
  * the DEGRADE-TO-IDENTITY tripwires, which fail if a dead or malformed API ever produces a
    partially-reordered batch instead of today's deterministic one.
"""
from __future__ import annotations

import pytest

from sharpen.crucible.agentic import JevRanker, RankedProposer
from sharpen.crucible.agentic.jev_ranker import _PROMPT_HASH
from sharpen.crucible.agentic.proposer import HypothesisProposal, ProposalContext, Proposer

# Same forbidden-value sentinel discipline the CR-1 moat tests use elsewhere in this suite.
_FORBIDDEN_VALUES = ("secret prior", "0.42", "3.1", "2.2", "PROMISING", "NO_GO", "DSR", "holdout")


def _p(name: str, *, candidate_type: str = "cross_sectional",
       formula: str = "rank(delta(close, 5))") -> HypothesisProposal:
    return HypothesisProposal(
        name=name, hypothesis=f"hypothesis for {name}",
        family="101alpha" if candidate_type == "cross_sectional" else "altdata",
        expected_sign=1, candidate_type=candidate_type, formula=formula,
        economic_rationale=f"rationale for {name}")


def _ctx(**kw) -> ProposalContext:
    base = dict(
        available_terminals=("close", "volume", "fred:DGS10", "cot:comm_net"),
        asset_classes=("macro",), max_proposals=4, panel_n=18,
        feature_slot_bars=(("cot:comm_net", 900), ("fred:DGS10", 4000)))
    base.update(kw)
    return ProposalContext(**base)


class _Transport:
    """Returns a canned score per question, recording every (url, headers, body) call.

    ``scores_by_name`` maps a proposal NAME to its score; the fixture resolves each ``qN`` back to a
    name via the request body, so a test states its intent by name rather than by question index.
    """

    def __init__(self, scores_by_name: dict[str, float], *, fail: bool = False,
                 response_override: dict | None = None):
        self.scores_by_name = scores_by_name
        self.fail = fail
        self.response_override = response_override
        self.calls: list[tuple[str, dict, dict]] = []

    def __call__(self, url: str, headers: dict, body: dict) -> dict:
        self.calls.append((url, headers, body))
        if self.fail:
            raise RuntimeError("simulated transport failure")
        if self.response_override is not None:
            return self.response_override
        answers = {}
        for qid, q in body["questions"].items():
            name = q["instructions"].split("name: ", 1)[1].split("\n", 1)[0].strip()
            answers[qid] = {"type": "score", "score": self.scores_by_name[name],
                            "legend": {"0": "a", "1": "b", "2": "c"},
                            "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}, "confidence": 0.9}
        return {"model": "jev-1", "answers": answers,
                "usage": {"input_tokens": 300, "output_tokens": 20}}


class _PositionalTransport:
    """Scores strictly by question INDEX, so a test can give two proposals the same ``name`` and
    still assign them different scores — which a name-keyed implementation cannot represent."""

    def __init__(self, scores: list[float]):
        self.scores = scores
        self.calls: list[tuple[str, dict, dict]] = []

    def __call__(self, url: str, headers: dict, body: dict) -> dict:
        self.calls.append((url, headers, body))
        offset = sum(len(c[2]["questions"]) for c in self.calls[:-1])
        answers = {qid: {"type": "score", "score": self.scores[offset + int(qid[1:])],
                         "confidence": 0.9}
                   for qid in body["questions"]}
        return {"model": "jev-1", "answers": answers,
                "usage": {"input_tokens": 100, "output_tokens": 10}}


class _FixedProposer:
    """Inner proposer returning a fixed pool, truncated at ``context.max_proposals`` exactly as
    ``LibrarySeedProposer`` does — so the oversample test exercises the real interaction."""

    model_id = "fixed-v1"

    def __init__(self, pool: list[HypothesisProposal]):
        self.pool = pool
        self.last_max_proposals: int | None = None

    def propose(self, context: ProposalContext) -> list[HypothesisProposal]:
        self.last_max_proposals = context.max_proposals
        return self.pool[: context.max_proposals]


# ----------------------------------------------------------------- protocol + provenance ----

def test_ranked_proposer_satisfies_proposer_protocol() -> None:
    ranker = JevRanker(api_key="x", transport=_Transport({}))
    assert isinstance(RankedProposer(_FixedProposer([]), ranker), Proposer)


def test_model_id_pins_model_and_prompt_hash() -> None:
    ranker = JevRanker(api_key="x", model="jev-latest")
    assert ranker.model_id == f"jev-jev-latest-prompt{_PROMPT_HASH}"
    wrapped = RankedProposer(_FixedProposer([]), ranker, oversample=3)
    assert "fixed-v1" in wrapped.model_id and _PROMPT_HASH in wrapped.model_id
    assert "os3" in wrapped.model_id


# ----------------------------------------------------------------------------- fails closed ----

def test_rank_fails_closed_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    ranker = JevRanker()
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        ranker.rank([_p("a"), _p("b")], _ctx())


def test_rank_reads_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "env-key")
    transport = _Transport({"a": 2.0, "b": 1.0})
    JevRanker(transport=transport).rank([_p("a"), _p("b")], _ctx())
    assert transport.calls[0][1]["Authorization"] == "Bearer env-key"


# --------------------------------------------------------------------------------- ordering ----

def test_rank_orders_by_score_within_a_stratum() -> None:
    proposals = [_p("low"), _p("high"), _p("mid")]
    ranker = JevRanker(api_key="x", transport=_Transport({"low": 0.1, "high": 2.0, "mid": 1.0}))
    assert [p.name for p in ranker.rank(proposals, _ctx())] == ["high", "mid", "low"]


def test_rank_is_a_pure_reorder_never_a_filter() -> None:
    proposals = [_p(f"c{i}") for i in range(6)]
    ranker = JevRanker(api_key="x", transport=_Transport({f"c{i}": float(i) for i in range(6)}))
    ranked = ranker.rank(proposals, _ctx())
    assert len(ranked) == len(proposals)
    assert {p.name for p in ranked} == {p.name for p in proposals}


def test_equal_scores_preserve_deterministic_input_order() -> None:
    proposals = [_p("a"), _p("b"), _p("c")]
    ranker = JevRanker(api_key="x", transport=_Transport({"a": 1.0, "b": 1.0, "c": 1.0}))
    assert [p.name for p in ranker.rank(proposals, _ctx())] == ["a", "b", "c"]


# ------------------------------------------------------------------- STRATIFICATION tripwires ----

def test_cap_cannot_become_a_candidate_type_filter() -> None:
    """The v12.2 bug: 'a batch cap must not silently become a type filter'.

    Jev is made to rank EVERY cross-sectional above EVERY overlay. A global sort would then let the
    ``max_proposals=4`` cut delete the overlay leg entirely. Stratified ranking must not.
    """
    xsec = [_p(f"cs{i}") for i in range(4)]
    overlay = [_p(f"ov{i}", candidate_type="overlay", formula=f"delta(fred:DGS10, {i + 2})")
               for i in range(4)]
    scores = {f"cs{i}": 2.0 for i in range(4)} | {f"ov{i}": 0.0 for i in range(4)}
    ranker = JevRanker(api_key="x", transport=_Transport(scores))

    kept = ranker.rank(xsec + overlay, _ctx())[:4]
    types = {p.candidate_type for p in kept}
    assert types == {"cross_sectional", "overlay"}, (
        f"cap became a candidate-type filter — kept only {types}")


def test_cap_cannot_become_a_source_filter() -> None:
    """The ``_round_robin_by_source`` bug: a source-dependent cut silently dropped four of six COT
    markets. A ranking that prefers every ``fred:`` overlay must not delete the ``cot:`` leg."""
    fred = [_p(f"f{i}", candidate_type="overlay", formula=f"delta(fred:DGS10, {i + 2})")
            for i in range(4)]
    cot = [_p(f"c{i}", candidate_type="overlay", formula=f"decay_linear(cot:comm_net, {i + 2})")
           for i in range(4)]
    scores = {f"f{i}": 2.0 for i in range(4)} | {f"c{i}": 0.0 for i in range(4)}
    ranker = JevRanker(api_key="x", transport=_Transport(scores))

    kept = ranker.rank(fred + cot, _ctx())[:4]
    assert any(p.name.startswith("c") for p in kept), (
        f"cap became a source filter — kept only {[p.name for p in kept]}")
    assert any(p.name.startswith("f") for p in kept)


def test_duplicate_proposal_names_are_scored_positionally_not_collapsed() -> None:
    """A proposer may emit two proposals with the same ``name`` — ``LlmProposer`` takes the name from
    the model, and the Author's dedup runs on the FORMULA hash only AFTER propose() returns, so with
    RankedProposer the duplicates reach the ranker first. Scoring keyed by name silently collapses
    them onto one score and mis-orders the batch; this fails unless scoring is positional.
    """
    dup_a = _p("dup", formula="rank(delta(close, 5))")
    dup_b = _p("dup", formula="rank(delta(close, 20))")
    uniq = _p("uniq", formula="rank(delta(close, 60))")
    ranker = JevRanker(api_key="x", transport=_PositionalTransport([0.0, 2.0, 1.0]))

    ranked = ranker.rank([dup_a, dup_b, uniq], _ctx())
    assert [p.formula for p in ranked] == [dup_b.formula, uniq.formula, dup_a.formula]


def test_rank_never_raises_even_with_duplicate_names() -> None:
    """``rank`` is contractually total: it degrades, never propagates. A name-keyed score map could
    raise KeyError out of the sort, past the transport try/except."""
    proposals = [_p("same") for _ in range(4)]
    ranker = JevRanker(api_key="x", transport=_PositionalTransport([3.0, 1.0, 4.0, 2.0]))
    assert len(ranker.rank(proposals, _ctx())) == 4


def test_unspaced_ternary_colon_is_not_mistaken_for_a_connector_source() -> None:
    """``cond ? a : b`` also contains a colon. The library writes it spaced, so the regex misses it —
    but that is a formatting convention, not a guarantee. An OHLCV name is never a connector."""
    from sharpen.crucible.agentic.jev_ranker import _source_of
    assert _source_of(_p("t", formula="rank(close>0?close:open)")) == ""
    assert _source_of(_p("t", formula="rank(volume>0?high:low)")) == ""
    # A real connector prefix still resolves, even when a ternary precedes it.
    assert _source_of(_p("t", candidate_type="overlay",
                         formula="close>0?fred:DGS10")) == "fred"
    assert _source_of(_p("t", candidate_type="overlay",
                         formula="delta(cot:comm_net, 20)")) == "cot"


def test_ranking_still_decides_the_winner_inside_a_stratum() -> None:
    """Stratification must not neuter the ranker: within one stratum the best prior still wins."""
    overlay = [_p(f"ov{i}", candidate_type="overlay", formula=f"delta(fred:DGS10, {i + 2})")
               for i in range(4)]
    scores = {"ov0": 0.0, "ov1": 0.5, "ov2": 2.0, "ov3": 1.0}
    ranker = JevRanker(api_key="x", transport=_Transport(scores))
    assert [p.name for p in ranker.rank(overlay, _ctx())] == ["ov2", "ov3", "ov1", "ov0"]


# --------------------------------------------------------------- DEGRADE-TO-IDENTITY tripwires ----

def test_transport_failure_degrades_to_identity_order() -> None:
    proposals = [_p("a"), _p("b"), _p("c")]
    ranker = JevRanker(api_key="x", transport=_Transport({}, fail=True))
    assert [p.name for p in ranker.rank(proposals, _ctx())] == ["a", "b", "c"]


def test_malformed_response_degrades_to_identity_order() -> None:
    proposals = [_p("a"), _p("b"), _p("c")]
    for bad in ({"no_answers_key": 1}, {"answers": {"q0": {"score": "NaN-ish"}}},
                {"answers": {}}, {"answers": {"q0": {}, "q1": {}, "q2": {}}}):
        ranker = JevRanker(api_key="x", transport=_Transport({}, response_override=bad))
        assert [p.name for p in ranker.rank(proposals, _ctx())] == ["a", "b", "c"], bad


def test_partial_chunk_failure_degrades_the_whole_batch_not_half_of_it() -> None:
    """A half-ranked batch would be a third behaviour that is neither today's deterministic order nor
    a full ranking — and it would be silent. One dead chunk must degrade everything to identity."""

    class _FailSecondChunk:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict, dict]] = []

        def __call__(self, url: str, headers: dict, body: dict) -> dict:
            self.calls.append((url, headers, body))
            if len(self.calls) >= 2:
                raise RuntimeError("second chunk dies")
            return {"model": "jev-1",
                    "answers": {qid: {"type": "score", "score": 1.0, "confidence": 0.5}
                                for qid in body["questions"]},
                    "usage": {"input_tokens": 10, "output_tokens": 1}}

    proposals = [_p(f"c{i}") for i in range(6)]
    ranker = JevRanker(api_key="x", transport=_FailSecondChunk(), max_questions_per_request=2)
    assert [p.name for p in ranker.rank(proposals, _ctx())] == [f"c{i}" for i in range(6)]
    assert ranker.last_usage["total_tokens"] == 0


# ----------------------------------------------------------------------------- CR-1 / CRU-2 ----

def test_cr1_no_score_or_verdict_can_reach_the_request_body() -> None:
    """The moat tripwire. The ranker's input surface is the proposal text + ProposalContext; a
    context carries no score field by construction, so no forbidden sentinel can appear on the wire.
    """
    transport = _Transport({"a": 1.0, "b": 2.0})
    ranker = JevRanker(api_key="x", transport=transport)
    ranker.rank([_p("a"), _p("b")], _ctx())
    payload = str(transport.calls[0][2])
    for forbidden in _FORBIDDEN_VALUES:
        assert forbidden not in payload, f"{forbidden!r} reached the Jev request body"


def test_cr1_instructions_forbid_predicting_the_outcome() -> None:
    """The rubric must rate the PRIOR, not guess performance — asking Jev to predict profitability
    would invite exactly the outcome-conditioning CR-1 exists to prevent."""
    transport = _Transport({"a": 1.0, "b": 2.0})
    JevRanker(api_key="x", transport=transport).rank([_p("a"), _p("b")], _ctx())
    instructions = transport.calls[0][2]["questions"]["q0"]["instructions"]
    assert "must not infer" in instructions
    assert "QUALITY OF THE PRIOR" in instructions


def test_request_shape_matches_the_documented_systemone_contract() -> None:
    transport = _Transport({"a": 1.0, "b": 2.0})
    JevRanker(api_key="x", transport=transport).rank([_p("a"), _p("b")], _ctx())
    url, headers, body = transport.calls[0]
    assert url == "https://api.typesafe.ai/v1/systemone"
    assert headers["Authorization"].startswith("Bearer ")
    assert set(body) == {"model", "state", "questions"}
    q0 = body["questions"]["q0"]
    assert q0["type"] == "score"
    assert isinstance(q0["criteria"], list) and 2 <= len(q0["criteria"]) <= 10


# ------------------------------------------------------------------------------- chunking ----

def test_chunking_scores_every_proposal_across_multiple_requests() -> None:
    proposals = [_p(f"c{i}") for i in range(5)]
    transport = _Transport({f"c{i}": float(i) for i in range(5)})
    ranker = JevRanker(api_key="x", transport=transport, max_questions_per_request=2)
    ranked = ranker.rank(proposals, _ctx())
    assert len(transport.calls) == 3
    assert [p.name for p in ranked] == ["c4", "c3", "c2", "c1", "c0"]
    assert ranker.last_usage["total_tokens"] == 3 * (300 + 20)


# ------------------------------------------------------------------------- RankedProposer ----

def test_ranked_proposer_oversamples_then_truncates() -> None:
    """The whole point: rank a pool LARGER than the cap, then cut to the cap."""
    pool = [_p(f"c{i}") for i in range(20)]
    inner = _FixedProposer(pool)
    # Reverse the library order so the ranking is observable.
    transport = _Transport({f"c{i}": float(20 - i) for i in range(20)})
    wrapped = RankedProposer(inner, JevRanker(api_key="x", transport=transport), oversample=4)

    out = wrapped.propose(_ctx(max_proposals=4))
    assert inner.last_max_proposals == 16, "inner proposer was not oversampled"
    assert len(out) == 4
    assert [p.name for p in out] == ["c0", "c1", "c2", "c3"]


def test_ranked_proposer_returns_exactly_max_proposals() -> None:
    pool = [_p(f"c{i}") for i in range(50)]
    transport = _Transport({f"c{i}": float(i % 7) for i in range(50)})
    wrapped = RankedProposer(_FixedProposer(pool),
                             JevRanker(api_key="x", transport=transport), oversample=5)
    assert len(wrapped.propose(_ctx(max_proposals=8))) == 8


def test_oversample_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="oversample must be >= 1"):
        RankedProposer(_FixedProposer([]), JevRanker(api_key="x"), oversample=0)


def test_ranked_proposer_degrades_to_the_inner_order_when_jev_is_down() -> None:
    """CRU-1 safety: with the API dead, the wrapped proposer must emit exactly the batch the
    unwrapped deterministic proposer would have emitted."""
    pool = [_p(f"c{i}") for i in range(20)]
    inner = _FixedProposer(pool)
    wrapped = RankedProposer(inner, JevRanker(api_key="x", transport=_Transport({}, fail=True)),
                             oversample=4)
    assert [p.name for p in wrapped.propose(_ctx(max_proposals=4))] == ["c0", "c1", "c2", "c3"]
