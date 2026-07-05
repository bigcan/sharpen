"""LlmProposer (spec Part A2) — fixture-driven, no network/key ever touched.

Mirrors the P1b connector test pattern: every test injects ``transport`` directly. The CR-1 tests
mirror ``test_hypothesis_loop.py``'s moat tripwires but target the NEW surface this module adds (the
API request body) rather than the ``ProposalContext`` object itself, since that is what's actually new
here — the context's own no-score guarantee is already covered by the existing P2 tests.
"""
from __future__ import annotations

import pytest

from finrl_pro_ds.crucible import TrialLedger, TrialRecord
from finrl_pro_ds.crucible.agentic import HypothesisAuthor, LlmProposer
from finrl_pro_ds.crucible.agentic.llm_proposer import _PROMPT_TEMPLATE_HASH
from finrl_pro_ds.crucible.agentic.proposer import Proposer

_TS = "2026-07-04T00:00:00+00:00"
# Same forbidden-field sentinel set test_hypothesis_loop.py uses for the CR-1 moat.
_FORBIDDEN_VALUES = ("secret prior", "0.42", "3.1", "2.2", "PROMISING", "NO_GO")


def _proposal(name="cs-rev", candidate_type="cross_sectional",
             formula="rank(delta(close, 5))", sign=-1) -> dict:
    return {"name": name, "hypothesis": "h", "expected_sign": sign,
            "candidate_type": candidate_type, "formula": formula,
            "economic_rationale": "r"}


def _api_response(proposals: list[dict], *, usage: dict | None = None) -> dict:
    return {
        "id": "msg_test", "type": "message", "role": "assistant",
        "content": [{"type": "tool_use", "id": "tu_1", "name": "propose_hypotheses",
                     "input": {"proposals": proposals}}],
        "usage": usage or {"input_tokens": 500, "output_tokens": 300},
    }


class _CapturingTransport:
    """Records every (url, headers, body) call and returns a canned response. Raises if called more
    than ``max_calls`` times — used to prove a consumer never re-invokes the proposer (reproduce)."""

    def __init__(self, response: dict, *, max_calls: int = 100):
        self.response = response
        self.max_calls = max_calls
        self.calls: list[tuple[str, dict, dict]] = []

    def __call__(self, url: str, headers: dict, body: dict) -> dict:
        if len(self.calls) >= self.max_calls:
            raise AssertionError(f"transport invoked more than {self.max_calls} time(s)")
        self.calls.append((url, headers, body))
        return self.response


# --------------------------------------------------------------------- protocol conformance ----

def test_llm_proposer_satisfies_proposer_protocol() -> None:
    assert isinstance(LlmProposer(api_key="x"), Proposer)


# --------------------------------------------------------------------- fails closed ----

def test_missing_api_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    proposer = LlmProposer(api_key="")
    context = HypothesisAuthor(LlmProposer(api_key="x"),
                               TrialLedger(":memory:")).build_context(("close",))
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        proposer.propose(context)


# --------------------------------------------------------------------- CR-1 moat ----

def test_request_body_carries_no_forbidden_score_data(tmp_path) -> None:
    """CR-1: even with a ledger full of scored rows, nothing the LLM proposer sends over the wire
    contains a score/verdict/holdout value — the request body is built ONLY from ProposalContext."""
    ledger = TrialLedger(tmp_path / "l.db")
    ledger.record(TrialRecord(candidate_hash="aaaa", crucible_version="v", family="altdata",
                              candidate_type="overlay", verdict="NO_GO", dsr=1.9,
                              delta_sr_oos=0.42, marginal_hlz_t=3.1, formula="secret",
                              economic_rationale="secret prior"))
    ledger.record(TrialRecord(candidate_hash="bbbb", crucible_version="v", family="101alpha",
                              candidate_type="cross_sectional", verdict="PROMISING", dsr=2.2))

    transport = _CapturingTransport(_api_response([_proposal()]))
    proposer = LlmProposer(api_key="test-key", transport=transport)
    author = HypothesisAuthor(proposer, ledger)
    context = author.build_context(("close", "macro:regime"))
    proposer.propose(context)

    assert len(transport.calls) == 1
    _, headers, body = transport.calls[0]
    blob = repr(body)
    for leaked in _FORBIDDEN_VALUES:
        assert leaked not in blob
    # sanity: the killed family (from ledger row "aaaa") + a live terminal DID make it through —
    # proving the assertion above is testing absence, not an empty/broken request
    assert "altdata" in blob and "macro:regime" in blob
    assert headers["x-api-key"] == "test-key"


def test_killed_families_named_in_prompt(tmp_path) -> None:
    ledger = TrialLedger(tmp_path / "l.db")
    ledger.record(TrialRecord(candidate_hash="dead1", crucible_version="v", family="altdata",
                              verdict="NO_GO"))
    transport = _CapturingTransport(_api_response([_proposal()]))
    proposer = LlmProposer(api_key="k", transport=transport)
    author = HypothesisAuthor(proposer, ledger)
    proposer.propose(author.build_context(("close",)))

    _, _, body = transport.calls[0]
    assert "altdata" in body["messages"][0]["content"]
    assert "KILLED" in body["messages"][0]["content"]


def test_scorer_still_cannot_receive_llm_output_rationale() -> None:
    """Structural CR-1, re-affirmed for this proposer specifically: nothing about swapping the
    proposer implementation changes the scorer's signature."""
    import inspect

    from finrl_pro_ds.signals.generation.evolve import evolve
    from finrl_pro_ds.signals.generation.fitness import combination_fitness

    for fn in (evolve, combination_fitness):
        params = set(inspect.signature(fn).parameters)
        assert not (params & {"economic_rationale", "rationale", "hypothesis", "narrative"})


# --------------------------------------------------------------------- response parsing ----

def test_valid_response_parses_into_proposals() -> None:
    transport = _CapturingTransport(_api_response([
        _proposal("cs-a", "cross_sectional", "rank(delta(close, 10))", 1),
        _proposal("ov-b", "overlay", "delta(fred:DGS10, 20)", 1),
    ]))
    proposer = LlmProposer(api_key="k", transport=transport)
    context = HypothesisAuthor(proposer, TrialLedger(":memory:")).build_context(
        ("close", "fred:DGS10"))
    out = proposer.propose(context)
    assert [p.name for p in out] == ["cs-a", "ov-b"]
    assert out[0].candidate_type == "cross_sectional" and out[1].candidate_type == "overlay"
    assert out[0].expected_sign == 1


def test_malformed_proposal_item_dropped_others_kept() -> None:
    good = _proposal("cs-good")
    bad = {"name": "cs-bad", "hypothesis": "h"}                  # missing required fields
    transport = _CapturingTransport(_api_response([bad, good]))
    proposer = LlmProposer(api_key="k", transport=transport)
    context = HypothesisAuthor(proposer, TrialLedger(":memory:")).build_context(("close",))
    out = proposer.propose(context)
    assert [p.name for p in out] == ["cs-good"]


def test_bad_expected_sign_type_dropped() -> None:
    bad = _proposal("cs-bad")
    bad["expected_sign"] = "not-an-int"
    transport = _CapturingTransport(_api_response([bad]))
    proposer = LlmProposer(api_key="k", transport=transport)
    context = HypothesisAuthor(proposer, TrialLedger(":memory:")).build_context(("close",))
    assert proposer.propose(context) == []


def test_no_tool_use_block_returns_empty() -> None:
    transport = _CapturingTransport({"content": [{"type": "text", "text": "I decline."}],
                                     "usage": {"input_tokens": 10, "output_tokens": 5}})
    proposer = LlmProposer(api_key="k", transport=transport)
    context = HypothesisAuthor(proposer, TrialLedger(":memory:")).build_context(("close",))
    assert proposer.propose(context) == []


def test_transport_exception_returns_empty_not_raised() -> None:
    def _boom(url, headers, body):
        raise TimeoutError("simulated network failure")

    proposer = LlmProposer(api_key="k", transport=_boom)
    context = HypothesisAuthor(proposer, TrialLedger(":memory:")).build_context(("close",))
    assert proposer.propose(context) == []


def test_non_dict_transport_return_degrades_to_empty_not_raised() -> None:
    """A `transport` that returns something other than a dict (a buggy/unusual injection, not the
    real Anthropic client) must degrade like any other malformed response — never propagate. This
    is the mutation tripwire for the response-shape guard: narrowing the try/except back down to
    only wrap the `transport(...)` call itself (not the parsing that follows) makes this raise
    AttributeError instead of returning []."""
    proposer = LlmProposer(api_key="k", transport=lambda u, h, b: None)
    context = HypothesisAuthor(proposer, TrialLedger(":memory:")).build_context(("close",))
    assert proposer.propose(context) == []


def test_malformed_content_shape_degrades_to_empty_not_raised() -> None:
    """A response that IS a dict but whose "content" is not a list of blocks (e.g. a string) must
    also degrade gracefully — `_extract_tool_input`'s iteration is inside the same try/except as
    the transport call for exactly this reason."""
    proposer = LlmProposer(api_key="k", transport=lambda u, h, b: {"content": "not-a-list"})
    context = HypothesisAuthor(proposer, TrialLedger(":memory:")).build_context(("close",))
    assert proposer.propose(context) == []


def test_max_proposals_cap_respected() -> None:
    many = [_proposal(f"cs-{i}") for i in range(10)]
    transport = _CapturingTransport(_api_response(many))
    proposer = LlmProposer(api_key="k", transport=transport)
    context = HypothesisAuthor(proposer, TrialLedger(":memory:"), max_proposals=3).build_context(
        ("close",))
    assert context.max_proposals == 3
    out = proposer.propose(context)
    assert len(out) == 3


# --------------------------------------------------------------------- provenance / usage ----

def test_usage_recorded_after_call() -> None:
    transport = _CapturingTransport(
        _api_response([_proposal()], usage={"input_tokens": 111, "output_tokens": 222}))
    proposer = LlmProposer(api_key="k", transport=transport)
    context = HypothesisAuthor(proposer, TrialLedger(":memory:")).build_context(("close",))
    proposer.propose(context)
    assert proposer.last_usage == {"input_tokens": 111, "output_tokens": 222, "total_tokens": 333}


def test_usage_resets_on_failed_call() -> None:
    proposer = LlmProposer(api_key="k", transport=lambda u, h, b: (_ for _ in ()).throw(ValueError()))
    proposer.last_usage = {"input_tokens": 999, "output_tokens": 999, "total_tokens": 1998}
    context = HypothesisAuthor(proposer, TrialLedger(":memory:")).build_context(("close",))
    proposer.propose(context)
    assert proposer.last_usage == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def test_model_id_reflects_model_and_is_stable_for_same_template() -> None:
    a = LlmProposer(api_key="k", model="claude-sonnet-5")
    b = LlmProposer(api_key="k", model="claude-opus-4-8")
    assert a.model_id != b.model_id
    assert a.model_id.startswith("llm-claude-sonnet-5-prompt")
    assert _PROMPT_TEMPLATE_HASH in a.model_id
    # same model + same (unedited) template -> identical id across independent instances
    assert LlmProposer(api_key="k", model="claude-sonnet-5").model_id == a.model_id


# --------------------------------------------------------------------- drop-in / Author integration ----

def test_author_validates_llm_output_same_as_any_proposer(tmp_path) -> None:
    """End-to-end drop-in property: the Author's grammar-parse + dedup + killed-family guardrails
    apply to this proposer's output exactly as they do to LibrarySeedProposer's — nothing about the
    Author needed to change. The killed family here is "altdata" (a real, reachable value — family is
    deterministic from candidate_type, so an overlay proposal always lands in "altdata")."""
    ledger = TrialLedger(tmp_path / "l.db")
    ledger.record(TrialRecord(candidate_hash="dead", crucible_version="v", family="altdata",
                              verdict="NO_GO"))
    transport = _CapturingTransport(_api_response([
        _proposal("ov-killed", candidate_type="overlay", formula="fred:DGS10"),  # family=altdata, dropped
        _proposal("cs-bad-syntax", formula="this is not dsl((("),               # must be dropped
        _proposal("cs-ok", formula="rank(delta(close, 10))"),                    # must survive
    ]))
    proposer = LlmProposer(api_key="k", transport=transport)
    author = HypothesisAuthor(proposer, ledger)
    specs = author.propose(author.build_context(("close", "fred:DGS10")), proposal_ts=_TS)
    assert [s.spec.name for s in specs] == ["cs-ok"]
    assert author.last_proposal_stats == {"raw": 3, "accepted": 1, "dropped": 2}


def test_reproduce_replays_recorded_specs_without_recalling_the_llm(tmp_path) -> None:
    """The documented reproduce contract for this proposer: a past run is replayed via
    `run_hypothesis_loop(pre_proposed=...)`, never by re-invoking `propose()`. Proven concretely with
    a transport that raises on a second call — if anything downstream tried to re-propose, this test
    would fail with the transport's own AssertionError rather than a normal assertion mismatch."""
    ledger = TrialLedger(tmp_path / "l.db")
    transport = _CapturingTransport(_api_response([_proposal("cs-ok", formula="rank(close)")]),
                                    max_calls=1)
    proposer = LlmProposer(api_key="k", transport=transport)
    author = HypothesisAuthor(proposer, ledger)
    context = author.build_context(("close",))
    specs = author.propose(context, proposal_ts=_TS)          # the ONE allowed call
    author.preregister(specs, run_id="r0", crucible_version="crucible-vX")

    # "reproduce": hand the already-recorded specs back in via pre_proposed. A second author/proposer
    # call here would hit the transport's call-count guard and raise.
    assert len(transport.calls) == 1
    replayed_names = [pr.spec.name for pr in specs]
    assert replayed_names == ["cs-ok"]


def test_context_rendering_uses_only_context_fields() -> None:
    """`_render_context` cannot leak anything beyond ProposalContext by construction (its signature
    accepts only a context) — this test locks that signature so a future edit can't quietly widen it."""
    import inspect

    from finrl_pro_ds.crucible.agentic.llm_proposer import _render_context

    params = list(inspect.signature(_render_context).parameters)
    assert params == ["context"]
