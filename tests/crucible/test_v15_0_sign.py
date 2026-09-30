"""crucible-v15.0: the pre-registered DIRECTION is the tested direction (deep audit 2026-09-30).

Every scoring path downstream of the Author trades the formula STRING as written, with a ONE-SIDED test.
``expected_sign`` was recorded and never applied, so a -1 hypothesis was scored in the mirror-image
direction (15 of 53 LLM specs in the taiwan_v2 store). These tests fail on the pre-v15 Author.
"""
from __future__ import annotations

import numpy as np

from sharpen.crucible.agentic.hypothesis import (
    HypothesisAuthor,
    candidate_hash,
    sign_folded_formula,
)
from sharpen.crucible.agentic.llm_proposer import _PROMPT_TEMPLATE
from sharpen.crucible.agentic.proposer import (
    _CS_SEED_BANK,
    HypothesisProposal,
    LibrarySeedProposer,
    ProposalContext,
)
from sharpen.crucible.ledger import TrialLedger
from sharpen.signals.generation.dsl_signal import eval_on_panel
from sharpen.signals.generation.grammar import parse, to_formula
from sharpen.signals.library._alpha_formulas import FORMULAS

_TS = "2026-09-30T00:00:00+00:00"


class _Emit:
    model_id = "test-emit"

    def __init__(self, props):
        self._p = props

    def propose(self, context):
        return list(self._p)


def _author(tmp_path, props) -> HypothesisAuthor:
    return HypothesisAuthor(_Emit(props), TrialLedger(tmp_path / "l.db"))


def test_minus_one_is_folded_into_the_tested_formula(tmp_path) -> None:
    p = HypothesisProposal("rev", "5d reversal", "101alpha", -1, "cross_sectional",
                           "rank(delta(close, 5))", "reversal")
    [spec] = _author(tmp_path, [p]).propose(ProposalContext(), proposal_ts=_TS)
    assert spec.formula == to_formula(parse("-(rank(delta(close, 5)))"))
    assert spec.candidate_hash == candidate_hash(spec.formula)
    assert spec.spec.expected_sign == 1                       # the tested formula is pre-signed
    assert "sign-folded" in spec.economic_rationale


def test_folded_formula_scores_the_mirror_of_the_unfolded_one() -> None:
    from sharpen.crucible.orchestrator.substrate import PreparedSubstrate  # noqa: F401  (import check)
    rng = np.random.default_rng(1)
    t, n = 200, 6
    close = np.exp(np.cumsum(0.01 * rng.standard_normal((t, n)), axis=0) + 4.0)
    from sharpen.signals.features import Panel
    dates = (np.datetime64("2015-01-02") + np.arange(t) * np.timedelta64(1, "D")).astype("datetime64[ns]")
    panel = Panel(dates, tuple(f"S{i}" for i in range(n)), close, close, close, close,
                  np.ones((t, n)), np.ones((t, n), bool), close, np.zeros(n, int), {})
    f = "rank(delta(close, 5))"
    a = eval_on_panel(f, panel)
    b = eval_on_panel(to_formula(parse(sign_folded_formula(f, -1))), panel)
    fin = np.isfinite(a)
    assert np.array_equal(fin, np.isfinite(b)) and np.allclose(b[fin], -a[fin])


def test_plus_one_leaves_the_formula_and_hash_unchanged(tmp_path) -> None:
    f = "rank(delta(close, 5))"
    p = HypothesisProposal("mom", "5d momentum", "101alpha", 1, "cross_sectional", f)
    [spec] = _author(tmp_path, [p]).propose(ProposalContext(), proposal_ts=_TS)
    assert spec.formula == to_formula(parse(f)) and spec.candidate_hash == candidate_hash(f)


def test_opposite_directions_are_distinct_hypotheses(tmp_path) -> None:
    f = "rank(delta(close, 5))"
    props = [HypothesisProposal("up", "h", "101alpha", 1, "cross_sectional", f),
             HypothesisProposal("down", "h", "101alpha", -1, "cross_sectional", f)]
    specs = _author(tmp_path, props).propose(ProposalContext(), proposal_ts=_TS)
    assert len({s.candidate_hash for s in specs}) == 2


def test_library_bank_is_presigned_so_its_batch_is_unchanged(tmp_path) -> None:
    """The WQ101 formulas carry their own sign; every library proposal is +1, so folding can never
    touch the offline bank (its formulas and hashes are exactly the verbatim canonical WQ101 strings)."""
    assert all(sign == 1 for _, _, _, sign in _CS_SEED_BANK)
    ctx = ProposalContext(available_terminals=("close", "macro:regime"), max_proposals=512)
    props = LibrarySeedProposer(extended_cs_bank=True).propose(ctx)
    assert all(p.expected_sign == 1 for p in props)
    specs = _author(tmp_path, props).propose(ctx, proposal_ts=_TS)
    curated = {FORMULAS[i] for i, _, _, _ in _CS_SEED_BANK}
    got = {s.formula for s in specs}
    assert {to_formula(parse(f)) for f in curated} <= got


def test_llm_prompt_states_the_sign_convention() -> None:
    assert "SIGN CONVENTION" in _PROMPT_TEMPLATE and "NEGATED" in _PROMPT_TEMPLATE
