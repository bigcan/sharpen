"""The questionnaire is the strategy's pre-registration in code. These tests pin what makes a score honest:
the answer mapping, the signs, the scoping, NaN-not-zero for an unscored block, and a hash that moves
whenever anything that defines the questionnaire moves."""
from __future__ import annotations

import dataclasses
import math

import pytest

from sharpen.jev import JevAnswer
from sharpen.jev import questionnaire as qn


def _noul(p: float) -> JevAnswer:
    return JevAnswer("noul", p, None, None, "jev-test")


def _score(s: float) -> JevAnswer:
    return JevAnswer("score", s, 0.9, None, "jev-test")


Q = {q.qid: q for q in qn.QUESTIONS_V1}


def test_noul_maps_to_evidence_of_presence_so_a_no_contributes_nothing():
    assert qn.map_answer(Q["E1"], _noul(1.0)) == 1.0
    assert qn.map_answer(Q["E1"], _noul(0.0)) == 0.0             # "no" is the null state, not -1
    assert qn.map_answer(Q["E1"], _noul(0.5)) == 0.5
    assert qn.map_answer(Q["E1"], _noul(1.2)) == 1.0             # clipped


def test_score_is_bipolar_and_centred_on_the_neutral_level():
    assert qn.map_answer(Q["E3"], _score(0.0)) == -1.0           # 5 levels: 0..4
    assert qn.map_answer(Q["E3"], _score(2.0)) == 0.0            # "stable, or not discussed" is neutral
    assert qn.map_answer(Q["E3"], _score(4.0)) == 1.0
    assert qn.map_answer(Q["E3"], _score(3.0)) == pytest.approx(0.5)


def test_a_news_free_filing_scores_neutral_whatever_questions_it_was_asked():
    """The Math-audit defect this mapping fixes: with 2p-1 a clean event filing (negatively-signed "no"s
    only) scored ~+0.9 while a news-free earnings release scored ~+0.6 — rank by filing TYPE, not news."""
    quiet_release = {"E1": _noul(0.02), "E2": _noul(0.02), "E3": _score(2.0), "E4": _noul(0.02),
                     "E5": _noul(0.02), "M1": _noul(0.02)}
    quiet_event = {"M1": _noul(0.02)}
    a = qn.filing_score(quiet_release, qn.questions_for(["2.02"]))
    b = qn.filing_score(quiet_event, qn.questions_for(["2.06"]))
    assert abs(a) < 0.05 and abs(b) < 0.05


def test_mapping_rejects_a_type_mismatch():
    with pytest.raises(ValueError, match="answer type"):
        qn.map_answer(Q["E1"], _score(2.0))


def test_signs_make_a_higher_score_better_news():
    qs = qn.questions_for(["2.02"])
    good = {"E1": _noul(0.95), "E2": _noul(0.05), "E3": _score(4.0), "E4": _noul(0.05), "E5": _noul(0.05),
            "M1": _noul(0.05)}
    bad = {"E1": _noul(0.05), "E2": _noul(0.95), "E3": _score(0.0), "E4": _noul(0.95), "E5": _noul(0.95),
           "M1": _noul(0.95)}
    assert qn.filing_score(good, qs) > 0.25 and qn.filing_score(bad, qs) < -0.6
    # tone inflation enters NEGATED: optimism beyond the numbers lowers the tone block
    assert qn.filing_score({"E4": _noul(0.9)}, qs, blocks=["tone"]) == pytest.approx(-0.9)
    # a lowered outlook is bad news on its own; the absence of a raise does not add to it
    assert qn.filing_score({"E1": _noul(0.02), "E2": _noul(0.95)}, qs, blocks=["surprise"]) == \
        pytest.approx((0.02 - 0.95) / 2)


def test_filing_score_is_a_mean_over_the_answered_questions_of_the_block():
    qs = qn.questions_for(["2.02"])
    ans = {"E1": _noul(0.75), "E2": _noul(0.25), "E3": _score(3.0)}
    # surprise block: (+0.75) + (-0.25) + (+0.5) -> mean 1/3
    assert qn.filing_score(ans, qs, blocks=["surprise"]) == pytest.approx(1.0 / 3.0)


def test_unscored_block_is_nan_never_zero():
    qs = qn.questions_for(["2.06"])                             # impairment filing: M1 only
    s = qn.filing_score({"M1": _noul(0.2)}, qs, blocks=["surprise"])
    assert math.isnan(s)


def test_unknown_block_is_rejected():
    with pytest.raises(ValueError, match="unknown blocks"):
        qn.filing_score({}, qn.QUESTIONS_V1, blocks=["sentiment"])


@pytest.mark.parametrize("items, expect", [
    (["2.02", "9.01"], ["E1", "E2", "E3", "E4", "E5", "M1"]),
    (["5.02"], []),                                             # left scope with the dropped M2
    (["2.06"], ["M1"]),
    (["2.02", "5.02"], ["E1", "E2", "E3", "E4", "E5", "M1"]),
    (["8.01", "9.01"], []),                                     # out of scope
    ([], []),
])
def test_scoping(items, expect):
    assert [q.qid for q in qn.questions_for(items)] == expect


def test_every_question_has_a_sign_mechanism_and_valid_block():
    for q in qn.QUESTIONS_V1:
        assert q.sign in (-1, 1) and q.block in qn.BLOCKS and q.mechanism
        if q.qtype == "score":
            assert len(q.criteria) >= 2
        req = q.request()
        assert req["instructions"].startswith(qn.PREAMBLE)


@pytest.mark.parametrize("mutate", [
    lambda q: dataclasses.replace(q, text=q.text + " "),         # wording
    lambda q: dataclasses.replace(q, sign=-q.sign),              # sign
    lambda q: dataclasses.replace(q, block="tone" if q.block != "tone" else "quality"),
    lambda q: dataclasses.replace(q, scope="adverse" if q.scope != "adverse" else "earnings"),
])
def test_hash_moves_when_anything_that_defines_the_questionnaire_moves(mutate):
    base = qn.questionnaire_hash()
    changed = (mutate(qn.QUESTIONS_V1[0]),) + qn.QUESTIONS_V1[1:]
    assert qn.questionnaire_hash(changed) != base
    assert qn.questionnaire_hash(tuple(reversed(qn.QUESTIONS_V1))) == base   # order-insensitive


def test_hash_covers_the_in_scope_item_list(monkeypatch):
    base = qn.questionnaire_hash()
    monkeypatch.setattr(qn, "ADVERSE_ITEMS", qn.ADVERSE_ITEMS | {"8.01"})
    assert qn.questionnaire_hash() != base


def test_hash_covers_the_scoring_rule(monkeypatch):
    base = qn.questionnaire_hash()
    monkeypatch.setattr(qn, "SCORING", qn.SCORING.replace("noul->p", "noul->2p-1"))
    assert qn.questionnaire_hash() != base


def test_dropped_question_stays_dropped():
    assert "M2" not in {q.qid for q in qn.QUESTIONS_V1}
    assert "5.02" not in qn.ADVERSE_ITEMS


def test_preregistered_hash_matches_the_code():
    """The declaration in configs/atl_jev.gates.yaml binds: an edit to the questionnaire without a new
    pre-registration fails here (and the Phase-3 scorer refuses to run)."""
    from pathlib import Path

    import yaml
    g = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs" / "atl_jev.gates.yaml").read_text(encoding="utf-8"))
    assert g["phase1"]["questionnaire"]["hash"] == qn.questionnaire_hash()
    assert g["phase1"]["questionnaire"]["version"] == qn.VERSION
