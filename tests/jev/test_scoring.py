"""The ATL x Jev scorer (architecture §3). These pin what makes a score trustworthy: the instrument is the
pre-registered one, Jev sees only masked text and questions (never a return, date or identifier), block scores
follow the hashed mapping with NaN for unasked blocks, a second served model stops the run, and a rerun replays
the answer cache without a call. All offline."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sharpen.jev import JevClient
from sharpen.jev.corpus import SHARD_COLUMNS, pipeline_sha
from sharpen.jev.questionnaire import VERSION, questionnaire_hash, questions_for
from sharpen.jev.scoring import (
    ServedVersionChanged,
    check_corpus_stamp,
    check_questionnaire,
    score_corpus,
    to_filing_scores,
)

ROOT = Path(__file__).resolve().parents[2]
PHASE1 = yaml.safe_load((ROOT / "configs" / "atl_jev.gates.yaml").read_text(encoding="utf-8"))["phase1"]
# The fake model's answer to each question, keyed by a phrase from its wording.
_ANSWERS = {"raised its financial outlook": ("noul", 0.9), "lowered or withdrew": ("noul", 0.1),
            "forward trajectory": ("score", 3.0), "noticeably more optimistic": ("noul", 0.2),
            "flattered by one-time": ("noul", 0.3), "material adverse development": ("noul", 0.6)}


class _Jev:
    """Spy transport: records every request body and answers from ``_ANSWERS``."""

    def __init__(self, served=("jev-1.13.0",)):
        self.bodies: list[dict] = []
        self.served = list(served)

    def __call__(self, url, headers, body):
        self.bodies.append(copy.deepcopy(body))
        model = self.served[min(len(self.bodies), len(self.served)) - 1]
        answers = {}
        for key, q in body["questions"].items():
            qtype, value = next(v for phrase, v in _ANSWERS.items() if phrase in q["instructions"])
            answers[key] = {"type": qtype, qtype: value}
        return {"model": model, "answers": answers, "usage": {"input_tokens": len(body["state"]) // 4,
                                                             "output_tokens": 0}}


def _row(acc, items, text, tickers="ACME", day="2019-02-01 21:05"):
    qs = questions_for(items.split(","))
    return {"accession": acc, "cik": 111, "tickers": tickers, "items": items, "accepted_utc": pd.Timestamp(day),
            "earnings": "2.02" in items, "doc_source": "EX-99.1", "qids": ",".join(q.qid for q in qs),
            "raw_chars": len(text), "text_len": len(text), "truncated": False, "text": text,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "questionnaire_hash": questionnaire_hash()}


def _corpus(*rows):
    df = pd.DataFrame(list(rows), columns=list(SHARD_COLUMNS))
    df["accepted_utc"] = df["accepted_utc"].astype("datetime64[ns]")
    return df


EARN = _row("0000000111-19-000001", "2.02,9.01", "[COMPANY] raised its outlook for the year.")
EVENT = _row("0000000111-19-000003", "1.02,9.01", "Item 1.02 Termination of a Material Definitive Agreement.",
             day="2019-04-01 14:00")
EMPTY = _row("0000000111-19-000005", "2.02", "", day="2019-06-03 20:00")


def _client(t, tmp_path=None):
    return JevClient(api_key="test", transport=t, cache_path=None if tmp_path is None else tmp_path / "c.sqlite",
                     sleep=lambda s: None)


# --------------------------------------------------------------------------- instrument and corpus gates
def test_refuses_a_questionnaire_that_is_not_the_preregistered_one():
    check_questionnaire(PHASE1)
    for key, bad in (("hash", "000000000000"), ("version", "v2")):
        p = copy.deepcopy(PHASE1)
        p["questionnaire"][key] = bad
        with pytest.raises(SystemExit, match="refusing to score"):
            check_questionnaire(p)
    assert PHASE1["questionnaire"] == {"version": VERSION, "hash": questionnaire_hash()}


def test_refuses_a_corpus_built_by_another_pipeline_mode_or_questionnaire():
    ok = {"mode": "screening", "questionnaire_hash": questionnaire_hash(), "pipeline_sha": pipeline_sha()}
    check_corpus_stamp(ok, mode="screening")
    for key, bad in (("mode", "clean"), ("questionnaire_hash", "x"), ("pipeline_sha", "y")):
        with pytest.raises(SystemExit, match="does not match"):
            check_corpus_stamp({**ok, key: bad}, mode="screening")


# --------------------------------------------------------------------------- information boundary
def test_jev_sees_only_the_masked_text_and_the_questions():
    t = _Jev()
    score_corpus(_corpus(EARN, EVENT), _client(t), concurrency=2)
    assert len(t.bodies) == 2
    for body, row in zip(sorted(t.bodies, key=lambda b: len(b["questions"]), reverse=True), (EARN, EVENT)):
        assert set(body) == {"model", "state", "questions"}
        assert body["state"] == row["text"]
        asked = [q.request() for q in questions_for(row["items"].split(","))]
        assert list(body["questions"].values()) == asked
        blob = json.dumps(body)
        for leak in (row["accession"], "ACME", "2019", str(row["cik"]) + ",", row["text_sha256"]):
            assert leak not in blob, f"{leak!r} reached Jev"


def test_a_frame_carrying_returns_is_refused_before_any_call():
    t = _Jev()
    with pytest.raises(ValueError, match="corpus frame only"):
        score_corpus(_corpus(EARN).assign(fwd_ret_5d=0.012), _client(t), concurrency=1)
    with pytest.raises(ValueError, match="missing"):
        score_corpus(_corpus(EARN).drop(columns=["qids"]), _client(t), concurrency=1)
    assert t.bodies == []


def test_a_corpus_row_scoped_by_another_questionnaire_is_refused():
    drifted = {**EARN, "qids": "M1"}
    with pytest.raises(ValueError, match="not this questionnaire's scope"):
        score_corpus(_corpus(drifted), _client(_Jev()), concurrency=1)


# --------------------------------------------------------------------------- scores
def test_block_scores_follow_the_preregistered_mapping():
    df, stats = score_corpus(_corpus(EARN, EVENT, EMPTY), _client(_Jev()), concurrency=2)
    earn, event, empty = (df.iloc[i] for i in range(3))
    # surprise: +E1, -E2, +E3 on [-1, 1] (score 3 of 0..4 -> 0.5); tone: -E4; quality: -E5, -M1
    assert earn.surprise == pytest.approx((0.9 - 0.1 + 0.5) / 3)
    assert earn.tone == pytest.approx(-0.2) and earn.quality == pytest.approx((-0.3 - 0.6) / 2)
    assert (earn.ans_E1, earn.ans_E3, earn.ans_M1) == pytest.approx((0.9, 0.5, 0.6))
    assert math.isnan(event.surprise) and math.isnan(event.tone), "an unasked block must be NaN, never 0"
    assert event.quality == pytest.approx(-0.6) and math.isnan(event.ans_E1)
    assert all(math.isnan(empty[b]) for b in ("surprise", "tone", "quality")) and empty.n_answered == 0
    assert (stats.scored, stats.empty_text, stats.questions) == (2, 1, 7)
    assert list(df.served_model) == ["jev-1.13.0", "jev-1.13.0", ""]
    assert str(df.accepted_utc.dtype) == "datetime64[ns]" and set(df.questionnaire_hash) == {questionnaire_hash()}


def test_empty_text_is_never_sent():
    t = _Jev()
    score_corpus(_corpus(EMPTY), _client(t), concurrency=1)
    assert t.bodies == []


def test_a_second_served_version_stops_the_run():
    t = _Jev(served=("jev-1.13.0", "jev-1.14.0"))
    with pytest.raises(ServedVersionChanged, match="new model"):
        score_corpus(_corpus(EARN, EVENT), _client(t), concurrency=1, chunk=1)


def test_a_rerun_replays_the_cache_without_calls(tmp_path):
    first, _ = score_corpus(_corpus(EARN, EVENT), _client(_Jev(), tmp_path), concurrency=2)
    t = _Jev()
    again, stats = score_corpus(_corpus(EARN, EVENT), _client(t, tmp_path), concurrency=2)
    assert t.bodies == [] and stats.served_models == {"jev-1.13.0": 7}
    pd.testing.assert_frame_equal(first, again)


def test_filing_scores_carry_one_row_per_share_class():
    df, _ = score_corpus(_corpus({**EARN, "tickers": "GOOG,GOOGL"}, EVENT), _client(_Jev()), concurrency=1)
    fs = to_filing_scores(df)
    assert list(fs.ticker) == ["GOOG", "GOOGL", "ACME"]
    assert fs.surprise[0] == fs.surprise[1] == pytest.approx((0.9 - 0.1 + 0.5) / 3)
    assert fs.accepted_utc.dtype == np.dtype("datetime64[ns]") and math.isnan(fs.surprise[2])
