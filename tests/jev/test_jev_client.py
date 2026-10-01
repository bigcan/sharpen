"""JevClient — fixture-driven, no network or key ever touched (every test injects ``transport``).

The families that matter:
  * FAIL-CLOSED / NEVER-SILENT: a missing key, a dead API or a malformed answer RAISES — a research
    probe must never score a missing answer as "uncertain" (the ranker's degrade-to-identity is right
    for an optional reorder and wrong here).
  * CACHE = REPRODUCIBILITY RECORD: a cached question is never re-sent, survives a new client, and is
    never overwritten.
  * KEY HYGIENE: the key reaches only the Authorization header, never the body or the logs.
"""
from __future__ import annotations

import io
import logging
import urllib.error

import pytest

from sharpen.jev import JevClient, JevError, question_key

_SECRET = "sk-test-sentinel-DO-NOT-LEAK"


def _noul(text: str) -> dict:
    return {"type": "noul", "instructions": text}


class _Transport:
    """Answers every noul question with a fixed probability; records every call."""

    def __init__(self, *, p: float = 0.8, served: str = "jev-9.9.9", fail_codes: list[int] | None = None,
                 response_override: dict | None = None):
        self.p, self.served = p, served
        self.fail_codes = list(fail_codes or [])
        self.response_override = response_override
        self.calls: list[tuple[str, dict, dict]] = []

    def __call__(self, url: str, headers: dict, body: dict) -> dict:
        self.calls.append((url, headers, body))
        if self.fail_codes:
            code = self.fail_codes.pop(0)
            raise urllib.error.HTTPError(url, code, "fail", hdrs=None, fp=io.BytesIO(b"rate limited"))
        if self.response_override is not None:
            return self.response_override
        answers = {}
        for qid, q in body["questions"].items():
            if q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": self.p}
            elif q["type"] == "score":
                answers[qid] = {"type": "score", "score": 1.5, "confidence": 0.9,
                                "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5}}
            else:
                first = next(iter(q["criteria"]))
                answers[qid] = {"type": "choice", "choice": first, "confidence": 1.0,
                                "probabilities": {first: 1.0}}
        return {"model": self.served, "answers": answers,
                "usage": {"input_tokens": 100 * len(answers), "output_tokens": 5}}


def _client(transport, tmp_path=None, **kw) -> JevClient:
    return JevClient(api_key=_SECRET, transport=transport, sleep=lambda s: None,
                     cache_path=None if tmp_path is None else tmp_path / "jev.sqlite", **kw)


def test_answers_map_back_to_caller_ids_with_types_and_served_version():
    t = _Transport(p=0.73)
    qs = {"hot": _noul("Is it hot?"),
          "level": {"type": "score", "instructions": "How warm?", "criteria": ["cold", "mild", "hot"]},
          "season": {"type": "choice", "instructions": "Season?",
                     "criteria": {"summer": "summer", "winter": "winter"}}}
    out = _client(t).ask("38C at noon", qs)
    assert list(out) == ["hot", "level", "season"]
    assert out["hot"].value == pytest.approx(0.73) and out["hot"].qtype == "noul"
    assert out["level"].value == pytest.approx(1.5) and out["level"].probabilities["2"] == 0.5
    assert out["season"].value == "summer"
    assert {a.served_model for a in out.values()} == {"jev-9.9.9"}


def test_cache_hit_never_resends_and_survives_a_new_client(tmp_path):
    t = _Transport()
    qs = {"a": _noul("A?"), "b": _noul("B?")}
    first = _client(t, tmp_path).ask("state", qs)
    assert len(t.calls) == 1
    again = _client(t, tmp_path).ask("state", qs)          # fresh client, same cache file
    assert len(t.calls) == 1, "a cached question was re-sent"
    assert {k: v.value for k, v in again.items()} == {k: v.value for k, v in first.items()}


def test_cache_is_first_write_wins(tmp_path):
    _client(_Transport(p=0.9), tmp_path).ask("s", {"a": _noul("A?")})
    c2 = _client(_Transport(p=0.1), tmp_path)
    c2._cache.put_many({question_key("jev-latest", "s", _noul("A?")):          # direct overwrite attempt
                        c2._call("s", [("a", _noul("A?"))])["a"]})
    assert c2.ask("s", {"a": _noul("A?")})["a"].value == pytest.approx(0.9)


def test_returned_answer_is_the_cached_one_when_another_writer_won_the_race(tmp_path):
    """Two threads can both miss and both call; first-write-wins keeps one answer. The run must use
    the STORED answer, or the run and its cache replay disagree (reproducibility tripwire)."""
    q = _noul("A?")
    c = _client(None, tmp_path)
    rival = _Transport(p=0.9, served="jev-rival")

    def racing_transport(url, headers, body):
        c._cache.put_many({question_key("jev-latest", "s", q):          # the other thread lands first
                           _client(rival)._call("s", [("a", q)])["a"]})
        return _Transport(p=0.1)(url, headers, body)

    c._transport = racing_transport
    got = c.ask("s", {"a": q})["a"]
    assert got.value == pytest.approx(0.9) and got.served_model == "jev-rival"
    assert _client(_Transport(p=0.5), tmp_path).ask("s", {"a": q})["a"].value == pytest.approx(0.9)


def test_only_misses_are_sent_and_chunking_respects_the_cap(tmp_path):
    t = _Transport()
    c = _client(t, tmp_path, max_questions_per_request=64)
    c.ask("s", {f"q{i}": _noul(f"Q{i}?") for i in range(5)})
    t.calls.clear()
    c.ask("s", {f"q{i}": _noul(f"Q{i}?") for i in range(70)})        # 5 cached + 65 new
    assert [len(b["questions"]) for _, _, b in t.calls] == [64, 1]
    assert c.usage.cache_hits == 5


def test_missing_key_fails_closed(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(JevError, match="TYPESAFE_API_KEY"):
        JevClient(api_key=None).ask("s", {"a": _noul("A?")})


def test_retryable_http_is_retried_then_succeeds():
    t = _Transport(fail_codes=[429, 503])
    c = _client(t, max_retries=4)
    assert c.ask("s", {"a": _noul("A?")})["a"].value == pytest.approx(0.8)
    assert c.usage.retries == 2 and c.usage.requests == 1


def test_overloaded_529_is_retried_not_fatal():
    t = _Transport(fail_codes=[529, 529])
    c = _client(t, max_retries=4)
    assert c.ask("s", {"a": _noul("A?")})["a"].value == pytest.approx(0.8)
    assert c.usage.retries == 2


def test_non_retryable_http_raises_immediately_with_the_body():
    t = _Transport(fail_codes=[400])
    with pytest.raises(JevError, match="400.*rate limited"):
        _client(t, max_retries=4).ask("s", {"a": _noul("A?")})
    assert len(t.calls) == 1


def test_retries_exhausted_raises():
    with pytest.raises(JevError, match="429"):
        _client(_Transport(fail_codes=[429] * 3), max_retries=2).ask("s", {"a": _noul("A?")})


def test_missing_answer_raises_never_defaults():
    bad = {"model": "jev-9.9.9", "answers": {"q0": {"type": "noul", "noul": 0.6}}}   # q1 missing
    with pytest.raises(JevError, match="shape"):
        _client(_Transport(response_override=bad)).ask("s", {"a": _noul("A?"), "b": _noul("B?")})


def test_answer_type_mismatch_raises():
    bad = {"model": "m", "answers": {"q0": {"type": "score", "score": 1.0}}}
    with pytest.raises(JevError, match="does not match"):
        _client(_Transport(response_override=bad)).ask("s", {"a": _noul("A?")})


@pytest.mark.parametrize("q", [
    {"type": "yesno", "instructions": "x"},
    {"type": "noul", "instructions": "  "},
    {"type": "score", "instructions": "x", "criteria": ["only-one"]},
    {"type": "score", "instructions": "x", "criteria": "cold,hot"},          # a str is a Sequence too
    {"type": "choice", "instructions": "x", "criteria": ["not", "a", "map"]},
])
def test_malformed_questions_are_rejected_before_any_call(q):
    t = _Transport()
    with pytest.raises(ValueError):
        _client(t).ask("s", {"a": q})
    assert t.calls == []


def test_key_only_in_authorization_header_never_in_body_or_logs(caplog):
    t = _Transport(fail_codes=[429])
    with caplog.at_level(logging.DEBUG, logger="sharpen.jev"):
        _client(t).ask("s", {"a": _noul("A?")})
    _, headers, body = t.calls[-1]
    assert headers["Authorization"] == f"Bearer {_SECRET}"
    assert _SECRET not in repr(body)
    assert _SECRET not in caplog.text


def test_usage_and_cost_accounting():
    c = _client(_Transport())
    c.ask("s", {"a": _noul("A?"), "b": _noul("B?")})
    assert c.usage.input_tokens == 200 and c.usage.questions_sent == 2
    rep = c.usage.to_json(usd_per_m_input=0.042)
    assert rep["cost_usd"] == pytest.approx(200 * 0.042 / 1e6)
    assert rep["served_models"] == {"jev-9.9.9": 2}
