"""Typed client for TypeSafe's Jev System One model (ATL x Jev workstream, plan item B1).

Jev answers typed questions about a text STATE — ``noul`` (P(yes)), ``score`` (a position on an
ordered rubric) and ``choice`` (one option id) — and never generates text. Three properties of that
contract shape this client:

* Questions are INDEPENDENT ("one question's answer is not hidden context for another"), so the cache
  is per QUESTION, keyed on (requested model, state, question). Batches can be re-chunked freely and a
  re-run of a probe costs nothing.
* ``jev-latest`` is an ALIAS. The served version (``jev-1.13.0`` on 2026-09-23) comes back in every
  response and is stored on every answer; a study that mixes served versions is mixing models (plan
  C6), which :meth:`JevUsage.to_json` makes visible.
* A live call is not bit-reproducible, so the cache IS the reproducibility record: probes and the
  strategy replay from it rather than re-calling.

Fail-closed like :class:`~sharpen.crucible.agentic.JevRanker`: a cache miss with no
``TYPESAFE_API_KEY`` raises. Unlike the ranker — which degrades a failed batch to identity order,
the right call for an optional reordering — this client RAISES after retries on any transport or
shape failure: a research probe must never score a missing answer as "uncertain".
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("sharpen.jev")

API_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
QUESTION_TYPES = frozenset({"noul", "score", "choice"})
# 529 is TypeSafe's "system_overloaded, try again later" (first seen 2026-09-24, corpus scoring).
_RETRYABLE_HTTP = frozenset({408, 429, 500, 502, 503, 504, 529})

Transport = Callable[[str, dict, dict], dict]


class JevError(RuntimeError):
    """A Jev call failed after retries, or returned a shape this client cannot trust."""


@dataclass(frozen=True)
class JevAnswer:
    """One typed answer. ``value`` is P(yes) for ``noul``, the rubric position for ``score`` and the
    option id for ``choice``."""

    qtype: str
    value: float | str
    confidence: float | None
    probabilities: dict[str, float] | None
    served_model: str

    def to_json(self) -> dict:
        return {"qtype": self.qtype, "value": self.value, "confidence": self.confidence,
                "probabilities": self.probabilities, "served_model": self.served_model}

    @classmethod
    def from_json(cls, d: Mapping) -> "JevAnswer":
        return cls(str(d["qtype"]), d["value"], d.get("confidence"), d.get("probabilities"),
                   str(d["served_model"]))


def question_key(model: str, state: str, question: Mapping) -> str:
    """Cache key of one question: canonical JSON of (requested model, state, question)."""
    payload = json.dumps({"model": model, "state": state, "question": question},
                         sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_question(qid: str, q: Mapping) -> None:
    qtype = q.get("type")
    if qtype not in QUESTION_TYPES:
        raise ValueError(f"question {qid!r}: type must be one of {sorted(QUESTION_TYPES)}, got {qtype!r}")
    if not isinstance(q.get("instructions"), str) or not q["instructions"].strip():
        raise ValueError(f"question {qid!r}: non-empty 'instructions' required")
    crit = q.get("criteria")
    if qtype == "score" and not (isinstance(crit, Sequence) and not isinstance(crit, str) and len(crit) >= 2):
        raise ValueError(f"question {qid!r}: 'score' needs an ordered 'criteria' list of >= 2 levels")
    if qtype == "choice" and not (isinstance(q.get("criteria"), Mapping) and len(q["criteria"]) >= 2):
        raise ValueError(f"question {qid!r}: 'choice' needs a 'criteria' map of >= 2 options")


def _parse_answer(raw: Mapping, qtype: str, served_model: str) -> JevAnswer:
    rtype = raw.get("type", qtype)
    if rtype != qtype:
        raise JevError(f"answer type {rtype!r} does not match question type {qtype!r}")
    if qtype == "noul":
        value: float | str = float(raw["noul"])
    elif qtype == "score":
        value = float(raw["score"])
    else:
        value = str(raw["choice"])
    conf = raw.get("confidence")
    probs = raw.get("probabilities")
    return JevAnswer(qtype, value, None if conf is None else float(conf),
                     None if probs is None else {str(k): float(v) for k, v in probs.items()},
                     served_model)


def _default_transport(url: str, headers: dict, body: dict) -> dict:
    """Live HTTPS POST -> parsed JSON. ``HTTPError`` propagates so the caller can classify it."""
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers,
                                 method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:     # noqa: S310 - fixed https host
        return json.loads(resp.read().decode("utf-8"))


class _AnswerCache:
    """Per-question answer store (SQLite). First answer wins: a cached record is never overwritten,
    because it is the reproducibility record of what the model said."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        with self._lock:
            self._db.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, "
                             "answer TEXT NOT NULL, served_model TEXT NOT NULL, created_utc TEXT NOT NULL)")
            self._db.commit()

    def get_many(self, keys: Sequence[str]) -> dict[str, JevAnswer]:
        out: dict[str, JevAnswer] = {}
        uniq = list(dict.fromkeys(keys))
        with self._lock:
            for i in range(0, len(uniq), 500):
                part = uniq[i:i + 500]
                rows = self._db.execute(
                    f"SELECT key, answer FROM answers WHERE key IN ({','.join('?' * len(part))})",
                    part).fetchall()
                out.update({k: JevAnswer.from_json(json.loads(a)) for k, a in rows})
        return out

    def put_many(self, items: Mapping[str, JevAnswer]) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            self._db.executemany(
                "INSERT OR IGNORE INTO answers (key, answer, served_model, created_utc) VALUES (?,?,?,?)",
                [(k, json.dumps(a.to_json()), a.served_model, now) for k, a in items.items()])
            self._db.commit()


@dataclass
class JevUsage:
    """Running totals for one client. ``served_models`` counts answers by served version, cached
    answers included, so a mixed-version study is visible in every artifact."""

    requests: int = 0
    questions_sent: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0
    latencies_s: list[float] = field(default_factory=list)
    served_models: dict[str, int] = field(default_factory=dict)

    def cost_usd(self, usd_per_m_input: float) -> float:
        return self.input_tokens * usd_per_m_input / 1e6

    def to_json(self, usd_per_m_input: float | None = None) -> dict:
        lat = sorted(self.latencies_s)

        def pct(p: float) -> float | None:
            return None if not lat else round(lat[min(len(lat) - 1, int(p * len(lat)))], 3)

        out = {"requests": self.requests, "questions_sent": self.questions_sent,
               "cache_hits": self.cache_hits, "input_tokens": self.input_tokens,
               "output_tokens": self.output_tokens, "retries": self.retries,
               "latency_s": {"p50": pct(0.50), "p90": pct(0.90), "max": pct(1.0)},
               "served_models": dict(self.served_models)}
        if usd_per_m_input is not None:
            out["cost_usd"] = round(self.cost_usd(usd_per_m_input), 9)   # probe costs are sub-cent
        return out


class JevClient:
    """Typed, cached, fail-closed Jev client.

    Inject ``transport`` and no network or key is touched (the ``JevRanker`` / connector seam). The
    live path needs ``TYPESAFE_API_KEY``; the key is only ever placed in the ``Authorization`` header
    and is never logged.
    """

    def __init__(self, api_key: str | None = None, *, model: str = DEFAULT_MODEL,
                 transport: Transport | None = None, cache_path: str | Path | None = None,
                 max_questions_per_request: int = 64, max_retries: int = 4,
                 backoff_s: float = 2.0, sleep: Callable[[float], None] = time.sleep) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY", "")
        self.model = model
        self._transport = transport
        self._cache = _AnswerCache(Path(cache_path)) if cache_path is not None else None
        self.max_questions_per_request = max(1, int(max_questions_per_request))
        self.max_retries = max(0, int(max_retries))
        self.backoff_s = float(backoff_s)
        self._sleep = sleep
        self._lock = threading.Lock()
        self.usage = JevUsage()

    # -- public ---------------------------------------------------------------------------------
    def ask(self, state: str, questions: Mapping[str, Mapping]) -> dict[str, JevAnswer]:
        """Answer every question about ``state``, from cache where possible. Returns answers keyed
        by the caller's question ids, in the caller's order."""
        for qid, q in questions.items():
            _validate_question(qid, q)
        keys = {qid: question_key(self.model, state, q) for qid, q in questions.items()}
        cached = self._cache.get_many(list(keys.values())) if self._cache is not None else {}
        out = {qid: cached[k] for qid, k in keys.items() if k in cached}
        missing = [qid for qid in questions if qid not in out]
        step = self.max_questions_per_request
        for start in range(0, len(missing), step):
            chunk = missing[start:start + step]
            fresh = self._call(state, [(qid, questions[qid]) for qid in chunk])
            if self._cache is not None:
                self._cache.put_many({keys[qid]: fresh[qid] for qid in chunk})
                # Return what the CACHE holds, not what this call said: if another thread answered the
                # same question first, first-write-wins kept ITS answer, and a re-run will replay that
                # one — so this run must use it too, or the run and its replay disagree.
                stored = self._cache.get_many([keys[qid] for qid in chunk])
                fresh = {qid: stored.get(keys[qid], fresh[qid]) for qid in chunk}
            out.update(fresh)
        with self._lock:
            self.usage.cache_hits += len(questions) - len(missing)
            for a in out.values():
                self.usage.served_models[a.served_model] = self.usage.served_models.get(a.served_model, 0) + 1
        return {qid: out[qid] for qid in questions}

    def ask_many(self, jobs: Sequence[tuple[str, Mapping[str, Mapping]]], *,
                 concurrency: int = 8) -> list[dict[str, JevAnswer]]:
        """``ask`` over many (state, questions) jobs with a thread pool; results in job order."""
        with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as pool:
            return list(pool.map(lambda job: self.ask(job[0], job[1]), jobs))

    # -- the one external boundary --------------------------------------------------------------
    def _call(self, state: str, items: Sequence[tuple[str, Mapping]]) -> dict[str, JevAnswer]:
        if self._transport is None and not self._api_key:
            raise JevError("JevClient requires TYPESAFE_API_KEY for a cache miss "
                           "(fails closed, like ANTHROPIC_API_KEY / FRED_API_KEY)")
        body = {"model": self.model, "state": state,
                "questions": {f"q{i}": dict(q) for i, (_, q) in enumerate(items)}}
        headers = {"Authorization": f"Bearer {self._api_key}", "content-type": "application/json"}
        transport = self._transport or _default_transport
        resp: dict | None = None
        latency = 0.0
        for attempt in range(self.max_retries + 1):
            t0 = time.monotonic()
            try:
                resp = transport(API_URL, headers, body)
                latency = time.monotonic() - t0
                break
            except urllib.error.HTTPError as exc:
                retryable = exc.code in _RETRYABLE_HTTP
                if not retryable or attempt == self.max_retries:
                    try:
                        detail = exc.read().decode("utf-8", "replace")[:500]
                    except Exception:                               # noqa: BLE001 - best effort
                        detail = "(body unreadable)"
                    raise JevError(f"Jev HTTP {exc.code}: {detail}") from exc
                reason = f"HTTP {exc.code}"
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                if attempt == self.max_retries:
                    raise JevError(f"Jev transport failed after {attempt + 1} attempts: {exc!r}") from exc
                reason = type(exc).__name__
            wait = self.backoff_s * (2 ** attempt)
            logger.warning("Jev call failed (%s); retry %d/%d in %.1fs", reason, attempt + 1,
                           self.max_retries, wait)
            with self._lock:
                self.usage.retries += 1
            self._sleep(wait)
        if resp is None:                                    # unreachable: the loop breaks or raises
            raise JevError("Jev call produced no response")
        try:
            served = str(resp.get("model") or self.model)
            raw = resp["answers"]
            parsed = {qid: _parse_answer(raw[f"q{i}"], str(q["type"]), served)
                      for i, (qid, q) in enumerate(items)}
            usage = resp.get("usage") or {}
            in_tok, out_tok = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
        except JevError:
            raise
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise JevError(f"unexpected Jev response shape: {exc!r}") from exc
        with self._lock:
            self.usage.requests += 1
            self.usage.questions_sent += len(items)
            self.usage.input_tokens += in_tok
            self.usage.output_tokens += out_tok
            self.usage.latencies_s.append(latency)
        return parsed
