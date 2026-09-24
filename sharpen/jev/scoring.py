"""Score the ATL x Jev corpus with the frozen questionnaire (Phase 2 step 4; architecture §3).

* **Fail closed on the instrument.** ``phase1.questionnaire`` must name this code's version and hash, and the
  corpus must have been built under the same questionnaire and text pipeline. Otherwise nothing is scored.
* **Information boundary.** Jev receives the masked filing text and the questions, nothing else. The scorer accepts
  only the corpus's own columns, so a frame that carries prices, returns or verdicts is refused before any call.
* **One model.** Every answer records the served Jev version. A second version appearing during a run stops it
  (plan C6): mixing versions would be mixing models.
* **Scores.** A filing's block score is :func:`sharpen.jev.questionnaire.filing_score` over that block's questions;
  a block the filing was not asked is NaN, never 0. The mapped answer to every question is kept for audits.
* **Resume.** Answers come from the shared per-question cache, so an interrupted run replays them for free.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from sharpen.jev.client import JevAnswer, JevClient
from sharpen.jev.corpus import SHARD_COLUMNS, pipeline_sha
from sharpen.jev.questionnaire import BLOCKS, QUESTIONS_V1, VERSION, filing_score, map_answer, questionnaire_hash, \
    questions_for
from sharpen.signals.library.jev_filings import FilingScores

ANSWER_COLUMNS = tuple(f"ans_{q.qid}" for q in QUESTIONS_V1)
SCORE_COLUMNS = ("accession", "cik", "tickers", "accepted_utc", "text_sha256", *BLOCKS, "n_answered",
                 "served_model", "questionnaire_hash", *ANSWER_COLUMNS)


class ServedVersionChanged(RuntimeError):
    """Jev served a second model version within one scoring run (plan C6)."""


def check_questionnaire(phase1: Mapping) -> None:
    """``phase1.questionnaire`` must equal this code's questionnaire, or the run is refused."""
    q = phase1.get("questionnaire") or {}
    if str(q.get("version")) != VERSION or str(q.get("hash")) != questionnaire_hash():
        raise SystemExit(f"phase1.questionnaire {dict(q)} != this code's {VERSION}/{questionnaire_hash()}: "
                         "refusing to score with an instrument that is not the pre-registered one")


def check_corpus_stamp(stamp: Mapping, *, mode: str) -> None:
    """The corpus was built for ``mode`` by the current questionnaire scope and text pipeline."""
    problems = []
    if stamp.get("mode") != mode:
        problems.append(f"mode {stamp.get('mode')!r} != {mode!r}")
    if stamp.get("questionnaire_hash") != questionnaire_hash():
        problems.append(f"questionnaire {stamp.get('questionnaire_hash')} != {questionnaire_hash()}")
    if stamp.get("pipeline_sha") != pipeline_sha():
        problems.append(f"text pipeline {stamp.get('pipeline_sha')} != {pipeline_sha()} (rebuild the corpus)")
    if problems:
        raise SystemExit("corpus stamp does not match this code: " + "; ".join(problems))


@dataclass
class ScoreStats:
    filings: int = 0
    scored: int = 0
    empty_text: int = 0
    questions: int = 0
    served_models: Counter = field(default_factory=Counter)

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["served_models"] = dict(self.served_models)
        return d


def _row_scores(r, answers: Mapping[str, JevAnswer] | None) -> dict:
    qs = questions_for(str(r.items).split(",") if r.items else [])
    out = {"accession": r.accession, "cik": int(r.cik), "tickers": r.tickers, "accepted_utc": r.accepted_utc,
           "text_sha256": r.text_sha256, "questionnaire_hash": questionnaire_hash(),
           **{c: math.nan for c in ANSWER_COLUMNS}}
    if answers is None:                                    # no text: no answer, no score
        return {**out, **{b: math.nan for b in BLOCKS}, "n_answered": 0, "served_model": ""}
    for b in BLOCKS:
        out[b] = filing_score(answers, qs, blocks=[b])
    for q in qs:
        out[f"ans_{q.qid}"] = map_answer(q, answers[q.qid])
    served = sorted({a.served_model for a in answers.values()})
    return {**out, "n_answered": len(answers), "served_model": ",".join(served)}


def score_corpus(corpus: pd.DataFrame, client: JevClient, *, concurrency: int, chunk: int = 256,
                 on_chunk: Callable[[int, int, ScoreStats], None] | None = None) -> tuple[pd.DataFrame, ScoreStats]:
    """Scores for every corpus row, in corpus order. Raises :class:`ServedVersionChanged` on a second model."""
    cols = set(corpus.columns)
    if cols != set(SHARD_COLUMNS):
        raise ValueError(f"the scorer reads a corpus frame only: unexpected {sorted(cols - set(SHARD_COLUMNS))}, "
                         f"missing {sorted(set(SHARD_COLUMNS) - cols)}")
    stats = ScoreStats(filings=len(corpus))
    out: list[dict] = []
    for start in range(0, len(corpus), max(1, int(chunk))):
        part = list(corpus.iloc[start:start + chunk].itertuples(index=False))
        jobs, which = [], []
        for k, r in enumerate(part):
            qs = questions_for(str(r.items).split(",") if r.items else [])
            if ",".join(q.qid for q in qs) != r.qids:
                raise ValueError(f"{r.accession}: corpus qids {r.qids!r} are not this questionnaire's scope "
                                 f"{','.join(q.qid for q in qs)!r}")
            if r.text:
                jobs.append((r.text, {q.qid: q.request() for q in qs}))
                which.append(k)
        answered = dict(zip(which, client.ask_many(jobs, concurrency=concurrency)))
        for k, r in enumerate(part):
            ans = answered.get(k)
            out.append(_row_scores(r, ans))
            if ans is None:
                stats.empty_text += 1
                continue
            stats.scored += 1
            stats.questions += len(ans)
            stats.served_models.update(a.served_model for a in ans.values())
        if len(stats.served_models) > 1:
            raise ServedVersionChanged(f"Jev served {dict(stats.served_models)} within one run; a version change "
                                       "is a new model (plan C6)")
        if on_chunk is not None:
            on_chunk(min(start + chunk, len(corpus)), len(corpus), stats)
    df = pd.DataFrame(out, columns=list(SCORE_COLUMNS))
    df["accepted_utc"] = pd.to_datetime(df["accepted_utc"]).astype("datetime64[ns]")
    return df, stats


def write_scores(df: pd.DataFrame, out_dir: str | Path, manifest: Mapping) -> Path:
    """Scores parquet, then its manifest, each replaced atomically."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "filing_scores.parquet"
    tmp = p.with_name(p.name + ".tmp")
    df.to_parquet(tmp, compression="zstd", index=False)
    os.replace(tmp, p)
    m = out / "_manifest.json"
    tmp_m = m.with_name(m.name + ".tmp")
    tmp_m.write_text(json.dumps(dict(manifest), indent=1, default=str), encoding="utf-8")
    os.replace(tmp_m, m)
    return p


def to_filing_scores(scores: pd.DataFrame) -> FilingScores:
    """One row per (filing, universe ticker): dual share classes carry the same filing scores."""
    t = scores["tickers"].astype(str).str.split(",")
    ex = scores.assign(ticker=t).explode("ticker", ignore_index=True)
    ex = ex[ex["ticker"].astype(str).str.len() > 0]
    return FilingScores(ticker=ex["ticker"].to_numpy(dtype=str),
                        accepted_utc=pd.to_datetime(ex["accepted_utc"]).to_numpy(dtype="datetime64[ns]"),
                        surprise=ex["surprise"].to_numpy(dtype=np.float64),
                        tone=ex["tone"].to_numpy(dtype=np.float64),
                        quality=ex["quality"].to_numpy(dtype=np.float64))
