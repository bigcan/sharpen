"""The Jev questionnaire over 8-K filings — the strategy's pre-registration in code (plan item B2).

Every question is EXTRACTION-only: its answer is in the document, so Jev is asked to read, never to
forecast (a forecast is where a model's memory of later events leaks in — Phase 0 C1/C3). Each question
carries an a-priori SIGN and the mechanism that justifies it (``docs/research/atl_jev_phase1_research.md``):

* surprise block (E1-E3, text surprise beyond the number — PEAD.txt): outlook raised +, lowered -, forward
  demand language +;
* tone block (E4, tone management): optimism beyond what the reported results justify is - (the delayed
  negative reaction of Huang, Teoh & Zhang 2014);
* quality block (E5, M1): one-off-flattered results and adverse developments are - (earnings quality; the
  negative post-filing drift of 8-K events). A draft M2 (unplanned CEO/CFO exit) was DROPPED by the
  pre-freeze instrument check: 17/17 sampled Item 5.02 filings were routine and it never varied.

Signs are applied INSIDE :func:`filing_score`, so a higher score always means better news and every signal
built on it is registered with ``expected_sign = +1``.

The wording, criteria, signs and scope are content-hashed by :func:`questionnaire_hash`. The hash is written
into the pre-registration and the gates file BEFORE any filing is scored; editing anything here is a new
questionnaire (a new trial), exactly like editing a gates file.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass

from .client import JevAnswer

VERSION = "v1"
EARNINGS_ITEM = "2.02"
# 8-K items whose filings are in scope for the adverse-development question (plus every 2.02 release):
# termination of a material agreement, bankruptcy, accelerated obligation, exit costs, impairment,
# delisting notice, auditor change, non-reliance on financials. (5.02 left scope with the dropped M2.)
ADVERSE_ITEMS = frozenset({"1.02", "1.03", "2.04", "2.05", "2.06", "3.01", "4.01", "4.02"})
# The scoring rule is part of the pre-registration: it is hashed with the questions, so changing the
# mapping moves the hash exactly like rewording a question (Math audit M-1 / hash-coverage finding).
SCORING = "noul->p on [0,1]; score->2s/(L-1)-1 on [-1,1]; filing=mean(sign*mapped) over answered questions"
BLOCKS = ("surprise", "tone", "quality")

PREAMBLE = ("Answer only from what this document itself states. Do not use outside knowledge about the "
            "company, its industry or later events. Names, tickers, places and dates may have been masked. ")


@dataclass(frozen=True)
class Question:
    qid: str
    qtype: str                         # "noul" | "score"
    text: str
    sign: int                          # +1: a high answer is good news; -1: a high answer is bad news
    scope: str                         # "earnings" | "adverse"
    block: str                         # one of BLOCKS
    criteria: tuple[str, ...] = ()     # ordered levels for "score"; empty for "noul"
    mechanism: str = ""

    def request(self) -> dict:
        """The Jev question payload (PREAMBLE + text; ordered criteria for a score)."""
        q: dict = {"type": self.qtype, "instructions": PREAMBLE + self.text}
        if self.qtype == "score":
            q["criteria"] = list(self.criteria)
        return q


QUESTIONS_V1: tuple[Question, ...] = (
    Question("E1", "noul",
             "Does the document state that the company raised its financial outlook or guidance (for example "
             "revenue, earnings per share, margin or cash-flow guidance) compared with the outlook it had "
             "previously given?", +1, "earnings", "surprise",
             mechanism="text surprise beyond the number (PEAD.txt)"),
    Question("E2", "noul",
             "Does the document state that the company lowered or withdrew its financial outlook or guidance "
             "compared with the outlook it had previously given?", -1, "earnings", "surprise",
             mechanism="text surprise beyond the number (PEAD.txt)"),
    Question("E3", "score",
             "How does management describe the forward trajectory of demand, orders, bookings, backlog or "
             "business momentum?", +1, "earnings", "surprise",
             criteria=("sharply deteriorating", "softening", "stable, or not discussed", "improving",
                       "strongly improving"),
             mechanism="forward details behind the earnings number (PEAD.txt)"),
    Question("E4", "noul",
             "Is the language of the document noticeably more optimistic than the financial results it "
             "reports would justify?", -1, "earnings", "tone",
             mechanism="tone management: delayed negative reaction (Huang, Teoh & Zhang 2014)"),
    Question("E5", "noul",
             "Are the headline results materially flattered by one-time, non-recurring or non-operating items "
             "that the document itself identifies (for example gains on asset sales, tax benefits or reserve "
             "releases)?", -1, "earnings", "quality",
             mechanism="earnings quality: investors fixate on headline earnings"),
    Question("M1", "noul",
             "Does the document disclose a material adverse development, for example an impairment charge, "
             "restructuring or exit costs, a restatement or non-reliance on prior financial statements, the "
             "loss of a major customer or contract, a covenant breach or going-concern doubt, or the "
             "acceleration of a financial obligation?", -1, "adverse", "quality",
             mechanism="negative post-filing drift of adverse 8-K events (Lerman & Livnat 2010)"),
)


def questionnaire_hash(questions: Sequence[Question] = QUESTIONS_V1) -> str:
    """12-hex content hash of everything that defines the questionnaire: wording, criteria, signs, blocks,
    scopes, which 8-K items put a filing in scope, AND the scoring rule (order-insensitive by qid)."""
    rows = [asdict(q) for q in sorted(questions, key=lambda q: q.qid)]
    payload = json.dumps({"version": VERSION, "preamble": PREAMBLE, "questions": rows, "scoring": SCORING,
                          "scope_items": {"earnings": EARNINGS_ITEM, "adverse": sorted(ADVERSE_ITEMS)}},
                         sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def questions_for(items: Iterable[str], questions: Sequence[Question] = QUESTIONS_V1) -> list[Question]:
    """The questions a filing with these 8-K ``items`` must answer; empty = the filing is out of scope."""
    items = {str(i).strip() for i in items}
    earnings = EARNINGS_ITEM in items
    keep = {"earnings": earnings, "adverse": earnings or bool(items & ADVERSE_ITEMS)}
    return [q for q in questions if keep[q.scope]]


def map_answer(q: Question, ans: JevAnswer) -> float:
    """Answer BEFORE the sign.

    * ``noul`` -> p, the EVIDENCE that the event occurs, on [0, 1]. Every noul here asks "does X occur?"
      and "no" is the null state, not the opposite pole: "did it raise guidance? no" is not evidence it
      lowered it. (A 2p-1 mapping — the pre-freeze draft, rejected in the Math audit — scored every "no"
      as -1, so a filing's neutral level depended on how many negatively-signed questions it was asked,
      and a clean officer-change 8-K outranked a news-free earnings release.)
    * ``score`` -> 2s/(L-1)-1 on [-1, 1], centred on the neutral middle level ("stable, or not
      discussed"): a genuinely bipolar judgement.
    """
    if ans.qtype != q.qtype:
        raise ValueError(f"{q.qid}: answer type {ans.qtype!r} != question type {q.qtype!r}")
    v = float(ans.value)
    if q.qtype == "noul":
        return min(1.0, max(0.0, v))
    return min(1.0, max(-1.0, 2.0 * v / (len(q.criteria) - 1) - 1.0))


def filing_score(answers: Mapping[str, JevAnswer], questions: Sequence[Question],
                 blocks: Iterable[str] | None = None) -> float:
    """Mean of ``sign * map_answer`` over the answered questions in ``blocks`` (all blocks if None).
    NaN when none apply — the filing then carries no score for that signal (never a silent 0)."""
    want = set(BLOCKS if blocks is None else blocks)
    unknown = want - set(BLOCKS)
    if unknown:
        raise ValueError(f"unknown blocks {sorted(unknown)}")
    vals = [q.sign * map_answer(q, answers[q.qid]) for q in questions
            if q.block in want and q.qid in answers]
    return float(sum(vals) / len(vals)) if vals else math.nan
