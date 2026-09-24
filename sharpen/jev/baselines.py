"""Non-LLM text baselines for P3 (Phase 2 step 5; prereg §5, architecture §5, ADR-9).

P3 asks whether Jev adds information beyond two cheap text measures computed on the same documents (the corpus text
Jev reads) and released on the same rows:

* **Loughran–McDonald net tone:** (positive − negative) / words, with the LM Master Dictionary's sentiment lists.
  The dictionary is operator-supplied (ADR-9: an academic-only license, used here as research). Code never downloads
  it and git never holds it; the loader refuses a missing file, an unpinned one, or one whose sha256 differs from
  the pin in ``configs/atl_jev_baselines.yaml``.
* **Prior-release similarity (Lazy Prices):** the cosine similarity of word counts between an earnings release
  and the same company's previous earnings release in the corpus. The first release has none (NaN).

Both are defined on earnings releases only (NaN for event filings). They become daily signals through the step-2
construction, the latest release within the hold window on its LEAK-2 release row, so they occupy exactly the rows
the Jev signal uses.
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sharpen.jev.release import ReleaseRule
from sharpen.signals.library.jev_filings import Construction, FilingScores, JevFilingSignal

_WORD = re.compile(r"[A-Za-z]+")
BASELINE_COLUMNS = ("lm_tone", "similarity")


def words(text: str) -> list[str]:
    """Alphabetic tokens, upper-cased (the dictionary's own case)."""
    return [w.upper() for w in _WORD.findall(text)]


@dataclass(frozen=True)
class Lexicon:
    positive: frozenset[str]
    negative: frozenset[str]
    sha256: str


def load_lm_lexicon(path: str | Path, *, sha256: str | None) -> Lexicon:
    """The LM Master Dictionary's Positive and Negative lists. A word belongs to a list when its column holds a
    positive number (the year it was added); zero means never, a negative number means removed."""
    if not sha256:
        raise ValueError("the Loughran-McDonald lexicon is not pinned (configs/atl_jev_baselines.yaml "
                         "lm_lexicon.sha256): P3 cannot run without its baseline")
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"operator-supplied Loughran-McDonald lexicon not found at {p} (ADR-9)")
    raw = p.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != sha256:
        raise ValueError(f"{p.name} sha256 {got} != pinned {sha256}: not the registered dictionary release")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    cols = {c.strip().lower(): c for c in reader.fieldnames or []}
    missing = {"word", "positive", "negative"} - set(cols)
    if missing:
        raise ValueError(f"{p.name} lacks columns {sorted(missing)}")
    pos, neg = set(), set()
    for r in reader:
        w = str(r[cols["word"]]).strip().upper()
        if float(r[cols["positive"]] or 0) > 0:
            pos.add(w)
        if float(r[cols["negative"]] or 0) > 0:
            neg.add(w)
    return Lexicon(frozenset(pos), frozenset(neg), got)


def lm_net_tone(text: str, lex: Lexicon) -> float:
    """(positive − negative) / words; NaN for a text with no words."""
    ws = words(text)
    if not ws:
        return math.nan
    return (sum(w in lex.positive for w in ws) - sum(w in lex.negative for w in ws)) / len(ws)


def cosine(a: Mapping[str, int], b: Mapping[str, int]) -> float:
    """Cosine similarity of two count vectors; NaN when either is empty."""
    if not a or not b:
        return math.nan
    dot = sum(a[w] * b[w] for w in a.keys() & b.keys())
    return dot / (math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values())))


def prior_similarity(corpus: pd.DataFrame) -> pd.Series:
    """Per corpus row: cosine to the same CIK's previous earnings release (by acceptance time). NaN for event
    filings and for a CIK's first release."""
    out = pd.Series(np.nan, index=corpus.index, dtype=np.float64)
    earn = corpus[corpus["earnings"].astype(bool)].sort_values(["cik", "accepted_utc", "accession"])
    prev_cik, prev_counts = None, None
    for idx, r in earn.iterrows():
        counts = Counter(words(r["text"]))
        if r["cik"] == prev_cik and prev_counts is not None:
            out[idx] = cosine(counts, prev_counts)
        prev_cik, prev_counts = r["cik"], counts
    return out


def baseline_scores(corpus: pd.DataFrame, lex: Lexicon) -> pd.DataFrame:
    """Per corpus row: the two baselines on earnings releases, NaN on event filings."""
    earn = corpus["earnings"].astype(bool)
    tone = [lm_net_tone(t, lex) if e else math.nan for t, e in zip(corpus["text"], earn)]
    out = corpus[["accession", "cik", "tickers", "accepted_utc"]].copy()
    out["lm_tone"] = np.asarray(tone, dtype=np.float64)
    out["similarity"] = prior_similarity(corpus).to_numpy()
    return out


def baseline_signal(name: str, scores: pd.DataFrame, column: str, hold_days: int, construction: Construction,
                    calendar: np.ndarray, rule: ReleaseRule,
                    min_filing_accepted: np.datetime64 | None = None) -> JevFilingSignal:
    """A daily baseline signal built by the Jev signal's own machinery: the value rides the surprise slot, whose
    registered rule (``latest_in_window``) is the one the architecture prescribes for baselines."""
    if column not in BASELINE_COLUMNS:
        raise ValueError(f"unknown baseline {column!r}; have {BASELINE_COLUMNS}")
    ex = scores.assign(ticker=scores["tickers"].astype(str).str.split(",")).explode("ticker", ignore_index=True)
    ex = ex[ex["ticker"].astype(str).str.len() > 0]
    nan = np.full(len(ex), np.nan)
    fs = FilingScores(ticker=ex["ticker"].to_numpy(dtype=str),
                      accepted_utc=pd.to_datetime(ex["accepted_utc"]).to_numpy(dtype="datetime64[ns]"),
                      surprise=ex[column].to_numpy(dtype=np.float64), tone=nan, quality=nan.copy())
    return JevFilingSignal(name, ("surprise",), hold_days, fs, construction, calendar, rule, min_filing_accepted)
