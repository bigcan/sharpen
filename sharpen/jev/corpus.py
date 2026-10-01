"""The ATL x Jev filing corpus: Phase 2 step 3 (``docs/research/atl_jev_phase2_architecture.md`` §2).

Turns the S&P 500 point-in-time universe into the masked 8-K texts Jev will read, as the frozen
pre-registration describes them (``docs/research/atl_jev_prereg.md`` §2, clarified in §9):

* **Filers (ADR-4).** A universe ticker maps to SEC CIKs through exact keys only: the constituents file and
  SEC's own ``company_tickers.json``. Where the two disagree every CIK is kept and the conflict is reported
  (ExxonMobil moved to a new holding-company CIK in 2026). Unmapped tickers are reported, never guessed.
* **Scope.** Form ``8-K`` only; amendments (``8-K/A``) are excluded, as in the instrument check. A filing is in
  scope when :func:`sharpen.jev.questionnaire.questions_for` gives it any question, and it is fetched only if
  accepted while the filer was an index member (each membership spell opened early by the warm-up): a filing
  released on a non-member row can never enter a signal, which masks those rows.
* **Text.** An Item 2.02 filing is read from its press-release exhibit, or from the 8-K body if it has none. An
  event filing is the 8-K body (cover page stripped) followed by any press-release exhibit. The press release
  is exhibit 99.1 under either observed label (``EX-99.1``, ``EX-99.01``), else an unnumbered ``EX-99``.
  Masked by :func:`sharpen.jev.anonymize.anonymize` and cut to ``phase1.filings.max_chars``.
* **Firewall (ADR-3).** Screening mode ends at ``screening_window[1]``. Clean mode is P4-only and refuses to
  start without the P4 sentinel. A filing outside the build window is an error, never a silent skip.
* **Storage (ADR-6).** One zstd parquet shard per CIK, written atomically, with a sidecar holding the stamp it
  was built under. An interrupted build resumes; a changed window, questionnaire or text pipeline rebuilds.
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import logging
import math
import os
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sharpen.jev import anonymize as _anonymize_module
from sharpen.jev.anonymize import anonymize
from sharpen.jev.questionnaire import EARNINGS_ITEM, questionnaire_hash, questions_for
from sharpen.jev.stamps import source_sha

log = logging.getLogger(__name__)

UNIVERSE = "sp500_pit"
FORM = "8-K"
# The press-release exhibit in priority order: exhibit 99.1 under its two observed labels, then an unnumbered
# exhibit 99. A strict "EX-99.1" match fed Jev a one-paragraph 8-K stub for 9 of 60 sampled Item 2.02 filings.
PRESS_RELEASE_TYPES = ("EX-99.1", "EX-99.01", "EX-99")
DOC_TYPES = ("8-K", *PRESS_RELEASE_TYPES)
SHARD_COLUMNS = ("accession", "cik", "tickers", "items", "accepted_utc", "earnings", "doc_source", "qids",
                 "raw_chars", "text_len", "truncated", "text", "text_sha256", "questionnaire_hash")


class FirewallError(RuntimeError):
    """A filing outside the build window reached the builder: a broken filter, never something to skip."""


def _norm_ticker(t: object) -> str:
    return str(t).strip().upper().replace(".", "-")


def _utc_day(d: str) -> datetime:
    return datetime.fromisoformat(str(d)).replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------- universe and CIK map (ADR-4)
def load_pit_rows(path: str | Path) -> tuple[np.ndarray, list[frozenset[str]]]:
    """``sp500_pit_members.csv`` -> (change dates ``datetime64[D]``, the member set from each date on)."""
    dates, members = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            dates.append(np.datetime64(r["date"], "D"))
            members.append(frozenset(_norm_ticker(t) for t in r["tickers"].split(",") if t.strip()))
    d = np.array(dates, dtype="datetime64[D]")
    if d.size and not np.all(d[1:] > d[:-1]):
        raise ValueError(f"{path}: membership dates are not strictly increasing")
    return d, members


def _validity(dates: np.ndarray, k: int) -> tuple[np.datetime64, np.datetime64 | None]:
    return dates[k], (dates[k + 1] if k + 1 < dates.size else None)


def _check_span(dates: np.ndarray, start: np.datetime64, end: np.datetime64) -> None:
    if dates.size == 0 or start < dates[0]:
        raise ValueError(f"membership is unknown before {dates[0] if dates.size else 'any date'}; window starts {start}")
    if end > dates[-1]:
        # the last row stays in force only until the file's next refresh: past it the universe is unknown
        raise ValueError(f"membership file ends {dates[-1]}; window ends {end}. Refresh it first (ADR-8)")


def membership_spells(dates: np.ndarray, members: Sequence[frozenset[str]], tickers: Iterable[str], start: str,
                      end: str) -> list[tuple[np.datetime64, np.datetime64]]:
    """[first day, day after the last) runs within [start, end] during which any of ``tickers`` was a member."""
    s, e = np.datetime64(start, "D"), np.datetime64(end, "D")
    _check_span(dates, s, e)
    want = {_norm_ticker(t) for t in tickers}
    spells: list[tuple[np.datetime64, np.datetime64]] = []
    for k in range(dates.size):
        lo, hi = _validity(dates, k)
        lo, hi = max(lo, s), min(hi if hi is not None else e + 1, e + 1)
        if hi <= lo or not (want & members[k]):
            continue
        if spells and spells[-1][1] >= lo:
            spells[-1] = (spells[-1][0], max(spells[-1][1], hi))
        else:
            spells.append((lo, hi))
    return spells


def universe_tickers(dates: np.ndarray, members: Sequence[frozenset[str]], start: str, end: str) -> set[str]:
    """Every ticker that was an index member on some day of [start, end]."""
    s, e = np.datetime64(start, "D"), np.datetime64(end, "D")
    _check_span(dates, s, e)
    out: set[str] = set()
    for k in range(dates.size):
        lo, hi = _validity(dates, k)
        if lo > e:
            break
        if hi is not None and hi <= s:
            continue
        out |= members[k]
    return out


@dataclass(frozen=True)
class CikMap:
    mapped: dict[str, tuple[int, ...]]          # universe ticker -> CIK(s)
    unmapped: tuple[str, ...]
    conflicts: dict[str, tuple[int, ...]]       # tickers whose exact-key sources disagree (all CIKs kept)

    def filers(self) -> dict[int, tuple[str, ...]]:
        """CIK -> the universe tickers it files for (dual share classes share one CIK)."""
        out: dict[int, list[str]] = {}
        for t, ciks in sorted(self.mapped.items()):
            for c in ciks:
                out.setdefault(c, []).append(t)
        return {c: tuple(ts) for c, ts in sorted(out.items())}

    def to_json(self) -> dict:
        return {"n_mapped": len(self.mapped), "n_unmapped": len(self.unmapped),
                "mapped": {t: list(c) for t, c in self.mapped.items()}, "unmapped": list(self.unmapped),
                "conflicts": {t: list(c) for t, c in self.conflicts.items()}}


def build_cik_map(tickers: Iterable[str], *, constituents: Mapping[str, int],
                  sec_tickers: Mapping[str, int]) -> CikMap:
    """Exact-key ticker -> CIK. Source keys must already be normalized (upper case, ``.`` -> ``-``)."""
    mapped: dict[str, tuple[int, ...]] = {}
    conflicts: dict[str, tuple[int, ...]] = {}
    unmapped: list[str] = []
    for t in sorted({_norm_ticker(x) for x in tickers if str(x).strip()}):
        ciks = tuple(sorted({int(src[t]) for src in (constituents, sec_tickers) if t in src}))
        if not ciks:
            unmapped.append(t)
            continue
        mapped[t] = ciks
        if len(ciks) > 1:
            conflicts[t] = ciks
    return CikMap(mapped, tuple(unmapped), conflicts)


def load_constituents(path: str | Path) -> tuple[dict[str, int], dict[int, str]]:
    """Constituents CSV (``Symbol``, ``Security``, ``CIK``) -> (ticker -> CIK, CIK -> security name)."""
    sym: dict[str, int] = {}
    names: dict[int, str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            cik = int(r["CIK"])
            sym[_norm_ticker(r["Symbol"])] = cik
            names.setdefault(cik, r["Security"].strip())
    return sym, names


def load_sec_tickers(path: str | Path) -> tuple[dict[str, int], dict[int, tuple[str, ...]]]:
    """SEC ``company_tickers.json`` -> (ticker -> CIK, CIK -> registrant titles)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    sym: dict[str, int] = {}
    titles: dict[int, set[str]] = {}
    for r in payload.values() if isinstance(payload, dict) else payload:
        t, cik = _norm_ticker(r.get("ticker", "")), str(r.get("cik_str", r.get("cik", ""))).strip()
        if t and cik.isdigit():
            sym[t] = int(cik)
            titles.setdefault(int(cik), set()).add(str(r.get("title", "")).strip())
    return sym, {c: tuple(sorted(x for x in v if x)) for c, v in titles.items()}


def coverage_by_year(cmap: CikMap, dates: np.ndarray, members: Sequence[frozenset[str]], start: str,
                     end: str) -> dict[str, dict]:
    """Per calendar year: members, the share with a CIK, and the share of member-days (calendar days in force)."""
    s, e = np.datetime64(start, "D"), np.datetime64(end, "D")
    _check_span(dates, s, e)
    names: dict[int, set[str]] = {}
    days: dict[int, list[int]] = {}
    for k in range(dates.size):
        lo, hi = _validity(dates, k)
        lo, hi = max(lo, s), min(hi if hi is not None else e + 1, e + 1)
        if hi <= lo:
            continue
        m = members[k]
        n_mapped = sum(t in cmap.mapped for t in m)
        y0, y1 = int(str(lo)[:4]), int(str(hi - 1)[:4])
        for y in range(y0, y1 + 1):
            a = max(lo, np.datetime64(f"{y}-01-01", "D"))
            b = min(hi, np.datetime64(f"{y + 1}-01-01", "D"))
            n = int((b - a).astype(int))
            if n <= 0:
                continue
            names.setdefault(y, set()).update(m)
            acc = days.setdefault(y, [0, 0])
            acc[0] += n * len(m)
            acc[1] += n * n_mapped
    return {str(y): {"names": len(names[y]),
                     "names_mapped_share": round(sum(t in cmap.mapped for t in names[y]) / len(names[y]), 4),
                     "member_days_mapped_share": round(days[y][1] / days[y][0], 4)}
            for y in sorted(names)}


# --------------------------------------------------------------------------- build window (ADR-3)
@dataclass(frozen=True)
class CorpusWindow:
    mode: str               # "screening" | "clean"
    since: datetime         # UTC, inclusive: first acceptance time kept (screening: includes the warm-up)
    until: datetime         # UTC, exclusive
    rows_start: str         # the evaluation window whose members form the universe
    rows_end: str

    def to_json(self) -> dict:
        return {"mode": self.mode, "since": self.since.isoformat(), "until": self.until.isoformat(),
                "rows": [self.rows_start, self.rows_end]}


def warmup_start(first_day: str, max_hold_days: int) -> datetime:
    """Earliest acceptance time whose release row can still reach ``first_day``'s hold window (prereg §4.3).

    ``max_hold_days`` trading days span at most ``ceil(max_hold_days * 365 / 252)`` calendar days plus holidays;
    two weeks cover the holidays and an after-close acceptance released a day later. Starting early is
    harmless (those filings fall outside every hold window); starting late would starve the first rows.
    """
    return _utc_day(first_day) - timedelta(days=math.ceil(max_hold_days * 365 / 252) + 14)


def max_hold_days(phase1: Mapping) -> int:
    return max(int(s["hold_days"]) for s in phase1["signals"])


def acceptance_spells(spells: Iterable[tuple[np.datetime64, np.datetime64]], window: CorpusWindow,
                      hold_days: int) -> tuple[tuple[datetime, datetime], ...]:
    """Acceptance-time intervals whose filings can reach a member row: each membership spell, opened early by
    the same warm-up as the window (a filing released just before a name joins counts on its first member rows),
    and closed at the spell's end (a later release lands on a non-member row). Clipped to the build window."""
    out: list[tuple[datetime, datetime]] = []
    for a, b in spells:
        lo = max(warmup_start(str(a), hold_days), window.since)
        hi = min(_utc_day(str(b)), window.until)
        if hi <= lo:
            continue
        if out and out[-1][1] >= lo:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return tuple(out)


def _require_p4_authorization(path: str | Path | None) -> None:
    if path is None:
        raise PermissionError("clean mode is P4-only: pass the P4 sentinel (results/atl_jev/phase4/OPENED.json)")
    p = Path(path)
    if not p.is_file():
        raise PermissionError(f"no P4 sentinel at {p}: the clean window is unopened")
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PermissionError(f"unreadable P4 sentinel {p}: {exc!r}") from exc
    if "opened_utc" not in rec:
        raise PermissionError(f"{p} is not a P4 sentinel (no opened_utc)")


def corpus_window(phase1: Mapping, *, mode: str, p4_authorization: str | Path | None = None,
                  narrow: tuple[datetime, datetime] | None = None) -> CorpusWindow:
    """The acceptance-time window a build may touch. ``narrow`` (a smoke test) must lie inside it."""
    if str(phase1.get("universe")) != UNIVERSE:
        raise ValueError(f"phase1.universe={phase1.get('universe')!r}; only {UNIVERSE!r} is implemented")
    max_hold = max_hold_days(phase1)
    if mode == "screening":
        start, end = (str(x) for x in phase1["screening_window"])
        w = CorpusWindow("screening", warmup_start(start, max_hold), _utc_day(end) + timedelta(days=1), start, end)
    elif mode == "clean":
        _require_p4_authorization(p4_authorization)
        start, end = (str(x) for x in phase1["clean_window"])
        since = _utc_day(str(phase1["p4_clean_window"]["min_filing_accepted"]))
        w = CorpusWindow("clean", since, _utc_day(end) + timedelta(days=1), start, end)
    else:
        raise ValueError(f"mode must be 'screening' or 'clean', got {mode!r}")
    if narrow is not None:
        lo, hi = narrow
        if not (w.since <= lo < hi <= w.until):
            raise FirewallError(f"narrowed window [{lo}, {hi}) is not inside the {mode} window "
                                f"[{w.since}, {w.until})")
        w = dataclasses.replace(w, since=lo, until=hi)
    return w


# --------------------------------------------------------------------------- document text (prereg §2)
@dataclass(frozen=True)
class Filer:
    cik: int
    tickers: tuple[str, ...]             # universe tickers this CIK files for
    names: tuple[str, ...] = ()          # extra exact names (constituents security, SEC titles) to mask
    spells: tuple[tuple[datetime, datetime], ...] = ()   # acceptance intervals kept (acceptance_spells); () = all

    def spells_json(self) -> list[list[str]]:
        return [[a.isoformat(), b.isoformat()] for a, b in self.spells]


@dataclass(frozen=True)
class DocText:
    text: str               # masked, truncated: exactly what Jev reads
    source: str             # documents used, e.g. "EX-99.1", "8-K", "8-K+EX-99"; "" when there were none
    raw_chars: int          # length of the unmasked source documents
    truncated: bool


def masking_inputs(filer: Filer, submissions: Mapping) -> tuple[str, list[str], list[str]]:
    """(legal name, aliases, tickers) for :func:`anonymize`: every exact name and ticker known for the filer."""
    legal = str(submissions.get("name") or "").strip() or (filer.names[0] if filer.names else "")
    aliases = [str(f.get("name", "")).strip() for f in (submissions.get("formerNames") or [])]
    aliases += [n for n in filer.names if n]
    tickers = sorted({*filer.tickers, *(_norm_ticker(t) for t in (submissions.get("tickers") or []) if t)})
    return legal, [a for a in aliases if a], tickers


def press_release(docs: Mapping[str, str]) -> tuple[str, str]:
    """(label, raw text) of the press-release exhibit, or ("", "")."""
    for t in PRESS_RELEASE_TYPES:
        if docs.get(t):
            return t, docs[t]
    return "", ""


def document_text(docs: Mapping[str, str], *, earnings: bool, legal_name: str, tickers: Iterable[str],
                  aliases: Iterable[str], max_chars: int) -> DocText:
    """The text Jev reads for one filing (module docstring, *Text*)."""
    tickers, aliases = list(tickers), list(aliases)

    def mask(s: str, cover: bool) -> str:
        return anonymize(s, legal_name=legal_name, tickers=tickers, aliases=aliases, drop_cover_page=cover)

    label, release = press_release(docs)
    parts, src, raw = [], [], 0
    if earnings and release:
        parts.append(mask(release, False))
        src.append(label)
        raw += len(release)
    else:
        body = docs.get("8-K") or ""
        if body:
            parts.append(mask(body, True))
            src.append("8-K")
            raw += len(body)
        if release:
            parts.append(mask(release, False))
            src.append(label)
            raw += len(release)
    full = "\n\n".join(parts)
    return DocText(full[:max_chars], "+".join(src), raw, len(full) > max_chars)


# --------------------------------------------------------------------------- shards (ADR-6)
def pipeline_sha() -> str:
    """Hash of the code that turns cached documents into Jev's text; a change rebuilds every shard."""
    return source_sha(Path(_anonymize_module.__file__), Path(__file__))


def build_stamp(window: CorpusWindow, max_chars: int) -> dict:
    return {**window.to_json(), "form": FORM, "doc_types": list(DOC_TYPES), "max_chars": int(max_chars),
            "questionnaire_hash": questionnaire_hash(), "pipeline_sha": pipeline_sha()}


def shard_path(out_dir: str | Path, cik: int) -> Path:
    return Path(out_dir) / f"cik={int(cik)}.parquet"


def _sidecar(p: Path) -> Path:
    return p.with_suffix(".json")


def shard_is_current(out_dir: str | Path, cik: int, stamp: Mapping, spells: list[list[str]] | None = None) -> bool:
    """The shard exists, is complete, and was built under this stamp (and these membership spells)."""
    p = shard_path(out_dir, cik)
    if not (p.is_file() and _sidecar(p).is_file()):
        return False
    try:
        rec = json.loads(_sidecar(p).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return rec.get("stamp") == dict(stamp) and (spells is None or rec.get("spells") == spells)


def write_shard(out_dir: str | Path, cik: int, rows: Sequence[Mapping], stamp: Mapping, extra: Mapping) -> Path:
    """Parquet first, then the sidecar, each replaced atomically: a sidecar never vouches for a partial shard."""
    p = shard_path(out_dir, cik)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows), columns=list(SHARD_COLUMNS))
    df["accepted_utc"] = pd.to_datetime(df["accepted_utc"]).astype("datetime64[ns]")
    tmp = p.with_name(p.name + ".tmp")
    df.to_parquet(tmp, compression="zstd", index=False)
    os.replace(tmp, p)
    sc = _sidecar(p)
    tmp_sc = sc.with_name(sc.name + ".tmp")
    tmp_sc.write_text(json.dumps({"cik": int(cik), "n_rows": len(df), "stamp": dict(stamp), **extra}, indent=1),
                      encoding="utf-8")
    os.replace(tmp_sc, sc)
    return p


def read_corpus(out_dir: str | Path, *, expect_stamp: Mapping | None = None) -> pd.DataFrame:
    """Every complete shard, sorted by acceptance time. ``expect_stamp`` refuses a stale or mixed corpus."""
    frames = []
    for p in sorted(Path(out_dir).glob("cik=*.parquet")):
        sc = _sidecar(p)
        if not sc.is_file():
            continue                                    # an interrupted write: not part of the corpus
        if expect_stamp is not None:
            got = json.loads(sc.read_text(encoding="utf-8")).get("stamp")
            if got != dict(expect_stamp):
                raise ValueError(f"{p.name} was built under a different stamp: {got} != {dict(expect_stamp)}")
        frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame(columns=list(SHARD_COLUMNS))
    df = pd.concat(frames, ignore_index=True)
    df["accepted_utc"] = pd.to_datetime(df["accepted_utc"]).astype("datetime64[ns]")
    dup = df.duplicated(["accession", "cik"])
    if dup.any():
        raise ValueError(f"duplicate (accession, cik) rows: {df.loc[dup, ['accession', 'cik']].values[:5].tolist()}")
    return df.sort_values(["accepted_utc", "accession", "cik"]).reset_index(drop=True)


# --------------------------------------------------------------------------- builder
@dataclass
class CorpusStats:
    filers: int = 0
    reused: int = 0
    built: int = 0
    failed: dict[int, str] = field(default_factory=dict)
    listed: int = 0
    outside_spells: int = 0          # listed while the filer was not an index member (never used by a signal)
    in_scope: int = 0
    earnings: int = 0
    events: int = 0
    empty_text: int = 0
    truncated: int = 0
    doc_sources: Counter = field(default_factory=Counter)

    def absorb(self, other: CorpusStats) -> None:
        for k in ("listed", "outside_spells", "in_scope", "earnings", "events", "empty_text", "truncated"):
            setattr(self, k, getattr(self, k) + getattr(other, k))
        self.doc_sources.update(other.doc_sources)

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["doc_sources"] = dict(self.doc_sources.most_common())
        d["failed"] = {str(k): v for k, v in self.failed.items()}
        return d


def _filer_rows(f: Filer, window: CorpusWindow, edgar, max_chars: int, local: CorpusStats,
                workers: int) -> list[dict]:
    sub = edgar.submissions(f.cik)
    legal, aliases, tickers = masking_inputs(f, sub)
    refs = edgar.filings(f.cik, forms=(FORM,), since=window.since, until=window.until - timedelta(seconds=1))
    for r in refs:
        if r.form.upper() != FORM or not (window.since <= r.accepted_utc < window.until):
            raise FirewallError(f"{r.accession} ({r.form}, accepted {r.accepted_utc.isoformat()}) is outside the "
                                f"{window.mode} build: form {FORM}, accepted in [{window.since}, {window.until})")
    local.listed += len(refs)
    if f.spells:
        kept = [r for r in refs if any(lo <= r.accepted_utc < hi for lo, hi in f.spells)]
        local.outside_spells += len(refs) - len(kept)
        refs = kept
    scoped = [(r, qs) for r in refs if (qs := questions_for(r.items))]
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        docs_list = list(pool.map(lambda rq: edgar.documents(rq[0], types=DOC_TYPES), scoped))
    rows = []
    qhash = questionnaire_hash()
    for (r, qs), docs in zip(scoped, docs_list):
        earnings = EARNINGS_ITEM in r.items
        dt = document_text(docs, earnings=earnings, legal_name=legal, tickers=tickers, aliases=aliases,
                           max_chars=max_chars)
        rows.append({"accession": r.accession, "cik": int(r.cik), "tickers": ",".join(f.tickers),
                     "items": ",".join(r.items), "accepted_utc": r.accepted_utc.replace(tzinfo=None),
                     "earnings": earnings, "doc_source": dt.source, "qids": ",".join(q.qid for q in qs),
                     "raw_chars": dt.raw_chars, "text_len": len(dt.text), "truncated": dt.truncated,
                     "text": dt.text, "text_sha256": hashlib.sha256(dt.text.encode("utf-8")).hexdigest(),
                     "questionnaire_hash": qhash})
        local.in_scope += 1
        local.earnings += int(earnings)
        local.events += int(not earnings)
        local.empty_text += int(not dt.text)
        local.truncated += int(dt.truncated)
        local.doc_sources[dt.source or "(none)"] += 1
    return rows


def build_corpus(filers: Sequence[Filer], window: CorpusWindow, edgar, out_dir: str | Path, *, max_chars: int,
                 workers: int = 4, on_progress: Callable[[int, int, Filer, CorpusStats], None] | None = None,
                 ) -> CorpusStats:
    """Build (or resume) one shard per filer. A filer that fails keeps no shard, so a rerun retries it; a
    :class:`FirewallError` aborts the whole build."""
    stamp = build_stamp(window, max_chars)
    stats = CorpusStats(filers=len(filers))
    for i, f in enumerate(filers, 1):
        if shard_is_current(out_dir, f.cik, stamp, f.spells_json()):
            stats.reused += 1
            continue
        local = CorpusStats()
        try:
            rows = _filer_rows(f, window, edgar, max_chars, local, workers)
        except FirewallError:
            raise
        except Exception as exc:                           # noqa: BLE001 - one filer must not stop the build
            stats.failed[f.cik] = repr(exc)[:300]
            log.warning("filer %s %s failed, no shard written: %r", f.cik, ",".join(f.tickers), exc)
            continue
        write_shard(out_dir, f.cik, rows, stamp, {"tickers": list(f.tickers), "spells": f.spells_json()})
        stats.built += 1
        stats.absorb(local)
        if on_progress is not None:
            on_progress(i, len(filers), f, local)
    return stats
