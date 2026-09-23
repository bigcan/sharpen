"""SEC EDGAR filing-TEXT connector (ATL x Jev plan item B4).

Complements :mod:`sharpen.crucible.data.edgar` (XBRL facts, filed DATE only). Filing text needs the
exact ``acceptanceDateTime`` — the submissions API carries it per filing, to the second, in UTC — plus
the documents themselves. Two EDGAR endpoints, both keyless:

* ``data.sec.gov/submissions/CIK##########.json`` — filing list with form, 8-K ``items``,
  ``acceptanceDateTime`` and ``primaryDocument`` (``filings.recent``; older pages are listed under
  ``filings.files`` and fetched on demand).
* ``www.sec.gov/Archives/edgar/data/<cik>/<acc>/<acc-with-dashes>.txt`` — the full submission, every
  document wrapped in ``<DOCUMENT><TYPE>…<TEXT>…</TEXT></DOCUMENT>``. Parsing ``<TYPE>`` is the reliable
  way to find the EX-99.1 press release that carries an Item 2.02 earnings release.

SEC fair-access policy: a descriptive ``User-Agent`` is mandatory (``SEC_EDGAR_UA``) and requests are
throttled below the 10 req/s ceiling. Cleaned text is cached gzip-compressed (the raw submission,
which can carry uuencoded graphics, is never stored). Inject ``transport`` for offline tests.

Timestamp semantics (LEAK-2) — WHEN a filing may first move a position — belong to the signal layer
and land with its negative test in Phase 2; this module only reports ``accepted_utc`` faithfully.
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_SUBMISSIONS_PAGE = "https://data.sec.gov/submissions/{name}"
_FULL_TEXT = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{acc}.txt"
_DOC_RE = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.S | re.I)
_TYPE_RE = re.compile(r"<TYPE>([^\s<]+)", re.I)
_TEXT_RE = re.compile(r"<TEXT>(.*?)</TEXT>", re.S | re.I)

Transport = Callable[[str], bytes]


@dataclass(frozen=True)
class FilingRef:
    """One filing from the submissions index. ``items`` is the 8-K item list (e.g. ('2.02','9.01'))."""

    cik: int
    accession: str            # with dashes, e.g. 0000320193-26-000071
    form: str
    accepted_utc: datetime    # tz-aware UTC, to the second
    items: tuple[str, ...]
    primary_document: str


class _Throttle:
    def __init__(self, per_second: float) -> None:
        self._gap = 1.0 / per_second
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self._gap
        if delay > 0:
            time.sleep(delay)


_RETRYABLE_HTTP = frozenset({429, 500, 502, 503, 504})


def _default_transport_factory(user_agent: str, per_second: float, max_retries: int = 3,
                               backoff_s: float = 2.0) -> Transport:
    """Throttled GET with retry on transient failures (429/5xx, timeouts, resets). A transient error
    must not silently DROP a filing: at corpus scale, skips are non-random (they cluster on the large
    filers whose submission indexes span many pages) and would bias the sample."""
    throttle = _Throttle(per_second)

    def _get(url: str) -> bytes:
        for attempt in range(max_retries + 1):
            throttle.wait()
            req = urllib.request.Request(url, headers={"User-Agent": user_agent,
                                                       "Accept-Encoding": "gzip"})
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:   # noqa: S310 - fixed https host
                    data = resp.read()
                    return gzip.decompress(data) if resp.headers.get("Content-Encoding") == "gzip" else data
            except urllib.error.HTTPError as exc:
                if exc.code not in _RETRYABLE_HTTP or attempt == max_retries:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == max_retries:
                    raise
            wait = backoff_s * (2 ** attempt)
            logger.warning("EDGAR transient failure on %s; retry %d/%d in %.0fs", url, attempt + 1,
                           max_retries, wait)
            time.sleep(wait)
        raise RuntimeError("unreachable")                            # loop returns or raises
    return _get


def _parse_accepted(s: str) -> datetime:
    # "2026-07-30T20:30:28.000Z" -> aware UTC
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def parse_filing_block(block: dict, cik: int) -> list[FilingRef]:
    """``filings.recent``-shaped column block -> FilingRefs (row-wise)."""
    out = []
    for i, form in enumerate(block.get("form", [])):
        items = tuple(x.strip() for x in (block["items"][i] or "").split(",") if x.strip())
        out.append(FilingRef(cik, block["accessionNumber"][i], form,
                             _parse_accepted(block["acceptanceDateTime"][i]), items,
                             block["primaryDocument"][i]))
    return out


def split_documents(submission_text: str) -> list[tuple[str, str]]:
    """Full-submission text -> [(TYPE, raw document text)] in file order."""
    docs = []
    for m in _DOC_RE.finditer(submission_text):
        body = m.group(1)
        t = _TYPE_RE.search(body)
        x = _TEXT_RE.search(body)
        if t and x:
            docs.append((t.group(1).upper(), x.group(1)))
    return docs


def html_to_text(raw: str) -> str:
    """Visible text of an EDGAR HTML/text document, whitespace-normalised, tables flattened."""
    if "<" not in raw[:2000] and "<" not in raw[-2000:]:
        text = raw
    else:
        from bs4 import BeautifulSoup                    # local: only the live path needs bs4/lxml
        soup = BeautifulSoup(raw, "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(" ")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\s*\n\s*", "\n", text).strip()


class EdgarFilingsClient:
    """List filings and fetch their cleaned document text, with an on-disk text cache."""

    def __init__(self, *, user_agent: str | None = None, cache_dir: str | Path | None = None,
                 transport: Transport | None = None, per_second: float = 4.0) -> None:
        if transport is None:
            ua = user_agent or os.environ.get("SEC_EDGAR_UA", "")
            if not ua:
                raise RuntimeError("EDGAR requires a descriptive User-Agent: set SEC_EDGAR_UA "
                                   "(SEC fair-access policy; fails closed)")
            transport = _default_transport_factory(ua, per_second)
        self._get = transport
        self._cache = Path(cache_dir) if cache_dir is not None else None
        self._subs: dict[int, dict] = {}

    def submissions(self, cik: int) -> dict:
        """Raw submissions JSON — registrant ``name``, ``tickers``, ``formerNames`` and the filing
        index — memoized per client."""
        cik = int(cik)
        if cik not in self._subs:
            self._subs[cik] = json.loads(self._get(_SUBMISSIONS.format(cik=cik)))
        return self._subs[cik]

    def filings(self, cik: int, *, forms: Iterable[str] = ("8-K",), since: datetime | None = None,
                until: datetime | None = None, include_older_pages: bool = True) -> list[FilingRef]:
        d = self.submissions(cik)
        refs = parse_filing_block(d["filings"]["recent"], int(cik))
        if include_older_pages:
            oldest = min((r.accepted_utc for r in refs), default=None)
            if since is None or oldest is None or oldest > since:
                for page in d["filings"].get("files", []):
                    refs += parse_filing_block(json.loads(self._get(_SUBMISSIONS_PAGE.format(
                        name=page["name"]))), int(cik))
        forms = {f.upper() for f in forms}
        return sorted((r for r in refs if r.form.upper() in forms
                       and (since is None or r.accepted_utc >= since)
                       and (until is None or r.accepted_utc <= until)),
                      key=lambda r: r.accepted_utc)

    def documents(self, ref: FilingRef, *, types: Iterable[str] = ("8-K", "EX-99.1")) -> dict[str, str]:
        """{TYPE: cleaned text} for the requested document types (first occurrence of each)."""
        want = {t.upper() for t in types}
        path = None
        if self._cache is not None:
            path = self._cache / str(ref.cik) / f"{ref.accession}.json.gz"
            if path.exists():
                cached = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
                if want <= set(cached.get("_types_requested", [])):
                    return {k: v for k, v in cached.items() if k in want}
        url = _FULL_TEXT.format(cik=ref.cik, acc_nodash=ref.accession.replace("-", ""), acc=ref.accession)
        sub = self._get(url).decode("utf-8", "replace")
        out: dict[str, str] = {}
        for dtype, raw in split_documents(sub):
            if dtype in want and dtype not in out:
                out[dtype] = html_to_text(raw)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(gzip.compress(json.dumps(
                {**out, "_types_requested": sorted(want)}).encode("utf-8")))
        return out
