"""EdgarFilingsClient — offline (injected transport). Parsing of the two EDGAR shapes, form/window
filtering, the gzip text cache, and fail-closed on a missing User-Agent."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from sharpen.crucible.data.edgar_filings import (
    EdgarFilingsClient,
    html_to_text,
    parse_filing_block,
    split_documents,
)

_BLOCK = {
    "accessionNumber": ["0000320193-26-000071", "0000320193-26-000050", "0000320193-25-000010"],
    "form": ["8-K", "10-Q", "8-K"],
    "acceptanceDateTime": ["2026-07-30T20:30:28.000Z", "2026-05-01T20:05:00.000Z", "2025-01-30T21:31:00.000Z"],
    "items": ["2.02,9.01", "", "2.02,9.01"],
    "primaryDocument": ["aapl-20260730.htm", "aapl-10q.htm", "aapl-20250130.htm"],
}
_SUBMISSION = """<SEC-DOCUMENT>
<DOCUMENT><TYPE>8-K<SEQUENCE>1<TEXT><html><body><p>Item 2.02 Results</p><script>x()</script></body></html></TEXT></DOCUMENT>
<DOCUMENT><TYPE>EX-99.1<SEQUENCE>2<TEXT><html><body><p>Revenue&nbsp;rose</p>
<table><tr><td>Q</td><td>$1</td></tr></table></body></html></TEXT></DOCUMENT>
<DOCUMENT><TYPE>GRAPHIC<SEQUENCE>3<TEXT>begin 644 logo.jpg</TEXT></DOCUMENT>
</SEC-DOCUMENT>"""


class _Transport:
    def __init__(self):
        self.urls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        if "submissions" in url:
            return json.dumps({"filings": {"recent": _BLOCK, "files": []}}).encode()
        return _SUBMISSION.encode()


def test_parse_filing_block_is_rowwise_with_aware_utc_and_items():
    refs = parse_filing_block(_BLOCK, 320193)
    assert [r.form for r in refs] == ["8-K", "10-Q", "8-K"]
    assert refs[0].accepted_utc == datetime(2026, 7, 30, 20, 30, 28, tzinfo=timezone.utc)
    assert refs[0].items == ("2.02", "9.01") and refs[1].items == ()


def test_filings_filter_by_form_and_window():
    c = EdgarFilingsClient(transport=_Transport())
    refs = c.filings(320193, forms=("8-K",), since=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert [r.accession for r in refs] == ["0000320193-26-000071"]


def test_split_documents_reads_types_in_order():
    assert [t for t, _ in split_documents(_SUBMISSION)] == ["8-K", "EX-99.1", "GRAPHIC"]


def test_documents_clean_text_and_cache_hit(tmp_path):
    t = _Transport()
    c = EdgarFilingsClient(transport=t, cache_dir=tmp_path)
    ref = parse_filing_block(_BLOCK, 320193)[0]
    docs = c.documents(ref)
    assert set(docs) == {"8-K", "EX-99.1"}
    assert "x()" not in docs["8-K"] and "Item 2.02" in docs["8-K"]
    assert "Revenue rose" in docs["EX-99.1"]
    n = len(t.urls)
    assert c.documents(ref) == docs and len(t.urls) == n, "cached filing was re-downloaded"


def test_missing_user_agent_fails_closed(monkeypatch):
    monkeypatch.delenv("SEC_EDGAR_UA", raising=False)
    with pytest.raises(RuntimeError, match="SEC_EDGAR_UA"):
        EdgarFilingsClient()


class _Resp:
    headers: dict = {}

    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch_urlopen(monkeypatch, outcomes):
    import sharpen.crucible.data.edgar_filings as ef
    calls = []

    def fake(req, timeout=None):
        calls.append(req.full_url)
        o = outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return _Resp(o)
    monkeypatch.setattr(ef.urllib.request, "urlopen", fake)
    monkeypatch.setattr(ef.time, "sleep", lambda s: None)
    return ef, calls


def test_transient_failures_are_retried_not_dropped(monkeypatch):
    import urllib.error
    ef, calls = _patch_urlopen(monkeypatch, [urllib.error.URLError(TimeoutError("t/o")),
                                             urllib.error.HTTPError("u", 503, "busy", {}, None), b"ok"])
    get = ef._default_transport_factory("UA test@example.com", per_second=1000.0)
    assert get("https://www.sec.gov/x") == b"ok" and len(calls) == 3


def test_non_retryable_http_error_propagates_immediately(monkeypatch):
    import urllib.error
    ef, calls = _patch_urlopen(monkeypatch, [urllib.error.HTTPError("u", 404, "missing", {}, None), b"never"])
    get = ef._default_transport_factory("UA test@example.com", per_second=1000.0)
    with pytest.raises(urllib.error.HTTPError):
        get("https://www.sec.gov/x")
    assert len(calls) == 1


def _429():
    import urllib.error
    return urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None)


def test_rate_limit_pauses_every_request_past_the_sec_window_then_retries(monkeypatch):
    ef, calls = _patch_urlopen(monkeypatch, [_429(), b"ok"])
    slept: list[float] = []
    monkeypatch.setattr(ef.time, "sleep", slept.append)
    get = ef._default_transport_factory("UA test@example.com", per_second=1000.0, cooldown_s=660.0)
    assert get("https://www.sec.gov/x") == b"ok" and len(calls) == 2
    # SEC lifts a block only after 10 minutes below its limit; a seconds-scale retry would prolong it
    assert max(slept) > 600


def test_rate_limit_spends_no_transient_retry_and_is_bounded(monkeypatch):
    import urllib.error
    ef, calls = _patch_urlopen(monkeypatch, [_429(), _429(), _429(), _429()])
    get = ef._default_transport_factory("UA test@example.com", per_second=1000.0, max_retries=0,
                                        max_cooldowns=3)
    with pytest.raises(urllib.error.HTTPError, match="Too Many"):
        get("https://www.sec.gov/x")
    assert len(calls) == 4, "three cool-downs, then the fourth 429 propagates"


def test_html_to_text_passes_plain_text_through():
    assert html_to_text("plain   text\n\n  line") == "plain text\nline"
