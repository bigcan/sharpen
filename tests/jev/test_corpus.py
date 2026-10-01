"""The ATL x Jev filing corpus (architecture §2; prereg §2 and §9). These pin what decides the text Jev reads and
which filings exist at all: the exact-key CIK map (ADR-4), the point-in-time universe, the press-release exhibit
under its observed labels, the document construction (parity with the instrument check where labels agree), the
clean-window firewall (ADR-3), and a resumable build that never drops a filing silently. All offline."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from sharpen.crucible.data.edgar_filings import EdgarFilingsClient, FilingRef
from sharpen.jev.corpus import (
    DOC_TYPES,
    CikMap,
    Filer,
    FirewallError,
    acceptance_spells,
    build_cik_map,
    build_corpus,
    build_stamp,
    corpus_window,
    coverage_by_year,
    document_text,
    masking_inputs,
    membership_spells,
    press_release,
    read_corpus,
    shard_path,
    universe_tickers,
    warmup_start,
)
from sharpen.jev.questionnaire import questionnaire_hash

ROOT = Path(__file__).resolve().parents[2]
PHASE1 = yaml.safe_load((ROOT / "configs" / "atl_jev.gates.yaml").read_text(encoding="utf-8"))["phase1"]
UTC = timezone.utc


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


# --------------------------------------------------------------------------- CIK map and universe
def test_cik_map_is_exact_key_normalized_and_reports_gaps_and_conflicts():
    cons = {"BRK-B": 1067983, "XOM": 34088}
    sec = {"XOM": 2115436, "AAPL": 320193, "GOOG": 1652044, "GOOGL": 1652044}
    m = build_cik_map(["BRK.B", "xom", "AAPL", "GOOG", "GOOGL", "FB", "  "], constituents=cons, sec_tickers=sec)
    assert m.mapped["BRK-B"] == (1067983,)
    assert m.mapped["XOM"] == (34088, 2115436) and m.conflicts == {"XOM": (34088, 2115436)}
    assert m.unmapped == ("FB",), "a renamed ticker must stay unmapped, never be guessed"
    assert m.filers()[1652044] == ("GOOG", "GOOGL"), "dual share classes file under one CIK"
    assert m.filers()[34088] == ("XOM",) and m.filers()[2115436] == ("XOM",)


def _pit():
    dates = np.array(["2011-06-01", "2011-12-15", "2012-06-01", "2025-03-01"], dtype="datetime64[D]")
    members = [frozenset({"A", "B"}), frozenset({"A", "C"}), frozenset({"C", "D"}), frozenset({"E"})]
    return dates, members


def test_universe_is_every_member_in_force_during_the_window():
    dates, members = _pit()
    # the row in force ON the start date counts (A, C); a row that ended before it does not (B); nor one after it (E)
    assert universe_tickers(dates, members, "2012-01-01", "2024-12-31") == {"A", "C", "D"}


def test_universe_fails_closed_outside_the_membership_file():
    dates, members = _pit()
    with pytest.raises(ValueError, match="unknown before"):
        universe_tickers(dates, members, "2011-01-01", "2012-12-31")
    with pytest.raises(ValueError, match="Refresh"):
        universe_tickers(dates, members, "2024-01-01", "2026-01-01")


def test_membership_spells_split_on_a_gap_and_merge_contiguous_rows():
    dates = np.array(["2011-06-01", "2011-12-15", "2012-06-01", "2013-01-01", "2025-03-01"], dtype="datetime64[D]")
    members = [frozenset({"A", "B"}), frozenset({"A", "C"}), frozenset({"C", "D"}), frozenset({"A", "D"}),
               frozenset({"E"})]
    d = lambda s: np.datetime64(s, "D")  # noqa: E731
    assert membership_spells(dates, members, ["A"], "2012-01-01", "2024-12-31") == [
        (d("2012-01-01"), d("2012-06-01")), (d("2013-01-01"), d("2025-01-01"))]
    # dual share classes: any of the filer's tickers makes it a member
    assert membership_spells(dates, members, ["A", "C"], "2012-01-01", "2024-12-31") == [
        (d("2012-01-01"), d("2025-01-01"))]


def test_acceptance_spells_open_each_spell_by_the_warmup_and_clip_to_the_window():
    w = corpus_window(PHASE1, mode="screening")
    d = lambda s: np.datetime64(s, "D")  # noqa: E731
    got = acceptance_spells([(d("2012-01-01"), d("2012-06-01")), (d("2016-03-01"), d("2018-01-01"))], w, 63)
    assert got == ((w.since, _utc("2012-06-01")), (warmup_start("2016-03-01", 63), _utc("2018-01-01")))


def test_coverage_by_year_weights_member_days():
    dates, members = _pit()
    m = CikMap({"A": (1,), "D": (4,)}, ("C",), {})
    cov = coverage_by_year(m, dates, members, "2012-01-01", "2012-12-31")
    # {A, C} for 152 days (Jan 1 - May 31, leap year), then {C, D} for 214 days: one mapped name of two throughout
    assert cov["2012"]["names"] == 3
    assert cov["2012"]["names_mapped_share"] == round(2 / 3, 4)
    assert cov["2012"]["member_days_mapped_share"] == 0.5


# --------------------------------------------------------------------------- document text
_COVER = ("UNITED STATES SECURITIES AND EXCHANGE COMMISSION FORM 8-K ACME CORP (Exact name of registrant) "
          "Commission File Number 1-1234 ")


def test_press_release_prefers_99_1_then_99_01_then_unnumbered_99():
    assert press_release({"EX-99": "c", "EX-99.01": "b", "EX-99.1": "a"}) == ("EX-99.1", "a")
    assert press_release({"EX-99": "c", "EX-99.01": "b"}) == ("EX-99.01", "b")
    assert press_release({"EX-99": "c", "8-K": "body"}) == ("EX-99", "c")
    assert press_release({"8-K": "body", "EX-99.1": ""}) == ("", "")


def _dt(docs, earnings, max_chars=24_000):
    return document_text(docs, earnings=earnings, legal_name="ACME CORP", tickers=["ACME"], aliases=[],
                         max_chars=max_chars)


def test_earnings_release_filed_as_ex_99_01_is_read_from_the_exhibit_not_the_8k_stub():
    docs = {"8-K": _COVER + "Item 2.02 Results. See Exhibit 99.01.",
            "EX-99.01": "ACME CORP raised its full-year outlook; revenue rose 12%."}
    d = _dt(docs, earnings=True)
    assert d.source == "EX-99.01"
    assert "raised its full-year outlook" in d.text and "See Exhibit" not in d.text
    assert "ACME" not in d.text and "[COMPANY]" in d.text


def test_earnings_without_any_release_exhibit_falls_back_to_the_8k_body_without_cover():
    d = _dt({"8-K": _COVER + "Item 2.02 Results of Operations. Revenue rose."}, earnings=True)
    assert d.source == "8-K" and d.text.startswith("Item 2.02") and "Commission File" not in d.text


def test_event_filing_is_body_then_release_and_is_truncated_to_max_chars():
    docs = {"8-K": _COVER + "Item 1.02 Termination of a Material Definitive Agreement.",
            "EX-99": "Press release: the agreement ended."}
    d = _dt(docs, earnings=False)
    assert d.source == "8-K+EX-99"
    assert d.text.index("Item 1.02") < d.text.index("Press release") and not d.truncated
    short = _dt(docs, earnings=False, max_chars=20)
    assert short.truncated and len(short.text) == 20 and short.text == d.text[:20]


def test_no_documents_gives_empty_text_not_a_placeholder():
    d = _dt({}, earnings=True)
    assert d.text == "" and d.source == "" and d.raw_chars == 0


def test_parity_with_the_instrument_check_when_the_release_is_labelled_ex_99_1():
    spec = importlib.util.spec_from_file_location("jev_instrument_check",
                                                  ROOT / "scripts" / "research" / "jev_instrument_check.py")
    ic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ic)
    kw = dict(name="ACME CORP", tickers=["ACME"], aliases=["ACME HOLDINGS"], max_chars=24_000)
    for docs, earnings in (({"8-K": _COVER + "Item 2.02 x", "EX-99.1": "ACME CORP (NYSE: ACME) grew."}, True),
                           ({"8-K": _COVER + "Item 2.05 Exit costs at ACME."}, False),
                           ({"8-K": _COVER + "Item 4.02 Non-reliance", "EX-99.1": "ACME restates."}, False)):
        old = ic.document_text(docs, earnings, kw["name"], kw["tickers"], kw["aliases"], kw["max_chars"])
        new = document_text(docs, earnings=earnings, legal_name=kw["name"], tickers=kw["tickers"],
                            aliases=kw["aliases"], max_chars=kw["max_chars"]).text
        assert new == old


def test_masking_inputs_use_every_exact_name_and_ticker():
    sub = {"name": "META PLATFORMS, INC.", "tickers": ["META"], "formerNames": [{"name": "FACEBOOK INC"}]}
    legal, aliases, tickers = masking_inputs(Filer(1326801, ("META",), ("Meta Platforms",)), sub)
    assert legal == "META PLATFORMS, INC."
    assert aliases == ["FACEBOOK INC", "Meta Platforms"] and tickers == ["META"]


# --------------------------------------------------------------------------- window and firewall (ADR-3)
def test_screening_window_reads_phase1_and_includes_the_warmup():
    w = corpus_window(PHASE1, mode="screening")
    assert w.until == _utc(PHASE1["screening_window"][1]) + timedelta(days=1)
    assert w.since == warmup_start(PHASE1["screening_window"][0], 63) and w.since < _utc("2012-01-01")
    moved = copy.deepcopy(PHASE1)
    moved["screening_window"] = ["2013-01-01", "2020-12-31"]
    w2 = corpus_window(moved, mode="screening")
    assert w2.until == _utc("2021-01-01") and w2.rows_start == "2013-01-01", "the window must be READ from phase1"


def test_warmup_reaches_63_trading_days_before_the_first_row():
    holidays = ["2011-11-24", "2011-12-26", "2012-01-02"]
    first_row = np.busday_offset("2012-01-01", 0, roll="forward", holidays=holidays)
    earliest_release = np.busday_offset(first_row, -62, holidays=holidays)   # 63 rows including the first
    # an after-close acceptance is released a trading day later, so acceptance may precede that row again
    earliest_acceptance = np.busday_offset(earliest_release, -1, holidays=holidays)
    assert warmup_start("2012-01-01", 63) < _utc(str(earliest_acceptance))


def test_clean_mode_refuses_without_the_p4_sentinel(tmp_path):
    with pytest.raises(PermissionError, match="P4-only"):
        corpus_window(PHASE1, mode="clean")
    with pytest.raises(PermissionError, match="unopened"):
        corpus_window(PHASE1, mode="clean", p4_authorization=tmp_path / "OPENED.json")
    fake = tmp_path / "fake.json"
    fake.write_text("{}", encoding="utf-8")
    with pytest.raises(PermissionError, match="not a P4 sentinel"):
        corpus_window(PHASE1, mode="clean", p4_authorization=fake)
    ok = tmp_path / "OPENED.json"
    ok.write_text(json.dumps({"opened_utc": "2027-01-01T00:00:00+00:00"}), encoding="utf-8")
    w = corpus_window(PHASE1, mode="clean", p4_authorization=ok)
    assert w.since == _utc(PHASE1["p4_clean_window"]["min_filing_accepted"])


def test_other_universe_or_mode_is_refused():
    other = {**PHASE1, "universe": "djia30"}
    with pytest.raises(ValueError, match="only 'sp500_pit'"):
        corpus_window(other, mode="screening")
    with pytest.raises(ValueError, match="mode"):
        corpus_window(PHASE1, mode="both")


def test_a_narrowed_window_may_not_leave_the_mode_window():
    with pytest.raises(FirewallError):
        corpus_window(PHASE1, mode="screening", narrow=(_utc("2024-06-01"), _utc("2025-02-01")))
    w = corpus_window(PHASE1, mode="screening", narrow=(_utc("2019-01-01"), _utc("2020-01-01")))
    assert (w.since, w.until) == (_utc("2019-01-01"), _utc("2020-01-01"))


# --------------------------------------------------------------------------- builder (offline EDGAR)
def _full_text(docs: dict[str, str]) -> str:
    return "<SEC-DOCUMENT>" + "".join(f"<DOCUMENT><TYPE>{t}<SEQUENCE>{i}<TEXT>{b}</TEXT></DOCUMENT>"
                                      for i, (t, b) in enumerate(docs.items(), 1)) + "</SEC-DOCUMENT>"


class _Edgar:
    """Transport serving two filers; CIK 333's submissions always fail."""

    FILINGS = {111: [("0000000111-19-000001", "8-K", "2019-02-01T21:05:00.000Z", "2.02,9.01"),
                     ("0000000111-19-000002", "8-K", "2019-03-01T14:00:00.000Z", "5.02"),
                     ("0000000111-19-000003", "8-K", "2019-04-01T14:00:00.000Z", "1.02,9.01"),
                     ("0000000111-19-000004", "8-K/A", "2019-05-01T14:00:00.000Z", "2.02"),
                     ("0000000111-19-000005", "8-K", "2019-06-03T20:00:00.000Z", "2.02"),
                     ("0000000111-18-000009", "8-K", "2018-11-01T21:00:00.000Z", "2.02")],
               222: [("0000000222-19-000001", "8-K", "2019-07-01T12:00:00.000Z", "2.02,9.01")]}
    DOCS = {"0000000111-19-000001": {"8-K": _COVER + "Item 2.02 See Exhibit 99.01",
                                     "EX-99.01": "ACME CORP (NYSE: ACME) raised its outlook."},
            "0000000111-19-000003": {"8-K": _COVER + "Item 1.02 Termination of agreement",
                                     "EX-99.1": "ACME ended the deal."},
            "0000000111-19-000005": {"8-K": _COVER + "Item 2.02 Results of Operations for ACME"},
            "0000000222-19-000001": {"8-K": "Item 2.02", "EX-99": "Beta Inc. results.", "GRAPHIC": "begin 644"}}

    def __init__(self):
        self.urls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        if "submissions/CIK" in url:
            cik = int(url.rsplit("CIK", 1)[1].split(".")[0])
            if cik == 333:
                raise OSError("EDGAR is down for this filer")
            rows = self.FILINGS[cik]
            name = {111: "ACME CORP", 222: "BETA INC"}[cik]
            return json.dumps({"name": name, "tickers": [], "formerNames": [], "filings": {"recent": {
                "accessionNumber": [r[0] for r in rows], "form": [r[1] for r in rows],
                "acceptanceDateTime": [r[2] for r in rows], "items": [r[3] for r in rows],
                "primaryDocument": ["x.htm"] * len(rows)}, "files": []}}).encode()
        return _full_text(self.DOCS[url.rsplit("/", 1)[1][:-4]]).encode()


FILERS = [Filer(111, ("ACME",)), Filer(222, ("BETA", "BETB")), Filer(333, ("GAMA",))]


def _window():
    return corpus_window(PHASE1, mode="screening", narrow=(_utc("2019-01-01"), _utc("2020-01-01")))


def test_build_scopes_masks_stamps_and_records_failures(tmp_path):
    t = _Edgar()
    stats = build_corpus(FILERS, _window(), EdgarFilingsClient(transport=t, cache_dir=tmp_path / "cache"),
                         tmp_path / "corpus", max_chars=24_000, workers=2)
    assert (stats.built, stats.reused, list(stats.failed)) == (2, 0, [333])
    assert not shard_path(tmp_path / "corpus", 333).exists(), "a failed filer must keep no shard"
    df = read_corpus(tmp_path / "corpus", expect_stamp=build_stamp(_window(), 24_000))
    acme = df[df.cik == 111].set_index("accession")
    # 5.02-only is out of scope, the 8-K/A amendment and the 2018 filing are outside the build
    assert list(acme.index) == ["0000000111-19-000001", "0000000111-19-000003", "0000000111-19-000005"]
    assert list(acme.doc_source) == ["EX-99.01", "8-K+EX-99.1", "8-K"]
    assert list(acme.qids) == ["E1,E2,E3,E4,E5,M1", "M1", "E1,E2,E3,E4,E5,M1"]
    assert all("ACME" not in x for x in acme.text), "filer name or ticker leaked into Jev's text"
    for text, sha in zip(df.text, df.text_sha256):
        assert sha == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert set(df.questionnaire_hash) == {questionnaire_hash()}
    beta = df[df.cik == 222].iloc[0]
    assert beta.tickers == "BETA,BETB" and beta.doc_source == "EX-99" and beta.text == "[COMPANY]. results."
    assert str(df.accepted_utc.dtype) == "datetime64[ns]"
    assert df.accepted_utc.iloc[0] == np.datetime64("2019-02-01T21:05:00", "ns")


def test_build_resumes_without_requests_and_rebuilds_on_a_new_stamp(tmp_path):
    out, cache = tmp_path / "corpus", tmp_path / "cache"
    build_corpus(FILERS[:2], _window(), EdgarFilingsClient(transport=_Edgar(), cache_dir=cache), out,
                 max_chars=24_000)
    t = _Edgar()
    again = build_corpus(FILERS[:2], _window(), EdgarFilingsClient(transport=t, cache_dir=cache), out,
                         max_chars=24_000)
    assert (again.reused, again.built, t.urls) == (2, 0, []), "a current shard was rebuilt"
    t2 = _Edgar()
    rebuilt = build_corpus(FILERS[:2], _window(), EdgarFilingsClient(transport=t2, cache_dir=cache), out,
                           max_chars=5_000)
    assert rebuilt.built == 2 and all("submissions" in u for u in t2.urls), "documents must come from the cache"
    with pytest.raises(ValueError, match="different stamp"):
        read_corpus(out, expect_stamp=build_stamp(_window(), 24_000))


def test_only_filings_accepted_in_a_membership_spell_are_fetched(tmp_path):
    # ACME is a member from 2019-06-01; the warm-up reaches back to the 2019-04-01 event but not to February
    spell = acceptance_spells([(np.datetime64("2019-06-01", "D"), np.datetime64("2020-01-01", "D"))], _window(), 63)
    t = _Edgar()
    stats = build_corpus([Filer(111, ("ACME",), spells=spell)], _window(),
                         EdgarFilingsClient(transport=t, cache_dir=tmp_path / "c"), tmp_path / "corpus",
                         max_chars=24_000)
    df = read_corpus(tmp_path / "corpus")
    assert list(df.accession) == ["0000000111-19-000003", "0000000111-19-000005"]
    assert stats.outside_spells == 1 and stats.listed == 4      # only the February release precedes the warm-up
    assert not any("0000000111-19-000001" in u for u in t.urls), "a non-member filing was downloaded"
    moved = acceptance_spells([(np.datetime64("2019-01-15", "D"), np.datetime64("2020-01-01", "D"))], _window(), 63)
    again = build_corpus([Filer(111, ("ACME",), spells=moved)], _window(),
                         EdgarFilingsClient(transport=_Edgar(), cache_dir=tmp_path / "c"), tmp_path / "corpus",
                         max_chars=24_000)
    assert again.built == 1, "a changed membership (e.g. a refreshed PIT file) must rebuild the shard"
    assert "0000000111-19-000001" in set(read_corpus(tmp_path / "corpus").accession)


class _LeakyEdgar(EdgarFilingsClient):
    """A broken listing filter that hands the builder one filing it must never accept."""

    def __init__(self, leak: FilingRef, **kw):
        super().__init__(**kw)
        self._leak = leak

    def filings(self, cik, **kw):
        return [self._leak]


@pytest.mark.parametrize("leak", [
    FilingRef(111, "0000000111-25-000001", "8-K", _utc("2025-03-03T21:00:00"), ("2.02",), "x.htm"),   # clean window
    FilingRef(111, "0000000111-19-000009", "8-K/A", _utc("2019-05-01T14:00:00"), ("2.02",), "x.htm"),  # amendment
], ids=["clean-window-filing", "amendment"])
def test_a_filing_the_listing_should_have_excluded_aborts_the_build(tmp_path, leak):
    with pytest.raises(FirewallError, match="outside the screening build"):
        build_corpus(FILERS[:1], _window(), _LeakyEdgar(leak, transport=_Edgar(), cache_dir=tmp_path / "c"),
                     tmp_path / "corpus", max_chars=24_000)
    assert not shard_path(tmp_path / "corpus", 111).exists()


def test_doc_types_requested_cover_every_press_release_label():
    assert set(DOC_TYPES) == {"8-K", "EX-99.1", "EX-99.01", "EX-99"}
