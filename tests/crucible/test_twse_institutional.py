"""TWSE T86 three-institutional-investors connector — Crucible v2.8.

Fixture field names/shape mirror the LIVE response captured 2026-07-04 (session S553-cont-115):
one ``fields`` array of 19 Chinese column names, one row per security, no per-row date (the whole
response describes one queried calendar day), comma-formatted (possibly negative) number strings.

Every connector construction injects ``store_path=`` (a pytest ``tmp_path`` file) so a test never
writes the repo's real default accumulation store (the same rule the TAIFEX connector's audit note
established for its second injectable seam).
"""
from __future__ import annotations

import numpy as np

from finrl_pro_ds.crucible.data import assert_asof_join_causal, validate_series
from finrl_pro_ds.crucible.data.twse_institutional import TwseInstitutionalConnector

_FIELDS = [
    "證券代號", "證券名稱",
    "外陸資買進股數(不含外資自營商)", "外陸資賣出股數(不含外資自營商)", "外陸資買賣超股數(不含外資自營商)",
    "外資自營商買進股數", "外資自營商賣出股數", "外資自營商買賣超股數",
    "投信買進股數", "投信賣出股數", "投信買賣超股數",
    "自營商買賣超股數",
    "自營商買進股數(自行買賣)", "自營商賣出股數(自行買賣)", "自營商買賣超股數(自行買賣)",
    "自營商買進股數(避險)", "自營商賣出股數(避險)", "自營商買賣超股數(避險)",
    "三大法人買賣超股數",
]

# A daily calendar the connector's series must join to causally.
_BARS = np.arange(np.datetime64("2026-06-01"), np.datetime64("2026-07-15"),
                  np.timedelta64(1, "D")).astype("datetime64[ns]")


def _row(ticker: str, name: str, foreign_net: str, trust_net: str, dealer_net: str,
         total_net: str) -> list[str]:
    """Build a full 19-column T86 row; only the 5 fields this connector reads carry real values,
    the rest are placeholder zeros (never read by the parser, which looks up columns by name)."""
    row = ["0"] * len(_FIELDS)
    row[0], row[1] = ticker, name
    row[4] = foreign_net       # 外陸資買賣超股數(不含外資自營商)
    row[10] = trust_net        # 投信買賣超股數
    row[11] = dealer_net       # 自營商買賣超股數
    row[18] = total_net        # 三大法人買賣超股數
    return row


def _t86_payload(date: str, rows: list[list[str]]) -> dict:
    return {"stat": "OK", "date": date, "title": "115年 test 三大法人買賣超日報",
            "hints": "單位：股", "fields": list(_FIELDS), "data": rows}


# Pre-2018 T86 schema: 16 columns with a SINGLE foreign column `外資買賣超股數` (no mainland/foreign-
# dealer split), live-verified against 2015 payloads (2026-07-06). Only the columns the connector reads.
_OLD_FIELDS = [
    "證券代號", "證券名稱",
    "外資買賣超股數",          # the pre-2018 single foreign column (modern reports split this)
    "投信買賣超股數", "自營商買賣超股數", "三大法人買賣超股數",
]


def _old_row(ticker: str, foreign_net: str, trust_net: str, dealer_net: str, total_net: str) -> list:
    return [ticker, "x", foreign_net, trust_net, dealer_net, total_net]


def _t86_old_payload(date: str, rows: list[list]) -> dict:
    return {"stat": "OK", "date": date, "title": "104年 test", "fields": list(_OLD_FIELDS), "data": rows}


def _transport_for(days: dict[str, dict]):
    calls: list[str] = []

    def transport(url: str) -> dict:
        calls.append(url)
        date = url.split("date=")[1].split("&")[0]
        if date not in days:
            raise RuntimeError(f"no fixture for {date}")
        return days[date]
    transport.calls = calls   # type: ignore[attr-defined]
    return transport


def test_parses_named_columns_and_stamps_release_lag(tmp_path) -> None:
    days = {"20260701": _t86_payload("20260701", [
        _row("0050", "元大台灣50       ", "12,345,678", "-1,000,000", "500,000", "11,845,678"),
    ])}
    conn = TwseInstitutionalConnector(transport=_transport_for(days), tickers=("0050",),
                                      sleep_seconds=0, store_path=tmp_path / "t86.json")
    refs = {r.series_id: r for r in conn.discover()}
    assert set(refs) == {"0050:foreign_net", "0050:trust_net", "0050:dealer_net", "0050:total_net"}

    data = conn.fetch(refs["0050:foreign_net"], "2026-07-01", "2026-07-01")
    assert data.n_obs == 1
    assert data.value[0] == 12_345_678.0
    assert data.reference_period[0] == np.datetime64("2026-07-01", "ns")
    assert data.release_timestamp[0] == np.datetime64("2026-07-02", "ns")   # +1 calendar day
    assert validate_series(data).passed
    assert_asof_join_causal(data, _BARS)


def test_parses_comma_and_negative_numbers(tmp_path) -> None:
    days = {"20260701": _t86_payload("20260701", [
        _row("0050", "x", "1,234,567", "-445,000", "-2,500", "787,067"),
    ])}
    conn = TwseInstitutionalConnector(transport=_transport_for(days), tickers=("0050",),
                                      sleep_seconds=0, store_path=tmp_path / "t86.json")
    refs = {r.series_id: r for r in conn.discover()}
    trust = conn.fetch(refs["0050:trust_net"], "2026-07-01", "2026-07-01")
    dealer = conn.fetch(refs["0050:dealer_net"], "2026-07-01", "2026-07-01")
    assert trust.value[0] == -445_000.0
    assert dealer.value[0] == -2_500.0


def test_bare_numeric_value_is_kept_not_dropped(tmp_path) -> None:
    """Regression (S553-cont-117): T86 occasionally returns a bare JSON number instead of a string
    for a round value (live 2026-07-06: 00635U's trust_net came back as int 0). The old parser called
    int.replace -> AttributeError, which escaped the (TypeError, ValueError) guard and dropped the
    ENTIRE series at the altdata bridge. A bare 0 is a real zero-flow observation — keep it. Tripwire:
    reverting _parse_num's isinstance branch makes this raise instead of returning value 0.0."""
    row = _row("00635U", "gold-etf", "1,000", "0", "-2,000", "-1,000")
    row[10] = 0            # trust_net as a bare int, exactly as the live endpoint sent it
    days = {"20260701": _t86_payload("20260701", [row])}
    conn = TwseInstitutionalConnector(transport=_transport_for(days), tickers=("00635U",),
                                      sleep_seconds=0, store_path=tmp_path / "t86.json")
    refs = {r.series_id: r for r in conn.discover()}
    trust = conn.fetch(refs["00635U:trust_net"], "2026-07-01", "2026-07-01")
    assert trust.n_obs == 1          # not dropped
    assert trust.value[0] == 0.0     # kept as a real zero-flow observation


def test_transport_failure_not_persisted_as_holiday(tmp_path) -> None:
    """C1-06 tripwire: a TRANSPORT failure for a day must NOT be recorded as a non-trading day (it is
    left unpolled so a later run retries it), whereas a genuine no-data reply (valid dict, stat != OK)
    IS persisted as an empty polled day. Reverting _fetch_day_live to collapse both to None (→ stored
    {}) silently deletes up to a full rate-limited backfill window from all future ticks."""
    def transport(url: str) -> dict:
        date = url.split("date=")[1].split("&")[0]
        if date == "20260703":
            raise RuntimeError("rate limited")            # transient transport failure
        if date == "20260704":
            return {"stat": "很抱歉，沒有符合條件的資料!"}   # valid reply: a genuine non-trading day
        return _t86_payload(date, [_row("0050", "x", "1,000", "0", "0", "1,000")])
    conn = TwseInstitutionalConnector(transport=transport, tickers=("0050",), sleep_seconds=0,
                                      max_live_days_per_fetch=100, store_path=tmp_path / "t86.json")
    conn.fetch(conn.discover()[0], "2026-07-03", "2026-07-04")
    persisted = conn._store.load()
    assert "20260703" not in persisted        # transport failure NOT recorded → retried next run
    assert persisted.get("20260704") == {}    # genuine no-data recorded as an empty polled day


def test_null_value_cell_drops_row_not_the_whole_series(tmp_path) -> None:
    """Sibling of the bare-int regression, found during the cont-117 audit: a JSON `null` value cell
    (Python None) hits None.replace -> AttributeError, which — before the guard was widened — escaped
    the (TypeError, ValueError) except and killed the ENTIRE series at the altdata bridge. Unlike a
    bare 0 (a real zero-flow observation, KEPT), a null means genuinely no data for that cell, so the
    correct behavior is to DROP just that field. Tripwire: narrowing _extract_day's except back to
    (TypeError, ValueError) makes this raise instead of returning an empty series."""
    row = _row("00635U", "gold-etf", "1,000", "0", "-2,000", "-1,000")
    row[10] = None         # trust_net as JSON null
    days = {"20260701": _t86_payload("20260701", [row])}
    conn = TwseInstitutionalConnector(transport=_transport_for(days), tickers=("00635U",),
                                      sleep_seconds=0, store_path=tmp_path / "t86.json")
    refs = {r.series_id: r for r in conn.discover()}
    trust = conn.fetch(refs["00635U:trust_net"], "2026-07-01", "2026-07-01")
    assert trust.n_obs == 0          # field dropped, series intact (no crash)
    # a sibling field on the SAME row still parses — proves only the null cell was dropped
    dealer = conn.fetch(refs["00635U:dealer_net"], "2026-07-01", "2026-07-01")
    assert dealer.n_obs == 1
    assert dealer.value[0] == -2_000.0


def test_non_string_ticker_cell_is_skipped_not_a_crash(tmp_path) -> None:
    """Cont-117 audit finding F3, closed by the P0 rewrite: a non-string TICKER cell used to hit
    `.strip()` OUTSIDE any guard and crash the day. `_extract_day` now skips it. The real 0050 row on
    the same day still extracts, proving one malformed row does not poison the day."""
    good = _row("0050", "x", "1,000", "2,000", "3,000", "6,000")
    junk = _row("0050", "x", "9", "9", "9", "27")
    junk[0] = 50           # ticker cell as a bare int (non-string) — must be skipped, not crash
    days = {"20260701": _t86_payload("20260701", [junk, good])}
    conn = TwseInstitutionalConnector(transport=_transport_for(days), tickers=("0050",),
                                      sleep_seconds=0, store_path=tmp_path / "t86.json")
    refs = {r.series_id: r for r in conn.discover()}
    fore = conn.fetch(refs["0050:foreign_net"], "2026-07-01", "2026-07-01")
    assert fore.n_obs == 1
    assert fore.value[0] == 1_000.0     # the valid string row, not the junk-ticker row


def test_store_shares_one_http_call_across_series(tmp_path) -> None:
    days = {"20260701": _t86_payload("20260701", [
        _row("0050", "x", "1", "2", "3", "6"),
    ])}
    transport = _transport_for(days)
    conn = TwseInstitutionalConnector(transport=transport, tickers=("0050",), sleep_seconds=0,
                                      store_path=tmp_path / "t86.json")
    refs = {r.series_id: r for r in conn.discover()}
    conn.fetch(refs["0050:foreign_net"], "2026-07-01", "2026-07-01")
    conn.fetch(refs["0050:trust_net"], "2026-07-01", "2026-07-01")
    conn.fetch(refs["0050:dealer_net"], "2026-07-01", "2026-07-01")
    conn.fetch(refs["0050:total_net"], "2026-07-01", "2026-07-01")
    assert len(transport.calls) == 1        # 4 series, same day -> exactly 1 HTTP call


def test_missing_ticker_returns_empty_series_not_a_crash(tmp_path) -> None:
    days = {"20260701": _t86_payload("20260701", [_row("2330", "x", "1", "1", "1", "3")])}
    conn = TwseInstitutionalConnector(transport=_transport_for(days), tickers=("0050",),
                                      sleep_seconds=0, store_path=tmp_path / "t86.json")
    ref = conn.discover()[0]
    data = conn.fetch(ref, "2026-07-01", "2026-07-01")
    assert data.n_obs == 0


def test_non_ok_response_treated_as_no_data(tmp_path) -> None:
    days = {"20260704": {"stat": "很抱歉，沒有符合條件的資料!"}}     # a non-trading-day-shaped response
    conn = TwseInstitutionalConnector(transport=_transport_for(days), tickers=("0050",),
                                      sleep_seconds=0, store_path=tmp_path / "t86.json")
    ref = conn.discover()[0]
    data = conn.fetch(ref, "2026-07-04", "2026-07-04")
    assert data.n_obs == 0


def test_respects_max_lookback_days_cap(tmp_path) -> None:
    conn = TwseInstitutionalConnector(transport=_transport_for({}), tickers=("0050",),
                                      max_lookback_days=5, sleep_seconds=0,
                                      store_path=tmp_path / "t86.json")
    ref = conn.discover()[0]
    # requested a 30-day window but every day will 404/raise in the fixture (empty `days` dict) —
    # the cap only needs to be verified via `meta`, not via how many days actually had data.
    data = conn.fetch(ref, "2026-06-01", "2026-07-01")
    assert data.meta["n_days_queried"] == 6     # capped to max_lookback_days + 1 (inclusive span)


def test_negative_lag_misconfiguration_is_caught_by_the_shared_pit_gate(tmp_path) -> None:
    """CAUS-05-style negative tripwire: this connector has no revision/holiday-lag model to break the
    way COT's did, so the meaningful mutation is a sign/arithmetic error in `release_lag_days` (an
    author mistake, not a real-world assumption). Proves the shared `validate_series` PIT-sanity check
    (release must not predate its own reference period) actually fires against THIS connector's
    output, not just in the abstract."""
    days = {"20260701": _t86_payload("20260701", [_row("0050", "x", "1", "1", "1", "3")])}
    broken = TwseInstitutionalConnector(transport=_transport_for(days), tickers=("0050",),
                                        release_lag_days=-1, sleep_seconds=0,
                                        store_path=tmp_path / "t86.json")   # sign-flipped bug
    ref = broken.discover()[0]
    data = broken.fetch(ref, "2026-07-01", "2026-07-01")
    report = validate_series(data)
    assert not report.passed
    assert any("PIT violation" in r for r in report.reasons)


# --- P0 accumulation store (S553-cont-117) ---------------------------------------------------------

def test_pre_2018_foreign_net_column_alias(tmp_path) -> None:
    """T86's schema changed ~2018 (live-verified): pre-2018 payloads carry a SINGLE `外資買賣超股數`
    foreign column instead of the modern `外陸資買賣超股數(不含外資自營商)` split. The connector must
    read both via the ordered candidate names in _FIELD_COLUMNS, stitching one continuous foreign_net
    series. Regression: this was found from a real backfill sample (2015 0050 had trust/dealer/total
    but NO foreign_net); without the alias every pre-2018 foreign observation is silently dropped.
    Tripwire: removing the pre-2018 fallback name makes foreign_net.n_obs drop to 0 here."""
    days = {"20150102": _t86_old_payload("20150102", [_old_row("0050", "5,000", "10", "20", "5,030")])}
    conn = TwseInstitutionalConnector(transport=_transport_for(days), tickers=("0050",),
                                      sleep_seconds=0, store_path=tmp_path / "t86.json")
    refs = {r.series_id: r for r in conn.discover()}
    fore = conn.fetch(refs["0050:foreign_net"], "2015-01-02", "2015-01-02")
    assert fore.n_obs == 1
    assert fore.value[0] == 5_000.0      # extracted from the pre-2018 single-foreign column
    # the stable-named fields still resolve on the old 16-column schema too
    for fld, exp in (("trust_net", 10.0), ("dealer_net", 20.0), ("total_net", 5_030.0)):
        d = conn.fetch(refs[f"0050:{fld}"], "2015-01-02", "2015-01-02")
        assert d.n_obs == 1 and d.value[0] == exp



def test_store_persists_across_instances_no_refetch(tmp_path) -> None:
    """The whole point of P0: a day fetched once persists to disk, so a LATER connector instance (a
    subsequent orchestrator tick) reads it WITHOUT any network call. Instance B shares the store path
    but is given a transport with NO fixture for that day — it must still answer from disk, and must
    NOT have called transport at all (a call would raise 'no fixture' and be swallowed to no-data)."""
    store = tmp_path / "t86.json"
    days = {"20260701": _t86_payload("20260701", [_row("0050", "x", "1,000", "2,000", "3,000", "6,000")])}
    a = TwseInstitutionalConnector(transport=_transport_for(days), tickers=("0050",),
                                   sleep_seconds=0, store_path=store)
    a.fetch(a.discover()[0], "2026-07-01", "2026-07-01")          # populates + persists the store
    assert store.exists()

    b_transport = _transport_for({})                             # empty: any live call would 'fail'
    b = TwseInstitutionalConnector(transport=b_transport, tickers=("0050",), sleep_seconds=0,
                                   store_path=store)
    refs = {r.series_id: r for r in b.discover()}
    data = b.fetch(refs["0050:foreign_net"], "2026-07-01", "2026-07-01")
    assert data.n_obs == 1
    assert data.value[0] == 1_000.0
    assert len(b_transport.calls) == 0          # answered entirely from the persisted store


def test_backfill_deep_range_reads_full_history_causally(tmp_path) -> None:
    """A backfill-style multi-day fill: three trading days across a span (with a gap = non-trading
    days), then one fetch reads all three back, each stamped release = ref + 1 calendar day, ascending
    and causal against the bar calendar. Proves the store is a faithful multi-day extension of the
    single-day path, not just a one-day cache."""
    days = {
        "20260701": _t86_payload("20260701", [_row("0050", "x", "100", "1", "1", "102")]),
        "20260702": _t86_payload("20260702", [_row("0050", "x", "200", "1", "1", "202")]),
        "20260706": _t86_payload("20260706", [_row("0050", "x", "300", "1", "1", "302")]),
        # 07-03..07-05 absent from the fixture -> live 'fail' -> recorded as polled empty days
    }
    conn = TwseInstitutionalConnector(transport=_transport_for(days), tickers=("0050",),
                                      sleep_seconds=0, store_path=tmp_path / "t86.json")
    refs = {r.series_id: r for r in conn.discover()}
    data = conn.fetch(refs["0050:foreign_net"], "2026-07-01", "2026-07-06")
    assert data.n_obs == 3
    assert list(data.value) == [100.0, 200.0, 300.0]
    assert list(data.reference_period.astype("datetime64[D]").astype(str)) == \
        ["2026-07-01", "2026-07-02", "2026-07-06"]
    # each release is exactly +1 calendar day on its OWN reference day (never a business-day count)
    assert list(data.release_timestamp.astype("datetime64[D]").astype(str)) == \
        ["2026-07-02", "2026-07-03", "2026-07-07"]
    assert validate_series(data).passed
    assert_asof_join_causal(data, _BARS)


def test_as_of_cutoff_hides_days_not_yet_released(tmp_path) -> None:
    """PIT/LEAK-2 on the store path: with two stored days, an as_of BETWEEN the first day's release
    (D1+1) and the second day's release (D2+1) must expose ONLY the first — the second has not been
    published yet at that as_of point. This is the negative guard that a backfilled history cannot
    leak a value before its own +1-day release."""
    days = {
        "20260701": _t86_payload("20260701", [_row("0050", "x", "111", "1", "1", "113")]),
        "20260702": _t86_payload("20260702", [_row("0050", "x", "222", "1", "1", "224")]),
    }
    conn = TwseInstitutionalConnector(transport=_transport_for(days), tickers=("0050",),
                                      sleep_seconds=0, store_path=tmp_path / "t86.json")
    ref = {r.series_id: r for r in conn.discover()}["0050:foreign_net"]
    # as_of = 2026-07-02: day1 (release 07-02) is <= cutoff and visible; day2 (release 07-03) is NOT.
    data = conn.fetch(ref, "2026-07-01", "2026-07-02", as_of="2026-07-02")
    assert data.n_obs == 1
    assert data.value[0] == 111.0
    # widen the as_of past day2's release -> both appear
    data2 = conn.fetch(ref, "2026-07-01", "2026-07-02", as_of="2026-07-03")
    assert data2.n_obs == 2


def test_live_fetch_budget_bounds_network_and_fills_recent_first(tmp_path) -> None:
    """A cold-store tick must not hang on a multi-thousand-day inline crawl: the per-instance
    live-fetch budget caps network calls, and fills MOST-RECENT-first so a truncated tick still gets
    the useful tail. Budget 2 over a 10-day empty range -> exactly 2 calls, for the 2 latest days."""
    transport = _transport_for({})               # every day 'fails' -> recorded polled-empty
    conn = TwseInstitutionalConnector(transport=transport, tickers=("0050",), sleep_seconds=0,
                                      max_live_days_per_fetch=2, store_path=tmp_path / "t86.json")
    ref = conn.discover()[0]
    conn.fetch(ref, "2026-06-20", "2026-06-29")
    assert len(transport.calls) == 2
    fetched_dates = sorted(u.split("date=")[1].split("&")[0] for u in transport.calls)
    assert fetched_dates == ["20260628", "20260629"]      # the two most-recent days, not the oldest
