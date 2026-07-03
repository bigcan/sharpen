"""Crucible P1b — free-data acquisition subsystem (spec §4).

The load-bearing test is the AS-OF-JOIN RECONSTRUCTION TRIPWIRE (P1b exit-gate, spec §8 P1b row /
§4.3 item 3): a release-lagged series must join to bars by *publication timestamp*, never by
reference period. The negative half proves the tripwire actually catches the leak Tier-0 is blind
to. The rest drive the FRED and CFTC-COT connectors entirely from fixtures — NO network, NO key.
"""
from __future__ import annotations

import numpy as np
import pytest

from finrl_pro_ds.crucible.catalog import DataCatalog
from finrl_pro_ds.crucible.data import (
    CftcCotConnector,
    DataConnector,
    FredConnector,
    SeriesData,
    SeriesRef,
    asof_join,
    assert_asof_join_causal,
    assert_feature_causal,
    naive_reference_period_join,
    register_series,
    validate_series,
)


def _cot_like_series() -> SeriesData:
    """COT-style: weekly Tuesday report dates, released the following Friday (+3d), strictly
    increasing distinct values so no join can coincidentally match the wrong observation."""
    tuesdays = np.array([f"2024-01-{d:02d}" for d in (2, 9, 16, 23, 30)], dtype="datetime64[ns]")
    values = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
    releases = tuesdays + np.timedelta64(3, "D")          # Friday public release
    return SeriesData(
        ref=SeriesRef("cot", "067651:comm_net", "positioning"),
        reference_period=tuesdays, value=values, release_timestamp=releases)


# ------------------------------------------------------------------ P1b EXIT GATE ----

def test_asof_join_is_causal_and_naive_join_leaks() -> None:
    """THE P1b exit gate. (1) the release-time as-of join passes the causality tripwire; (2) the
    reference-period join produces a DIFFERENT series that FAILS the tripwire — a look-ahead."""
    series = _cot_like_series()
    # Daily calendar spanning the reports and their releases.
    bars = np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-02-05"),
                     np.timedelta64(1, "D")).astype("datetime64[ns]")

    asof = asof_join(series, bars)
    naive = naive_reference_period_join(series, bars)

    # (1) our join is causal — does not raise.
    assert_asof_join_causal(series, bars)

    # (2) the joins genuinely differ (else the test proves nothing): on the Tue..Thu window after a
    # report but before its Friday release, the naive join already shows the new value; as-of not.
    wed = np.datetime64("2024-01-03T00:00:00", "ns")      # day after first Tuesday report
    i = int(np.where(bars == wed)[0][0])
    assert naive[i] == 100.0                              # naive leaks Tuesday's not-yet-public value
    assert np.isnan(asof[i])                              # as-of: nothing released by Wed

    # (3) the tripwire CATCHES the leak: feeding the naive join must raise.
    with pytest.raises(AssertionError):
        assert_feature_causal(naive, series, bars)


def test_asof_join_respects_release_lag_and_revisions() -> None:
    series = _cot_like_series()
    fri1 = np.array(["2024-01-05"], dtype="datetime64[ns]")   # first Friday release
    thu1 = np.array(["2024-01-04"], dtype="datetime64[ns]")
    assert asof_join(series, fri1)[0] == 100.0            # available exactly on release day
    assert np.isnan(asof_join(series, thu1)[0])           # not the day before

    # A revision of the LATEST period (2024-01-30, first released Feb-02 as 140.0), re-released
    # Feb-09 as 145.0, supersedes the current level only from the revision's release date. (A
    # revision of an OLD period would NOT change the current level — that is correct PIT semantics
    # for a "latest available reading" overlay feature.)
    revised = SeriesData(
        ref=series.ref,
        reference_period=np.concatenate([series.reference_period, series.reference_period[-1:]]),
        value=np.concatenate([series.value, [145.0]]),
        release_timestamp=np.concatenate(
            [series.release_timestamp, np.array(["2024-02-09"], dtype="datetime64[ns]")]))
    assert asof_join(revised, np.array(["2024-02-05"], dtype="datetime64[ns]"))[0] == 140.0
    assert asof_join(revised, np.array(["2024-02-12"], dtype="datetime64[ns]"))[0] == 145.0
    assert_asof_join_causal(revised, np.arange(
        np.datetime64("2024-01-01"), np.datetime64("2024-02-20"),
        np.timedelta64(1, "D")).astype("datetime64[ns]"))


# ------------------------------------------------------------------ FRED connector ----

_FRED_FIXTURE = {
    "observations": [
        {"realtime_start": "2020-01-15", "realtime_end": "9999-12-31",
         "date": "2020-01-01", "value": "1.50"},
        {"realtime_start": "2020-02-15", "realtime_end": "9999-12-31",
         "date": "2020-02-01", "value": "1.75"},
        {"realtime_start": "2020-03-15", "realtime_end": "9999-12-31",
         "date": "2020-03-01", "value": "."},          # missing → dropped
    ]
}


def test_fred_fetch_from_fixture_stamps_release_from_reference_plus_lag() -> None:
    captured: dict[str, str] = {}

    def transport(url: str) -> dict:
        captured["url"] = url
        return _FRED_FIXTURE

    conn = FredConnector(transport=transport, release_lag_days=1)   # no key needed with transport
    ref = conn.discover()[0]
    data = conn.fetch(ref, "2020-01-01", "2020-12-31")

    assert data.n_obs == 2                                # the "." row dropped
    assert list(data.value) == [1.5, 1.75]
    assert data.reference_period[0] == np.datetime64("2020-01-01", "ns")
    # release = reference + lag (FRED's realtime_start is 'today' on the plain endpoint, useless as
    # a release time; the all-vintages endpoint is pivoted + capped at 2000 vintages — see fred.py).
    assert data.release_timestamp[0] == np.datetime64("2020-01-02", "ns")
    assert "observation_start=2020-01-01" in captured["url"]
    assert "output_type" not in captured["url"]           # plain endpoint, not the pivoted vintage one


def test_fred_asof_sets_vintage_realtime_params() -> None:
    captured: dict[str, str] = {}

    def transport(url: str) -> dict:
        captured["url"] = url
        return {"observations": []}

    conn = FredConnector(transport=transport)
    conn.fetch(conn.discover()[0], "2020-01-01", "2020-12-31", as_of="2020-06-30")
    assert "realtime_start=2020-06-30" in captured["url"]
    assert "realtime_end=2020-06-30" in captured["url"]


def test_fred_live_fetch_without_key_fails_closed() -> None:
    conn = FredConnector(api_key="")                      # no key, no transport
    with pytest.raises(RuntimeError, match="FRED_API_KEY"):
        conn.fetch(conn.discover()[0], "2020-01-01", "2020-12-31")


# ------------------------------------------------------------------ COT connector ----

def _cot_rows() -> list[dict]:
    return [
        {"report_date_as_yyyy_mm_dd": "2024-01-02T00:00:00.000",
         "cftc_contract_market_code": "067651",
         "comm_positions_long_all": "100", "comm_positions_short_all": "40",
         "noncomm_positions_long_all": "50", "noncomm_positions_short_all": "70",
         "open_interest_all": "300"},
        {"report_date_as_yyyy_mm_dd": "2024-01-09T00:00:00.000",
         "cftc_contract_market_code": "067651",
         "comm_positions_long_all": "120", "comm_positions_short_all": "30",
         "noncomm_positions_long_all": "60", "noncomm_positions_short_all": "55",
         "open_interest_all": "320"},
    ]


def test_cot_fetch_computes_net_and_release_lag() -> None:
    conn = CftcCotConnector(transport=lambda url: _cot_rows(),
                            markets=(("067651", "GOLD - COMMODITY EXCHANGE"),))
    ref = SeriesRef("cot", "067651:comm_net", "positioning")
    data = conn.fetch(ref, "2024-01-01", "2024-01-31")
    assert list(data.value) == [60.0, 90.0]               # long - short
    assert data.reference_period[0] == np.datetime64("2024-01-02", "ns")
    assert data.release_timestamp[0] == np.datetime64("2024-01-05", "ns")   # +3d Friday


def test_cot_asof_excludes_unreleased_rows() -> None:
    conn = CftcCotConnector(transport=lambda url: _cot_rows(),
                            markets=(("067651", "x"),))
    ref = SeriesRef("cot", "067651:comm_net", "positioning")
    # As of Jan-08: only the first report (released Jan-05) is public; the second (released Jan-12)
    # is not — the vintage must exclude it.
    data = conn.fetch(ref, "2024-01-01", "2024-01-31", as_of="2024-01-08")
    assert data.n_obs == 1
    assert data.value[0] == 60.0


def test_cot_pct_oi_field() -> None:
    conn = CftcCotConnector(transport=lambda url: _cot_rows(), markets=(("067651", "x"),))
    data = conn.fetch(SeriesRef("cot", "067651:comm_net_pct_oi", "positioning"),
                      "2024-01-01", "2024-01-31")
    assert data.value[0] == pytest.approx((100 - 40) / 300)


# ------------------------------------------------------------------ quality gate ----

def test_validate_series_flags_gap_outlier_and_pit_violation() -> None:
    clean = _cot_like_series()
    assert validate_series(clean).passed

    # PIT violation: a release stamped BEFORE its reference period.
    bad_pit = SeriesData(clean.ref, clean.reference_period, clean.value,
                         clean.reference_period - np.timedelta64(1, "D"))
    assert not validate_series(bad_pit).passed

    # Staleness: a > max_gap_days hole between reference periods.
    gapped = SeriesData(
        clean.ref,
        np.array(["2020-01-01", "2024-01-01"], dtype="datetime64[ns]"),
        np.array([1.0, 2.0]),
        np.array(["2020-01-02", "2024-01-02"], dtype="datetime64[ns]"))
    assert not validate_series(gapped).passed

    # Outlier / unit shift: one 1000x spike among a majority of normal readings (MAD needs an
    # inlier majority — a robustness property, not a bug).
    weeks = np.arange(np.datetime64("2024-01-02"), np.datetime64("2024-03-26"),
                      np.timedelta64(7, "D")).astype("datetime64[ns]")
    vals = np.array([1.0, 1.1, 1.05, 0.95, 1.2, 1.0, 1000.0, 1.1, 1.05, 0.9, 1.15, 1.0])
    rel = weeks + np.timedelta64(3, "D")
    # A MARKET-class series hard-rejects the spike (a >12σ tick there is usually a bad print).
    market_spike = SeriesData(SeriesRef("stooq", "^spx", "market"), weeks,
                              vals[: weeks.shape[0]], rel)
    report = validate_series(market_spike)
    assert not report.passed
    assert report.n_outliers >= 1
    assert not report.outliers_advisory


def test_validate_series_outliers_advisory_for_fat_tailed_classes() -> None:
    # Fat-tailed alt-data (macro/positioning/fundamental): a crisis-sized move is SIGNAL, not
    # corruption, so the same spike is COUNTED but does NOT reject (advisory). Negative tripwire:
    # dropping these classes from advisory_outlier_classes re-rejects the series.
    weeks = np.arange(np.datetime64("2024-01-02"), np.datetime64("2024-03-26"),
                      np.timedelta64(7, "D")).astype("datetime64[ns]")
    vals = np.array([1.0, 1.1, 1.05, 0.95, 1.2, 1.0, 1000.0, 1.1, 1.05, 0.9, 1.15, 1.0])
    rel = weeks + np.timedelta64(3, "D")
    for cls in ("macro", "positioning", "fundamental"):
        spike = SeriesData(SeriesRef("fred", "VIXCLS", cls), weeks, vals[: weeks.shape[0]], rel)
        report = validate_series(spike)
        assert report.passed, f"{cls} outlier should be advisory, not a reject"
        assert report.n_outliers >= 1
        assert report.outliers_advisory
        # tripwire: with no advisory classes the spike hard-rejects again.
        assert not validate_series(spike, advisory_outlier_classes=frozenset()).passed


# ------------------------------------------------------------------ catalog + protocol ----

def test_register_series_roundtrips_and_snapshot_is_deterministic(tmp_path) -> None:
    series = _cot_like_series()
    db = tmp_path / "catalog.sqlite"
    with DataCatalog(db) as cat:
        entry = register_series(cat, series, freshness="2024-02-01")
        rows = cat.list_series("positioning")
        assert len(rows) == 1
        assert rows[0]["source_id"] == "cot"
        assert rows[0]["snapshot_hash"] == entry.snapshot_hash
        h1 = cat.snapshot_hash()
    # Re-register the identical series → identical catalog snapshot (reproducibility, CR-5).
    with DataCatalog(tmp_path / "catalog2.sqlite") as cat2:
        register_series(cat2, series, freshness="2024-02-01")
        assert cat2.snapshot_hash() == h1


def test_connectors_satisfy_protocol() -> None:
    assert isinstance(FredConnector(transport=lambda u: {}), DataConnector)
    assert isinstance(CftcCotConnector(transport=lambda u: []), DataConnector)
