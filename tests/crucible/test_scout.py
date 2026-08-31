"""Crucible P5 — the Data Scout (Stage-1 ACQUIRE driver), spec §3 Stage 1 / §7.1 role 1.

The Scout must (a) accept only series that pass BOTH the quality gate and the as-of-join reconstruction
tripwire, (b) REJECT a leaking source rather than register it, (c) keep registration a separate,
post-review step, and (d) flag ids that need a DSL-legal alias (COT/EDGAR-style). CR-1/CR-6: it never
touches a gate value or verdict — it only proves data PIT-clean and widens the frontier.
"""
from __future__ import annotations

import numpy as np

from sharpen.crucible import DataCatalog
from sharpen.crucible.agentic import DataScout
from sharpen.crucible.data import (
    EdgarConnector,
    GdeltConnector,
    Provenance,
    SeriesData,
    SeriesRef,
    StooqConnector,
)

_BARS = np.arange(np.datetime64("2019-06-01"), np.datetime64("2020-06-01"),
                  np.timedelta64(1, "D")).astype("datetime64[ns]")


class _LeakyConnector:
    """A source whose observations are released BEFORE their reference period (a PIT violation) — the
    Scout must reject it, never register it."""

    source_id = "leaky"
    asset_class = "macro"

    def discover(self) -> list[SeriesRef]:
        return [SeriesRef("leaky", "bad", "macro")]

    def fetch(self, ref, start, end, *, as_of=None) -> SeriesData:
        rp = np.array(["2020-02-01"], dtype="datetime64[ns]")
        return SeriesData(ref, rp, np.array([1.0]), rp - np.timedelta64(5, "D"))

    def provenance(self, ref) -> Provenance:
        return Provenance("leaky", "u", "lic", "none-rejected")


def _clean_connectors() -> list:
    csv = "Date,Open,High,Low,Close,Volume\n" + "\n".join(
        f"2020-01-{d:02d},10,11,9,{100 + d},0" for d in range(1, 29))
    gpay = {"timeline": [{"series": "Average Tone", "data": [
        {"date": f"202001{d:02d}T000000Z", "value": -1.0 + 0.1 * d} for d in range(1, 29)]}]}
    eunits = {"units": {"USD": [{"end": "2019-12-31", "val": 200.0, "filed": "2020-02-15",
                                 "form": "10-K"}]}}
    return [
        StooqConnector(transport=lambda u: csv, symbols=(("^spx", "S&P 500"),)),
        GdeltConnector(transport=lambda u: gpay, themes=(("gold", "gold tone"),)),
        EdgarConnector(transport=lambda u: eunits,
                       concepts=(("320193", "RevenueFromContractWithCustomerExcludingAssessedTax",
                                  "AAPL rev"),)),
    ]


def test_scout_accepts_clean_rejects_leaky() -> None:
    scout = DataScout([*_clean_connectors(), _LeakyConnector()])
    report = scout.survey("2019-01-01", "2020-12-31", _BARS)

    assert len(report.findings) == 4
    accepted = {(f.source_id, f.accepted) for f in report.findings}
    assert ("stooq", True) in accepted
    assert ("gdelt", True) in accepted
    assert ("edgar", True) in accepted
    # the leaking source is rejected with a PIT reason, not registered.
    leaky = next(f for f in report.findings if f.source_id == "leaky")
    assert not leaky.accepted
    assert any("PIT" in r or "LEAK" in r for r in leaky.reasons)
    # new uncorrelated domains reach the funnel (spec §8 P5 gate).
    assert set(report.asset_classes()) == {"market", "sentiment", "fundamental"}


def test_scout_flags_non_dsl_legal_terminals() -> None:
    scout = DataScout(_clean_connectors())
    report = scout.survey("2019-01-01", "2020-12-31", _BARS)
    by_source = {f.source_id: f for f in report.findings}
    assert by_source["gdelt"].terminal_dsl_legal is True            # gdelt:gold — legal
    assert by_source["stooq"].terminal_dsl_legal is False           # '^spx' — needs an alias
    assert by_source["edgar"].terminal_dsl_legal is False           # cik:concept — needs an alias


def test_scout_registration_is_separate_and_accepted_only(tmp_path) -> None:
    scout = DataScout([*_clean_connectors(), _LeakyConnector()])
    report = scout.survey("2019-01-01", "2020-12-31", _BARS)

    with DataCatalog(tmp_path / "catalog.db") as cat:
        assert len(cat.list_series()) == 0                          # survey did NOT touch the catalog
        entries = scout.register_accepted(report, cat)
        assert len(entries) == 3                                    # only the accepted three
        classes = {r["asset_class"] for r in cat.list_series()}
        assert "macro" not in classes                              # the leaky one never registered


def test_scout_report_writes_json_and_markdown(tmp_path) -> None:
    scout = DataScout(_clean_connectors())
    report = scout.survey("2019-01-01", "2020-12-31", _BARS)
    report.write(tmp_path)
    assert (tmp_path / "scout_report.json").exists()
    assert (tmp_path / "scout_report.md").exists()
    assert "Data Scout report" in (tmp_path / "scout_report.md").read_text(encoding="utf-8")


def test_scout_handles_connector_fetch_failure() -> None:
    class _Broken:
        source_id = "broken"
        asset_class = "macro"

        def discover(self):
            return [SeriesRef("broken", "x", "macro")]

        def fetch(self, ref, start, end, *, as_of=None):
            raise RuntimeError("endpoint down")

        def provenance(self, ref):
            return Provenance("broken", "u", "lic", "release-lag")

    report = DataScout([_Broken()]).survey("2019-01-01", "2020-12-31", _BARS)
    assert len(report.findings) == 1
    assert not report.findings[0].accepted
    assert any("fetch failed" in r for r in report.findings[0].reasons)
