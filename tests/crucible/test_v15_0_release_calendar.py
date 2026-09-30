"""crucible-v15.0 LEAK-2 tripwires: modelled release stamps vs the ACTUAL publication record.

Every as-of join trusts a connector's ``release_timestamp`` as ground truth, so a mis-stamped release is a
leak no downstream causality check can see (the as-of tripwire compares the join against itself). These
pin stamps to dates the agencies actually published. Each FAILS on the pre-v15 connectors.
"""
from __future__ import annotations

import numpy as np
import pytest

from sharpen.crucible.data.cftc_cot import CftcCotConnector
from sharpen.crucible.data.connector import SeriesRef
from sharpen.crucible.data.fred import FredConnector


def _cot(report: str) -> str:
    return str(np.datetime64(CftcCotConnector()._release_for(np.datetime64(report)), "D"))


@pytest.mark.parametrize("report,published", [
    ("2025-09-30", "2025-11-19"),     # 2025 shutdown: first delayed report (CFTC 9138-25 / 9147-25)
    ("2025-12-16", "2026-01-06"),     # later of the two published catch-up schedules
    ("2023-01-31", "2023-02-24"),     # ION cyber incident (Historical Special Announcements)
    ("2018-12-24", "2019-02-01"),     # 2018-19 shutdown (CFTC 7864-19)
    ("2013-10-01", "2013-10-25"),     # 2013 shutdown (CFTC 6745-13)
])
def test_disrupted_windows_use_the_actual_publication_date(report, published) -> None:
    assert _cot(report) == published


@pytest.mark.parametrize("report,published", [
    ("2025-01-07", "2025-01-13"),     # Carter national day of mourning closed Thu Jan 9 → Monday
    ("2014-12-23", "2014-12-30"),     # executive-order closure Dec 26 → CFTC published Tue Dec 30
    ("2020-12-21", "2020-12-28"),     # Monday data, Dec 24 closure + Dec 25 → Mon Dec 28 (CFTC note)
])
def test_ad_hoc_federal_closures_delay_the_release(report, published) -> None:
    assert _cot(report) == published


def test_ordinary_week_is_unchanged() -> None:
    assert _cot("2024-03-05") == "2024-03-08"          # Tuesday → Friday


def test_pinned_table_can_only_delay() -> None:
    from sharpen.crucible.data.cftc_cot import _PINNED_RELEASES
    cal_only = CftcCotConnector()
    for report, pinned in _PINNED_RELEASES.items():
        assert np.datetime64(pinned) >= np.datetime64(report) + np.timedelta64(3, "D")
        assert _cot(report) == pinned or np.datetime64(_cot(report)) > np.datetime64(pinned)
    assert cal_only is not None


def _fred_release(series_id: str, dates: list[str], freq: str = "W") -> list[str]:
    payload = {"observations": [{"date": d, "value": "1.0"} for d in dates]}
    fc = FredConnector(api_key="x", transport=lambda url: payload)
    data = fc.fetch(SeriesRef("fred", series_id, "macro", title="t", frequency=freq),
                    dates[0], dates[-1])
    return [str(np.datetime64(r, "D")) for r in data.release_timestamp]


def test_walcl_thanksgiving_week_is_not_a_session_early() -> None:
    rel = _fred_release("WALCL", ["2025-11-19", "2025-11-26"])
    assert rel == ["2025-11-21", "2025-12-01"]           # normal Friday; Thanksgiving → Monday


def test_fred_period_start_dated_series_fail_closed() -> None:
    with pytest.raises(ValueError, match="period START"):
        _fred_release("PAYEMS", ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"], freq="M")
    with pytest.raises(ValueError, match="period START"):   # detected from spacing, even if undeclared
        _fred_release("PAYEMS", ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"], freq="")
