"""The ATL x Jev screening panel (ADR-3, ADR-8): nothing after the screening end can influence it, membership
decides ``active``, the recorded OHLC bracketing fixes exactly the violating bars, and the metadata never claims
survivorship-free data."""
from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest

import sharpen.jev.panel as jp
from sharpen.jev.panel import bracket_ohlc, screening_panel, sector_ids
from sharpen.signals.features import Panel, ohlc_violations

WINDOW = {"screening_window": ["2012-01-02", "2012-02-29"]}


def _raw(seed=0, tickers=("AAA", "BBB", "ZZZ")):
    dates = np.array([d for d in np.arange("2011-11-01", "2012-04-30", dtype="datetime64[D]")
                      if np.is_busday(d)], dtype="datetime64[ns]")
    rng = np.random.default_rng(seed)
    T, N = dates.size, len(tickers)
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, (T, N)), axis=0))
    open_ = close * (1 + rng.normal(0, 0.003, (T, N)))
    high, low = np.maximum(open_, close) * 1.004, np.minimum(open_, close) * 0.996
    vol = rng.uniform(1e5, 1e6, (T, N))
    return Panel(dates, tuple(tickers), open_, high, low, close, vol, np.ones((T, N), bool), np.full((T, N), np.nan),
                 np.zeros(N, int), {"survivorship_free": True, "source": "test"})


@pytest.fixture
def members(monkeypatch):
    """AAA and BBB are members throughout; ZZZ never is."""
    def fake(dates, tickers):
        return np.array([[t in ("AAA", "BBB") for t in tickers]] * len(dates))
    monkeypatch.setattr(jp, "_membership_matrix", fake)


def _cons(tmp_path):
    p = tmp_path / "cons.csv"
    p.write_text("Symbol,Security,GICS Sector,CIK\nAAA,A Co,Energy,1\nBBB,B Co,Utilities,2\n", encoding="utf-8")
    return p


def test_nothing_after_the_screening_end_can_move_the_panel(tmp_path, members):
    a, b = _raw(), _raw()
    after = b.dates.astype("datetime64[D]") > np.datetime64("2012-02-29")
    for f in ("open", "high", "low", "close", "volume"):
        getattr(b, f)[after] *= 7.0                        # wild post-window prices and volumes
    pa, ca, ra = screening_panel(a, WINDOW, _cons(tmp_path))
    pb, cb, rb = screening_panel(b, WINDOW, _cons(tmp_path))
    for f in ("open", "high", "low", "close", "volume", "adv_usd", "active"):
        np.testing.assert_array_equal(getattr(pa, f), getattr(pb, f), err_msg=f)
    assert np.array_equal(ca, cb) and ra == rb
    assert str(pa.dates[-1])[:10] == "2012-02-29" and ca[-1] == np.datetime64("2012-02-29")


def test_rows_start_at_the_window_but_the_calendar_keeps_the_warmup(tmp_path, members):
    p, cal, rep = screening_panel(_raw(), WINDOW, _cons(tmp_path))
    assert str(p.dates[0])[:10] == "2012-01-02" and cal[0] == np.datetime64("2011-11-01")
    assert rep["first_row"] == "2012-01-02" and rep["calendar_days"] == cal.size


def test_membership_decides_active_and_metadata_is_honest(tmp_path, members):
    p, _, rep = screening_panel(_raw(), WINDOW, _cons(tmp_path))
    assert p.active[:, :2].all() and not p.active[:, 2].any()
    assert p.meta["survivorship_free"] is False and p.meta["verdict_cap"] == "PROMISING"
    assert rep["unknown_sector_names"] == 1 and list(p.sector_id) == [0, 1, 2]
    assert np.isnan(p.adv_usd[0]).all() or np.isfinite(p.adv_usd[-1]).all()


def test_bracketing_fixes_exactly_the_violating_active_bars():
    p = _raw()
    hi, lo = p.high.copy(), p.low.copy()
    hi[5, 0] = min(p.open[5, 0], p.close[5, 0]) - 1.0     # high below the body
    lo[7, 1] = max(p.open[7, 1], p.close[7, 1]) + 1.0     # low above the body
    hi[9, 2] = min(p.open[9, 2], p.close[9, 2]) - 1.0     # violating but inactive: left alone
    act = p.active.copy()
    act[9, 2] = False
    bad = dataclasses.replace(p, high=hi, low=lo, active=act)
    v = ohlc_violations(bad)                               # each planted bar also has high < low
    assert (v["high_lt_max_oc"], v["low_gt_min_oc"], v["high_lt_low"]) == (1, 1, 2)
    fixed, n = bracket_ohlc(bad)
    assert n == 2 and ohlc_violations(fixed)["total"] == 0
    assert fixed.high[5, 0] == max(p.open[5, 0], p.close[5, 0]) and fixed.low[7, 1] == min(p.open[7, 1], p.close[7, 1])
    assert fixed.high[9, 2] == hi[9, 2], "an inactive bar is not the funnel's concern and is not rewritten"
    untouched = np.ones_like(hi, bool)
    untouched[5, 0] = untouched[7, 1] = False
    assert np.array_equal(fixed.high[untouched], bad.high[untouched])


def test_sector_ids_bucket_former_members(tmp_path):
    sid, names = sector_ids(("AAA", "BBB", "OLD", "BRK.B"), _cons(tmp_path))
    assert names == ["Energy", "Utilities", "Unknown"] and list(sid) == [0, 1, 2, 2]


def test_the_raw_panel_is_not_modified(tmp_path, members):
    raw = _raw()
    snap = copy.deepcopy(raw)
    screening_panel(raw, WINDOW, _cons(tmp_path))
    for f in ("open", "high", "low", "close", "volume"):
        np.testing.assert_array_equal(getattr(raw, f), getattr(snap, f), err_msg=f)
