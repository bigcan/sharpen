"""Offline tripwires for scripts/research/finrlx_strategy/data_commodity.py (commodity long-history proxies).

No network: every source is an inline string or a synthetic frame. Pinned properties:
  - parsers keep the right column / unit / exchange date, and unit guards reject percent, cents and GBP-style
    mistakes;
  - the futures roll never books a calendar spread as a return (a constant-contango market with a common
    price factor must give EXACTLY the factor's return), and a naive C1 splice or an LTD off by one trading
    day fails that check (planted bugs);
  - the CL / heating-oil / natural-gas expiry calendars hit known dates;
  - fees accrue on calendar days, T-bill collateral is added once, composites re-route missing energy legs;
  - the seal: ``window="backward"`` raises while sealed, dev levels start at 1.0 and are blind to any change
    in sealed data (a loader that cumulates before the seal view fails that check: planted bug).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from research.finrlx_strategy import data_commodity as dc  # noqa: E402
from research.finrlx_strategy import seal  # noqa: E402

LBMA_RAW = (b'[{"is_cms_locked":0,"d":"1973-03-01","v":[4.5,1.8,null]},'
            b'{"is_cms_locked":0,"d":"1973-03-02","v":[null,null,null]},'
            b'{"is_cms_locked":0,"d":"1973-03-05","v":[4.6,1.85,null]}]')

HOLIDAYS_2020 = ["2020-01-01", "2020-01-20", "2020-02-17", "2020-04-10", "2020-05-25", "2020-07-03",
                 "2020-09-07", "2020-11-26", "2020-12-25"]


def _yahoo_raw(ts: list[int], close: list[float | None], currency: str = "USD") -> bytes:
    return json.dumps({"chart": {"result": [{
        "meta": {"currency": currency, "symbol": "^TEST", "exchangeTimezoneName": "America/New_York"},
        "timestamp": ts, "indicators": {"quote": [{"close": close}]}}]}}).encode()


# ------------------------------------------------------------------------------------------- parsing

def test_parse_lbma_takes_usd_and_keeps_nulls_as_nan():
    df = dc.parse_lbma(LBMA_RAW)
    assert list(df.columns) == ["usd", "gbp", "eur"]
    assert df.dtypes.eq(float).all()
    assert df.loc["1973-03-01", "usd"] == 4.5 and df.loc["1973-03-01", "gbp"] == 1.8
    assert np.isnan(df.loc["1973-03-02", "usd"])
    assert dc.parse_source("lbma_silver", LBMA_RAW).index.tolist() == [pd.Timestamp("1973-03-01"),
                                                                      pd.Timestamp("1973-03-05")]


def test_parse_lbma_rejects_duplicate_or_unordered_dates():
    dup = b'[{"d":"1990-01-02","v":[400,250,null]},{"d":"1990-01-02","v":[401,251,null]}]'
    back = b'[{"d":"1990-01-03","v":[400,250,null]},{"d":"1990-01-02","v":[401,251,null]}]'
    for raw in (dup, back):
        with pytest.raises(ValueError):
            dc.parse_lbma(raw)


def test_silver_fix_drops_the_saturday_keying_error_and_unit_guard_rejects_cents():
    fx = dc.FIXES["lbma_silver"][0]
    assert pd.Timestamp(fx["date"]).dayofweek == 5 and fx["reason"]          # a Saturday: no London fixing
    raw = (b'[{"d":"1983-02-04","v":[14.152,9.334,null]},{"d":"1983-02-05","v":[7.54,4.097,null]},'
           b'{"d":"1983-02-07","v":[13.69,8.998,null]}]')
    s = dc.parse_source("lbma_silver", raw)
    assert pd.Timestamp("1983-02-05") not in s.index and len(s) == 2
    cents = raw.replace(b"14.152", b"1415.2")
    with pytest.raises(ValueError, match="unit range"):
        dc.parse_source("lbma_silver", cents)


def _eia_frame(key: str, label: str, rows: list[tuple[str, float | None]]) -> pd.DataFrame:
    head = [["Back to Contents", "Data 1: test"], ["Sourcekey", key], ["Date", label]]
    return pd.DataFrame(head + [[pd.Timestamp(d), v] for d, v in rows])


def test_parse_eia_frame_checks_sourcekey_and_unit_label():
    rows = [("1990-01-02", 22.5), ("1990-01-03", None), ("1990-01-04", 23.0)]
    s = dc.parse_eia_frame(_eia_frame("RCLC1", "Crude Oil Future Contract 1 (Dollars per Barrel)", rows), "RCLC1")
    assert s.index.tolist() == [pd.Timestamp("1990-01-02"), pd.Timestamp("1990-01-04")]
    assert s.dtype == float and s.iloc[-1] == 23.0
    with pytest.raises(ValueError, match="sourcekey"):
        dc.parse_eia_frame(_eia_frame("RCLC2", "x (Dollars per Barrel)", rows), "RCLC1")
    with pytest.raises(ValueError, match="unit label"):          # heating oil is per GALLON, not barrel
        dc.parse_eia_frame(_eia_frame("RCLC1", "Heating Oil (Dollars per Gallon)", rows), "RCLC1")


def test_parse_yahoo_uses_exchange_date_and_drops_live_row():
    ts = [int(pd.Timestamp("1984-01-03 14:30", tz="UTC").timestamp()),   # 09:30 New York
          int(pd.Timestamp("2008-06-25 04:00", tz="UTC").timestamp()),   # 00:00 New York (EDT)
          int(pd.Timestamp("2008-06-26 14:30", tz="UTC").timestamp()),
          int(pd.Timestamp("2026-09-25 05:37", tz="UTC").timestamp())]   # live snapshot, fetch day
    s = dc.parse_yahoo_chart(_yahoo_raw(ts, [100.0, 838.5, None, 748.9]),
                             drop_on_or_after=pd.Timestamp("2026-09-25"))
    assert s.index.tolist() == [pd.Timestamp("1984-01-03"), pd.Timestamp("2008-06-25")]
    with pytest.raises(ValueError, match="not USD"):
        dc.parse_yahoo_chart(_yahoo_raw(ts, [1.0, 1.0, 1.0, 1.0], currency="EUR"))


# ------------------------------------------------------------------------------------ units and math

def test_rf_units_percent_is_rejected():
    rf = pd.Series([0.0001, 0.00025, 0.00063], index=pd.bdate_range("1981-01-05", periods=3))
    dc.check_rf_units(rf)
    with pytest.raises(ValueError, match="PERCENT"):          # planted units bug: raw Ken French percent
        dc.check_rf_units(rf * 100)
    with pytest.raises(ValueError, match="PERCENT"):
        dc.rf_on_index(rf * 100, rf.index)


def test_rf_on_index_holidays_zero_tail_carried_head_nan():
    idx = pd.DatetimeIndex(["2001-01-08", "2001-01-09", "2001-01-11", "2001-01-12"])   # Wed 10th missing
    rf = pd.Series([1e-4, 2e-4, 3e-4, 4e-4], index=idx)
    grid = pd.bdate_range("2001-01-05", "2001-01-16")
    out = dc.rf_on_index(rf, grid)
    assert np.isnan(out.loc["2001-01-05"])
    assert out.loc["2001-01-10"] == 0.0
    assert out.loc["2001-01-15"] == 4e-4 and out.loc["2001-01-16"] == 4e-4


def test_price_returns_skip_nonpositive_prints():
    p = pd.Series([18.31, -36.98, 8.91, 13.64], index=pd.bdate_range("2020-04-17", periods=4))
    r = dc.price_returns(p)
    assert r.index.tolist() == [pd.Timestamp("2020-04-21"), pd.Timestamp("2020-04-22")]
    assert r.iloc[0] == pytest.approx(8.91 / 18.31 - 1) and r.iloc[1] == pytest.approx(13.64 / 8.91 - 1)


def test_place_returns_zero_on_gaps_nan_outside_and_rejects_weekends():
    grid = pd.bdate_range("2001-01-08", "2001-01-12")
    r_obs = pd.Series([0.01, 0.02], index=pd.DatetimeIndex(["2001-01-08", "2001-01-11"]))
    out = dc.place_returns(r_obs, grid)
    assert out.loc["2001-01-09"] == 0.0 and out.loc["2001-01-10"] == 0.0
    assert out.loc["2001-01-11"] == 0.02 and np.isnan(out.loc["2001-01-12"])
    with pytest.raises(ValueError, match="weekend"):
        dc.place_returns(pd.Series([0.01], index=pd.DatetimeIndex(["2001-01-13"])), grid)


def test_fee_accrues_on_calendar_days():
    grid = pd.bdate_range("2001-01-01", "2002-01-01")
    r = pd.Series(0.0, index=grid)
    net = dc.apply_fee(r, 40.0)
    lvl = (1.0 + net.iloc[1:]).prod()
    assert lvl == pytest.approx(0.996 ** (365 / 365.25), rel=1e-12)
    monday = net.loc["2001-01-08"]                                   # Fri -> Mon: three days of fee
    assert 1 + monday == pytest.approx(0.996 ** (3 / 365.25), rel=1e-12)


def test_composite_reroutes_missing_energy_to_wti_and_drops_other_legs():
    idx = pd.bdate_range("2000-01-03", periods=4)
    comp = pd.DataFrame({"wti": [0.01, 0.01, 0.01, np.nan], "ho": [0.03, np.nan, 0.03, 0.03],
                         "gold": [-0.02, -0.02, np.nan, -0.02]}, index=idx)
    out = dc.composite_returns(comp, {"wti": 1.0, "ho": 1.0, "gold": 2.0})
    assert out.iloc[0] == pytest.approx((0.01 + 0.03 - 0.04) / 4)
    assert out.iloc[1] == pytest.approx((2 * 0.01 - 0.04) / 4)        # heating oil's weight moved to WTI
    assert out.iloc[2] == pytest.approx((0.01 + 0.03) / 2)             # gold missing: renormalised away
    assert np.isnan(out.iloc[3])                                       # no WTI: no composite
    assert sum(dc.DBC_WEIGHTS[k] for k in dc.ENERGY) == pytest.approx(55.0)
    assert dc.DBC_WEIGHTS["gold"] + dc.DBC_WEIGHTS["silver"] == pytest.approx(10.0)


def test_levels_start_at_one_and_skip_first_return():
    r = pd.DataFrame({"x": [np.nan, 0.5, 0.1, -0.1, np.nan]}, index=pd.bdate_range("2007-01-01", periods=5))
    lv = dc.levels_from_returns(r)["x"]
    assert np.isnan(lv.iloc[0]) and lv.iloc[1] == 1.0
    assert lv.iloc[3] == pytest.approx(1.1 * 0.9) and np.isnan(lv.iloc[4])


# --------------------------------------------------------------------------------- futures calendars

def test_cl_calendar_hits_known_2020_expiries():
    cal = pd.bdate_range("2020-01-01", "2020-12-31").difference(pd.DatetimeIndex(HOLIDAYS_2020))
    got = [str(d.date()) for d in dc.cl_last_trade_dates(cal)]
    assert got == ["2020-01-21", "2020-02-20", "2020-03-20", "2020-04-21", "2020-05-19", "2020-06-22",
                   "2020-07-21", "2020-08-20", "2020-09-22", "2020-10-20", "2020-11-20", "2020-12-21"]


def test_month_end_and_natural_gas_calendars():
    cal = pd.bdate_range("1995-11-01", "1997-03-31").difference(pd.DatetimeIndex(["1995-12-25", "1996-12-25"]))
    ho = dc.futures_ltd("month_last", cal)
    assert pd.Timestamp("1995-12-29") in ho and pd.Timestamp("1996-06-28") in ho
    ng = dc.futures_ltd("ng", cal)
    assert pd.Timestamp("1995-12-21") in ng       # 6 business days before the delivery month (old rule)
    assert pd.Timestamp("1996-06-24") in ng       # 5 business days (1996-01 .. 1997-01)
    assert pd.Timestamp("1997-02-26") in ng       # 3 business days (current rule)


def _contango_market(seed: int = 0, noise: float = 0.02) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Every contract k trades at base_k * g(t): a 5%/contract contango and one common price factor g.
    The rolled return of ANY correct construction is therefore exactly g(t)/g(t-1) - 1."""
    cal = pd.bdate_range("2019-01-01", "2019-12-31")
    ltd = dc.cl_last_trade_dates(cal)
    front = np.searchsorted(ltd.values, cal.values, side="left")
    g = pd.Series(np.exp(np.cumsum(np.random.default_rng(seed).normal(0, noise, len(cal)))), index=cal)
    c1 = pd.Series(50 * 1.05 ** front * g.to_numpy(), index=cal)
    c2 = pd.Series(50 * 1.05 ** (front + 1) * g.to_numpy(), index=cal)
    return c1, c2, g


def _assert_roll_invariant(r: pd.Series, g: pd.Series) -> None:
    want = g.pct_change().loc[r.index]
    err = float(np.nanmax(np.abs(r.to_numpy() - want.to_numpy())))
    assert not r.isna().any(), "rolled returns contain NaN"
    assert err < 1e-12, f"roll booked a calendar spread as a return (max error {err:.4f})"


@pytest.mark.parametrize("k", [0, 3, 10])
def test_rolled_returns_never_book_the_calendar_spread(k):
    c1, c2, g = _contango_market()
    _assert_roll_invariant(dc.rolled_front_returns(c1, c2, roll_bdays_before_ltd=k), g)


def test_planted_bug_naive_front_splice_is_caught():
    c1, c2, g = _contango_market()
    naive = c1.pct_change().iloc[1:]
    with pytest.raises(AssertionError, match="calendar spread"):
        _assert_roll_invariant(naive, g)


@pytest.mark.parametrize("shift", [-1, 1])
def test_planted_bug_ltd_off_by_one_day_is_caught(monkeypatch, shift):
    c1, c2, g = _contango_market()                  # built on the TRUE calendar
    true_ltd = dc.cl_last_trade_dates

    def shifted(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
        cal = pd.DatetimeIndex(sorted(set(days)))
        pos = cal.get_indexer(true_ltd(days)) + shift
        return cal[pos[(pos >= 0) & (pos < len(cal))]]

    monkeypatch.setattr(dc, "cl_last_trade_dates", shifted)
    with pytest.raises(AssertionError, match="calendar spread"):
        _assert_roll_invariant(dc.rolled_front_returns(c1, c2, roll_bdays_before_ltd=10), g)


def test_roll_alignment_counts_prefer_the_true_calendar():
    c1, c2, _ = _contango_market(noise=0.003)
    counts = dc.roll_alignment_counts(c1, c2)
    n = counts["as_computed.dev.n_roll_days_gap_gt_3pct"]
    assert n >= 10 and counts["as_computed.dev.n_roll_days_gap_gt_3pct_consistent"] == n
    for wrong in ("ltd_minus_1", "ltd_plus_1"):
        assert counts[f"{wrong}.dev.n_roll_days_gap_gt_3pct_consistent"] == 0


# --------------------------------------------------------------------------------------------- seal

def _synthetic_returns(seed: int = 1) -> pd.DataFrame:
    idx = pd.bdate_range("2005-10-03", "2006-03-31", name="date")
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"GLD": rng.normal(0, 0.01, len(idx)), "DBC": rng.normal(0, 0.02, len(idx))}, index=idx)


def test_backward_raises_while_sealed(monkeypatch):
    monkeypatch.setattr(dc, "_daily_returns", lambda **_: _synthetic_returns())
    monkeypatch.setattr(seal, "prereg_status", lambda *a, **k: (False, "sealed (test)"))
    with pytest.raises(seal.SealedError):
        dc.load_commodity("backward")


def test_backward_raises_in_current_repo_state(monkeypatch):
    if seal.is_unsealed():
        pytest.skip("pre-registration committed: the clean window is legitimately open")
    monkeypatch.setattr(dc, "_daily_returns", lambda **_: _synthetic_returns())
    with pytest.raises(seal.SealedError):
        dc.load_commodity("backward")


def test_unsealed_backward_returns_only_sealed_rows(monkeypatch):
    monkeypatch.setattr(dc, "_daily_returns", lambda **_: _synthetic_returns())
    monkeypatch.setattr(seal, "prereg_status", lambda *a, **k: (True, "unsealed (test)"))
    lv = dc.load_commodity("backward")
    assert lv.index.max() <= seal.SEAL_END and (lv.iloc[0] == 1.0).all()


def test_invalid_window_rejected(monkeypatch):
    monkeypatch.setattr(dc, "_daily_returns", lambda **_: _synthetic_returns())
    with pytest.raises(ValueError):
        dc.load_commodity("all")


def _dev_levels_correct(rets: pd.DataFrame) -> pd.DataFrame:
    return dc.levels_from_returns(seal.dev_view(rets))


def _dev_levels_leaky(rets: pd.DataFrame) -> pd.DataFrame:     # planted bug: cumulate, THEN take the view
    return seal.dev_view(dc.levels_from_returns(rets))


def _perturb_sealed(rets: pd.DataFrame) -> pd.DataFrame:
    out = rets.copy()
    sealed = out.index <= seal.SEAL_END
    out.loc[sealed] = out.loc[sealed] * -3.0 + 0.01
    return out


def test_dev_levels_are_rebased_and_blind_to_sealed_data(monkeypatch):
    base = _synthetic_returns()
    monkeypatch.setattr(dc, "_daily_returns", lambda **_: base)
    lv = dc.load_commodity("dev")
    seal.assert_no_sealed_rows(lv)
    assert (lv.iloc[0] == 1.0).all()
    monkeypatch.setattr(dc, "_daily_returns", lambda **_: _perturb_sealed(base))
    pd.testing.assert_frame_equal(dc.load_commodity("dev"), lv)


def test_planted_bug_cumulating_before_the_seal_view_is_caught():
    base = _synthetic_returns()
    pd.testing.assert_frame_equal(_dev_levels_correct(base), _dev_levels_correct(_perturb_sealed(base)))
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(_dev_levels_leaky(base), _dev_levels_leaky(_perturb_sealed(base)))


# ----------------------------------------------------------------------------------------- fidelity

def _etf_and_returns(seed: int = 2) -> tuple[pd.Series, pd.Series]:
    idx = pd.bdate_range("2007-01-01", "2008-12-31")
    r = pd.Series(np.random.default_rng(seed).normal(0.0003, 0.015, len(idx)), index=idx)
    return (1 + r).cumprod(), r


def test_fidelity_metrics_math():
    etf, r = _etf_and_returns()
    same = dc.fidelity_metrics(3.0 * etf, etf)
    assert same["corr_daily"] == pytest.approx(1.0) and same["corr_weekly"] == pytest.approx(1.0)
    assert same["vol_ratio"] == pytest.approx(1.0) and same["te_ann"] == pytest.approx(0.0, abs=1e-12)
    assert same["growth_ratio"] == pytest.approx(1.0)
    double = dc.fidelity_metrics((1 + 2 * r).cumprod(), etf)
    assert double["vol_ratio"] == pytest.approx(2.0, rel=1e-9)


def test_planted_bug_sign_flip_and_one_day_misalignment_are_caught():
    etf, r = _etf_and_returns()
    assert dc.fidelity_metrics(etf, etf)["corr_daily"] > 0.9
    lagged = dc.fidelity_metrics((1 + r.shift(1)).cumprod(), etf)
    flipped = dc.fidelity_metrics((1 - r).cumprod(), etf)
    assert lagged["corr_daily"] < 0.2
    assert flipped["corr_daily"] < -0.9


def test_fidelity_refuses_sealed_rows():
    idx = pd.bdate_range("2005-06-01", "2006-06-30")
    s = pd.Series(np.linspace(1, 2, len(idx)), index=idx)
    with pytest.raises(seal.SealedError):
        dc.fidelity_metrics(s, s)


# ------------------------------------------------------------------------------------ hygiene, fetch

def test_hygiene_reports_counts_only(monkeypatch):
    idx = pd.bdate_range("2004-01-01", "2007-12-31")
    s = pd.Series(np.linspace(10, 20, len(idx)), index=idx)
    s.iloc[100] = -5.0
    monkeypatch.setattr(dc, "read_source", lambda key: s.rename(key))
    h = dc.hygiene_commodity()
    allowed = {"first_valid", "last_valid", "n_obs", "n_nan_inside", "n_sealed_obs", "n_nonpositive",
               "n_big_moves", "n_repeats", "max_gap_days", "n_nonpositive_raw", "n_fixes"}
    assert set(h.columns) == allowed
    assert (h["n_nonpositive_raw"] == 1).all() and len(h) == len(dc.SOURCES)


class _Resp:
    def __init__(self, content: bytes):
        self.content, self.status_code = content, 200

    def raise_for_status(self) -> None:
        return None


def test_fetch_is_cache_aware_and_writes_manifest(monkeypatch, tmp_path):
    raw = b'[{"d":"1968-04-01","v":[37.7,15.68,null]},{"d":"1968-04-02","v":[37.3,15.5,null]}]'
    calls: list[str] = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        return _Resp(raw)

    monkeypatch.setattr(dc, "RAW_DIR", tmp_path)
    monkeypatch.setattr(dc.requests, "get", fake_get)
    man = dc.fetch_source("lbma_gold_pm")
    assert len(calls) == 1 and calls[0] == dc.SOURCES["lbma_gold_pm"].url
    on_disk = json.loads((tmp_path / "lbma_gold_pm.json.manifest.json").read_text(encoding="utf-8"))
    for key in ("url", "fetched_at", "sha256", "parsed_rows", "first_date", "last_date", "notes", "fixes"):
        assert key in on_disk
    assert on_disk["sha256"] == dc._sha256(raw) and man["parsed_rows"] == 2
    assert on_disk["first_date"] == "1968-04-01" and on_disk["fetched_at"].endswith("Z")
    dc.fetch_source("lbma_gold_pm")
    assert len(calls) == 1                                   # cached: no second download
    dc.fetch_source("lbma_gold_pm", force=True)
    assert len(calls) == 2
    (tmp_path / "lbma_gold_pm.json").write_bytes(raw + b" ")
    with pytest.raises(RuntimeError, match="sha256"):
        dc.fetch_source("lbma_gold_pm")
