"""Offline tripwires for scripts/research/finrlx_strategy/data_equity.py (equity long-history proxies).

No network: every source is a tiny inline fixture. Each construction property is pinned to a hand value, and
the planted-bug tests show the checks have power: a percent-for-decimal slip, a same-month (look-ahead)
dividend yield, a UTC-instead-of-exchange-date bar stamp and a one-day proxy misalignment each make a check
fail. The seal tests assert that the sealed window cannot be read and that dev fidelity ignores it.
"""

from __future__ import annotations

import hashlib
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

from research.finrlx_strategy import data_equity as de  # noqa: E402
from research.finrlx_strategy import seal  # noqa: E402

KF_INDUSTRY_CSV = """This file was created using the 202607 CRSP database.
It contains value- and equal-weighted returns for  5 industry portfolios.

Missing data are indicated by -99.99 or -999.


  Average Value Weighted Returns -- Daily
,Cnsmr,HiTec
20000103   ,   1.00 ,  -2.50
20000104   , -99.99 ,   0.10
20000105   ,   0.00 ,   -999


  Average Equal Weighted Returns -- Daily
,Cnsmr,HiTec
20000103   ,   5.00 ,   5.00

  Annual Value Weighted Returns
,Cnsmr,HiTec
2000   ,  10.00 ,  20.00

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""

KF_FACTOR_CSV = """This file was created by using the 202607 CRSP database.
The Tbill return is the simple daily rate that, over the number of trading days
compounds to 1-month TBill rate.

,Mkt-RF,SMB,HML,RF
19260701,    0.09,   -0.25,   -0.27,    0.01
19260702,    0.45,   -0.33,   -0.06,    0.01
19260706,   -1.50,    0.10,    0.20,    0.01

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


# --------------------------------------------------------------------------------------------------------
# Ken French parsing and units
# --------------------------------------------------------------------------------------------------------
def test_kf_sections_units_and_missing_codes() -> None:
    out = de.parse_kf_csv(KF_INDUSTRY_CSV)
    assert list(out) == ["Average Value Weighted Returns -- Daily", "Average Equal Weighted Returns -- Daily"]
    vw = out[de.VW_SECTION]
    assert vw.loc["2000-01-03", "HiTec"] == pytest.approx(-0.025)       # -2.50 percent -> -0.025
    assert vw.loc["2000-01-03", "Cnsmr"] == pytest.approx(0.01)
    assert np.isnan(vw.loc["2000-01-04", "Cnsmr"])                      # -99.99 missing code
    assert np.isnan(vw.loc["2000-01-05", "HiTec"])                      # -999 missing code
    assert vw.loc["2000-01-05", "Cnsmr"] == 0.0
    assert out["Average Equal Weighted Returns -- Daily"].loc["2000-01-03", "HiTec"] == pytest.approx(0.05)


def test_kf_single_table_factor_file_and_total_return() -> None:
    out = de.parse_kf_csv(KF_FACTOR_CSV)
    assert list(out) == [""]                                          # title line is blank-separated
    tr = de.kf_total_return(out[""])
    assert tr.tolist() == pytest.approx([0.0010, 0.0046, -0.0149])    # (Mkt-RF) + RF, decimals
    lv = de.levels_from_returns(tr)
    assert lv.tolist() == pytest.approx([1.0, 1.0046, 1.0046 * (1 - 0.0149)])


def test_kf_without_daily_rows_raises() -> None:
    with pytest.raises(ValueError, match="no daily table"):
        de.parse_kf_csv("x\n\n,A\n202001,1.0\n202002,2.0\n")


def test_planted_units_bug_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """PLANTED BUG: forget the percent-to-decimal division. check_units must refuse the series."""
    r = de.kf_total_return(de.parse_kf_csv(KF_FACTOR_CSV)[""])
    de.levels_from_returns(r)                                          # correct units pass
    monkeypatch.setattr(de, "KF_PCT", 1.0)
    bad = de.kf_total_return(de.parse_kf_csv(KF_FACTOR_CSV)[""])
    with pytest.raises(ValueError, match="units are not decimal"):
        de.levels_from_returns(bad)


# --------------------------------------------------------------------------------------------------------
# return construction math
# --------------------------------------------------------------------------------------------------------
def test_levels_from_returns_base_and_compounding() -> None:
    r = pd.Series([0.10, 0.10, -0.50], index=pd.bdate_range("2020-01-01", periods=3), name="x")
    assert de.levels_from_returns(r).tolist() == pytest.approx([1.0, 1.1, 0.55])


def test_levels_from_prices() -> None:
    p = pd.Series([50.0, 55.0, 44.0], index=pd.bdate_range("2020-01-01", periods=3), name="x")
    assert de.levels_from_prices(p).tolist() == pytest.approx([1.0, 1.1, 0.88])


def test_price_plus_cash_dividends() -> None:
    df = pd.DataFrame({"close": [10.0, 9.9, 9.9], "dividend": [0.0, 0.2, 0.0]},
                      index=pd.bdate_range("2020-01-01", periods=3))
    r = de.price_plus_cash_dividends(df)
    assert r.tolist() == pytest.approx([(9.9 + 0.2) / 10.0 - 1.0, 0.0])


def test_shiller_date_october_is_not_january() -> None:
    assert de._shiller_date(1871.01) == pd.Timestamp("1871-01-01")
    assert de._shiller_date(1871.1) == pd.Timestamp("1871-10-01")     # the 1871.10 float trap
    assert de._shiller_date(2023.12) == pd.Timestamp("2023-12-01")
    assert de._shiller_date(1999.09) == pd.Timestamp("1999-09-01")


def _shiller_step() -> pd.DataFrame:
    """Yield 2.4% in Jan-2000, 4.8% from Feb-2000: a step that exposes a same-month (look-ahead) lookup."""
    months = pd.date_range("1999-12-01", "2000-04-01", freq="MS")
    return pd.DataFrame({"P": 1000.0, "D": [24.0, 24.0, 48.0, 48.0, 48.0]}, index=months)


def _check_dividend_is_lagged() -> None:
    days = pd.bdate_range("2000-01-03", "2000-03-31")
    close = pd.Series(100.0, index=days, name="flat")               # flat price: TR is pure dividend
    r = de.price_plus_dividend_yield(close, _shiller_step())
    feb = r.loc["2000-02"]
    mar = r.loc["2000-03"]
    assert feb.to_numpy() == pytest.approx(0.024 / 252), "February must use January's yield (known)"
    assert mar.to_numpy() == pytest.approx(0.048 / 252), "March uses February's yield"


def test_dividend_yield_is_prior_month() -> None:
    _check_dividend_is_lagged()


def test_planted_lookahead_dividend_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """PLANTED BUG: use the same month's yield (not yet published). The lag check must fail."""
    monkeypatch.setattr(de, "DIV_LAG_MONTHS", 0)
    with pytest.raises(AssertionError):
        _check_dividend_is_lagged()


def test_yield_forward_fill_only_after_last_publication() -> None:
    days = pd.bdate_range("2000-05-01", "2000-06-30")
    y = de.lagged_monthly_yield(_shiller_step(), days)
    assert y.to_numpy() == pytest.approx(0.048)                       # last known yield carried forward


# --------------------------------------------------------------------------------------------------------
# Yahoo parsing (exchange-local dates, dividends, in-progress bar)
# --------------------------------------------------------------------------------------------------------
def _ts(s: str) -> int:
    return int(pd.Timestamp(s, tz="UTC").timestamp())


def _yahoo_json(ts: list[int], close: list[float], divs: dict[int, float] | None = None,
                rmt: int | None = None, reg: tuple[int, int] | None = None) -> bytes:
    meta: dict = {"exchangeTimezoneName": "America/New_York"}
    if rmt is not None and reg is not None:
        meta["regularMarketTime"] = rmt
        meta["currentTradingPeriod"] = {"regular": {"start": reg[0], "end": reg[1]}}
    res = {"meta": meta, "timestamp": ts,
           "indicators": {"quote": [{"close": close}], "adjclose": [{"adjclose": close}]}}
    if divs:
        res["events"] = {"dividends": {str(k): {"amount": v, "date": k} for k, v in divs.items()}}
    return json.dumps({"chart": {"result": [res], "error": None}}).encode()


def _check_yahoo_dates() -> None:
    # 2024-01-03 01:00 UTC is 2024-01-02 20:00 in New York: the bar belongs to the 2 January session.
    raw = _yahoo_json([_ts("2024-01-02 14:30"), _ts("2024-01-04 01:00")], [100.0, 101.0])
    df = de.parse_yahoo_chart(raw)
    assert list(df.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")], list(df.index)


def test_yahoo_bars_use_exchange_local_date() -> None:
    _check_yahoo_dates()


def test_planted_utc_date_misalignment_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """PLANTED BUG: stamp bars by their UTC date (shifts evening stamps one day). The date check must fail."""
    orig = de.parse_yahoo_chart

    def utc_dated(raw: bytes) -> pd.DataFrame:
        df = orig(raw)
        j = json.loads(raw)["chart"]["result"][0]
        df.index = pd.DatetimeIndex(pd.to_datetime(j["timestamp"], unit="s").normalize(), name="date")
        return df

    monkeypatch.setattr(de, "parse_yahoo_chart", utc_dated)
    with pytest.raises(AssertionError):
        _check_yahoo_dates()


def test_yahoo_dividends_and_nan_rows() -> None:
    t = [_ts("2024-01-02 14:30"), _ts("2024-01-03 14:30"), _ts("2024-01-04 14:30")]
    df = de.parse_yahoo_chart(_yahoo_json(t, [10.0, None, 10.1], divs={t[2]: 0.25}))
    assert list(df.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04")]
    assert df.loc["2024-01-04", "dividend"] == pytest.approx(0.25)


def test_yahoo_in_progress_bar_is_dropped() -> None:
    t = [_ts("2024-01-02 14:30"), _ts("2024-01-03 14:30")]
    reg = (_ts("2024-01-03 14:30"), _ts("2024-01-03 21:00"))
    df = de.parse_yahoo_chart(_yahoo_json(t, [10.0, 10.5], rmt=_ts("2024-01-03 16:00"), reg=reg))
    assert list(df.index) == [pd.Timestamp("2024-01-02")]
    closed = de.parse_yahoo_chart(_yahoo_json(t, [10.0, 10.5], rmt=_ts("2024-01-03 21:00"), reg=reg))
    assert len(closed) == 2


# --------------------------------------------------------------------------------------------------------
# full construction on synthetic sources
# --------------------------------------------------------------------------------------------------------
DAYS = pd.DatetimeIndex(["2005-12-28", "2005-12-29", "2005-12-30", "2006-01-03", "2006-01-04"])
HOLIDAY = pd.Timestamp("2006-01-02")        # US market closed; ex-US data has a row


def _yahoo_frame(close: list[float], div: list[float] | None = None) -> pd.DataFrame:
    return pd.DataFrame({"close": close, "adjclose": close, "dividend": div or [0.0] * len(close)},
                        index=DAYS)


def _synthetic_sources() -> dict:
    ret = pd.DataFrame({"Mkt-RF": [0.01, -0.02, 0.0, 0.03, 0.01], "RF": 0.0001}, index=DAYS)
    ind = pd.DataFrame({"HiTec": [0.02, 0.0, 0.01, -0.01, 0.0]}, index=DAYS)
    me = pd.DataFrame({lbl: [0.001 * k, 0.0, 0.0, 0.0, 0.0] for k, lbl in
                       enumerate(de.ME_CANDIDATES.values(), 1)}, index=DAYS)
    dx_idx = DAYS.insert(3, HOLIDAY)
    dx = pd.DataFrame({"Mkt-RF": [0.0, 0.0, 0.0, 0.05, 0.10, 0.0], "RF": 0.0}, index=dx_idx)
    shiller = pd.DataFrame({"P": 1000.0, "D": 25.2}, index=pd.date_range("2005-10-01", "2006-01-01",
                                                                           freq="MS"))
    flat = [100.0, 101.0, 102.0, 103.0, 104.0]
    return {
        "kf_ff3_daily": {"": ret}, "kf_ind5_daily": {de.VW_SECTION: ind},
        "kf_me_daily": {de.VW_SECTION: me}, "kf_devxus3_daily": {"": dx},
        "shiller_ie_data": shiller, "yh_gspc": _yahoo_frame(flat), "yh_sp500tr": _yahoo_frame(flat),
        "yh_ndx": _yahoo_frame(flat), "yh_ixic": _yahoo_frame(flat), "yh_rut": _yahoo_frame(flat),
        "yh_veiex": _yahoo_frame([10.0, 10.0, 10.0, 10.0, 10.0], [0.0, 0.1, 0.0, 0.0, 0.0]),
    }


def test_construct_levels_columns_calendar_and_holiday_fold() -> None:
    lv = de.construct_levels(_synthetic_sources())
    for tkr, cand in de.RECOMMENDED.items():
        assert tkr in lv.columns and cand in lv.columns
        pd.testing.assert_series_equal(lv[tkr], lv[cand], check_names=False)
    assert (lv.index.dayofweek < 5).all()
    assert HOLIDAY not in lv.index                                     # US-holiday row dropped...
    efa = lv["EFA__kf_devxus"]
    assert efa.loc["2006-01-03"] == pytest.approx(1.05 * 1.10)          # ...its move folded into next session
    # gspc + dividend accrual: yield 25.2/1000 = 2.52% -> 0.0001 per day on top of the price return
    spx = lv["SPX_TR__gspc_shiller"]
    assert spx.loc["2005-12-30"] / spx.loc["2005-12-29"] == pytest.approx(102 / 101 + 0.0001)
    assert lv["EEM__veiex"].loc["2005-12-30"] == pytest.approx(1.0)     # dividend return then flat NAV
    assert lv["SPX_TR__kf_mkt"].iloc[0] == 1.0


# --------------------------------------------------------------------------------------------------------
# seal
# --------------------------------------------------------------------------------------------------------
def _span_levels() -> pd.DataFrame:
    idx = pd.bdate_range("2004-01-01", "2008-12-31")
    rng = np.random.default_rng(0)
    cols = {c: np.cumprod(1 + rng.normal(0, 0.01, len(idx)))
            for c in ("SPX_TR__gspc_shiller", "SPX_TR__kf_mkt", "SPX_TR__sp500tr", "QQQ__ndx")}
    lv = pd.DataFrame(cols, index=idx)
    lv["SPY"] = lv["SPX_TR__gspc_shiller"]
    lv["QQQ"] = lv["QQQ__ndx"]
    return lv


def test_backward_window_raises_while_sealed(monkeypatch: pytest.MonkeyPatch) -> None:
    if seal.is_unsealed():
        pytest.skip("pre-registration committed: the clean window is legitimately unsealed")
    monkeypatch.setattr(de, "_levels_full", lambda raw_dir=None: _span_levels())
    with pytest.raises(seal.SealedError):
        de.load_equity("backward")
    dev = de.load_equity("dev")
    seal.assert_no_sealed_rows(dev)
    assert dev.index.min() > seal.SEAL_END
    with pytest.raises(ValueError):
        de.load_equity("all")


def test_pair_metrics_refuses_sealed_rows() -> None:
    lv = _span_levels()
    with pytest.raises(seal.SealedError):
        de.pair_metrics(lv["SPY"], lv["QQQ"])


def test_fidelity_ignores_the_sealed_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scrambling every sealed value must leave the dev fidelity numbers bit-identical."""
    lv = _span_levels()
    etf = seal.dev_view(pd.DataFrame({"SPY": lv["SPX_TR__sp500tr"], "QQQ": lv["QQQ__ndx"] * 1.0}))
    monkeypatch.setattr(de, "_etf_levels", lambda: etf)
    monkeypatch.setattr(de, "_levels_full", lambda raw_dir=None: lv)
    a = de.fidelity_equity()
    scrambled = lv.copy()
    sealed = scrambled.index <= seal.SEAL_END
    scrambled.loc[sealed] = scrambled.loc[sealed].to_numpy()[::-1] * 7.0
    monkeypatch.setattr(de, "_levels_full", lambda raw_dir=None: scrambled)
    b = de.fidelity_equity()
    assert json.dumps(a["vs_etf"], sort_keys=True) == json.dumps(b["vs_etf"], sort_keys=True)
    assert a["vs_etf"]["QQQ__ndx"]["corr_daily"] == pytest.approx(1.0)
    assert a["vs_etf"]["QQQ__ndx"]["te_ann"] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------------------------------------
# fidelity metrics
# --------------------------------------------------------------------------------------------------------
def _dev_pair() -> tuple[pd.Series, pd.Series]:
    idx = pd.bdate_range("2010-01-01", periods=600)
    r = np.random.default_rng(1).normal(0.0003, 0.01, len(idx))
    e = pd.Series(np.cumprod(1 + r), index=idx)
    return e.copy(), e


def _check_alignment(p: pd.Series, e: pd.Series) -> None:
    m = de.pair_metrics(p, e)
    assert m["corr_daily"] > 0.9, f"daily corr {m['corr_daily']:.3f}: proxy misaligned by a day?"


def test_pair_metrics_identity_and_leverage() -> None:
    p, e = _dev_pair()
    m = de.pair_metrics(p, e)
    assert m["corr_daily"] == pytest.approx(1.0) and m["corr_weekly"] == pytest.approx(1.0)
    assert m["vol_ratio"] == pytest.approx(1.0) and m["te_ann"] == pytest.approx(0.0, abs=1e-12)
    assert m["growth_ratio"] == pytest.approx(1.0)
    lev = de.levels_from_returns((2 * e.pct_change()).fillna(0.0))
    assert de.pair_metrics(lev, e)["vol_ratio"] == pytest.approx(2.0, rel=1e-6)
    _check_alignment(p, e)


def test_planted_one_day_misalignment_is_caught() -> None:
    """PLANTED BUG: proxy levels shifted by one session. The alignment check must fail."""
    p, e = _dev_pair()
    with pytest.raises(AssertionError, match="misaligned"):
        _check_alignment(p.shift(1).dropna(), e)


def test_recommend_by_rule_prefers_full_coverage() -> None:
    fid = {"QQQ__ndx": {"te_ann": 0.02}, "QQQ__ixic": {"te_ann": 0.04}, "QQQ__kf_hitec": {"te_ann": 0.05},
           "EFA__kf_devxus": {"te_ann": 0.14}}
    fv = pd.Series({"QQQ__ndx": pd.Timestamp("1985-10-01"), "QQQ__ixic": pd.Timestamp("1971-02-05"),
                    "QQQ__kf_hitec": pd.Timestamp("1926-07-01"), "EFA__kf_devxus": pd.Timestamp("1990-07-02")})
    rec = de.recommend_by_rule(fid, fv)
    assert rec == {"QQQ": "QQQ__ixic", "EFA": "EFA__kf_devxus"}      # NDX better TE but starts 1985


# --------------------------------------------------------------------------------------------------------
# cache + manifest
# --------------------------------------------------------------------------------------------------------
def test_cache_hit_skips_network_and_sha_mismatch_raises(tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("F-F_Research_Data_Factors_daily.csv", KF_FACTOR_CSV)
    content = buf.getvalue()
    calls: list[str] = []
    monkeypatch.setattr(de, "_http_get", lambda url, timeout=90: calls.append(url) or content)
    de.fetch_kf("kf_ff3_daily", raw_dir=tmp_path)
    man = json.loads((tmp_path / "kf_ff3_daily.manifest.json").read_text())
    assert man["sha256"] == hashlib.sha256(content).hexdigest()
    assert man["parsed_rows"] == 3 and man["first_date"] == "1926-07-01"
    assert {"url", "fetched_at", "notes"} <= set(man)
    de.fetch_kf("kf_ff3_daily", raw_dir=tmp_path)                      # cached: no second download
    assert len(calls) == 1
    (tmp_path / "kf_ff3_daily.zip").write_bytes(content + b"tamper")
    with pytest.raises(RuntimeError, match="sha256"):
        de.fetch_kf("kf_ff3_daily", raw_dir=tmp_path)
