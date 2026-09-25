"""Offline tripwires for scripts/research/finrlx_strategy/data_fx.py (FX long-history proxies).

No network: every source is a tiny inline fixture or a synthetic frame. Pinned properties:
  - parsers read the real file shapes (H.10 SDMX XML incl. 'ND', H.10 legacy HTML, OECD csv, BIS csv);
  - quoting: USD-per-unit series are inverted to units per USD, legacy pages are checked to say "X/US$";
  - units: percent -> decimal exactly once (a decimal input or an unconverted percent chain fails);
  - the return math (spot x deposit carry - fee; DXY geometric formula; UUP funded vs excess);
  - causality (LEAK-2): a monthly average is used only from the next month;
  - the euro splice (irrevocable 1.95583 DEM/EUR) and the legacy/euro DXY identity after 1999;
  - the seal: window='backward' raises SealedError while sealed; the dev view is re-based;
  - PLANTED BUGS (inverted quote, one-day misalignment, missing euro conversion, leaky monthly rate,
    units error) are each shown to be caught by the check that guards them.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from research.finrlx_strategy import data_fx as fx  # noqa: E402
from research.finrlx_strategy import seal  # noqa: E402

ALL_DXY = sorted(set(fx.DXY_WEIGHTS_EURO) | set(fx.DXY_WEIGHTS_LEGACY))

# ---------------------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------------------
H10_XML = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<message:MessageGroup xmlns:message="http://www.SDMX.org/resources/SDMXML/schemas/v1_0/message"
 xmlns:common="http://www.SDMX.org/resources/SDMXML/schemas/v1_0/common"
 xmlns:frb="http://www.federalreserve.gov/structure/compact/common">
 <frb:DataSet id="H10" xmlns:kf="http://www.federalreserve.gov/structure/compact/H10_H10">
  <kf:Series CURRENCY="EUR" FREQ="9" FX="EUR" SERIES_NAME="RXI$US_N.B.EU" UNIT="Currency" UNIT_MULT="1">
   <frb:Annotations><common:Annotation>
    <common:AnnotationType>Short Description</common:AnnotationType>
    <common:AnnotationText>EMU Members Euro (USD per EUR)</common:AnnotationText>
   </common:Annotation></frb:Annotations>
   <frb:Obs OBS_STATUS="A" OBS_VALUE="1.2500" TIME_PERIOD="2007-01-02" />
   <frb:Obs OBS_STATUS="ND" OBS_VALUE="-9999" TIME_PERIOD="2007-01-03" />
   <frb:Obs OBS_STATUS="A" OBS_VALUE="1.2000" TIME_PERIOD="2007-01-04" />
  </kf:Series>
  <kf:Series CURRENCY="JPY" FREQ="9" FX="JPY" SERIES_NAME="RXI_N.B.JA" UNIT="Currency" UNIT_MULT="1">
   <frb:Obs OBS_STATUS="A" OBS_VALUE="118.50" TIME_PERIOD="2007-01-02" />
   <frb:Obs OBS_STATUS="A" OBS_VALUE="119.00" TIME_PERIOD="2007-01-03" />
   <frb:Obs OBS_STATUS="A" OBS_VALUE="120.00" TIME_PERIOD="2007-01-04" />
  </kf:Series>
  <kf:Series CURRENCY="USD" FREQ="129" FX="EUR" SERIES_NAME="RXI$US_N.M.EU" UNIT="Currency" UNIT_MULT="1">
   <frb:Obs OBS_STATUS="A" OBS_VALUE="1.3000" TIME_PERIOD="2007-01-01" />
  </kf:Series>
 </frb:DataSet>
</message:MessageGroup>
"""

LEGACY_HTML = """<html><body><pre>GERMANY -- SPOT EXCHANGE RATE, DM/US$</pre>
<table border="1" cellpadding="0" cellspacing="0" summary="GERMANY -- SPOT EXCHANGE RATE, DM/US$">
<tr><th id="header1">Date</th><th id="header2">Rate</th></tr>
<tr><th id="rowheader1" axis="Date">&nbsp;4-<ABBR title="January">Jan-</ABBR>71&nbsp;</th><td headers="header2 rowheader1">&nbsp;&nbsp;&nbsp;3.6434&nbsp;</td></tr>
<tr><th id="rowheader2" axis="Date">&nbsp;5-<ABBR title="January">Jan-</ABBR>71&nbsp;</th><td headers="header2 rowheader2">&nbsp;&nbsp;<ABBR title="No Data">ND</ABBR>&nbsp;</td></tr>
<tr><th id="rowheader3" axis="Date">31-<ABBR title="December">Dec-</ABBR>98&nbsp;</th><td headers="header2 rowheader3">&nbsp;1.6670&nbsp;</td></tr>
</table></body></html>
"""  # noqa: E501


def _oecd_csv(values: list[float], unit: str = "PA") -> bytes:
    head = ("DATAFLOW,REF_AREA,FREQ,MEASURE,UNIT_MEASURE,ACTIVITY,ADJUSTMENT,TRANSFORMATION,"
            "TIME_HORIZ,METHODOLOGY,TIME_PERIOD,OBS_VALUE,OBS_STATUS,UNIT_MULT,DECIMALS,BASE_PER\n")
    periods = pd.period_range("2000-01", periods=len(values), freq="M")
    rows = [f"X,DEU,M,IR3TIB,{unit},_Z,_Z,_Z,_Z,N,{p},{v},A,0,2," for p, v in zip(periods, values)]
    return (head + "\n".join(rows) + "\n").encode()


BIS_CSV = (
    'STRUCTURE,STRUCTURE_ID,ACTION,FREQ:Frequency,REF_AREA:Reference area,'
    'TIME_PERIOD:Time period or range,OBS_VALUE:Observation Value,UNIT_MEASURE:Unit of measure\n'
    'dataflow,BIS:WS_CBPOL(1.0),I,D: Daily,JP: Japan,1980-01-02,6.25,368: Per cent per year\n'
    'dataflow,BIS:WS_CBPOL(1.0),I,D: Daily,JP: Japan,1980-01-03,6.25,368: Per cent per year\n'
    'dataflow,BIS:WS_CBPOL(1.0),I,D: Daily,GB: United Kingdom,1980-01-02,17.0,368: Per cent per year\n'
    'dataflow,BIS:WS_CBPOL(1.0),I,M: Monthly,JP: Japan,1980-01,6.25,368: Per cent per year\n'
    'dataflow,BIS:WS_CBPOL(1.0),I,D: Daily,ZZ: Nowhere,1980-01-02,99.0,368: Per cent per year\n'
).encode()


def _synthetic_src(start: str = "1998-06-01", end: str = "1999-06-30", seed: int = 0) -> dict:
    """A tiny but complete source bundle (all H.10 series, legacy, OECD, BIS, FF) across the euro join."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end, name="date")
    n = len(idx)

    def walk(level: float) -> np.ndarray:
        return level * np.exp(np.cumsum(rng.normal(0, 0.004, n)))

    legacy_levels = {"DEM": 1.70, "FRF": 5.70, "ITL": 1680.0, "NLG": 1.92, "BEF": 35.0}
    pre = idx < fx.EURO_START
    dem = walk(legacy_levels["DEM"])
    eur_upu = dem / fx.EURO_CONVERSION["DEM"] * np.exp(rng.normal(0, 0.001, n))
    h10 = pd.DataFrame(index=idx)
    h10["RXI$US_N.B.EU"] = np.where(pre, np.nan, 1.0 / eur_upu)
    h10["RXI$US_N.B.UK"] = 1.0 / walk(0.61)
    h10["RXI$US_N.B.AL"] = 1.0 / walk(1.60)
    h10["RXI$US_N.B.NZ"] = 1.0 / walk(1.90)
    h10["RXI_N.B.JA"] = walk(140.0)
    h10["RXI_N.B.CA"] = walk(1.50)
    h10["RXI_N.B.SD"] = walk(8.0)
    h10["RXI_N.B.SZ"] = walk(1.45)
    legacy = {}
    for c, lv in legacy_levels.items():
        s = dem if c == "DEM" else walk(lv)
        # make each legacy currency end consistent with the euro at its irrevocable rate
        s = s * (fx.EURO_CONVERSION[c] / fx.EURO_CONVERSION["DEM"]) / (s[pre][-1] / dem[pre][-1]) \
            if c != "DEM" else s
        legacy[c] = pd.Series(s[pre], index=idx[pre], name=c)
    months = pd.period_range(pd.Timestamp(start) - pd.DateOffset(months=3), end, freq="M")
    oecd = pd.DataFrame({f"{a}:IR3TIB": 0.04 for a in fx.OECD_AREAS}, index=months)
    bis = pd.DataFrame({a: 0.03 for a in fx.BIS_AREAS},
                       index=pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=30), end))
    ff = pd.DataFrame({"RF": 0.0002}, index=idx)
    return {"h10": h10, "h10_attrs": {}, "legacy": legacy, "oecd": oecd, "bis": bis, "ff": ff}


# ---------------------------------------------------------------------------------------------------
# parsers + units
# ---------------------------------------------------------------------------------------------------
def test_parse_h10_xml_daily_only_nd_and_zip() -> None:
    df, attrs = fx.parse_h10_xml(H10_XML.encode())
    assert list(df.columns) == ["RXI$US_N.B.EU", "RXI_N.B.JA"]          # monthly series dropped
    assert df.loc["2007-01-02", "RXI$US_N.B.EU"] == pytest.approx(1.25)
    assert np.isnan(df.loc["2007-01-03", "RXI$US_N.B.EU"])               # ND -> NaN, not -9999
    assert "USD per EUR" in attrs["RXI$US_N.B.EU"]["description"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("H10_data.xml", H10_XML)
    df2, _ = fx.parse_h10_xml(buf.getvalue())
    pd.testing.assert_frame_equal(df, df2)


def test_quote_convention_inverts_usd_per_unit_only() -> None:
    h10 = pd.DataFrame({s: [2.0] for s, _ in fx.H10_FX_SERIES.values()},
                       index=pd.DatetimeIndex(["2010-01-04"]))
    upu = fx.to_units_per_usd(h10)
    for ccy, (_, quote) in fx.H10_FX_SERIES.items():
        assert upu[ccy].iloc[0] == pytest.approx(0.5 if quote == "usd_per_unit" else 2.0)
    assert fx.H10_FX_SERIES["EUR"][1] == fx.H10_FX_SERIES["GBP"][1] == "usd_per_unit"
    assert fx.H10_FX_SERIES["AUD"][1] == "usd_per_unit"
    assert fx.H10_FX_SERIES["JPY"][1] == "units_per_usd"


def test_units_band_catches_inverted_or_scaled_quote() -> None:
    idx = pd.DatetimeIndex(["2010-01-04", "2010-01-05"])
    fx.check_units_per_usd(pd.DataFrame({"JPY": [90.0, 91.0], "EUR": [0.70, 0.71]}, index=idx))
    with pytest.raises(ValueError, match="JPY"):
        fx.check_units_per_usd(pd.DataFrame({"JPY": [1 / 90.0, 1 / 91.0]}, index=idx))  # inverted
    with pytest.raises(ValueError, match="ITL"):
        fx.check_units_per_usd(pd.DataFrame({"ITL": [1.6, 1.61]}, index=idx))  # thousands lost


def test_parse_legacy_html_dates_nd_and_title_unit() -> None:
    s = fx.parse_h10_legacy_html(LEGACY_HTML.encode(), "DEM")
    assert s.index[0] == pd.Timestamp("1971-01-04") and s.iloc[0] == pytest.approx(3.6434)
    assert np.isnan(s.loc["1971-01-05"])
    assert s.index[-1] == pd.Timestamp("1998-12-31")
    with pytest.raises(ValueError, match="quoting convention"):
        fx.parse_h10_legacy_html(LEGACY_HTML.replace("DM/US$", "US$/DM").encode(), "DEM")


def test_oecd_percent_to_decimal_and_planted_units_errors() -> None:
    vals = [5.0 + 0.01 * i for i in range(60)]
    out = fx.parse_oecd_csv(_oecd_csv(vals))
    assert out["DEU:IR3TIB"].iloc[0] == pytest.approx(0.05)             # percent -> decimal once
    assert isinstance(out.index, pd.PeriodIndex)
    with pytest.raises(ValueError, match="DECIMAL"):                    # PLANTED: file already decimal
        fx.parse_oecd_csv(_oecd_csv([v / 100 for v in vals]))
    with pytest.raises(ValueError, match="BASIS POINTS"):               # PLANTED: basis points
        fx.parse_oecd_csv(_oecd_csv([v * 100 for v in vals]))
    with pytest.raises(ValueError, match="PA"):
        fx.parse_oecd_csv(_oecd_csv(vals, unit="IX"))


def test_parse_bis_daily_only_known_areas_decimal() -> None:
    out = fx.parse_bis_cbpol(BIS_CSV)
    assert set(out.columns) == {"JP", "GB"}                             # ZZ filtered, M dropped
    assert out.loc["1980-01-02", "GB"] == pytest.approx(0.17)
    assert len(out) == 2


def test_planted_unconverted_percent_chain_is_caught() -> None:
    idx = pd.bdate_range("2010-01-01", periods=5)
    fx.check_rate_units_decimal(pd.Series(0.05, index=idx), "ok")
    with pytest.raises(ValueError, match="units"):
        fx.check_rate_units_decimal(pd.Series(5.0, index=idx), "planted")


# ---------------------------------------------------------------------------------------------------
# causality (LEAK-2)
# ---------------------------------------------------------------------------------------------------
def _assert_monthly_mapping_causal(fn) -> None:
    """Perturbing the month-m average must not change any daily value inside month m or before."""
    months = pd.period_range("2010-01", "2010-04", freq="M")
    base = pd.Series([0.01, 0.02, 0.03, 0.04], index=months)
    idx = pd.bdate_range("2010-02-01", "2010-04-30")
    a = fn(base, idx)
    bumped = base.copy()
    bumped[pd.Period("2010-03", "M")] = 0.99
    b = fn(bumped, idx)
    upto_march = idx <= pd.Timestamp("2010-03-31")
    np.testing.assert_allclose(a[upto_march].to_numpy(), b[upto_march].to_numpy())
    assert (b[idx >= pd.Timestamp("2010-04-01")] == 0.99).all()


def test_monthly_rate_used_only_from_next_month() -> None:
    _assert_monthly_mapping_causal(fx.monthly_to_daily_causal)
    months = pd.period_range("2010-01", "2010-02", freq="M")
    d = fx.monthly_to_daily_causal(pd.Series([0.01, 0.02], index=months),
                                   pd.bdate_range("2010-02-01", "2010-03-05"))
    assert (d.loc["2010-02"] == 0.01).all() and (d.loc["2010-03"] == 0.02).all()


def test_planted_leaky_monthly_mapping_is_caught() -> None:
    def leaky(m: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:   # month m used IN month m
        s = pd.Series(m.to_numpy(), index=m.index.to_timestamp(how="start"))
        return s.reindex(s.index.union(idx)).ffill().reindex(idx)

    with pytest.raises(AssertionError):
        _assert_monthly_mapping_causal(leaky)


def test_monthly_rate_goes_stale_then_falls_back() -> None:
    months = pd.period_range("2010-01", "2010-01", freq="M")
    idx = pd.bdate_range("2010-02-01", "2010-04-30")
    d = fx.monthly_to_daily_causal(pd.Series([0.01], index=months), idx)
    assert d.loc["2010-02"].notna().all() and d.loc["2010-04"].isna().all()


def test_accrual_uses_rate_known_at_previous_day() -> None:
    idx = pd.bdate_range("2010-01-04", periods=3)
    upu = pd.Series(1.0, index=idx)
    rate = pd.Series([0.0, 0.36, 0.0], index=idx)                   # a rate that exists only on day 2
    r = fx.fx_trust_returns(upu, rate, 360, 0.0)
    assert r.iloc[1] == pytest.approx(0.0)                          # day-2 rate NOT used for (1, 2]
    assert r.iloc[2] == pytest.approx(0.36 / 360)                   # used for (2, 3]


# ---------------------------------------------------------------------------------------------------
# return construction
# ---------------------------------------------------------------------------------------------------
def test_fx_trust_return_hand_computed() -> None:
    idx = pd.DatetimeIndex(["2020-01-03", "2020-01-06"])           # Friday -> Monday, 3 calendar days
    upu = pd.Series([0.9, 0.8], index=idx)                          # foreign currency appreciates
    rate = pd.Series([0.036, 0.036], index=idx)
    r = fx.fx_trust_returns(upu, rate, 360, 0.004)
    expected = (0.9 / 0.8) * (1 + 0.036 * 3 / 360) - 1 - 0.004 * 3 / 365
    assert r.iloc[1] == pytest.approx(expected, rel=1e-12)
    spot = fx.fx_trust_returns(upu, rate, 360, 0.004, carry=False)
    assert spot.iloc[1] == pytest.approx(0.9 / 0.8 - 1)


def test_levels_start_at_one_and_reject_inner_nan() -> None:
    r = pd.Series([np.nan, 0.1, -0.5], index=pd.bdate_range("2010-01-04", periods=3), name="x")
    lv = fx.levels_from_returns(r)
    np.testing.assert_allclose(lv.to_numpy(), [1.0, 1.1, 0.55])
    with pytest.raises(ValueError, match="NaN"):
        fx.levels_from_returns(pd.Series([0.0, np.nan, 0.1], index=r.index, name="x"))


def test_dxy_formula_constant_weights_and_direction() -> None:
    assert sum(fx.DXY_WEIGHTS_EURO.values()) == pytest.approx(1.0)
    assert sum(fx.DXY_WEIGHTS_LEGACY.values()) == pytest.approx(1.0)
    assert sum(fx.DXY_WEIGHTS_LEGACY[c] for c in fx.EURO_CONVERSION) == pytest.approx(
        fx.DXY_WEIGHTS_EURO["EUR"])
    idx = pd.bdate_range("2010-01-04", periods=2)
    ones = pd.DataFrame({c: [1.0, 1.01] for c in ALL_DXY}, index=idx)
    assert fx.dxy_level(ones).iloc[0] == pytest.approx(fx.DXY_CONST)     # all rates 1 -> constant
    assert fx.dxy_log_returns(ones).iloc[1] == pytest.approx(np.log(1.01))  # USD up 1% vs all -> +1%
    # ICE sign convention: EURUSD^-0.576 -- a stronger euro (fewer EUR per USD) lowers the index
    eur_up = ones.copy()
    eur_up.loc[idx[1]] = 1.0
    eur_up.loc[idx[1], "EUR"] = 0.9
    assert fx.dxy_log_returns(eur_up).iloc[1] == pytest.approx(0.576 * np.log(0.9))


def test_legacy_and_euro_dxy_formulas_agree_after_1999() -> None:
    upu = fx.spot_panel(_synthetic_src(), fixes=[])
    lr = np.log(upu / upu.shift(1))
    legacy = sum(w * lr[c] for c, w in fx.DXY_WEIGHTS_LEGACY.items())
    euro = sum(w * lr[c] for c, w in fx.DXY_WEIGHTS_EURO.items())
    post = upu.index > fx.EURO_START
    np.testing.assert_allclose(legacy[post].to_numpy(), euro[post].to_numpy(), atol=1e-12)
    for c, k in fx.EURO_CONVERSION.items():                          # legacy legs implied from EUR
        np.testing.assert_allclose(upu.loc[post, c].to_numpy(), (upu.loc[post, "EUR"] * k).to_numpy())


def test_euro_splice_uses_irrevocable_rate_and_planted_omission_is_caught() -> None:
    upu = fx.spot_panel(_synthetic_src(), fixes=[])
    before = upu.loc[upu.index < fx.EURO_START, "EUR"]
    dem = upu.loc[before.index, "DEM"]
    np.testing.assert_allclose(before.to_numpy(), (dem / 1.95583).to_numpy())
    fx.check_euro_join(upu)
    broken = upu.copy()                                              # PLANTED: DEM used as EUR
    broken.loc[broken.index < fx.EURO_START, "EUR"] = dem
    with pytest.raises(ValueError, match="euro join"):
        fx.check_euro_join(broken)


def test_uup_funded_is_excess_plus_collateral() -> None:
    idx = pd.bdate_range("2010-01-04", periods=3)
    upu = pd.DataFrame({c: 1.0 for c in ALL_DXY}, index=idx)
    rates = pd.DataFrame({c: 0.036 for c in ALL_DXY}, index=idx)
    us = pd.Series(0.0002, index=idx)
    out = fx.uup_returns(upu, rates, us, fee=0.0)
    fb = sum(w * 0.036 / fx.DAY_BASIS[c] for c, w in fx.DXY_WEIGHTS_EURO.items())   # 1 calendar day
    assert out["spot"].iloc[1] == pytest.approx(0.0)
    assert out["excess"].iloc[1] == pytest.approx(0.0002 - fb)
    assert out["funded"].iloc[1] == pytest.approx(2 * 0.0002 - fb)


def test_construct_levels_end_to_end_synthetic() -> None:
    lv, meta = fx.construct_levels(_synthetic_src(), fixes=[])
    for tk in fx.TICKERS:
        pd.testing.assert_series_equal(lv[tk], lv[f"{tk}__{fx.RECOMMENDED[tk]}"], check_names=False)
        assert lv[tk].dropna().iloc[0] == pytest.approx(1.0)
        assert lv[tk].loc[lv[tk].first_valid_index():].notna().all()
    assert all(v == 0 for v in meta["days_without_foreign_rate"].values())
    assert meta["rate_segments"]["EUR"][0]["source"].startswith("oecd:DEU")
    assert meta["rate_segments"]["EUR"][-1]["source"].startswith("oecd:EA20")


# ---------------------------------------------------------------------------------------------------
# holidays, fixes, spike screen
# ---------------------------------------------------------------------------------------------------
def test_fill_holidays_limit_and_known_closure() -> None:
    idx = pd.bdate_range("2010-01-04", periods=12)
    s = pd.Series([1.0] + [np.nan] * 3 + [1.1] * 8, index=idx, name="X")
    assert fx.fill_holidays(s, idx).notna().all()
    long_gap = pd.Series([1.0] + [np.nan] * 7 + [1.1] * 4, index=idx, name="X")
    with pytest.raises(ValueError, match="gap longer"):
        fx.fill_holidays(long_gap, idx)
    closure = [(str(idx[1].date()), str(idx[7].date()), "test closure")]
    assert fx.fill_holidays(long_gap, idx, closures=closure).notna().all()


def test_apply_fixes_and_stale_fix_list_raises() -> None:
    idx = pd.bdate_range("2010-01-04", periods=3)
    df = pd.DataFrame({"SEK": [7.0, 7.5, 7.0]}, index=idx)
    out = fx.apply_fixes(df, [{"series": "SEK", "date": "2010-01-05", "action": "nan"}])
    assert np.isnan(out.loc["2010-01-05", "SEK"]) and not np.isnan(df.loc["2010-01-05", "SEK"])
    with pytest.raises(ValueError, match="stale"):
        fx.apply_fixes(df, [{"series": "SEK", "date": "2011-01-05", "action": "nan"}])


def test_isolated_spike_screen_flags_keying_error_not_common_move() -> None:
    idx = pd.bdate_range("2010-01-04", periods=6)
    base = pd.DataFrame({c: np.full(6, 2.0) for c in ["A", "B", "C"]}, index=idx)
    typo = base.copy()
    typo.loc[idx[3], "A"] = 2.1                                     # +4.9% then back: one currency
    hits = fx.find_isolated_spikes(typo)
    assert [(h["series"], h["date"]) for h in hits] == [("A", str(idx[3].date()))]
    common = base.copy()
    common.loc[idx[3]] = 2.1                                        # every currency moves: real event
    assert fx.find_isolated_spikes(common) == []


def test_data_fixes_are_well_formed_and_sparse() -> None:
    assert len(fx.DATA_FIXES) <= 20
    for f in fx.DATA_FIXES:
        assert f["action"] == "nan" and f["series"] in fx.UNITS_PER_USD_BANDS
        pd.Timestamp(f["date"])


# ---------------------------------------------------------------------------------------------------
# seal
# ---------------------------------------------------------------------------------------------------
def _levels_across_seal() -> pd.DataFrame:
    idx = pd.bdate_range("2005-12-20", "2006-01-20")
    return pd.DataFrame({"FXE": np.linspace(3.0, 3.3, len(idx))}, index=idx)


def test_backward_window_raises_while_sealed(tmp_path: Path) -> None:
    gates = tmp_path / "missing.gates.yaml"
    with pytest.raises(seal.SealedError):
        fx.load_fx("backward", levels=_levels_across_seal(), gates_path=gates)
    # without levels it must refuse BEFORE building anything (empty raw dir, no network)
    with pytest.raises(seal.SealedError):
        fx.load_fx("backward", raw_dir=tmp_path / "empty", gates_path=gates)
    if not seal.is_unsealed():                                     # the real repository state
        with pytest.raises(seal.SealedError):
            fx.load_fx("backward", levels=_levels_across_seal())


def test_dev_window_is_rebased_and_has_no_sealed_rows() -> None:
    dev = fx.load_fx("dev", levels=_levels_across_seal())
    seal.assert_no_sealed_rows(dev)
    assert dev.index.min() > seal.SEAL_END
    assert dev["FXE"].iloc[0] == pytest.approx(1.0)                 # no sealed growth carried in
    with pytest.raises(ValueError):
        fx.load_fx("all", levels=_levels_across_seal())


def test_fidelity_ignores_sealed_rows() -> None:
    idx = pd.bdate_range("2004-01-01", "2007-12-31")
    rng = np.random.default_rng(1)
    etf = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.006, len(idx)))), index=idx)
    proxy = etf.copy()
    proxy[idx <= seal.SEAL_END] *= rng.uniform(0.5, 2.0, int((idx <= seal.SEAL_END).sum()))
    f = fx.fidelity_pair(proxy, etf)                                # sealed part is garbage
    assert pd.Timestamp(f["first"]) > seal.SEAL_END
    assert f["daily_corr"] == pytest.approx(1.0) and f["growth_ratio"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------------------------------
# PLANTED BUGS caught by the dev-window fidelity checks
# ---------------------------------------------------------------------------------------------------
def _dev_fixture(seed: int = 3) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic H.10 JPY quotes (dev window) and a synthetic FXY-like ETF built from the truth."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", "2012-12-31", name="date")
    jpy = 90.0 * np.exp(np.cumsum(rng.normal(0, 0.006, len(idx))))
    h10 = pd.DataFrame({s: 1.0 for s, _ in fx.H10_FX_SERIES.values()}, index=idx)
    h10["RXI_N.B.JA"] = jpy
    etf = pd.Series((1.0 / jpy) * np.exp(rng.normal(0, 0.0005, len(idx))), index=idx, name="FXY")
    return h10, etf


def _fxy_proxy(h10: pd.DataFrame) -> pd.Series:
    upu = fx.to_units_per_usd(h10)
    r = fx.fx_trust_returns(upu["JPY"], pd.Series(0.0, index=upu.index), 365, fx.FEES["FXY"])
    return fx.levels_from_returns(r.rename("FXY"))


def test_planted_quote_inversion_is_caught_by_fidelity_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    h10, etf = _dev_fixture()
    good = {"proxies": {"FXY": fx.fidelity_pair(_fxy_proxy(h10), etf)}}
    fx.fidelity_gate(good, ["FXY"])                                 # the correct build passes
    monkeypatch.setitem(fx.H10_FX_SERIES, "JPY", ("RXI_N.B.JA", "usd_per_unit"))  # PLANTED sign flip
    bad = {"proxies": {"FXY": fx.fidelity_pair(_fxy_proxy(h10), etf)}}
    assert bad["proxies"]["FXY"]["weekly_corr"] < 0
    with pytest.raises(ValueError, match="fidelity gate"):
        fx.fidelity_gate(bad, ["FXY"])


def test_planted_one_day_misalignment_is_caught() -> None:
    h10, etf = _dev_fixture()
    proxy = _fxy_proxy(h10)
    good = fx.fidelity_pair(proxy, etf)
    shifted = fx.fidelity_pair(proxy.shift(1), etf)                 # PLANTED: stamped one day late
    assert good["daily_corr"] > 0.95
    assert shifted["daily_corr"] < 0.2                              # lag-0 correlation collapses
    assert shifted["daily_corr_proxy_lag1"] > 0.95                  # ...and moves to the lag
