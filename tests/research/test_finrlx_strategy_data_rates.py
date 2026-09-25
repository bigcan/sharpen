"""Offline tripwires for scripts/research/finrlx_strategy/data_rates.py (rates-area long-history proxies).

No network: every fixture is synthetic or an inline CSV / XML string. Pinned properties:
  - H.15 parsing: 'ND' / blank / OBS_STATUS != 'A' (sentinel -9999) become NaN; values stay in percent;
  - units: exactly one percent -> decimal conversion, guarded (percent or -9999 in a decimal path raises);
  - par-bond math: 5% -> 5.1% at M=10 re-prices by -0.7758% (hand annuity formula), carry y_{t-1}*dt with
    weekend dt = 3/365, D and C match finite differences of the exact price, roll-down sign;
  - T-bill discount -> bond-equivalent conversion (hand value);
  - splicing: no dy across two series (no jump), no double-counted return;
  - the seal: window='backward' raises SealedError while sealed; 'dev' has no sealed rows; fidelity ignores
    sealed rows;
  - planted bugs (sign flip, percent units, carry/one-day misalignment, naive splice) are CAUGHT.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from research.finrlx_strategy import data_rates as dr  # noqa: E402
from research.finrlx_strategy import seal  # noqa: E402

# ---------------------------------------------------------------------------------------------------
# Hand-computed reference: par bond, 10y, semi-annual, 5% -> 5.1%.
#   (1.025)^20 = 1.638616440 ; D = (1 - 1/1.638616440) / 0.05 = 7.794581
#   price at 5.1% of a 5% coupon: 0.025 * (1 - 1.0255^-20) / 0.0255 + 1.0255^-20 = 0.992242
# ---------------------------------------------------------------------------------------------------
HAND_D_5PCT_10Y = 7.794581
HAND_EXACT_DP = 0.025 * (1 - 1.0255 ** -20) / 0.0255 + 1.0255 ** -20 - 1.0  # = -0.0077579


def _hand_example_ok(fn) -> bool:
    """The check a correct bond-return function must pass: price move, carry and weekend accrual."""
    move = float(fn(0.05, 0.051, 0.0, 10.0))
    carry = float(fn(0.05, 0.05, 1.0 / 365.0, 10.0))
    weekend = float(fn(0.05, 0.05, 3.0 / 365.0, 10.0))
    carry_on_move = float(fn(0.05, 0.051, 1.0 / 365.0, 10.0)) - move
    return (abs(move - HAND_EXACT_DP) < 1e-6 and abs(carry - 0.05 / 365.0) < 1e-12
            and abs(weekend - 3 * 0.05 / 365.0) < 1e-12 and abs(carry_on_move - 0.05 / 365.0) < 1e-12)


# ---------------------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------------------
DDP_CSV = '''"Series Description","10-year","20-year"
"Unit:","Percent:_Per_Year","Percent:_Per_Year"
"Multiplier:","1","1"
"Currency:","NA","NA"
"Unique Identifier: ","H15/H15/RIFLGFCY10_N.B","H15/H15/RIFLGFCY20_N.B"
"Time Period","RIFLGFCY10_N.B","RIFLGFCY20_N.B"
1962-01-02,4.06,4.07
1962-01-03,ND,ND
1962-01-04,3.99,
'''

SDMX_XML = '''<?xml version="1.0"?><message:MessageGroup>
<kf:Series FREQ="9" SERIES_NAME="RIFSGFSM03_N.B" UNIT="Percent:_Per_Year">
  <frb:Obs OBS_STATUS="A" OBS_VALUE="1.33" TIME_PERIOD="1954-01-04" />
  <frb:Obs OBS_STATUS="ND" OBS_VALUE="-9999" TIME_PERIOD="1954-01-05" />
  <frb:Obs OBS_STATUS="A" OBS_VALUE="1.28" TIME_PERIOD="1954-01-06" />
</kf:Series>
<kf:Series FREQ="129" SERIES_NAME="RIFSGFSM03_N.M" UNIT="Percent:_Per_Year">
  <frb:Obs OBS_STATUS="A" OBS_VALUE="99.0" TIME_PERIOD="1954-01-31" />
</kf:Series>
<kf:Series FREQ="9" SERIES_NAME="RIMLPAAAR_N.B" UNIT="Percent:_Per_Year">
  <frb:Obs OBS_STATUS="NA" OBS_VALUE="-9999" TIME_PERIOD="1954-01-04" />
  <frb:Obs OBS_STATUS="A" OBS_VALUE="3.10" TIME_PERIOD="1954-01-05" />
</kf:Series>
</message:MessageGroup>'''


def test_parse_ddp_csv_nd_blank_and_percent():
    df = dr.parse_ddp_csv(DDP_CSV)
    assert list(df.columns) == ["y10", "y20"]
    assert df.index[0] == pd.Timestamp("1962-01-02")
    assert df.at[pd.Timestamp("1962-01-02"), "y10"] == pytest.approx(4.06)  # percent, not decimal
    assert np.isnan(df.at[pd.Timestamp("1962-01-03"), "y10"])               # 'ND'
    assert np.isnan(df.at[pd.Timestamp("1962-01-04"), "y20"])               # blank


def test_parse_sdmx_status_sentinel_and_series_selection():
    df = dr.parse_h15_sdmx(SDMX_XML, {"tb3m": "RIFSGFSM03_N.B", "aaa": "RIMLPAAAR_N.B"})
    assert df.at[pd.Timestamp("1954-01-04"), "tb3m"] == pytest.approx(1.33)
    assert np.isnan(df.at[pd.Timestamp("1954-01-05"), "tb3m"])   # ND sentinel -9999 dropped
    assert np.isnan(df.at[pd.Timestamp("1954-01-04"), "aaa"])    # NA status dropped
    assert (df.stack() > 0).all()                                 # no -9999 survives
    assert 99.0 not in df.to_numpy()                              # monthly series not mixed in
    with pytest.raises(KeyError):
        dr.parse_h15_sdmx(SDMX_XML, {"baa": "RIMLPBAAR_N.B"})


def test_cross_check_catches_a_disagreeing_encoding():
    idx = pd.bdate_range("2007-01-01", periods=5)
    a = pd.DataFrame({k: np.linspace(4.0, 4.1, 5) for k in dr.CMT_KEYS}, index=idx)
    dr._cross_check_cmt(a, a.copy())
    b = a.copy()
    b.iloc[2, 0] += 0.01
    with pytest.raises(ValueError, match="disagree"):
        dr._cross_check_cmt(a, b)


# ---------------------------------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------------------------------
def test_units_guard_rejects_percent_and_sentinel():
    dr._require_decimal(pd.Series([0.05, 0.12]), "ok")
    with pytest.raises(ValueError, match="percent"):
        dr._require_decimal(pd.Series([5.0, 5.1]), "percent")
    with pytest.raises(ValueError):
        dr._require_decimal(pd.Series([0.05, -99.99]), "sentinel/100")
    assert dr.pct_to_decimal(pd.Series([5.0])).iloc[0] == pytest.approx(0.05)


def test_ff_rf_units_are_daily_decimal():
    # FF daily RF 0.0002 (decimal/day) annualises to ~5%: passes; a percent value 0.02 would annualise to 5.04.
    dr._require_decimal(pd.Series([0.0002]) * 252, "rf")
    with pytest.raises(ValueError):
        dr._require_decimal(pd.Series([0.02]) * 252, "rf in percent")


# ---------------------------------------------------------------------------------------------------
# Bond math
# ---------------------------------------------------------------------------------------------------
def test_par_bond_hand_example():
    assert float(dr.par_duration(0.05, 10)) == pytest.approx(HAND_D_5PCT_10Y, abs=1e-6)
    r = float(dr.bond_return(0.05, 0.051, 0.0, 10))
    assert r == pytest.approx(-0.0077579, abs=2e-7)            # "~ -0.78%"
    assert r == pytest.approx(HAND_EXACT_DP, abs=1e-6)          # vs exact re-pricing
    assert dr.coupon_bond_price(0.05, 0.051, 10) - 1 == pytest.approx(HAND_EXACT_DP, abs=1e-12)
    assert _hand_example_ok(dr.bond_return)


def test_duration_convexity_match_finite_differences():
    for y, m in ((0.03, 25.0), (0.08, 8.5), (0.12, 30.0), (0.05, 10.0)):
        h = 1e-5
        p0 = dr.coupon_bond_price(y, y, m)
        pu, pd_ = dr.coupon_bond_price(y, y + h, m), dr.coupon_bond_price(y, y - h, m)
        assert p0 == pytest.approx(1.0, abs=1e-12)                                  # par
        assert float(dr.par_duration(y, m)) == pytest.approx(-(pu - pd_) / (2 * h), rel=1e-6)
        assert float(dr.par_convexity(y, m)) == pytest.approx((pu - 2 * p0 + pd_) / h**2, rel=1e-4)


def test_leg_returns_weekend_carry_accrues_on_monday():
    idx = pd.DatetimeIndex(["2007-01-04", "2007-01-05", "2007-01-08"])   # Thu, Fri, Mon
    leg = dr.Leg("y", pd.Series([0.05, 0.05, 0.05], index=idx))
    out = dr.leg_returns(leg, dr._par_fn(10.0))
    assert out["r"].iloc[1] == pytest.approx(0.05 / 365.0, abs=1e-15)
    assert out["r"].iloc[2] == pytest.approx(3 * 0.05 / 365.0, abs=1e-15)
    assert out["prev"].iloc[2] == pd.Timestamp("2007-01-05")


def test_roll_down_sign_and_size():
    # Upward curve (+10bp per year of maturity): the held bond's yield falls as it ages -> positive return.
    r = float(dr.roll_down_return(0.04, 0.001, 1.0, 8.5))
    assert r == pytest.approx(float(dr.par_duration(0.04, 8.5)) * 0.001, rel=1e-12)
    assert r > 0
    fn = dr._par_fn(8.5)
    base = fn(np.array([0.04]), np.array([0.04]), np.array([1 / 365]), np.array([0.0]))[0]
    rolled = fn(np.array([0.04]), np.array([0.04]), np.array([1 / 365]), np.array([0.001]))[0]
    assert rolled - base == pytest.approx(float(dr.par_duration(0.04, 8.5)) * 0.001 / 365, rel=1e-9)


def test_tbill_discount_to_bond_equivalent_hand_value():
    # 5% discount, 91 days: 365 * 0.05 / (360 - 0.05 * 91) = 18.25 / 355.45
    assert float(dr.tbill_discount_to_bey(0.05)) == pytest.approx(18.25 / 355.45, abs=1e-15)
    assert float(dr.tbill_discount_to_bey(0.05)) > 0.05          # BEY exceeds the discount rate


# ---------------------------------------------------------------------------------------------------
# Splicing
# ---------------------------------------------------------------------------------------------------
def _two_legs_with_gap() -> tuple[pd.Series, pd.Series]:
    idx = pd.bdate_range("2001-01-01", periods=40)
    y20 = pd.Series(0.05, index=idx)
    y30 = pd.Series(0.06, index=idx)
    y30.iloc[10:25] = np.nan                                      # the 30y goes missing, then returns
    return y20, y30


def _max_excess_over_carry(r: pd.Series) -> float:
    """With every series constant, a correct index earns only carry: at most max(y) * dt per step."""
    dt = r.index.to_series().diff().dt.days / 365.0
    return float((r.abs() - 0.06 * dt).max())


def test_composite_splice_has_no_jump_and_uses_fallback():
    y20, y30 = _two_legs_with_gap()
    legs = [dr.Leg("blend", dr._mean2(y20, y30)), dr.Leg("y30", y30), dr.Leg("y20", y20)]
    comp = dr.composite(legs, dr._par_fn(25.0))
    assert comp.n_unbridged == 0
    assert _max_excess_over_carry(comp.returns.dropna()) <= 1e-12
    assert set(comp.source.iloc[11:25]) == {"y20"}                # gap bridged by the 20y leg
    assert comp.source.iloc[26] == "blend"                        # blend resumes once both exist twice


def test_planted_naive_splice_jump_is_caught():
    # BUG: blend with nanmean -> the yield silently switches from 5.5% to 5.0% inside ONE series -> dy jump.
    y20, y30 = _two_legs_with_gap()
    naive = pd.concat([y20, y30], axis=1).mean(axis=1)
    comp = dr.composite([dr.Leg("naive", naive)], dr._par_fn(25.0))
    assert _max_excess_over_carry(comp.returns.dropna()) > 0.01   # a ~+7% phantom return


def test_composite_does_not_double_count():
    idx = pd.bdate_range("2003-01-01", periods=30)
    rng = np.random.default_rng(0)
    y = pd.Series(0.05 + np.cumsum(rng.normal(0, 5e-4, len(idx))), index=idx)
    hole = y.copy()
    hole.iloc[[5, 6, 17]] = np.nan
    fn = dr._par_fn(10.0)
    ref = dr.composite([dr.Leg("full", y)], fn).returns
    comp = dr.composite([dr.Leg("hole", hole), dr.Leg("full", y)], fn)
    # identical yields in both legs: the spliced index must equal the single-leg index exactly
    assert np.allclose(comp.returns.dropna(), ref.dropna(), atol=1e-15)
    assert set(comp.source.iloc[[5, 6, 7, 17, 18]]) == {"full"}


def test_backfill_leg_bridges_first_date_of_primary():
    idx = pd.bdate_range("1982-12-27", periods=10)
    y10 = pd.Series(0.10, index=idx)
    corp = pd.Series(np.nan, index=idx)
    corp.iloc[5:] = 0.12
    legs = [dr.Leg("corp", corp), dr.Leg("fill", y10, before=idx[5] + pd.Timedelta(days=1))]
    comp = dr.composite(legs, dr._par_fn(10.0))
    assert comp.n_unbridged == 0
    assert list(comp.source.iloc[1:6]) == ["fill"] * 5 and list(comp.source.iloc[6:]) == ["corp"] * 4


def test_levels_start_at_one_ffill_holidays_nan_outside():
    grid = pd.DatetimeIndex(["2007-01-03", "2007-01-04", "2007-01-08"])  # 01-05 missing (holiday)
    r = pd.Series([np.nan, 0.01, 0.02], index=grid)
    bdays = pd.bdate_range("2007-01-01", "2007-01-10")
    lvl = dr.levels_from_returns(r, bdays=bdays)
    assert np.isnan(lvl.loc["2007-01-02"]) and np.isnan(lvl.loc["2007-01-09"])
    assert lvl.loc["2007-01-03"] == 1.0
    assert lvl.loc["2007-01-05"] == pytest.approx(1.01)         # flat over the holiday
    assert lvl.loc["2007-01-08"] == pytest.approx(1.01 * 1.02)


# ---------------------------------------------------------------------------------------------------
# End-to-end on a synthetic H.15 frame
# ---------------------------------------------------------------------------------------------------
def _synthetic_h15(start: str = "2004-06-01", n: int = 600) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n, name="date")
    rng = np.random.default_rng(1)
    walk = np.cumsum(rng.normal(0, 3, n)) / 100.0               # percent-point random walk
    base = {"y5": 4.0, "y7": 4.2, "y10": 4.4, "y20": 4.8, "y30": 4.9, "cmt3m": 3.0, "tb3m": 2.95,
            "aaa": 5.5, "baa": 6.5}
    df = pd.DataFrame({k: v + walk for k, v in base.items()}, index=idx)
    df.iloc[7] = np.nan                                           # a bond-market holiday (ND everywhere)
    df.loc[df.index[100:140], "y30"] = np.nan                     # a 30y gap
    return df[list(dr.SERIES)]


def test_build_rates_synthetic_end_to_end():
    lv, info = dr.build_rates(_synthetic_h15())
    for tkr in ("TLT", "IEF", "LQD", "CASH"):
        assert tkr in lv.columns and info[tkr]["alias_of"] == dr.RECOMMENDED[tkr]
        s = lv[tkr].dropna()
        assert s.iloc[0] == 1.0 and (s > 0).all()
        assert s.isna().sum() == 0
    assert all(v.get("n_unbridged", 0) == 0 for v in info.values())
    assert lv.index.dayofweek.max() <= 4                          # business-day index


def test_planted_units_bug_is_caught(monkeypatch):
    # BUG: forget the percent -> decimal conversion. The units guard must stop construction.
    monkeypatch.setattr(dr, "pct_to_decimal", lambda y: y)
    with pytest.raises(ValueError, match="percent"):
        dr.build_rates(_synthetic_h15())


@pytest.mark.parametrize("bug", ["sign_flip", "percent_units", "carry_on_today", "wrong_maturity"])
def test_planted_bond_math_bugs_fail_the_hand_check(bug):
    def sign_flip(yp, y, dt, m):
        return yp * dt + dr.par_duration(yp, m) * (y - yp) + 0.5 * dr.par_convexity(yp, m) * (y - yp) ** 2

    def percent_units(yp, y, dt, m):
        return dr.bond_return(yp * 100, y * 100, dt, m)

    def carry_on_today(yp, y, dt, m):  # one-day misalignment: accrue at y_t instead of y_{t-1}
        return dr.bond_return(yp, y, dt, m) + (y - yp) * dt

    def wrong_maturity(yp, y, dt, m):  # duration evaluated at the wrong maturity
        return dr.bond_return(yp, y, dt, 2 * m)

    fn = {"sign_flip": sign_flip, "percent_units": percent_units, "carry_on_today": carry_on_today,
          "wrong_maturity": wrong_maturity}[bug]
    assert _hand_example_ok(dr.bond_return)
    assert not _hand_example_ok(fn)


# ---------------------------------------------------------------------------------------------------
# Fidelity + seal
# ---------------------------------------------------------------------------------------------------
def _etf_like(start: str = "2004-01-01", n: int = 900) -> pd.Series:
    idx = pd.bdate_range(start, periods=n, name="date")
    r = np.random.default_rng(2).normal(0.0002, 0.006, n)
    return pd.Series(np.cumprod(1 + r), index=idx)


def test_fidelity_identity_and_dev_only():
    etf = _etf_like()
    f = dr.fidelity_pair(etf, etf)
    assert f["daily_corr"] == pytest.approx(1.0) and f["weekly_corr"] == pytest.approx(1.0)
    assert f["vol_ratio"] == pytest.approx(1.0) and f["tracking_error_ann"] == pytest.approx(0.0, abs=1e-12)
    assert f["growth_ratio"] == pytest.approx(1.0)
    assert pd.Timestamp(f["start"]) > seal.SEAL_END
    # a wild SEALED-window divergence must not move any dev fidelity number
    proxy = etf.copy()
    proxy.loc[:"2005-12-31"] *= np.linspace(1, 3, int((proxy.index <= seal.SEAL_END).sum()))
    g = dr.fidelity_pair(proxy, etf)
    assert all(g[k] == pytest.approx(f[k]) for k in ("daily_corr", "vol_ratio", "tracking_error_ann"))
    # the fee adjustment charges the proxy: exp(-er * years)
    h = dr.fidelity_pair(etf, etf, expense_ratio=0.01)
    assert h["growth_ratio_fee_adj"] == pytest.approx(np.exp(-0.01 * h["years"]))


def test_planted_one_day_misalignment_is_caught_by_fidelity():
    etf = _etf_like()
    good = dr.fidelity_pair(etf, etf)
    lagged = etf.shift(1).dropna()                                # proxy stamped one day late
    bad = dr.fidelity_pair(lagged, etf)
    assert good["daily_corr"] > 0.9
    assert bad["daily_corr"] < 0.3                                # the daily-corr check flags it


def test_select_candidate_prefers_low_noise_plus_drift_error():
    pairs = {
        "A": {"tracking_error_ann": 0.020, "growth_ratio_fee_adj": 0.88, "years": 20.0},   # drift -0.64%/yr
        "B": {"tracking_error_ann": 0.021, "growth_ratio_fee_adj": 1.00, "years": 20.0},
        "C": {"tracking_error_ann": 0.030, "growth_ratio_fee_adj": 1.00, "years": 20.0},
    }
    assert dr.select_candidate(pairs) == "B"
    assert dr.selection_score(pairs["A"]) == pytest.approx(0.020 + abs(np.log(0.88)) / 20)


def _fake_levels() -> pd.DataFrame:
    idx = pd.bdate_range("2005-12-01", "2006-02-28", name="date")
    return pd.DataFrame({"TLT": np.linspace(1, 1.1, len(idx)), "CASH": 1.0}, index=idx)


def test_load_backward_raises_while_sealed(monkeypatch):
    monkeypatch.setattr(dr, "_build_cached", lambda: (_fake_levels(), {}))
    monkeypatch.setattr(seal, "prereg_status", lambda *a, **k: (False, "test: sealed"))
    with pytest.raises(seal.SealedError):
        dr.load_rates("backward")
    dev = dr.load_rates("dev")
    seal.assert_no_sealed_rows(dev)
    assert len(dev) and dev.index.min() > seal.SEAL_END
    with pytest.raises(ValueError):
        dr.load_rates("all")


def test_real_seal_state_blocks_backward(monkeypatch):
    if seal.is_unsealed():
        pytest.skip("pre-registration committed: the clean window is legitimately open")
    monkeypatch.setattr(dr, "_build_cached", lambda: (_fake_levels(), {}))
    with pytest.raises(seal.SealedError):
        dr.load_rates("backward")


def test_hygiene_outlier_counts_are_counts_only():
    idx = pd.bdate_range("1980-01-01", periods=12)
    y = pd.DataFrame({"y10": [10.0, 10.0, 10.0, 10.0, 10.0, 10.5, 10.02, 10.1, 11.0, 11.1, 11.2, 11.3]}, index=idx)
    c = dr.yield_outlier_counts(y)
    assert c.loc["y10", "n_spike_reversals"] == 1                  # 10.0 -> 10.5 -> 10.02
    assert c.loc["y10", "n_abs_dy_gt_50bp"] == 1                   # 10.1 -> 11.0 (+90bp); +50bp is not > 50
    assert c.loc["y10", "n_stale_runs_ge5"] == 1
    assert dr.suspicious_dates(y) == {"y10": [str(idx[5].date())]}
