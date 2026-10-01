"""The Taiwan small/mid-cap universe builder — its sealed default and its two opt-in corrections.

`build_membership` defines the band every Taiwan small-cap result was scored on, so the first thing
pinned here is that the DEFAULT call has not moved: it is compared, byte for byte, against a frozen
copy of the builder as it stood before the corrections existed. The corrections are opt-in:

  * ``max_stale_days`` — a name that stopped trading no longer keeps its cap-rank slot;
  * ``seg_gap_days``   — a re-issued stock code earns its own history and its own ADV.

Synthetic bars only (`data/` is gitignored, and no real price belongs in `tests/`). The fixture is
deterministic and small enough to reason about by hand: every name has a fixed share count and a
close that wiggles ±2% around a level, with levels at least 14% apart, so the cap ORDER never
flips and each test can say exactly who is in the band.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research import taiwan_smallcap_universe as tsu  # noqa: E402

_CAL = pd.bdate_range("2018-01-01", "2021-12-31")
_REBALANCES = tsu.month_end_rebalances(_CAL)
_BAND = {"lo_rank": 2, "hi_rank": 5}          # 4 slots over 7-9 names, so both cuts bind

#: code -> close level. Every name has the same share count, so this is also the cap order.
_LIVE = {"1001": 100.0, "1002": 50.0, "1004": 30.0, "1006": 15.0, "1007": 10.0, "1008": 5.0}
_GHOST, _GHOST_CLOSE = "1003", 40.0           # ranks 3rd while it exists
_REUSED = "1005"                              # first company closes ~20, second ~25: ranks 4th
_STRAY, _STRAY_CLOSE = "1009", 35.0


# --------------------------------------------------------------------------- #
# SEALED REFERENCE — the builder exactly as it stood before the opt-in corrections were added.
# DO NOT EDIT, and do not "tidy" it to match the live module: it is the definition of "today's
# behaviour" that the default path is pinned to.
# --------------------------------------------------------------------------- #
def _sealed_month_end_rebalances(dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    s = pd.Series(1, index=pd.DatetimeIndex(sorted(set(dates))))
    return list(s.groupby([s.index.year, s.index.month]).apply(lambda g: g.index.max()))


def _sealed_shares_asof(shareholding: pd.DataFrame) -> pd.DataFrame:
    if shareholding.empty or "total_shares" not in shareholding.columns:
        return pd.DataFrame(columns=["stock_id", "avail_date", "total_shares"])
    sh = shareholding.dropna(subset=["total_shares"]).copy()
    sh["avail_date"] = pd.to_datetime(sh["avail_date"])
    return (sh[["stock_id", "avail_date", "total_shares"]]
            .sort_values(["stock_id", "avail_date"]).reset_index(drop=True))


def _sealed_build_membership(prices: pd.DataFrame, shareholding: pd.DataFrame, *,
                             lo_rank: int = 51, hi_rank: int = 250, adv_window: int = 60,
                             min_adv_twd: float = 5.0e6, min_history: int = 250) -> pd.DataFrame:
    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"])
    px["ticker"] = px["ticker"].astype(str)
    px = px.dropna(subset=["close"]).sort_values(["ticker", "date"])
    px["dollar"] = px["close"] * px["volume"].where(px["volume"] > 0)
    px["adv"] = (px.groupby("ticker")["dollar"]
                 .transform(lambda s: s.rolling(adv_window, min_periods=max(10, adv_window // 3)).mean()))
    px["nobs"] = px.groupby("ticker").cumcount() + 1

    shares = _sealed_shares_asof(shareholding)
    rebalances = _sealed_month_end_rebalances(pd.DatetimeIndex(px["date"].unique()))
    out_rows: list[pd.DataFrame] = []

    for t in rebalances:
        snap = (px[px["date"] <= t].groupby("ticker").tail(1)     # last causal bar per name <= t
                .loc[:, ["ticker", "close", "adv", "nobs"]])
        snap = snap[(snap["nobs"] >= min_history) & (snap["adv"] >= min_adv_twd) & (snap["close"] > 0)]
        if snap.empty:
            continue
        # causal as-of shares: last register with avail_date <= t
        if not shares.empty:
            sh_t = (shares[shares["avail_date"] <= t].groupby("stock_id").tail(1)
                    .rename(columns={"stock_id": "ticker"})[["ticker", "total_shares"]])
            snap = snap.merge(sh_t, on="ticker", how="left")
        else:
            snap["total_shares"] = np.nan
        snap = snap.dropna(subset=["total_shares"])
        if snap.empty:
            continue
        snap["market_cap"] = snap["close"] * snap["total_shares"]
        snap = snap.sort_values("market_cap", ascending=False).reset_index(drop=True)
        snap["rank"] = snap.index + 1
        members = snap[(snap["rank"] >= lo_rank) & (snap["rank"] <= hi_rank)].copy()
        members["rebalance_date"] = t
        out_rows.append(members[["rebalance_date", "ticker", "rank", "market_cap", "adv"]]
                        .rename(columns={"ticker": "stock_id", "adv": "adv_twd"}))

    if not out_rows:
        return pd.DataFrame(columns=["rebalance_date", "stock_id", "rank", "market_cap", "adv_twd"])
    return pd.concat(out_rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# Fixture
# --------------------------------------------------------------------------- #
def _bars(code: str, dates: pd.DatetimeIndex, *, close: float, volume: float = 1.0e7) -> pd.DataFrame:
    """Deterministic bars: close wiggles ±2%, volume ±50%, so the rolling ADV is a non-trivial float."""
    i = np.arange(len(dates), dtype=np.float64)
    return pd.DataFrame({"date": dates, "ticker": code,
                         "close": close * (1.0 + 0.02 * np.sin(i / 9.0)),
                         "volume": volume * (1.0 + 0.5 * np.cos(i / 7.0))})


def _live() -> list[pd.DataFrame]:
    return [_bars(code, _CAL, close=level) for code, level in _LIVE.items()]


def _shares(prices: pd.DataFrame) -> pd.DataFrame:
    """Two register prints per name (the second 5% higher) — an as-of merge with something to do."""
    codes = sorted(prices["ticker"].unique())
    first = pd.DataFrame({"stock_id": codes, "avail_date": _CAL[0], "total_shares": 1.0e9})
    second = first.assign(avail_date=pd.Timestamp("2020-01-03"), total_shares=1.05e9)
    return pd.concat([first, second], ignore_index=True)


def _universe(*extra: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = pd.concat([*_live(), *extra], ignore_index=True)
    return prices, _shares(prices)


def _ghost(last_bar: str = "2019-06-12") -> pd.DataFrame:
    """A name that trades from the start and then stops for good."""
    return _bars(_GHOST, _CAL[_CAL <= last_bar], close=_GHOST_CLOSE)


_FIRST_ENDS, _SECOND_STARTS = pd.Timestamp("2019-02-28"), pd.Timestamp("2020-06-01")   # 459 days apart
_SECOND = _CAL[_CAL >= _SECOND_STARTS]


def _reused(second_volume: float = 1.0e7) -> list[pd.DataFrame]:
    """One stock code, two companies, more than a year of silence between them."""
    return [_bars(_REUSED, _CAL[_CAL <= _FIRST_ENDS], close=20.0),
            _bars(_REUSED, _SECOND, close=25.0, volume=second_volume)]


_STRAY_LAST_REAL = pd.Timestamp("2019-03-15")
_STRAY_PRINTS = _CAL[(_CAL >= "2021-06-21") & (_CAL <= "2021-06-28")]      # six bars, two years on


def _stray(prints: pd.DatetimeIndex = _STRAY_PRINTS) -> list[pd.DataFrame]:
    """A name that stopped trading, plus a handful of prints dated after its last real bar."""
    return [_bars(_STRAY, _CAL[_CAL <= _STRAY_LAST_REAL], close=_STRAY_CLOSE),
            _bars(_STRAY, prints, close=_STRAY_CLOSE)]


def _member_dates(mem: pd.DataFrame, code: str) -> set[pd.Timestamp]:
    return set(mem.loc[mem["stock_id"] == code, "rebalance_date"])


def _band_at(mem: pd.DataFrame, t: pd.Timestamp) -> list[str]:
    return list(mem.loc[mem["rebalance_date"] == t].sort_values("rank")["stock_id"])


def _assert_bitwise_equal(got: pd.DataFrame, want: pd.DataFrame) -> None:
    """Same columns, dtypes, index, and the same BYTES in every column (stricter than allclose)."""
    assert list(got.columns) == list(want.columns)
    assert got.dtypes.to_dict() == want.dtypes.to_dict()
    assert got.index.equals(want.index)
    for c in want.columns:
        a, b = got[c].to_numpy(), want[c].to_numpy()
        if a.dtype == object:
            assert a.tolist() == b.tolist(), f"column {c} differs"
        else:
            assert a.tobytes() == b.tobytes(), f"column {c} differs at the byte level"


# --------------------------------------------------------------------------- #
# 1. The default has not moved
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("params", [
    _BAND,
    {"lo_rank": 1, "hi_rank": 3, "adv_window": 20, "min_adv_twd": 1.0e6, "min_history": 60},
    {"lo_rank": 3, "hi_rank": 8, "min_adv_twd": 2.0e8},          # the ADV floor binds here
])
def test_default_reproduces_the_sealed_builder_byte_for_byte(params):
    """With both switches off — omitted, or passed as None — the output is the sealed builder's."""
    prices, shares = _universe(_ghost(), *_reused(), *_stray())
    want = _sealed_build_membership(prices, shares, **params)
    assert len(want) > 0, "an empty membership would make this comparison vacuous"

    _assert_bitwise_equal(tsu.build_membership(prices, shares, **params), want)
    _assert_bitwise_equal(tsu.build_membership(prices, shares, **params, max_stale_days=None,
                                               seg_gap_days=None), want)


def test_default_still_has_the_rows_the_switches_remove():
    """The comparison above must be on a fixture where the corrections WOULD change something: with
    nothing to correct, a builder with the switches hard-wired ON would match the sealed one too."""
    prices, shares = _universe(_ghost(), *_reused(), *_stray())
    default = tsu.build_membership(prices, shares, **_BAND)
    last = _REBALANCES[-1]
    assert last in _member_dates(default, _GHOST), "the stopped name should still hold its slot"
    assert last in _member_dates(default, _STRAY)
    both = tsu.build_membership(prices, shares, **_BAND, max_stale_days=10, seg_gap_days=365)
    assert len(both) == len(default), "the band is the same size either way — only WHO is in it moves"
    assert set(map(tuple, both[["rebalance_date", "stock_id"]].to_numpy())) != \
        set(map(tuple, default[["rebalance_date", "stock_id"]].to_numpy()))


# --------------------------------------------------------------------------- #
# 2. max_stale_days — a name that stopped trading leaves the band
# --------------------------------------------------------------------------- #
def test_a_stopped_name_leaves_the_band_at_the_next_rebalance():
    last_bar = pd.Timestamp("2019-06-12")
    prices, shares = _universe(_ghost(str(last_bar.date())))
    off = tsu.build_membership(prices, shares, **_BAND)
    on = tsu.build_membership(prices, shares, **_BAND, max_stale_days=10)

    eligible = [t for t in _REBALANCES if int((_CAL <= t).sum()) >= 250]
    traded, gone = [t for t in eligible if t <= last_bar], [t for t in eligible if t > last_bar]
    assert traded[-1] == pd.Timestamp("2019-05-31") and gone[0] == pd.Timestamp("2019-06-28")

    # sealed behaviour: ranked on its final close at every later rebalance, for good
    assert _member_dates(off, _GHOST) == set(eligible)
    # corrected: a member while it trades, out at the first rebalance after it stops
    assert _member_dates(on, _GHOST) == set(traded)

    # while the name trades the switch changes nothing at all
    cut = on["rebalance_date"] <= traded[-1]
    _assert_bitwise_equal(on[cut].reset_index(drop=True),
                          off[off["rebalance_date"] <= traded[-1]].reset_index(drop=True))
    # afterwards its slot goes to the next tradeable name; the band keeps its size
    for t in gone:
        assert _band_at(off, t) == ["1002", _GHOST, "1004", "1006"]
        assert _band_at(on, t) == ["1002", "1004", "1006", "1007"]


@pytest.mark.parametrize(("last_bar", "still_member"), [
    ("2019-06-18", True),      # exactly 10 calendar days before the 2019-06-28 rebalance — kept
    ("2019-06-17", False),     # 11 days — dropped
])
def test_the_staleness_bound_is_inclusive_and_in_calendar_days(last_bar, still_member):
    prices, shares = _universe(_ghost(last_bar))
    on = tsu.build_membership(prices, shares, **_BAND, max_stale_days=10)
    assert (pd.Timestamp("2019-06-28") in _member_dates(on, _GHOST)) is still_member
    assert pd.Timestamp("2019-07-31") not in _member_dates(on, _GHOST)


def test_zero_stale_days_is_a_bound_not_off():
    """`0` means "traded on the rebalance day itself"; it must not be read as falsy / switched off."""
    prices, shares = _universe(_ghost("2019-06-28"))       # last bar ON the June rebalance
    on = tsu.build_membership(prices, shares, **_BAND, max_stale_days=0)
    assert pd.Timestamp("2019-06-28") in _member_dates(on, _GHOST)
    assert pd.Timestamp("2019-07-31") not in _member_dates(on, _GHOST)
    assert _band_at(on, pd.Timestamp("2019-07-31")) == ["1002", "1004", "1006", "1007"]


def _stale_members(mem: pd.DataFrame, prices: pd.DataFrame, max_stale_days: int) -> set[tuple]:
    """(rebalance, name) member rows whose name had NOT traded within `max_stale_days` of the
    rebalance — re-derived from the raw bars, with no reference to how the builder decides."""
    bars = {tk: np.sort(g["date"].to_numpy()) for tk, g in prices.groupby("ticker")}
    out = set()
    for row in mem.itertuples():
        seen = bars[row.stock_id]
        last = pd.Timestamp(seen[seen <= np.datetime64(row.rebalance_date)].max())
        if (row.rebalance_date - last).days > max_stale_days:
            out.add((row.rebalance_date, row.stock_id))
    return out


def test_removing_the_recency_check_would_be_caught():
    """NEGATIVE test for the recency check.

    The property is "no member is staler than the bound". It holds with the switch on. With the
    switch off — which is exactly the builder with the recency check deleted — it must FAIL on this
    fixture, and fail on precisely the stopped name's post-exit rebalances. If someone removes the
    check (or stops passing `max_stale_days` through), `on` becomes `off` and the first assertion
    below reports those rows.
    """
    last_bar = pd.Timestamp("2019-06-12")
    prices, shares = _universe(_ghost(str(last_bar.date())))
    on = tsu.build_membership(prices, shares, **_BAND, max_stale_days=10)
    off = tsu.build_membership(prices, shares, **_BAND)

    assert _stale_members(on, prices, 10) == set(), "a member had not traded within the bound"
    assert _stale_members(off, prices, 10) == {(t, _GHOST) for t in _REBALANCES if t > last_bar}, (
        "without the recency check the stopped name should hold its slot at every later rebalance "
        "— if it does not, this fixture cannot tell the check from its absence")


# --------------------------------------------------------------------------- #
# 3. seg_gap_days — a re-issued code earns its own history and its own ADV
# --------------------------------------------------------------------------- #
def test_a_reused_code_needs_its_own_250_bars():
    prices, shares = _universe(*_reused())
    off = tsu.build_membership(prices, shares, **_BAND)
    on = tsu.build_membership(prices, shares, **_BAND, seg_gap_days=365)

    after = [t for t in _REBALANCES if t >= _SECOND_STARTS]
    own_bars = {t: int((_SECOND <= t).sum()) for t in after}
    assert own_bars[after[0]] == 22, "the fixture's second listing should be one month old here"
    seasoned = {t for t in after if own_bars[t] >= 250}
    assert seasoned and seasoned != set(after), "need rebalances on both sides of the 250-bar line"

    # sealed behaviour: the first company's bars count, so the newcomer is in the band at once
    assert set(after) <= _member_dates(off, _REUSED)
    # corrected: in the band exactly when the SECOND listing has 250 bars of its own
    assert _member_dates(on, _REUSED) & set(after) == seasoned
    assert min(seasoned) == min(t for t in _REBALANCES if t >= _SECOND[249])

    # segmenting is not a recency check: during the silence the code keeps its slot either way
    silence = {t for t in _REBALANCES if _FIRST_ENDS < t < _SECOND_STARTS}
    assert silence and silence <= _member_dates(on, _REUSED)
    # and before the gap, where there is one segment, nothing changes
    cut = on["rebalance_date"] <= _FIRST_ENDS
    _assert_bitwise_equal(on[cut].reset_index(drop=True),
                          off[off["rebalance_date"] <= _FIRST_ENDS].reset_index(drop=True))


@pytest.mark.parametrize(("seg_gap_days", "carries_history"), [
    (365, True),       # a silence of exactly 365 days is NOT longer than 365 — same segment
    (364, False),      # ...but it is longer than 364 — new segment, one bar of history
])
def test_a_segment_starts_only_after_a_silence_strictly_longer_than_the_gap(seg_gap_days,
                                                                            carries_history):
    relisted = pd.Timestamp("2020-02-28")                  # 365 days after the first listing's last bar
    assert (relisted - _FIRST_ENDS).days == 365 and relisted in _REBALANCES
    prices, shares = _universe(_bars(_REUSED, _CAL[_CAL <= _FIRST_ENDS], close=20.0),
                               _bars(_REUSED, _CAL[_CAL >= relisted], close=25.0))
    on = tsu.build_membership(prices, shares, **_BAND, seg_gap_days=seg_gap_days)
    assert (relisted in _member_dates(on, _REUSED)) is carries_history


def test_zero_gap_days_is_a_bound_not_off():
    """`seg_gap_days=0` is degenerate — every bar is its own segment, so nothing has history — but it
    is a value. Reading it as falsy would quietly return the uncorrected universe."""
    prices, shares = _universe(*_reused())
    assert len(tsu.build_membership(prices, shares, **_BAND)) > 0
    assert len(tsu.build_membership(prices, shares, **_BAND, seg_gap_days=0)) == 0


def test_a_reused_code_earns_its_own_adv():
    """The ADV window must not reach back across the gap into the first company's turnover."""
    first_rebalance = pd.Timestamp("2020-06-30")           # the second listing has 22 bars
    kw = {**_BAND, "min_history": 20}                      # take the history floor out of the way

    # (a) a liquid newcomer: admitted either way, but the ADV it is admitted ON differs
    prices, shares = _universe(*_reused(second_volume=1.0e6))
    own = prices[(prices["ticker"] == _REUSED) & (prices["date"] >= _SECOND_STARTS)
                 & (prices["date"] <= first_rebalance)]
    assert len(own) == 22
    own_adv = float((own["close"] * own["volume"]).mean())
    row = {}
    for label, seg in (("off", None), ("on", 365)):
        mem = tsu.build_membership(prices, shares, **kw, seg_gap_days=seg)
        row[label] = mem[(mem["stock_id"] == _REUSED) & (mem["rebalance_date"] == first_rebalance)]
        assert len(row[label]) == 1
    assert row["on"]["adv_twd"].iloc[0] == pytest.approx(own_adv, rel=1e-12)
    assert row["off"]["adv_twd"].iloc[0] > 3.0 * own_adv, (
        "the sealed 60-bar window should still be averaging in the first company's turnover")

    # (b) a newcomer too thin for the liquidity floor passes it on the first company's turnover
    prices, shares = _universe(*_reused(second_volume=1.0e5))      # ~2.5e6 a day against a 5e6 floor
    off = tsu.build_membership(prices, shares, **kw)
    on = tsu.build_membership(prices, shares, **kw, seg_gap_days=365)
    assert first_rebalance in _member_dates(off, _REUSED)
    after = {t for t in _REBALANCES if t >= _SECOND_STARTS}
    assert not (_member_dates(on, _REUSED) & after), "admitted on turnover that was never its own"


# --------------------------------------------------------------------------- #
# 4. The two switches together
# --------------------------------------------------------------------------- #
def test_stray_prints_readmit_a_stopped_name_unless_both_switches_are_on():
    """A few prints dated long after the last real bar make a stopped name look recent again.

    The recency check alone therefore lets it back in for that month; only the per-segment history
    floor (the prints are a segment of six bars) keeps it out. This is why the two are documented
    as a pair.
    """
    prices, shares = _universe(*_stray())
    while_trading = {t for t in _REBALANCES
                     if t <= _STRAY_LAST_REAL and int((_CAL <= t).sum()) >= 250}
    assert len(while_trading) == 3
    print_month = pd.Timestamp("2021-06-30")
    assert (print_month - _STRAY_PRINTS[-1]).days <= 10

    recency_only = tsu.build_membership(prices, shares, **_BAND, max_stale_days=10)
    assert _member_dates(recency_only, _STRAY) == while_trading | {print_month}

    both = tsu.build_membership(prices, shares, **_BAND, max_stale_days=10, seg_gap_days=365)
    assert _member_dates(both, _STRAY) == while_trading


def test_stray_prints_inside_the_segment_window_get_past_both_switches():
    """The limit of the pair, pinned so it is a known fact rather than a surprise.

    Prints 276 days after the last real bar do not open a new segment under a 365-day rule, so the
    name carries its old history AND looks recent: it is back in the band for the print month.
    Nothing in the membership builder can tell these from real trades — a tighter gap does (at the
    price of resetting genuinely suspended names sooner), or the rows are dropped at the source.
    """
    prints = _CAL[(_CAL >= "2019-12-16") & (_CAL <= "2019-12-23")]
    assert len(prints) == 6 and 30 < (prints[0] - _STRAY_LAST_REAL).days < 365
    prices, shares = _universe(*_stray(prints))
    print_month = pd.Timestamp("2019-12-31")

    both = tsu.build_membership(prices, shares, **_BAND, max_stale_days=10, seg_gap_days=365)
    assert print_month in _member_dates(both, _STRAY), "expected the documented gap in the pair"
    assert pd.Timestamp("2019-11-29") not in _member_dates(both, _STRAY)
    assert pd.Timestamp("2020-01-31") not in _member_dates(both, _STRAY)

    tighter = tsu.build_membership(prices, shares, **_BAND, max_stale_days=10, seg_gap_days=180)
    assert print_month not in _member_dates(tighter, _STRAY)


# --------------------------------------------------------------------------- #
# 5. Causality (LEAK-2) — neither switch may read past the rebalance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("switches", [
    {}, {"max_stale_days": 10}, {"seg_gap_days": 365}, {"max_stale_days": 10, "seg_gap_days": 365},
])
@pytest.mark.parametrize("cut", ["2019-09-30", "2020-09-30"])    # inside the gap / after the relisting
def test_membership_is_unchanged_when_the_future_is_deleted(switches, cut):
    """Rebuild as of a past rebalance: every row at or before it must be byte-identical.

    A staleness or segment rule that looked FORWARD (the gap to the next bar rather than the
    previous one, say) would move rows here.
    """
    cut = pd.Timestamp(cut)
    assert cut in _REBALANCES
    prices, shares = _universe(_ghost(), *_reused(), *_stray())
    full = tsu.build_membership(prices, shares, **_BAND, **switches)
    past = tsu.build_membership(prices[prices["date"] <= cut], shares[shares["avail_date"] <= cut],
                                **_BAND, **switches)
    assert len(past) > 0
    _assert_bitwise_equal(past, full[full["rebalance_date"] <= cut].reset_index(drop=True))


# --------------------------------------------------------------------------- #
# 6. Arguments and CLI
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [{"max_stale_days": -1}, {"seg_gap_days": -1}])
def test_negative_day_counts_are_rejected(bad):
    prices, shares = _universe()
    with pytest.raises(ValueError, match="must be >= 0 or None"):
        tsu.build_membership(prices, shares, **_BAND, **bad)


def _write_dataset(d: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices, shares = _universe(_ghost(), *_reused(), *_stray())
    prices.to_parquet(d / "prices.parquet", index=False)
    shares.to_parquet(d / "shareholding.parquet", index=False)
    return prices, shares


def _run_cli(monkeypatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["taiwan_smallcap_universe.py", "--lo-rank", "2",
                                      "--hi-rank", "5", *argv])
    return tsu.main()


def test_cli_default_writes_the_sealed_membership(tmp_path, monkeypatch):
    prices, shares = _write_dataset(tmp_path)
    assert _run_cli(monkeypatch, "--data", str(tmp_path)) == 0
    got = pd.read_parquet(tmp_path / "universe" / "membership.parquet")
    _assert_bitwise_equal(got, _sealed_build_membership(prices, shares, **_BAND))


def test_cli_flags_reach_the_builder(tmp_path, monkeypatch):
    prices, shares = _write_dataset(tmp_path)
    out = tmp_path / "corrected"
    assert _run_cli(monkeypatch, "--data", str(tmp_path), "--out", str(out),
                    "--max-stale-days", "10", "--seg-gap-days", "365") == 0
    got = pd.read_parquet(out / "membership.parquet")
    want = tsu.build_membership(prices, shares, **_BAND, max_stale_days=10, seg_gap_days=365)
    _assert_bitwise_equal(got, want)
    assert _member_dates(got, _GHOST) != _member_dates(
        _sealed_build_membership(prices, shares, **_BAND), _GHOST), "the flags did nothing"


@pytest.mark.parametrize("flags", [("--max-stale-days", "10"), ("--seg-gap-days", "365"),
                                   ("--max-stale-days", "0")])
def test_cli_will_not_write_a_corrected_universe_over_the_default_location(tmp_path, monkeypatch,
                                                                           flags):
    """Without --out the script writes to <data>/universe — where the pre-registered membership
    lives. A corrected universe must be sent somewhere on purpose."""
    _write_dataset(tmp_path)
    assert _run_cli(monkeypatch, "--data", str(tmp_path), *flags) == 2
    assert not (tmp_path / "universe").exists(), "the refusal still created or wrote the directory"
    # an empty --out falls back to the default location too, so it must be refused the same way
    assert _run_cli(monkeypatch, "--data", str(tmp_path), "--out", "", *flags) == 2
    assert not (tmp_path / "universe").exists()
