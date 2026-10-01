"""The `taiwan_smallcap` substrate builder — universe, causality and channel contracts.

Built on a SYNTHETIC fixture rather than the real parquets: `data/` is gitignored, so a test that
needed the 111 MB price file would silently skip on every fresh clone — which is exactly how a
causality tripwire stops being a tripwire. The fixture is small enough to reason about by hand and
carries the one property each test cares about.

The load-bearing risks pinned here:
  * an alt-data value must never enter a bar EARLIER than its `avail_date` (LEAK-2). Every channel
    routes through `asof_grid`, so one leak here leaks all six.
  * the margin and short legs of `margin_short.parquet` differ ONLY in which column is divided.
    They were written twice, in two probe scripts, and a divergence would have been invisible —
    both produce well-formed numbers either way. `balance_util` is now the single implementation
    and this pins that the two legs agree when the data is swapped under them.
  * a name must not be active before the first monthly rebalance admits it.
  * the dividend add-back must be FORWARD-only — a back-adjusted feed would rewrite past bars with
    a future factor, the classic total-return look-ahead.
  * the opt-in age cap (`max_age_days`) is OFF by default, and the default must not move: the
    uncapped join is compared byte for byte against a frozen copy of the pre-cap `asof_grid`.
  * the opt-in tie-break (`stable_ties`) is OFF by default too. Switched on, two prints of one
    name that share an avail date resolve to the later one, every time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sharpen.crucible.data import taiwan_smallcap_panel as tsp

_TICKERS = ("1101", "2330", "6505")
_N_DAYS = 260


@pytest.fixture
def data_dir(tmp_path):
    """A 3-name, ~1-year synthetic substrate on disk, in the real parquet layout."""
    dates = pd.bdate_range("2020-01-01", periods=_N_DAYS)
    rng = np.random.default_rng(11)
    rows = []
    for i, tk in enumerate(_TICKERS):
        px = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, _N_DAYS)))
        rows.append(pd.DataFrame({
            "date": dates, "ticker": tk, "open": px, "high": px * 1.01,
            "low": px * 0.99, "close": px, "volume": 1e6 * (i + 1),
        }))
    prices = pd.concat(rows, ignore_index=True)
    prices.to_parquet(tmp_path / "prices.parquet")

    # monthly membership: every name from the first rebalance, which is 30 bars in
    (tmp_path / "universe").mkdir()
    rebals = dates[30::21]
    mem = pd.DataFrame([{"rebalance_date": d, "stock_id": tk} for d in rebals for tk in _TICKERS])
    mem.to_parquet(tmp_path / "universe" / "membership.parquet")

    pd.DataFrame({"stock_id": list(_TICKERS),
                  "sector": ["cement", "semis", "biotech"]}).to_parquet(
        tmp_path / "pool.frozen.parquet")

    # month revenue: 24 monthly prints per name so a YoY is defined for the last 12
    mr = []
    for tk in _TICKERS:
        for k in range(24):
            y, m = 2019 + k // 12, k % 12 + 1
            mr.append({"stock_id": tk, "revenue_year": y, "revenue_month": m,
                       "revenue": 1000.0 * (1 + 0.01 * k),
                       "avail_date": pd.Timestamp(y, m, 10) + pd.DateOffset(months=1)})
    pd.DataFrame(mr).to_parquet(tmp_path / "month_revenue.parquet")

    # daily margin + short balances, and a share count that predates them
    ms = []
    for tk in _TICKERS:
        for d in dates:
            ms.append({"stock_id": tk, "date": d, "margin_balance": 5e5, "short_balance": 2e5})
    pd.DataFrame(ms).to_parquet(tmp_path / "margin_short.parquet")
    pd.DataFrame([{"stock_id": tk, "avail_date": dates[0], "total_shares": 1e8,
                   "big_holder_pct": 40.0 + j} for j, tk in enumerate(_TICKERS)]).to_parquet(
        tmp_path / "shareholding.parquet")

    inst = []
    for tk in _TICKERS:
        for d in dates:
            inst.append({"stock_id": tk, "date": d, "avail_date": d + pd.tseries.offsets.BDay(1),
                         "foreign_net": 1000.0, "trust_net": -500.0})
    pd.DataFrame(inst).to_parquet(tmp_path / "institutional.parquet")
    return tmp_path


def test_panel_shape_and_channels(data_dir):
    p = tsp.build_taiwan_smallcap_panel(data_dir, channels=tsp.ALL_CHANNELS)
    assert p.T == _N_DAYS and p.N == len(_TICKERS)
    assert set(p.feature_slots) == set(tsp.ALL_CHANNELS)
    for name, slot in p.feature_slots.items():
        assert slot.shape == (p.T, p.N), f"{name} is not a per-name (T,N) matrix"
    assert p.meta["panel"] == "taiwan_smallcap"
    # the honest downgrade must survive into meta — the scorecard prints it as an UPPER BOUND
    assert p.meta["survivorship_free"] is False


def test_default_channels_are_the_locked_three(data_dir):
    """The probe scripts delegate here; adding channels by default would change sealed scorecards."""
    p = tsp.build_taiwan_smallcap_panel(data_dir)
    assert tuple(p.feature_slots) == tsp.CORE_CHANNELS


def test_unknown_channel_is_rejected(data_dir):
    with pytest.raises(ValueError, match="unknown taiwan_smallcap channel"):
        tsp.build_taiwan_smallcap_panel(data_dir, channels=("mrev_yoy", "insider_buys"))


def test_missing_prices_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="prices.parquet"):
        tsp.build_taiwan_smallcap_panel(tmp_path)


# --------------------------------------------------------------------------- #
# LEAK-2
# --------------------------------------------------------------------------- #
def test_asof_grid_never_reads_a_future_avail_date():
    dates = np.array(pd.bdate_range("2020-01-01", periods=10), dtype="datetime64[ns]")
    ev = pd.DataFrame({"stock_id": ["A", "A"],
                       "avail_date": [pd.Timestamp("2020-01-08"), pd.Timestamp("2020-01-13")],
                       "v": [1.0, 2.0]})
    grid = tsp.asof_grid(ev, dates, ("A",), "v")
    assert np.isnan(grid[:5, 0]).all(), "a value appeared before its avail_date"
    assert (grid[5:8, 0] == 1.0).all()
    assert (grid[8:, 0] == 2.0).all(), "the second print never arrives"


def _truncate_source(src: Path, dst: Path, cut: pd.Timestamp) -> None:
    """Copy the on-disk substrate to `dst`, dropping every observation dated after `cut`.

    Truncation is applied to whichever date column each file carries, INCLUDING `avail_date` —
    the point is to delete the future as the builder would have seen it on `cut`.
    """
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "universe").mkdir(exist_ok=True)
    for rel in ("prices.parquet", "month_revenue.parquet", "margin_short.parquet",
                "shareholding.parquet", "institutional.parquet", "pool.frozen.parquet",
                "universe/membership.parquet"):
        p = src / rel
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        for col in ("date", "avail_date", "rebalance_date", "ex_date"):
            if col in df.columns:
                df = df[pd.to_datetime(df[col]) <= cut]
        df.to_parquet(dst / rel)


def test_rebuilding_on_truncated_source_reproduces_every_past_cell(data_dir, tmp_path):
    """The REAL causality tripwire for the builder: rebuild it as of a past date and every cell
    at or before that date must be bit-identical.

    NOTE this deliberately does NOT use `Panel.truncated()`. That method only SLICES the arrays of
    an already-built panel, so comparing `built.truncated(k)[:k]` against `built[:k]` is true for
    any array whatsoever and proves nothing about the builder — a vacuous green. Deleting the
    future from the SOURCE and rebuilding is what actually catches a builder that peeks forward
    (a `merge_asof(direction="forward"/"nearest")`, a bfill, a full-series normalization, or a
    dividend factor applied backwards).
    """
    full = tsp.build_taiwan_smallcap_panel(data_dir, channels=tsp.ALL_CHANNELS)
    cut_ix = int(full.T * 0.7)
    cut = pd.Timestamp(full.dates[cut_ix])

    trunc_dir = tmp_path / "as_of"
    _truncate_source(data_dir, trunc_dir, cut)
    past = tsp.build_taiwan_smallcap_panel(trunc_dir, channels=tsp.ALL_CHANNELS)

    assert past.T == cut_ix + 1, "the as-of rebuild did not end at the cut date"
    assert past.tickers == full.tickers, "the universe changed — the comparison would be misaligned"
    np.testing.assert_allclose(past.close, full.close[:past.T], equal_nan=True,
                               err_msg="a past CLOSE moved when the future was deleted")
    np.testing.assert_array_equal(past.active, full.active[:past.T],
                                  err_msg="a past ACTIVE cell moved when the future was deleted")
    for name in tsp.ALL_CHANNELS:
        np.testing.assert_allclose(
            np.asarray(past.feature_slots[name]),
            np.asarray(full.feature_slots[name])[:past.T], equal_nan=True,
            err_msg=f"slot {name} moved at t<=cut when the future was deleted — look-ahead")


def test_the_asof_tripwire_has_teeth(data_dir):
    """Mutation check: the alignment above must FAIL if the as-of join is made forward-looking.

    Without this, `test_rebuilding_on_truncated_source_reproduces_every_past_cell` could be green
    because the channel is empty, constant, or never actually joined — CAUS-05.
    """
    dates = np.array(pd.bdate_range("2020-01-01", periods=10), dtype="datetime64[ns]")
    ev = pd.DataFrame({"stock_id": ["A", "A"],
                       "avail_date": [pd.Timestamp("2020-01-08"), pd.Timestamp("2020-01-13")],
                       "v": [1.0, 2.0]})
    causal = tsp.asof_grid(ev, dates, ("A",), "v")

    # the same join, forward-looking — what a one-word edit to `direction=` would produce
    d = pd.DataFrame({"avail_date": pd.to_datetime(dates)})
    g = ev[["avail_date", "v"]].sort_values("avail_date")
    leaky = pd.merge_asof(d, g, on="avail_date", direction="forward")["v"].to_numpy()

    assert not np.allclose(causal[:, 0], leaky, equal_nan=True), (
        "a forward-looking join is INDISTINGUISHABLE from the causal one on this fixture — the "
        "causality tests above would pass with the leak reintroduced")
    assert np.isnan(causal[0, 0]) and leaky[0] == 1.0, "the leak should surface a value at bar 0"


def test_flow_events_value_lands_strictly_after_its_window(data_dir):
    """A T86 flow summed over bars <= t may only enter the panel on a session AFTER t."""
    p = tsp.build_taiwan_smallcap_panel(data_dir, channels=("foreign_flow",))
    slot = p.feature_slots["foreign_flow"]
    # FLOW_WINDOW bars are needed before the first sum exists, and it publishes T+1 => the first
    # finite row must be strictly later than bar FLOW_WINDOW-1.
    first = int(np.argmax(np.isfinite(slot[:, 0]))) if np.isfinite(slot[:, 0]).any() else -1
    assert first >= tsp.FLOW_WINDOW, f"a {tsp.FLOW_WINDOW}-bar sum surfaced at bar {first}"


def test_membership_is_as_of(data_dir):
    p = tsp.build_taiwan_smallcap_panel(data_dir)
    members = pd.read_parquet(data_dir / "universe" / "membership.parquet")
    first = pd.to_datetime(members["rebalance_date"]).min()
    before = pd.DatetimeIndex(p.dates) < first
    assert p.active[before].sum() == 0, "a name is active before the first rebalance admits it"
    assert p.active[~before].any(), "no name is ever active — the fixture would prove nothing"


def test_dividend_add_back_is_forward_only():
    """Bars BEFORE an ex-date must be byte-identical; only later bars may be scaled up."""
    idx = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=10))
    close = pd.DataFrame({"A": np.full(10, 100.0)}, index=idx)
    div = pd.DataFrame({"stock_id": ["A"], "ex_date": [idx[5]], "amount": [5.0]})
    cf = tsp.causal_total_return_factor(close, div)
    assert (cf.to_numpy()[:5] == 1.0).all(), "a past bar was rewritten by a later dividend"
    assert (cf.to_numpy()[5:] > 1.0).all(), "the add-back never applied"
    # and an empty dividend table is the identity
    cf0 = tsp.causal_total_return_factor(close, div.iloc[0:0])
    assert (cf0.to_numpy() == 1.0).all()


# --------------------------------------------------------------------------- #
# One implementation for the two balance legs
# --------------------------------------------------------------------------- #
def test_balance_util_legs_agree_when_the_columns_are_swapped(data_dir):
    """margin_util and short_util must differ ONLY by which column is divided."""
    margin = pd.read_parquet(data_dir / "margin_short.parquet")
    shares = pd.read_parquet(data_dir / "shareholding.parquet")
    a = tsp.balance_util(margin, shares, balance_col="margin_balance", out_col="margin_util")
    swapped = margin.rename(columns={"margin_balance": "short_balance",
                                     "short_balance": "margin_balance"})
    b = tsp.balance_util(swapped, shares, balance_col="short_balance", out_col="short_util")
    np.testing.assert_allclose(a["margin_util"].to_numpy(), b["short_util"].to_numpy())


def test_balance_util_is_empty_without_share_counts(data_dir):
    margin = pd.read_parquet(data_dir / "margin_short.parquet")
    out = tsp.balance_util(margin, pd.DataFrame(), balance_col="margin_balance",
                           out_col="margin_util")
    assert out.empty, "utilization was computed with no share count to normalize by"


# --------------------------------------------------------------------------- #
# The sector map is a neutralization control, and it moved a sealed result once (P1-REPRO-01)
# --------------------------------------------------------------------------- #
def test_sector_map_prefers_the_frozen_pool(data_dir):
    live = pd.DataFrame({"stock_id": list(_TICKERS), "sector": ["X", "Y", "Z"]})
    live.to_parquet(data_dir / "pool.parquet")
    _ids, prov = tsp.sector_map(data_dir, _TICKERS)
    assert prov["source"] == "pool.frozen.parquet" and prov["frozen"] is True


def test_sector_map_sha_moves_only_with_this_panels_partition(data_dir):
    _ids, a = tsp.sector_map(data_dir, _TICKERS)
    # an unrelated listing joining the pool must NOT restate this panel
    pool = pd.read_parquet(data_dir / "pool.frozen.parquet")
    pd.concat([pool, pd.DataFrame([{"stock_id": "9999", "sector": "other"}])],
              ignore_index=True).to_parquet(data_dir / "pool.frozen.parquet")
    _ids, b = tsp.sector_map(data_dir, _TICKERS)
    assert a["sector_map_sha"] == b["sector_map_sha"]
    # but a genuine re-classification of one of OUR names must trip it
    pool.loc[pool["stock_id"] == "2330", "sector"] = "reclassified"
    pool.to_parquet(data_dir / "pool.frozen.parquet")
    _ids, c = tsp.sector_map(data_dir, _TICKERS)
    assert c["sector_map_sha"] != a["sector_map_sha"]


# --------------------------------------------------------------------------- #
# Opt-in age cap (`max_age_days`) — off by default, and the default must not move
# --------------------------------------------------------------------------- #
def _sealed_asof_grid(events, dates, tickers, value_col, avail_col="avail_date",
                      id_col="stock_id", *, max_age_days=None, stable_ties=False):
    """SEALED REFERENCE — `asof_grid` exactly as it stood before the age cap existed. DO NOT EDIT.

    `max_age_days` and `stable_ties` are accepted only so this can stand in for `tsp.asof_grid`
    inside the builder; the default build must never hand a cap or the stable tie-break to any
    channel, which the asserts enforce. The body below them is the pre-switch function, untouched.
    """
    assert max_age_days is None, "the default build passed an age cap to a channel"
    assert stable_ties is False, "the default build asked a channel for the stable tie-break"
    T, N = len(dates), len(tickers)
    grid = np.full((T, N), np.nan, dtype=np.float64)
    if events is None or events.empty or value_col not in events.columns:
        return grid
    d = pd.DataFrame({avail_col: pd.to_datetime(dates)}).sort_values(avail_col).reset_index(drop=True)
    ev = events.dropna(subset=[avail_col, value_col]).copy()
    ev[avail_col] = pd.to_datetime(ev[avail_col])
    ev[id_col] = ev[id_col].astype(str)
    col = {tk: j for j, tk in enumerate(tickers)}
    for tk, g in ev.groupby(id_col):
        j = col.get(str(tk))
        if j is None:
            continue
        g = g[[avail_col, value_col]].sort_values(avail_col)
        # collapse duplicate avail dates to the LAST print that day (keep it public-consistent)
        g = g.groupby(avail_col, as_index=False).last()
        merged = pd.merge_asof(d, g, on=avail_col, direction="backward")
        grid[:, j] = merged[value_col].to_numpy(dtype=np.float64)
    return grid


_GRID_TICKERS = ("A", "B", "C")


def _irregular_events() -> pd.DataFrame:
    """Prints 1-119 days apart, plus a duplicate avail date, a NaN value and an off-grid ticker."""
    rng = np.random.default_rng(5)
    rows = []
    for tk in (*_GRID_TICKERS, "Z"):                      # Z is not on the grid
        t = pd.Timestamp("2020-01-01")
        for _ in range(12):
            t = t + pd.Timedelta(days=int(rng.integers(1, 120)))
            rows.append({"stock_id": tk, "avail_date": t, "v": float(rng.normal())})
    rows.append({"stock_id": "A", "avail_date": rows[3]["avail_date"], "v": 9.0})
    rows.append({"stock_id": "B", "avail_date": pd.Timestamp("2020-06-01"), "v": np.nan})
    return pd.DataFrame(rows)


_GRID_DATES = np.array(pd.bdate_range("2020-01-01", periods=700), dtype="datetime64[ns]")


def test_asof_grid_default_is_the_sealed_join_byte_for_byte():
    ev = _irregular_events()
    want = _sealed_asof_grid(ev, _GRID_DATES, _GRID_TICKERS, "v")
    assert np.isfinite(want).any() and np.isnan(want).any()
    assert tsp.asof_grid(ev, _GRID_DATES, _GRID_TICKERS, "v").tobytes() == want.tobytes()
    assert tsp.asof_grid(ev, _GRID_DATES, _GRID_TICKERS, "v",
                         max_age_days=None).tobytes() == want.tobytes()


def test_a_stale_value_becomes_nan_after_the_cap():
    dates = np.array(pd.date_range("2020-01-01", periods=20, freq="D"), dtype="datetime64[ns]")
    ev = pd.DataFrame({"stock_id": ["A", "A"],
                       "avail_date": [pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-15")],
                       "v": [1.0, 2.0]})
    capped = tsp.asof_grid(ev, dates, ("A",), "v", max_age_days=5)[:, 0]
    assert np.isnan(capped[:2]).all()
    assert (capped[2:8] == 1.0).all(), "ages 0..5 must be carried — the cap is inclusive"
    assert np.isnan(capped[8:14]).all(), "a print older than the cap was still carried"
    assert (capped[14:] == 2.0).all(), "a newer print did not bring the cell back"
    # the sealed behaviour carries the first print across that whole stretch — this is what the
    # cap removes, and why a cap that silently did nothing would be caught here
    assert (tsp.asof_grid(ev, dates, ("A",), "v")[8:14, 0] == 1.0).all()


def test_the_age_cap_counts_calendar_days_not_bars():
    dates = np.array(pd.bdate_range("2020-01-06", periods=10), dtype="datetime64[ns]")   # a Monday
    ev = pd.DataFrame({"stock_id": ["A"], "avail_date": [pd.Timestamp("2020-01-10")], "v": [1.0]})
    two = tsp.asof_grid(ev, dates, ("A",), "v", max_age_days=2)[:, 0]
    # Friday the 10th is bar 4. Monday the 13th is ONE bar later but three calendar days older.
    assert two[4] == 1.0 and np.isnan(two[5:]).all()
    assert tsp.asof_grid(ev, dates, ("A",), "v", max_age_days=3)[5, 0] == 1.0


@pytest.mark.parametrize("cap", [0, 1, 30, 75, 400])
def test_age_cap_agrees_with_an_explicit_age_computation(cap):
    """The cap rides on `merge_asof(tolerance=...)`. Re-derive it WITHOUT that argument: carry the
    avail date of the print in force through the uncapped join, measure its age at every bar, and
    blank what is older. Pins the inclusive boundary against a change in pandas' semantics."""
    ev = _irregular_events()
    valid = ev.dropna(subset=["v"])                       # the print in force is the last VALID one
    epoch = pd.Timestamp("1970-01-01")
    stamped = valid.assign(day=(valid["avail_date"] - epoch) / pd.Timedelta(days=1))
    in_force = tsp.asof_grid(stamped, _GRID_DATES, _GRID_TICKERS, "day")
    today = ((pd.DatetimeIndex(_GRID_DATES) - epoch) / pd.Timedelta(days=1)).to_numpy()[:, None]
    uncapped = tsp.asof_grid(ev, _GRID_DATES, _GRID_TICKERS, "v")
    want = np.where(today - in_force > cap, np.nan, uncapped)

    got = tsp.asof_grid(ev, _GRID_DATES, _GRID_TICKERS, "v", max_age_days=cap)
    np.testing.assert_array_equal(got, want)
    blanked = int((np.isfinite(uncapped) & np.isnan(got)).sum())
    if cap <= 75:
        assert blanked > 0 and np.isfinite(got).any(), "the cap never bound — nothing was compared"
    else:
        assert blanked == 0 and got.tobytes() == uncapped.tobytes(), "a non-binding cap changed a cell"


def test_asof_grid_rejects_a_negative_cap():
    with pytest.raises(ValueError, match="max_age_days must be >= 0"):
        tsp.asof_grid(_irregular_events(), _GRID_DATES, _GRID_TICKERS, "v", max_age_days=-1)


def _stop_feed(data_dir: Path, ticker: str, after: str) -> None:
    """Delete `ticker`'s month-revenue prints available after `after`: a vendor feed that stops
    while the name itself keeps trading."""
    mr = pd.read_parquet(data_dir / "month_revenue.parquet")
    gone = (mr["stock_id"] == ticker) & (pd.to_datetime(mr["avail_date"]) > pd.Timestamp(after))
    assert gone.any()
    mr[~gone].to_parquet(data_dir / "month_revenue.parquet")


def _assert_same_panel(got, want) -> None:
    assert got.tickers == want.tickers
    for name in ("dates", "open", "high", "low", "close", "volume", "active", "adv_usd",
                 "sector_id"):
        assert getattr(got, name).tobytes() == getattr(want, name).tobytes(), f"{name} moved"
    assert list(got.feature_slots) == list(want.feature_slots)
    for name, slot in want.feature_slots.items():
        assert np.asarray(got.feature_slots[name]).tobytes() == np.asarray(slot).tobytes(), (
            f"slot {name} moved")
    assert got.meta == want.meta


def test_default_panel_is_the_sealed_panel(data_dir, monkeypatch):
    """No cap asked for => the panel the pre-cap builder gave, on a fixture where a cap WOULD bind."""
    _stop_feed(data_dir, "6505", "2020-04-30")
    got = tsp.build_taiwan_smallcap_panel(data_dir, channels=tsp.ALL_CHANNELS)
    assert "max_age_days" not in got.meta, "the default meta gained a key — sealed scorecards move"

    monkeypatch.setattr(tsp, "asof_grid", _sealed_asof_grid)
    want = tsp.build_taiwan_smallcap_panel(data_dir, channels=tsp.ALL_CHANNELS)
    _assert_same_panel(got, want)


@pytest.mark.parametrize("off", [None, {}, {"mrev_yoy": None}])
def test_an_empty_age_cap_is_the_default(data_dir, off):
    _stop_feed(data_dir, "6505", "2020-04-30")
    _assert_same_panel(tsp.build_taiwan_smallcap_panel(data_dir, max_age_days=off),
                       tsp.build_taiwan_smallcap_panel(data_dir))


def test_a_stopped_feed_drops_out_of_the_panel_after_the_cap(data_dir):
    """The last print for the stopped feed is due 2020-04-10 (a Friday), so it is usable from the
    next session, Monday 2020-04-13. With a 75-day cap it is carried through 2020-06-27 and NaN
    after; the uncapped panel carries that one value to the last bar."""
    _stop_feed(data_dir, "6505", "2020-04-30")
    sealed = tsp.build_taiwan_smallcap_panel(data_dir)
    capped = tsp.build_taiwan_smallcap_panel(data_dir, max_age_days={"mrev_yoy": 75})
    j = sealed.tickers.index("6505")
    d = pd.DatetimeIndex(sealed.dates)
    stale = np.asarray(d > pd.Timestamp("2020-04-13") + pd.Timedelta(days=75))
    assert stale.sum() > 100 and d[stale][0] == pd.Timestamp("2020-06-29")

    before, after = sealed.feature_slots["mrev_yoy"], capped.feature_slots["mrev_yoy"]
    assert np.isfinite(before[stale, j]).all() and len(set(before[stale, j])) == 1, (
        "the uncapped panel should be carrying one frozen value here")
    assert np.isnan(after[stale, j]).all(), "a value older than the cap survived"
    assert after[~stale, j].tobytes() == before[~stale, j].tobytes()
    # the two names still reporting every month are never 75 days stale — untouched
    others = [k for k in range(sealed.N) if k != j]
    assert after[:, others].tobytes() == before[:, others].tobytes()

    # ...and the cap is per channel: nothing else in the panel moves
    for name in ("margin_util", "holder_conc"):
        assert capped.feature_slots[name].tobytes() == sealed.feature_slots[name].tobytes()
    for name in ("close", "volume", "active", "adv_usd"):
        assert getattr(capped, name).tobytes() == getattr(sealed, name).tobytes()
    assert capped.meta["max_age_days"] == {"mrev_yoy": 75}, "a capped panel must say it is capped"
    assert {k: v for k, v in capped.meta.items() if k != "max_age_days"} == sealed.meta


def test_age_cap_is_per_channel(data_dir):
    """Capping one channel leaves the others exactly as they were."""
    sealed = tsp.build_taiwan_smallcap_panel(data_dir)
    capped = tsp.build_taiwan_smallcap_panel(data_dir, max_age_days={"holder_conc": 30})
    # the fixture's register is a single print on the first bar: uncapped it is carried all year
    assert np.isfinite(sealed.feature_slots["holder_conc"]).all()
    d = pd.DatetimeIndex(sealed.dates)
    old = np.asarray(d > d[0] + pd.Timedelta(days=30))
    assert np.isnan(capped.feature_slots["holder_conc"][old]).all()
    assert np.isfinite(capped.feature_slots["holder_conc"][~old]).all()
    for name in ("mrev_yoy", "margin_util"):
        assert capped.feature_slots[name].tobytes() == sealed.feature_slots[name].tobytes()


def test_age_cap_on_a_channel_that_is_not_built_is_rejected(data_dir):
    """A cap that names nothing would leave the panel uncapped while the caller thinks otherwise."""
    with pytest.raises(ValueError, match="not being built"):          # real channel, not requested
        tsp.build_taiwan_smallcap_panel(data_dir, max_age_days={"short_util": 30})
    with pytest.raises(ValueError, match="not being built"):          # misspelt
        tsp.build_taiwan_smallcap_panel(data_dir, max_age_days={"mrev": 75})


@pytest.mark.parametrize("bad", [-1, 7.5])
def test_a_malformed_age_cap_is_rejected(data_dir, bad):
    with pytest.raises(ValueError, match="whole number of days"):
        tsp.build_taiwan_smallcap_panel(data_dir, max_age_days={"mrev_yoy": bad})


@pytest.mark.parametrize("bare", [0, 75])
def test_a_bare_number_is_not_accepted_as_a_panel_age_cap(data_dir, bare):
    """`asof_grid` takes a number, the builder takes {channel: days}. A bare 0 is falsy, so without
    this check it would be read as "no cap" and build an uncapped panel without a word."""
    with pytest.raises(TypeError, match="must be a mapping"):
        tsp.build_taiwan_smallcap_panel(data_dir, max_age_days=bare)


def test_capped_rebuild_on_truncated_source_reproduces_every_past_cell(data_dir, tmp_path):
    """LEAK-2 for the capped path: the truncate-the-SOURCE-and-rebuild tripwire, with caps that bind."""
    caps = {"mrev_yoy": 20, "holder_conc": 30}
    full = tsp.build_taiwan_smallcap_panel(data_dir, channels=tsp.ALL_CHANNELS, max_age_days=caps)
    uncapped = tsp.build_taiwan_smallcap_panel(data_dir, channels=tsp.ALL_CHANNELS)
    for name in caps:
        assert (np.isfinite(uncapped.feature_slots[name])
                & np.isnan(full.feature_slots[name])).any(), f"the {name} cap never binds"

    cut_ix = int(full.T * 0.7)
    trunc_dir = tmp_path / "as_of"
    _truncate_source(data_dir, trunc_dir, pd.Timestamp(full.dates[cut_ix]))
    past = tsp.build_taiwan_smallcap_panel(trunc_dir, channels=tsp.ALL_CHANNELS, max_age_days=caps)

    assert past.T == cut_ix + 1 and past.tickers == full.tickers
    for name in tsp.ALL_CHANNELS:
        np.testing.assert_array_equal(
            np.asarray(past.feature_slots[name]), np.asarray(full.feature_slots[name])[:past.T],
            err_msg=f"capped slot {name} moved at t<=cut when the future was deleted — look-ahead")


# --------------------------------------------------------------------------- #
# Opt-in tie-break (`stable_ties`) — off by default, and the default must not move
# --------------------------------------------------------------------------- #
def _later_print_reference(events: pd.DataFrame, dates, tickers, value_col: str) -> np.ndarray:
    """The as-of grid re-derived with no sort at all: walk the rows in the order given, let a later
    row overwrite an earlier one with the same (ticker, avail date), then look each bar up."""
    grid = np.full((len(dates), len(tickers)), np.nan)
    for j, tk in enumerate(tickers):
        last: dict[pd.Timestamp, float] = {}
        for avail, v in events.loc[events["stock_id"] == tk, ["avail_date", value_col]].itertuples(
                index=False):
            if pd.notna(avail) and pd.notna(v):
                last[pd.Timestamp(avail)] = float(v)
        if not last:
            continue
        keys = np.array(sorted(last), dtype="datetime64[ns]")
        vals = np.array([last[pd.Timestamp(k)] for k in keys])
        ix = np.searchsorted(keys, dates, side="right") - 1
        grid[ix >= 0, j] = vals[ix[ix >= 0]]
    return grid


_TIE_WEEKS = 120
_TIE_DATES = np.array(pd.bdate_range("2018-01-01", periods=_TIE_WEEKS * 5), dtype="datetime64[ns]")


def _six_day_weeks() -> pd.DataFrame:
    """One name printing Monday to Saturday, each print public the next business day — so every
    Friday and Saturday print share a Monday avail date. Rows are in observation-date order and the
    value is the row number, so the later print of a tied pair is the larger value."""
    obs = pd.date_range("2018-01-01", periods=_TIE_WEEKS * 7, freq="D")
    obs = obs[obs.dayofweek < 6]
    return pd.DataFrame({"stock_id": "A", "obs": obs, "avail_date": obs + pd.tseries.offsets.BDay(1),
                         "v": np.arange(len(obs), dtype=np.float64)})


def test_two_prints_sharing_an_avail_date_resolve_to_the_later_one():
    ev = _six_day_weeks()
    tied = ev[ev.duplicated("avail_date", keep=False)]
    assert len(tied) == 2 * _TIE_WEEKS and set(tied["obs"].dt.dayofweek) == {4, 5}

    got = tsp.asof_grid(ev, _TIE_DATES, ("A",), "v", stable_ties=True)[:, 0]
    np.testing.assert_array_equal(got, _later_print_reference(ev, _TIE_DATES, ("A",), "v")[:, 0])

    # ...and spelled out for the tied bars: every Monday carries the SATURDAY print, never Friday's
    sat = ev[ev["obs"].dt.dayofweek == 5]
    fri = ev[ev["obs"].dt.dayofweek == 4]
    mondays = pd.DatetimeIndex(_TIE_DATES).get_indexer(pd.DatetimeIndex(sat["avail_date"]))
    assert (mondays >= 0).sum() == _TIE_WEEKS - 1, "the last tie falls after the grid, the rest on it"
    on = mondays >= 0
    assert (sat["v"].to_numpy() != fri["v"].to_numpy()).all(), "tied prints are indistinguishable"
    np.testing.assert_array_equal(got[mondays[on]], sat["v"].to_numpy()[on])
    # untied bars are whatever the default gives: the switch only decides ties
    untied = np.setdiff1d(np.arange(len(_TIE_DATES)), mondays[on])
    np.testing.assert_array_equal(
        got[untied], tsp.asof_grid(ev, _TIE_DATES, ("A",), "v")[untied, 0])


def test_stable_ties_follows_row_order_not_value_order():
    """The winner is the row that comes last in `events`, whatever its value — so a caller that
    hands rows in observation-date order gets the later-dated print."""
    ev = _six_day_weeks()
    flipped = ev.iloc[::-1].reset_index(drop=True)                    # Saturday now precedes Friday
    got = tsp.asof_grid(flipped, _TIE_DATES, ("A",), "v", stable_ties=True)[:, 0]
    np.testing.assert_array_equal(
        got, _later_print_reference(flipped, _TIE_DATES, ("A",), "v")[:, 0])
    fri = ev[ev["obs"].dt.dayofweek == 4]
    mondays = pd.DatetimeIndex(_TIE_DATES).get_indexer(pd.DatetimeIndex(fri["avail_date"]))
    on = mondays >= 0
    np.testing.assert_array_equal(got[mondays[on]], fri["v"].to_numpy()[on])


def test_asof_grid_default_is_the_sealed_join_on_tied_prints():
    """`stable_ties` off => the pre-switch function byte for byte, on a fixture made of ties."""
    ev = _six_day_weeks()
    want = _sealed_asof_grid(ev, _TIE_DATES, ("A",), "v")
    assert tsp.asof_grid(ev, _TIE_DATES, ("A",), "v").tobytes() == want.tobytes()
    assert tsp.asof_grid(ev, _TIE_DATES, ("A",), "v", stable_ties=False).tobytes() == want.tobytes()


def test_stable_ties_and_the_age_cap_compose():
    ev = _six_day_weeks()
    stopped = ev[ev["obs"] < "2019-01-01"]                           # the feed stops; the grid runs on
    got = tsp.asof_grid(stopped, _TIE_DATES, ("A",), "v", stable_ties=True, max_age_days=10)[:, 0]
    ref = _later_print_reference(stopped, _TIE_DATES, ("A",), "v")[:, 0]
    d = pd.DatetimeIndex(_TIE_DATES)
    old = np.asarray(d > stopped["avail_date"].max() + pd.Timedelta(days=10))
    assert old.sum() > 50 and np.isnan(got[old]).all()
    np.testing.assert_array_equal(got[~old], ref[~old])


def _add_saturday_sessions(data_dir: Path) -> None:
    """Give the balance and flow files a make-up Saturday after every Friday, with values unlike
    any weekday's. The price calendar stays Monday-Friday, so the tie shows on the Monday bar."""
    ms = pd.read_parquet(data_dir / "margin_short.parquet")
    sat = ms[pd.to_datetime(ms["date"]).dt.dayofweek == 4].copy()
    sat["date"] = pd.to_datetime(sat["date"]) + pd.Timedelta(days=1)
    k = np.arange(len(sat), dtype=np.float64)
    sat["margin_balance"], sat["short_balance"] = 9e5 + k, 7e5 + k
    pd.concat([ms, sat], ignore_index=True).to_parquet(data_dir / "margin_short.parquet")

    inst = pd.read_parquet(data_dir / "institutional.parquet")
    sat = inst[pd.to_datetime(inst["date"]).dt.dayofweek == 4].copy()
    sat["date"] = pd.to_datetime(sat["date"]) + pd.Timedelta(days=1)
    sat["avail_date"] = sat["date"] + pd.tseries.offsets.BDay(1)
    k = np.arange(len(sat), dtype=np.float64)
    sat["foreign_net"], sat["trust_net"] = 5000.0 + k, -3000.0 - k
    pd.concat([inst, sat], ignore_index=True).to_parquet(data_dir / "institutional.parquet")


def _channel_events(data_dir: Path) -> dict[str, pd.DataFrame]:
    """The event streams of the four channels whose prints can share an avail date."""
    margin = pd.read_parquet(data_dir / "margin_short.parquet")
    shares = pd.read_parquet(data_dir / "shareholding.parquet")
    inst = pd.read_parquet(data_dir / "institutional.parquet")
    return {
        "margin_util": tsp.balance_util(margin, shares, balance_col="margin_balance",
                                        out_col="margin_util"),
        "short_util": tsp.balance_util(margin, shares, balance_col="short_balance",
                                       out_col="short_util"),
        "foreign_flow": tsp.flow_events(inst, "foreign_net", "foreign_flow"),
        "trust_flow": tsp.flow_events(inst, "trust_net", "trust_flow"),
    }


def test_default_panel_is_the_sealed_panel_when_prints_tie(data_dir, monkeypatch):
    _add_saturday_sessions(data_dir)
    got = tsp.build_taiwan_smallcap_panel(data_dir, channels=tsp.ALL_CHANNELS)
    assert "stable_ties" not in got.meta, "the default meta gained a key — sealed scorecards move"
    off = tsp.build_taiwan_smallcap_panel(data_dir, channels=tsp.ALL_CHANNELS, stable_ties=False)

    monkeypatch.setattr(tsp, "asof_grid", _sealed_asof_grid)
    want = tsp.build_taiwan_smallcap_panel(data_dir, channels=tsp.ALL_CHANNELS)
    _assert_same_panel(got, want)
    _assert_same_panel(off, want)


def test_stable_ties_panel_carries_the_saturday_print_on_monday(data_dir):
    _add_saturday_sessions(data_dir)
    sealed = tsp.build_taiwan_smallcap_panel(data_dir, channels=tsp.ALL_CHANNELS)
    stable = tsp.build_taiwan_smallcap_panel(data_dir, channels=tsp.ALL_CHANNELS, stable_ties=True)
    d = pd.DatetimeIndex(stable.dates)
    monday = np.asarray(d.dayofweek == 0)

    for name, ev in _channel_events(data_dir).items():
        ties = ev[ev.duplicated(["stock_id", "avail_date"], keep=False)]
        assert len(ties) >= 2 * 40 * len(_TICKERS), f"{name}: the fixture has no tied prints"
        slot = stable.feature_slots[name]
        np.testing.assert_array_equal(
            slot, _later_print_reference(ev, stable.dates, stable.tickers, name),
            err_msg=f"{name}: a tied avail date did not resolve to the later print")
        # a tie can only move the Monday bar; every other cell is the sealed one
        assert slot[~monday].tobytes() == sealed.feature_slots[name][~monday].tobytes(), name

    # the Saturday balances are the only ones at or above these floors, so a cell names its print
    shares = 1e8
    for name, floor in (("margin_util", 9e5), ("short_util", 7e5)):
        slot = stable.feature_slots[name]
        live = monday & (d > d[5])
        assert live.sum() >= 40 and (slot[live] * shares >= floor - 0.5).all(), (
            f"{name}: a Monday bar is carrying the Friday print")
        rest = ~monday & np.isfinite(slot[:, 0])
        assert rest.sum() > 150 and (slot[rest] * shares < floor - 0.5).all()

    # channels without ties, and everything that is not a channel, are untouched
    for name in ("mrev_yoy", "holder_conc"):
        assert stable.feature_slots[name].tobytes() == sealed.feature_slots[name].tobytes()
    for name in ("dates", "open", "high", "low", "close", "volume", "active", "adv_usd",
                 "sector_id"):
        assert getattr(stable, name).tobytes() == getattr(sealed, name).tobytes(), f"{name} moved"
    assert stable.meta["stable_ties"] is True, "a tie-broken panel must say so"
    assert {k: v for k, v in stable.meta.items() if k != "stable_ties"} == sealed.meta


def test_stable_ties_changes_nothing_when_no_prints_tie(data_dir):
    """The fixture as shipped has no shared avail dates, so the switch must be a no-op on values."""
    sealed = tsp.build_taiwan_smallcap_panel(data_dir, channels=tsp.ALL_CHANNELS)
    stable = tsp.build_taiwan_smallcap_panel(data_dir, channels=tsp.ALL_CHANNELS, stable_ties=True)
    for ev in _channel_events(data_dir).values():
        assert not ev.duplicated(["stock_id", "avail_date"]).any()
    for name in tsp.ALL_CHANNELS:
        assert stable.feature_slots[name].tobytes() == sealed.feature_slots[name].tobytes(), name


@pytest.mark.parametrize("bad", [1, 0, "yes", None])
def test_a_non_bool_tie_break_is_rejected(data_dir, bad):
    with pytest.raises(TypeError, match="stable_ties must be a bool"):
        tsp.build_taiwan_smallcap_panel(data_dir, stable_ties=bad)


def test_stable_ties_rebuild_on_truncated_source_reproduces_every_past_cell(data_dir, tmp_path):
    """LEAK-2 for the tie-broken path: delete the future from the SOURCE, rebuild, compare."""
    _add_saturday_sessions(data_dir)
    full = tsp.build_taiwan_smallcap_panel(data_dir, channels=tsp.ALL_CHANNELS, stable_ties=True)
    cut_ix = int(full.T * 0.7)
    trunc_dir = tmp_path / "as_of"
    _truncate_source(data_dir, trunc_dir, pd.Timestamp(full.dates[cut_ix]))
    past = tsp.build_taiwan_smallcap_panel(trunc_dir, channels=tsp.ALL_CHANNELS, stable_ties=True)

    assert past.T == cut_ix + 1 and past.tickers == full.tickers
    for name in tsp.ALL_CHANNELS:
        np.testing.assert_array_equal(
            np.asarray(past.feature_slots[name]), np.asarray(full.feature_slots[name])[:past.T],
            err_msg=f"tie-broken slot {name} moved at t<=cut when the future was deleted")
