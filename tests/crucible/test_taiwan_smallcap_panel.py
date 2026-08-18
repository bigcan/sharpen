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
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finrl_pro_ds.crucible.data import taiwan_smallcap_panel as tsp

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
