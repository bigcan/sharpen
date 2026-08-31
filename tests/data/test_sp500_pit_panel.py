"""BALLAST P0 — PIT S&P 500 panel builder tests.

The load-bearing test here is :class:`TestMembershipCausality` — the negative test that fails if
as-of membership is ever back-filled (LEAK-2). Everything else guards the survivorship accounting,
which is the number every BALLAST result has to be read against.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sharpen.data import sp500_pit_panel as pit


# --------------------------------------------------------------------------- #
# Fixtures — a tiny synthetic membership history + price fetch, no network.
# --------------------------------------------------------------------------- #
@pytest.fixture
def members_csv(tmp_path):
    """AAA is always in; BBB joins 2020-07-01; CCC leaves 2020-07-01."""
    path = tmp_path / "members.csv"
    pd.DataFrame({
        "date": ["2020-01-02", "2020-07-01", "2021-01-04"],
        "tickers": ["AAA,CCC", "AAA,BBB", "AAA,BBB"],
    }).to_csv(path, index=False)
    return path


def _fake_fetch(assets, start, end, **_kw):
    """Deterministic OHLCV for every requested ticker except ``DEAD`` (all-NaN = delisted)."""
    idx = pd.bdate_range("2019-07-01", "2021-06-30")
    out = {}
    for field in ("open", "high", "low", "close", "volume"):
        cols = {}
        for k, a in enumerate(assets):
            if a == "DEAD":
                cols[a] = np.full(len(idx), np.nan)
            elif field == "volume":
                cols[a] = np.full(len(idx), 5e6)
            else:
                cols[a] = 100.0 + k * 10 + np.arange(len(idx)) * 0.01
        out[field] = pd.DataFrame(cols, index=idx)
    return out


def _no_clean(wide):
    return wide, {c: {"stale_flagged": False} for c in wide["close"].columns}


# --------------------------------------------------------------------------- #
class TestMembershipCausality:
    """LEAK-2: membership on date t may only use snapshots stamped <= t."""

    def test_membership_is_asof_not_backfilled(self, members_csv):
        snap_d, snap_m, union = pit.load_membership(members_csv)
        dates = np.array(["2020-06-30", "2020-07-01", "2020-07-02"], dtype="datetime64[ns]")
        M = pit.membership_matrix(dates, snap_d, snap_m, union)
        j = {t: i for i, t in enumerate(union)}

        # BBB joins ON 2020-07-01 -> absent the day before, present from that day.
        assert not M[0, j["BBB"]], "BBB back-filled before its addition date (LEAK-2)"
        assert M[1, j["BBB"]] and M[2, j["BBB"]]
        # CCC leaves ON 2020-07-01 -> present the day before, absent from that day.
        assert M[0, j["CCC"]]
        assert not M[1, j["CCC"]] and not M[2, j["CCC"]]

    def test_dates_before_first_snapshot_are_empty(self, members_csv):
        snap_d, snap_m, union = pit.load_membership(members_csv)
        dates = np.array(["2019-12-31"], dtype="datetime64[ns]")
        M = pit.membership_matrix(dates, snap_d, snap_m, union)
        assert not M.any(), "pre-history dates must be empty, never back-filled from snapshot 0"

    def test_truncating_the_panel_does_not_change_earlier_membership(self, members_csv):
        """The Tier-0 style tripwire: M[:k] computed on a truncated date axis is identical."""
        snap_d, snap_m, union = pit.load_membership(members_csv)
        dates = pd.bdate_range("2020-06-01", "2020-08-01").to_numpy(dtype="datetime64[ns]")
        full = pit.membership_matrix(dates, snap_d, snap_m, union)
        for k in (5, 20, len(dates) - 1):
            trunc = pit.membership_matrix(dates[:k + 1], snap_d, snap_m, union)
            assert np.array_equal(trunc, full[:k + 1]), f"membership at t<={k} depends on the future"

    def test_unknown_tickers_in_snapshot_are_ignored(self, members_csv):
        snap_d, snap_m, _ = pit.load_membership(members_csv)
        dates = np.array(["2020-08-03"], dtype="datetime64[ns]")
        M = pit.membership_matrix(dates, snap_d, snap_m, ["AAA"])   # BBB not in the price panel
        assert M.shape == (1, 1) and M[0, 0]


class TestLoadMembership:
    def test_normalizes_share_class_tickers(self, tmp_path):
        path = tmp_path / "m.csv"
        pd.DataFrame({"date": ["2020-01-02"], "tickers": ["BRK.B,BF.B"]}).to_csv(path, index=False)
        _, members, union = pit.load_membership(path)
        assert union == ["BF-B", "BRK-B"] and members[0] == frozenset({"BRK-B", "BF-B"})

    def test_start_filter_and_missing_file(self, members_csv, tmp_path):
        snap_d, _, _ = pit.load_membership(members_csv, start="2020-07-01")
        assert len(snap_d) == 2
        with pytest.raises(FileNotFoundError, match="PIT membership CSV not found"):
            pit.load_membership(tmp_path / "nope.csv")


class TestBuildPitPanel:
    def _panel(self, members_csv, tmp_path, **kw):
        kw.setdefault("min_adv_usd", 0.0)
        return pit.build_pit_panel(
            start="2020-01-02", end="2021-06-30", members_csv=members_csv,
            cache_path=tmp_path / "p.pkl", fetch_fn=_fake_fetch, clean_fn=_no_clean, **kw)

    def test_active_mask_is_membership_and_priced_and_liquid(self, members_csv, tmp_path):
        p = self._panel(members_csv, tmp_path)
        j = {t: i for i, t in enumerate(p.tickers)}
        t_before = int(np.searchsorted(p.dates, np.datetime64("2020-06-30")))
        t_after = int(np.searchsorted(p.dates, np.datetime64("2020-07-01")))
        assert not p.active[t_before, j["BBB"]] and p.active[t_after, j["BBB"]]
        assert p.active[t_before, j["CCC"]] and not p.active[t_after, j["CCC"]]
        # AAA is a member for the whole evaluated window, but NOT in the pre-history warmup rows
        # (dates before the first snapshot are empty by design, never back-filled).
        t0 = int(np.searchsorted(p.dates, np.datetime64("2020-01-02")))
        assert p.active[t0:, j["AAA"]].all()
        assert not p.active[:t0, j["AAA"]].any()

    def test_liquidity_floor_excludes_thin_names(self, members_csv, tmp_path):
        p = self._panel(members_csv, tmp_path, min_adv_usd=1e12)
        assert not p.active.any(), "an unreachable ADV floor must empty the universe"

    def test_unpriceable_names_are_dropped_and_accounted(self, members_csv, tmp_path):
        path = tmp_path / "m2.csv"
        pd.DataFrame({"date": ["2020-01-02"], "tickers": ["AAA,DEAD"]}).to_csv(path, index=False)
        p = pit.build_pit_panel(
            start="2020-01-02", end="2021-06-30", members_csv=path,
            cache_path=tmp_path / "p2.pkl", fetch_fn=_fake_fetch, clean_fn=_no_clean,
            min_adv_usd=0.0)
        s = p.meta["survivorship"]
        assert "DEAD" not in p.tickers
        assert s["unpriceable_tickers"] == ["DEAD"]
        assert s["n_ever_members"] == 2 and s["n_priceable"] == 1
        assert s["unpriceable_frac"] == pytest.approx(0.5)

    def test_meta_caps_the_verdict_and_flags_the_bias(self, members_csv, tmp_path):
        """A free-data panel must never be able to present itself as survivorship-free."""
        p = self._panel(members_csv, tmp_path)
        assert p.meta["survivorship_free"] is False
        assert p.meta["verdict_cap"] == "PROMISING"
        assert "UPWARD" in p.meta["survivorship"]["bias_direction"]

    def test_fetch_start_warms_up_before_the_evaluated_start(self, members_csv, tmp_path):
        p = self._panel(members_csv, tmp_path)
        assert p.dates[0] < np.datetime64("2020-01-02"), "no warmup rows before the eval start"

    def test_cache_roundtrip(self, members_csv, tmp_path):
        a = self._panel(members_csv, tmp_path)
        b = pit.build_pit_panel(   # no fetch_fn: must come from cache, not the network
            start="2020-01-02", end="2021-06-30", members_csv=members_csv,
            cache_path=tmp_path / "p.pkl", min_adv_usd=0.0)
        assert np.array_equal(a.close, b.close) and a.tickers == b.tickers
        assert (tmp_path / "p.manifest.json").exists()

    def test_failed_fetch_chunk_does_not_kill_the_build(self, members_csv, tmp_path):
        calls = {"n": 0}

        def flaky(assets, start, end, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("yfinance rate limit")
            return _fake_fetch(assets, start, end, **kw)

        p = pit.build_pit_panel(
            start="2020-01-02", end="2021-06-30", members_csv=members_csv,
            cache_path=tmp_path / "p3.pkl", fetch_fn=flaky, clean_fn=_no_clean,
            min_adv_usd=0.0, batch=1)
        assert p.N >= 1 and calls["n"] > 1


class TestSectorIds:
    def test_unmapped_former_members_get_their_own_unknown_bucket(self):
        ids = pit._sector_ids(["AAA", "BBB", "GONE"], {"AAA": "Tech", "BBB": "Energy"})
        assert ids[2] not in ids[:2], "Unknown must not merge into a real sector"
        assert len(set(ids)) == 3
