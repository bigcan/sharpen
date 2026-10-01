"""crucible-v17.0 — the data / store items the 2026-09-30 deep audit listed as open or not fixed:
shared trial ledgers, the T86 store's ticker set, the bridge fetching every series twice, the Taiwan
small-cap base book charged the stock tax, and the ``balance_util`` build cost."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.research.crucible_orchestrator import foreign_ledger_substrates
from sharpen.crucible.data import FredConnector
from sharpen.crucible.data.altdata_bridge import bridge_altdata_feature_slots
from sharpen.crucible.data.taiwan_smallcap_panel import balance_util
from sharpen.crucible.data.twse_institutional import (
    _DEFAULT_TICKERS,
    _META_KEY,
    TwseInstitutionalConnector,
)
from sharpen.crucible.ledger import TrialLedger, TrialRecord
from sharpen.signals.generation.config import load_generation_meta

ROOT = Path(__file__).resolve().parents[2]


# ---- shared ledger ------------------------------------------------------------------------------

def _ledger(tmp_path, runs: dict[str, str]) -> TrialLedger:
    led = TrialLedger(tmp_path / "trial_ledger.db")
    for h, run in runs.items():
        led.record(TrialRecord(candidate_hash=h, crucible_version="t", formula="rank(close)",
                               first_seen_run=run))
    return led


def test_ledger_reports_which_substrates_wrote_it(tmp_path) -> None:
    led = _ledger(tmp_path, {
        "a": "tick-us_equity-2026-08-10T23:52:00+00:00",
        "b": "tick-us_equity-2026-08-11T01:00:00+00:00",
        "c": "tick-cross_asset-2026-07-02T23:53:17.374661+00:00",
        "d": "loop-manual-1"})                                   # not an orchestrator tick
    assert led.substrate_row_counts() == {"us_equity": 2, "cross_asset": 1}
    assert foreign_ledger_substrates(led, "us_equity") == {"cross_asset": 1}
    assert foreign_ledger_substrates(led, "taiwan_smallcap") == {"us_equity": 2, "cross_asset": 1}


def test_a_ledger_of_its_own_substrate_is_not_shared(tmp_path) -> None:
    led = _ledger(tmp_path, {"a": "tick-us_equity-2026-08-10T23:52:00+00:00",
                             "b": "tick-us_equity-night-3"})     # a non-ISO stamp of its own
    assert foreign_ledger_substrates(led, "us_equity") == {}
    assert foreign_ledger_substrates(TrialLedger(tmp_path / "new.db"), "us_equity") == {}


# ---- T86 store ticker set -----------------------------------------------------------------------

def _t86_payload(rows: list[list[str]]) -> dict:
    return {"stat": "OK",
            "fields": ["證券代號", "證券名稱", "外陸資買賣超股數(不含外資自營商)", "投信買賣超股數",
                       "自營商買賣超股數", "三大法人買賣超股數"],
            "data": rows}


def _t86(tmp_path, tickers, calls=None) -> TwseInstitutionalConnector:
    def transport(url: str) -> dict:
        if calls is not None:
            calls.append(url)
        return _t86_payload([["0050", "x", "1", "2", "3", "6"], ["2330", "y", "4", "5", "6", "15"]])
    return TwseInstitutionalConnector(
        transport=transport, tickers=tickers, store_path=tmp_path / "t86.json", sleep_seconds=0,
        today=lambda: np.datetime64("2026-10-01", "D"), max_lookback_days=30)


def _fetch(conn: TwseInstitutionalConnector, ticker: str):
    ref = next(r for r in conn.discover() if r.series_id == f"{ticker}:foreign_net")
    return conn.fetch(ref, "2026-09-01", "2026-09-02")


def test_t86_store_refuses_a_ticker_it_was_not_built_with(tmp_path) -> None:
    assert _fetch(_t86(tmp_path, ("0050",)), "0050").n_obs == 2
    with pytest.raises(ValueError, match="built without tickers"):
        _fetch(_t86(tmp_path, ("0050", "2330")), "2330")
    # Before v17.0 the wider instance read both stored days as "2330 had no T86 row": n_obs == 0,
    # no error, and no later poll would repair it (the days are already marked polled).


def test_t86_store_serves_the_same_or_a_narrower_set_without_refetching(tmp_path) -> None:
    _fetch(_t86(tmp_path, ("0050", "2330")), "2330")
    calls: list[str] = []
    assert _fetch(_t86(tmp_path, ("2330",), calls), "2330").n_obs == 2
    assert calls == []


def test_t86_legacy_store_is_read_as_the_default_ticker_set(tmp_path) -> None:
    import json
    (tmp_path / "t86.json").write_text(
        json.dumps({"20260901": {"0050": {"foreign_net": 1.0}}}), encoding="utf-8")   # no _meta
    assert _fetch(_t86(tmp_path, tuple(_DEFAULT_TICKERS)), "0050").n_obs == 2   # stored day + polled day
    with pytest.raises(ValueError, match="built without tickers"):
        _fetch(_t86(tmp_path, ("2330",)), "2330")
    assert json.loads((tmp_path / "t86.json").read_text(encoding="utf-8"))[_META_KEY] == {
        "tickers": sorted(_DEFAULT_TICKERS)}                       # stamped on the first save


# ---- alt-data bridge: one fetch per accepted series -----------------------------------------------

def test_bridge_fetches_each_accepted_series_once() -> None:
    bars = np.arange(np.datetime64("2019-06-01"), np.datetime64("2020-06-01"),
                     np.timedelta64(1, "D")).astype("datetime64[ns]")
    days = np.arange(np.datetime64("2019-05-01"), np.datetime64("2020-06-01"), np.timedelta64(1, "D"))
    obs = [{"date": str(d), "value": f"{4.0 + 0.001 * i:.4f}"} for i, d in enumerate(days)]
    calls: list[str] = []

    def transport(url: str) -> dict:
        calls.append(url)
        return {"observations": obs}

    conn = FredConnector(transport=transport, series=(("DGS10", "10Y yield", "D"),),
                         release_lag_days=1)
    slots = bridge_altdata_feature_slots(bar_dates=bars, start="2019-06-01", end="2020-05-31",
                                         connectors=[conn])
    assert list(slots) == ["fred:DGS10"] and np.isfinite(slots["fred:DGS10"]).sum() > 300
    assert len(calls) == 1                                       # was 2: the survey, then the build


# ---- Taiwan small-cap base book cost ------------------------------------------------------------

def test_taiwan_smallcap_base_book_has_its_own_cost() -> None:
    meta = load_generation_meta(ROOT / "configs" / "taiwan_smallcap_signal_eval.gates.yaml")
    assert meta["panel"] == "taiwan_smallcap"
    assert meta["base_cost_bps"] == 0.0010                       # futures book: no 0.30% stock tax
    for other in ("signal_eval", "taiwan_signal_eval", "us_equity_signal_eval"):
        m = load_generation_meta(ROOT / "configs" / f"{other}.gates.yaml")
        assert m["base_cost_bps"] is None                        # every other substrate: unchanged


# ---- balance_util: grouped once, bit-identical ----------------------------------------------------

def _ref_balance_util(margin, shareholding, *, balance_col, out_col):
    """The pre-v17.0 loop, verbatim (a full-frame boolean filter per ticker)."""
    mg = margin.copy()
    mg["stock_id"] = mg["stock_id"].astype(str)
    mg["date"] = pd.to_datetime(mg["date"])
    sh = shareholding.dropna(subset=["total_shares"]).copy()
    sh["stock_id"] = sh["stock_id"].astype(str)
    sh["avail_date"] = pd.to_datetime(sh["avail_date"])
    frames = []
    for tk, g in mg.groupby("stock_id"):
        s = sh[sh["stock_id"] == tk][["avail_date", "total_shares"]].sort_values("avail_date")
        if s.empty:
            continue
        g = g.sort_values("date")
        merged = pd.merge_asof(
            g[["date", balance_col]].rename(columns={"date": "avail_date"}),
            s, on="avail_date", direction="backward")
        util = np.where(merged["total_shares"] > 0, merged[balance_col] / merged["total_shares"],
                        np.nan)
        frames.append(pd.DataFrame({
            "stock_id": tk,
            "avail_date": pd.to_datetime(g["date"].to_numpy()) + pd.tseries.offsets.BDay(1),
            out_col: util}))
    return pd.concat(frames, ignore_index=True).dropna(subset=[out_col])


def test_balance_util_matches_the_per_ticker_filter_bit_for_bit() -> None:
    rng = np.random.default_rng(0)
    tickers = [f"{1000 + i}" for i in range(25)]
    days = pd.bdate_range("2020-01-01", periods=260)
    margin = pd.DataFrame({
        "stock_id": np.repeat(tickers, len(days)), "date": np.tile(days, len(tickers)),
        "margin_balance": rng.integers(0, 10_000, len(tickers) * len(days)).astype(float)})
    margin = margin.sample(frac=1.0, random_state=1).reset_index(drop=True)      # unsorted input
    weeks = days[::5]
    sh = pd.DataFrame({
        "stock_id": np.repeat(tickers[:-2], len(weeks)),                           # 2 names lack shares
        "avail_date": np.tile(weeks, len(tickers) - 2),
        "total_shares": rng.integers(0, 3, (len(tickers) - 2) * len(weeks)) * 1e6})
    sh.loc[sh.sample(frac=0.05, random_state=2).index, "total_shares"] = np.nan
    sh = pd.concat([sh, sh.sample(frac=0.03, random_state=3)]).sample(frac=1.0, random_state=4)
    got = balance_util(margin, sh, balance_col="margin_balance", out_col="margin_util")
    ref = _ref_balance_util(margin, sh, balance_col="margin_balance", out_col="margin_util")
    assert len(got) > 1000
    pd.testing.assert_frame_equal(got, ref, check_exact=True)
