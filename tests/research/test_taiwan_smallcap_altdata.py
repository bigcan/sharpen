"""Tripwires for the Taiwan small/mid-cap alt-data probes (pre-registration 2026-07-15).

Guards the load-bearing correctness properties — all offline, NO FinMind token:
  - the pre-registration SPEC-HASH SEAL (a spec edit that moves a hash fails here);
  - MANDATORY size-neutralization in every spec;
  - the LEAK-2 causal availability lags (month-rev 10th-of-next-month; margin T+1; 集保 +6d) and
    the as-of alignment that must NOT surface a value before its avail_date;
  - the per-signal Tier-0 truncation-equivalence (compute(truncated)[t] == compute(full)[t]);
  - a full evaluate_batch smoke that exercises neutralization (incl. size) end-to-end and confirms
    the harness's own causality tripwire PASSES for all three probes.
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

from finrl_pro_ds.signals import Gates, evaluate_batch, make_synthetic_panel  # noqa: E402
from finrl_pro_ds.signals.features import ohlc_violations  # noqa: E402
from scripts.data.fetch_taiwan_fundamentals_finmind import (  # noqa: E402
    _is_tier_block,
    big_holder_percent,
    month_revenue_avail_date,
    parse_holding_lower_bound,
)
from scripts.research.taiwan_smallcap_altdata_eval import (  # noqa: E402
    PREREG_HASHES,
    _asof_grid,
    _causal_total_return_factor,
    _daily_membership,
    _month_revenue_yoy,
    build_panel,
    build_signals,
)


# --------------------------------------------------------------------------- #
# 1. Pre-registration seal: spec hashes + mandatory size-neutralization
# --------------------------------------------------------------------------- #
def test_spec_hashes_match_preregistration():
    sigs = build_signals()                                 # raises SystemExit on drift
    assert set(sigs) == set(PREREG_HASHES)
    for name, s in sigs.items():
        assert s.spec.content_hash() == PREREG_HASHES[name]


def test_size_neutralization_is_mandatory():
    for s in build_signals().values():
        assert "size" in s.spec.neutralization, f"{s.spec.name} dropped size-neutralization"
        assert s.spec.neutralization == ("winsor", "zscore", "sector", "size")


def test_expected_signs_are_pre_committed():
    sigs = build_signals()
    assert sigs["tw_smallcap_mom_rev"].spec.expected_sign == 1
    assert sigs["tw_smallcap_margin_crowd"].spec.expected_sign == -1
    assert sigs["tw_smallcap_holder_conc"].spec.expected_sign == 1


# --------------------------------------------------------------------------- #
# 2. Causal availability lags (LEAK-2)
# --------------------------------------------------------------------------- #
def test_month_revenue_availability_is_tenth_of_next_month():
    assert month_revenue_avail_date(2020, 4) == pd.Timestamp("2020-05-10")
    assert month_revenue_avail_date(2020, 12) == pd.Timestamp("2021-01-10")   # year roll
    assert month_revenue_avail_date(2019, 1) == pd.Timestamp("2019-02-10")


def test_tier_block_detection_halts_not_skips():
    # systematic tier/permission block → must be detected (so the fetcher HALTS, not skips 1000×)
    assert _is_tier_block(RuntimeError("FinMind HTTP 400 on TaiwanStockMonthRevenue/2330: level is register"))
    assert _is_tier_block(RuntimeError("permission denied"))
    # transient / single-id misses → NOT a tier block (skip-and-continue is correct)
    assert not _is_tier_block(RuntimeError("FinMind HTTP 500 on X/2330: timeout"))
    assert not _is_tier_block(RuntimeError("data_id not exist"))


def test_holding_tier_parse():
    assert parse_holding_lower_bound("1-999") == 1
    assert parse_holding_lower_bound("400,001-600,000") == 400_001
    assert parse_holding_lower_bound("more than 1,000,001") == 1_000_001
    assert parse_holding_lower_bound("total") is None
    assert parse_holding_lower_bound(12) == 400_001                            # integer TDCC code
    g = pd.DataFrame({
        "HoldingSharesLevel": ["1-999", "400,001-600,000", "more than 1,000,001", "total"],
        "percent": [60.0, 25.0, 15.0, 100.0], "unit": [0, 0, 0, 1_000_000]})
    assert big_holder_percent(g) == pytest.approx(40.0)                        # 25 + 15, > 400 lots


# --------------------------------------------------------------------------- #
# 3. As-of alignment never surfaces a value before its avail_date (the leak guard)
# --------------------------------------------------------------------------- #
def test_asof_grid_is_causal_and_forward_filled():
    dates = np.array(pd.date_range("2020-01-01", periods=10, freq="D"), dtype="datetime64[ns]")
    tickers = ("A", "B")
    ev = pd.DataFrame({
        "stock_id": ["A", "A", "B"],
        "avail_date": [pd.Timestamp("2020-01-04"), pd.Timestamp("2020-01-08"),
                       pd.Timestamp("2020-01-06")],
        "val": [1.0, 2.0, 9.0]})
    grid = _asof_grid(ev, dates, tickers, "val")
    # A: NaN before the 4th, 1.0 from the 4th, 2.0 from the 8th
    assert np.isnan(grid[:3, 0]).all()
    assert (grid[3:7, 0] == 1.0).all()
    assert (grid[7:, 0] == 2.0).all()
    # B: NaN before the 6th, then 9.0
    assert np.isnan(grid[:5, 1]).all()
    assert (grid[5:, 1] == 9.0).all()


def test_month_revenue_yoy_value_and_availability():
    months = pd.period_range("2019-01", "2020-06", freq="M")
    rev = pd.DataFrame({
        "stock_id": "1234",
        "revenue_year": [p.year for p in months],
        "revenue_month": [p.month for p in months],
        "revenue": [100.0 * (1.10 ** i) for i in range(len(months))],       # +10% MoM compounding
        "avail_date": [month_revenue_avail_date(p.year, p.month) for p in months],
    })
    yoy = _month_revenue_yoy(rev)
    # first YoY is for 2020-01 (needs 2019-01 base); availability = 2020-02-10
    row = yoy[yoy["avail_date"] == pd.Timestamp("2020-02-10")].iloc[0]
    assert row["yoy"] == pytest.approx(1.10 ** 12 - 1.0, rel=1e-6)
    # and it must NOT be visible before 2020-02-10
    dates = np.array(pd.date_range("2020-01-01", "2020-03-01", freq="D"), dtype="datetime64[ns]")
    grid = _asof_grid(yoy, dates, ("1234",), "yoy")
    before = dates < np.datetime64("2020-02-10")
    assert np.isnan(grid[before, 0]).all()
    assert np.isfinite(grid[dates >= np.datetime64("2020-02-10"), 0]).any()


# --------------------------------------------------------------------------- #
# 4. Daily membership expansion (monthly rebalance → causal daily mask)
# --------------------------------------------------------------------------- #
def test_daily_membership_active_from_rebalance():
    dates = np.array(pd.date_range("2020-01-01", periods=90, freq="D"), dtype="datetime64[ns]")
    members = pd.DataFrame({
        "rebalance_date": [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")],
        "stock_id": ["A", "B"]})                          # A in Jan-rebal, B in Feb-rebal
    mask = _daily_membership(members, dates, ("A", "B"))
    d = pd.DatetimeIndex(dates)
    assert not mask[d < pd.Timestamp("2020-01-31")].any()                     # nothing before rebal-1
    assert mask[(d >= pd.Timestamp("2020-01-31")) & (d < pd.Timestamp("2020-02-29")), 0].all()  # A on
    assert not mask[(d >= pd.Timestamp("2020-01-31")) & (d < pd.Timestamp("2020-02-29")), 1].any()  # B off
    assert mask[d >= pd.Timestamp("2020-02-29"), 1].all()                     # B on from rebal-2


# --------------------------------------------------------------------------- #
# 5. Per-signal Tier-0 truncation equivalence (LEAK-2 by construction)
# --------------------------------------------------------------------------- #
def _panel_with_slots(T=1100, N=80, seed=3):
    rng = np.random.default_rng(seed)
    slots = {
        "mrev_yoy": np.cumsum(0.01 * rng.standard_normal((T, N)), axis=0),
        "margin_util": 0.05 + 0.01 * np.cumsum(rng.standard_normal((T, N)), axis=0) / 50,
        "holder_conc": 30.0 + np.cumsum(0.1 * rng.standard_normal((T, N)), axis=0),
    }
    return make_synthetic_panel(T=T, N=N, n_sectors=6, seed=seed, feature_slots=slots)


@pytest.mark.parametrize("name", ["tw_smallcap_mom_rev", "tw_smallcap_margin_crowd",
                                  "tw_smallcap_holder_conc"])
def test_signal_truncation_equivalence(name):
    panel = _panel_with_slots()
    sig = build_signals()[name]
    full = np.asarray(sig.compute(panel), dtype=np.float64)
    for t in (300, 700, 1099):
        trunc = np.asarray(sig.compute(panel.truncated(t)), dtype=np.float64)
        assert trunc.shape[0] == t + 1
        a, b = full[t], trunc[t]
        assert np.array_equal(np.isnan(a), np.isnan(b))
        m = ~np.isnan(a)
        assert np.allclose(a[m], b[m], atol=1e-12)


# --------------------------------------------------------------------------- #
# 6. End-to-end evaluate_batch smoke — exercises neutralization (incl. size) + harness tripwire
# --------------------------------------------------------------------------- #
def test_evaluate_batch_runs_and_is_causal():
    panel = _panel_with_slots()
    gates = Gates.from_yaml(ROOT / "configs" / "taiwan_smallcap_altdata.gates.yaml")
    rs = evaluate_batch(build_signals(), panel, gates, "taiwan_smallcap_altdata_smoke")
    assert rs.n_trials == 3
    assert {c.name for c in rs.cards} == set(PREREG_HASHES)
    for c in rs.cards:
        assert c.spec_hash == PREREG_HASHES[c.name]
        assert c.verdict in {"GATE_FAIL", "LOGGED", "PROMISING"}
        # the harness's OWN Tier-0 causality tripwire must pass for every probe
        assert c.hygiene.passed, f"{c.name} failed hygiene: {c.hygiene}"


# --------------------------------------------------------------------------- #
# 7. Dividend total-return add-back is forward (LEAK-2) and OHLC-consistent
# --------------------------------------------------------------------------- #
def test_dividend_factor_ratchets_forward_only():
    idx = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=6, freq="D"))
    close = pd.DataFrame({"A": [10.0] * 6}, index=idx)
    div = pd.DataFrame({"ex_date": [pd.Timestamp("2020-01-04")], "stock_id": ["A"], "amount": [1.0]})
    cf = _causal_total_return_factor(close, div)["A"].to_numpy()
    assert (cf[:3] == 1.0).all()                  # bars BEFORE the ex-date are untouched
    assert cf[3] == pytest.approx(1.10)           # 1 + 1.0/10.0 on the ex-date
    assert np.allclose(cf[3:], 1.10)              # and it only ratchets, never falls back


def _write_synth_dataset(d: Path, nt=80, seed=5):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=330)
    tickers = [f"{2001 + i}" for i in range(nt)]
    shares = np.round(rng.uniform(5e7, 5e9, nt))
    prows, drows = [], []
    for i, tk in enumerate(tickers):
        close = float(rng.uniform(15, 120)) * np.exp(np.cumsum(0.015 * rng.standard_normal(len(dates))))
        op = close * np.exp(0.001 * rng.standard_normal(len(dates)))
        hi = np.maximum(op, close) * np.exp(np.abs(0.003 * rng.standard_normal(len(dates))))
        lo = np.minimum(op, close) * np.exp(-np.abs(0.003 * rng.standard_normal(len(dates))))
        prows.append(pd.DataFrame({"date": dates, "ticker": tk, "open": op, "high": hi,
                                   "low": lo, "close": close, "volume": rng.uniform(2e5, 8e6, len(dates))}))
        exd = rng.choice(dates, size=2, replace=False)
        drows.append(pd.DataFrame({"ex_date": pd.to_datetime(exd), "stock_id": tk,
                                   "amount": rng.uniform(0.5, 3.0, 2)}))
    wk = pd.date_range("2019-01-04", dates[-1], freq="W-FRI")
    sh = pd.concat([pd.DataFrame({"date": wk, "stock_id": tk,
                                  "big_holder_pct": np.clip(35 + np.cumsum(0.5 * rng.standard_normal(len(wk))), 5, 85),
                                  "total_shares": shares[i], "avail_date": wk + pd.Timedelta(days=6)})
                    for i, tk in enumerate(tickers)], ignore_index=True)
    mem = pd.DataFrame({"rebalance_date": pd.Timestamp("2019-06-28"), "stock_id": tickers,
                        "rank": range(51, 51 + nt), "market_cap": 1.0, "adv_twd": 1e7})
    pd.concat(prows, ignore_index=True).to_parquet(d / "prices.parquet", index=False)
    pd.concat(drows, ignore_index=True).to_parquet(d / "dividends.parquet", index=False)
    sh.to_parquet(d / "shareholding.parquet", index=False)
    pd.DataFrame({"stock_id": tickers, "name": tickers, "sector": "Electronics",
                  "type": "twse"}).to_parquet(d / "pool.parquet", index=False)
    (d / "universe").mkdir()
    mem.to_parquet(d / "universe" / "membership.parquet", index=False)


def test_build_panel_ohlc_clean_with_dividend_adjustment(tmp_path):
    """Regression: the causal total-return factor must scale ALL FOUR OHLC fields, not just close
    (scaling close alone pierces high/low → OHLC violations → the whole batch GATE_FAILs)."""
    _write_synth_dataset(tmp_path)
    panel = build_panel(tmp_path, adv_window=20)
    assert panel.meta["adjusted"] is True
    assert panel.meta["return_basis"] == "causal_total_return_forward"
    assert ohlc_violations(panel)["total"] == 0            # would be ~1 per active bar if close-only
    assert panel.active.any()


# --------------------------------------------------------------------------- #
# 6. Sector-map provenance — the 2026-07-31 silent-restatement regression
# --------------------------------------------------------------------------- #

def _sectorise(d: Path, mapping: dict[str, str], *, name: str = "pool.parquet") -> None:
    """Rewrite a pool file with an explicit ticker -> sector mapping."""
    pool = pd.read_parquet(d / "pool.parquet")
    pool["sector"] = pool["stock_id"].astype(str).map(mapping).fillna("Electronics")
    pool.to_parquet(d / name, index=False)


def test_sector_map_sha_tracks_the_partition_not_the_file(tmp_path):
    """The 2026-07-31 defect: re-enumerating `pool.parquet` silently restated a SEALED scorecard.

    `sector` is a mandatory neutralization control, so its partition is load-bearing on every
    downstream number — yet `spec_hash`, `liquid_days_ge25` and `n_names_pool` are all blind to it,
    which is exactly why the drift read as a code regression. `sector_map_sha` closes that hole.

    It must key on the PANEL'S OWN (ticker, sector) pairs: a pool that merely gains unrelated
    listings (FinMind re-enumerates on every fetch) must NOT trip it, or the stamp would cry wolf on
    every data pull and get ignored — but any genuine change to this panel's partition must.
    """
    _write_synth_dataset(tmp_path)
    base = build_panel(tmp_path, adv_window=20).meta["sector_map"]
    assert base["n_unknown"] == 0 and base["n_sectors"] == 1      # fixture is single-sector

    # (a) unrelated listings appended to the pool -> same panel partition -> SAME sha
    pool = pd.read_parquet(tmp_path / "pool.parquet")
    extra = pd.DataFrame({"stock_id": ["9001", "9002"], "name": ["x", "y"],
                          "sector": ["Finance", "Steel"], "type": ["twse", "twse"]})
    pd.concat([pool, extra], ignore_index=True).to_parquet(tmp_path / "pool.parquet", index=False)
    assert build_panel(tmp_path, adv_window=20).meta["sector_map"]["sector_map_sha"] == \
        base["sector_map_sha"]

    # (b) one panel name re-classified -> partition really changed -> DIFFERENT sha
    _sectorise(tmp_path, {"2001": "Finance"})
    moved = build_panel(tmp_path, adv_window=20).meta["sector_map"]
    assert moved["sector_map_sha"] != base["sector_map_sha"]
    assert moved["n_sectors"] == 2


def test_frozen_pool_wins_over_a_rewritten_live_pool(tmp_path):
    """The fetcher only ever writes `pool.parquet`; the evaluation must read the frozen snapshot so
    an unrelated data pull cannot move a recorded result out from under it."""
    _write_synth_dataset(tmp_path)
    _sectorise(tmp_path, {"2001": "Finance", "2002": "Steel"}, name="pool.frozen.parquet")
    frozen = build_panel(tmp_path, adv_window=20).meta["sector_map"]
    assert frozen["source"] == "pool.frozen.parquet" and frozen["frozen"] is True
    assert frozen["n_sectors"] == 3

    # rewriting the LIVE pool (what the fetcher does) must not perturb the pinned result
    _sectorise(tmp_path, {t: f"S{i}" for i, t in enumerate(
        pd.read_parquet(tmp_path / "pool.parquet")["stock_id"].astype(str))})
    after = build_panel(tmp_path, adv_window=20).meta["sector_map"]
    assert after["sector_map_sha"] == frozen["sector_map_sha"]

    # ...and with the freeze removed, that same rewrite DOES move it (proves the test has teeth)
    (tmp_path / "pool.frozen.parquet").unlink()
    thawed = build_panel(tmp_path, adv_window=20).meta["sector_map"]
    assert thawed["frozen"] is False and thawed["sector_map_sha"] != frozen["sector_map_sha"]
