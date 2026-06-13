"""Regression tests for the stale-print / inter-bar-continuity scan in
scripts/clean_ohlcv.py (detect_stale_runs).

Guards two failure modes that 500+ sessions of green /audit runs missed:
  1. The gmgp1-gold flat-OHLC teleport print (a zero-range bar at a wrong level,
     reverting) must be FLAGGED — R1's identical-close-run logic alone misses it.
  2. Normal data with many benign zero-range bars (BTC 1-min runs ~18% flat-OHLC)
     must NOT be flagged — only the round-trip *spike* discriminates corruption.

The synthetic tests are self-contained (no data files). Two integration
tripwires read the real corrupt gold file and a real exchange file; they skip
cleanly if the data is not present (CI without the data dir).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from clean_ohlcv import detect_stale_runs  # noqa: E402

GOLD_CORRUPT = ROOT / "data" / "cme" / "gold_2025_2026q1_15min_stitched.parquet"
BTC_REAL = ROOT / "data" / "bitfinex" / "btc_usdt_perp_2025_1min.parquet"


def _r1_run_metrics(df: pd.DataFrame) -> dict:
    """Canonical 1-min identical-close-run metrics, verbatim from the R1 probe."""
    close = df["close"].to_numpy(np.float64)
    vol = df["volume"].to_numpy(np.float64)
    n = len(close)
    same = np.zeros(n, dtype=bool)
    same[1:] = close[1:] == close[:-1]
    starts = np.flatnonzero(~same)
    ends = np.r_[starts[1:] - 1, n - 1]
    lens = ends - starts + 1
    volsum = np.add.reduceat(vol, starts)
    sus = (lens >= 30) & (volsum > 0)
    suspect = np.zeros(n, dtype=bool)
    for i0, i1 in zip(starts[sus], ends[sus]):
        suspect[i0:i1 + 1] = True
    logret = np.zeros(n)
    logret[1:] = np.log(np.where(close[1:] > 0, close[1:], np.nan)
                        / np.where(close[:-1] > 0, close[:-1], np.nan))
    logret = np.nan_to_num(logret)
    boundary = np.zeros(n, dtype=bool)
    boundary[1:] = suspect[1:] != suspect[:-1]
    tot = np.abs(logret).sum()
    sps = float(np.abs(logret[boundary]).sum() / tot) if tot > 0 else 0.0
    return {"stale_suspect_frac": round(float(suspect.mean()), 5),
            "stale_pnl_share": round(sps, 5),
            "stale_runs": int(sus.sum())}


def _walk(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    o = close * (1 + rng.normal(0, 2e-4, n))
    h = np.maximum(o, close) * (1 + np.abs(rng.normal(0, 2e-4, n)))
    lo = np.minimum(o, close) * (1 - np.abs(rng.normal(0, 2e-4, n)))
    vol = rng.uniform(1, 100, n)
    ts = pd.date_range("2025-01-01", periods=n, freq="1min")
    return pd.DataFrame({"timestamp": ts, "open": o, "high": h, "low": lo,
                         "close": close, "volume": vol})


def test_reproduces_r1_run_metrics_at_1min():
    """At 1-min the run-based metrics match the canonical R1 scan exactly."""
    df = _walk()
    # inject a 40-bar identical-close run (R1's staleness sub-type)
    df.loc[2000:2039, "close"] = df.loc[2000, "close"]
    out = detect_stale_runs(df)
    ref = _r1_run_metrics(df)
    assert out["min_run_bars"] == 30
    for k in ("stale_suspect_frac", "stale_pnl_share", "stale_runs"):
        assert out[k] == ref[k], f"{k}: {out[k]} != R1 {ref[k]}"


def test_timeframe_scaling():
    """min_run_bars auto-scales: 30 @1-min, floor 6 on coarser bars."""
    assert detect_stale_runs(_walk())["min_run_bars"] == 30
    df15 = _walk(n=4000).iloc[::15].copy()
    assert detect_stale_runs(df15)["min_run_bars"] == 6
    ts = pd.date_range("2015-01-01", periods=400, freq="1D")
    cl = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, 400)))
    dfd = pd.DataFrame({"timestamp": ts, "open": cl, "high": cl * 1.001,
                        "low": cl * 0.999, "close": cl, "volume": np.ones(400)})
    assert detect_stale_runs(dfd)["min_run_bars"] == 6


def test_clean_data_not_flagged():
    out = detect_stale_runs(_walk(seed=7))
    assert out["flagged"] is False
    assert out["flat_ohlc_spikes"] == 0


def test_flat_ohlc_teleport_spike_flagged():
    """The gmgp1-gold artifact: an isolated zero-range bar at a +5% wrong level
    that reverts. Must be flagged via flat_ohlc_spikes."""
    df = _walk(seed=3)
    i = 3000
    bad = float(df.loc[i - 1, "close"]) * 1.05            # +5% teleport
    for col in ("open", "high", "low", "close"):
        df.loc[i, col] = bad                              # flat-OHLC (zero range)
    df.loc[i + 1, "close"] = float(df.loc[i - 1, "close"])  # revert
    out = detect_stale_runs(df)
    assert out["flat_ohlc_spikes"] >= 1
    assert out["flagged"] is True
    assert out["flat_ohlc_sample"], "spike sample should be populated"


def test_benign_flat_ohlc_fraction_not_flagged():
    """Regression: many zero-range bars AT the local level (no jump) — like real
    1-min crypto's ~18% flat bars — must NOT trip the gate. Only spikes do."""
    df = _walk(seed=11)
    rng = np.random.default_rng(99)
    idx = rng.choice(np.arange(100, len(df) - 100), size=int(0.15 * len(df)),
                     replace=False)
    for i in idx:                                         # zero range, no teleport
        c = float(df.loc[i, "close"])
        df.loc[i, ["open", "high", "low"]] = c
    out = detect_stale_runs(df)
    assert out["flat_ohlc_frac"] > 0.10                   # high fraction...
    assert out["flat_ohlc_spikes"] == 0                   # ...but no spikes
    assert out["flagged"] is False                        # ...so NOT flagged


def test_missing_columns_graceful():
    df = _walk()
    detect_stale_runs(df.drop(columns=["volume"]))        # no volume
    detect_stale_runs(df.drop(columns=["timestamp"]))     # no time axis
    out = detect_stale_runs(df.iloc[:2])                  # too short
    assert out["flagged"] is False
    # close-only input must not manufacture flat-OHLC spikes from a price jump
    # (the o=h=lo=close fallback would otherwise read as a degenerate flat bar)
    co = df[["timestamp", "close", "volume"]].copy()
    co.loc[3000, "close"] = float(co.loc[2999, "close"]) * 1.05
    co.loc[3001, "close"] = float(co.loc[2999, "close"])
    assert detect_stale_runs(co)["flat_ohlc_spikes"] == 0


@pytest.mark.skipif(not GOLD_CORRUPT.exists(), reason="corrupt gold file absent")
def test_real_corrupt_gold_is_flagged():
    """Tripwire: the real file Fable identified (flat-OHLC stale prints supplying
    ~21% of best-fold P&L) MUST be flagged. This is the blind spot item 1 closes."""
    out = detect_stale_runs(pd.read_parquet(GOLD_CORRUPT))
    assert out["flagged"] is True
    assert out["flat_ohlc_spikes"] > 0


@pytest.mark.skipif(not BTC_REAL.exists(), reason="real BTC file absent")
def test_real_exchange_data_not_flagged():
    """Specificity tripwire: real continuous exchange data (17%+ flat-OHLC bars)
    must NOT be flagged — guards against the frac-gate false positive."""
    out = detect_stale_runs(pd.read_parquet(BTC_REAL))
    assert out["flat_ohlc_spikes"] == 0
    assert out["flagged"] is False
