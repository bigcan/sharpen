"""Tripwire tests for the VWAP / anchored-VWAP falsification probes.

The probes' verdict (NO_GO) is load-bearing, so these pin the three properties an
adversarial auditor would attack:

  1. **Matched control is apples-to-apples** — VWAP and its time-weighted control
     differ ONLY by volume-weighting. Proven structurally: when volume is CONSTANT,
     rolling/anchored/session VWAP must equal the SMA control EXACTLY. If they don't,
     the uplift comparison (the decisive gate) is comparing two different things.
  2. **The backtest is causal (no same-bar leak)** — a signal that perfectly predicts
     the SAME bar's return earns ~0 through the real `backtest()` (weights shift 1),
     while the leaked no-shift variant earns a huge Sharpe. Confirms the NO_GO cannot
     be hiding an inflated/leaked GO, and that costs/shift are not silently broken.
  3. **The NO_GO is NOT a return-suppression bug** — a real gross (frictionless) trend
     signal exists; the verdict comes from costs + control-redundancy, not a bug that
     zeroes returns.

These are mutation tripwires: reverting the `.shift(1)` in `backtest`, or making the
control use a different window, makes a test here fail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.research.vwap_avwap_falsification as vw
import scripts.research.vwap_intraday_falsification as vwi


def _panel(t: int = 400, n_assets: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=t, freq="B")
    cols = [f"A{i}" for i in range(n_assets)]
    px = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (t, n_assets)), axis=0),
                      index=idx, columns=cols)
    return idx, cols, px


# --- 1. matched control apples-to-apples (constant volume => VWAP == SMA) ---------
def test_rolling_vwap_equals_sma_when_volume_constant():
    idx, cols, tp = _panel()
    vol = pd.DataFrame(1.0, index=idx, columns=cols)
    for n in (21, 63):
        vwap = vw.rolling_vwap(tp, vol, n)
        sma = tp.rolling(n, min_periods=n).mean()        # the probe's control line
        np.testing.assert_allclose(vwap.values, sma.values, rtol=1e-9, equal_nan=True)


def test_anchored_vwap_equals_anchored_sma_when_volume_constant():
    idx, cols, tp = _panel()
    vol = pd.DataFrame(1.0, index=idx, columns=cols)
    for blk in (63, 252):
        avwap = vw.anchored_vwap(tp, vol, blk)
        asma = vw.anchored_sma(tp, blk)
        np.testing.assert_allclose(avwap.values, asma.values, rtol=1e-9, equal_nan=True)


def test_intraday_session_vwap_equals_sma_when_volume_constant():
    t = 200
    idx = pd.date_range("2024-01-01", periods=t, freq="15min")
    rng = np.random.default_rng(1)
    price = 2000 + np.cumsum(rng.normal(0, 1.0, t))
    df = pd.DataFrame({"timestamp": idx, "high": price, "low": price,
                       "close": price, "volume": 1.0})
    df["tp"] = (df["high"] + df["low"] + df["close"]) / 3.0
    refs = vwi.session_refs(df, "cal")
    np.testing.assert_allclose(refs["vwap"].values, refs["sma"].values,
                               rtol=1e-9, equal_nan=True)


# --- 2. backtest causality: no same-bar leak (mutation tripwire on .shift(1)) ------
def test_backtest_is_causal_no_same_bar_leak():
    idx, cols, px = _panel(t=800, seed=3)
    rets = px.pct_change()
    sig = np.sign(rets).fillna(0.0)                       # perfect SAME-bar predictor
    gross, _cost, _turn = vw.backtest(sig, rets)          # real harness (shifts weights)
    causal_sh = vw.sharpe(gross)
    leaked = (sig * rets).sum(axis=1)                     # the leak: same-bar w x same-bar ret
    leaked_sh = vw.sharpe(leaked)
    assert leaked_sh > 5.0, "test setup insensitive — same-bar signal should be hugely +"
    assert abs(causal_sh) < 1.5, (
        f"backtest leaked same-bar info (causal Sharpe {causal_sh}); is .shift(1) intact?")
    assert causal_sh < leaked_sh - 4.0                    # causal strictly worse than leaked


# --- 3. costs reduce return monotonically + one-way (no double count) --------------
def test_cost_is_one_way_and_monotonic():
    idx, cols, px = _panel(t=300, seed=5)
    rets = px.pct_change()
    w = pd.DataFrame(0.0, index=idx, columns=cols)
    w.iloc[:, 0] = 1.0                                    # establish once, hold forever
    gross, cost_daily, turn = vw.backtest(w, rets)
    years = (idx[-1] - idx[0]).days / 365.25
    np.testing.assert_allclose(turn, 1.0 / years, rtol=1e-6)   # one-way (NOT 2x round-trip)
    # net total return strictly decreasing in fee (costs subtract, never add)
    tot = {cm: float((gross - cost_daily[cm]).sum()) for cm in vw.COST_MODELS}
    assert tot["frictionless"] >= tot["standard_2bps"] >= tot["harsh_10bps"]


# --- 4. the verdict is NO_GO AND is not a return-suppression artifact --------------
def test_swing_verdict_no_go_and_not_suppression_bug():
    v = vw.main()                                         # reads the cached clean panel
    assert v["decision"] == "NO_GO"
    assert v["uplift_vwap_minus_control"] < vw.GATES["min_uplift_vs_matched_control"]
    # NOT a bug that zeroes returns: a real gross trend signal exists (else NO_GO would
    # be meaningless). The verdict comes from cost + control-redundancy, not suppression.
    assert v["frictionless_sharpe"] > 0.3
