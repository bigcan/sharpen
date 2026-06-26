"""Tripwire tests for the SMB VWAP-scalp + MACD-scalp falsification probe.

The probe's NO_GO is load-bearing, so these pin the properties an adversarial
auditor would attack:

  1. **Causal entry (no same-bar leak)** — a signal at bar i must execute at bar
     i+1's OPEN, never at the signal bar. Reverting the i+1 in `simulate` fails here.
  2. **Stop-before-target on a straddling bar** — when one bar contains BOTH the
     stop and the target, the STOP fills (adverse/conservative). A bug that books
     the target on straddles would inflate every bracket strategy's win rate.
  3. **Session-flat** — a position that resolves neither stop nor target before the
     session's last bar exits there ("session_end"), never carrying overnight.
  4. **Cost monotonicity** — higher round-trip cost => weakly lower net expectancy
     and Sharpe. Confirms cost is actually charged.
  5. **Win-rate-trap mechanic** — tightening the TP (tp_mult<1) moves the target
     closer to entry (=> higher win rate, the thing the videos sell). Pins that the
     trap demonstration is wired correctly.
  6. **Metrics are not suppressed** — a clearly-profitable synthetic trade set
     reports POSITIVE expectancy/Sharpe, so the NO_GO is data-driven, not a bug.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.research.smb_macd_scalp_eval as sc


def _df(bars):
    """bars: list of (open,high,low,close,vol). One synthetic session unless dates given."""
    n = len(bars)
    ts = pd.date_range("2025-01-01 09:30", periods=n, freq="5min")
    d = pd.DataFrame(bars, columns=["open", "high", "low", "close", "volume"])
    d["timestamp"] = ts
    d["date"] = d["timestamp"].dt.date
    d["hour"] = d["timestamp"].dt.hour
    d["tp"] = (d["high"] + d["low"] + d["close"]) / 3.0
    return d


# --- 1. causal entry at next-bar open --------------------------------------------
def test_entry_is_next_bar_open():
    d = _df([(100, 101, 99, 100, 1)] * 8)
    sess_id = d["date"].to_numpy()
    sess_last = np.zeros(len(d), dtype=bool)
    sess_last[-1] = True
    # long signal at i=2; wide stop/target so it survives to session end
    tr = sc.simulate(d, [{"i": 2, "side": 1, "stop": 50, "target": 200}], sess_id, sess_last)
    assert len(tr) == 1
    assert tr["entry_i"].iloc[0] == 3                     # i+1, not i
    assert tr["entry"].iloc[0] == d["open"].iloc[3]       # filled at the OPEN


# --- 2. stop fills before target on a straddling bar -----------------------------
def test_stop_before_target_on_straddle():
    # entry at bar1 open=100; bar2 spans low=90 (hits stop 95) AND high=110 (hits tgt 105)
    d = _df([(100, 100, 100, 100, 1), (100, 100, 100, 100, 1), (100, 110, 90, 100, 1),
             (100, 101, 99, 100, 1)])
    sess_id = d["date"].to_numpy()
    sess_last = np.zeros(len(d), dtype=bool)
    sess_last[-1] = True
    tr = sc.simulate(d, [{"i": 0, "side": 1, "stop": 95, "target": 105}], sess_id, sess_last)
    assert tr["reason"].iloc[0] == "stop"                 # NOT "target"
    assert tr["gross_ret"].iloc[0] < 0


# --- 3. flat at session end (no overnight carry) ---------------------------------
def test_session_flat_exit():
    d = _df([(100, 101, 99, 100, 1)] * 6)
    d.loc[3:, "date"] = pd.Timestamp("2025-01-02").date()  # second session from bar 3
    sess_id = d["date"].to_numpy()
    sess_last = np.zeros(len(d), dtype=bool)
    sess_last[2] = True                                    # last bar of session 1
    sess_last[5] = True
    # signal bar0 -> entry bar1; never hits stop/target -> must exit at bar2 (session end)
    tr = sc.simulate(d, [{"i": 0, "side": 1, "stop": 1, "target": 1e9}], sess_id, sess_last)
    assert tr["reason"].iloc[0] == "session_end"
    assert tr["exit_i"].iloc[0] == 2                       # not carried into session 2


# --- 4. cost monotonicity --------------------------------------------------------
def test_cost_monotonic():
    rng = np.random.default_rng(0)
    g = rng.normal(0.0005, 0.01, 300)
    tr = pd.DataFrame({"gross_ret": g, "r_mult": g / 0.01, "r_risk": 0.01,
                       "bars_held": 5})
    m0 = sc.trade_metrics(tr, 0.0, 1.0, None)
    m1 = sc.trade_metrics(tr, 0.0005, 1.0, None)
    assert m1["expectancy_bps"] < m0["expectancy_bps"]
    assert m1["net_sharpe"] <= m0["net_sharpe"]


# --- 5. win-rate-trap mechanic: tighter TP => closer target ----------------------
def test_tighter_tp_moves_target_closer():
    # upward VWAP flip at bar t: build prices so close crosses a constant VWAP proxy
    bars = [(100, 100.5, 99.5, 99.0, 1)] * 6 + [(99, 103, 99, 102, 5)] + [(102, 103, 101, 102, 1)] * 3
    d = _df(bars)
    active = pd.Series(True, index=d.index)
    vwap = pd.Series(100.0, index=d.index)                # flat VWAP; close flips below->above at t=6
    s_full = sc.sig_vwap_flip(d, vwap, active, swing=3, body_k=0.0, tp_mult=1.0)
    s_tight = sc.sig_vwap_flip(d, vwap, active, swing=3, body_k=0.0, tp_mult=0.33)
    assert s_full and s_tight
    e = d["open"].iloc[s_full[0]["i"] + 1]
    # tighter TP target is strictly closer to entry than the full-1R target
    assert abs(s_tight[0]["target"] - e) < abs(s_full[0]["target"] - e)


# --- 6. metrics not suppressed (a real winner reports positive) ------------------
def test_positive_trades_report_positive():
    tr = pd.DataFrame({"gross_ret": np.full(200, 0.002), "r_mult": np.full(200, 0.2),
                       "r_risk": np.full(200, 0.01), "bars_held": np.full(200, 5)})
    m = sc.trade_metrics(tr, 0.0, 1.0, None)
    assert m["expectancy_bps"] > 0
    assert m["net_pf"] > 1.0
    assert m["win_rate"] == 1.0


# --- 7b. pure-MACD position stream is single-bar causal (no double-lag) -----------
def test_pos_metrics_single_bar_alignment():
    # pos[t] (already shift-aligned to "held during bar t") must earn ret[t], NOT ret[t+1].
    # Reverting pos_metrics to `np.r_[ret[1:],0]` (the double-lag bug) fails this.
    d = _df([(100, 100, 100, 100, 1), (101, 101, 101, 101, 1), (103, 103, 103, 103, 1),
             (106, 106, 106, 106, 1), (110, 110, 110, 110, 1), (115, 115, 115, 115, 1)])
    pos = pd.Series([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], index=d.index)   # long held only during bar 3
    net = sc.pos_metrics(d, pos, 0.0, 1.0)["_net"]
    ret3 = (106 - 103) / 103          # bar-3 return (decision known at close[2])
    ret4 = (110 - 106) / 106          # bar-4 return (what the double-lag bug would book)
    assert abs(net[3] - ret3) < 1e-9
    assert abs(net[3] - ret4) > 1e-4


# --- 7. deflated-Sharpe helpers are sane -----------------------------------------
def test_deflation_helpers():
    assert sc.expected_max_sr(10) > sc.expected_max_sr(2) > 0      # more trials => higher null max
    assert abs(sc._phi(0.0) - 0.5) < 1e-9
    assert abs(sc._z(0.975) - 1.959964) < 1e-3                     # inverse-normal accuracy
