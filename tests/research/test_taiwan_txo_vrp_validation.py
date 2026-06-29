"""Tripwires for the fuller TXO VRP validation (cont-87 NEXT plan).

Locks the load-bearing correctness properties of
``scripts/research/taiwan_txo_vrp_validation.py`` — each test fails if the property is
reverted (mutation tripwire), per the Audit skill's negative-test requirement:

  1. BS legs / IV inversion / forward-delta identity (the hedge ratio math).
  2. Hedge-sign: a long hedge ratio on a RISING path makes positive hedge PnL
     (flip the sign in straddle_delta_legs and this breaks).
  3. Instrument realism: the documented finding — hedging the basis-free SPOT path
     scores materially higher than a basis-noised (real-future-like) path. This is the
     whole reason the scout GO downgrades to NO-GO; if the script silently reverts to a
     spot hedge the gap collapses and this test fails.
  4. Causality: a cycle whose entry is NOT strictly before settlement is dropped.
  5. Gate wiring: a require_all name not produced by the checks dict raises (no silent
     pass of an unmeasured gate).
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from scripts.research import taiwan_txo_vrp_validation as V

_N = NormalDist().cdf

# minimal valid config blocks (mirror configs/taiwan_txo_vrp_validation.gates.yaml shape)
_COSTS = {"half_spread_pts_per_leg": 1.0, "fee_pts_per_leg": 0.4, "tax_rate_on_premium": 0.001,
          "hedge_half_spread_pts": 1.0, "vol_floor": 0.0014}
_GATE = {"net_sharpe_floor": 0.5, "oos_net_sharpe_min": 0.0, "bootstrap_p_max": 0.05,
         "max_equity_beta_abs": 0.15, "vrp_positive_frac_min": 0.50,
         "delta_robust_min_sharpe": 0.5, "dsr_min": 0.95}
_DEFL = {"n_trials_gate": 50, "n_trials_report": [10, 50]}
_NM = {"n_boot": 2000}
_SCOUT = {"oos_split_frac": 0.5, "rv_window_trading_days": 252}
_REQUIRE = ["net_sharpe_above_floor", "oos_net_sharpe_positive", "beats_bootstrap_null",
            "equity_beta_below_ceiling", "vrp_premium_exists", "delta_models_robust",
            "dsr_above_floor"]


def test_bs_parity_and_iv_roundtrip():
    s, k, tau, sig = 15000.0, 15000.0, 30 / 365, 0.18
    c, p = V.bs_call_put(s, k, tau, sig)
    assert abs((c - p) - (s - k)) < 1e-6                 # put-call parity, r=q=0
    assert abs(V.implied_vol_leg(c, s, k, tau, True) - sig) < 1e-4
    assert abs(V.implied_vol_leg(p, s, k, tau, False) - sig) < 1e-4


def test_forward_delta_identity_and_signs():
    s, k, tau, sig = 15000.0, 15000.0, 30 / 365, 0.18
    d1 = (math.log(s / k) + 0.5 * sig * sig * tau) / (sig * math.sqrt(tau))
    assert abs(V.straddle_delta_legs(s, k, tau, sig, sig) - (2 * _N(d1) - 1)) < 1e-9
    # spot above strike => call side dominates => long-straddle delta > 0
    assert V.straddle_delta_legs(15500.0, k, tau, sig, sig) > 0
    assert V.straddle_delta_legs(14500.0, k, tau, sig, sig) < 0
    assert V.straddle_delta_legs(15500.0, k, 0.0, sig, sig) == 1.0   # expiry indicator


def test_hedge_pnl_sign_on_rising_path():
    """Long hedge units on a strictly rising future path => positive hedge PnL.
    (Negating the return in straddle_delta_legs would flip this.)"""
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
    path = np.array([15000.0, 15100.0, 15200.0, 15300.0, 15400.0])
    hedge_pnl, turnover = V.hedge_cycle_tx(dates.to_numpy(), path, pd.Timestamp("2024-02-01"),
                                           14000.0, 0.18, 0.18, 0.0014)   # deep ITM call => H>0
    assert hedge_pnl > 0
    assert turnover > 0


def _toy_panel(basis_noise_std: float, seed: int):
    """A synthetic option panel + a hedge-instrument path that is spot + optional basis noise."""
    dates = pd.bdate_range("2020-01-02", "2024-12-31")
    rng = np.random.default_rng(seed)
    px = 15000 * np.cumprod(1 + rng.normal(0.0002, 0.008, len(dates)))
    spot = pd.DataFrame({"date": dates, "close": px})
    fpx = px * (1.0 + rng.normal(0.0, basis_noise_std, len(dates)))    # futures-like basis noise
    fut = pd.DataFrame({"date": dates, "close": fpx, "high": fpx * 1.001, "low": fpx * 0.999})
    rows_c, rows_s = [], []
    for m in pd.date_range("2020-02-01", "2024-12-01", freq="MS"):
        entry = dates[dates >= m][0]
        exp = dates[dates >= (m + pd.Timedelta(days=28))][0]
        s0 = float(spot.loc[spot["date"] == entry, "close"].iloc[0])
        st = float(spot.loc[spot["date"] == exp, "close"].iloc[0])
        cm = m.strftime("%Y%m")
        k = round(s0 / 100) * 100
        prem = 0.030 * s0
        rows_c += [{"contract_month": cm, "entry_date": entry, "entry_spot": s0, "strike": float(k),
                    "call_put": cp, "close": prem / 2, "volume": 1, "oi": 1}
                   for cp in ("call", "put")]
        rows_s.append({"contract_month": cm, "settle_date": exp, "settle_price": st})
    return pd.DataFrame(rows_c), pd.DataFrame(rows_s), spot, fut


def test_instrument_realism_basis_degrades_sharpe():
    """The documented finding: a basis-free hedge path scores materially higher than a
    basis-noised (real-future-like) one. Locks why the scout GO downgrades."""
    chains, settle, spot, _ = _toy_panel(0.0, seed=1)
    cyc_clean = V.build_cycles(chains, settle,
                               spot.assign(high=spot["close"], low=spot["close"]),
                               spot, _COSTS, 252, "entry_iv")
    chains2, settle2, spot2, fut2 = _toy_panel(0.004, seed=1)
    cyc_basis = V.build_cycles(chains2, settle2, fut2, spot2, _COSTS, 252, "entry_iv")
    sh = lambda c: V.sharpe(c["net_ret"].to_numpy(float), 12.0)   # noqa: E731
    assert sh(cyc_clean) > sh(cyc_basis) + 0.1


def test_causality_drops_noncausal_cycle():
    """entry_date not strictly before settle_date => cycle excluded."""
    entry = pd.Timestamp("2024-02-01")
    days = pd.bdate_range("2024-01-01", "2024-03-01")
    spot = pd.DataFrame({"date": days, "close": np.linspace(15000, 15500, len(days))})
    fut = spot.assign(high=spot["close"], low=spot["close"])
    chains = pd.DataFrame([{"contract_month": "202402", "entry_date": entry, "entry_spot": 15300.0,
                            "strike": 15300.0, "call_put": cp, "close": 150.0, "volume": 1, "oi": 1}
                           for cp in ("call", "put")])
    settle = pd.DataFrame([{"contract_month": "202402", "settle_date": entry,   # == entry, NOT after
                            "settle_price": 15300.0}])
    cyc = V.build_cycles(chains, settle, fut, spot, _COSTS, 252, "entry_iv")
    assert len(cyc) == 0


def test_gate_wiring_rejects_unknown_check():
    """A require_all name with no matching check must raise (no silent unmeasured gate)."""
    chains, settle, spot, fut = _toy_panel(0.0, seed=3)
    modes = V._build_all_modes(chains, settle, fut, spot, _COSTS, 252)
    import tempfile
    from pathlib import Path
    with pytest.raises(KeyError):
        V.evaluate(modes, _GATE, _DEFL, _NM, _SCOUT, _REQUIRE + ["bogus_gate"], 12.0, 0.0,
                   Path(tempfile.mkdtemp()))


def test_min_path_points_gates_short_cycles():
    """min_path_points controls how many daily marks a cycle needs to be hedged/kept.
    A 4-trading-day cycle survives at min_path=3 but is dropped at min_path=5 (weekly relevance)."""
    days = pd.bdate_range("2024-01-02", "2024-01-31")
    spot = pd.DataFrame({"date": days, "close": np.linspace(15000, 15200, len(days))})
    fut = spot.assign(high=spot["close"], low=spot["close"])
    entry, exp = pd.Timestamp("2024-01-15"), pd.Timestamp("2024-01-18")   # ~4 trading days
    chains = pd.DataFrame([{"contract_month": "202403W2", "entry_date": entry, "entry_spot": 15100.0,
                            "strike": 15100.0, "call_put": cp, "close": 90.0, "volume": 1, "oi": 1}
                           for cp in ("call", "put")])
    settle = pd.DataFrame([{"contract_month": "202403W2", "settle_date": exp, "settle_price": 15120.0}])
    keep = V.build_cycles(chains, settle, fut, spot, {**_COSTS, "min_path_points": 3}, 252, "entry_iv")
    drop = V.build_cycles(chains, settle, fut, spot, {**_COSTS, "min_path_points": 5}, 252, "entry_iv")
    assert len(keep) == 1 and len(drop) == 0


def test_selftest_runs():
    assert V._selftest(_COSTS, _GATE, _DEFL, _NM, _SCOUT,
                       {"session_start": "08:45", "session_end": "13:45"},
                       252, 12.0, _REQUIRE) == 0
