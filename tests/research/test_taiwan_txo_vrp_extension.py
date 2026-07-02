"""Tripwires for the TXO VRP backward-extension confirmatory test.

Guard the load-bearing plumbing: period-correct schedule boundaries, hedge-loop equivalence
with the audited stage-1 loop, entry/expiry causality (LEAK-2 class), the pick-then-filter
volume floor, the settlement hole-fill cross-check, NW-t, and the gap stress.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.fetch_taiwan_options_finmind_ext import fill_settlement_holes  # noqa: E402
from scripts.research.taiwan_txo_vrp_extension import (  # noqa: E402
    build_cycles_ext, gap_stress, hedge_cycle_tx_ext, newey_west_t, sched_rate,
)
from scripts.research.taiwan_txo_vrp_validation import hedge_cycle_tx  # noqa: E402

TAX_SCHED = [
    {"through": "2005-12-31", "rate": 0.00025},
    {"through": "2008-10-05", "rate": 0.0001},
    {"through": "2013-03-31", "rate": 0.00004},
    {"through": "9999-12-31", "rate": 0.00002},
]


# --------------------------------------------------------------------------- #
# 1. period-correct schedule boundaries
# --------------------------------------------------------------------------- #
def test_sched_rate_boundaries():
    assert sched_rate(TAX_SCHED, pd.Timestamp("2005-12-31"), "rate") == 0.00025
    assert sched_rate(TAX_SCHED, pd.Timestamp("2006-01-01"), "rate") == 0.0001
    assert sched_rate(TAX_SCHED, pd.Timestamp("2008-10-05"), "rate") == 0.0001
    assert sched_rate(TAX_SCHED, pd.Timestamp("2008-10-06"), "rate") == 0.00004
    assert sched_rate(TAX_SCHED, pd.Timestamp("2013-03-31"), "rate") == 0.00004
    assert sched_rate(TAX_SCHED, pd.Timestamp("2013-04-01"), "rate") == 0.00002
    assert sched_rate(TAX_SCHED, pd.Timestamp("2026-01-01"), "rate") == 0.00002


# --------------------------------------------------------------------------- #
# 2. hedge loop == audited stage-1 loop (pnl & turnover), on random paths
# --------------------------------------------------------------------------- #
def test_hedge_loop_equivalence_with_stage1():
    rng = np.random.default_rng(3)
    for _ in range(5):
        n = int(rng.integers(8, 25))
        dates = pd.bdate_range("2010-01-04", periods=n).to_numpy()
        f = 8000 * np.cumprod(1 + rng.normal(0, 0.01, n))
        expiry = pd.Timestamp(dates[-1])
        k, sc, sp = float(f[0].round(-2)), 0.2, 0.22
        pnl0, turn0 = hedge_cycle_tx(dates, f, expiry, k, sc, sp, 0.0014)
        pnl1, turn1, cost, h_last = hedge_cycle_tx_ext(
            dates, f, expiry, k, sc, sp, 0.0014, np.full(n, 1.0))
        assert math.isclose(pnl0, pnl1, rel_tol=0, abs_tol=1e-9)
        assert math.isclose(turn0, turn1, rel_tol=0, abs_tol=1e-9)
        # flat unit cost 1.0 => cost == turnover
        assert math.isclose(cost, turn1, rel_tol=0, abs_tol=1e-9)
        assert np.isfinite(h_last)


def _mini_market(settle_regime_gap_days: int = 0):
    """One clean synthetic monthly cycle. settle_regime_gap_days=1 emulates the pre-2008
    Thursday-settlement regime (TX path ends the day BEFORE the settle date)."""
    dates = pd.bdate_range("2005-01-03", "2005-03-31")
    px = pd.Series(6000.0 + np.arange(len(dates)) * 2.0, index=dates)
    spot = pd.DataFrame({"date": dates, "close": px.values})
    entry, expiry_trade = dates[5], dates[25]
    expiry = dates[25 + settle_regime_gap_days]
    s0 = float(px.iloc[5])
    st = float(px.iloc[25 + settle_regime_gap_days]) + 1.0
    cm = "200502"
    k = round(s0 / 100) * 100
    chains = pd.DataFrame([
        {"contract_month": cm, "entry_date": entry, "entry_spot": s0, "strike": float(k),
         "call_put": cp, "close": 90.0, "volume": 500, "oi": 100} for cp in ("call", "put")])
    settle = pd.DataFrame({"contract_month": [cm], "settle_date": [expiry],
                           "settle_price": [st]})
    seg = dates[(dates >= entry) & (dates <= expiry_trade)]
    tx = pd.DataFrame({"date": seg, "contract_month": cm,
                       "close": px.loc[seg].values - 3.0,
                       "high": px.loc[seg].values, "low": px.loc[seg].values - 6.0,
                       "settlement_price": 0.0})
    costs = {"half_spread_pts_per_leg": 1.0, "fee_pts_per_leg": 0.4,
             "tax_rate_on_premium": 0.001, "hedge_half_spread_pts": 1.0,
             "vol_floor": 0.0014, "min_path_points": 5, "min_atm_leg_volume": 10,
             "dte_min": 10, "dte_max": 45}
    return chains, settle, tx, spot, costs, cm, k, st


# --------------------------------------------------------------------------- #
# 3. causality / LEAK-2 class
# --------------------------------------------------------------------------- #
def test_entry_leg_immune_to_settlement_change():
    chains, settle, tx, spot, costs, cm, k, st = _mini_market()
    a = build_cycles_ext(chains, settle, tx, spot, costs, None, 252, "entry_iv")
    settle2 = settle.assign(settle_price=st + 500.0)
    b = build_cycles_ext(chains, settle2, tx, spot, costs, None, 252, "entry_iv")
    assert len(a) == 1 and len(b) == 1
    # entry-determined quantities identical; only payoff-side moves
    for col in ("premium", "strike", "iv", "spot0"):
        assert a.loc[0, col] == b.loc[0, col]
    assert a.loc[0, "intrinsic"] != b.loc[0, "intrinsic"]


def test_hedge_path_never_exceeds_expiry_and_old_regime_final_mark():
    chains, settle, tx, spot, costs, cm, k, st = _mini_market(settle_regime_gap_days=1)
    out = build_cycles_ext(chains, settle, tx, spot, costs, None, 252, "entry_iv")
    assert len(out) == 1
    # old regime: final mark is the settlement print itself
    assert out.loc[0, "f_final"] == pytest.approx(st)
    # and nothing after expiry leaks in: shifting post-expiry TX prices changes nothing
    tx2 = tx.copy()
    extra = pd.DataFrame({"date": [settle.loc[0, "settle_date"] + pd.Timedelta(days=1)],
                          "contract_month": [cm], "close": [9999.0], "high": [9999.0],
                          "low": [9999.0], "settlement_price": [0.0]})
    tx2 = pd.concat([tx2, extra], ignore_index=True)
    out2 = build_cycles_ext(chains, settle, tx2, spot, costs, None, 252, "entry_iv")
    assert out2.loc[0, "net_ret"] == pytest.approx(out.loc[0, "net_ret"])


# --------------------------------------------------------------------------- #
# 4. volume floor: pick-then-filter (never re-picks a different strike)
# --------------------------------------------------------------------------- #
def test_volume_floor_drops_cycle_not_repicks():
    chains, settle, tx, spot, costs, cm, k, st = _mini_market()
    # add a liquid FAR strike; make the ATM strike ILLIQUID
    far = chains.copy()
    far["strike"] = far["strike"] + 400.0
    far["volume"] = 5000
    chains.loc[:, "volume"] = 1  # ATM legs below the floor of 10
    both = pd.concat([chains, far], ignore_index=True)
    out = build_cycles_ext(both, settle, tx, spot, costs, None, 252, "entry_iv")
    assert len(out) == 0  # dropped — NOT re-picked to the liquid far strike
    both.loc[both["strike"] == float(k), "volume"] = 50
    out2 = build_cycles_ext(both, settle, tx, spot, costs, None, 252, "entry_iv")
    assert len(out2) == 1 and out2.loc[0, "strike"] == float(k)


# --------------------------------------------------------------------------- #
# 5. NW t
# --------------------------------------------------------------------------- #
def test_newey_west_t():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 1.0, 4000)
    t_nw = newey_west_t(x, lags=3)
    t_iid = float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))
    assert abs(t_nw - t_iid) / t_iid < 0.05      # iid: NW ~= plain t
    # positively autocorrelated series: NW must shrink the t
    e = rng.normal(0, 1.0, 4000)
    y = np.empty(4000)
    y[0] = e[0]
    for i in range(1, 4000):
        y[i] = 0.6 * y[i - 1] + e[i]
    y = y + 0.5
    assert abs(newey_west_t(y, lags=10)) < abs(
        float(y.mean() / (y.std(ddof=1) / math.sqrt(len(y)))))


# --------------------------------------------------------------------------- #
# 6. gap stress arithmetic (hand-checked)
# --------------------------------------------------------------------------- #
def test_gap_stress_hand_example():
    cyc = pd.DataFrame([{
        "contract_month": "200502", "net_pts": 50.0, "h_final": 0.5, "f_final": 6000.0,
        "settle": 6000.0, "strike": 6000.0, "intrinsic": 0.0, "spot0": 6000.0,
        "net_ret": 50.0 / 6000.0}])
    res = gap_stress(cyc, 0.10)
    # -10%: hedge 0.5*6000*(-0.1) = -300 ; intrinsic 600-0 ; net = 50-300-600 = -850
    assert res["worst_net_ret"] == pytest.approx(-850.0 / 6000.0, abs=1e-4)
    assert res["worst_direction"] == "-10%"


# --------------------------------------------------------------------------- #
# 7. settlement hole-fill cross-check (halts on mismatch)
# --------------------------------------------------------------------------- #
def test_fill_settlement_holes_and_halt():
    months = ["201112", "201201", "201202"]
    exp = pd.to_datetime(["2011-12-21", "2012-01-18", "2012-02-15"])
    px = [7000.0, 7100.0, 7200.0]
    txo = pd.DataFrame({"contract_month": months[:1], "settle_date": exp[:1],
                        "settle_price": px[:1]})
    tx_fs = pd.DataFrame({"contract_month": months, "settle_date": exp, "settle_price": px})
    filled, st = fill_settlement_holes(txo, tx_fs, 1.0, 0.98)
    assert len(filled) == 3 and st["n_filled_from_tx"] == 2 and st["match_rate"] == 1.0
    assert (filled.sort_values("settle_date")["source"].tolist()
            == ["txo", "tx_fill", "tx_fill"])
    with pytest.raises(RuntimeError):
        fill_settlement_holes(txo.assign(settle_price=px[0] + 99.0), tx_fs, 1.0, 0.98)
