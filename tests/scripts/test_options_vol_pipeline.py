"""Tests for the Phase-5 Options-VRP pipeline (``scripts/options_vol_pipeline.py``).

The KEYSTONE is ``test_linear_core_through_pipeline_matches_falsification``: the
gate baseline that ``evaluate_linear_core`` feeds into ``rl_beats_linear`` must
reproduce the validated Phase-1 linear short-straddle core (the BTC net-Sharpe-1.10
number in ``results/options_vrp/verdict.json``) through the FULL pipeline wrappers
(``make_options_env`` → ``_drive`` → metric helpers), not merely inside the env. A
future refactor of any wrapper that silently broke the baseline would let the gate
compare RL against a wrong number — this fails loud. The remaining tests guard the
WF slicing causality (test never overlaps its own train; embargo gaps present), the
config→env mapping + HPO override application, and that the frictionless probe truly
zeroes the trading frictions.

All data is synthetic and deterministic — no network, no cached parquet, no SB3.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from finrl_pro_ds.crypto import options_pricing as op
from finrl_pro_ds.crypto import options_vrp_sim as ovs  # sim core promoted from the script (MS-ADR-9)

_ROOT = Path(__file__).resolve().parents[2]


def _load(mod_name: str, rel: str):
    spec = importlib.util.spec_from_file_location(mod_name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# scripts/ is not a package — load the pipeline + the falsification oracle by path.
ovp = _load("ovp_pipeline_test", "scripts/options_vol_pipeline.py")
fals = _load("ovp_fals_oracle_test", "scripts/research/options_vrp_falsification.py")


# Identical synthetic generator to tests/envs/test_options_vol_harvest.py.
def synthetic_series(T=500, seed=3, iv_level=0.65, vol=0.55):
    rng = np.random.default_rng(seed)
    dt = 1.0 / op.ANN
    rets = rng.normal(0.0, vol * math.sqrt(dt), size=T)
    spot = 30_000.0 * np.exp(np.cumsum(rets))
    iv = np.empty(T)
    iv[0] = iv_level
    for t in range(1, T):
        iv[t] = iv[t - 1] + 0.15 * (iv_level - iv[t - 1]) + rng.normal(0, 0.01)
    iv = np.clip(iv, 0.10, 2.0)
    funding = np.full(T, 0.0001)
    return spot, iv, funding


def _arrays(spot, iv, funding):
    T = len(spot)
    return {
        "spot": spot, "iv": iv, "iv_rv_spread": np.zeros(T),
        "rv_ref": np.full(T, 0.5), "funding": funding,
        "timestamps": np.arange(T, dtype=np.int64),
    }


# Minimal config mirroring the env defaults / Phase-1 SimConfig.
CFG = {"env": {
    "type": "options_vol_harvest", "initial_capital": 100_000.0,
    "base_premium_frac": 0.05, "max_premium_frac": 0.10, "roll_days": 21,
    "entry_tenor_days": 30, "option_fee_pct_underlying": 0.0003,
    "option_fee_cap_pct_premium": 0.125, "perp_taker_fee": 0.0005,
    "option_spread_vol_pts": 1.0, "reward_type": "dsr_cvar",
}}


# ---------------------------------------------------------------------------
# KEYSTONE: the pipeline's frozen linear core == the Phase-1 falsification core
# ---------------------------------------------------------------------------
def test_linear_core_through_pipeline_matches_falsification():
    spot, iv, funding = synthetic_series(T=500, seed=3)

    ref = fals.simulate_asset(spot, iv, funding, fals.SimConfig())
    ref_sharpe = fals._sharpe(fals._returns_from_pnl(ref["daily_pnl"], ref["eq_curve"]))
    ref_pf = fals._profit_factor(ref["daily_pnl"])
    ref_dd = fals._max_drawdown(ref["eq_curve"])

    core = ovp.evaluate_linear_core(_arrays(spot, iv, funding), CFG)

    assert abs(core["net_sharpe"] - ref_sharpe) < 1e-6
    assert abs(core["net_pf"] - ref_pf) < 1e-6
    assert abs(core["net_max_dd"] - ref_dd) < 1e-9


def test_ann_is_365():
    """The Sharpe annualization must match options_pricing / the falsification (365),
    NOT the allocator's 252 — else the RL net Sharpe is on a different scale than the
    documented 1.10 core and the gate is mis-anchored."""
    assert ovp.ANN == op.ANN == 365.0


# ---------------------------------------------------------------------------
# Walk-forward slicing causality: test never overlaps its own train; embargoed.
# ---------------------------------------------------------------------------
def test_wf_schedule_test_disjoint_from_train_and_embargoed():
    dates = pd.date_range("2021-04-01", periods=1892, freq="D", tz="UTC")
    wf = {"train_bars": 730, "val_bars": 180, "test_bars": 180,
          "step_bars": 180, "embargo_bars": 14}
    sched = ovp.build_wf_schedule(dates, wf)
    assert len(sched) >= 4
    for w in sched:
        # embargo gaps present (val after train, test after val)
        assert w["va_s"] == w["tr_e"] + 14
        assert w["te_s"] == w["va_e"] + 14
        # the load-bearing WF property: a window's TEST is strictly after its own
        # TRAIN end (no train↔test overlap → no in-sample leakage into the verdict)
        assert w["te_s"] > w["tr_e"]
        assert w["tr_s"] < w["tr_e"] <= w["va_s"] < w["va_e"] <= w["te_s"] < w["te_e"]


def test_slice_arrays_is_contiguous_forward_slice():
    full = _arrays(*synthetic_series(T=300, seed=1))
    sl = ovp.slice_arrays(full, 100, 200)
    for k in ("spot", "iv", "funding", "timestamps"):
        assert len(sl[k]) == 100
        np.testing.assert_array_equal(sl[k], full[k][100:200])


# ---------------------------------------------------------------------------
# config → env mapping + HPO override application
# ---------------------------------------------------------------------------
def test_make_options_env_maps_config_and_applies_overrides():
    spot, iv, funding = synthetic_series(T=80, seed=2)
    env = ovp.make_options_env(
        _arrays(spot, iv, funding), CFG,
        overrides={"turnover_penalty": 0.007, "cvar_penalty": 0.9}, eval_mode=True)
    assert env.roll_days == 21
    assert abs(env.base_premium_frac - 0.05) < 1e-12
    assert abs(env.turnover_penalty - 0.007) < 1e-12   # HPO override wins over config
    assert abs(env.cvar_penalty - 0.9) < 1e-12
    assert env.random_start is False                   # eval_mode forces determinism


# ---------------------------------------------------------------------------
# frictionless probe (g_cost_gap / AlphaSeek artifact detector) truly zeroes costs
# ---------------------------------------------------------------------------
def test_frictionless_overrides_zero_all_costs():
    spot, iv, funding = synthetic_series(T=120, seed=5)
    arr = _arrays(spot, iv, funding)
    fric_arrays = {**arr, "funding": np.zeros_like(arr["funding"])}
    env = ovp.make_options_env(fric_arrays, CFG,
                               overrides=dict(ovp._FRICTIONLESS_OVERRIDES), eval_mode=True)
    neutral = np.array([env.neutral_conviction, 0.0])
    env.reset()
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(neutral)
        done = term or trunc
    # zero option fee + zero spread + zero perp taker ⇒ no fees ever booked
    assert env.cumulative_fees == 0.0


# ---------------------------------------------------------------------------
# V3-03 regression: the close-fee straddle_price(S, K, sigma, tau) arg order. A swap
# is INERT at the shipped 21d config (the premium cap never binds) but corrupts the
# fee the instant the cap binds — the latent unit landmine. These lock the fix and
# document why hundreds of green runs never surfaced it.
# ---------------------------------------------------------------------------
def test_straddle_price_arg_order_is_load_bearing():
    """sigma and tau are NOT interchangeable; the swapped call is a different price."""
    S, sigma, tau = 30_000.0, 0.60, 9.0 / op.ANN
    correct = op.straddle_price(S, S, sigma, tau)
    swapped = op.straddle_price(S, S, tau, sigma)   # the old falsification.py:144 order
    assert abs(correct - swapped) / correct > 0.5   # materially different


def test_close_fee_cap_binding_uses_bs_correct_value():
    """When the 12.5%-of-premium cap binds, the close fee must use the BS-correct
    straddle premium (sigma, tau). With the swapped args the 'premium' is a different
    number, so the booked cap would be wrong — the V3-03 landmine."""
    cfg = fals.SimConfig()
    # very short tenor + low vol => tiny straddle value => the 12.5%-of-premium cap
    # falls below the 0.06%-of-notional underlying fee => the cap is the binding branch.
    S, sigma, tau = 30_000.0, 0.08, 1.0 / op.ANN
    n = 5.0
    V_correct = op.straddle_price(S, S, sigma, max(tau, 1.0 / op.ANN))
    fee = ovs._option_fees(n, S, n * V_correct, cfg)
    fee_underlying = 2.0 * n * cfg.option_fee_pct_underlying * S
    fee_cap = cfg.option_fee_cap_pct_premium * n * V_correct
    assert fee_cap < fee_underlying                 # the cap is the binding branch
    assert abs(fee - fee_cap) < 1e-9                # fee uses the BS-correct premium
    # the swapped-arg premium would book a different cap => the bug is real once it binds
    V_swapped = op.straddle_price(S, S, max(tau, 1.0 / op.ANN), sigma)
    assert abs(cfg.option_fee_cap_pct_premium * n * V_swapped - fee_cap) > 1e-6


def test_close_fee_cap_does_not_bind_at_shipped_aged_straddle():
    """Why the swap was inert in the shipped verdict: at a 21d roll the closed straddle
    still has ~9-30d left, so the underlying fee is well below the premium cap — the cap
    never binds, so the V that only feeds it is dead. Reproduces 'cap binds 0/90'."""
    cfg = fals.SimConfig()
    S, sigma, tau = 30_000.0, 0.60, 9.0 / op.ANN     # an aged straddle at a 21d roll
    n = 5.0
    V = op.straddle_price(S, S, sigma, tau)
    fee_underlying = 2.0 * n * cfg.option_fee_pct_underlying * S
    fee_cap = cfg.option_fee_cap_pct_premium * n * V
    assert fee_underlying < fee_cap                  # underlying branch wins => swap inert


def test_establishment_hedge_fee_booked_at_open():
    """V2-05: the first open books option fee + half-spread + the perp-hedge
    establishment taker fee (q_prev 0 -> initial delta hedge) — previously omitted."""
    spot, iv, funding = synthetic_series(T=120, seed=7)
    cfg = fals.SimConfig()
    res = fals.simulate_asset(spot, iv, funding, cfg)
    t0 = res["t0"]
    S, sigma, tau = spot[t0], iv[t0], cfg.entry_tenor_days / op.ANN
    unit = op.straddle_price(S, S, sigma, tau)
    n = cfg.premium_frac * cfg.initial_capital / unit
    opt_fee = ovs._option_fees(n, S, n * unit, cfg)
    spread = ovs._spread_cost(n, S, sigma, tau, cfg)
    est_fee = cfg.perp_taker_fee * abs(n * op.straddle_delta(S, S, sigma, tau)) * S
    assert est_fee > 0.0
    assert abs(res["daily_pnl"][t0] - (-(opt_fee + spread + est_fee))) < 1e-6
