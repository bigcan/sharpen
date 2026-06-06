"""Tests for OptionsVolHarvestEnv (crypto-options VRP harvester, Phase 3).

The KEYSTONE is ``test_baseline_parity_matches_linear_core``: driven by the NEUTRAL
action the env must reproduce the validated Phase-1 linear short-straddle book
(``scripts/research/options_vrp_falsification.simulate_asset``) bit-for-bit on
identical inputs — proving the options-MTM / cost / roll engine is the same one
that produced ``results/options_vrp/verdict.json``. Everything else (caps, reward,
causality) guards the RL machinery layered on top of that frozen core.

All data is synthetic and deterministic — no network, no cached parquet.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

from finrl_pro_ds.crypto import options_pricing as op
from finrl_pro_ds.crypto.envs.options_vol_harvest_env import OptionsVolHarvestEnv

# --- load the (non-package) research falsification module for the parity oracle --
_FALS_PATH = Path(__file__).resolve().parents[2] / "scripts" / "research" / "options_vrp_falsification.py"
_spec = importlib.util.spec_from_file_location("ovf_fals_test", _FALS_PATH)
fals = importlib.util.module_from_spec(_spec)
sys.modules["ovf_fals_test"] = fals
_spec.loader.exec_module(fals)


# ---------------------------------------------------------------------------
# Synthetic data: positive spot (GBM), mean-reverting IV kept above realised so a
# VRP exists, small funding. Returns 1-D float64 arrays of length T.
# ---------------------------------------------------------------------------
def synthetic_series(T=400, seed=0, iv_level=0.65, vol=0.55):
    rng = np.random.default_rng(seed)
    dt = 1.0 / op.ANN
    rets = rng.normal(0.0, vol * math.sqrt(dt), size=T)
    spot = 30_000.0 * np.exp(np.cumsum(rets))
    # IV mean-reverts around iv_level (> realised `vol` ⇒ harvestable premium)
    iv = np.empty(T)
    iv[0] = iv_level
    for t in range(1, T):
        iv[t] = iv[t - 1] + 0.15 * (iv_level - iv[t - 1]) + rng.normal(0, 0.01)
    iv = np.clip(iv, 0.10, 2.0)
    funding = np.full(T, 0.0001)  # 1 bp/day carry on the hedge
    return spot, iv, funding


def make_env(spot, iv, funding, **kwargs):
    T = len(spot)
    return OptionsVolHarvestEnv(
        spot=spot, iv=iv,
        iv_rv_spread=np.zeros(T), rv_ref=np.full(T, 0.5),
        funding=funding, timestamps=np.arange(T, dtype=np.int64),
        **kwargs,
    )


def run_env(env, action):
    """Roll the env to done; return reconstructed (daily_pnl, eq_curve) arrays
    in the EXACT convention of simulate_asset (daily_pnl[0] = -open_cost)."""
    _, info = env.reset()
    daily_pnl = [-info["initial_open_cost"]]
    eq_curve = [env.initial_capital]
    done = False
    while not done:
        _, _, term, trunc, info = env.step(action)
        daily_pnl.append(info["step_pnl"])
        eq_curve.append(info["portfolio_value"])
        done = term or trunc
    return np.array(daily_pnl), np.array(eq_curve)


# ---------------------------------------------------------------------------
# KEYSTONE: baseline parity
# ---------------------------------------------------------------------------
def test_baseline_parity_matches_linear_core():
    """Neutral action ⇒ env equity curve == simulate_asset's, bit-for-bit."""
    spot, iv, funding = synthetic_series(T=500, seed=3)

    ref = fals.simulate_asset(spot, iv, funding, fals.SimConfig())

    env = make_env(spot, iv, funding, reward_type="dsr_cvar")
    neutral = np.array([env.neutral_conviction, 0.0])
    env_pnl, env_eq = run_env(env, neutral)

    # Same length and the same per-bar equity to floating-point tolerance.
    assert len(env_eq) == len(ref["eq_curve"])
    np.testing.assert_allclose(env_eq, ref["eq_curve"], atol=1e-6, rtol=0)

    # And the headline metrics computed the identical way both match.
    ref_sharpe = fals._sharpe(fals._returns_from_pnl(ref["daily_pnl"], ref["eq_curve"]))
    env_sharpe = fals._sharpe(fals._returns_from_pnl(env_pnl, env_eq))
    assert abs(env_sharpe - ref_sharpe) < 1e-6


def test_neutral_conviction_reproduces_base_premium_frac():
    """The ADR-4 fixed point: neutral conviction maps to the linear-core size."""
    spot, iv, funding = synthetic_series(T=80, seed=1)
    env = make_env(spot, iv, funding)
    env.reset()
    env.equity = 100_000.0
    S, sigma = spot[0], iv[0]
    m = env._sized_position(env.neutral_conviction, S, sigma)
    prem_frac = abs(m) * op.straddle_price(S, S, sigma, env.entry_tau) / env.equity
    assert m > 0  # short straddle (positive m == short)
    assert abs(prem_frac - env.base_premium_frac) < 1e-9


# ---------------------------------------------------------------------------
# Hard risk caps (ADR-6) — clipping the realized position
# ---------------------------------------------------------------------------
def test_gross_premium_cap_binds():
    spot, iv, funding = synthetic_series(T=80, seed=2)
    env = make_env(spot, iv, funding, max_gross_premium_frac=0.02, max_net_vega_pct=99.0)
    env.reset()
    env.equity = 100_000.0
    S, sigma = spot[0], iv[0]
    # max short conviction would target premium_frac 0.10; cap is 0.02.
    m = env._sized_position(-1.0, S, sigma)
    prem_frac = abs(m) * op.straddle_price(S, S, sigma, env.entry_tau) / env.equity
    assert prem_frac <= 0.02 + 1e-9


def test_net_vega_cap_binds():
    spot, iv, funding = synthetic_series(T=80, seed=4)
    env = make_env(spot, iv, funding, max_gross_premium_frac=0.99, max_net_vega_pct=0.05)
    env.reset()
    env.equity = 100_000.0
    S, sigma = spot[0], iv[0]
    m = env._sized_position(-1.0, S, sigma)
    vega_pct = abs(m) * op.straddle_vega(S, S, sigma, env.entry_tau) / env.equity
    assert vega_pct <= 0.05 + 1e-9


def test_allow_long_vol_false_blocks_long():
    spot, iv, funding = synthetic_series(T=80, seed=6)
    env = make_env(spot, iv, funding, allow_long_vol=False)
    env.reset()
    env.equity = 100_000.0
    # positive conviction would be LONG vol; blocked ⇒ flat (m == 0).
    m = env._sized_position(0.8, spot[0], iv[0])
    assert abs(m) < 1e-12


# ---------------------------------------------------------------------------
# SHORT-ACCT: options-MTM, no notional_debt; short straddle earns theta in calm
# ---------------------------------------------------------------------------
def test_no_notional_debt_field():
    """SHORT-ACCT: short vol is pure MTM (premium − Δmark); no debt bookkeeping."""
    spot, iv, funding = synthetic_series(T=40, seed=8)
    env = make_env(spot, iv, funding)
    env.reset()
    assert not hasattr(env, "notional_debt")
    # equity is a single scalar account — never a (margin, debt) pair.
    assert isinstance(env.equity, float)


def test_short_straddle_earns_theta_in_calm():
    """Flat spot + IV at entry ⇒ a delta-hedged short straddle bleeds in time value
    (theta) to the seller: option_pnl > 0 most bars, equity drifts up net of cost."""
    T = 40
    spot = np.full(T, 30_000.0)                 # perfectly flat ⇒ no gamma loss
    iv = np.full(T, 0.60)                        # constant IV ⇒ pure theta decay
    funding = np.zeros(T)
    env = make_env(spot, iv, funding, perp_taker_fee=0.0, option_spread_vol_pts=0.0)
    env.reset()
    neutral = np.array([env.neutral_conviction, 0.0])
    option_pnls = []
    for _ in range(15):
        _, _, term, trunc, info = env.step(neutral)
        option_pnls.append(info["option_pnl"])
        if term or trunc:
            break
    assert np.mean(option_pnls) > 0.0           # seller earns theta
    assert env.equity > env.initial_capital * 0.999


def test_short_straddle_loses_in_vol_spike():
    """A large spot jump (gamma loss) makes the short straddle lose on that bar."""
    T = 40
    spot = np.full(T, 30_000.0)
    spot[20:] = 39_000.0                          # +30% jump at bar 20
    iv = np.full(T, 0.60)
    funding = np.zeros(T)
    env = make_env(spot, iv, funding, perp_taker_fee=0.0, option_spread_vol_pts=0.0)
    env.reset()
    neutral = np.array([env.neutral_conviction, 0.0])
    jump_pnl = None
    for t in range(25):
        _, _, term, trunc, info = env.step(neutral)
        if env.step_idx == 20:                    # the t->t+1 move into the jump bar
            jump_pnl = info["option_pnl"] + info["hedge_pnl"]
        if term or trunc:
            break
    assert jump_pnl is not None and jump_pnl < 0.0  # net short-gamma loss on the jump


# ---------------------------------------------------------------------------
# DSR + CVaR reward
# ---------------------------------------------------------------------------
def test_cvar_excess_warmup_and_gain_are_zero():
    spot, iv, funding = synthetic_series(T=40, seed=9)
    env = make_env(spot, iv, funding, cvar_alpha=0.05)
    env.returns_history = [0.001] * 5            # below warmup threshold (need >= 20)
    assert env._cvar_excess(-0.05) == 0.0
    # warm but the step is a GAIN ⇒ no penalty
    env.returns_history = list(np.random.default_rng(0).normal(0, 0.01, 60))
    assert env._cvar_excess(+0.05) == 0.0


def test_cvar_excess_penalizes_tail_loss():
    spot, iv, funding = synthetic_series(T=40, seed=10)
    env = make_env(spot, iv, funding, cvar_alpha=0.05)
    # a calm buffer of small returns; a -8% step is deep in the loss tail
    env.returns_history = list(np.random.default_rng(1).normal(0.0005, 0.008, 80))
    excess = env._cvar_excess(-0.08)
    assert excess > 0.0
    # 1/alpha scaling makes it O(0.1+), not O(0.001) — must actually bite vs DSR ~O(1)
    assert excess > 0.1


def test_cvar_penalty_lowers_reward_on_tail_loss():
    """The dsr_cvar reward = clip(DSR - cvar_penalty*excess) on a tail loss — strictly
    below the bare DSR. Wide clip range isolates the term from the [-5,5] floor; an
    independent DSR probe seeded to the env's PRE-compute state gives the bare DSR
    (calling _calc_reward mutates the env's own DSR EMA, so we must not read it after).
    """
    from finrl_pro_ds.envs.dsr import DSRCalculator

    buf = list(np.random.default_rng(2).normal(0.0005, 0.008, 80))
    spot, iv, funding = synthetic_series(T=40, seed=11)
    env = make_env(spot, iv, funding, reward_type="dsr_cvar", cvar_penalty=1.0,
                   reward_clip_range=(-100.0, 100.0))
    env.returns_history = list(buf)

    excess = env._cvar_excess(-0.08)               # does not mutate DSR
    assert excess > 0.0

    probe = DSRCalculator(eta=env.dsr_eta, scale=env.reward_scaling)
    probe._A, probe._B, probe._warmup = env._dsr._A, env._dsr._B, env._dsr._warmup
    dsr_base = probe.compute(-0.08)                # bare DSR from identical pre-state

    r_cvar = env._calc_reward(-0.08, turnover_frac=0.0)
    assert abs(r_cvar - (dsr_base - 1.0 * excess)) < 1e-9   # exactly the penalty applied
    assert r_cvar < dsr_base                                 # strictly lowers reward


# ---------------------------------------------------------------------------
# LEAK-2 / T+1 causality
# ---------------------------------------------------------------------------
def test_future_bar_perturbation_does_not_change_past_steps():
    """Bars > k+1 perturbed ⇒ steps 0..k-1 (whose t->t+1 moves never touch them)
    are byte-identical. Guards against any forward look-ahead in step()/obs."""
    spot, iv, funding = synthetic_series(T=200, seed=12)
    k = 50

    def run(sp, vol):
        env = make_env(sp, vol, funding)
        _, _ = env.reset()
        rng = np.random.default_rng(99)
        out = []
        for _ in range(k):
            _, r, term, trunc, info = env.step(rng.uniform(-1, 1, size=2))
            out.append((info["portfolio_value"], info["step_return"], r))
            if term or trunc:
                break
        return out

    base = run(spot, iv)
    sp2, iv2 = spot.copy(), iv.copy()
    sp2[k + 2:] *= 1.4                            # perturb only FUTURE bars
    iv2[k + 2:] = np.clip(iv2[k + 2:] * 1.5, 0.1, 2.0)
    after = run(sp2, iv2)

    for (pv0, sr0, r0), (pv1, sr1, r1) in zip(base[:k - 1], after[:k - 1]):
        assert abs(pv0 - pv1) < 1e-9
        assert abs(sr0 - sr1) < 1e-12
        assert abs(r0 - r1) < 1e-12


def test_obs_action_spaces():
    spot, iv, funding = synthetic_series(T=40, seed=13)
    env = make_env(spot, iv, funding)
    obs, _ = env.reset()
    assert env.observation_space.shape == (OptionsVolHarvestEnv.OBS_DIM,)
    assert env.action_space.shape == (2,)
    assert obs.dtype == np.float32 and obs.shape == (OptionsVolHarvestEnv.OBS_DIM,)
    assert np.isfinite(obs).all()
