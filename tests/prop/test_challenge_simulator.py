"""Tests for the prop-firm challenge simulator (sharpen/prop/challenge_simulator).

Deterministic (seeded); covers the breach/target/timeout logic, the scalar<->vectorized
agreement (the vectorized core is load-bearing), bank-and-derisk, vol normalization, and
monotonicity sanity (higher leverage -> more daily breaches).
"""
from __future__ import annotations

import numpy as np

from sharpen.prop.challenge_simulator import (
    DAILY_BREACH,
    DD_BREACH,
    PASS,
    TIMEOUT,
    FirmRules,
    SizingPolicy,
    _ACTIVE,
    _DAILY,
    _DD,
    _PASS,
    _simulate_vectorized,
    evaluate,
    moving_block_bootstrap,
    normalize_to_vol,
    simulate_path,
    sweep,
)

FTMO = FirmRules(name="ftmo", profit_target=0.10, max_total_dd=0.10, daily_loss_limit=0.05,
                 max_days=60, min_trading_days=0, dd_mode="static")
POL = SizingPolicy(vol_multiplier=1.0)


# --------------------------------------------------------------------------- #
# scalar path logic
# --------------------------------------------------------------------------- #
def test_pass_on_reaching_target():
    path = np.full(60, 0.004)                       # steady +0.4%/day compounds past +10%
    outcome, day, final = simulate_path(path, FTMO, POL)
    assert outcome == PASS
    assert final >= FTMO.profit_target


def test_daily_breach_on_big_single_day_loss():
    path = np.array([0.01, -0.06] + [0.0] * 58)     # day 2 loss 6% > 5% daily limit
    outcome, day, _ = simulate_path(path, FTMO, POL)
    assert outcome == DAILY_BREACH
    assert day == 2


def test_dd_breach_on_slow_bleed():
    # ~2%/day losses (each under the 5% daily limit) compound past the 10% static floor.
    path = np.full(60, -0.02)
    outcome, day, _ = simulate_path(path, FTMO, POL)
    assert outcome == DD_BREACH
    assert day >= 5                                 # not a daily breach; a cumulative one


def test_timeout_when_flat():
    path = np.full(60, 0.0)
    outcome, day, _ = simulate_path(path, FTMO, POL)
    assert outcome == TIMEOUT
    assert day == 60


def test_min_trading_days_delays_pass():
    path = np.full(60, 0.02)                        # hits +10% by ~day 5
    early = FirmRules(name="x", profit_target=0.10, max_total_dd=0.20, daily_loss_limit=0.50,
                      max_days=60, min_trading_days=20)
    outcome, day, _ = simulate_path(path, early, POL)
    assert outcome == PASS
    assert day >= 20                                # cannot pass before the minimum


def test_bank_and_derisk_reduces_daily_moves():
    # Rise to +6% (banks, vol cut to 0.4x) BEFORE reaching the +10% target, then a -10% shock.
    path = np.array([0.03, 0.03, 0.005, -0.10] + [0.0] * 56)   # cum ~6.6% by day3 (< target), then -10%
    banked = SizingPolicy(vol_multiplier=1.0, bank_threshold=0.06, derisk_multiplier=0.4)
    o_bank, _, _ = simulate_path(path, FTMO, banked)
    o_nobank, _, _ = simulate_path(path, FTMO, POL)
    # unbanked: day4 -10%*1.0 breaches the 5% daily limit before the target is reached
    assert o_nobank == DAILY_BREACH
    # banked: day4 -10%*0.4 = -4% does NOT breach; the derisked book then times out below target
    assert o_bank == TIMEOUT


# --------------------------------------------------------------------------- #
# scalar <-> vectorized agreement (the vectorized core is load-bearing)
# --------------------------------------------------------------------------- #
def test_vectorized_matches_scalar():
    rng = np.random.default_rng(3)
    paths = rng.normal(0.0005, 0.02, (500, 60))
    for pol in (POL, SizingPolicy(vol_multiplier=2.5, bank_threshold=0.06, derisk_multiplier=0.4),
                SizingPolicy(vol_multiplier=2.0, intraday_mae_mult=1.4)):
        codes, days = _simulate_vectorized(paths, FTMO, pol)
        code_map = {PASS: _PASS, DD_BREACH: _DD, DAILY_BREACH: _DAILY}
        for i in range(paths.shape[0]):
            o, d, _ = simulate_path(paths[i], FTMO, pol)
            assert codes[i] == code_map.get(o, codes[i]) or o == TIMEOUT
            if o != TIMEOUT:
                assert days[i] == d, f"path {i}: vec day {days[i]} != scalar {d}"
        assert not (codes == _ACTIVE).any()         # every path resolved


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_normalize_to_vol_hits_target():
    rng = np.random.default_rng(1)
    r = rng.normal(0.001, 0.03, 3000)
    out = normalize_to_vol(r, target_vol_ann=0.10)
    ann_vol = out.std(ddof=1) * np.sqrt(252)
    assert abs(ann_vol - 0.10) < 1e-6


def test_bootstrap_shape_and_finite():
    rng = np.random.default_rng(2)
    r = rng.normal(0.0005, 0.02, 2000)
    paths = moving_block_bootstrap(r, 100, 60, block=10, rng=rng)
    assert paths.shape == (100, 60)
    assert np.isfinite(paths).all()


# --------------------------------------------------------------------------- #
# aggregate + monotonicity
# --------------------------------------------------------------------------- #
def test_higher_leverage_more_daily_breaches():
    rng = np.random.default_rng(9)
    r = rng.normal(0.0004, 0.011, 4000)             # a modest-Sharpe book
    lo = evaluate(r, FTMO, SizingPolicy(vol_multiplier=1.0), n_paths=3000, seed=5)
    hi = evaluate(r, FTMO, SizingPolicy(vol_multiplier=4.0), n_paths=3000, seed=5)
    assert hi["p_daily_breach"] >= lo["p_daily_breach"]
    assert 0.0 <= lo["p_pass"] <= 1.0 and 0.0 <= hi["p_pass"] <= 1.0
    probs = lo["p_pass"] + lo["p_dd_breach"] + lo["p_daily_breach"] + lo["p_timeout"]
    assert abs(probs - 1.0) < 1e-3                  # outcomes partition the paths (each p rounded to 4dp)


def test_sweep_returns_grid_and_best():
    rng = np.random.default_rng(4)
    r = rng.normal(0.0004, 0.011, 4000)
    res = sweep(r, FTMO, vol_multipliers=[1.0, 2.0, 3.0], bank_thresholds=[None, 0.06],
                n_paths=2000, seed=5)
    assert len(res.rows) == 6
    best = res.best("p_pass")
    assert best["p_pass"] == max(row["p_pass"] for row in res.rows)
