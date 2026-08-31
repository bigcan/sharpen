"""Regression test for V7 ContinuousSwingEnv earn-timing causality (no same-bar leak).

Pins audit finding **P3-01** ("V7 base-scale same-bar look-ahead"), raised by the
2026-06-01 deep lifecycle audit and **REFUTED** that session (S553-cont-23): the
finder+skeptic mistook the within-step observation advance — which prepares the
NEXT action — for the current action's earn. This test makes the refutation a
permanent guard so a *real* same-bar leak can never be silently reintroduced
(the original throwaway probe lived at ``C:\\tmp\\p301_causality_tripwire.py``).

Mechanism under test — the "echo policy":
  At each step the agent observes the newest base-scale log return
  ``r_obs = obs["scale_0"][-1, 0]`` (= ``log(close_k / close_{k-1})``;
  ``multiscale_handler._compute_scale_features`` writes ``log_return`` to column 0)
  and acts ``sign(r_obs)``. ``ContinuousSwingEnv.step`` then advances one bar and
  pays the held position the NEXT (``k -> k+1``) price move:

    - CAUSAL env  -> earns ``log(close_{k+1}/close_k)``, which on an i.i.d. random
                     walk is independent of ``r_obs``  ->  echo win-rate ~ 50%,
                     ``corr(sign(r_obs), earned) ~ 0``.
    - LEAKY env   -> earns the SAME observed bar's move  ->  echo win-rate ~ 100%,
                     ``corr ~ +1`` (the agent prints money by reading the present).

The env is run frictionless (``taker_fee = slippage = deadband = 0``, ATR cap
disabled, ``max_leverage = 1``) so the per-unit equity delta equals the exact
earned price return and ``sign(PnL) == sign(position * earned-return)``. The
synthetic price path is a zero-drift Gaussian random walk, so consecutive
base-bar returns are independent: the causal signature (~50% / ~0) and the leaky
signature (~100% / ~+1) are unmistakably separated.

Companion to ``tests/data/test_multiscale_causality.py``, which guards the
*coarse*-bar X2 leak; this guards the *base*-scale earn-timing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sharpen.data.multiscale_handler import MultiScaleOHLCVHandler
from sharpen.envs.continuous_swing_env import ContinuousSwingEnv

# Mirrors the gmgp1-btc / sg1-btc V7 octave; the earn-timing under test is
# scale-agnostic (it lives in ContinuousSwingEnv.step), but using a real config's
# scales keeps the guard recognisable.
_SCALES = [15, 60, 240]
_BASE_SCALE = min(_SCALES)
_WINDOW = 30
# ~36k 1-min bars -> ~2.4k base (15m) bars -> a few thousand scored echo steps,
# enough for a win-rate std ~1% (clean separation of 50% from 100%).
_N_MIN = 36000
_SEED = 20260602


def _write_random_walk_parquet(path) -> None:
    """Zero-drift Gaussian random-walk 1-min OHLCV.

    Disjoint base-bar intervals aggregate disjoint sets of i.i.d. increments, so
    consecutive base-bar log returns are independent and roughly sign-balanced —
    exactly the BTC-momentum-autocorr~0 property the live probe relied on.
    """
    rng = np.random.RandomState(_SEED)
    incr = rng.normal(0.0, 5e-4, size=_N_MIN)          # i.i.d. 1-min log increments, no drift
    close = 100.0 * np.exp(np.cumsum(incr))
    half_range = np.abs(rng.normal(0.0, 3e-4, size=_N_MIN)) * close + 1e-6
    open_ = close * np.exp(rng.normal(0.0, 2e-4, size=_N_MIN))
    high = np.maximum(close + half_range, np.maximum(open_, close))
    low = np.minimum(close - half_range, np.minimum(open_, close))
    vol = np.abs(rng.normal(0.0, 1.0, size=_N_MIN)) * 100.0 + 10.0
    ts = pd.date_range("2024-01-01", periods=_N_MIN, freq="1min")
    pd.DataFrame(
        {"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": vol}
    ).to_parquet(path)


def _make_frictionless_env(parquet) -> ContinuousSwingEnv:
    """V7 env with every friction/sizing knob neutralised so PnL sign == earn sign."""
    handler = MultiScaleOHLCVHandler(
        file_path=str(parquet),
        ticker="SYNTH",
        feature_config={"scales": _SCALES, "window_size": _WINDOW},
    )
    config = {
        "mdp_version": "v7",
        "initial_balance": 100000.0,
        "window_size": _WINDOW,
        "features_per_scale": 8,
        "scales": _SCALES,
        "taker_fee": 0.0,            # no fee  -> earned == position * price_return
        "slippage_base_bps": 0.0,    # no slippage
        "deadband_threshold": 0.0,   # every action trades
        "atr_cap_percentile": 100,   # cap never trips (current_atr can't exceed buffer max)
        "atr_cap_max_position": 1.0,
        "max_leverage": 1.0,
        "episode_length": 0,         # run to data exhaustion
        "random_start": False,
        "max_drawdown_pct": 1.0,     # never DD-terminate
        "gap_detection": False,      # never zero a return
        "reward": {"mode": "raw"},
    }
    return ContinuousSwingEnv(config=config, data_handler=handler)


def _run_echo_policy(env, max_steps: int = 10000):
    """Act sign(newest observed base return); record what the env actually pays."""
    obs, _ = env.reset()
    echo_wins = 0
    n = 0
    obs_signs: list[int] = []
    earned_per_unit: list[float] = []
    for _ in range(max_steps):
        r_obs = float(obs["scale_0"][-1, 0])           # newest OBSERVED base-scale log return
        act = 1.0 if r_obs >= 0 else -1.0
        eq0 = env.equity
        obs, _, term, trunc, _ = env.step(np.array([act], dtype=np.float32))
        if term or trunc:
            break
        pos = env.current_position
        if abs(pos) < 1e-9:
            continue
        earned = (env.equity - eq0) / eq0              # == pos * next-bar return (fee == 0)
        obs_signs.append(1 if r_obs >= 0 else -1)
        earned_per_unit.append(earned / pos)           # the raw bar return the env PAID
        if earned > 0:
            echo_wins += 1
        n += 1
    return n, echo_wins, np.asarray(obs_signs), np.asarray(earned_per_unit)


def test_v7_env_earn_timing_is_causal(tmp_path):
    parquet = tmp_path / "random_walk_1min.parquet"
    _write_random_walk_parquet(parquet)
    env = _make_frictionless_env(parquet)

    n, echo_wins, obs_signs, earned = _run_echo_policy(env)

    assert n >= 500, f"too few scored steps ({n}) — synthetic window too short to be conclusive"

    win_rate = echo_wins / n
    corr = float(np.corrcoef(obs_signs, earned)[0, 1]) if earned.std() > 0 else float("nan")

    # Leaky signature would be win-rate ~ 100% and corr ~ +1 (agent earns the bar
    # it just read). Causal signature is ~50% / ~0 on an i.i.d. random walk.
    assert win_rate < 0.65, (
        f"SAME-BAR LEAK: echo win-rate {win_rate:.3%} over {n} steps (causal ~50%, leaky ~100%). "
        f"The V7 env is paying the position the return it just OBSERVED."
    )
    assert abs(corr) < 0.20, (
        f"SAME-BAR LEAK: corr(sign(observed return), earned-per-unit) = {corr:+.4f} "
        f"(causal ~0, leaky ~+1). Earned move is correlated with the just-observed bar."
    )
    # Sanity floor: data is balanced and the env is actually trading (not a stuck
    # all-flat / sign-inverted degenerate path masquerading as 'not leaky').
    assert win_rate > 0.35, (
        f"echo win-rate {win_rate:.3%} is implausibly low for an i.i.d. walk — "
        f"the env may be paying the NEGATED next-bar return; investigate earn timing."
    )
