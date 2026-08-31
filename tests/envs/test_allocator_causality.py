"""T+1 causality tripwire for MultiAssetAllocatorEnv (LEAK-2).

Weights are decided at bar t (from obs/vol at t) and applied to the price move
t -> t+1. Two guards:
  1. A position opened from flat enters at price[t+1] (the NEXT bar), never price[t].
  2. Perturbing a FUTURE price bar changes only the step whose t->t+1 move it touches —
     no earlier step's return moves (no forward look-ahead).
"""
from __future__ import annotations

import numpy as np

from sharpen.envs.multi_asset_allocator_env import MultiAssetAllocatorEnv

from .conftest import build_arrays, causal_vol, synthetic_prices


def _env(price, vol):
    arrays = build_arrays(price, vol=vol)
    return MultiAssetAllocatorEnv(
        **arrays, target_vol_asset=0.10, lev_cap=2.0, max_gross_exposure=10.0,
        taker_fee_pct=0.0, slippage_base_bps=0.0, slippage_impact_bps=0.0,
        min_trade_pct=0.0, turnover_penalty=0.0, reward_type="simple",
        circuit_breaker_threshold=0.0,
    )


def test_entry_price_is_next_bar_not_decision_bar():
    """Decision at bar 0 enters at price[1], proving the T+1 execution lag."""
    price = np.array([100.0, 137.0, 137.0], dtype=np.float64).reshape(-1, 1)
    vol = np.full((3, 1), 0.10)
    env = _env(price, vol)
    env.reset()                                       # step_idx = 0, sees price[0]=100
    env.step(np.array([1.0]))                         # advance to bar 1, enter
    assert abs(env.entry_prices[0] - 137.0) < 1e-9    # price[1], NOT price[0]=100


def test_future_price_perturbation_does_not_change_past_returns():
    """Bump the LAST price bar; every step return except the final one is unchanged."""
    price = synthetic_prices(T=120, n=3, seed=21)
    vol = causal_vol(price)

    def run(prices):
        env = _env(prices, vol)
        env.reset()
        rng = np.random.default_rng(5)               # identical action stream both runs
        rets = []
        for _ in range(prices.shape[0] - 1):
            _, _, term, trunc, info = env.step(rng.uniform(-1.0, 1.0, size=3))
            rets.append(info["step_return"])
            if term or trunc:
                break
        return np.array(rets)

    base = run(price)
    bumped = price.copy()
    bumped[-1] *= 1.5                                 # perturb only the final bar
    after = run(bumped)

    # All returns before the final t->t+1 move are byte-identical.
    np.testing.assert_allclose(base[:-1], after[:-1], atol=1e-12)
    # The final step (whose t->t+1 move touches the perturbed bar) DID change.
    assert abs(base[-1] - after[-1]) > 1e-9


def test_step_vol_scales_with_decision_bar_not_next_bar():
    """MUTATION TRIPWIRE (CAUS-05): step() must vol-scale using vol_ary at the DECISION
    bar (read before the clock advances). If it instead read vol_ary[t+1] (a 1-bar
    look-ahead on vol), the realized position weight would change — this test fails.

    The keystone parity test uses a constant vol_ary, and the action unit tests call
    _action_to_weights directly with a manual step_idx, so neither guards the read
    ORDER inside step(); this does.
    """
    price = np.array([100.0, 100.0, 100.0], dtype=np.float64).reshape(-1, 1)
    vol = np.array([[0.10], [0.40], [0.40]])          # decision bar 0 → scale 1.0
    env = _env(price, vol)
    env.reset()                                        # step_idx = 0
    env.step(np.array([1.0]))                          # decide at bar 0 (vol=0.10)
    # scale at decision bar 0 = target/0.10 = 1.0 ⇒ weight 1.0.
    # A leak reading vol[1]=0.40 would give scale 0.25 ⇒ weight 0.25.
    assert abs(env.positions[0] - 1.0) < 1e-9


def test_future_vol_perturbation_does_not_change_past_weights():
    """vol-scaling at decision bar t uses vol[t] only; bumping vol at t' > t leaves the
    weight chosen at t unchanged (vol_ary is already causal, but guard the read index)."""
    price = synthetic_prices(T=60, n=3, seed=7)
    vol = causal_vol(price)
    env = _env(price, vol)
    env.step_idx = 30
    w_before = env._action_to_weights(np.array([1.0, -1.0, 0.5]))

    vol2 = vol.copy()
    vol2[31:] *= 3.0                                  # change only FUTURE vol rows
    env2 = _env(price, vol2)
    env2.step_idx = 30
    w_after = env2._action_to_weights(np.array([1.0, -1.0, 0.5]))
    np.testing.assert_allclose(w_before, w_after, atol=1e-12)
