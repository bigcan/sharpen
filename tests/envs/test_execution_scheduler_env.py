"""ExecutionSchedulerEnv — the RL execution overlay over a FIXED 2-sleeve target.

Step 2 of the execution-overlay build (execution_overlay_architecture.md). Pins the
structural guarantees that make this execution-ONLY:
  - neutral action ⇒ TWAP equal slices; H=1 ⇒ snap;
  - completion: W_held → W_target at horizon end for ANY urgency (forced φ=1 last bar);
  - gross cap: W_held is a convex combination of held-in and target ⇒ never over-levers;
  - reward is direction-free (symmetric in post-arrival drift) + Spooner dampener clips
    favorable-timing profit;
  - LEAK-2: obs at the decision bar is invariant to all future bars; reward reads no bar
    beyond the execution bar;
  - PropFirmWrapperV7 composes (augments private with 3 risk dims).
"""
from __future__ import annotations

import numpy as np
import pytest

from sharpen.envs.execution_scheduler_env import (
    MARKET_DIM,
    PRIVATE_DIM,
    ExecutionSchedulerEnv,
)


def _synthetic(*, U=3, T=30, held=None, targets=None, rebal=(10, 20), seed=0,
               volume=1e9, **overrides):
    """Build env kwargs with a step-function target (conviction constant between rebalances,
    jumping at `rebal`), positive prices, and ample volume (small participation)."""
    rng = np.random.default_rng(seed)
    # geometric-ish positive prices around 100
    rets = rng.normal(0.0, 0.01, size=(T, U))
    price = 100.0 * np.cumprod(1.0 + rets, axis=0)
    K = T - 1
    if held is None:
        held = np.array([0.3, -0.2, 0.1])[:U]
    if targets is None:
        # conviction blocks: [..r0) , [r0..r1), [r1..]
        blocks = [np.array([0.3, -0.2, 0.1])[:U],
                  np.array([0.1, 0.3, -0.2])[:U],
                  np.array([-0.1, 0.1, 0.3])[:U]]
    else:
        blocks = targets
    W = np.zeros((K, U), dtype=np.float64)
    r0, r1 = rebal
    W[:r0] = blocks[0]
    W[r0:r1] = blocks[1]
    W[r1:] = blocks[2]
    kwargs = dict(
        target_weights=W,
        price_ary=price,
        volume_ary=np.full((T, U), float(volume)),
        carry_ary=np.zeros((T, U)),
        timestamps=(np.arange(T, dtype=np.int64) * 86400 + 1_500_000_000),
        assets=[f"A{i}" for i in range(U)],
        rebalance_steps=np.array(rebal, dtype=np.int64),
        horizon_bars=4,
        random_start=False,
    )
    kwargs.update(overrides)
    return kwargs


def _run(env, actions):
    """Step `actions` through `env` from its first valid rebalance; return per-step info."""
    env.reset(options={"rebalance_step": int(env.rebalance_steps[0])})
    out = []
    for a in actions:
        obs, r, term, trunc, info = env.step(np.array([a], dtype=np.float32))
        out.append((obs, r, term, trunc, info))
        if term or trunc:
            break
    return out


# --------------------------------------------------------------------------- #
# Schedule mechanics: neutral=TWAP, H=1=snap, completion
# --------------------------------------------------------------------------- #
def test_neutral_action_is_twap_equal_slices():
    """a=0 ⇒ m=1 ⇒ closed-loop TWAP ⇒ equal slices of the parent each bar, completing in H."""
    env = ExecutionSchedulerEnv(**_synthetic())                  # H=4
    steps = _run(env, [0.0, 0.0, 0.0, 0.0])
    executed = [s[4]["executed_l1"] for s in steps]
    gap0_l1 = abs(np.array([0.1, 0.3, -0.2]) - np.array([0.3, -0.2, 0.1])).sum()  # 1.0
    for e in executed:
        assert abs(e - gap0_l1 / 4) < 1e-9                       # equal TWAP slices
    assert steps[-1][2] is True                                  # TERMINATED at horizon end (TERM-01)
    assert steps[-1][3] is False                                 # ...not truncated
    np.testing.assert_allclose(env._W_held, env.W_target[env._r], atol=1e-9)  # completion


def test_horizon_one_is_snap():
    """H=1 ⇒ the single bar forces φ=1 ⇒ full snap to the target (== the _replay behavior)."""
    env = ExecutionSchedulerEnv(**_synthetic(horizon_bars=1))
    steps = _run(env, [-1.0])                                    # urgency irrelevant; forced φ=1
    assert steps[0][2] is True and steps[0][3] is False          # terminated, not truncated (TERM-01)
    np.testing.assert_allclose(env._W_held, env.W_target[env._r], atol=1e-12)
    gap0_l1 = abs(np.array([0.1, 0.3, -0.2]) - np.array([0.3, -0.2, 0.1])).sum()
    assert abs(steps[0][4]["executed_l1"] - gap0_l1) < 1e-12


@pytest.mark.parametrize("urgency", [-1.0, -0.5, 0.0, 0.5, 1.0])
def test_completion_guarantee_any_urgency(urgency):
    """For ANY constant urgency (incl. full pause a=-1), forced φ=1 on the last bar drives
    W_held → W_target at horizon end (the execution-only guarantee)."""
    env = ExecutionSchedulerEnv(**_synthetic())
    _run(env, [urgency] * 4)
    np.testing.assert_allclose(env._W_held, env.W_target[env._r], atol=1e-9)


def test_pause_then_forced_completion_concentrates_in_last_bar():
    """a=-1 (φ=0) for the first H-1 bars executes nothing; the forced last bar snaps the
    whole parent (so under-execution defers cost to a high-impact lump — the agent's
    incentive to NOT leave too much)."""
    env = ExecutionSchedulerEnv(**_synthetic())
    steps = _run(env, [-1.0, -1.0, -1.0, -1.0])
    ex = [s[4]["executed_l1"] for s in steps]
    assert ex[0] == 0.0 and ex[1] == 0.0 and ex[2] == 0.0
    assert abs(ex[3] - 1.0) < 1e-9                               # whole parent in the forced bar


# --------------------------------------------------------------------------- #
# Gross cap: convex combination ⇒ never over-levers / overshoots
# --------------------------------------------------------------------------- #
def test_gross_never_exceeds_endpoints():
    """W_held_new = (1−φ)·W_held + φ·W_target ⇒ gross(W_held) ≤ max(gross endpoints) ≤ cap,
    for ANY action sequence (random)."""
    held = np.array([1.0, -0.6, 0.4])
    blocks = [np.array([1.0, -0.6, 0.4]), np.array([-0.8, 0.7, 0.5]), np.array([0.3, -0.9, 0.6])]
    env = ExecutionSchedulerEnv(**_synthetic(held=held, targets=blocks, max_gross_exposure=3.0))
    rng = np.random.default_rng(1)
    env.reset(options={"rebalance_step": int(env.rebalance_steps[0])})
    bound = max(np.abs(env.W_target[env._r - 1]).sum(),
                np.abs(env.W_target[env._r:env._r + env.H]).sum(axis=1).max())
    for _ in range(env.H):
        a = rng.uniform(-1, 1, size=1).astype(np.float32)
        _, _, term, trunc, info = env.step(a)
        assert info["gross_exposure"] <= bound + 1e-9
        assert info["gross_exposure"] <= env.max_gross_exposure + 1e-9
        if term or trunc:
            break


# --------------------------------------------------------------------------- #
# Observation contract
# --------------------------------------------------------------------------- #
def test_obs_shape_and_space():
    env = ExecutionSchedulerEnv(**_synthetic())
    obs, _ = env.reset(options={"rebalance_step": 10})
    assert set(obs) == {"scale_0", "private"}
    assert obs["scale_0"].shape == (MARKET_DIM,) and obs["scale_0"].dtype == np.float32
    assert obs["private"].shape == (PRIVATE_DIM,) and obs["private"].dtype == np.float32
    assert env.observation_space["scale_0"].shape == (MARKET_DIM,)
    assert env.observation_space["private"].shape == (PRIVATE_DIM,)
    assert np.isfinite(obs["scale_0"]).all() and np.isfinite(obs["private"]).all()


def test_private_time_and_inventory_track_progress():
    """time_remaining decreases each bar; inventory_remaining falls as the gap is worked."""
    env = ExecutionSchedulerEnv(**_synthetic())
    env.reset(options={"rebalance_step": 10})
    t_prev, inv_prev = 1.0, 1.0
    for _ in range(env.H):
        obs, _, term, trunc, _ = env.step(np.array([0.0], dtype=np.float32))
        assert obs["private"][0] <= t_prev + 1e-9               # time_remaining ↓
        assert obs["private"][1] <= inv_prev + 1e-9             # inventory_remaining ↓
        t_prev, inv_prev = obs["private"][0], obs["private"][1]
        if term or trunc:
            break


# --------------------------------------------------------------------------- #
# Reward: direction-free + Spooner asymmetric dampener
# --------------------------------------------------------------------------- #
def _drift_reward(direction: float, dampen: float) -> float:
    """Cumulative reward for a pure BUY parent when the price drifts `direction` after
    arrival, under a fixed (neutral) schedule."""
    U, T, r0 = 1, 20, 8
    price = np.full((T, U), 100.0)
    for k in range(r0, T):                                      # post-arrival geometric drift
        price[k, 0] = 100.0 * (1.0 + direction * 0.01) ** (k - r0)
    W = np.zeros((T - 1, U))
    W[:r0] = 0.0
    W[r0:] = 0.5                                                # buy 0 → 0.5 at the rebalance
    env = ExecutionSchedulerEnv(
        target_weights=W, price_ary=price, volume_ary=np.full((T, U), 1e9),
        carry_ary=np.zeros((T, U)), timestamps=np.arange(T, dtype=np.int64) * 86400,
        assets=["A0"], rebalance_steps=np.array([r0]), horizon_bars=4,
        random_start=False, asymmetric_dampen=dampen, impact_penalty=0.0,
    )
    total = 0.0
    env.reset(options={"rebalance_step": r0})
    for _ in range(env.H):
        _, rew, term, trunc, _ = env.step(np.array([0.0], dtype=np.float32))
        total += rew
        if term or trunc:
            break
    return total


def test_reward_is_direction_free_symmetric():
    """IS struck vs the fixed arrival price ⇒ reward responds symmetrically to the SIGN of
    post-arrival drift (a rising price hurts a buy as much as a falling price helps it).
    No directional bias is baked in (without the dampener)."""
    up = _drift_reward(+1.0, dampen=0.0)
    flat = _drift_reward(0.0, dampen=0.0)
    down = _drift_reward(-1.0, dampen=0.0)
    assert up < flat < down                                     # rising price ⇒ worse fill for a buy
    # symmetric magnitude (impact off, pv≈capital): |up−flat| ≈ |down−flat|
    assert abs((flat - up) - (down - flat)) < 0.05 * abs(down - flat)


def test_spooner_dampener_clips_favorable_timing_profit():
    """η>0 removes the *gain* from a favorable (price-falls-for-a-buy) drift while keeping
    the loss — structurally blocks disguised trend-following."""
    down_plain = _drift_reward(-1.0, dampen=0.0)
    down_damped = _drift_reward(-1.0, dampen=1.0)
    up_plain = _drift_reward(+1.0, dampen=0.0)
    up_damped = _drift_reward(+1.0, dampen=1.0)
    assert down_damped < down_plain                            # favorable profit clipped
    assert abs(up_damped - up_plain) < 1e-9                    # adverse (loss) side untouched


# --------------------------------------------------------------------------- #
# LEAK-2 causality tripwire
# --------------------------------------------------------------------------- #
def test_obs_invariant_to_all_future_bars():
    """The obs at the decision bar must use ONLY bars ≤ that bar. Mutating every future bar
    (price + volume from r+1 on) must NOT change the reset obs — fails if any feature peeks."""
    kw = _synthetic()
    env_a = ExecutionSchedulerEnv(**kw)
    obs_a, _ = env_a.reset(options={"rebalance_step": 10})

    kw2 = dict(kw)
    price2 = kw["price_ary"].copy()
    vol2 = kw["volume_ary"].copy()
    price2[11:] *= 3.7                                          # wild future-tail mutation
    vol2[11:] *= 0.1
    kw2["price_ary"], kw2["volume_ary"] = price2, vol2
    env_b = ExecutionSchedulerEnv(**kw2)
    obs_b, _ = env_b.reset(options={"rebalance_step": 10})

    np.testing.assert_array_equal(obs_a["scale_0"], obs_b["scale_0"])
    np.testing.assert_array_equal(obs_a["private"], obs_b["private"])


def test_reward_reads_no_bar_beyond_execution():
    """Step-k reward uses the execution bar k+1 (the realized fill) but NOTHING beyond it;
    mutating price[k+2:] must not change the step-k reward."""
    kw = _synthetic()
    env_a = ExecutionSchedulerEnv(**kw)
    env_a.reset(options={"rebalance_step": 10})
    _, r_a, _, _, _ = env_a.step(np.array([0.3], dtype=np.float32))   # decision bar k=10, fill at 11

    kw2 = dict(kw)
    price2 = kw["price_ary"].copy()
    price2[12:] *= 2.5                                          # beyond the execution bar (11)
    kw2["price_ary"] = price2
    env_b = ExecutionSchedulerEnv(**kw2)
    env_b.reset(options={"rebalance_step": 10})
    _, r_b, _, _, _ = env_b.step(np.array([0.3], dtype=np.float32))
    assert r_a == r_b


def test_relative_equity_uses_per_bar_return_not_cumulative():
    """ADR-5 / MATH-EXEC-3: the execution tracking-error equity must accumulate the PER-BAR
    return, NOT the cumulative-from-arrival return. With a paused (lagged) position and a
    multi-bar +10%/bar rise, each bar's tracking loss is −0.6·cap·0.10; the cumulative-return
    bug would give 0.10, 0.21, 0.33…. Held-in = 0 ⇒ pv_before = capital (no MtM drift) so the
    expected value is exact."""
    U, T, r0, H = 1, 16, 8, 4
    price = np.full((T, U), 100.0)
    for j, k in enumerate(range(r0, r0 + H + 1)):
        price[k, 0] = 100.0 * (1.10 ** j)                       # +10% per bar from arrival
    W = np.zeros((T - 1, U))
    W[r0:] = 0.6                                                # buy 0 → 0.6 at the rebalance
    env = ExecutionSchedulerEnv(
        target_weights=W, price_ary=price, volume_ary=np.full((T, U), 1e9),
        carry_ary=np.zeros((T, U)), timestamps=np.arange(T, dtype=np.int64) * 86400,
        assets=["A0"], rebalance_steps=np.array([r0]), horizon_bars=H, random_start=False,
        relative_equity=True,
    )
    env.reset(options={"rebalance_step": r0})
    pv, last_cost = None, 0.0
    for _ in range(H):
        _, _, term, trunc, info = env.step(np.array([-1.0], dtype=np.float32))   # pause
        pv, last_cost = info["portfolio_value"], info["fill_cost"]
        if term or trunc:
            break
    # 3 paused bars each lose −0.6·cap·0.10 (per-bar 10%); the forced-complete bar pays cost only.
    expect = env.initial_capital - 3 * (0.6 * env.initial_capital * 0.10) - last_cost
    assert abs(pv - expect) < 1e-6


def test_daily_drift_completion_tracks_latest_target():
    """When the target drifts WITHIN the horizon, completion re-anchors to the LATEST
    daily target at horizon end (the overlay tracks the fixed daily trajectory)."""
    U, T, r0 = 2, 24, 10
    rng = np.random.default_rng(3)
    price = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, size=(T, U)), axis=0)
    W = np.zeros((T - 1, U))
    W[:r0] = np.array([0.2, -0.1])
    # post-rebalance conviction + small daily drift each bar
    for k in range(r0, T - 1):
        W[k] = np.array([0.4, 0.3]) + 0.002 * (k - r0)
    env = ExecutionSchedulerEnv(
        target_weights=W, price_ary=price, volume_ary=np.full((T, U), 1e9),
        carry_ary=np.zeros((T, U)), timestamps=np.arange(T, dtype=np.int64) * 86400,
        assets=["A0", "A1"], rebalance_steps=np.array([r0]), horizon_bars=4, random_start=False,
    )
    _run(env, [0.2, -0.4, 0.6, 0.0])
    k_last = env._r + env.H - 1
    np.testing.assert_allclose(env._W_held, env.W_target[k_last], atol=1e-9)


# --------------------------------------------------------------------------- #
# PropFirmWrapperV7 composition (ADR-5 / ADR-9)
# --------------------------------------------------------------------------- #
def test_propfirm_wrapper_composes():
    """The env exposes the contract PropFirmWrapperV7 needs; augment_obs appends 3 dims to
    private and step flows info['portfolio_value']."""
    from sharpen.envs.prop_firm_wrapper import PropFirmWrapperV7
    base = ExecutionSchedulerEnv(**_synthetic())
    env = PropFirmWrapperV7(base, augment_obs=True, max_trailing_drawdown_pct=0.05,
                            static_peak=True)
    obs, _ = env.reset(options={"rebalance_step": 10})
    assert obs["private"].shape == (PRIVATE_DIM + 3,)          # +[dd_remaining, daily_remaining, profit_progress]
    assert obs["scale_0"].shape == (MARKET_DIM,)
    obs, rew, term, trunc, info = env.step(np.array([0.0], dtype=np.float32))
    assert "portfolio_value" in info
    assert obs["private"].shape == (PRIVATE_DIM + 3,)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_no_valid_rebalance_step_rejected():
    with pytest.raises(ValueError):
        ExecutionSchedulerEnv(**_synthetic(rebal=(10, 20), horizon_bars=100))  # H > data span


def test_bad_target_shape_rejected():
    kw = _synthetic()
    kw["target_weights"] = kw["target_weights"][:, :2]         # wrong U
    with pytest.raises(AssertionError):
        ExecutionSchedulerEnv(**kw)


# --------------------------------------------------------------------------- #
# NAN-01 — degenerate (empty) parent orders + float16-safe observations
#
# Regression tripwires for the crash that killed the first real GPU training run
# (randd_log S553-cont-164): 3/171 horizon-valid rebalance steps in the TRAIN window had
# W_target[r] == W_target[r-1] (the linear book's signal-warmup months, combined_w ≡ 0), so
# `_gap0_l1 == 0`. Every parent-normalized quantity then divided by ~0 and emitted ~1e11 —
# finite in float64, therefore invisible to the env's `np.isfinite` guard, but `+/-inf` the
# instant AMP narrowed the batch to float16. The encoder's first LayerNorm turned that inf
# into a per-ROW NaN, which surfaced as an invalid `loc` in
# `torch.distributions.Normal(mu, sigma)` inside the SAC actor.
#
# Both tests are NEGATIVE: they fail if either half of the fix is reverted.
# --------------------------------------------------------------------------- #
_FP16_MAX = 65504.0


def _degenerate_target_kwargs(**overrides):
    """A target whose FIRST rebalance is empty (block 0 == block 1) and whose second is real
    — i.e. the signal-warmup shape that produced the crash."""
    zero = np.zeros(3)
    kw = _synthetic(targets=[zero, zero, np.array([-0.1, 0.1, 0.3])], held=zero)
    kw.update(overrides)
    return kw


def test_degenerate_parent_order_dropped_from_calendar():
    """A rebalance step whose target does not move is NOT a schedulable parent order."""
    env = ExecutionSchedulerEnv(**_degenerate_target_kwargs())
    assert 10 not in set(int(r) for r in env.rebalance_steps), \
        "empty parent order (‖ΔW_target‖₁ == 0) must be dropped from the calendar"
    assert 20 in set(int(r) for r in env.rebalance_steps), "real parent order must survive"
    # ...and every surviving step is genuinely non-degenerate.
    for r in env.rebalance_steps:
        assert np.abs(env.W_target[r] - env.W_target[r - 1]).sum() >= env.min_parent_l1


def test_all_degenerate_calendar_rejected():
    """If NOTHING in the window moves the target there is no execution problem to pose."""
    zero = np.zeros(3)
    with pytest.raises(ValueError, match="degenerate"):
        ExecutionSchedulerEnv(**_synthetic(targets=[zero, zero, zero], held=zero))


def test_obs_is_float16_representable():
    """Every emitted feature must survive the float16 cast AMP applies.

    A merely-huge finite float64 passes `np.isfinite` and only becomes inf downstream — so
    the env, not the agent, has to bound it. Drives every valid episode under adversarial
    (max-urgency, min-urgency, neutral) actions.
    """
    env = ExecutionSchedulerEnv(**_synthetic(volume=1.0))       # tiny volume ⇒ huge participation
    for r in (int(x) for x in env.rebalance_steps):
        for a in (-1.0, 0.0, 1.0):
            obs, _ = env.reset(options={"rebalance_step": r})
            done = False
            while not done:
                for key, blk in obs.items():
                    assert np.isfinite(blk).all(), f"{key} non-finite at r={r}"
                    assert np.abs(blk).max() <= _FP16_MAX, (
                        f"{key} at r={r} exceeds float16 max ({np.abs(blk).max():.4g}) — "
                        "will become inf under AMP and NaN after LayerNorm")
                    assert np.isfinite(np.asarray(blk, dtype=np.float16)).all()
                obs, _rew, term, trunc, _info = env.step(np.array([a], dtype=np.float32))
                done = term or trunc


def test_obs_clip_bounds_an_empty_parent_order():
    """The clip is an INDEPENDENT backstop, not decoration.

    Disable the calendar filter (``min_parent_l1=0``, i.e. exactly the pre-fix behaviour) and
    drive the empty parent order that crashed the GPU run. Unclipped, `inventory_remaining`
    /`schedule_deviation`/`cost_so_far_bps` reach ~1e11 here; the obs must still be
    float16-representable so neither half of the fix is load-bearing on its own.
    """
    # The production shape (randd_log S553-cont-164, train step 61 / 2006-03-31): an EMPTY
    # parent order (‖ΔW_target‖₁ == 0 at r) whose horizon straddles the bar where the linear
    # book's signal warmup completes and the target jumps 0 → O(1). gap0_l1 == 0 then
    # normalizes an O(1) mid-horizon gap ⇒ inventory_remaining ~ 1e11.
    zero = np.zeros(3)
    env = ExecutionSchedulerEnv(**_synthetic(
        targets=[zero, zero, np.array([-0.5, 0.5, 0.3])], held=zero,
        rebal=(10, 12),                                         # target turns on at k=12, INSIDE
        horizon_bars=4, min_parent_l1=0.0))                     # the r=10 horizon (k=10..13)
    assert 10 in set(int(r) for r in env.rebalance_steps)       # filter off ⇒ still admissible
    obs, _ = env.reset(options={"rebalance_step": 10})
    done = False
    while not done:
        for key, blk in obs.items():
            assert np.isfinite(np.asarray(blk, dtype=np.float16)).all(), (
                f"{key} not float16-representable ({np.abs(blk).max():.4g}) on an empty "
                "parent order — the obs clip is the last line of defence")
        obs, _rew, term, trunc, _info = env.step(np.array([1.0], dtype=np.float32))
        done = term or trunc


# --------------------------------------------------------------------------- #
# TERM-01 — the horizon end is a genuine MDP terminal, not a time limit
#
# `truncated` and `terminated` are not interchangeable here: SACTrainer stores
# `dones_for_buffer = terms` (terminated only), so a TRUNCATED horizon end writes done=0
# and the critic bootstraps across the episode boundary into an unrelated parent order
# months away. With done never set, the value function has no terminal anywhere in the MDP
# and targets the infinite-horizon sum (1/(1-gamma) = 100 bars = 20 parent orders) instead
# of the H=5 episodic one. That is the cvar_q_mean ~ -880-and-climbing measured in the 75K
# seedcheck, against an episodic CVaR of ~ -42 (randd_log S553-cont-165).
#
# Negative tests: they fail if the horizon end reverts to signalling truncation.
# --------------------------------------------------------------------------- #
def test_horizon_end_terminates_not_truncates():
    env = ExecutionSchedulerEnv(**_synthetic())                  # H=4
    steps = _run(env, [0.0, 0.0, 0.0, 0.0])
    for i, (_obs, _r, term, trunc, _info) in enumerate(steps[:-1]):
        assert not term and not trunc, f"step {i} ended the episode early"
    _, _, term, trunc, _ = steps[-1]
    assert term is True, "horizon end must TERMINATE (done=1 reaches the replay buffer)"
    assert trunc is False, "horizon end must not be reported as a time-limit truncation"


def test_terminated_is_what_reaches_the_replay_buffer():
    """Pins the CONSEQUENCE, not the flag. SACTrainer derives done from `terminated` alone,
    so this asserts the training signal the critic actually sees at the boundary."""
    env = ExecutionSchedulerEnv(**_synthetic())
    steps = _run(env, [0.0, 0.0, 0.0, 0.0])
    dones = [float(term) for _o, _r, term, _t, _i in steps]      # sac_trainer: dones = terms
    assert dones == [0.0, 0.0, 0.0, 1.0], (
        f"critic would bootstrap past the parent order's end; dones={dones}")


def test_completion_still_holds_at_the_terminal_step():
    """The terminal flag must not disturb the structural completion invariant."""
    env = ExecutionSchedulerEnv(**_synthetic())
    _run(env, [-1.0, -1.0, -1.0, -1.0])                          # pause as long as possible
    np.testing.assert_allclose(env._W_held, env.W_target[env._r], atol=1e-9)
