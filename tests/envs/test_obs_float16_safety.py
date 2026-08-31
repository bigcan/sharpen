"""NAN-01 — every emitted observation feature must survive the float16 cast AMP applies.

Regression tripwires for the crash that killed the first execution-overlay GPU run
(randd_log S553-cont-164/165) and for the same latent exposure found in the sibling envs
by the follow-up audit.

The failure shape is always the same: a ratio whose denominator is guarded only against
being *exactly* zero (a wiped-out equity, a dust-volume bar) emits a value that is
perfectly finite in float64 — so ``np.isfinite``/``np.nan_to_num`` in the env sees nothing
— and becomes ``+/-inf`` the instant ``training.use_amp`` narrows the batch to float16
(max finite 65504). The encoder's first ``LayerNorm`` turns that inf into a NaN for that
ROW ONLY, which reaches ``torch.distributions.Normal`` as an invalid ``loc`` inside
``SACActorNetwork.sample``.

Every test here is NEGATIVE: it fails if either half of the fix — the source bound or the
``sanitize_obs`` backstop — is reverted, and the ``*_backstop`` tests disable the source
bound explicitly so neither half is load-bearing on its own.

Companion suite for the crypto envs: ``tests/crypto/test_obs_float16_safety.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from sharpen.crypto.envs import options_vol_harvest_env as opt_mod
from sharpen.envs import multi_asset_allocator_env as alloc_mod
from sharpen.envs.multi_asset_allocator_env import MultiAssetAllocatorEnv
from sharpen.envs.obs_guard import FP16_MAX, OBS_CLIP, sanitize_obs

from .conftest import build_arrays, synthetic_prices
from .test_options_vol_harvest import make_env as make_options_env
from .test_options_vol_harvest import synthetic_series


def assert_fp16_safe(obs, label=""):
    """Every block of ``obs`` (array or dict of arrays) must round-trip through float16."""
    blocks = obs.items() if isinstance(obs, dict) else [("obs", obs)]
    for key, blk in blocks:
        blk = np.asarray(blk)
        assert np.isfinite(blk).all(), f"{label}{key}: non-finite in float64"
        peak = float(np.abs(blk).max())
        assert peak <= FP16_MAX, (
            f"{label}{key}: |max| = {peak:.4g} exceeds float16 max ({FP16_MAX}) — "
            "will become inf under AMP and NaN after the encoder's LayerNorm")
        assert np.isfinite(np.asarray(blk, dtype=np.float16)).all(), (
            f"{label}{key}: not float16-representable")


# --------------------------------------------------------------------------- #
# The shared guard itself
# --------------------------------------------------------------------------- #
def test_sanitize_obs_bounds_a_huge_finite():
    """The guard's whole point: a merely-HUGE finite float64 must be clipped, not passed."""
    buf = np.array([6.9e11, -6.9e11, 0.5], dtype=np.float32)
    assert np.isfinite(buf).all()                            # passes the pre-fix guard
    with np.errstate(over="ignore"):                         # ...but not the AMP cast
        assert not np.isfinite(buf.astype(np.float16)).all()
    out = sanitize_obs(buf)
    assert out is buf                                        # in-place: no hot-path realloc
    assert np.allclose(out, [OBS_CLIP, -OBS_CLIP, 0.5])
    assert np.isfinite(out.astype(np.float16)).all()


def test_sanitize_obs_scrubs_nan_and_inf():
    buf = np.array([np.nan, np.inf, -np.inf, 1.5], dtype=np.float32)
    assert np.allclose(sanitize_obs(buf), [0.0, 0.0, 0.0, 1.5])


def test_obs_clip_leaves_fp16_headroom():
    """OBS_CLIP bounds the ENV's output; the encoder's own activations still need room
    under the float16 ceiling."""
    assert OBS_CLIP * 6 < FP16_MAX


# --------------------------------------------------------------------------- #
# MultiAssetAllocatorEnv — dust dollar-volume bar
# --------------------------------------------------------------------------- #
_ALLOC_PARAMS = dict(
    target_vol_asset=0.10, lev_cap=2.0, max_gross_exposure=5.0,
    slippage_impact_bps=10.0, min_trade_pct=0.0, random_start=False,
    circuit_breaker_threshold=0.0,
)


def _allocator(volume: float = 1e12, **kw) -> MultiAssetAllocatorEnv:
    price = synthetic_prices(T=120, n=3, seed=11)
    arrays = build_arrays(price, vol=np.full((120, 3), 0.10), volume=volume)
    return MultiAssetAllocatorEnv(**arrays, **{**_ALLOC_PARAMS, **kw})


def _opened_on_a_dust_bar() -> MultiAssetAllocatorEnv:
    """A book holding a real position when the bar's DOLLAR volume collapses to a tiny
    but NONZERO value (halt, holiday stub, data gap).

    ``1e-4`` clears the env's ``> 1e-6`` guard, so the cost-to-unwind feature divides an
    O(capital) notional by ~0. The position is opened on healthy volume first — otherwise
    the (deliberately unbounded) slippage actually CHARGED in ``_calc_costs`` wipes the
    account before any position exists and the test is vacuous.
    """
    env = _allocator()
    env.reset()
    for _ in range(5):
        env.step(np.ones(3, dtype=np.float32))
    assert np.abs(env.positions).max() > 0.0, "test is vacuous — no position was opened"
    env.volume_ary[:] = 1e-4
    return env


def _cost_block(env: MultiAssetAllocatorEnv, obs: np.ndarray) -> np.ndarray:
    n = env.n_assets
    off = 1 + n * env.tech_dim + 3 * n      # pv, tech, positions, unrealized, carry
    return obs[off:off + n]


def test_allocator_obs_fp16_safe_on_dust_volume():
    env = _opened_on_a_dust_bar()
    assert_fp16_safe(env._get_obs(), "dust bar: ")
    for _ in range(10):
        obs, _r, term, trunc, _i = env.step(np.ones(3, dtype=np.float32))
        assert_fp16_safe(obs, f"step {env.step_idx}: ")
        if term or trunc:
            break


def test_allocator_participation_cap_binds_at_the_source():
    """The source bound is real: the cost feature saturates instead of exploding."""
    env = _opened_on_a_dust_bar()
    block = _cost_block(env, env._get_obs())
    assert np.abs(block).max() > 0.0, "test is vacuous — cost feature is empty"
    assert np.abs(block).max() <= OBS_CLIP


def test_allocator_obs_clip_is_an_independent_backstop(monkeypatch):
    """Disable the source bound (pre-fix behaviour) — ``sanitize_obs`` must still hold.

    Also pins the magnitude the pre-fix env actually emitted, so this stays a real
    tripwire rather than a tautology if the dust threshold is ever retuned.
    """
    env = _opened_on_a_dust_bar()
    monkeypatch.setattr(alloc_mod, "MAX_OBS_PARTICIPATION", np.inf)
    raw = _cost_block(env, env._get_obs().copy())
    assert np.abs(raw).max() >= OBS_CLIP, (
        "unbounded participation no longer overflows — the scenario has gone stale")
    assert_fp16_safe(env._get_obs(), "no-source-bound: ")


def test_allocator_normal_volume_obs_is_untouched():
    """The clip must be a backstop, not a semantic transform: on a healthy book nothing
    goes anywhere near OBS_CLIP."""
    env = _allocator(slippage_impact_bps=1.0)
    env.reset()
    peak = 0.0
    for _ in range(40):
        obs, _r, term, trunc, _i = env.step(np.ones(3, dtype=np.float32))
        peak = max(peak, float(np.abs(obs).max()))
        if term or trunc:
            break
    assert peak < 100.0, f"healthy-book obs reached {peak:.4g} — the clip would be semantic"


# --------------------------------------------------------------------------- #
# OptionsVolHarvestEnv — short gamma can drive equity through zero
# --------------------------------------------------------------------------- #
def _wiped_options_env(equity: float):
    """An env holding a real straddle whose equity has been driven to/through zero.

    The greek features (premium frac, vega, $-gamma, residual delta) are all divided by
    equity, and the obs for the ruin bar is built and pushed into the replay buffer BEFORE
    ``terminated`` is honoured — so this state is genuinely reachable on-policy. The
    vega/premium caps are enforced against equity AT THE ROLL, not at the current bar.
    """
    spot, iv, funding = synthetic_series(T=60, seed=3)
    env = make_options_env(spot, iv, funding)
    env.reset()
    for _ in range(5):
        env.step(np.array([-1.0, 0.0], dtype=np.float32))
    assert env.m != 0.0 and env.K > 0, "test is vacuous — no straddle open"
    env.equity = equity
    return env


@pytest.mark.parametrize("equity", [0.0, -50_000.0, 1e-9])
def test_options_obs_fp16_safe_on_wiped_equity(equity):
    assert_fp16_safe(_wiped_options_env(equity)._get_obs(), f"equity={equity}: ")


def test_options_obs_clip_is_an_independent_backstop(monkeypatch):
    """Drop the equity floor back to the pre-fix ``max(equity, 1e-6)`` scale and feed a
    tiny-but-POSITIVE equity — the exact shape that produces a huge FINITE float64 rather
    than an inf. The clip must still hold on its own.
    """
    env = _wiped_options_env(1e-9)
    monkeypatch.setattr(opt_mod, "EQ_FLOOR_FRAC", 0.0)
    raw = env._get_obs().copy()
    assert np.abs(raw).max() >= OBS_CLIP, (
        "unbounded equity denominator no longer overflows — the scenario has gone stale")
    assert_fp16_safe(env._get_obs(), "no-equity-floor: ")


def test_options_running_sharpe_saturates():
    """A near-constant return buffer drove ``mean/sd*sqrt(252)`` to ~1e9 pre-fix."""
    spot, iv, funding = synthetic_series(T=40, seed=5)
    env = make_options_env(spot, iv, funding)
    env.reset()
    env.returns_history = [1e-3 + 1e-11 * i for i in range(20)]
    s = env._running_sharpe()
    assert np.isfinite(s) and abs(s) <= opt_mod.SHARPE_CLIP, f"running sharpe = {s:.4g}"
