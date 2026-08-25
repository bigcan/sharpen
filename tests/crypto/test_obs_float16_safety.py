"""NAN-01 — float16-safe observations for the crypto envs.

Same contract as ``tests/envs/test_obs_float16_safety.py`` (read its header for the
mechanism): under ``training.use_amp: true`` the batch is narrowed to float16, whose max
finite value is 65504, so a merely-HUGE finite float64 feature passes the env's
``np.isfinite``/``np.nan_to_num`` guard and only becomes ``inf`` inside the encoder — where
the first ``LayerNorm`` turns it into a per-ROW NaN and the SAC actor gets an invalid
``loc``.

Two reachable sources here:

* ``FundingArbEnv`` — ``_get_portfolio_value`` returns ``max(pv, 0.0)``, i.e. EXACTLY zero
  on a wipe-out, and the pre-fix margin/capital-usage features divided by ``pv + 1e-10``
  ⇒ ~1e15. The terminal obs is built and stored in the replay buffer before the circuit
  breaker fires, so the state is on-policy reachable. That env also had NO finite guard at
  all on the concatenated obs — only ``tech_flat`` was scrubbed.
* ``CryptoPerpEnv`` — the cost-to-rebalance feature's ``> 1e-6`` volume guard only catches
  a volume of exactly ~zero; a dust-volume bar divides an O(capital) notional by ~0.

Every test is NEGATIVE, and the ``*_backstop`` tests disable the source bound so neither
half of the fix is load-bearing on its own.
"""
from __future__ import annotations

import numpy as np

from finrl_pro_ds.crypto.envs import crypto_perp_env as perp_mod
from finrl_pro_ds.crypto.envs import funding_arb_env as arb_mod
from finrl_pro_ds.envs.obs_guard import FP16_MAX, OBS_CLIP

from .test_crypto_perp_env_long_only import _make_env as _make_perp_env
from .test_funding_arb_env import _make_env as _make_arb_env


def assert_fp16_safe(obs: np.ndarray, label: str = "") -> None:
    obs = np.asarray(obs)
    assert np.isfinite(obs).all(), f"{label}non-finite in float64"
    peak = float(np.abs(obs).max())
    assert peak <= FP16_MAX, (
        f"{label}|max| = {peak:.4g} exceeds float16 max ({FP16_MAX}) — will become inf "
        "under AMP and NaN after the encoder's LayerNorm")
    assert np.isfinite(np.asarray(obs, dtype=np.float16)).all(), (
        f"{label}not float16-representable")


# --------------------------------------------------------------------------- #
# FundingArbEnv — portfolio value driven to exactly zero
# --------------------------------------------------------------------------- #
def _wiped_arb_env():
    """An arb book with both legs open whose portfolio value has been driven to zero."""
    env = _make_arb_env()
    env.reset()
    for _ in range(6):
        env.step(np.full(env.n_assets, 0.5, dtype=np.float32))
    assert env.perp_notionals.sum() > 0.0, "test is vacuous — no position was opened"
    env.margin_balance = 0.0                      # ⇒ _get_portfolio_value() == 0.0 exactly
    spot = env.spot_price_ary[env.step_idx]
    perp = env.perp_price_ary[env.step_idx]
    assert env._get_portfolio_value(spot, perp) == 0.0
    return env


def test_funding_arb_obs_fp16_safe_on_wiped_portfolio():
    assert_fp16_safe(_wiped_arb_env()._get_obs(), "wiped pv: ")


def test_funding_arb_obs_has_a_finite_guard_at_all():
    """Pre-fix only ``tech_flat`` was scrubbed; a NaN anywhere else went straight out."""
    env = _make_arb_env()
    env.reset()
    env.step(np.full(env.n_assets, 0.5, dtype=np.float32))
    env.cumulative_funding[0] = np.nan
    env.spot_notionals[0] = np.inf
    assert np.isfinite(env._get_obs()).all()


def test_funding_arb_obs_clip_is_an_independent_backstop(monkeypatch):
    """Remove the portfolio-value floor (pre-fix behaviour) and confirm both that the
    scenario still overflows and that the clip alone contains it."""
    env = _wiped_arb_env()
    monkeypatch.setattr(arb_mod, "PV_FLOOR_FRAC", 1e-15)
    raw = env._get_obs().copy()
    assert np.abs(raw).max() >= OBS_CLIP, (
        "unbounded pv denominator no longer overflows — the scenario has gone stale")
    assert_fp16_safe(env._get_obs(), "no-pv-floor: ")


def test_funding_arb_healthy_obs_is_untouched():
    """The clip is a backstop, not a semantic transform."""
    env = _make_arb_env()
    env.reset()
    peak = 0.0
    for _ in range(40):
        obs, _r, term, trunc, _i = env.step(np.full(env.n_assets, 0.3, dtype=np.float32))
        peak = max(peak, float(np.abs(obs).max()))
        if term or trunc:
            break
    assert peak < 100.0, f"healthy arb obs reached {peak:.4g} — the clip would be semantic"


# --------------------------------------------------------------------------- #
# CryptoPerpEnv — dust-volume bar
# --------------------------------------------------------------------------- #
def _perp_on_a_dust_bar():
    """Positions opened on healthy volume, then the bar volume collapses to a tiny but
    NONZERO value (which clears the env's ``> 1e-6`` guard)."""
    env = _make_perp_env()
    env.reset()
    for _ in range(5):
        env.step(np.full(env.n_assets, 0.6, dtype=np.float32))
    assert np.abs(env.positions).max() > 0.0, "test is vacuous — no position was opened"
    env.volume_ary[:] = 1e-4
    return env


def _perp_cost_block(env, obs: np.ndarray) -> np.ndarray:
    n = env.n_assets
    off = 1 + n * env.tech_dim + 3 * n       # pv, tech, positions, unrealized, funding
    return obs[off:off + n]


def test_crypto_perp_obs_fp16_safe_on_dust_volume():
    env = _perp_on_a_dust_bar()
    assert_fp16_safe(env._get_obs(), "dust bar: ")


def test_crypto_perp_participation_cap_binds_at_the_source():
    env = _perp_on_a_dust_bar()
    block = _perp_cost_block(env, env._get_obs())
    assert np.abs(block).max() > 0.0, "test is vacuous — cost feature is empty"
    assert np.abs(block).max() <= OBS_CLIP


def test_crypto_perp_obs_clip_is_an_independent_backstop(monkeypatch):
    env = _perp_on_a_dust_bar()
    monkeypatch.setattr(perp_mod, "MAX_OBS_PARTICIPATION", np.inf)
    raw = _perp_cost_block(env, env._get_obs().copy())
    assert np.abs(raw).max() >= OBS_CLIP, (
        "unbounded participation no longer overflows — the scenario has gone stale")
    assert_fp16_safe(env._get_obs(), "no-source-bound: ")


def test_crypto_perp_healthy_obs_is_untouched():
    env = _make_perp_env()
    env.reset()
    peak = 0.0
    for _ in range(40):
        obs, _r, term, trunc, _i = env.step(np.full(env.n_assets, 0.3, dtype=np.float32))
        peak = max(peak, float(np.abs(obs).max()))
        if term or trunc:
            break
    assert peak < 100.0, f"healthy perp obs reached {peak:.4g} — the clip would be semantic"
