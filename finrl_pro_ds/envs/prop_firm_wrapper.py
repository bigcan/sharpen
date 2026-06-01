"""Prop Firm Challenge Wrapper — V7 (Deprecated Adapter).

Retained for backward-compat. Subclasses :class:`RiskShapingWrapper` and
re-adds the profit-target termination + success_bonus behaviour that the
parent deliberately omits.

The decoupling (see ``.agent/artifacts/prop_firm_decoupling_architecture.md``)
moves profit-target handling into ``ChallengeStateMachine`` on the live path.
New configs should use ``env.risk:`` + the parent wrapper directly.
``scripts/validate_config.py`` warns on ``env.prop_firm:`` at any stage and
rejects it at stage ``paper-deploy``.

``info["profit_progress"]`` is emitted here (not by the parent) with a
``DeprecationWarning`` on first access. Removed in Step 6 once all
workstreams migrate.
"""

from __future__ import annotations

import logging
import warnings

import gymnasium as gym
import numpy as np

from finrl_pro_ds.envs.risk_shaping_wrapper import (
    RiskShapingWrapper,
)

logger = logging.getLogger(__name__)


_PROFIT_PROGRESS_DEPRECATION = (
    "info['profit_progress'] is deprecated — it is emitted only by the "
    "PropFirmWrapperV7 adapter for backward compat and will be removed "
    "when the adapter retires. Migrate to env.risk: + ChallengeStateMachine."
)


class _DeprecatedProfitProgressDict(dict):
    """Dict that emits a DeprecationWarning on 'profit_progress' access.

    Behaves identically to a regular dict for every other key. Used by
    :class:`PropFirmWrapperV7` to surface its continued emission of the
    legacy key without forcing every downstream log-parser to change
    immediately.
    """

    __slots__ = ()

    def __getitem__(self, key):
        if key == "profit_progress":
            warnings.warn(
                _PROFIT_PROGRESS_DEPRECATION,
                DeprecationWarning,
                stacklevel=2,
            )
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "profit_progress":
            warnings.warn(
                _PROFIT_PROGRESS_DEPRECATION,
                DeprecationWarning,
                stacklevel=2,
            )
        return super().get(key, default)


class PropFirmWrapperV7(RiskShapingWrapper):
    """Deprecated adapter — adds profit-target termination + success_bonus.

    All parent (:class:`RiskShapingWrapper`) behaviour is preserved. The
    only additions are:

    - Early termination when cumulative return >= ``profit_target_pct``.
    - One-shot ``success_bonus`` reward on that termination.
    - ``info["profit_progress"]``, ``info["challenge_passed"]`` keys.
    - Legacy ``augment_obs: bool`` API (``True`` → 3 dims = parent's 2 dims
      + ``profit_progress``; ``False`` → no augmentation).

    New code should use :class:`RiskShapingWrapper` + ``ChallengeStateMachine``.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        profit_target_pct: float = 0.10,
        max_trailing_drawdown_pct: float = 0.10,
        max_daily_loss_pct: float = 0.0,
        eod_hour_utc: int = 0,
        drawdown_penalty_start: float = 0.05,
        drawdown_penalty_scale: float = 5.0,
        daily_loss_penalty_start: float = 0.0,
        daily_loss_penalty_scale: float = 0.0,
        success_bonus: float = 10.0,
        augment_obs: bool = True,
        static_peak: bool = True,
    ) -> None:
        # V7 uses a legacy 3-dim augmentation (dd_remaining, daily_remaining,
        # profit_progress). The parent only knows "off" or "2d", so we keep
        # augmentation off at the parent level and handle all 3 dims here.
        self._v7_augment = bool(augment_obs)
        super().__init__(
            env,
            max_trailing_drawdown_pct=max_trailing_drawdown_pct,
            max_daily_loss_pct=max_daily_loss_pct,
            eod_hour_utc=eod_hour_utc,
            drawdown_penalty_start=drawdown_penalty_start,
            drawdown_penalty_scale=drawdown_penalty_scale,
            daily_loss_penalty_start=daily_loss_penalty_start,
            daily_loss_penalty_scale=daily_loss_penalty_scale,
            augment_obs="off",
            static_peak=static_peak,
        )

        self.profit_target_pct = float(profit_target_pct)
        self.success_bonus = float(success_bonus)
        # Keep the old public attribute name for tests that read it directly
        self.augment_obs = augment_obs

        if self._v7_augment:
            if self._dict_obs:
                assert isinstance(env.observation_space, gym.spaces.Dict)
                spaces = dict(env.observation_space.spaces)
                private_space = spaces["private"]
                assert private_space.shape is not None
                old_dim = private_space.shape[0]
                spaces["private"] = gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(old_dim + 3,), dtype=np.float32,
                )
                self.observation_space = gym.spaces.Dict(spaces)
            else:
                base_shape = env.observation_space.shape
                assert base_shape is not None and len(base_shape) == 1, (
                    "PropFirmWrapperV7 requires flat Box or Dict obs"
                )
                new_dim = base_shape[0] + 3
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(new_dim,), dtype=np.float32,
                )

    # ------------------------------------------------------------------
    # step / reset — add profit-target termination + V7-shaped info dict
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        if self._v7_augment:
            obs = self._augment_v7(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Parent already handled DD + daily-loss termination. Only add
        # profit-target termination if the parent did not terminate.
        if not terminated:
            cumulative_return = info.get("cumulative_return", 0.0)
            if cumulative_return >= self.profit_target_pct:
                terminated = True
                reward += self.success_bonus
                info["prop_firm_termination"] = "profit_target_reached"
                info["challenge_passed"] = True
                logger.info(
                    f"PropFirm: Profit target reached! Return {cumulative_return:.4f} >= "
                    f"{self.profit_target_pct:.4f} — challenge PASSED"
                )

        # V7-only info keys. Wrap dict so profit_progress access emits
        # DeprecationWarning.
        cumulative_return = info.get("cumulative_return", 0.0)
        profit_progress = (
            min(1.0, cumulative_return / self.profit_target_pct)
            if self.profit_target_pct > 0 else 0.0
        )
        info["profit_progress"] = profit_progress
        info = _DeprecatedProfitProgressDict(info)

        if self._v7_augment:
            obs = self._augment_v7(obs)

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # V7 3-dim augmentation (parent's 2 dims + profit_progress)
    # ------------------------------------------------------------------

    def _compute_extra_v7(self) -> np.ndarray:
        """3-dim constraint vector: [dd_remaining, daily_remaining, profit_progress]."""
        parent_extra = self._compute_extra()  # (2,) [dd_remaining, daily_remaining]
        cum_ret = (
            (self._current_equity - self._initial_capital) / self._initial_capital
            if self._initial_capital > 0 else 0.0
        )
        profit_progress = (
            float(np.clip(cum_ret / self.profit_target_pct, 0.0, 1.0))
            if self.profit_target_pct > 0 else 0.0
        )
        return np.concatenate([parent_extra, np.array([profit_progress], dtype=np.float32)])

    def _augment_v7(self, obs: dict | np.ndarray) -> dict | np.ndarray:
        extra = self._compute_extra_v7()
        if self._dict_obs:
            obs = dict(obs)
            obs["private"] = np.concatenate([obs["private"], extra])
            return obs
        return np.concatenate([obs, extra])
