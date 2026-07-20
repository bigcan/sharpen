"""Risk-shaping wrapper — phase-invariant DD + daily-loss shaping for training.

Gymnasium wrapper that imposes drawdown and daily-loss shaping on any base
environment. Supports both flat ``Box`` observations (crypto envs) and
``Dict`` observations (ContinuousSwingEnv V7 multi-scale).

This wrapper is the *training-only* half of the prop-firm challenge split
(see ``.agent/artifacts/prop_firm_decoupling_architecture.md``). It does
NOT handle profit targets or success bonuses — those are a live-engine
concern handled by ``ChallengeStateMachine``.

Adds:
- EOD trailing drawdown (floor updates at session close, not tick-by-tick)
- Daily loss limit (terminate if daily loss exceeds threshold)
- Reward shaping: soft penalty for approaching drawdown / daily-loss limits
- Optional observation augmentation: 2 extra dims for risk awareness

Compatible with any base env that:
1. Returns ``portfolio_value`` in its ``info`` dict.
2. Has a ``timestamps`` array (int64 epoch-seconds or datetime64).
3. Has a ``step_idx`` attribute.
4. Uses either a flat ``Box`` or a ``Dict`` observation space.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Literal

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


AugmentMode = Literal["off", "2d"]


def _epoch_to_utc_date(epoch_s: int) -> int:
    """Return an integer YYYYMMDD from epoch seconds (UTC)."""
    dt = datetime.fromtimestamp(int(epoch_s), tz=timezone.utc)
    return dt.year * 10000 + dt.month * 100 + dt.day


def _ts_to_epoch(ts_val: Any) -> int:
    """Convert a timestamp value (int64 epoch or datetime64) to epoch seconds."""
    if isinstance(ts_val, (np.datetime64,)):
        return int(ts_val.astype("datetime64[s]").astype("int64"))
    return int(ts_val)


class RiskShapingWrapper(gym.Wrapper):
    """Phase-invariant risk shaping wrapper.

    Parameters
    ----------
    env : gym.Env
        Base environment (ContinuousSwingEnv V7, CryptoPerpEnv, etc.).
    max_trailing_drawdown_pct : float
        Maximum EOD trailing drawdown before termination (e.g. 0.10 = 10%).
    max_daily_loss_pct : float
        Maximum single-day loss before termination (0.0 = disabled).
    eod_hour_utc : int
        UTC hour that defines end-of-day for drawdown floor updates.
    drawdown_penalty_start : float
        Drawdown fraction at which reward penalty begins.
    drawdown_penalty_scale : float
        Multiplier for the quadratic drawdown penalty.
    daily_loss_penalty_start : float
        Daily-loss fraction at which reward penalty begins (0.0 disables).
    daily_loss_penalty_scale : float
        Multiplier for the quadratic daily-loss penalty (0.0 disables).
    augment_obs : {"off", "2d"}
        Obs-space augmentation mode. "off" = passthrough (production path).
        "2d" = append [dd_remaining, daily_remaining] to private obs.
        The legacy 3-dim mode is explicitly unsupported (see ADR-1).
    static_peak : bool
        If True (default), peak equity locks at ``initial_capital`` for the
        whole episode — matching FTMO Phase 1 Challenge's static Maximum Loss
        rule. If False, peak ratchets up at each EOD boundary.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        max_trailing_drawdown_pct: float = 0.10,
        max_daily_loss_pct: float = 0.0,
        eod_hour_utc: int = 0,
        drawdown_penalty_start: float = 0.05,
        drawdown_penalty_scale: float = 5.0,
        daily_loss_penalty_start: float = 0.0,
        daily_loss_penalty_scale: float = 0.0,
        augment_obs: AugmentMode = "off",
        static_peak: bool = True,
    ) -> None:
        super().__init__(env)

        if augment_obs not in ("off", "2d"):
            raise ValueError(
                f"augment_obs must be 'off' or '2d', got {augment_obs!r}. "
                "The legacy '3d_legacy' mode is explicitly rejected — "
                "see ADR-1 in prop_firm_decoupling_architecture.md."
            )

        self.max_trailing_dd_pct = float(max_trailing_drawdown_pct)
        self.max_daily_loss_pct = float(max_daily_loss_pct)
        self.eod_hour_utc = int(eod_hour_utc)
        self.dd_penalty_start = float(drawdown_penalty_start)
        self.dd_penalty_scale = float(drawdown_penalty_scale)
        self.daily_loss_penalty_start = float(daily_loss_penalty_start)
        self.daily_loss_penalty_scale = float(daily_loss_penalty_scale)
        self.augment_obs: AugmentMode = augment_obs
        self.static_peak = bool(static_peak)

        self._dict_obs = isinstance(env.observation_space, gym.spaces.Dict)
        self._augment_dim = 2 if augment_obs == "2d" else 0

        if self._augment_dim > 0:
            if self._dict_obs:
                assert isinstance(env.observation_space, gym.spaces.Dict)
                spaces = dict(env.observation_space.spaces)
                private_space = spaces["private"]
                assert private_space.shape is not None, "private space must have shape"
                old_dim = private_space.shape[0]
                spaces["private"] = gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(old_dim + self._augment_dim,), dtype=np.float32,
                )
                self.observation_space = gym.spaces.Dict(spaces)
            else:
                base_shape = env.observation_space.shape
                assert base_shape is not None and len(base_shape) == 1, (
                    "RiskShapingWrapper requires flat Box or Dict obs"
                )
                new_dim = base_shape[0] + self._augment_dim
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(new_dim,), dtype=np.float32,
                )

        self._initial_capital: float = 0.0
        self._peak_eod_equity: float = 0.0
        self._last_eod_date: int = 0
        self._daily_start_equity: float = 0.0
        self._daily_date: int = 0
        self._current_equity: float = 0.0

    def _get_timestamp_epoch(self) -> int | None:
        """Extract the current-bar epoch seconds from the base env.

        Reads ``timestamps[step_idx]`` (lookahead semantic) rather than
        ``step_idx - 1``. Both V7 and crypto envs expose ``step_idx`` as
        a data pointer where ``timestamps[step_idx]`` is the timestamp
        of the bar associated with the just-returned ``portfolio_value``.
        This alignment is necessary for EOD-boundary detection: the
        original crypto wrapper behaviour (tests in
        ``tests/crypto/test_prop_firm_wrapper.py``) asserts that the
        trailing-peak ratchets to the previous day's closing equity at
        the moment the new-day timestamp arrives alongside that equity.
        """
        timestamps = getattr(self.env, "timestamps", None)
        step_idx = getattr(self.env, "step_idx", None)
        if timestamps is None or step_idx is None or len(timestamps) == 0:
            return None
        idx = min(int(step_idx), len(timestamps) - 1)
        if idx < 0:
            return None
        return _ts_to_epoch(timestamps[idx])

    def _get_reset_date_int(self) -> int:
        """Return the YYYYMMDD for the first bar of the episode, or 0 if unavailable."""
        ts_epoch = self._get_timestamp_epoch()
        return _epoch_to_utc_date(ts_epoch) if ts_epoch is not None else 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # audit F8: V7 ContinuousSwingEnv exposes `initial_balance` (not the crypto
        # env's `initial_capital`) and returns an empty info dict on reset, so the
        # old fallback silently used the 100k default for any account != 100k,
        # disabling the DD reference. Prefer reset info, then either env attr.
        _env_initial = getattr(self.env, "initial_capital", None)
        if _env_initial is None:
            _env_initial = getattr(self.env, "initial_balance", 100_000.0)
        self._initial_capital = float(info.get("portfolio_value", _env_initial))
        self._peak_eod_equity = self._initial_capital
        self._current_equity = self._initial_capital
        self._daily_start_equity = self._initial_capital

        date_int = self._get_reset_date_int()
        self._last_eod_date = date_int
        self._daily_date = date_int

        if self._augment_dim > 0:
            obs = self._augment(obs)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated:
            if self._augment_dim > 0:
                obs = self._augment(obs)
            return obs, reward, terminated, truncated, info

        self._current_equity = float(info.get("portfolio_value", self._current_equity))

        current_date = 0
        ts_epoch = self._get_timestamp_epoch()
        if ts_epoch is not None:
            current_date = _epoch_to_utc_date(ts_epoch)

        if not self.static_peak:
            if current_date != self._last_eod_date and self._last_eod_date > 0:
                self._peak_eod_equity = max(self._peak_eod_equity, self._current_equity)
                self._last_eod_date = current_date

        eod_drawdown = 0.0
        if self._peak_eod_equity > 0:
            eod_drawdown = 1.0 - self._current_equity / self._peak_eod_equity

        if eod_drawdown > self.max_trailing_dd_pct:
            terminated = True
            info["prop_firm_termination"] = "eod_trailing_drawdown"
            info["eod_drawdown"] = eod_drawdown
            logger.info(
                f"RiskShaping: EOD trailing drawdown {eod_drawdown:.4f} > "
                f"{self.max_trailing_dd_pct:.4f} — episode terminated"
            )

        if self.max_daily_loss_pct > 0 and not terminated:
            if current_date != self._daily_date and self._daily_date > 0:
                self._daily_start_equity = self._current_equity
                self._daily_date = current_date

            daily_loss = 0.0
            if self._daily_start_equity > 0:
                daily_loss = 1.0 - self._current_equity / self._daily_start_equity

            if daily_loss > self.max_daily_loss_pct:
                terminated = True
                info["prop_firm_termination"] = "daily_loss_limit"
                info["daily_loss"] = daily_loss
                logger.info(
                    f"RiskShaping: Daily loss {daily_loss:.4f} > "
                    f"{self.max_daily_loss_pct:.4f} — episode terminated"
                )

        cumulative_return = (
            (self._current_equity - self._initial_capital) / self._initial_capital
            if self._initial_capital > 0 else 0.0
        )

        if not terminated and self.dd_penalty_scale > 0 and eod_drawdown > self.dd_penalty_start:
            dd_frac = (eod_drawdown - self.dd_penalty_start) / (
                self.max_trailing_dd_pct - self.dd_penalty_start + 1e-10
            )
            reward += -self.dd_penalty_scale * dd_frac * dd_frac

        if (not terminated and self.daily_loss_penalty_scale > 0
                and self.max_daily_loss_pct > 0 and self._daily_start_equity > 0):
            daily_loss_frac = 1.0 - self._current_equity / self._daily_start_equity
            if daily_loss_frac > self.daily_loss_penalty_start:
                dl_frac = (daily_loss_frac - self.daily_loss_penalty_start) / (
                    self.max_daily_loss_pct - self.daily_loss_penalty_start + 1e-10
                )
                reward += -self.daily_loss_penalty_scale * dl_frac * dl_frac

        info["eod_drawdown"] = eod_drawdown
        info["eod_peak_equity"] = self._peak_eod_equity
        info["cumulative_return"] = cumulative_return
        info["drawdown_budget_remaining"] = max(0.0, self.max_trailing_dd_pct - eod_drawdown)

        if self._augment_dim > 0:
            obs = self._augment(obs)

        return obs, reward, terminated, truncated, info

    def _compute_extra(self) -> np.ndarray:
        """Compute the 2 risk-shaping constraint dims.

        Returns
        -------
        np.ndarray of shape (2,):
            [0] drawdown_remaining_frac: (max_dd - current_dd) / max_dd  in [0, 1]
            [1] daily_loss_remaining_frac: (max_daily - current_daily) / max_daily  in [0, 1]
        """
        eod_dd = (
            1.0 - self._current_equity / self._peak_eod_equity
            if self._peak_eod_equity > 0 else 0.0
        )
        dd_remaining = (
            max(0.0, (self.max_trailing_dd_pct - eod_dd) / self.max_trailing_dd_pct)
            if self.max_trailing_dd_pct > 0 else 1.0
        )

        if self.max_daily_loss_pct > 0 and self._daily_start_equity > 0:
            daily_loss = 1.0 - self._current_equity / self._daily_start_equity
            daily_remaining = max(
                0.0, (self.max_daily_loss_pct - daily_loss) / self.max_daily_loss_pct
            )
        else:
            daily_remaining = 1.0

        return np.array([dd_remaining, daily_remaining], dtype=np.float32)

    def _augment(self, obs: dict | np.ndarray) -> dict | np.ndarray:
        """Append risk-shaping constraint dims to the observation."""
        extra = self._compute_extra()

        if self._dict_obs:
            obs = dict(obs)
            obs["private"] = np.concatenate([obs["private"], extra])
            return obs
        return np.concatenate([obs, extra])
