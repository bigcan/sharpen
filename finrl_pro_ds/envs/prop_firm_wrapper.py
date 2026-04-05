"""Prop Firm Challenge Wrapper — V7-Compatible (Dict + Flat Obs).

Gymnasium wrapper that imposes prop firm evaluation constraints on any
base environment.  Supports both flat ``Box`` observations (crypto envs)
and ``Dict`` observations (ContinuousSwingEnv V7 multi-scale).

Adds:
- EOD trailing drawdown (floor updates at session close, not tick-by-tick)
- Daily loss limit (terminate if daily loss exceeds threshold)
- Profit target (early termination on success)
- Reward shaping: soft penalty for approaching drawdown limits
- Observation augmentation: 3 extra dims for constraint awareness

Compatible with any base env that:
1. Returns ``portfolio_value`` in its ``info`` dict.
2. Has a ``timestamps`` array (int64 epoch-seconds or datetime64).
3. Has a ``step_idx`` attribute.
4. Uses either a flat ``Box`` or a ``Dict`` observation space.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


def _epoch_to_utc_date(epoch_s: int) -> int:
    """Return an integer YYYYMMDD from epoch seconds (UTC)."""
    dt = datetime.fromtimestamp(int(epoch_s), tz=timezone.utc)
    return dt.year * 10000 + dt.month * 100 + dt.day


def _ts_to_epoch(ts_val: Any) -> int:
    """Convert a timestamp value (int64 epoch or datetime64) to epoch seconds."""
    if isinstance(ts_val, (np.datetime64,)):
        return int(ts_val.astype("datetime64[s]").astype("int64"))
    return int(ts_val)


class PropFirmWrapperV7(gym.Wrapper):
    """Gymnasium wrapper imposing prop firm challenge constraints.

    Supports both flat Box and Dict observation spaces.

    Parameters
    ----------
    env : gym.Env
        Base environment (ContinuousSwingEnv V7, CryptoPerpEnv, etc.).
    profit_target_pct : float
        Cumulative return target to pass the challenge (e.g. 0.10 = 10%).
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
    success_bonus : float
        One-time reward bonus when profit target is reached.
    augment_obs : bool
        If True, append 3 constraint-awareness dims to observation.
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
        success_bonus: float = 10.0,
        augment_obs: bool = True,
    ) -> None:
        super().__init__(env)

        self.profit_target_pct = float(profit_target_pct)
        self.max_trailing_dd_pct = float(max_trailing_drawdown_pct)
        self.max_daily_loss_pct = float(max_daily_loss_pct)
        self.eod_hour_utc = int(eod_hour_utc)
        self.dd_penalty_start = float(drawdown_penalty_start)
        self.dd_penalty_scale = float(drawdown_penalty_scale)
        self.success_bonus = float(success_bonus)
        self.augment_obs = augment_obs

        # Detect obs type and extend space if augmenting
        self._dict_obs = isinstance(env.observation_space, gym.spaces.Dict)

        if self.augment_obs:
            if self._dict_obs:
                # Extend the 'private' sub-space by 3 dims
                assert isinstance(env.observation_space, gym.spaces.Dict)
                spaces = dict(env.observation_space.spaces)
                private_space = spaces["private"]
                assert private_space.shape is not None, "private space must have shape"
                old_dim = private_space.shape[0]
                spaces["private"] = gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(old_dim + 3,), dtype=np.float32,
                )
                self.observation_space = gym.spaces.Dict(spaces)
            else:
                # Flat Box obs — extend by 3
                base_shape = env.observation_space.shape
                assert base_shape is not None and len(base_shape) == 1, (
                    "PropFirmWrapperV7 requires flat Box or Dict obs"
                )
                new_dim = base_shape[0] + 3
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(new_dim,), dtype=np.float32,
                )

        # State (reset in reset())
        self._initial_capital: float = 0.0
        self._peak_eod_equity: float = 0.0
        self._last_eod_date: int = 0  # YYYYMMDD
        self._daily_start_equity: float = 0.0
        self._daily_date: int = 0  # YYYYMMDD
        self._current_equity: float = 0.0

    def _get_timestamp_epoch(self) -> int | None:
        """Extract current bar's epoch seconds from the base env."""
        timestamps = getattr(self.env, "timestamps", None)
        step_idx = getattr(self.env, "step_idx", None)
        if timestamps is not None and step_idx is not None and step_idx > 0:
            if step_idx <= len(timestamps):
                return _ts_to_epoch(timestamps[step_idx - 1])
        return None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Initialize from base env
        self._initial_capital = float(
            info.get("portfolio_value", getattr(self.env, "initial_capital", 100_000.0))
        )
        self._peak_eod_equity = self._initial_capital
        self._current_equity = self._initial_capital
        self._daily_start_equity = self._initial_capital

        # Initialize date tracking from first timestamp
        ts_epoch = self._get_timestamp_epoch()
        if ts_epoch is not None:
            date_int = _epoch_to_utc_date(ts_epoch)
            self._last_eod_date = date_int
            self._daily_date = date_int
        else:
            self._last_eod_date = 0
            self._daily_date = 0

        if self.augment_obs:
            obs = self._augment(obs)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # If base env already terminated, pass through
        if terminated:
            if self.augment_obs:
                obs = self._augment(obs)
            return obs, reward, terminated, truncated, info

        # --- Extract equity ---
        self._current_equity = float(info.get("portfolio_value", self._current_equity))

        # --- Timestamp tracking ---
        current_date = 0
        ts_epoch = self._get_timestamp_epoch()
        if ts_epoch is not None:
            current_date = _epoch_to_utc_date(ts_epoch)

        # --- EOD trailing drawdown ---
        # Floor only ratchets up at end-of-day boundary crossings
        if current_date != self._last_eod_date and self._last_eod_date > 0:
            # Day boundary crossed: update EOD peak from PREVIOUS day's closing equity
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
                f"PropFirm: EOD trailing drawdown {eod_drawdown:.4f} > "
                f"{self.max_trailing_dd_pct:.4f} — challenge FAILED"
            )

        # --- Daily loss limit ---
        if self.max_daily_loss_pct > 0 and not terminated:
            if current_date != self._daily_date and self._daily_date > 0:
                # New day: reset daily tracking
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
                    f"PropFirm: Daily loss {daily_loss:.4f} > "
                    f"{self.max_daily_loss_pct:.4f} — challenge FAILED"
                )

        # --- Profit target ---
        cumulative_return = (self._current_equity - self._initial_capital) / self._initial_capital
        if cumulative_return >= self.profit_target_pct and not terminated:
            terminated = True
            reward += self.success_bonus
            info["prop_firm_termination"] = "profit_target_reached"
            info["challenge_passed"] = True
            info["cumulative_return"] = cumulative_return
            logger.info(
                f"PropFirm: Profit target reached! Return {cumulative_return:.4f} >= "
                f"{self.profit_target_pct:.4f} — challenge PASSED"
            )

        # --- Reward shaping: drawdown proximity penalty ---
        if not terminated and self.dd_penalty_scale > 0 and eod_drawdown > self.dd_penalty_start:
            dd_frac = (eod_drawdown - self.dd_penalty_start) / (
                self.max_trailing_dd_pct - self.dd_penalty_start + 1e-10
            )
            penalty = -self.dd_penalty_scale * dd_frac * dd_frac
            reward += penalty

        # --- Prop firm info ---
        info["eod_drawdown"] = eod_drawdown
        info["eod_peak_equity"] = self._peak_eod_equity
        info["cumulative_return"] = cumulative_return
        info["profit_progress"] = (
            min(1.0, cumulative_return / self.profit_target_pct)
            if self.profit_target_pct > 0 else 0.0
        )
        info["drawdown_budget_remaining"] = max(0.0, self.max_trailing_dd_pct - eod_drawdown)

        if self.augment_obs:
            obs = self._augment(obs)

        return obs, reward, terminated, truncated, info

    def _compute_extra(self) -> np.ndarray:
        """Compute the 3 prop-firm constraint dims.

        Returns
        -------
        np.ndarray of shape (3,):
            [0] drawdown_remaining_frac: (max_dd - current_dd) / max_dd  in [0, 1]
            [1] daily_loss_remaining_frac: (max_daily - current_daily) / max_daily  in [0, 1]
            [2] profit_progress_frac: cumulative_return / profit_target  in [0, 1]
        """
        # Drawdown remaining
        eod_dd = (
            1.0 - self._current_equity / self._peak_eod_equity
            if self._peak_eod_equity > 0 else 0.0
        )
        dd_remaining = (
            max(0.0, (self.max_trailing_dd_pct - eod_dd) / self.max_trailing_dd_pct)
            if self.max_trailing_dd_pct > 0 else 1.0
        )

        # Daily loss remaining
        if self.max_daily_loss_pct > 0 and self._daily_start_equity > 0:
            daily_loss = 1.0 - self._current_equity / self._daily_start_equity
            daily_remaining = max(
                0.0, (self.max_daily_loss_pct - daily_loss) / self.max_daily_loss_pct
            )
        else:
            daily_remaining = 1.0

        # Profit progress
        cum_ret = (
            (self._current_equity - self._initial_capital) / self._initial_capital
            if self._initial_capital > 0 else 0.0
        )
        profit_progress = (
            float(np.clip(cum_ret / self.profit_target_pct, 0.0, 1.0))
            if self.profit_target_pct > 0 else 0.0
        )

        return np.array([dd_remaining, daily_remaining, profit_progress], dtype=np.float32)

    def _augment(self, obs: dict | np.ndarray) -> dict | np.ndarray:
        """Append 3 prop-firm constraint dims to the observation."""
        extra = self._compute_extra()

        if self._dict_obs:
            # Dict obs: extend 'private' key
            obs = dict(obs)  # shallow copy
            obs["private"] = np.concatenate([obs["private"], extra])
            return obs
        else:
            # Flat obs
            return np.concatenate([obs, extra])
