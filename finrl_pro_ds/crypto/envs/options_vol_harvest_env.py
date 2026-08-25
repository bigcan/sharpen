"""OptionsVolHarvestEnv — crypto-options variance-risk-premium harvester (Phase 3).

Realizes the Architect design in
``.agent/artifacts/options_vol_harvest_architecture.md`` for the FREE-DATA scope:
**BTC-only**, a single **constant-maturity ATM straddle** repriced from the DVOL
implied-vol index (the only honest daily substrate the free Deribit API affords —
no per-option chain ⇒ no skew/strangle leg; that is a drop-in K=2 extension once
paid chain data is bought). The env's structural core is byte-for-byte the
validated Phase-1 linear book (``scripts/research/options_vrp_falsification.py``):
a rolling, daily-delta-hedged short straddle, net of Deribit option fees + perp
taker + funding carry + a modelled vol-point spread, priced with the SHARED
``crypto.options_pricing`` module. That sharing is what makes the baseline-parity
test exact — feed the NEUTRAL action every bar and the env reproduces the linear
core's equity curve.

What the RL adds on top of that frozen structure (ADR-3):
  * ``vol_conviction`` ∈ [-1, 1] — modulates HOW MUCH vol to sell at each roll
    (negative = short = harvest VRP; positive = long vol, if ``allow_long_vol``).
    The neutral value ``-base_premium_frac/max_premium_frac`` reproduces the
    linear core's fixed sizing.
  * ``hedge_residual`` ∈ [-1, 1] — cost-aware partial-hedge control applied EVERY
    bar over the structural full delta-hedge (the deep-hedging value-add: half the
    rehedge cost at similar risk). ``0`` = full structural hedge = the linear core.

Short vol = short gamma = crash-exposed, so tail safety is a set of HARD caps
(``max_gross_premium_frac``, ``max_net_vega_pct``) enforced by *clipping the
realized position* — a reward-seeking policy cannot trade through them (ADR-6) —
plus a CVaR downside term in the reward.

Invariants: LEAK-2 (action at bar ``t`` decides; option/hedge/funding PnL accrues
``t -> t+1``; rolls executed with info ``<= t``), SHORT-ACCT (short-straddle PnL =
premium received − Δmark; no ``notional_debt`` ever accrues), DATA-CLEAN (upstream
loader), no ``print`` (logging only). New class; touches no existing env (ADR-1).
"""

from __future__ import annotations

import logging
from typing import Optional

import gymnasium as gym
import numpy as np

from finrl_pro_ds.crypto.options_pricing import (
    ANN,
    straddle_delta,
    straddle_gamma,
    straddle_price,
    straddle_vega,
)
from finrl_pro_ds.envs.dsr import DSRCalculator
from finrl_pro_ds.envs.obs_guard import sanitize_obs

logger = logging.getLogger(__name__)

# NAN-01 source bounds.
#
# Short gamma means `equity` can go NEGATIVE in a crash bar, and the obs for that bar
# is built and pushed into the replay buffer before `terminated` is honoured. The
# greek features (premium frac, vega, $-gamma, residual delta) are all divided by
# equity; the old `max(equity, 1e-6)` floor turned a ruin step into ~1e10 — finite in
# float64, `inf` the instant fp16 AMP casts it, then a per-row NaN out of the
# encoder's LayerNorm and an invalid `loc` in the SAC actor's Normal. The vega/premium
# caps are enforced against equity AT THE ROLL, so they do not bound this ratio at the
# current bar. Floor the denominator at a fraction of initial capital instead — the
# termination threshold is already 0.01 * initial_capital, so this can only bind on a
# step the episode is ending on anyway.
EQ_FLOOR_FRAC = 0.01

# `r.mean() / sd` with sd barely above the old 1e-12 floor reaches ~1e9. A running
# Sharpe is meaningless past this band; saturate instead of emitting a huge finite.
SHARPE_CLIP = 100.0


class OptionsVolHarvestEnv(gym.Env):
    """Single-asset (BTC) constant-maturity short-straddle VRP harvester.

    Observation (flat float32, ``OBS_DIM`` dims):
        [ portfolio_value_pct,                 # 1  equity / initial_capital
          atm_iv, iv_rv_spread, rv_ref, funding,   # 4  causal vol signals (<= t)
          days_to_roll_frac,                   # 1  position age / roll_days
          position_premium_frac,               # 1  signed exposure (m*V_t / equity)
          net_vega_pct, net_gamma_pct, net_delta_residual_pct,  # 3  greeks / equity
          baseline_premium_frac,               # 1  ADR-4: the linear core's fixed size
          running_sharpe, running_drawdown ]   # 2

    Action ``Box(-1, 1, (2,))`` = ``[vol_conviction, hedge_residual]``.
    """

    metadata = {"render_modes": ["human"]}

    OBS_DIM = 13

    def __init__(
        self,
        *,
        spot: np.ndarray,                 # (T,) perp close (hedge / PnL leg)
        iv: np.ndarray,                   # (T,) ATM 30d implied vol (fraction)
        iv_rv_spread: np.ndarray,         # (T,) atm_iv - rv_ref (the VRP signal; obs only)
        rv_ref: np.ndarray,               # (T,) trailing realized vol (obs only)
        funding: np.ndarray,              # (T,) daily perp funding (fraction)
        timestamps: np.ndarray,           # (T,) int64 epoch-ms
        initial_capital: float = 100_000.0,
        # --- structural book (matches Phase-1 SimConfig) ---
        base_premium_frac: float = 0.05,      # linear-core size; the NEUTRAL conviction maps here
        max_premium_frac: float = 0.10,       # |target premium frac| at |conviction|=1
        roll_days: int = 21,
        entry_tenor_days: int = 30,
        # --- costs (Deribit schedule; matches Phase-1) ---
        option_fee_pct_underlying: float = 0.0003,
        option_fee_cap_pct_premium: float = 0.125,
        perp_taker_fee: float = 0.0005,
        option_spread_vol_pts: float = 1.0,
        # --- RL controls ---
        allow_long_vol: bool = True,
        hedge_residual_gain: float = 1.0,     # hedge_frac = clip(1 + gain*residual, 0, hedge_frac_max)
        hedge_frac_max: float = 2.0,
        # --- HARD risk caps (ADR-6) — config-bound, never hardcoded gates ---
        max_gross_premium_frac: float = 0.30,
        max_net_vega_pct: float = 0.50,       # |m*vega| / equity ceiling
        # --- reward ---
        reward_type: str = "dsr_cvar",        # {"dsr", "dsr_cvar", "sortino", "simple"}
        dsr_eta: float = 0.01,
        cvar_alpha: float = 0.05,
        cvar_penalty: float = 0.5,
        cvar_window: int = 63,
        turnover_penalty: float = 0.001,
        reward_scaling: float = 1.0,
        reward_clip_range: tuple = (-5.0, 5.0),
        random_start: bool = False,
        random_start_pct: float = 0.1,
    ) -> None:
        super().__init__()

        spot = np.asarray(spot, dtype=np.float64).ravel()
        iv = np.asarray(iv, dtype=np.float64).ravel()
        T = spot.shape[0]
        for name, arr in (("iv", iv), ("iv_rv_spread", iv_rv_spread),
                          ("rv_ref", rv_ref), ("funding", funding),
                          ("timestamps", timestamps)):
            assert np.asarray(arr).shape == (T,), f"{name} must be 1-D length {T}"
        assert T >= 4, "need >= 4 bars"

        self.spot = spot
        self.iv = iv
        self.iv_rv_spread = np.asarray(iv_rv_spread, dtype=np.float64).ravel()
        self.rv_ref = np.asarray(rv_ref, dtype=np.float64).ravel()
        self.funding = np.asarray(funding, dtype=np.float64).ravel()
        self.timestamps = np.asarray(timestamps, dtype=np.int64).ravel()
        self.max_step = T - 1

        self.initial_capital = float(initial_capital)
        self.base_premium_frac = float(base_premium_frac)
        self.max_premium_frac = float(max_premium_frac)
        self.roll_days = int(roll_days)
        self.entry_tenor_days = int(entry_tenor_days)
        self.option_fee_pct_underlying = float(option_fee_pct_underlying)
        self.option_fee_cap_pct_premium = float(option_fee_cap_pct_premium)
        self.perp_taker_fee = float(perp_taker_fee)
        self.option_spread_vol_pts = float(option_spread_vol_pts)
        self.allow_long_vol = bool(allow_long_vol)
        self.hedge_residual_gain = float(hedge_residual_gain)
        self.hedge_frac_max = float(hedge_frac_max)
        self.max_gross_premium_frac = float(max_gross_premium_frac)
        self.max_net_vega_pct = float(max_net_vega_pct)
        self.reward_type = reward_type
        self.dsr_eta = float(dsr_eta)
        self.cvar_alpha = float(cvar_alpha)
        self.cvar_penalty = float(cvar_penalty)
        self.cvar_window = int(cvar_window)
        self.turnover_penalty = float(turnover_penalty)
        self.reward_scaling = float(reward_scaling)
        self.reward_clip_range = reward_clip_range
        self.random_start = bool(random_start)
        self.random_start_pct = float(random_start_pct)

        self.tau_step = 1.0 / ANN
        self.entry_tau = self.entry_tenor_days / ANN
        # Conviction that reproduces the linear core's fixed sizing (ADR-4 fixed point).
        self.neutral_conviction = -self.base_premium_frac / self.max_premium_frac

        self._dsr = DSRCalculator(eta=self.dsr_eta, scale=self.reward_scaling)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
        )
        self._obs_buffer = np.empty(self.OBS_DIM, dtype=np.float32)

        # runtime state (set in reset)
        self.step_idx = 0
        self.equity = 0.0
        self.m = 0.0          # signed straddle count (>0 SHORT, <0 LONG)
        self.K = 0.0
        self.tau = 0.0
        self.q_prev = 0.0     # previous perp hedge qty (coin)
        self.days_held = 0
        self.initial_open_cost = 0.0
        self.cumulative_fees = 0.0
        self.returns_history: list[float] = []
        self.eq_peak = self.initial_capital

    # -----------------------------------------------------------------------
    # cost helpers (mirror options_vrp_falsification._option_fees/_spread_cost)
    # -----------------------------------------------------------------------
    def _option_fees(self, n_abs: float, S: float, premium_total_abs: float) -> float:
        """Deribit option fee for opening/closing ``n_abs`` straddles (2 legs each)."""
        fee_underlying = 2.0 * n_abs * self.option_fee_pct_underlying * S
        fee_premium_cap = self.option_fee_cap_pct_premium * premium_total_abs
        return min(fee_underlying, fee_premium_cap)

    def _spread_cost(self, n_abs: float, S: float, sigma: float, tau: float) -> float:
        """Half-spread (vol points -> USD via vega) for one side of the trade."""
        vega = straddle_vega(S, S, sigma, tau)
        return n_abs * vega * (self.option_spread_vol_pts / 100.0) * 0.5

    # -----------------------------------------------------------------------
    def _sized_position(self, conviction: float, S: float, sigma: float) -> float:
        """Signed straddle count from conviction, AFTER hard caps (clip — ADR-6).

        ``target_premium_frac = -conviction * max_premium_frac`` (>0 = short).
        Gross-premium cap clips the target frac; net-vega cap then scales the count.
        """
        if not self.allow_long_vol:
            conviction = min(conviction, 0.0)
        target_premium_frac = -conviction * self.max_premium_frac
        # HARD cap 1: gross premium fraction
        target_premium_frac = float(np.clip(
            target_premium_frac, -self.max_gross_premium_frac, self.max_gross_premium_frac))
        unit_prem = straddle_price(S, S, sigma, self.entry_tau)
        if unit_prem <= 0 or self.equity <= 0:
            return 0.0
        m = target_premium_frac * self.equity / unit_prem
        # HARD cap 2: net vega per equity
        vega = straddle_vega(S, S, sigma, self.entry_tau)
        if vega > 0:
            vega_pct = abs(m) * vega / self.equity
            if vega_pct > self.max_net_vega_pct:
                m *= self.max_net_vega_pct / vega_pct
        return m

    def _open(self, t: int, conviction: float | None) -> float:
        """Open a new constant-maturity straddle at bar ``t`` and return the open
        cost booked. ``conviction=None`` uses the base (linear-core) size — only at
        the initial reset open."""
        S, sigma = self.spot[t], self.iv[t]
        self.K = S
        self.tau = self.entry_tau
        if conviction is None:
            unit_prem = straddle_price(S, S, sigma, self.entry_tau)
            self.m = (self.base_premium_frac * self.equity / unit_prem) if unit_prem > 0 else 0.0
        else:
            self.m = self._sized_position(conviction, S, sigma)
        prem_total_abs = abs(self.m) * straddle_price(S, S, sigma, self.entry_tau)
        cost = self._option_fees(abs(self.m), S, prem_total_abs) + \
            self._spread_cost(abs(self.m), S, sigma, self.entry_tau)
        # Perp-hedge establishment / roll-jump taker fee (V2-05): the hedge is traded
        # from its prior level (``q_prev`` — the leftover hedge from the just-closed
        # straddle, or 0 at the reset open) to the new straddle's delta hedge. The
        # in-step rehedge fee only covers intra-hold delta adjustments, so without
        # this the establishment leg of every roll was free. Mirrors
        # ``options_vrp_falsification.open_straddle`` so baseline parity still holds.
        q_new = self.m * straddle_delta(S, self.K, sigma, self.entry_tau)
        cost += self.perp_taker_fee * abs(q_new - self.q_prev) * S
        self.equity -= cost
        self.cumulative_fees += cost
        # initial hedge so that q_t == q_prev on the next bar (zero rehedge at open)
        self.q_prev = q_new
        self.days_held = 0
        return cost

    def _close(self, t: int) -> float:
        """Close the active straddle at bar ``t``; return the cost booked."""
        if abs(self.m) <= 1e-12:
            return 0.0
        S, sigma = self.spot[t], self.iv[t]
        tau = max(self.tau, self.tau_step)
        V = straddle_price(S, self.K, sigma, tau)
        cost = self._option_fees(abs(self.m), S, abs(self.m) * V) + \
            self._spread_cost(abs(self.m), S, sigma, tau)
        self.equity -= cost
        self.cumulative_fees += cost
        self.m = 0.0
        return cost

    # -----------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if self.random_start and self.max_step > 2:
            max_offset = int(self.max_step * self.random_start_pct)
            t0 = int(self.np_random.integers(0, max_offset)) if max_offset > 0 else 0
        else:
            t0 = 0
        # advance to first finite/positive-iv bar
        while t0 < self.max_step - 1 and not (
            np.isfinite(self.spot[t0]) and np.isfinite(self.iv[t0]) and self.iv[t0] > 0
        ):
            t0 += 1
        self.step_idx = t0
        self.equity = self.initial_capital
        self.eq_peak = self.initial_capital
        self.cumulative_fees = 0.0
        self.m = 0.0
        self.K = 0.0
        self.tau = 0.0
        self.q_prev = 0.0
        self.days_held = 0
        self.returns_history = []
        self._dsr.reset()
        self.initial_open_cost = self._open(t0, conviction=None)
        return self._get_obs(), {"initial_open_cost": self.initial_open_cost}

    # -----------------------------------------------------------------------
    def step(self, action: np.ndarray):
        if hasattr(action, "cpu"):
            action = action.cpu()
        action = np.asarray(action, dtype=np.float64).ravel()
        np.nan_to_num(action, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        action = action.clip(-1.0, 1.0)
        vol_conviction = float(action[0])
        hedge_residual = float(action[1])

        t = self.step_idx
        S_t, sig_t = self.spot[t], self.iv[t]
        S_n, sig_n = self.spot[t + 1], self.iv[t + 1]
        if not (np.isfinite(S_n) and np.isfinite(sig_n) and sig_n > 0):
            S_n, sig_n = S_t, sig_t

        portfolio_value_before = self.equity

        # 1) hedge for today (structural full delta-hedge × RL residual fraction)
        hedge_frac = float(np.clip(1.0 + self.hedge_residual_gain * hedge_residual,
                                   0.0, self.hedge_frac_max))
        q_struct = self.m * straddle_delta(S_t, self.K, sig_t, max(self.tau, self.tau_step))
        q_t = hedge_frac * q_struct
        rehedge_cost = self.perp_taker_fee * abs(q_t - self.q_prev) * S_t

        # 2) advance one day: option MTM (short gains when value falls), hedge PnL, funding
        V_t = straddle_price(S_t, self.K, sig_t, max(self.tau, self.tau_step))
        tau_next = max(self.tau - self.tau_step, self.tau_step)
        V_n = straddle_price(S_n, self.K, sig_n, tau_next)
        option_pnl = self.m * (V_t - V_n)
        hedge_pnl = q_t * (S_n - S_t)
        funding_pnl = -q_t * S_t * self.funding[t]

        step_pnl = option_pnl + hedge_pnl - rehedge_cost + funding_pnl
        self.cumulative_fees += rehedge_cost
        self.equity += step_pnl
        self.q_prev = q_t
        self.tau = tau_next
        self.days_held += 1
        self.step_idx = t + 1

        # 3) roll if held long enough or tenor too short (structural — ADR-3)
        roll_cost = 0.0
        rolled = False
        if (self.days_held >= self.roll_days
                or self.tau <= (self.entry_tenor_days - self.roll_days - 1) / ANN):
            roll_cost += self._close(self.step_idx)
            if self.equity > 0:
                roll_cost += self._open(self.step_idx, conviction=vol_conviction)
            rolled = True
        step_pnl -= roll_cost

        # --- step return + reward ---
        if portfolio_value_before > 1e-6:
            step_return = (self.equity - portfolio_value_before) / portfolio_value_before
        else:
            step_return = 0.0
        self.returns_history.append(step_return)
        if len(self.returns_history) > self.cvar_window * 4:
            self.returns_history = self.returns_history[-self.cvar_window * 2:]

        self.eq_peak = max(self.eq_peak, self.equity)
        # Turnover penalty argument: option premium (re)traded this step (the costly
        # axis). Only rolls trade options; the cash cost is already booked into equity,
        # this just shapes the reward away from over-trading (the 7d-roll NO-GO lesson).
        turnover = self._roll_turnover_frac if rolled else 0.0
        reward = self._calc_reward(step_return, turnover)

        terminated = self.equity < 0.01 * self.initial_capital
        truncated = self.step_idx >= self.max_step

        net_vega = self.m * straddle_vega(self.spot[self.step_idx], self.K,
                                          self.iv[self.step_idx],
                                          max(self.tau, self.tau_step))
        info = {
            "portfolio_value": self.equity,
            "step_pnl": step_pnl,   # total equity change this step (daily PnL minus roll cost)
            "step_return": step_return,
            "option_pnl": option_pnl,
            "hedge_pnl": hedge_pnl,
            "funding_pnl": funding_pnl,
            "rehedge_cost": rehedge_cost,
            "roll_cost": roll_cost,
            "rolled": rolled,
            "position_m": self.m,
            "net_vega": net_vega,
            "net_delta_residual": self.m * straddle_delta(
                self.spot[self.step_idx], self.K, self.iv[self.step_idx],
                max(self.tau, self.tau_step)) - self.q_prev,
            "cumulative_fees": self.cumulative_fees,
            "hedge_frac": hedge_frac,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    @property
    def _roll_turnover_frac(self) -> float:
        """Premium fraction (re)traded at a roll: |close| + |open| ~ 2*base size.

        Used only as the turnover penalty's argument; the actual cash cost is
        already booked into equity. Approximated by the current gross premium frac.
        """
        S = self.spot[self.step_idx]
        sigma = self.iv[self.step_idx]
        unit_prem = straddle_price(S, S, sigma, self.entry_tau)
        prem_frac = abs(self.m) * unit_prem / max(self.equity, 1e-6)
        return 2.0 * prem_frac

    # -----------------------------------------------------------------------
    def _calc_reward(self, step_return: float, turnover_frac: float) -> float:
        """``base(step_return) - turnover_penalty*turnover - cvar_penalty*cvar_excess``.

        ``dsr``/``dsr_cvar``: Moody-Saffell DSR (shared calculator). The ``_cvar``
        variant adds a Rockafellar-Uryasev CVaR-tail penalty on the rolling return
        buffer to punish short-gamma crash tails (raw Sharpe under-penalizes
        negative skew). Verified by the Math skill — see test_options_reward.py.
        """
        if self.reward_type in ("dsr", "dsr_cvar"):
            reward = self._dsr.compute(step_return)
        elif self.reward_type == "sortino":
            reward = self._sortino_reward(step_return)
        else:
            reward = step_return * self.reward_scaling

        if self.reward_type == "dsr_cvar" and self.cvar_penalty > 0:
            reward -= self.cvar_penalty * self._cvar_excess(step_return)

        reward -= self.turnover_penalty * turnover_frac

        clip_lo, clip_hi = self.reward_clip_range
        return max(clip_lo, min(clip_hi, reward))

    def _cvar_excess(self, step_return: float) -> float:
        """Rockafellar-Uryasev CVaR surrogate excess loss for THIS step's return.

        Let loss ``L = -step_return`` and ``VaR_a`` be the (1-alpha) empirical
        quantile of the trailing loss buffer (a loss threshold). The RU surrogate
        contribution is ``max(0, L - VaR_a)`` — the amount by which this step's loss
        pierces the tail boundary. Returns 0 during warmup. Verified by Math skill.
        """
        n = len(self.returns_history)
        if n < max(10, int(1.0 / max(self.cvar_alpha, 1e-6))):
            return 0.0
        window = min(self.cvar_window, n)
        losses = -np.asarray(self.returns_history[-window:], dtype=np.float64)
        var_a = float(np.quantile(losses, 1.0 - self.cvar_alpha))
        loss = -step_return
        # RU surrogate excess, scaled by 1/alpha. The 1/alpha is the faithful
        # Rockafellar-Uryasev factor (CVaR_a = VaR_a + (1/alpha)*E[(L-VaR_a)+]); it
        # also lifts the fractional excess toward the *dimensionless, scale-invariant*
        # DSR's O(1) range so the penalty actually bites (Math finding CVAR-SCALE).
        # Floor the VaR threshold at 0 so a small GAIN is never penalized in the
        # rare-loss regime where the tail boundary VaR_a is itself negative.
        return max(0.0, loss - max(var_a, 0.0)) / self.cvar_alpha

    def _sortino_reward(self, step_return: float) -> float:
        if len(self.returns_history) < 2:
            return step_return * self.reward_scaling
        window = min(self.cvar_window, len(self.returns_history))
        recent = np.asarray(self.returns_history[-window:])
        downside_sq = np.minimum(recent, 0.0) ** 2
        dd = float(np.sqrt(np.sum(downside_sq) / max(len(downside_sq) - 1, 1)))
        if dd < 1e-8 or not np.isfinite(dd):
            return step_return * self.reward_scaling
        return (step_return / dd) * self.reward_scaling

    # -----------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        t = self.step_idx
        S, sigma = self.spot[t], self.iv[t]
        tau = max(self.tau, self.tau_step)
        buf = self._obs_buffer
        inv_cap = 1.0 / self.initial_capital
        eq = max(self.equity, self.initial_capital * EQ_FLOOR_FRAC)   # NAN-01

        V_t = straddle_price(S, self.K, sigma, tau) if self.K > 0 else 0.0
        vega = straddle_vega(S, self.K, sigma, tau) if self.K > 0 else 0.0
        gamma = straddle_gamma(S, self.K, sigma, tau) if self.K > 0 else 0.0
        delta = straddle_delta(S, self.K, sigma, tau) if self.K > 0 else 0.0

        buf[0] = self.equity * inv_cap
        buf[1] = sigma
        buf[2] = self.iv_rv_spread[t] if np.isfinite(self.iv_rv_spread[t]) else 0.0
        buf[3] = self.rv_ref[t] if np.isfinite(self.rv_ref[t]) else 0.0
        buf[4] = self.funding[t]
        buf[5] = self.days_held / max(self.roll_days, 1)
        buf[6] = self.m * V_t / eq                              # position premium frac (signed)
        buf[7] = self.m * vega / eq                             # net vega / equity
        buf[8] = self.m * gamma * S / eq                        # net $-gamma / equity
        buf[9] = (self.m * delta - self.q_prev) * S / eq        # residual delta / equity
        buf[10] = self.base_premium_frac                        # ADR-4 baseline book weight
        buf[11] = self._running_sharpe()
        buf[12] = float(1.0 - self.equity / self.eq_peak) if self.eq_peak > 0 else 0.0

        return sanitize_obs(buf)

    def _running_sharpe(self) -> float:
        if len(self.returns_history) < 8:
            return 0.0
        r = np.asarray(self.returns_history[-self.cvar_window:], dtype=np.float64)
        sd = r.std(ddof=1)
        if sd < 1e-12 or not np.isfinite(sd):
            return 0.0
        return float(np.clip(r.mean() / sd * np.sqrt(ANN), -SHARPE_CLIP, SHARPE_CLIP))

    # -----------------------------------------------------------------------
    @classmethod
    def from_panels(cls, panels, asset: str = "BTC", *, rv_ref_window: int = 30, **kwargs):
        """Build from an ``options_array_builder.OptionsPanels`` for one asset."""
        if asset not in panels.assets:
            raise ValueError(f"asset {asset!r} not in panels.assets={panels.assets}")
        i = panels.assets.index(asset)
        rv_ary = panels.rv_ary.get(rv_ref_window)
        if rv_ary is None:  # fall back to any available window
            rv_ary = next(iter(panels.rv_ary.values()))
        return cls(
            spot=panels.spot_ary[:, i],
            iv=panels.iv_ary[:, i],
            iv_rv_spread=panels.iv_rv_spread_ary[:, i],
            rv_ref=rv_ary[:, i],
            funding=panels.funding_ary[:, i],
            timestamps=panels.timestamps,
            **kwargs,
        )
