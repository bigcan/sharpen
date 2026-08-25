"""Execution-scheduler env — the RL execution overlay over the fixed 2-sleeve target.

Step 2 of the execution-overlay build (``.agent/artifacts/execution_overlay_architecture.md``;
ADR-1..9). The linear 2-sleeve book emits a **fixed** target-weight trajectory
``W_target[k]`` (``TwoSleeveExecutor.sim_oracle`` → ``combined_w``); the snap executor
(``ParityHarness._replay``) books the full gap to ``W_target[k]`` every bar. This env
inserts an RL scheduler that *shapes the trade path* — it works each monthly rebalance's
conviction shift over a short horizon to minimize **implementation shortfall** (timing
drift vs the arrival price + reactive impact cost) — WITHOUT changing the target.

**Control = a per-bar FILL FRACTION (ADR-4, the key structural choice).** The action is a
scalar urgency mapped to ``φ_k ∈ [0,1]`` of the current gap to the (fixed) daily target:

    gap_k       = W_target[k] − W_held
    φ_k         = clip( m_k · baseline_φ_h , 0, 1 ),   baseline_φ_h = 1/(H − h)   (closed-loop TWAP)
    trade_k     = φ_k · gap_k
    W_held     += trade_k          ⇒  W_held_new = (1−φ)·W_held + φ·W_target[k]  (CONVEX combo)

Consequences that make the invariants STRUCTURAL, not learned:
  - **Gross cap (MARGIN-CFG):** a convex combination of two ≤-cap weight vectors is ≤-cap
    (triangle inequality) ⇒ ``gross(W_held) ≤ gross(W_target) ≤ max_gross_exposure`` every
    bar; the overlay only ever *lags or matches* the target, never over-levers/overshoots.
  - **Completion (execution-only):** ``m=1 ⇒ φ=baseline ⇒`` exact TWAP (the known-good
    schedule); at the last horizon bar ``h=H−1`` ``baseline_φ=1`` and the env forces ``φ=1``
    so ``W_held → W_target`` — the target is ALWAYS reached, the overlay only chose the path.
    ``H=1 ⇒`` byte-equivalent to the snap ``_replay``.

**Reward = dense implementation shortfall (ADR-7), direction-free.** Struck vs the arrival
price ``P_arr = price[r]`` (the rebalance decision bar), computed from the executed slices —
NOT from the book's P&L (that directional P&L belongs to the core's target, not the
overlay):

    IS_timing_k = Σ_i trade_k,i · pv_before · (price[k+1,i] − P_arr_i)/P_arr_i      (USD)
    impact_k    = fees_k + slippage_k                                              (USD, incl. reactive)
    reward_k    = −( IS_timing_k + impact_penalty·impact_k ) / parent_notional · is_reward_scale
                  [ − asymmetric_dampen · max(0, −IS_timing_k)/parent_notional · is_reward_scale ]   (Spooner)

A perfect/instant fill gives reward ≈ 0 independent of price direction; the optional
Spooner dampener clips the *profit* from favorable timing (keeps the loss) so the overlay
cannot morph into disguised trend-following. CVaR-aware tail-slippage control comes from the
``DistributionalSACAgent`` (ADR-7); drawdown constraints from ``PropFirmWrapperV7`` (ADR-5).

Reuses ``PaperState`` (the env-faithful book accounting, verbatim) and
``ReactiveSimFillEngine`` (the √-law mean-reverting impact, step 1). Obs is a raw-numpy
DICT ``{"scale_0": market(M,), "private": exec-intrinsic(P,)}`` (ADR-9, ``obs_mode=
summary_stats``); exposes ``observation_space``/``initial_capital``/``timestamps``/
``step_idx``/``info["portfolio_value"]`` so ``PropFirmWrapperV7`` composes (it augments
``private`` with 3 risk-budget dims).

Invariants: LEAK-2 (every feature/arrival reads bars ≤ the decision bar; the reactive
impact is past-liquidity-only), SHORT-ACCT (``PaperState``), MARGIN-CFG (convex-combo
gross cap), completion (forced ``φ=1`` at horizon end). New reward formula ⇒ Math + Audit.
"""
from __future__ import annotations

import logging
from typing import Optional

import gymnasium as gym
import numpy as np

from finrl_pro_ds.envs.obs_guard import sanitize_obs
from finrl_pro_ds.paper.fill_engine import ReactiveSimFillEngine
from finrl_pro_ds.paper.paper_state import PaperState

logger = logging.getLogger(__name__)

_PRICE_EPS = 1e-10
_VOL_EPS = 1e-6
_L1_EPS = 1e-12

# Smallest L1 target shift that counts as a real parent order (NAN-01).
#
# `W_target[r] == W_target[r-1]` (e.g. the signal-warmup months where the linear book's
# lookbacks are unmet and `combined_w` is identically zero) opens a parent order with
# NOTHING to execute. Such an episode is not merely uninformative — every quantity the env
# normalizes BY the parent size (`inventory_remaining`, `schedule_deviation`,
# `cost_so_far_bps`, and the IS reward via `_parent_notional`) divides by ~0 and explodes.
# It is also undefined as an execution decision: overlay and TWAP book the same (zero)
# trade, so the gate's uplift on it is 0/0. Drop those steps from the calendar instead.
MIN_PARENT_L1 = 1e-4

# Private (execution-intrinsic) and market (microstructure-proxy) observation widths.
PRIVATE_DIM = 4
MARKET_DIM = 6


class ExecutionSchedulerEnv(gym.Env):
    """RL execution overlay: shape the trade path to a FIXED target, minimize shortfall.

    One episode = working one rebalance's parent order over ``horizon_bars`` bars. The
    target ``W_target`` and the arrival price are immutable inputs; the agent picks a per-bar
    fill-fraction urgency. Observation (raw-numpy dict):

        private (P=4): [time_remaining, inventory_remaining, schedule_deviation, exec_drawdown]
        scale_0 (M=6): [realized_short_vol, agg_participation, signed_basket_momentum,
                        participation_dispersion, cost_so_far_bps, parent_gross]

    Action: ``Box(-1,1,(1,))`` scalar urgency; ``a=0 → m=1 →`` baseline TWAP slice.

    **Calendar (NAN-01).** A rebalance step is admissible only if it is horizon-valid AND
    opens a real parent order (``‖W_target[r] − W_target[r−1]‖₁ ≥ min_parent_l1``). An empty
    parent order has nothing to schedule, is undefined as an execution decision (overlay and
    TWAP book the same zero trade), and would divide every parent-normalized obs/reward
    quantity by ~0 — producing values that are finite in float64 but ``inf`` under fp16 AMP.
    ``obs_guard.sanitize_obs`` is the independent backstop on the emitted obs.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        *,
        target_weights: np.ndarray,            # (K, U) FIXED W_target (K = T-1)
        price_ary: np.ndarray,                 # (T, U) union close
        volume_ary: np.ndarray,                # (T, U) union DOLLAR volume (F1 denominator)
        carry_ary: np.ndarray,                 # (T, U) union per-bar carry (zeros in v1)
        timestamps: np.ndarray,                # (T,) epoch-seconds int64
        assets: list[str],                     # U union asset names
        rebalance_steps: np.ndarray,           # (R,) int step-indices where a parent opens (>=1)
        initial_capital: float = 100_000.0,
        horizon_bars: int = 5,                 # H — bars to work each parent order
        urgency_min: float = 0.0,              # m at action=-1 (pause)
        urgency_max: float = 2.0,              # m at action=+1 (2x baseline); a=0 -> m=1 (baseline)
        is_reward_scale: float = 1.0e4,        # IS fraction -> bps for the reward
        impact_penalty: float = 1.0,           # alpha on modeled impact cost
        risk_penalty: float = 0.0,             # lambda on exec-drawdown (0 => risk via wrapper + CVaR)
        unexecuted_penalty: float = 50.0,      # beta terminal guard (forced phi=1 makes residual ~0)
        asymmetric_dampen: float = 0.0,        # eta Spooner dampener (0 => off)
        taker_fee_pct: float = 0.0002,
        slippage_base_bps: float = 1.0,
        slippage_impact_bps: float = 5.0,
        reactive_impact_bps: float = 8.0,      # ReactiveSimFillEngine temporary-impact coeff
        reactive_decay: float = 0.5,
        vol_window: int = 20,                  # trailing window for realized_short_vol
        mom_window: int = 5,                   # short-horizon window for signed_basket_momentum
        relative_equity: bool = True,          # info["portfolio_value"] = execution tracking-error equity (ADR-5)
        max_gross_exposure: float = 3.0,
        random_start: bool = True,             # train: start at a random rebalance event
        min_parent_l1: float = MIN_PARENT_L1,  # drop degenerate (empty) parent orders (NAN-01)
    ) -> None:
        super().__init__()

        W = np.asarray(target_weights, dtype=np.float64)
        price = np.asarray(price_ary, dtype=np.float64)
        volume = np.asarray(volume_ary, dtype=np.float64)
        carry = np.asarray(carry_ary, dtype=np.float64)
        ts = np.asarray(timestamps, dtype=np.int64)
        assert price.ndim == 2, "price_ary must be (T, U)"
        T, U = price.shape
        assert W.shape == (T - 1, U), f"target_weights {W.shape} != {(T - 1, U)}"
        for name, arr in (("volume_ary", volume), ("carry_ary", carry)):
            assert arr.shape == (T, U), f"{name} must be (T, U) == {(T, U)}"
        assert ts.shape == (T,), "timestamps must be (T,)"

        self.W_target = W
        self.price = price
        self.volume = volume
        self.carry = carry
        self.timestamps = ts
        self.assets = list(assets) if assets else [f"asset_{i}" for i in range(U)]
        self.n_assets = U
        self.T = T
        self.K = T - 1                                      # number of target steps

        self.initial_capital = float(initial_capital)
        self.H = max(int(horizon_bars), 1)
        self.urgency_min = float(urgency_min)
        self.urgency_max = float(urgency_max)
        self.is_reward_scale = float(is_reward_scale)
        self.impact_penalty = float(impact_penalty)
        self.risk_penalty = float(risk_penalty)
        self.unexecuted_penalty = float(unexecuted_penalty)
        self.asymmetric_dampen = float(asymmetric_dampen)
        self.vol_window = max(int(vol_window), 2)
        self.mom_window = max(int(mom_window), 1)
        self.relative_equity = bool(relative_equity)
        self.max_gross_exposure = float(max_gross_exposure)
        self.random_start = bool(random_start)
        self.min_parent_l1 = max(float(min_parent_l1), 0.0)

        # Valid parent-open steps: need price[r-1..r+H] in range ⇒ r in [1, K-H], AND a
        # non-degenerate parent order ‖W_target[r] − W_target[r−1]‖₁ ≥ min_parent_l1 (NAN-01
        # — an empty parent order has nothing to schedule and makes every parent-normalized
        # quantity divide by ~0; see MIN_PARENT_L1).
        rs = np.asarray(rebalance_steps, dtype=np.int64).ravel()
        lo, hi = 1, self.K - self.H
        in_range = sorted({int(r) for r in rs if lo <= int(r) <= hi})
        self.rebalance_steps = np.array(
            [r for r in in_range
             if float(np.abs(self.W_target[r] - self.W_target[r - 1]).sum()) >= self.min_parent_l1],
            dtype=np.int64)
        n_degenerate = len(in_range) - int(self.rebalance_steps.size)
        if n_degenerate:
            logger.info(
                "ExecutionSchedulerEnv: dropped %d/%d horizon-valid rebalance step(s) with an "
                "empty parent order (L1(dW_target) < %.3g) - nothing to execute there",
                n_degenerate, len(in_range), self.min_parent_l1)
        if self.rebalance_steps.size == 0:
            raise ValueError(
                f"no valid rebalance step in [1, {hi}] with L1(dW_target) >= "
                f"{self.min_parent_l1:g} (H={self.H}, K={self.K}, "
                f"{len(in_range)} in-range but all degenerate); data window too short for the "
                "horizon, or the target never moves in it")

        self.engine = ReactiveSimFillEngine(
            taker_fee_pct=taker_fee_pct, slippage_base_bps=slippage_base_bps,
            slippage_impact_bps=slippage_impact_bps, n_assets=U,
            reactive_impact_bps=reactive_impact_bps, decay=reactive_decay,
        )

        self.observation_space = gym.spaces.Dict({
            "scale_0": gym.spaces.Box(-np.inf, np.inf, shape=(MARKET_DIM,), dtype=np.float32),
            "private": gym.spaces.Box(-np.inf, np.inf, shape=(PRIVATE_DIM,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        # --- pre-allocated obs buffers (avoid per-step allocation) ---
        self._mkt_buf = np.empty(MARKET_DIM, dtype=np.float32)
        self._priv_buf = np.empty(PRIVATE_DIM, dtype=np.float32)

        # --- episode runtime state (set in reset) ---
        self.step_idx = 0
        self._r = 0
        self._h = 0
        self._P_arr: np.ndarray = np.zeros(U, dtype=np.float64)
        self._W_held: np.ndarray = np.zeros(U, dtype=np.float64)
        self._gap0_l1 = 0.0
        self._parent_notional = 1.0
        self._cum_cost = 0.0
        self._rel_equity = self.initial_capital
        self._rel_peak = self.initial_capital
        self.book: PaperState | None = None

    @property
    def W_held(self) -> np.ndarray:
        """The realized held weight vector after the latest :meth:`step` — the lagged path
        the overlay actually books (``W_held = (1−φ)·W_held + φ·W_target[k]`` per bar).
        ``TwoSleeveExecutor.run_with_overlay`` reads this each step to stitch the realized
        trajectory; a live view of the episode's running inventory (caller copies on use)."""
        return self._W_held

    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if options and "rebalance_step" in options:
            r = int(options["rebalance_step"])
            if r not in set(int(x) for x in self.rebalance_steps):
                raise ValueError(f"rebalance_step {r} not in the valid calendar")
        elif self.random_start:
            r = int(self.np_random.choice(self.rebalance_steps))
        else:
            r = int(self.rebalance_steps[0])

        self._r = r
        self._h = 0
        self.step_idx = r
        self._P_arr = self.price[r].copy()
        # Guard the arrival price against zero/invalid (dead asset) — never divide by it.
        self._P_arr = np.where(self._P_arr > _PRICE_EPS, self._P_arr, 1.0)

        # Warm the book to the position held coming INTO the rebalance (the prior target),
        # entered at price[r] so episode-start unrealized = 0 and pv_before(price[r]) = capital.
        held_in = self.W_target[r - 1].copy()
        self._W_held = held_in.copy()
        self.book = self._warm_book(held_in, self.price[r])
        self.engine.reset()

        gap0 = self.W_target[r] - held_in
        self._gap0_l1 = float(np.abs(gap0).sum())
        self._parent_notional = max(self._gap0_l1 * self.initial_capital, 1.0)
        self._cum_cost = 0.0
        self._rel_equity = self.initial_capital
        self._rel_peak = self.initial_capital

        return self._get_obs(), {"rebalance_step": r}

    def _warm_book(self, positions: np.ndarray, price: np.ndarray) -> PaperState:
        """A book holding ``positions`` entered at ``price`` (zero unrealized, margin =
        initial_capital) — the realistic state coming into the rebalance, so the execution
        accounting (and pv_before scaling) is faithful from bar r."""
        book = PaperState(n_assets=self.n_assets, initial_capital=self.initial_capital,
                          assets=self.assets)
        active = np.abs(positions) > _PRICE_EPS
        book.positions = positions.copy()
        book.entry_prices = np.where(active, price, 0.0)
        book.entry_notionals = np.where(active, np.abs(positions) * self.initial_capital, 0.0)
        book.margin_balance = self.initial_capital
        book.peak_equity = self.initial_capital
        return book

    # ------------------------------------------------------------------ #
    def step(self, action: np.ndarray):
        assert self.book is not None, "reset() must be called before step()"
        k = self.step_idx
        h = self._h
        a = float(np.asarray(action, dtype=np.float64).ravel()[0])
        a = min(max(a, -1.0), 1.0)
        if not np.isfinite(a):
            a = 0.0

        # urgency multiplier on the closed-loop TWAP slice; last bar forces full completion.
        m = self.urgency_min + (a + 1.0) * 0.5 * (self.urgency_max - self.urgency_min)
        baseline_phi = 1.0 / float(self.H - h)              # h in [0, H-1] ⇒ denom in [H, 1]
        if h >= self.H - 1:
            phi = 1.0                                       # terminal completion (W_held -> W_target)
        else:
            phi = min(max(m * baseline_phi, 0.0), 1.0)

        price_now = self.price[k + 1]
        prev_price = self.price[k]
        valid_next = price_now > _PRICE_EPS

        gap = self.W_target[k] - self._W_held
        trade = phi * gap
        trade[~valid_next] = 0.0                            # never trade into an invalid next price
        trade[np.abs(trade) < 1e-12] = 0.0

        pv_before = self.book.pv_before(prev_price)
        fill = self.engine.fill(
            delta_weights=trade, ref_prices=price_now,
            pv_before=pv_before, dollar_volume=self.volume[k],
        )
        info_book = self.book.step_bar(
            delta_weights=trade, fill=fill, prev_price=prev_price, price_now=price_now,
            carry_rates=self.carry[k + 1], pv_before=pv_before, as_of_ts=int(self.timestamps[k + 1]),
        )
        self._W_held = self._W_held + trade

        # --- implementation shortfall (direction-free; struck vs the arrival price) ---
        ret_vs_arrival: np.ndarray = np.zeros(self.n_assets, dtype=np.float64)
        np.divide(price_now, self._P_arr, out=ret_vs_arrival, where=valid_next)
        ret_vs_arrival = np.where(valid_next, ret_vs_arrival - 1.0, 0.0)
        is_timing = float(np.sum(trade * pv_before * ret_vs_arrival))   # USD; trade carries the sign
        impact_cost = float(fill.realized_cost)                          # USD (fees + slippage incl. reactive)
        self._cum_cost += impact_cost

        is_frac = is_timing / self._parent_notional
        impact_frac = impact_cost / self._parent_notional
        reward = -(is_frac + self.impact_penalty * impact_frac) * self.is_reward_scale
        if self.asymmetric_dampen > 0.0:                                 # Spooner: clip favorable-timing profit
            reward -= self.asymmetric_dampen * max(0.0, -is_frac) * self.is_reward_scale

        # --- execution tracking-error equity (ADR-5; isolates what the overlay controls) ---
        # rel_pnl = (held − snap-target)·BAR-return − this bar's execution cost. The directional
        # P&L common to overlay and the snap baseline cancels; only the timing+cost diff remains.
        # MUST use the per-bar return price[k+1]/price[k]−1 (a per-bar equity increment), NOT the
        # cumulative-vs-arrival return used by IS (which would over-count every bar after the first).
        ret_bar: np.ndarray = np.zeros(self.n_assets, dtype=np.float64)
        valid_bar = (prev_price > _PRICE_EPS) & valid_next
        np.divide(price_now, prev_price, out=ret_bar, where=valid_bar)
        ret_bar = np.where(valid_bar, ret_bar - 1.0, 0.0)
        snap_dev = self._W_held - self.W_target[k]
        rel_pnl = float(np.sum(snap_dev * pv_before * ret_bar)) - impact_cost
        self._rel_equity += rel_pnl
        self._rel_peak = max(self._rel_peak, self._rel_equity)
        exec_drawdown = self._rel_equity / self._rel_peak - 1.0 if self._rel_peak > 1e-9 else 0.0
        if self.risk_penalty > 0.0:
            reward -= self.risk_penalty * max(0.0, -exec_drawdown) * self.is_reward_scale

        # --- advance clock; terminal guard on any residual (forced phi=1 makes this ~0) ---
        self.step_idx += 1
        self._h += 1
        # HORIZON-END IS A GENUINE MDP TERMINAL, NOT A TIME LIMIT (TERM-01).
        #
        # This deliberately breaks the project's usual `truncated`-at-horizon convention, so
        # the reason matters. `truncated` means "cut off, but the task would have continued" —
        # and the trainer acts on exactly that reading (``sac_trainer``: ``dones_for_buffer =
        # terms``, so a truncated step stores ``done=0`` and the critic BOOTSTRAPS across the
        # boundary). That is right for a walker stopped at 1000 steps. It is wrong here: at
        # ``h = H−1`` the env forces ``phi=1``, ``W_held -> W_target``, and the parent order is
        # COMPLETE. There is no continuation to bootstrap into — the vec-env's next observation
        # belongs to a different rebalance, months away, with a freshly warmed book.
        #
        # With `truncated` the critic therefore never saw a terminal state anywhere in this MDP
        # and learned the discounted sum over an endless chain of UNRELATED parent orders: an
        # effective horizon of 1/(1−gamma) = 100 bars = 20 parent orders instead of H=5. That is
        # the measured `cvar_q_mean` ≈ −880-and-climbing of the 75K seedcheck (randd_log
        # S553-cont-165) — an order of magnitude past the episodic CVaR (≈ −42) and heading for
        # the infinite-horizon fixed point (≈ −400). Not divergence: convergence to the wrong
        # quantity, with ~95 of those 100 bars being rewards the agent cannot influence.
        terminated = self._h >= self.H
        truncated = False
        if terminated:
            residual_l1 = float(np.abs(self.W_target[k] - self._W_held).sum())
            if residual_l1 > 1e-9 and self.unexecuted_penalty > 0.0:
                reward -= self.unexecuted_penalty * residual_l1 / max(self._gap0_l1, _L1_EPS)

        reward = float(np.clip(reward, -1e4, 1e4))
        portfolio_value = self._rel_equity if self.relative_equity else info_book["portfolio_value"]
        obs = self._get_obs(terminal=True) if terminated else self._get_obs()
        info = {
            "portfolio_value": portfolio_value,
            "cumulative_return": portfolio_value / self.initial_capital - 1.0,
            "implementation_shortfall_bps": is_frac * 1e4,
            "impact_cost_bps": impact_frac * 1e4,
            "fill_cost": impact_cost,
            "executed_l1": float(np.abs(trade).sum()),
            "remaining_l1": float(np.abs(self.W_target[k] - self._W_held).sum()),
            "gross_exposure": float(np.abs(self._W_held).sum()),
            "phi": phi,
            "book_step_return": info_book["step_return"],
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def _get_obs(self, *, terminal: bool = False) -> dict:
        """Build the raw-numpy obs dict (private + market blocks). All features causal (≤ the
        current decision bar). ``terminal`` clamps the lookahead bar at the episode's last
        valid index so the post-final-step obs stays in range (it is never acted on)."""
        k = min(self.step_idx, self.K)                      # decision-bar index for W_target/features
        h = self._h
        priv = self._priv_buf
        mkt = self._mkt_buf

        gap = self.W_target[min(k, self.K - 1)] - self._W_held
        gap_l1 = float(np.abs(gap).sum())
        inv_remaining = gap_l1 / max(self._gap0_l1, _L1_EPS)
        cum_exec_frac = 1.0 - inv_remaining
        baseline_cum = min(h / float(self.H), 1.0)
        exec_dd = self._rel_equity / self._rel_peak - 1.0 if self._rel_peak > 1e-9 else 0.0

        priv[0] = max(self.H - h, 0) / float(self.H)        # time_remaining
        priv[1] = inv_remaining                             # inventory_remaining
        priv[2] = cum_exec_frac - baseline_cum              # schedule_deviation (ahead +, behind −)
        priv[3] = exec_dd                                   # exec_drawdown (<=0)

        # --- market block (microstructure proxies from close + dollar volume; LEAK-2: ≤ k) ---
        order_profile = np.abs(self.W_target[min(k, self.K - 1)] - self._W_held)
        prof_l1 = order_profile.sum()
        wprof = order_profile / prof_l1 if prof_l1 > _L1_EPS else np.zeros(self.n_assets)
        sign_parent = np.sign(self.W_target[min(k, self.K - 1)] - self._W_held)

        mkt[0] = self._basket_realized_vol(k, wprof)
        mkt[1] = self._agg_participation(k, gap)
        mkt[2] = self._signed_momentum(k, wprof, sign_parent)
        mkt[3] = self._participation_dispersion(k, gap)
        mkt[4] = (self._cum_cost / self._parent_notional) * 1e4   # cost_so_far_bps
        mkt[5] = self._gap0_l1                                    # parent_gross

        # NaN/inf → 0, then a HARD finite bound (NAN-01). The nan_to_num alone is not
        # enough: a merely-huge finite float64 (a near-empty parent order divided into an
        # O(1) gap) survives it and only becomes inf once autocast narrows the batch to
        # float16 — which the encoder's LayerNorm then turns into a per-row NaN in the
        # actor's `loc`. See `obs_guard` for the full mechanism.
        sanitize_obs(priv)
        sanitize_obs(mkt)
        return {"scale_0": mkt.copy(), "private": priv.copy()}

    def _basket_realized_vol(self, k: int, wprof: np.ndarray) -> float:
        """Std of the order-weighted basket's close-to-close returns over the trailing
        ``vol_window`` bars ending at ``k`` (returns ≤ k ⇒ causal)."""
        lo = max(k - self.vol_window, 0)
        if k - lo < 2:
            return 0.0
        seg = self.price[lo:k + 1]                          # (w+1, U), last row = price[k]
        prev, cur = seg[:-1], seg[1:]
        valid = (prev > _PRICE_EPS) & (cur > _PRICE_EPS)
        rets = np.zeros_like(prev)
        np.divide(cur, prev, out=rets, where=valid)
        rets = np.where(valid, rets - 1.0, 0.0)             # (w, U)
        basket = rets @ wprof                               # (w,)
        return float(basket.std(ddof=1)) if basket.size > 1 else 0.0

    def _agg_participation(self, k: int, gap: np.ndarray) -> float:
        """Σ|gap·pv| / Σ dollar_volume — the impact pressure if the whole gap snapped now."""
        pv = self.book.pv_before(self.price[k]) if self.book is not None else self.initial_capital
        notional = float(np.abs(gap).sum()) * pv
        vol = float(self.volume[k].sum())
        return notional / vol if vol > _VOL_EPS else (1.0 if notional > 0 else 0.0)

    def _signed_momentum(self, k: int, wprof: np.ndarray, sign_parent: np.ndarray) -> float:
        """``mom_window``-bar basket return in the parent's trade direction (≤ k ⇒ causal).
        Positive ⇒ price moving against the order (a buy facing a rising price)."""
        lo = max(k - self.mom_window, 0)
        if k <= lo:
            return 0.0
        prev, cur = self.price[lo], self.price[k]
        valid = (prev > _PRICE_EPS) & (cur > _PRICE_EPS)
        ret: np.ndarray = np.zeros(self.n_assets, dtype=np.float64)
        np.divide(cur, prev, out=ret, where=valid)
        ret = np.where(valid, ret - 1.0, 0.0)
        return float(np.sum(sign_parent * wprof * ret))

    def _participation_dispersion(self, k: int, gap: np.ndarray) -> float:
        """Std across active legs of per-asset participation |gap·pv|/volume — an illiquid-
        leg flag (a scalar urgency must still respect the worst leg)."""
        active = np.abs(gap) > 1e-9
        if active.sum() < 2:
            return 0.0
        pv = self.book.pv_before(self.price[k]) if self.book is not None else self.initial_capital
        notional = np.abs(gap[active]) * pv
        vol = self.volume[k, active]
        part = np.where(vol > _VOL_EPS, notional / np.maximum(vol, _VOL_EPS), 1.0)
        return float(part.std(ddof=1))

    # ------------------------------------------------------------------ #
    def render(self):
        logger.info(
            "ExecOverlay r=%d h=%d/%d step=%d gross=%.3f rel_eq=%.2f cum_cost=%.2f",
            self._r, self._h, self.H, self.step_idx,
            float(np.abs(self._W_held).sum()), self._rel_equity, self._cum_cost,
        )
