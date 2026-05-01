"""Maker-bias execution: post-only limits with same-bar probabilistic fills.

AlphaSeek v3 (S500 ADR-003). Defaults submit post-only at BBO; the agent's
intent only converts into an executed trade when the limit fills, modelled as
a Bernoulli draw with logit ``α·OFI_signed + β·spread_inv + γ·age``. Forced
closes (truncation, max-holding, stop-loss) bypass the maker queue and route
as taker — wired upstream in ``lob_trade_simulator._step``.

Vectorised across simulations so the same class works for the GPU sim
(num_sims = thousands) and live trading (num_sims = 1, device = cpu). On
Binance perp the live engine wraps this with ``timeInForce=GTX`` and a
single retry at ``BBO ± 1 tick`` on reject — see M11.

Calibration of α/β/γ comes from Stage 0.5 shadow logging; literature defaults
(α=2.0, β=1.0, γ=-0.05) are documented in
``decision_alphaseek_v3_kb_review.md``.
"""

from __future__ import annotations

import logging

import torch as th

logger = logging.getLogger(__name__)


class MakerBiasExecutor:
    """Stateful, vectorised post-only executor with same-bar fill probabilities.

    State (per sim):
        ``pending_dir`` ∈ {-1, 0, +1}  — direction of the resting limit (0 = none)
        ``pending_age`` int            — bars since submit (incremented in step)
        ``pending_price`` float        — BBO snapshot at submit (telemetry only)

    The executor does NOT update positions or PnL — it returns the effective
    ``action_int`` and the ``filled_mask`` which the caller (simulator or
    live engine) uses to apply state changes and accrue fees.
    """

    def __init__(
        self,
        num_sims: int,
        device: th.device,
        fill_alpha_ofi: float = 2.0,
        fill_beta_spread: float = 1.0,
        fill_gamma_age: float = -0.05,
        limit_max_age: int = 60,
        spread_clamp_bps: float = 0.5,
        seed: int | None = None,
    ):
        self.num_sims = num_sims
        self.device = device
        self.fill_alpha_ofi = float(fill_alpha_ofi)
        self.fill_beta_spread = float(fill_beta_spread)
        self.fill_gamma_age = float(fill_gamma_age)
        self.limit_max_age = int(limit_max_age)
        self.spread_clamp_bps = float(spread_clamp_bps)

        # rng kept on-device so vectorised Bernoulli draws don't stall on H2D.
        self._rng = th.Generator(device=device)
        if seed is not None:
            self._rng.manual_seed(int(seed))

        self.pending_dir = th.zeros(num_sims, dtype=th.long, device=device)
        self.pending_age = th.zeros(num_sims, dtype=th.long, device=device)
        self.pending_price = th.zeros(num_sims, dtype=th.float32, device=device)

    def reset(self) -> None:
        """Clear all pending limits (called from simulator._reset)."""
        self.pending_dir.zero_()
        self.pending_age.zero_()
        self.pending_price.zero_()

    def set_calibration(
        self,
        fill_alpha_ofi: float | None = None,
        fill_beta_spread: float | None = None,
        fill_gamma_age: float | None = None,
    ) -> None:
        """Runtime calibration update (Stage 0.5 shadow → live tune)."""
        if fill_alpha_ofi is not None:
            self.fill_alpha_ofi = float(fill_alpha_ofi)
        if fill_beta_spread is not None:
            self.fill_beta_spread = float(fill_beta_spread)
        if fill_gamma_age is not None:
            self.fill_gamma_age = float(fill_gamma_age)

    def step(
        self,
        action_int: th.Tensor,
        forced_close_mask: th.Tensor,
        ofi_signed: th.Tensor,
        bbo_bid: th.Tensor,
        bbo_ask: th.Tensor,
        mid_price: th.Tensor,
        old_position: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Resolve one bar of post-only limit dynamics.

        Args:
            action_int: agent intent in {-1, 0, +1}, shape (num_sims,)
            forced_close_mask: (num_sims,) bool — sim must exit this bar via taker
            ofi_signed: (num_sims,) — order flow imbalance feature, already normalised
            bbo_bid: (num_sims,) best bid for this bar
            bbo_ask: (num_sims,) best ask for this bar
            mid_price: (num_sims,) mid price for this bar
            old_position: (num_sims,) long — position before this bar's action

        Returns:
            effective_action_int: (num_sims,) long — what actually trades this bar
            filled_mask: (num_sims,) bool — True if a trade fills (taker or maker)
            used_taker_mask: (num_sims,) bool — True if the fill was taker
                (forced close or fallback). Maker fills get ``maker_fee``.
        """
        device = self.device
        n = self.num_sims

        effective_action = th.zeros_like(action_int)
        filled_mask = th.zeros(n, dtype=th.bool, device=device)
        used_taker_mask = th.zeros(n, dtype=th.bool, device=device)

        # ---- 1. Forced close: taker exit, drop any pending. -----------------
        if forced_close_mask.any():
            effective_action = th.where(
                forced_close_mask, -old_position, effective_action,
            )
            filled_mask = filled_mask | (forced_close_mask & old_position.ne(0))
            used_taker_mask = used_taker_mask | (
                forced_close_mask & old_position.ne(0)
            )
            self.pending_dir = th.where(
                forced_close_mask, th.zeros_like(self.pending_dir), self.pending_dir,
            )
            self.pending_age = th.where(
                forced_close_mask, th.zeros_like(self.pending_age), self.pending_age,
            )

        non_forced = ~forced_close_mask

        # ---- 2. Direction-change cancel BEFORE sampling. -------------------
        # If agent intent flipped vs. resting limit, drop the old quote first
        # so a stale +1 pending can't fill the bar the agent decided on -1.
        # Audit FIND-01 (S500): without this ordering the executor could
        # execute the previous direction's quote against current intent.
        if non_forced.any() and action_int.any():
            dir_change = (
                non_forced
                & self.pending_dir.ne(0)
                & action_int.ne(0)
                & (self.pending_dir * action_int < 0)
            )
            if dir_change.any():
                self.pending_dir = th.where(
                    dir_change, th.zeros_like(self.pending_dir), self.pending_dir,
                )
                self.pending_age = th.where(
                    dir_change, th.zeros_like(self.pending_age), self.pending_age,
                )

        # ---- 3. Sample fills on remaining pending limits. -------------------
        # Limit lives in queue; aging happens before sampling so age=0 means
        # "submitted this bar" → first fill chance comes next bar at age=1.
        has_pending = non_forced & self.pending_dir.ne(0)
        if has_pending.any():
            self.pending_age = th.where(
                has_pending, self.pending_age + 1, self.pending_age,
            )
            # Spread term: tighter book → higher fill prob (bps-scaled, clamped).
            rel_spread_bps = th.clamp(
                (bbo_ask - bbo_bid) / th.clamp(mid_price, min=1e-9) * 1e4,
                min=self.spread_clamp_bps,
            )
            spread_term = 1.0 / rel_spread_bps
            # OFI signed by pending direction: long pending wants bid-heavy OFI.
            ofi_dir = ofi_signed * self.pending_dir.float()
            logit = (
                self.fill_alpha_ofi * ofi_dir
                + self.fill_beta_spread * spread_term
                + self.fill_gamma_age * self.pending_age.float()
            )
            p_fill = th.sigmoid(logit)
            draws = th.rand(n, device=device, generator=self._rng)
            fill_now = has_pending & (draws < p_fill)

            if fill_now.any():
                effective_action = th.where(
                    fill_now, self.pending_dir, effective_action,
                )
                filled_mask = filled_mask | fill_now
                # used_taker stays False — these are maker fills.
                self.pending_dir = th.where(
                    fill_now, th.zeros_like(self.pending_dir), self.pending_dir,
                )
                self.pending_age = th.where(
                    fill_now, th.zeros_like(self.pending_age), self.pending_age,
                )

            # ---- 4. Cancel aged-out limits. ---------------------------------
            aged_out = (
                non_forced & self.pending_dir.ne(0)
                & (self.pending_age > self.limit_max_age)
            )
            if aged_out.any():
                self.pending_dir = th.where(
                    aged_out, th.zeros_like(self.pending_dir), self.pending_dir,
                )
                self.pending_age = th.where(
                    aged_out, th.zeros_like(self.pending_age), self.pending_age,
                )

        # ---- 5. Submit new post-only when the queue is empty. --------------
        # No same-bar fill on submit — first chance is next bar at age=1.
        new_submit = (
            non_forced & action_int.ne(0) & self.pending_dir.eq(0) & ~filled_mask
        )
        if new_submit.any():
            self.pending_dir = th.where(new_submit, action_int, self.pending_dir)
            self.pending_age = th.where(
                new_submit, th.zeros_like(self.pending_age), self.pending_age,
            )
            limit_price = th.where(action_int.gt(0), bbo_bid, bbo_ask)
            self.pending_price = th.where(
                new_submit, limit_price, self.pending_price,
            )

        return effective_action, filled_mask, used_taker_mask

    def state_summary(self) -> dict:
        """Telemetry snapshot (used by tests and the throughput gate)."""
        return {
            "pending_count": int(self.pending_dir.ne(0).sum().item()),
            "mean_age": float(self.pending_age.float().mean().item()),
            "max_age": int(self.pending_age.max().item()),
        }
