"""LOBTradeSimulator — Drop-in TradeSimulator using LOB parquet + FeatureEngine.

Replaces the contest TradeSimulator's dependency on BTC_1sec_predict.npy with
8 hand-crafted microstructure features computed directly from our LOB database
via AlphaSeekFeatureEngine.

Usage:
    sim = LOBTradeSimulator(
        lob_parquet_path="data/lob_parquet/btcusdt_lob_1s.parquet",
        step_gap=2,  # HPO-sweepable: {1, 2, 4, 8}
        num_sims=4096,
    )
    state = sim.reset()
    state, reward, done, info = sim.step(action)

    # Walk-forward: filter to specific segments for train/val/test splits
    train_sim = LOBTradeSimulator(..., segment_filter=[0,1,2,3,4,5,6,7])
    val_sim   = LOBTradeSimulator(..., segment_filter=[8])
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch as th

from .feature_engine import AlphaSeekFeatureEngine
from .maker_bias_executor import MakerBiasExecutor

logger = logging.getLogger(__name__)


class LOBTradeSimulator:
    """GPU-tensorized trade simulator using LOB-derived features.

    Mirrors the contest TradeSimulator API (reset/step/get_state) but loads
    data from LOB parquet and computes features via AlphaSeekFeatureEngine.
    """

    def __init__(
        self,
        lob_parquet_path: str,
        num_sims: int = 64,
        slippage: float = 5e-5,
        max_position: int = 1,
        step_gap: int = 2,
        delay_step: int = 1,
        num_ignore_step: int = 60,
        seq_len: int = 3600,
        stop_loss_thresh: float = 1e-3,
        norm_span: int = 120,
        momentum_window: int = 5,
        vol_window: int = 30,
        device: th.device = th.device("cpu"),
        gpu_id: int = -1,
        segment_filter: list[int] | None = None,
        # v3 fee-internalized scalping (S500 ADR-002 — steady-state fees, no curriculum)
        taker_fee: float = 0.0,
        maker_fee: float = 0.0,
        maker_bias: bool = False,
        per_trade_penalty_bps: float = 0.0,
        holding_bonus: float = 0.0,
        # v3 maker-bias execution (S500 ADR-003); MakerBiasExecutor calibration.
        # Defaults from decision_alphaseek_v3_kb_review.md; HPO locks these.
        fill_alpha_ofi: float = 2.0,
        fill_beta_spread: float = 1.0,
        fill_gamma_age: float = -0.05,
        limit_max_age: int = 60,
        max_idle_bars: int | None = None,
        seed: int | None = None,
    ):
        self.device = th.device(f"cuda:{gpu_id}") if gpu_id >= 0 else device
        self.num_sims = num_sims
        self.slippage = slippage
        self.delay_step = delay_step
        self.max_holding = 60 * 60 // step_gap
        self.max_position = max_position
        self.step_gap = step_gap
        self.seq_len = seq_len
        self.stop_loss_thresh = stop_loss_thresh
        self.sim_ids = th.arange(self.num_sims, device=self.device)
        # v3 fee + penalty config; defaults zero so legacy configs preserve v1/v2 reward
        # semantics. M5 yaml configs set 5e-4 / 2e-4 / 0.5 / 1e-7 per plan.
        self.taker_fee = float(taker_fee)
        self.maker_fee = float(maker_fee)
        self.maker_bias = bool(maker_bias)
        self.per_trade_penalty_bps = float(per_trade_penalty_bps)
        self.holding_bonus = float(holding_bonus)

        # --- Load LOB data and compute features ---
        logger.info(f"Loading LOB parquet: {lob_parquet_path}")
        df = pd.read_parquet(lob_parquet_path)
        logger.info(f"  Loaded {len(df):,} rows")

        # Filter to specific segments for walk-forward train/val/test splits
        if segment_filter is not None and "segment_id" in df.columns:
            df = df[df["segment_id"].isin(segment_filter)].reset_index(drop=True)
            logger.info(
                f"  Filtered to segments {segment_filter}: {len(df):,} rows",
            )

        # Extract segment IDs for LEAK-1 compliant normalization
        segment_ids = None
        if "segment_id" in df.columns:
            segment_ids = df["segment_id"].values.astype(np.int32)

        # Compute features
        engine = AlphaSeekFeatureEngine(
            norm_span=norm_span,
            momentum_window=momentum_window,
            vol_window=vol_window,
        )
        factor_ary = engine.process_batch(df, segment_ids=segment_ids)
        self.factor_ary = th.tensor(factor_ary, dtype=th.float32, device=self.device)

        # Extract price arrays: [bid, ask, mid]
        mid = df["mid_price"].values.astype(np.float64)
        best_bid = df["best_bid_price"].values.astype(np.float64)
        best_ask = df["best_ask_price"].values.astype(np.float64)
        price_ary = np.stack([best_bid, best_ask, mid], axis=1)
        self.price_ary = th.tensor(price_ary, dtype=th.float32, device=self.device)

        assert self.price_ary.shape[0] == self.factor_ary.shape[0], (
            f"Price/factor length mismatch: {self.price_ary.shape[0]} vs {self.factor_ary.shape[0]}"
        )

        # Segment boundaries for episode sampling
        self._segment_ids = segment_ids
        self._valid_start_mask = self._compute_valid_starts()

        self.full_seq_len = self.price_ary.shape[0]
        logger.info(
            f"  Features: {self.factor_ary.shape}, "
            f"Prices: {self.price_ary.shape}, "
            f"Valid starts: {self._valid_start_mask.sum():,}",
        )

        # Environment info (matches contest TradeSimulator). v3 extends state
        # 10 → 12 dims: appends `pending_limit_active` + `bars_since_last_trade
        # / max_idle_bars` (S500 M3). Normaliser defaults to `max_holding`.
        self.max_idle_bars = (
            int(max_idle_bars) if max_idle_bars is not None else int(self.max_holding)
        )
        self.env_name = "LOBTradeSimulator-v0"
        self.state_dim = 8 + 4  # features + (position, holding, pending, idle)
        self.action_dim = 3  # short, nothing, long
        self.if_discrete = True
        self.max_step = (self.seq_len - num_ignore_step) // step_gap
        self.target_return = +np.inf

        # State tensors
        self.step_i = 0
        self.step_is = th.zeros((num_sims,), dtype=th.long, device=self.device)
        self.action_int = th.zeros((num_sims,), dtype=th.long, device=self.device)

        self.position = th.zeros((num_sims,), dtype=th.long, device=self.device)
        self.holding = th.zeros((num_sims,), dtype=th.long, device=self.device)

        self.cash = th.zeros((num_sims,), dtype=th.float32, device=self.device)
        self.asset = th.zeros((num_sims,), dtype=th.float32, device=self.device)
        self.best_price = th.zeros((num_sims,), dtype=th.float32, device=self.device)

        # v3 throughput-gate counter (M4 Optuna pruning reads this)
        self.fill_counter = th.zeros((num_sims,), dtype=th.long, device=self.device)
        # Cumulative fee paid per sim (telemetry / fee-to-gross ratio gate)
        self.fee_cost_cum = th.zeros((num_sims,), dtype=th.float32, device=self.device)
        # M3: bars since last filled trade (resets on fill; feeds dim 11 of state).
        self.bars_since_last_trade = th.zeros(
            (num_sims,), dtype=th.long, device=self.device,
        )

        # v3 maker-bias executor — always constructed, only consulted when
        # ``maker_bias=True`` so unit tests can poke its state regardless.
        self.maker_executor = MakerBiasExecutor(
            num_sims=num_sims,
            device=self.device,
            fill_alpha_ofi=fill_alpha_ofi,
            fill_beta_spread=fill_beta_spread,
            fill_gamma_age=fill_gamma_age,
            limit_max_age=limit_max_age,
            seed=seed,
        )

    def set_fees(self, taker_fee: float, maker_fee: float | None = None) -> None:
        """Runtime fee-update hook.

        Mirrors `ContinuousSwingEnv.set_fees` (envs/continuous_swing_env.py:176)
        so live overlays / future-curriculum experiments share the same surface.
        Pass ``maker_fee=None`` to leave the maker-side rate unchanged.
        """
        self.taker_fee = float(taker_fee)
        if maker_fee is not None:
            self.maker_fee = float(maker_fee)

    def _compute_valid_starts(self) -> np.ndarray:
        """Compute boolean mask of valid episode start indices.

        An index is valid if:
        1. There's enough room for a full episode (seq_len bars ahead)
        2. The episode doesn't cross a segment boundary
        """
        n = self.price_ary.shape[0]
        valid = np.ones(n, dtype=bool)

        # Must have seq_len bars ahead
        valid[max(0, n - self.seq_len * 2) :] = False

        # Must not start too early (warmup)
        valid[: self.seq_len] = False

        # Segment-aware: episode must not cross segment boundaries (vectorized)
        if self._segment_ids is not None:
            end_indices = np.minimum(np.arange(n) + self.seq_len - 1, n - 1)
            valid &= self._segment_ids == self._segment_ids[end_indices]

        return valid

    def _sample_start_indices(self) -> np.ndarray:
        """Sample num_sims valid start indices."""
        valid_indices = np.where(self._valid_start_mask)[0]
        if len(valid_indices) == 0:
            # Fallback: sample from anywhere with enough room
            valid_indices = np.arange(
                self.seq_len, max(self.seq_len + 1, self.full_seq_len - self.seq_len * 2),
            )
        return np.random.choice(valid_indices, size=self.num_sims, replace=True)

    def _reset(self, slippage=None, _if_random=True):
        self.slippage = slippage if isinstance(slippage, float) else self.slippage

        num_sims = self.num_sims
        device = self.device

        if _if_random:
            i0s = self._sample_start_indices()
        else:
            # Deterministic: start from a fixed offset for evaluation
            i0s = np.full(
                self.num_sims,
                self.full_seq_len - self.seq_len * 2,
                dtype=np.int64,
            )

        self.step_i = 0
        self.step_is = th.tensor(i0s, dtype=th.long, device=self.device)
        self.cash = th.zeros((num_sims,), dtype=th.float32, device=device)
        self.asset = th.zeros((num_sims,), dtype=th.float32, device=device)

        self.holding = th.zeros((num_sims,), dtype=th.long, device=device)
        self.position = th.zeros((num_sims,), dtype=th.long, device=device)
        self.best_price = th.zeros((num_sims,), dtype=th.float32, device=self.device)

        # v3 telemetry reset (throughput gate + fee-to-gross diagnostics)
        self.fill_counter = th.zeros((num_sims,), dtype=th.long, device=device)
        self.fee_cost_cum = th.zeros((num_sims,), dtype=th.float32, device=device)
        self.bars_since_last_trade = th.zeros(
            (num_sims,), dtype=th.long, device=device,
        )
        self.maker_executor.reset()

        step_is = self.step_is + self.step_i
        state = self.get_state(step_is)
        return state

    def _step(self, action, _if_random=True):
        self.step_i += self.step_gap
        step_is = self.step_is + self.step_i

        action = action.squeeze(1).to(self.device)
        action_int = action - 1  # map (0,1,2) → (-1,0,+1)
        del action

        # BUG-04 snapshot: agent intent before any forced-close masking. Held
        # for telemetry / future regularizers; reward formula uses executed
        # action_int because PnL is realized on what filled, not on intent.
        direction_for_reward = action_int.clone()

        old_cash = self.cash
        old_asset = self.asset
        old_position = self.position

        mid_price = self.price_ary[step_is, 2]

        # Track which sims had this bar's exit forced (taker fee + skip
        # per-trade penalty when M5 config wires post-action attribution).
        forced_close_mask = th.zeros(
            (self.num_sims,), dtype=th.bool, device=self.device,
        )

        # Truncation: force close at episode end
        truncated = self.step_i >= (self.max_step * self.step_gap)
        if truncated:
            action_int = -old_position
            forced_close_mask = old_position.ne(0)
        else:
            new_position = (old_position + action_int).clip(
                -self.max_position, self.max_position,
            )
            action_int = new_position - old_position

            # Prevent flipping through zero in one step
            done_mask = (new_position * old_position).lt(0) & old_position.ne(0)
            if done_mask.sum() > 0:
                action_int[done_mask] = -old_position[done_mask]

        # Max holding enforcement
        self.holding = self.holding + 1
        mask_max_holding = self.holding.gt(self.max_holding) & old_position.ne(0)
        if mask_max_holding.sum() > 0:
            action_int[mask_max_holding] = -old_position[mask_max_holding]
            forced_close_mask = forced_close_mask | mask_max_holding
        self.holding[old_position == 0] = 0

        # Stop-loss
        direction_mask1 = old_position.gt(0)
        if direction_mask1.sum() > 0:
            _best = th.max(
                th.stack([self.best_price[direction_mask1], mid_price[direction_mask1]]),
                dim=0,
            )[0]
            self.best_price[direction_mask1] = _best

        direction_mask2 = old_position.lt(0)
        if direction_mask2.sum() > 0:
            _best = th.min(
                th.stack([self.best_price[direction_mask2], mid_price[direction_mask2]]),
                dim=0,
            )[0]
            self.best_price[direction_mask2] = _best

        sl_mask1 = th.logical_and(
            direction_mask1, (self.best_price - mid_price).gt(self.stop_loss_thresh),
        )
        sl_mask2 = th.logical_and(
            direction_mask2, (mid_price - self.best_price).gt(self.stop_loss_thresh),
        )
        sl_mask = th.logical_or(sl_mask1, sl_mask2)
        if sl_mask.sum() > 0:
            action_int[sl_mask] = -old_position[sl_mask]
            forced_close_mask = forced_close_mask | sl_mask

        # ---- M2: maker-bias execution override. -----------------------------
        # When ``maker_bias=True``, non-forced sims surrender their action to
        # the executor (post-only at BBO; same-bar Bernoulli fill model).
        # Forced sims keep the taker exit (executor short-circuits to taker on
        # ``forced_close_mask=True`` so the result is identical to taker mode
        # for those sims).
        if self.maker_bias:
            ofi_signed = self.factor_ary[step_is, 3]  # OFI feature dim
            bbo_bid = self.price_ary[step_is, 0]
            bbo_ask = self.price_ary[step_is, 1]
            action_int, filled_mask, used_taker_mask = self.maker_executor.step(
                action_int=action_int,
                forced_close_mask=forced_close_mask,
                ofi_signed=ofi_signed,
                bbo_bid=bbo_bid,
                bbo_ask=bbo_ask,
                mid_price=mid_price,
                old_position=old_position,
            )
        else:
            filled_mask = action_int.ne(0)
            used_taker_mask = filled_mask  # taker-only model

        # Execute
        new_position = old_position + action_int

        entry_mask = old_position.eq(0)
        if entry_mask.sum() > 0:
            self.best_price[entry_mask] = mid_price[entry_mask]

        direction = action_int.gt(0)
        cost = action_int * mid_price
        new_cash = old_cash - cost * th.where(
            direction, 1 + self.slippage, 1 - self.slippage,
        )
        new_asset = new_cash + new_position * mid_price

        # ---- v3 reward: gross PnL minus exchange fees, per-trade penalty,
        #      plus position-holding bonus. Taker fills (forced exits or
        #      taker-only mode) pay ``taker_fee``; maker fills pay
        #      ``maker_fee``. Sims that didn't fill pay nothing.
        gross_pnl = new_asset - old_asset
        notional = action_int.abs().float() * mid_price
        effective_fee = th.where(
            used_taker_mask,
            th.full_like(notional, self.taker_fee),
            th.full_like(notional, self.maker_fee),
        )
        fee_cost = notional * effective_fee * filled_mask.float()
        trade_penalty = (
            filled_mask.float() * notional * (self.per_trade_penalty_bps / 1e4)
        )
        # Holding bonus rewards staying in (old) position; flat sims get nothing.
        hold_bonus = (
            old_position.ne(0).float() * self.holding_bonus * mid_price
        )
        reward = gross_pnl - fee_cost - trade_penalty + hold_bonus

        # Telemetry: per-sim fill counter (M4 throughput gate) + cumulative fee.
        self.fill_counter = self.fill_counter + filled_mask.long()
        self.fee_cost_cum = self.fee_cost_cum + fee_cost
        # M3: idle-bars counter resets on fill, increments otherwise.
        self.bars_since_last_trade = th.where(
            filled_mask,
            th.zeros_like(self.bars_since_last_trade),
            self.bars_since_last_trade + 1,
        )

        self.cash = new_cash
        self.asset = new_asset
        self.position = new_position
        self.action_int = action_int

        state = self.get_state(step_is)
        info_dict = {
            "filled_mask": filled_mask,
            "forced_close_mask": forced_close_mask,
            "used_taker_mask": used_taker_mask,
            "fee_cost": fee_cost,
            "fill_count": self.fill_counter.clone(),
            "fee_cost_cum": self.fee_cost_cum.clone(),
            "direction_for_reward": direction_for_reward,
        }
        if truncated:
            terminal = th.ones_like(self.position, dtype=th.bool)
            state = self.reset()
        else:
            terminal = th.zeros_like(self.position, dtype=th.bool)

        return state, reward, terminal, info_dict

    def reset(self, slippage=None, date_strs=()):
        return self._reset(slippage=slippage, _if_random=True)

    def step(self, action):
        return self._step(action, _if_random=True)

    def get_state(self, step_is):
        factor_ary = self.factor_ary[step_is, :]
        pending_active = self.maker_executor.pending_dir.ne(0).float()
        idle_norm = (
            self.bars_since_last_trade.float() / max(1, self.max_idle_bars)
        ).clamp(max=1.0)
        return th.concat(
            (
                (self.position.float() / self.max_position)[:, None],
                (self.holding.float() / self.max_holding)[:, None],
                factor_ary,
                pending_active[:, None],
                idle_norm[:, None],
            ),
            dim=1,
        )


    @staticmethod
    def get_segment_info(lob_parquet_path: str) -> pd.DataFrame:
        """Return summary of segments in the LOB parquet file.

        Returns DataFrame with columns:
            segment_id, n_rows, start_time, end_time, duration_hours
        """
        import pyarrow.parquet as pq

        schema = pq.read_schema(lob_parquet_path)
        col_names = [f.name for f in schema]

        # Detect timestamp column name (timestamp or timestamp_ms)
        ts_col = "timestamp" if "timestamp" in col_names else "timestamp_ms"

        if "segment_id" not in col_names:
            n_rows = pq.read_metadata(lob_parquet_path).num_rows
            return pd.DataFrame(
                {"segment_id": [0], "n_rows": [n_rows], "duration_hours": [np.nan]},
            )

        df = pd.read_parquet(lob_parquet_path, columns=["segment_id", ts_col])
        info = df.groupby("segment_id").agg(
            n_rows=(ts_col, "count"),
            start_time=(ts_col, "min"),
            end_time=(ts_col, "max"),
        ).reset_index()

        # Handle both epoch ms and datetime timestamps
        start = pd.to_datetime(info["start_time"], unit="ms", errors="coerce")
        end = pd.to_datetime(info["end_time"], unit="ms", errors="coerce")
        if start.isna().all():
            start = pd.to_datetime(info["start_time"])
            end = pd.to_datetime(info["end_time"])
        info["duration_hours"] = (end - start).dt.total_seconds() / 3600
        return info


class EvalLOBTradeSimulator(LOBTradeSimulator):
    """Evaluation variant: deterministic start, tighter stop-loss."""

    def reset(self, slippage=None, date_strs=()):
        self.stop_loss_thresh = 1e-4
        return self._reset(slippage=slippage, _if_random=False)

    def step(self, action):
        return self._step(action, _if_random=False)
