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
        self.factor_ary = th.tensor(factor_ary, dtype=th.float32)  # CPU

        # Extract price arrays: [bid, ask, mid]
        mid = df["mid_price"].values.astype(np.float64)
        best_bid = df["best_bid_price"].values.astype(np.float64)
        best_ask = df["best_ask_price"].values.astype(np.float64)
        price_ary = np.stack([best_bid, best_ask, mid], axis=1)
        self.price_ary = th.tensor(price_ary, dtype=th.float32)  # CPU

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

        # Environment info (matches contest TradeSimulator)
        self.env_name = "LOBTradeSimulator-v0"
        self.state_dim = 8 + 2  # features + (position, holding)
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

        # Segment-aware: episode must not cross segment boundaries
        if self._segment_ids is not None:
            for i in range(n):
                if not valid[i]:
                    continue
                end_idx = min(i + self.seq_len, n)
                if end_idx >= n:
                    valid[i] = False
                    continue
                if self._segment_ids[i] != self._segment_ids[end_idx - 1]:
                    valid[i] = False

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

        step_is = self.step_is + self.step_i
        state = self.get_state(step_is_cpu=step_is.to(th.device("cpu")))
        return state

    def _step(self, action, _if_random=True):
        self.step_i += self.step_gap
        step_is = self.step_is + self.step_i
        step_is_cpu = step_is.to(th.device("cpu"))

        action = action.squeeze(1).to(self.device)
        action_int = action - 1  # map (0,1,2) → (-1,0,+1)
        del action

        old_cash = self.cash
        old_asset = self.asset
        old_position = self.position

        mid_price = self.price_ary[step_is_cpu, 2].to(self.device)

        # Truncation: force close at episode end
        truncated = self.step_i >= (self.max_step * self.step_gap)
        if truncated:
            action_int = -old_position
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
        mask_max_holding = self.holding.gt(self.max_holding)
        if mask_max_holding.sum() > 0:
            action_int[mask_max_holding] = -old_position[mask_max_holding]
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

        reward = new_asset - old_asset

        self.cash = new_cash
        self.asset = new_asset
        self.position = new_position
        self.action_int = action_int

        state = self.get_state(step_is_cpu)
        info_dict = {}
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

    def get_state(self, step_is_cpu):
        factor_ary = self.factor_ary[step_is_cpu, :].to(self.device)
        return th.concat(
            (
                (self.position.float() / self.max_position)[:, None],
                (self.holding.float() / self.max_holding)[:, None],
                factor_ary,
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
