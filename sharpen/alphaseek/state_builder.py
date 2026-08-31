"""AlphaSeek State Builder — implements StateBuilderProtocol for live trading.

Wraps AlphaSeekFeatureEngine and produces the 12-dim state tensor
(position_norm, holding_norm, feat_0..feat_7, pending_limit_active,
bars_since_last_trade_norm) expected by the v3 DQN ensemble.

Legacy v1/v2 checkpoints used a 10-dim layout (position, holding, feats);
v3 trains from scratch and adds the two execution-state dims so the policy
can observe its own pending-limit queue and recent-trade frequency.

Usage:
    from sharpen.alphaseek.feature_engine import AlphaSeekFeatureEngine
    from sharpen.alphaseek.state_builder import AlphaSeekStateBuilder

    engine = AlphaSeekFeatureEngine(norm_span=120)
    builder = AlphaSeekStateBuilder(engine, device="cuda:0")

    # Feed snapshots from BybitLOBFeed
    builder.ingest_snapshot(snapshot)

    # Get state for the ensemble
    state = builder.get_state(
        position=1, holding=42,
        pending_limit_active=0, bars_since_last_trade=0,
    )  # (1, 12) tensor
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from .feature_engine import AlphaSeekFeatureEngine

logger = logging.getLogger(__name__)


class AlphaSeekStateBuilder:
    """Live state builder implementing the StateBuilderProtocol.

    Protocol methods:
        get_state(position, holding, pending_limit_active, bars_since_last_trade)
            -> torch.Tensor  # (1, 12)
        is_ready -> bool
        get_current_mid_price() -> float
        get_current_spread() -> float
    """

    def __init__(
        self,
        feature_engine: AlphaSeekFeatureEngine,
        device: str = "cpu",
        max_position: int = 1,
        max_holding: int = 1800,
        max_idle_bars: int | None = None,
    ):
        self._engine = feature_engine
        self._device = torch.device(device)
        self._max_position = max_position
        self._max_holding = max_holding
        self._max_idle_bars = (
            int(max_idle_bars) if max_idle_bars is not None else int(max_holding)
        )

        self._latest_features: np.ndarray | None = None
        self._mid_price: float = 0.0
        self._spread: float = 0.0
        self._snapshot_count: int = 0

    def ingest_snapshot(self, snapshot: dict) -> None:
        """Process a new LOB snapshot and update internal state.

        Args:
            snapshot: dict with keys matching FeatureEngine.process_snapshot() contract.
        """
        self._latest_features = self._engine.process_snapshot(snapshot)
        self._snapshot_count += 1

        bid = float(snapshot.get("best_bid_price", 0.0))
        ask = float(snapshot.get("best_ask_price", 0.0))
        self._mid_price = (bid + ask) / 2.0
        self._spread = float(snapshot.get("spread", ask - bid))

    def get_state(
        self,
        position: int,
        holding: int,
        pending_limit_active: int = 0,
        bars_since_last_trade: int = 0,
    ) -> torch.Tensor:
        """Build the 12-dim state tensor.

        Returns:
            torch.Tensor of shape (1, 12) on self._device.
            Layout: [position_norm, holding_norm, feat_0..feat_7,
                     pending_limit_active, bars_since_last_trade_norm]
        """
        if self._latest_features is None:
            # Pre-warmup: return zeros
            return torch.zeros(
                1, 12, dtype=torch.float32, device=self._device,
            )

        position_norm = float(position) / max(self._max_position, 1)
        holding_norm = float(holding) / max(self._max_holding, 1)
        pending_norm = 1.0 if pending_limit_active else 0.0
        idle_norm = min(
            1.0, float(bars_since_last_trade) / max(1, self._max_idle_bars),
        )

        state = np.concatenate(
            [
                [position_norm, holding_norm],
                self._latest_features,
                [pending_norm, idle_norm],
            ],
        )

        return torch.tensor(
            state, dtype=torch.float32, device=self._device,
        ).unsqueeze(0)

    @property
    def is_ready(self) -> bool:
        """True after enough snapshots for EMA warmup."""
        return self._snapshot_count >= self._engine.warmup_ticks

    def get_current_mid_price(self) -> float:
        return self._mid_price

    def get_current_spread(self) -> float:
        return self._spread

    def reset(self) -> None:
        """Reset state builder and underlying feature engine."""
        self._engine.reset()
        self._latest_features = None
        self._mid_price = 0.0
        self._spread = 0.0
        self._snapshot_count = 0
