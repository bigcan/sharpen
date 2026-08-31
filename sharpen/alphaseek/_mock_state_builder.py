"""Mock state builder for dry-run testing.

Generates random but plausible 12-dim state vectors matching the AlphaSeek
v3 contract: position_norm, holding_norm, 8 LOB features, pending_limit_active,
bars_since_last_trade_norm. Simulates a slowly drifting BTC price for
mid_price and spread.
"""

import numpy as np
import torch


class MockStateBuilder:
    """Mock state builder for dry-run and testing."""

    def __init__(self, device: str = "cpu", base_price: float = 100000.0):
        self.device = torch.device(device)
        self._base_price = base_price
        self._mid_price = base_price
        self._spread = 0.5  # typical BTC spread
        self._tick = 0
        self._ready = True

    def get_state(
        self,
        position: int,
        holding: int,
        pending_limit_active: int = 0,
        bars_since_last_trade: int = 0,
    ) -> torch.Tensor:
        """Generate a mock 12-dim state vector."""
        self._tick += 1

        # Simulate random LSTM predictions (8 dims, small values near 0)
        lstm_preds = np.random.randn(8).astype(np.float32) * 0.01

        # Position and holding normalization (matching TradeSimulator)
        position_norm = float(position) / 1.0  # max_position = 1
        holding_norm = float(holding) / 1800.0  # max_holding = 1800
        pending_norm = 1.0 if pending_limit_active else 0.0
        idle_norm = min(1.0, float(bars_since_last_trade) / 1800.0)

        state = np.concatenate([
            [position_norm, holding_norm],
            lstm_preds,
            [pending_norm, idle_norm],
        ])

        # Simulate price drift
        self._mid_price += np.random.randn() * 10.0  # ~$10 random walk
        self._spread = max(0.1, self._spread + np.random.randn() * 0.05)

        return torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    @property
    def is_ready(self) -> bool:
        return self._ready

    def get_current_mid_price(self) -> float:
        return self._mid_price

    def get_current_spread(self) -> float:
        return self._spread
