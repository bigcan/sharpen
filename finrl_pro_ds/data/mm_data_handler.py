"""
Market Making Data Handler

Extends MultiScaleOHLCVHandler with OHLCV bar data (open, high, low, volume)
needed by fill models. Also stores base-scale open and volume arrays.

step() returns all fields from MultiScaleOHLCVHandler plus:
  - open: float (base-scale bar open)
  - high: float (base-scale bar high)
  - low: float (base-scale bar low)
  - volume: float (base-scale bar volume)
"""
import numpy as np
import logging
from typing import Dict, Optional

from finrl_pro_ds.data.multiscale_handler import MultiScaleOHLCVHandler

logger = logging.getLogger(__name__)


class MMDataHandler(MultiScaleOHLCVHandler):
    """Market Making data handler — extends MultiScaleOHLCVHandler with full OHLCV bar data."""

    def _load_data(self):
        """Load data via parent, then store additional base-scale arrays."""
        super()._load_data()

        base_df = self._scale_dfs[self._base_scale]
        self._base_open = base_df['open'].values.astype(np.float64)
        self._base_volume = base_df['volume'].values.astype(np.float64)

        logger.info(
            f"MMDataHandler: added open/volume arrays ({len(self._base_open)} bars)"
        )

    def step(self) -> Optional[Dict]:
        """Advance one bar, return multi-scale obs + full OHLCV bar data.

        Returns parent dict augmented with: open, high, low, volume
        """
        result = super().step()
        if result is None:
            return None

        # _ptr was already incremented by super().step(), so use _ptr - 1
        idx = self._ptr - 1
        result["open"] = float(self._base_open[idx])
        result["high"] = float(self._base_high[idx])
        result["low"] = float(self._base_low[idx])
        result["volume"] = float(self._base_volume[idx])

        return result
