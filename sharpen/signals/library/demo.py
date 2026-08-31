"""A few classic causal technical signals — the starter batch that proves the harness
end-to-end and seeds the structure the 101-alpha / TradingView libraries (P8) extend.

Each is a pure function of close/volume up to bar ``t`` (causal — verified by the Tier-0
truncation tripwire). Exposes ``SIGNALS`` for the CLI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import Panel
from ..spec import SignalSpec


def _trailing_logret(close: np.ndarray, lookback: int) -> np.ndarray:
    c = np.log(close)
    out = np.full(c.shape, np.nan)
    out[lookback:] = c[lookback:] - c[:-lookback]
    return out


class Momentum:
    """``lookback``-day trailing return — the continuation (momentum) hypothesis."""

    def __init__(self, lookback: int = 20) -> None:
        self.lookback = lookback
        self.spec = SignalSpec(
            name=f"mom_{lookback}d",
            hypothesis=f"{lookback}d trailing return predicts continuation",
            family="technical", expected_sign=1)

    def compute(self, panel: Panel) -> np.ndarray:
        return _trailing_logret(panel.close, self.lookback)


class Reversal:
    """Short-horizon trailing return — the reversal hypothesis (expected_sign=-1)."""

    def __init__(self, lookback: int = 5) -> None:
        self.lookback = lookback
        self.spec = SignalSpec(
            name=f"rev_{lookback}d",
            hypothesis=f"{lookback}d return reverses (recent losers outperform)",
            family="technical", expected_sign=-1)

    def compute(self, panel: Panel) -> np.ndarray:
        return _trailing_logret(panel.close, self.lookback)


class Volatility:
    """Trailing realized volatility — the low-vol-anomaly hypothesis (expected_sign=-1)."""

    def __init__(self, lookback: int = 20) -> None:
        self.lookback = lookback
        self.spec = SignalSpec(
            name=f"vol_{lookback}d",
            hypothesis="low trailing realized vol predicts higher risk-adjusted return",
            family="technical", expected_sign=-1)

    def compute(self, panel: Panel) -> np.ndarray:
        c = np.log(panel.close)
        r = np.full(c.shape, np.nan)
        r[1:] = c[1:] - c[:-1]
        return pd.DataFrame(r).rolling(self.lookback, min_periods=self.lookback).std().to_numpy()


SIGNALS = [Momentum(20), Momentum(60), Reversal(5), Volatility(20)]
