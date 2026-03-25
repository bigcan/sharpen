"""
Differential Sharpe Ratio (DSR) — Moody & Saffell 2001

Shared utility used by all trading environments that support DSR reward shaping.
Encapsulates the EMA state (A, B) and warmup logic so each env does not need to
duplicate the computation.
"""

import numpy as np


class DSRCalculator:
    """Stateful DSR calculator with EMA statistics.

    Parameters
    ----------
    eta : float
        EMA adaptation rate.  Default 0.001.
    scale : float
        Multiplicative scaling applied to the raw DSR before clipping.
        Default 1.0 (no extra scaling).
    """

    def __init__(self, eta: float = 0.001, scale: float = 1.0) -> None:
        self.eta = eta
        self.scale = scale
        self._A = 0.0   # EMA of returns
        self._B = 0.0   # EMA of squared returns
        self._warmup = 0

    def reset(self) -> None:
        """Reset EMA state for a new episode."""
        self._A = 0.0
        self._B = 0.0
        self._warmup = 0

    def compute(self, R_t: float) -> float:
        """Compute DSR for a single-step return.

        Implements the Moody & Saffell (2001) Differential Sharpe Ratio:

            DSR_t = (B_{t-1} * delta_A - 0.5 * A_{t-1} * delta_B) / var^{3/2}

        where delta_A = R_t - A_{t-1}, delta_B = R_t^2 - B_{t-1},
        var = B_{t-1} - A_{t-1}^2, and A/B are updated with EMA afterwards.

        Returns clipped DSR in [-10, 10], or 0.0 during warmup / near-zero
        variance.
        """
        delta_A = R_t - self._A
        delta_B = R_t * R_t - self._B

        prev_A, prev_B = self._A, self._B
        prev_variance = max(prev_B - prev_A ** 2, 0.0)

        self._A += self.eta * delta_A
        self._B += self.eta * delta_B
        self._warmup += 1

        if self._warmup > 1 and prev_variance > 1e-16:
            denom = prev_variance ** 1.5
            dsr = (prev_B * delta_A - 0.5 * prev_A * delta_B) / denom
            return float(np.clip(dsr * self.scale, -10.0, 10.0))

        return 0.0
