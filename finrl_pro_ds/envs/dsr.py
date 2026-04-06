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
    regime_eta_multipliers : dict or None
        Optional PRISM regime-adaptive eta multipliers.
        Keys: vol regime int (0=LOW_VOL, 1=NORMAL_VOL, 2=HIGH_VOL).
        Values: multiplier applied to base eta. E.g. {0: 2.0, 1: 1.0, 2: 0.3}.
    regime_blend : float
        Blending factor between base eta and regime eta [0, 1].
        0.0 = pure base eta, 1.0 = pure regime eta. Default 0.5.
    """

    def __init__(
        self,
        eta: float = 0.001,
        scale: float = 1.0,
        regime_eta_multipliers: dict[int, float] | None = None,
        regime_blend: float = 0.5,
    ) -> None:
        self._base_eta = eta
        self.eta = eta
        self.scale = scale
        self._A = 0.0   # EMA of returns
        self._B = 0.0   # EMA of squared returns
        self._warmup = 0

        # Regime-adaptive DSR (Path 2)
        self._regime_eta_mults = regime_eta_multipliers
        self._regime_blend = regime_blend

    def reset(self) -> None:
        """Reset EMA state for a new episode."""
        self._A = 0.0
        self._B = 0.0
        self._warmup = 0
        self.eta = self._base_eta

    def set_regime_context(self, vol_regime: int) -> None:
        """Set regime-adaptive eta for the current bar.

        Called before compute() when regime_eta_multipliers is configured.
        vol_regime: 0=LOW_VOL, 1=NORMAL_VOL, 2=HIGH_VOL, -1=unknown.
        """
        if self._regime_eta_mults is None or vol_regime < 0:
            self.eta = self._base_eta
            return
        mult = self._regime_eta_mults.get(vol_regime, 1.0)
        regime_eta = self._base_eta * mult
        # Smooth blend to avoid non-stationarity at regime transitions
        self.eta = (1.0 - self._regime_blend) * self._base_eta + self._regime_blend * regime_eta

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
