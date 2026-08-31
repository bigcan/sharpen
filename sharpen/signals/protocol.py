"""The Signal plug interface.

A Signal is any object carrying a :class:`SignalSpec` and a vectorized, whole-panel
``compute`` (ADR-2): it returns the full ``(T, N)`` score matrix in one call (honoring
the "no per-sample Python loops in hot paths" coding standard). The CAUSAL CONTRACT —
row ``t`` may use only data with date <= ``panel.dates[t]`` — is NOT assumed from the
signature; it is VERIFIED by the Tier-0 truncation tripwire. A leaky signal compiles and
runs but is rejected at the gate, never reaching the ranking.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from .spec import SignalSpec

if TYPE_CHECKING:  # avoid import cycle: features.py defines Panel
    from .features import Panel


@runtime_checkable
class Signal(Protocol):
    """Structural type for a candidate signal."""

    spec: SignalSpec

    def compute(self, panel: "Panel") -> np.ndarray:
        """Return an ``(panel.T, panel.N)`` float64 score matrix.

        Causal contract: ``compute(panel)[t]`` may depend only on panel data with date
        <= ``panel.dates[t]``. NaN where the score is undefined or the name is inactive.
        ``±inf`` is forbidden (the harness treats non-finite-but-not-NaN as a bug).
        """
        ...
