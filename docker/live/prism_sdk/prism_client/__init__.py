"""PRISM Python SDK — client for the PRISM financial intelligence API.

Provides typed access to Chronos-2 probabilistic forecasts and Dual GAHMM
regime detection for seamless integration with FinRL and other trading frameworks.
"""

from prism_client.client import PRISMClient
from prism_client.types import (
    ForecastResult,
    RegimeResult,
    HealthStatus,
    RefitResult,
    RegimeHistoryEntry,
    COMPOSITE_LABELS,
)

__version__ = "1.0.0"
__all__ = [
    "PRISMClient",
    "ForecastResult",
    "RegimeResult",
    "HealthStatus",
    "RefitResult",
    "RegimeHistoryEntry",
    "COMPOSITE_LABELS",
]
