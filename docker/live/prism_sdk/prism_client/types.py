"""Response types for the PRISM Python SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


COMPOSITE_LABELS = {
    0: "Grinding selloff",    # BEARISH / LOW_VOL
    1: "Correction",          # BEARISH / NORMAL_VOL
    2: "Crash/Panic",         # BEARISH / HIGH_VOL
    3: "Dead calm",           # NEUTRAL / LOW_VOL
    4: "Normal range",        # NEUTRAL / NORMAL_VOL
    5: "Choppy/Whipsaw",      # NEUTRAL / HIGH_VOL
    6: "Steady trend",        # BULLISH / LOW_VOL
    7: "Normal rally",        # BULLISH / NORMAL_VOL
    8: "Squeeze/Melt-up",     # BULLISH / HIGH_VOL
}


@dataclass
class ForecastResult:
    """Chronos-2 probabilistic forecast result.

    All quantile values are **log-returns** (not prices).
    To convert to price: ``target_price = current_price * exp(sum(p50))``.
    """

    ticker: str
    timestamp: str
    p10: List[float]            # 10th percentile log-returns per step
    p30: List[float]            # 30th percentile (only from feature provider)
    p50: List[float]            # Median log-returns per step
    p70: List[float]            # 70th percentile (only from feature provider)
    p90: List[float]            # 90th percentile log-returns per step
    quantile_spread: float      # Mean(p90 - p10) — forecast uncertainty
    samples: Optional[List[List[float]]] = None  # (100, H) sample matrix
    is_fallback: bool = False
    regime_context: Optional[Dict] = None
    calibration_diagnostics: Optional[Dict] = None

    @property
    def expected_return(self) -> float:
        """Cumulative median log-return over the forecast horizon."""
        return sum(self.p50)

    @property
    def horizon(self) -> int:
        return len(self.p50)


@dataclass
class RegimeResult:
    """Dual GAHMM regime detection result.

    Contains independent price direction and volatility regime predictions
    plus a composite 9-state regime code.
    """

    ticker: str
    timeframe: str
    timestamp: str
    price_regime: str           # "BEARISH", "NEUTRAL", or "BULLISH"
    vol_regime: str             # "LOW_VOL", "NORMAL_VOL", or "HIGH_VOL"
    composite_code: int         # 0-8: price * 3 + vol
    composite_label: str        # Human-readable label
    price_probabilities: Dict[str, float]  # {"BEARISH": 0.1, "NEUTRAL": 0.3, "BULLISH": 0.6}
    vol_probabilities: Dict[str, float]    # {"LOW_VOL": 0.2, "NORMAL_VOL": 0.5, "HIGH_VOL": 0.3}
    confidence: float           # min(max(price_probs), max(vol_probs))
    model_version: Optional[str] = None

    @property
    def is_bearish(self) -> bool:
        return self.price_regime == "BEARISH"

    @property
    def is_bullish(self) -> bool:
        return self.price_regime == "BULLISH"

    @property
    def is_high_vol(self) -> bool:
        return self.vol_regime == "HIGH_VOL"

    @property
    def is_crisis(self) -> bool:
        """Composite code 2: BEARISH + HIGH_VOL."""
        return self.composite_code == 2

    @property
    def position_multiplier(self) -> float:
        """Suggested position size multiplier based on vol regime.

        LOW_VOL: 1.3x, NORMAL_VOL: 1.0x, HIGH_VOL: 0.3x.
        """
        return {"LOW_VOL": 1.3, "NORMAL_VOL": 1.0, "HIGH_VOL": 0.3}.get(
            self.vol_regime, 1.0
        )

    def to_feature_vector(self) -> List[float]:
        """Convert to 7-element feature vector for RL observation space.

        Returns: [price_bear, price_neutral, price_bull,
                  vol_low, vol_normal, vol_high,
                  composite_code_normalized]
        """
        return [
            self.price_probabilities.get("BEARISH", 0.0),
            self.price_probabilities.get("NEUTRAL", 0.0),
            self.price_probabilities.get("BULLISH", 0.0),
            self.vol_probabilities.get("LOW_VOL", 0.0),
            self.vol_probabilities.get("NORMAL_VOL", 0.0),
            self.vol_probabilities.get("HIGH_VOL", 0.0),
            self.composite_code / 8.0,
        ]


@dataclass
class HealthStatus:
    """PRISM API health check result."""

    status: str
    chronos_loaded: bool
    hmm_service: str  # "operational" or "unavailable"


@dataclass
class RefitResult:
    """Result from triggering model refit."""

    status: str
    versions: Dict[str, str]  # {"price": "version_id", "vol": "version_id"}


@dataclass
class RegimeHistoryEntry:
    """Single entry from regime prediction history."""

    timestamp: str
    price_regime: int           # 0=BEARISH, 1=NEUTRAL, 2=BULLISH
    vol_regime: int             # 0=LOW_VOL, 1=NORMAL_VOL, 2=HIGH_VOL
    composite_code: int
    price_probs: Dict[str, float]
    vol_probs: Dict[str, float]
    confidence: float
