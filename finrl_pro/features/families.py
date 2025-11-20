"""Indicator family resolution utilities.

Maps high-level indicator families into stockstats indicator names used by
the upstream FeatureEngineer. This allows config-driven toggling of groups
without editing upstream code.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence


# Baseline mapping aligned with defaults in FinRLPodracer/finrl/StockTrading.py
FAMILY_MAP: Dict[str, List[str]] = {
    "trend": ["close_30_sma", "close_60_sma", "macd"],
    "momentum": ["rsi_30"],
    "vol": ["dx_30", "rsi_30", "cci_30", "mfi_30"],
    "volume": [],
}


def resolve_indicator_list(
    *,
    families: Mapping[str, bool] | None = None,
    overrides: Sequence[str] | None = None,
) -> List[str]:
    """Return a deduplicated indicator list given family toggles and overrides.

    - families: mapping of family -> enabled flag
    - overrides: explicit indicator names to include regardless of family
    """
    indicators: List[str] = []
    fams = families or {}
    for name, enabled in fams.items():
        if enabled:
            indicators.extend(FAMILY_MAP.get(name, []))
    if overrides:
        indicators.extend(list(overrides))
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: List[str] = []
    for x in indicators:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result

