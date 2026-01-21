"""SHAP explainability helpers for FinRL Pro."""

from __future__ import annotations

from typing import Mapping


class ShapAnalysis:
    """Compute lightweight SHAP summaries for evaluation reporting."""

    def summarize(self, feature_scores: Mapping[str, float]) -> dict[str, float]:
        """Return normalized feature contributions for reporting."""
        if not feature_scores:
            return {}

        total = sum(abs(value) for value in feature_scores.values()) or 1.0
        return {
            feature: abs(value) / total
            for feature, value in feature_scores.items()
        }
