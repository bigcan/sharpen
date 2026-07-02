"""Crucible free-data acquisition subsystem (spec §4) — Crucible P1b.

The small-operator edge is gated by *data access*, not math (``project_small_operator_strategy_reframe``),
so widening the data frontier with free, point-in-time-safe sources is a first-class feature. This
package owns the uniform connector contract, the non-OHLCV data-quality gate (including the
as-of-join reconstruction tripwire — CR-4/LEAK-2), and catalog registration.

P1b ships two Tier-A connectors (spec §10 decision 3): **FRED/ALFRED** (macro, vintage) and
**CFTC COT** (positioning, release-lagged). Both are testable against fixtures with NO network and
NO API key by injecting a ``transport`` callable; the live pull is gated behind ``FRED_API_KEY`` /
outbound HTTPS and fails closed with a clear error.

The load-bearing invariant (CR-4): a series enters a Panel feature slot only through
:func:`quality_gate.asof_join`, which binds bar ``t`` to the latest observation *released by* ``t``
— never by reference period. :func:`quality_gate.assert_asof_join_causal` is the P1b exit-gate.
"""
from __future__ import annotations

from .cftc_cot import CftcCotConnector
from .connector import (
    DataConnector,
    Observation,
    Provenance,
    SeriesData,
    SeriesRef,
)
from .fred import FredConnector
from .quality_gate import (
    QualityReport,
    asof_join,
    assert_asof_join_causal,
    assert_feature_causal,
    naive_reference_period_join,
    register_series,
    validate_series,
)

__all__ = [
    "CftcCotConnector",
    "DataConnector",
    "FredConnector",
    "Observation",
    "Provenance",
    "QualityReport",
    "SeriesData",
    "SeriesRef",
    "asof_join",
    "assert_asof_join_causal",
    "assert_feature_causal",
    "naive_reference_period_join",
    "register_series",
    "validate_series",
]
