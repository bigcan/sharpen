"""The uniform connector contract + point-in-time data types (spec §4.1).

Every free source implements :class:`DataConnector`, mirroring the existing loader pattern but
carrying the piece OHLCV loaders don't need and macro/positioning data lives or dies by: a
**release timestamp** per observation. The dangerous look-ahead is the *join* (aligning a series
to a bar by reference period instead of publication time — spec §4.3, CR-4), so the release
timestamp is a first-class field of every :class:`Observation`, not an afterthought.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, slots=True)
class Provenance:
    """License + point-in-time policy for a source/series (spec §4.3 item 4)."""

    source_id: str
    url: str
    license: str
    # PIT policy tag stored in the catalog: 'vintage-api' (ALFRED), 'release-lag' (COT), or
    # 'none-rejected' (a silently-revising series with no vintage → rejected by the gate, CR-4).
    as_of_policy: str
    release_lag_days: int = 0            # typical publication lag (reference period → public)
    revision_policy: str = "unknown"     # 'revised' | 'final' | 'unknown'


@dataclass(frozen=True, slots=True)
class SeriesRef:
    """A single series a source offers (what :meth:`DataConnector.discover` returns)."""

    source_id: str
    series_id: str                       # source-native key, e.g. 'T10Y2Y' or 'CFTC:067651'
    asset_class: str                     # one of crucible.catalog.ASSET_CLASSES
    title: str = ""
    frequency: str = ""                  # 'D' | 'W' | 'M' | ... (informational)

    @property
    def terminal(self) -> str:
        """The DSL terminal name this series is addressable as in a Panel feature slot
        (Crucible P1a grammar registry), e.g. ``fred:T10Y2Y`` / ``cot:comm_net``."""
        return f"{self.source_id}:{self.series_id}"


@dataclass(frozen=True, slots=True)
class Observation:
    """One point-in-time reading. The (reference_period, value, release_timestamp) triple is the
    unit of PIT correctness: ``value`` describes ``reference_period`` but only became public at
    ``release_timestamp``. A revision is a later observation for the same ``reference_period`` with
    a newer ``release_timestamp``."""

    reference_period: np.datetime64      # the date the datum describes (period-end)
    value: float
    release_timestamp: np.datetime64     # when the datum became publicly available (CR-4)


@dataclass(frozen=True, slots=True)
class SeriesData:
    """A fetched series as PIT observation arrays (parallel, not necessarily sorted). Arrays over
    dataclasses keep the as-of join vectorizable and reproducible."""

    ref: SeriesRef
    reference_period: np.ndarray         # (K,) datetime64[ns]
    value: np.ndarray                    # (K,) float64
    release_timestamp: np.ndarray        # (K,) datetime64[ns]
    provenance: Provenance | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        k = self.reference_period.shape[0]
        if not (self.value.shape[0] == k and self.release_timestamp.shape[0] == k):
            raise ValueError(
                f"SeriesData arrays must be equal length; got reference_period={k}, "
                f"value={self.value.shape[0]}, release_timestamp={self.release_timestamp.shape[0]}")

    @property
    def n_obs(self) -> int:
        return int(self.reference_period.shape[0])

    def observations(self) -> list[Observation]:
        """Materialize the triples (diagnostics/tests; the hot path uses the arrays)."""
        return [
            Observation(rp, float(v), rt)
            for rp, v, rt in zip(self.reference_period, self.value, self.release_timestamp)
        ]


@runtime_checkable
class DataConnector(Protocol):
    """Uniform, pluggable free-data source (spec §4.1). Connectors NEVER touch gates or verdicts —
    they only widen the data frontier the funnel can be pointed at (CR-1, CR-6)."""

    source_id: str
    asset_class: str

    def discover(self) -> list[SeriesRef]:
        """What series/tickers this source offers."""
        ...

    def fetch(
        self,
        ref: SeriesRef,
        start: np.datetime64 | str,
        end: np.datetime64 | str,
        *,
        as_of: np.datetime64 | str | None = None,
    ) -> SeriesData:
        """Pull one series. ``as_of`` requests the *vintage* as it stood on that date (ALFRED);
        connectors without a vintage API must stamp ``release_timestamp`` from the release lag."""
        ...

    def provenance(self, ref: SeriesRef) -> Provenance:
        """License, url, release lag, revision policy — recorded in the catalog (spec §4.3)."""
        ...
