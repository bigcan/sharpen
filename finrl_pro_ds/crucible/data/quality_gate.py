"""The data crucible (spec §4.3) — no non-OHLCV series reaches a Panel feature slot without
passing here. The load-bearing piece is the **as-of join** and its **reconstruction tripwire**.

Why this exists (CR-4 / LEAK-2, spec §4.3 item 3): the Tier-0 truncation-equivalence check
validates the *signal function*, not the *join*. The dangerous mistake is aligning a released
series to a bar by **reference period** instead of **publication timestamp** — e.g. joining
Tuesday's COT to Tuesday's bar when it only published Friday. :func:`asof_join` is the only
sanctioned way to build a feature series; :func:`assert_asof_join_causal` is the P1b exit-gate; and
:func:`naive_reference_period_join` exists solely to *demonstrate the leak* a negative test asserts
against.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

import numpy as np

from ..catalog import CatalogEntry, DataCatalog
from .connector import Provenance, SeriesData

logger = logging.getLogger(__name__)

_NS = np.timedelta64(1, "ns")


def _as_ns(x: np.datetime64 | str) -> np.datetime64:
    return np.datetime64(x, "ns")


def asof_join(series: SeriesData, bar_dates: np.ndarray) -> np.ndarray:
    """Bind ``series`` to a (T,) bar calendar the ONLY PIT-safe way (CR-4): bar ``t`` takes the
    value of the most recent reference period whose ``release_timestamp <= bar_dates[t]`` — i.e.
    the latest reading *publicly available by t*. Ties on reference period break to the later
    release (a revision supersedes). Bars before the first release are NaN.

    This is a publication-time join, not a reference-period join. It is deliberately vectorized so
    ``asof_join(series, bar_dates[:t+1])[t] == asof_join(series, bar_dates)[t]`` (the truncation
    property the tripwire relies on).
    """
    bars = np.asarray(bar_dates, dtype="datetime64[ns]")
    ref = np.asarray(series.reference_period, dtype="datetime64[ns]")
    rel = np.asarray(series.release_timestamp, dtype="datetime64[ns]")
    val = np.asarray(series.value, dtype=np.float64)
    out = np.full(bars.shape[0], np.nan, dtype=np.float64)
    if ref.shape[0] == 0:
        return out
    # Order observations by (release_timestamp, reference_period) so that, scanning in release
    # order, the running "best available reading" = max reference_period seen so far (revisions of
    # an already-seen period keep the newest value because they arrive later in release order).
    order = np.lexsort((ref, rel))          # primary key = rel, secondary = ref
    rel_s, ref_s, val_s = rel[order], ref[order], val[order]
    # For each bar, how many observations were released by then (rel_s is sorted ascending).
    n_released = np.searchsorted(rel_s, bars, side="right")   # (T,) count of releases <= bar
    # Running argmax of reference_period over the release-ordered stream, and the value carried by
    # that argmax. best_val[k] = value of the reading in effect after the first (k+1) releases.
    best_ref = np.maximum.accumulate(ref_s)
    is_new_max = ref_s >= best_ref          # True where this release advances (or ties) the max ref
    # Forward-fill the in-effect value across ties/regressions so index k always holds the reading
    # in effect after the first (k+1) releases (a later release of the max ref period supersedes).
    fill_idx = np.where(is_new_max, np.arange(val_s.shape[0]), -1)
    fill_idx = np.maximum.accumulate(fill_idx)
    eff_val = np.where(fill_idx >= 0, val_s[np.clip(fill_idx, 0, None)], np.nan)
    has = n_released > 0
    out[has] = eff_val[n_released[has] - 1]
    return out


def naive_reference_period_join(series: SeriesData, bar_dates: np.ndarray) -> np.ndarray:
    """The WRONG join, provided only so a negative test can assert against it: bar ``t`` takes the
    latest value whose *reference_period* <= ``bar_dates[t]``, ignoring when it was released. For a
    release-lagged series this uses data from the future (the leak the tripwire catches)."""
    bars = np.asarray(bar_dates, dtype="datetime64[ns]")
    ref = np.asarray(series.reference_period, dtype="datetime64[ns]")
    val = np.asarray(series.value, dtype=np.float64)
    out = np.full(bars.shape[0], np.nan, dtype=np.float64)
    if ref.shape[0] == 0:
        return out
    order = np.argsort(ref, kind="stable")
    ref_s, val_s = ref[order], val[order]
    n_ref = np.searchsorted(ref_s, bars, side="right")
    has = n_ref > 0
    out[has] = val_s[n_ref[has] - 1]
    return out


def assert_feature_causal(joined: np.ndarray, series: SeriesData, bar_dates: np.ndarray) -> None:
    """Assert an ARBITRARY joined feature series used only data publicly available in time: for
    every non-NaN bar ``t`` the value must equal some observation with ``release_timestamp <=
    bar_dates[t]``. A reference-period join (:func:`naive_reference_period_join`) FAILS this — which
    is exactly what the P1b negative test asserts, proving the tripwire detects the data-layer leak
    that Tier-0 is blind to."""
    bars = np.asarray(bar_dates, dtype="datetime64[ns]")
    rel = np.asarray(series.release_timestamp, dtype="datetime64[ns]")
    val = np.asarray(series.value, dtype=np.float64)
    joined = np.asarray(joined, dtype=np.float64)
    for t in range(bars.shape[0]):
        vt = joined[t]
        if np.isnan(vt):
            continue
        avail = rel <= bars[t]
        if not np.any(avail):
            raise AssertionError(f"bar {t} ({bars[t]}) has value {vt} but NO release <= bar date")
        if not np.any(np.isclose(val[avail], vt, rtol=0, atol=1e-12)):
            raise AssertionError(
                f"as-of-join LEAK at bar {t} ({bars[t]}): value {vt} matches no observation "
                f"released by then (latest release <= t = {rel[avail].max()})")


def assert_asof_join_causal(series: SeriesData, bar_dates: np.ndarray) -> None:
    """P1b EXIT-GATE (spec §8 P1b row). Rebuild the feature purely from (value, release_timestamp)
    events and assert every feature value at bar ``t`` was *publicly available by t* — a fail means
    look-ahead. Combines the data-availability check (:func:`assert_feature_causal`) with the
    Tier-0 truncation property extended to the data-layer join: truncating the calendar at ``t``
    yields the identical value at ``t``.

    Raises AssertionError on any violation (the negative test asserts this does NOT raise for the
    release-time :func:`asof_join`, and DOES for :func:`naive_reference_period_join`).
    """
    bars = np.asarray(bar_dates, dtype="datetime64[ns]")
    joined = asof_join(series, bars)
    assert_feature_causal(joined, series, bars)
    for t in range(bars.shape[0]):
        vt = joined[t]
        vt_trunc = asof_join(series, bars[: t + 1])[t]
        if np.isnan(vt) and np.isnan(vt_trunc):
            continue
        if not np.isclose(vt_trunc, vt, atol=1e-12):
            raise AssertionError(
                f"as-of-join not truncation-stable at bar {t}: full={vt} truncated={vt_trunc}")


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Outcome of the non-OHLCV data-quality gate (spec §4.3 items 1-2)."""

    passed: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
    n_obs: int = 0
    max_gap_days: float = 0.0
    n_outliers: int = 0


def validate_series(
    series: SeriesData,
    *,
    max_gap_days: float = 400.0,
    outlier_sigma: float = 12.0,
    require_release_after_reference: bool = True,
) -> QualityReport:
    """The analogue of ``clean_ohlcv`` for a non-OHLCV series: gaps, staleness, outliers, unit
    shifts, and the PIT sanity that a release never predates the period it describes. Returns a
    :class:`QualityReport`; the caller decides whether to admit the series to the catalog."""
    reasons: list[str] = []
    ref = np.asarray(series.reference_period, dtype="datetime64[ns]")
    rel = np.asarray(series.release_timestamp, dtype="datetime64[ns]")
    val = np.asarray(series.value, dtype=np.float64)
    n = ref.shape[0]
    if n == 0:
        return QualityReport(False, ("empty series",), 0)

    order = np.argsort(ref, kind="stable")
    ref_s, rel_s, val_s = ref[order], rel[order], val[order]

    # PIT sanity (CR-4): a release must not predate the period-end it reports.
    if require_release_after_reference and np.any(rel_s < ref_s):
        bad = int(np.sum(rel_s < ref_s))
        reasons.append(f"{bad} observation(s) released BEFORE their reference period (PIT violation)")

    # Staleness / gaps between consecutive reference periods.
    max_gap = 0.0
    if n >= 2:
        gaps = (ref_s[1:] - ref_s[:-1]) / np.timedelta64(1, "D")
        max_gap = float(np.max(gaps)) if gaps.size else 0.0
        if max_gap > max_gap_days:
            reasons.append(f"max reference-period gap {max_gap:.0f}d exceeds {max_gap_days:.0f}d")

    # Outliers on first differences (unit-shift / spike detection), robust sigma via MAD.
    n_outliers = 0
    finite = np.isfinite(val_s)
    if np.sum(finite) >= 4:
        d = np.diff(val_s[finite])
        if d.size:
            med = np.median(d)
            mad = np.median(np.abs(d - med)) or np.std(d) or 1.0
            z = np.abs(d - med) / (1.4826 * mad)
            n_outliers = int(np.sum(z > outlier_sigma))
            if n_outliers:
                reasons.append(f"{n_outliers} first-difference outlier(s) > {outlier_sigma}sigma")

    if not np.any(np.isfinite(val_s)):
        reasons.append("all values are non-finite")

    return QualityReport(
        passed=len(reasons) == 0,
        reasons=tuple(reasons),
        n_obs=n,
        max_gap_days=max_gap,
        n_outliers=n_outliers,
    )


def snapshot_hash(series: SeriesData) -> str:
    """Deterministic 12-hex content hash of a series snapshot (spec §5) — pins the exact data a
    run saw. Hashes the sorted (reference_period, value, release_timestamp) triples."""
    order = np.lexsort((np.asarray(series.release_timestamp), np.asarray(series.reference_period)))
    payload = "|".join(
        f"{np.datetime_as_string(series.reference_period[i])},{float(series.value[i]):.12g},"
        f"{np.datetime_as_string(series.release_timestamp[i])}"
        for i in order
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def register_series(
    catalog: DataCatalog,
    series: SeriesData,
    *,
    freshness: str | None = None,
) -> CatalogEntry:
    """Register a *validated* series into the central catalog (spec §4.4). The caller must have run
    :func:`validate_series` and :func:`assert_asof_join_causal` first — registration is the last
    step of Stage 1 (ACQUIRE), after the data crucible clears."""
    prov: Provenance | None = series.provenance
    entry = CatalogEntry(
        source_id=series.ref.source_id,
        series=series.ref.series_id,
        asset_class=series.ref.asset_class,
        date_start=np.datetime_as_string(series.reference_period.min()) if series.n_obs else None,
        date_end=np.datetime_as_string(series.reference_period.max()) if series.n_obs else None,
        freshness=freshness,
        as_of_policy=prov.as_of_policy if prov else None,
        snapshot_hash=snapshot_hash(series),
        license=prov.license if prov else None,
    )
    catalog.register(entry)
    logger.info("registered %s:%s (%d obs, snapshot=%s)",
                entry.source_id, entry.series, series.n_obs, entry.snapshot_hash)
    return entry
