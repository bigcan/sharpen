"""Data Scout — the Stage-1 ACQUIRE driver (spec §3 Stage 1, §7.1 role 1) — Crucible P5.

The small-operator edge is gated by *data access*, not math (``project_small_operator_strategy_reframe``),
so the Scout's job is to widen the data frontier SAFELY: survey the free connectors, and for every
series they offer, prove it clears the data crucible before it is ever trusted. For each candidate
series it runs, in order:

  1. ``connector.fetch`` — pull the PIT observations (offline when a transport is injected);
  2. :func:`quality_gate.validate_series` — gaps / staleness / outliers / the PIT sanity that a
     release never predates its reference period;
  3. :func:`quality_gate.assert_asof_join_causal` — the **as-of-join reconstruction tripwire** (CR-4 /
     LEAK-2): rebuild the feature purely from ``(value, release_timestamp)`` events against a real
     trading calendar and assert every bar value was publicly available by that bar. A source that
     leaks here is rejected, not registered.

A series is ACCEPTED iff both the quality gate and the join tripwire pass; the Scout emits a
:class:`ScoutReport` (the reviewable artifact — "output reviewed before a connector is trusted",
spec §7.1). **Registration into the catalog is a deliberately SEPARATE step** (:meth:`register_accepted`):
surveying never mutates the catalog, so an operator (or an LLM Scout) reviews the report first.

CR-1 / CR-6 boundary: the Scout NEVER reads a gate value, a verdict, or the trial ledger — it only
proves data is PIT-clean and widens what the funnel can be pointed at. Like the ``Proposer`` seam, an
LLM-backed Scout that *proposes new connectors/series* is a drop-in around this deterministic core.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..catalog import CatalogEntry, DataCatalog
from ..data.connector import DataConnector, SeriesData, SeriesRef
from ..data.panel_bridge import is_valid_terminal
from ..data.quality_gate import (
    assert_asof_join_causal,
    register_series,
    snapshot_hash,
    validate_series,
)

log = logging.getLogger("crucible.scout")


@dataclass(frozen=True, slots=True)
class ScoutFinding:
    """One surveyed series and the verdict of the data crucible on it (spec §4.3). Serializable — the
    ``SeriesData`` itself is held separately by the Scout for the register step, not in the report."""

    source_id: str
    series_id: str
    terminal: str                 # native source:series terminal (ref.terminal)
    terminal_dsl_legal: bool      # False -> a SlotRequest alias is required (COT/EDGAR-style ids)
    asset_class: str
    accepted: bool
    reasons: tuple[str, ...]      # empty iff accepted; else the failing checks
    n_obs: int
    date_start: str | None
    date_end: str | None
    snapshot_hash: str | None
    license: str | None
    as_of_policy: str | None
    release_lag_days: int


@dataclass(frozen=True, slots=True)
class ScoutReport:
    """The reviewable acquisition artifact — one finding per surveyed series (spec §7.1)."""

    findings: tuple[ScoutFinding, ...] = ()
    bar_start: str | None = None
    bar_end: str | None = None
    n_bars: int = 0
    extra: dict = field(default_factory=dict)

    def accepted(self) -> tuple[ScoutFinding, ...]:
        return tuple(f for f in self.findings if f.accepted)

    def rejected(self) -> tuple[ScoutFinding, ...]:
        return tuple(f for f in self.findings if not f.accepted)

    def asset_classes(self) -> tuple[str, ...]:
        """Distinct asset classes among ACCEPTED findings — the new domains reaching the funnel."""
        return tuple(sorted({f.asset_class for f in self.accepted()}))

    def to_json(self) -> dict:
        from dataclasses import asdict
        return {
            "n_surveyed": len(self.findings),
            "n_accepted": len(self.accepted()),
            "n_rejected": len(self.rejected()),
            "accepted_asset_classes": list(self.asset_classes()),
            "bar_start": self.bar_start, "bar_end": self.bar_end, "n_bars": self.n_bars,
            "findings": [asdict(f) for f in self.findings],
            "extra": self.extra,
        }

    def write(self, out_dir: str | Path) -> Path:
        """Write ``scout_report.json`` + a human-readable ``scout_report.md`` into ``out_dir``."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "scout_report.json").write_text(
            json.dumps(self.to_json(), indent=2, sort_keys=True), encoding="utf-8")
        (out / "scout_report.md").write_text(self._markdown(), encoding="utf-8")
        return out / "scout_report.json"

    def _markdown(self) -> str:
        lines = [
            "# Crucible Data Scout report",
            "",
            f"- surveyed: **{len(self.findings)}**  ·  accepted: **{len(self.accepted())}**  "
            f"·  rejected: **{len(self.rejected())}**",
            f"- new domains reaching the funnel: {', '.join(self.asset_classes()) or '(none)'}",
            f"- tested against calendar: {self.bar_start} → {self.bar_end} ({self.n_bars} bars)",
            "",
            "| status | source | series | class | PIT | terminal | reasons |",
            "|---|---|---|---|---|---|---|",
        ]
        for f in self.findings:
            status = "✅ accept" if f.accepted else "❌ reject"
            alias = "" if f.terminal_dsl_legal else " (alias req)"
            reasons = "; ".join(f.reasons) if f.reasons else ""
            lines.append(
                f"| {status} | {f.source_id} | {f.series_id} | {f.asset_class} | "
                f"{f.as_of_policy} | `{f.terminal}`{alias} | {reasons} |")
        lines += ["", "_Registration is a separate reviewed step; nothing here promotes past PROMISING._"]
        return "\n".join(lines)


class DataScout:
    """Surveys free-data connectors and proves each series PIT-clean before it can be trusted (§7.1)."""

    def __init__(self, connectors: list[DataConnector]) -> None:
        self._connectors = list(connectors)
        # Accepted SeriesData retained from the most recent survey, keyed (source_id, series_id), so
        # register_accepted() can register WITHOUT re-fetching. Cleared at the start of each survey.
        self._accepted_series: dict[tuple[str, str], SeriesData] = {}

    def survey(
        self,
        start: np.datetime64 | str,
        end: np.datetime64 | str,
        bar_dates: np.ndarray,
    ) -> ScoutReport:
        """Fetch → validate → as-of-join-tripwire every series every connector offers, against the
        ``bar_dates`` trading calendar. Returns a :class:`ScoutReport`; the catalog is NOT touched."""
        bars = np.asarray(bar_dates, dtype="datetime64[ns]")
        self._accepted_series = {}
        findings: list[ScoutFinding] = []
        for connector in self._connectors:
            for ref in connector.discover():
                findings.append(self._survey_one(connector, ref, start, end, bars))
        n = int(bars.shape[0])
        return ScoutReport(
            findings=tuple(findings),
            bar_start=(np.datetime_as_string(bars.min()) if n else None),
            bar_end=(np.datetime_as_string(bars.max()) if n else None),
            n_bars=n,
        )

    def _survey_one(self, connector: DataConnector, ref: SeriesRef, start, end,
                    bars: np.ndarray) -> ScoutFinding:
        prov = None
        try:
            prov = connector.provenance(ref)
        except Exception as exc:                              # noqa: BLE001 - provenance is best-effort
            log.warning("provenance failed for %s: %r", ref.terminal, exc)

        try:
            data = connector.fetch(ref, start, end)
        except Exception as exc:                              # noqa: BLE001 - connector failed closed
            return self._finding(ref, prov, accepted=False,
                                 reasons=(f"fetch failed: {exc!r}",), data=None)

        report = validate_series(data)
        reasons: list[str] = list(report.reasons)

        # The as-of-join reconstruction tripwire (CR-4) — a data-layer leak T0 is blind to.
        if data.n_obs and bars.shape[0]:
            try:
                assert_asof_join_causal(data, bars)
            except AssertionError as exc:
                reasons.append(f"as-of-join LEAK: {exc}")
        elif data.n_obs == 0:
            reasons.append("no observations in range")

        accepted = not reasons
        if accepted:
            self._accepted_series[(ref.source_id, ref.series_id)] = data
        return self._finding(ref, prov, accepted=accepted, reasons=tuple(reasons), data=data)

    def _finding(self, ref: SeriesRef, prov, *, accepted: bool, reasons: tuple[str, ...],
                 data: SeriesData | None) -> ScoutFinding:
        date_start: str | None = None
        date_end: str | None = None
        snap: str | None = None
        n_obs = 0
        if data is not None and data.n_obs > 0:            # narrows `data` for the accessors below
            n_obs = data.n_obs
            date_start = np.datetime_as_string(data.reference_period.min())
            date_end = np.datetime_as_string(data.reference_period.max())
            snap = snapshot_hash(data)
        return ScoutFinding(
            source_id=ref.source_id, series_id=ref.series_id, terminal=ref.terminal,
            terminal_dsl_legal=is_valid_terminal(ref.terminal), asset_class=ref.asset_class,
            accepted=accepted, reasons=reasons,
            n_obs=n_obs, date_start=date_start, date_end=date_end, snapshot_hash=snap,
            license=(prov.license if prov else None),
            as_of_policy=(prov.as_of_policy if prov else None),
            release_lag_days=(prov.release_lag_days if prov else 0),
        )

    def register_accepted(self, report: ScoutReport, catalog: DataCatalog, *,
                          freshness: str | None = None) -> list[CatalogEntry]:
        """Register ONLY the accepted findings from the most recent survey into ``catalog`` — the
        explicit, post-review Stage-1 commit. Re-uses the retained SeriesData (no re-fetch). A finding
        whose SeriesData is no longer cached (a stale report from an earlier survey) is skipped with a
        warning rather than silently re-fetched."""
        entries: list[CatalogEntry] = []
        for f in report.accepted():
            data = self._accepted_series.get((f.source_id, f.series_id))
            if data is None:
                log.warning("register_accepted: %s not in current survey cache — skipped "
                            "(re-run survey())", f.terminal)
                continue
            entries.append(register_series(catalog, data, freshness=freshness))
        log.info("registered %d/%d accepted series into the catalog", len(entries),
                 len(report.accepted()))
        return entries
