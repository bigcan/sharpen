"""crucible-v15.0: the Data Scout's bar-coverage floor (``altdata.min_bar_coverage``).

The scout rejected only an EMPTY series, so a series available for a sliver of the bar calendar (FRED's
~3-year ICE window on a 19-year clock, TAIFEX's 3-4 days) became a slot, spawned pre-registered overlay
specs and spent LORD++ wealth on near-powerless tests.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from sharpen.crucible.agentic import DataScout
from sharpen.crucible.data import Provenance, SeriesData, SeriesRef

_BARS = np.arange(np.datetime64("2010-01-01"), np.datetime64("2020-01-01"),
                  np.timedelta64(1, "D")).astype("datetime64[ns]")


class _Series:
    source_id = "fake"
    asset_class = "macro"

    def __init__(self, first: str) -> None:
        self._first = first

    def discover(self) -> list[SeriesRef]:
        return [SeriesRef("fake", f"s{self._first[:4]}", "macro", frequency="D")]

    def fetch(self, ref, start, end, *, as_of=None) -> SeriesData:
        rp = np.arange(np.datetime64(self._first), np.datetime64("2020-01-01"),
                       np.timedelta64(7, "D")).astype("datetime64[ns]")
        return SeriesData(ref, rp, np.linspace(0.0, 1.0, rp.size), rp + np.timedelta64(1, "D"))

    def provenance(self, ref) -> Provenance:
        return Provenance("fake", "u", "lic", "release-lag", release_lag_days=1)


def test_late_starting_series_is_rejected_by_the_floor() -> None:
    long, late = _Series("2010-01-05"), _Series("2018-06-01")
    rep = DataScout([long, late], min_bar_coverage=0.5).survey("2010-01-01", "2020-01-01", _BARS)
    by_id = {f.series_id: f for f in rep.findings}
    assert by_id["s2010"].accepted
    assert not by_id["s2018"].accepted
    assert any("min_bar_coverage" in r for r in by_id["s2018"].reasons)


def test_no_floor_is_the_legacy_behaviour() -> None:
    rep = DataScout([_Series("2018-06-01")]).survey("2010-01-01", "2020-01-01", _BARS)
    assert rep.findings[0].accepted


def test_shipped_altdata_gates_set_the_floor() -> None:
    """crucible-v16.0: the operator decision (delegated 2026-09-30) sets the floor at 0.50 — a series
    must cover at least half the panel, i.e. begin well before the holdout."""
    import scripts.research.crucible_orchestrator as orch

    shipped = Path(__file__).resolve().parents[2] / "configs/crucible_altdata.gates.yaml"
    cfg = yaml.safe_load(shipped.read_text(encoding="utf-8"))
    assert cfg["altdata"]["min_bar_coverage"] == 0.5
    assert orch._altdata_min_bar_coverage(shipped) == 0.5


def test_orchestrator_reads_the_floor_when_configured(tmp_path: Path) -> None:
    import scripts.research.crucible_orchestrator as orch

    p = tmp_path / "altdata.gates.yaml"
    p.write_text("altdata:\n  min_bar_coverage: 0.5\n", encoding="utf-8")
    assert orch._altdata_min_bar_coverage(p) == 0.5
