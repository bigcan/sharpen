"""C1.4 — combiner-selection gate wiring (``combiner_beats_static``, ADR-C1-4).

Exercises ``scripts.run_combiner_selection.select_combiner`` on the synthetic two-sleeve
bundle (no network): the in-sample-select → held-out-OOS verdict, and that the gate threshold
decides ship_dynamic vs ship_static_rp. Placed here (not tests/scripts/) to reuse the
``bundle`` / ``paper2_cfg`` fixtures.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_combiner_selection import select_combiner  # noqa: E402

_GRID = dict(tilts=(1.0, 3.0), perf_windows=(63, 126))      # small grid → fast


def _gates(min_uplift: float) -> dict:
    return {"gates": {"combiner_beats_static": {
        "metric": "oos_net_sharpe", "baseline": "static_inverse_vol",
        "min_uplift_vs_baseline": min_uplift, "else": "ship_static_rp"}}}


def test_ships_static_when_uplift_below_min(paper2_cfg, bundle):
    """An unreachable uplift bar ⇒ the gate ships the static inverse-vol book."""
    v = select_combiner(bundle, paper2_cfg, _gates(1e9), **_GRID)
    assert v["decision"] == "ship_static_rp"
    assert v["uplift_vs_static"] < v["min_uplift_vs_baseline"]


def test_ships_dynamic_when_uplift_above_min(paper2_cfg, bundle):
    """A trivially-met bar ⇒ ship_dynamic — proves the threshold is the only thing gating."""
    v = select_combiner(bundle, paper2_cfg, _gates(-1e9), **_GRID)
    assert v["decision"] == "ship_dynamic"


def test_verdict_structure_is_advisory_and_split_consistent(paper2_cfg, bundle):
    v = select_combiner(bundle, paper2_cfg, _gates(0.10), **_GRID)
    assert v["advisory"] is True                            # never a capital promotion
    assert v["binding_requires"].startswith("C2 CPCV")
    assert v["n_bars_in_sample"] + v["n_bars_oos"] == v["n_bars_total"]
    # selected config came from the swept grid.
    assert v["selected"]["tilt_strength"] in _GRID["tilts"]
    assert v["selected"]["perf_window"] in _GRID["perf_windows"]
    assert v["decision"] in ("ship_dynamic", "ship_static_rp")


def test_missing_gate_raises(paper2_cfg, bundle):
    with pytest.raises(KeyError):
        select_combiner(bundle, paper2_cfg, {"gates": {}}, **_GRID)
