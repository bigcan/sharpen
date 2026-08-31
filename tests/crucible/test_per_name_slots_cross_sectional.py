"""U3: per-name (T,N) alt-data is reachable by the CROSS-SECTIONAL search (2026-07-29).

Before this, every non-price dataset the project connected could only act as a market-timing overlay
— one scalar per day, T observations instead of T x N. `crucible-v2.9` had closed the cross-sectional
route wholesale (C2-06) because BROADCAST slots rank() to constant/dead genomes; the repair is to
filter by SHAPE rather than exclude every slot.

These tests pin both halves: the terminal registry admits (T,N) and still excludes (T,), and a
per-name slot actually produces a non-degenerate cross-sectional score end-to-end.
"""
from __future__ import annotations

import numpy as np
import pytest

from sharpen.signals.eval_harness import assert_causal
from sharpen.signals.features import make_synthetic_panel
from sharpen.signals.generation.dsl_signal import DslSignal, eval_on_panel
from sharpen.signals.generation.grammar import (
    INPUTS,
    available_terminals,
    cross_sectional_terminals,
    per_name_slots,
)

_T, _N = 260, 20


def _panel(*, per_name=True, broadcast=True):
    rng = np.random.default_rng(11)
    slots: dict[str, np.ndarray] = {}
    if per_name:
        slots["twse:inst_net"] = rng.standard_normal((_T, _N))          # (T,N) per-name
    if broadcast:
        slots["macro:regime"] = np.sin(np.arange(_T) / 30.0)            # (T,) broadcast
    return make_synthetic_panel(T=_T, N=_N, seed=3, feature_slots=slots)


def test_cross_sectional_registry_admits_per_name_and_excludes_broadcast():
    p = _panel()
    xs = cross_sectional_terminals(p)
    assert "twse:inst_net" in xs, "per-name slot is still unreachable cross-sectionally (U3 not wired)"
    assert "macro:regime" not in xs, (
        "broadcast slot leaked into the cross-sectional registry — that is the v2.9 C2-06 defect "
        "(it rank()s to a constant and pollutes gen_n / the DSR dispersion pool)")
    assert set(INPUTS).issubset(xs)


def test_overlay_registry_still_sees_every_slot():
    """The overlay path collapses the cross-section to a per-day scalar, so a broadcast series is
    exactly its input — its registry must be unchanged."""
    ov = available_terminals(_panel())
    assert {"twse:inst_net", "macro:regime"}.issubset(ov)


def test_ohlcv_only_panel_is_byte_identical_to_INPUTS():
    """Strict-extension guarantee: with no (T,N) slot the cross-sectional draw is exactly what it was
    at v2.9, so no existing search trajectory moves."""
    assert cross_sectional_terminals(make_synthetic_panel(T=60, N=8, seed=1)) == INPUTS
    assert cross_sectional_terminals(_panel(per_name=False)) == INPUTS


def test_per_name_slots_reports_only_2d():
    p = _panel()
    assert per_name_slots(p) == ("twse:inst_net",)
    assert per_name_slots(_panel(per_name=False)) == ()


def test_per_name_slot_yields_a_NON_degenerate_cross_sectional_score():
    """The point of the whole change. A rank() on a per-name slot must vary across names; the same
    construction on a broadcast slot is identically zero — which is precisely why the broadcast one
    stays excluded."""
    p = _panel()
    per_name = eval_on_panel("rank(twse:inst_net)", p)
    assert np.isfinite(per_name).all()
    xs_std = np.nanstd(per_name, axis=1)                 # dispersion ACROSS names, per day
    assert float(np.nanmean(xs_std)) > 0.1, "per-name slot produced a flat cross-section"

    broadcast = eval_on_panel("rank(macro:regime)", p)
    assert float(np.nanmax(np.abs(np.nanstd(broadcast, axis=1)))) < 1e-9, (
        "a centred rank on a constant row should be flat — if this fires the exclusion rationale "
        "no longer holds and the shape filter should be revisited")


def test_per_name_slot_signal_is_causal_under_the_tier0_tripwire():
    """A (T,N) slot must survive truncation-equivalence: compute(panel.truncated(t))[t] ==
    compute(panel)[t]. Panel._sliced_slots already slices both shapes on axis 0; this pins it."""
    ok, msg = assert_causal(DslSignal("rank(twse:inst_net)"), _panel())
    assert ok, msg
    ok2, msg2 = assert_causal(DslSignal("ts_rank(twse:inst_net, 10) - rank(close)"), _panel())
    assert ok2, msg2


def test_per_name_genome_is_SCORED_end_to_end_by_the_cross_sectional_search():
    """The end-to-end assertion the registry test cannot make: a genome referencing a (T,N) slot must
    survive the whole cross-sectional path — parse, eval_on_panel, rank-L/S book, CPCV fitness — and
    come back with a real score rather than being culled as degenerate or infeasible.

    The slot here is pure noise, so a NEGATIVE marginal ΔSR is the correct outcome; what is asserted
    is that it was measured at all."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    if str(root / "scripts") not in sys.path:
        sys.path.insert(0, str(root / "scripts"))
    from research.crucible_calibration import _panel_ts, _proxy_base_sleeves, load_calib

    from sharpen.signals.generation.evolve import evolve

    cc = load_calib(root / "configs" / "crucible_calibration.gates.yaml",
                    root / "configs" / "signal_eval.gates.yaml")
    ek = dict(cc.ek)
    ek.update(pop_size=8, n_generations=1)
    p = _panel(broadcast=False)
    rep = evolve(["rank(twse:inst_net)", "rank(close)"], p, _proxy_base_sleeves(p, hold=ek["hold_horizon"]),
                 _panel_ts(p), cc.fit_cfg, candidate_type="cross_sectional", **ek)

    hit = next((c for c in rep.hall_of_fame if "twse:inst_net" in c.formula), None)
    assert hit is not None, "the per-name seed never surfaced — it was culled before scoring"
    assert hit.result is not None and hit.reason == "", f"culled: {hit.reason!r}"
    assert np.isfinite(hit.result.delta_sr_oos), "no marginal ΔSR was measured for the per-name genome"
    assert hit.result.n_paths > 0


def test_build_panel_feature_slot_shapes_and_absent_names(monkeypatch):
    """The bridge assembly: per-ticker series stack into (T,N) in `tickers` order, and a ticker with
    no request becomes an ALL-NaN column — never a 0.0, which would be a tradeable value."""
    from sharpen.crucible.data import panel_bridge as pb

    bars = (np.datetime64("2020-01-01") + np.arange(5)).astype("datetime64[ns]")
    seen: list[str] = []

    class _Conn:
        def fetch(self, ref, start, end):
            seen.append(ref)
            return ref

    monkeypatch.setattr(pb, "validate_series", lambda d: type("R", (), {
        "passed": True, "reasons": (), "n_obs": 5, "max_gap_days": 0.0})())
    monkeypatch.setattr(pb, "asof_join", lambda d, b: np.full(len(b), float(d)))
    monkeypatch.setattr(pb, "assert_asof_join_causal", lambda d, b: None)

    reqs = {"AAA": pb.SlotRequest(_Conn(), 1.0), "CCC": pb.SlotRequest(_Conn(), 3.0)}
    out = pb.build_panel_feature_slot(reqs, ["AAA", "BBB", "CCC"], bars, terminal="x:flow",
                                      start="2020-01-01", end="2020-01-05")
    arr = out["x:flow"]
    assert arr.shape == (5, 3)
    assert np.allclose(arr[:, 0], 1.0) and np.allclose(arr[:, 2], 3.0)   # column order = tickers
    assert np.isnan(arr[:, 1]).all(), "absent ticker must be NaN, not 0.0"

    with pytest.raises(ValueError, match="not a DSL-legal name"):
        pb.build_panel_feature_slot({}, ["AAA"], bars, terminal="9bad:name",
                                    start="2020-01-01", end="2020-01-05")
