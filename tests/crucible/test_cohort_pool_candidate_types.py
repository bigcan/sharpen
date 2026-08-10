"""crucible-v12.1 — WHICH candidates the cohort gate is allowed to see.

Through v12.0 ``loop._evaluate_cohort_gate`` hard-filtered its pool to ``candidate_type ==
"overlay"``. That single line was the structural mispairing behind the 0-PROMISING record: the
high-information hypotheses (cross-sectional, carrying the panel's breadth and so the large
per-member δ in ``IR_cohort = δ·√K·hit_rate``) were adjudicated only by the per-candidate gate
measured at ~0% power, while the one gate with measured power — the selection-aware MC null — only
ever saw one-scalar-per-day overlays that are structurally correlated with the base book they tilt.

This file pins the new behaviour at the LOOP boundary (the statistics live in
``tests/signals/test_cohort_eval.py``): the pool is config-driven, the flag defaults to the old
overlay-only shape, each member is labelled with its own type, and the cross-sectional sleeve knobs
handed to the scorer are the MINE's, not the scorer's defaults.
"""
from __future__ import annotations

import numpy as np
import pytest

from finrl_pro_ds.crucible.agentic import loop as loop_mod
from finrl_pro_ds.crucible.agentic.hypothesis import PreRegisteredSpec
from finrl_pro_ds.signals.generation.cohort import CohortConfig
from finrl_pro_ds.signals.spec import SignalSpec

_MC = {"enabled": True, "n_reps": 8, "alpha_cohort": 0.05, "block_length": 21}


def _ccfg(*, include_cross_sectional: bool) -> CohortConfig:
    return CohortConfig(
        max_cohort_size=12, min_cohort_size=3, max_pairwise_corr=0.35, promising_dsr=0.90,
        cohort_hlz_t_min=3.0, min_book_uplift=0.10, combiner_redundancy_strength=0.5,
        include_cross_sectional=include_cross_sectional)


def _spec(name: str, formula: str, candidate_type: str) -> PreRegisteredSpec:
    return PreRegisteredSpec(
        spec=SignalSpec(name=name, hypothesis="h", family="101alpha", expected_sign=1,
                        horizons=(1,), neutralization=(), universe="u", cost_profile="c",
                        sample_window=(None, None), candidate_type=candidate_type),
        formula=formula, candidate_hash=f"hash-{name}", economic_rationale="r",
        proposal_ts="2026-08-10T00:00:00Z")


_SPECS = [
    _spec("ov-a", "fred:X00", "overlay"),
    _spec("ov-b", "fred:X01", "overlay"),
    _spec("xs-a", "rank(close)", "cross_sectional"),
    _spec("xs-b", "ts_rank(volume, 10)", "cross_sectional"),
    _spec("xs-c", "rank(stddev(close, 10))", "cross_sectional"),
]


@pytest.fixture()
def captured(monkeypatch):
    """Stub the SCORER and record exactly what pool the loop hands it. Returns a dict the test fills
    in by calling ``loop._evaluate_cohort_gate``; the stub returns None so no card is built (this
    file is about pool assembly, not verdicts)."""
    seen: dict = {}

    def _stub(panel, base_returns, timestamps, formulas, ccfg, fcfg, **kw):
        seen["formulas"] = dict(formulas)
        seen["candidate_types"] = dict(kw.get("candidate_types") or {})
        seen["hold_horizon"] = kw.get("hold_horizon")
        seen["ls_min_names"] = kw.get("ls_min_names")
        return None

    monkeypatch.setattr(loop_mod, "evaluate_cohort", _stub)
    return seen


def _run(cohort_cfg, ek=None):
    return loop_mod._evaluate_cohort_gate(
        specs=_SPECS, panel=object(), base_returns={}, timestamps=np.zeros(4),
        cfg=object(), ek=(ek if ek is not None else {}), run_id="run-1",
        crucible_version="crucible-v12.1", gates_hash="519158fa1450", proposal_ts="2026-08-10",
        data_snapshot_hash=None, cohort_cfg=cohort_cfg, cohort_mc_kwargs=dict(_MC),
        cohort_gates_hash="cohortgates1")


def test_overlay_only_is_the_default_pool(captured) -> None:
    """``include_cross_sectional: false`` reproduces the pre-v12.1 hard filter exactly."""
    _run(_ccfg(include_cross_sectional=False))
    assert set(captured["formulas"]) == {"hash-ov-a", "hash-ov-b"}
    assert set(captured["candidate_types"].values()) == {"overlay"}


def test_cross_sectional_candidates_reach_the_cohort_gate(captured) -> None:
    """The whole point of v12.1: with the flag on, the pool is the UNION and every member carries
    its own type so the scorer can route it to the right funnel path."""
    _run(_ccfg(include_cross_sectional=True))
    assert set(captured["formulas"]) == {f"hash-{n}" for n in ("ov-a", "ov-b", "xs-a", "xs-b", "xs-c")}
    assert captured["candidate_types"]["hash-xs-a"] == "cross_sectional"
    assert captured["candidate_types"]["hash-ov-a"] == "overlay"
    assert sum(1 for t in captured["candidate_types"].values() if t == "cross_sectional") == 3


def test_sleeve_knobs_come_from_the_mine_not_the_scorer_defaults(captured) -> None:
    """A cross-sectional cohort member must be scored with the SAME hold_horizon / ls_min_names the
    mine used, or the cohort adjudicates a different stream than the one pre-registered."""
    _run(_ccfg(include_cross_sectional=True), ek={"hold_horizon": 5, "ls_min_names": 11})
    assert captured["hold_horizon"] == 5
    assert captured["ls_min_names"] == 11


def test_pool_is_empty_when_no_admitted_type_is_present(captured) -> None:
    """An all-cross-sectional tick with the flag OFF must no-op (return no cards), not crash and not
    silently score cross-sectional specs through the overlay path."""
    cards, prov = loop_mod._evaluate_cohort_gate(
        specs=[s for s in _SPECS if s.spec.candidate_type == "cross_sectional"],
        panel=object(), base_returns={}, timestamps=np.zeros(4), cfg=object(), ek={},
        run_id="run-1", crucible_version="crucible-v12.1", gates_hash="519158fa1450",
        proposal_ts="2026-08-10", data_snapshot_hash=None,
        cohort_cfg=_ccfg(include_cross_sectional=False), cohort_mc_kwargs=dict(_MC),
        cohort_gates_hash="cohortgates1")
    assert cards == [] and prov == {}
    assert "formulas" not in captured                      # the scorer was never called


def test_disabled_gate_still_no_ops(captured) -> None:
    """``enabled: false`` short-circuits before any pool is assembled (manifest byte-identity)."""
    cards, prov = loop_mod._evaluate_cohort_gate(
        specs=_SPECS, panel=object(), base_returns={}, timestamps=np.zeros(4), cfg=object(), ek={},
        run_id="run-1", crucible_version="crucible-v12.1", gates_hash="519158fa1450",
        proposal_ts="2026-08-10", data_snapshot_hash=None,
        cohort_cfg=_ccfg(include_cross_sectional=True),
        cohort_mc_kwargs={**_MC, "enabled": False}, cohort_gates_hash="cohortgates1")
    assert cards == [] and prov == {}
    assert "formulas" not in captured


def test_shipped_gates_file_admits_cross_sectional() -> None:
    """The capability is only worth anything if it is actually ON in the shipped config — the
    'declared but unread' failure mode this project keeps rediscovering (v11.0 capturability, the
    unreachable cohort floor). Read from the file, never asserted from the code default."""
    from pathlib import Path

    from finrl_pro_ds.signals.generation.config import load_cohort_config

    root = Path(__file__).resolve().parents[2]
    ccfg, mc = load_cohort_config(root / "configs" / "signal_eval.gates.yaml",
                                  root / "configs" / "crucible_cohort.gates.yaml")
    assert mc["enabled"] is True
    assert ccfg.include_cross_sectional is True
