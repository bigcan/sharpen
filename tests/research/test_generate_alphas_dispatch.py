"""Dispatch tripwire — `generate_alphas --mode real` routes on `generation.panel` (S553-cont step 5).

Offline (loaders + evolve monkeypatched): asserts panel=taiwan takes the Taiwan loader + TX/TE/TF
base book and NEVER the cross_asset path, and that the report lands in the per-substrate subdir.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import sharpen.data.cross_asset_panel_loader as cpl
import sharpen.data.taiwan_panel_loader as tpl
import sharpen.signals.generation.base_sleeves as bs
import scripts.research.generate_alphas as ga

ROOT = Path(__file__).resolve().parents[2]
TW_GATES = ROOT / "configs" / "taiwan_signal_eval.gates.yaml"


def _fake_report() -> SimpleNamespace:
    return SimpleNamespace(gen_n_total=3, gen_n_eff=3, promising=[], hall_of_fame=[],
                           holdout_validation={}, pbo=None)


def test_real_mode_dispatches_to_taiwan(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}
    panel = ga._synthetic_panel(120, 10, planted=False, seed=0)

    def _fake_load_taiwan(start, end=None, **kw):
        calls["taiwan_panel"] = (start, end)
        return panel

    def _fake_taiwan_sleeves(p, **kw):
        calls["taiwan_sleeves"] = True
        return {"tsmom": np.zeros(p.T)}

    def _boom(*a, **k):
        raise AssertionError("cross_asset path must NOT run for panel=taiwan")

    monkeypatch.setattr(tpl, "load_taiwan_panel", _fake_load_taiwan)
    monkeypatch.setattr(bs, "taiwan_base_sleeves", _fake_taiwan_sleeves)
    monkeypatch.setattr(cpl, "load_cross_asset_panel", _boom)      # cross_asset must be untouched
    monkeypatch.setattr(ga, "evolve", lambda *a, **k: _fake_report())

    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "generate_alphas.py", "--mode", "real", "--force",
        "--config", str(TW_GATES), "--start", "2015-01-01", "--out", str(out)])

    assert ga.main() == 0
    assert calls.get("taiwan_sleeves") is True and "taiwan_panel" in calls
    report = json.loads((out / "taiwan" / "generation_report.json").read_text(encoding="utf-8"))
    assert report["panel"] == "taiwan" and report["mode"] == "real"
    assert report["n_promising"] == 0
