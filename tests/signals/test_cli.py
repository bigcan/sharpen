"""Integration smoke: the eval_signals CLI builds a panel, evaluates the demo registry,
and writes a scorecard."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "eval_signals_cli", ROOT / "scripts" / "research" / "eval_signals.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_smoke(tmp_path) -> None:
    cli = _load_cli()
    gpath = tmp_path / "gates.yaml"
    gpath.write_text(yaml.safe_dump({
        "universe": {"min_names_per_day": 4},
        "coverage": {"min_days": 150},
        "gross_power": {"horizons": [1, 5], "primary_horizon": 1},
    }))
    out = tmp_path / "out"
    rs = cli.main(["--batch", "smoke", "--panel", "synthetic:300,40",
                   "--gates", str(gpath), "--out", str(out)])

    from finrl_pro_ds.signals.library import demo
    assert rs.n_trials == len(demo.SIGNALS)          # all demo signals are causal -> deflated
    assert (out / "scorecard.json").exists() and (out / "scorecard.md").exists()
    data = json.loads((out / "scorecard.json").read_text(encoding="utf-8"))
    assert data["batch_name"] == "smoke"
    assert len(data["cards"]) == len(demo.SIGNALS)
    # every card carries the survivorship caveat (synthetic panel = not survivorship-free)
    assert all(any("UPPER BOUND" in cav for cav in c["caveats"]) for c in data["cards"])


def test_cli_unknown_panel_errors() -> None:
    cli = _load_cli()
    import pytest

    with pytest.raises(SystemExit):
        cli.build_panel("bogus")
    with pytest.raises(SystemExit):
        cli.build_panel("sharadar")
