"""scripts/sharpops_promotion_check.py (Tier-2 N3): every non-PASS state exits non-zero."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from sharpen.sharpops import gate_registry as gr

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def pc():
    spec = importlib.util.spec_from_file_location("sharpops_promotion_check",
                                                  ROOT / "scripts" / "sharpops_promotion_check.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _tree(tmp: Path, verdict: str = "PASS") -> Path:
    (tmp / "configs").mkdir()
    (tmp / "res").mkdir()
    (tmp / "data.parquet").write_bytes(b"data")
    (tmp / "configs" / "book.yaml").write_bytes(b"k: 1\n")
    ladder = tmp / "configs" / "ladder.gates.yaml"
    ladder.write_text(yaml.safe_dump({"ladder": {"workstreams": {"w": {
        "target_rung": "paper", "enforcing_artifact_glob": "res/art_*.json"}}}}), encoding="utf-8")
    h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    art = {"pins": {"files_sha256": {"data.parquet": h(tmp / "data.parquet")}},
           "lineage": {"config_sha256": {"configs/book.yaml": h(tmp / "configs" / "book.yaml")}},
           "verdict": {"decision": verdict}}
    (tmp / "res" / "art_1.json").write_text(json.dumps(art), encoding="utf-8")
    gr.register(["configs/ladder.gates.yaml"], "t", root=tmp, registry=tmp / "configs" / "gates_registry.json")
    return ladder


@pytest.mark.parametrize("verdict,code", [("PASS", 0), ("REVIEW", 3), ("BLOCK", 1), ("weird", 1)])
def test_verdict_maps_to_exit(pc, tmp_path, verdict, code):
    ladder = _tree(tmp_path, verdict)
    res = pc.evaluate("w", root=tmp_path, ladder=ladder)
    assert pc.EXIT[res["decision"]] == code


def test_changed_pinned_data_is_stale(pc, tmp_path):
    ladder = _tree(tmp_path)
    (tmp_path / "data.parquet").write_bytes(b"refetched")
    res = pc.evaluate("w", root=tmp_path, ladder=ladder)
    assert res["decision"] == "STALE" and pc.EXIT["STALE"] == 1


def test_changed_config_is_stale(pc, tmp_path):
    ladder = _tree(tmp_path)
    (tmp_path / "configs" / "book.yaml").write_bytes(b"k: 2\n")
    assert pc.evaluate("w", root=tmp_path, ladder=ladder)["decision"] == "STALE"


def test_dirty_registry_fails(pc, tmp_path):
    ladder = _tree(tmp_path)
    ladder.write_text(ladder.read_text(encoding="utf-8") + "# edit\n", encoding="utf-8")
    assert pc.evaluate("w", root=tmp_path, ladder=ladder)["decision"] == "FAIL"


def test_missing_artifact_and_unknown_workstream_fail(pc, tmp_path):
    ladder = _tree(tmp_path)
    (tmp_path / "res" / "art_1.json").unlink()
    assert pc.evaluate("w", root=tmp_path, ladder=ladder)["decision"] == "FAIL"
    assert pc.evaluate("nope", root=tmp_path, ladder=ladder)["decision"] == "FAIL"


def test_tailwind_is_blocked_on_todays_tree(pc):
    """N3 acceptance: promotion_check on today's tree exits non-zero (BLOCK)."""
    if not list(ROOT.glob("results/tailwind_v1/x1_recertification_*.json")):
        pytest.skip("X1 artifact not present in this checkout (results/ is gitignored)")
    assert pc.main(["--workstream", "tailwind-v1"]) != 0
