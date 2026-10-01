"""Hashed gate files: an edit to any configs/*.gates.yaml must be re-registered with a reason.

If this fails after you edited a gates file, that is the point: register the change with the
decision it rests on (`python scripts/sharpops_gate_registry.py --register <path> --reason ...`).
"""
from __future__ import annotations

import json

import pytest

from sharpen.sharpops import gate_registry as gr


def test_every_gates_file_is_registered_and_unchanged():
    res = gr.check()
    assert res == {"changed": [], "unregistered": [], "missing": []}, (
        "gates files changed without re-registration: " + json.dumps(res))


def test_every_registration_has_a_reason():
    for f, e in gr.load()["files"].items():
        assert e.get("reason", "").strip(), f"{f} registered without a reason"


def _mini(tmp_path):
    (tmp_path / "configs").mkdir()
    g = tmp_path / "configs" / "x.gates.yaml"
    g.write_bytes(b"a: 1\r\nb: 2\r\n")
    return g


def test_edit_is_detected_and_reregistration_needs_a_reason(tmp_path):
    g = _mini(tmp_path)
    reg = tmp_path / "configs" / "gates_registry.json"
    assert gr.check(tmp_path, reg)["unregistered"] == ["configs/x.gates.yaml"]
    with pytest.raises(ValueError, match="reason"):
        gr.register(["configs/x.gates.yaml"], "  ", root=tmp_path, registry=reg)
    gr.register(["configs/x.gates.yaml"], "initial", root=tmp_path, registry=reg, today="2026-09-30")
    assert gr.check(tmp_path, reg) == {"changed": [], "unregistered": [], "missing": []}
    g.write_bytes(b"a: 1\r\nb: 3\r\n")                          # a post-hoc threshold edit
    assert gr.check(tmp_path, reg)["changed"] == ["configs/x.gates.yaml"]
    out = gr.register(["configs/x.gates.yaml"], "memo X", root=tmp_path, registry=reg)
    assert "previous_sha256" in out["files"]["configs/x.gates.yaml"]


def test_line_endings_do_not_change_the_hash(tmp_path):
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_bytes(b"k: 1\r\nv: 2\r\n")
    b.write_bytes(b"k: 1\nv: 2\n")
    assert gr.content_sha256(a) == gr.content_sha256(b)


def test_deleted_file_is_reported_missing(tmp_path):
    g = _mini(tmp_path)
    reg = tmp_path / "configs" / "gates_registry.json"
    gr.register(["configs/x.gates.yaml"], "initial", root=tmp_path, registry=reg)
    g.unlink()
    assert gr.check(tmp_path, reg)["missing"] == ["configs/x.gates.yaml"]
