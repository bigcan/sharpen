"""Hashed gate files (Protocol v2 audit 2026-09-29 §6 item 7; the CRU-1 idea, repo-wide).

Every ``configs/*.gates.yaml`` is registered in ``configs/gates_registry.json`` with the sha256
of its content (line endings normalized, so a CRLF checkout and an LF checkout agree). Editing
a gates file without re-registering it fails ``tests/sharpops/test_gate_registry.py``, and a
re-registration must carry a written reason (a waiver cites the trial ledger). A post-hoc edit
is therefore always visible in the registry diff: GMGP1-BTC's G3 1.10 -> 1.05 (S528) would
have needed a stated reason at the moment of the edit.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "gates_registry.json"
PATTERN = "*.gates.yaml"


def content_sha256(path: Path) -> str:
    """sha256 of the file with CRLF/CR normalized to LF."""
    raw = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def gates_files(root: Path = ROOT) -> list[Path]:
    return sorted((root / "configs").glob(PATTERN))


def load(registry: Path = REGISTRY) -> dict:
    if not registry.exists():
        return {"schema": "gates_registry/v1", "files": {}}
    return json.loads(registry.read_text(encoding="utf-8"))


def check(root: Path = ROOT, registry: Path | None = None) -> dict:
    """{changed, unregistered, missing}: every list must be empty for a clean state."""
    reg = load(registry or root / "configs" / "gates_registry.json")["files"]
    on_disk = {p.relative_to(root).as_posix(): content_sha256(p) for p in gates_files(root)}
    return {
        "changed": sorted(f for f, h in on_disk.items() if f in reg and reg[f]["sha256"] != h),
        "unregistered": sorted(f for f in on_disk if f not in reg),
        "missing": sorted(f for f in reg if f not in on_disk),
    }


def register(paths: list[str], reason: str, *, root: Path = ROOT, registry: Path | None = None,
             today: str | None = None) -> dict:
    """(Re-)register ``paths`` (repo-relative) with a non-empty written reason. Returns the
    updated registry and writes it. A removed file is dropped only when named explicitly."""
    if not reason or not reason.strip():
        raise ValueError("a registration needs a written reason (cite the decision / ledger)")
    registry = registry or root / "configs" / "gates_registry.json"
    reg = load(registry)
    day = today or _dt.date.today().isoformat()
    for rel in paths:
        p = root / rel
        if not p.exists():
            reg["files"].pop(rel, None)
            continue
        prev = reg["files"].get(rel)
        entry = {"sha256": content_sha256(p), "registered_on": day, "reason": reason.strip()}
        if prev and prev["sha256"] != entry["sha256"]:
            entry["previous_sha256"] = prev["sha256"]
        reg["files"][rel] = entry
    reg["files"] = dict(sorted(reg["files"].items()))
    registry.write_text(json.dumps(reg, indent=2) + "\n", encoding="utf-8")
    return reg
