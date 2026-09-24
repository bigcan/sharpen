"""Tripwire: no deploy package may ship secrets to a remote GPU box.

All three packagers (bare-metal, Vast.ai, distributed HPO) zip the working tree. Before 2026-09-23
none excluded `.env`, `env.txt`, `instances.json` or private keys, so every deploy copied the API
keys, GPUHub passwords and the Kalshi private key to each rented instance. Each packager is run on a
small fake project; the secrets must be absent (at any depth) and ordinary source must survive.
"""
from __future__ import annotations

import importlib
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

SECRETS = [".env", ".env.local", "env.txt", "instances.json", "kalshi_private_key.pem",
           "sub/dir/.env", "sub/dir/server.key", "certs/client.pem"]
KEEP = ["keep.py", "requirements.txt", "sub/dir/module.py", "configs/x.yaml"]


@pytest.mark.parametrize("module", ["deploy_bare_metal", "deploy_vastai", "distributed_hpo_coordinator"])
def test_packager_excludes_secrets(module: str, tmp_path: Path) -> None:
    src = tmp_path / "proj"
    for rel in SECRETS + KEEP:
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")
    out = tmp_path / "pkg.zip"
    importlib.import_module(module).create_filtered_zip(src, str(out))
    names = {n.replace("\\", "/") for n in zipfile.ZipFile(out).namelist()}
    assert not names & set(SECRETS), f"{module} shipped secrets: {sorted(names & set(SECRETS))}"
    assert set(KEEP) <= names, f"{module} dropped source files: {sorted(set(KEEP) - names)}"
