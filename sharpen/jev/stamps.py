"""Reproducible stamps for ATL x Jev artifacts (``docs/research/atl_jev_phase2_architecture.md`` ADR-7).

This checkout runs with ``core.autocrlf``, so a working-copy file's raw bytes depend on the checkout, not on
its content. Every stamp here hashes CRLF->LF-normalized bytes, which equals the hash of the git blob's
content, so an artifact's stamp can be re-derived from the repository on any machine.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def lf_bytes(path: str | Path) -> bytes:
    return Path(path).read_bytes().replace(b"\r\n", b"\n")


def gates_sha(path: str | Path) -> str:
    """sha256 of a gates file's LF-normalized bytes."""
    return hashlib.sha256(lf_bytes(path)).hexdigest()


def source_sha(*paths: str | Path) -> str:
    """12-hex hash over the LF-normalized source of ``paths`` (order-sensitive, file names included)."""
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).name.encode("utf-8") + b"\0")
        h.update(lf_bytes(p) + b"\0")
    return h.hexdigest()[:12]


def git_commit(root: str | Path) -> dict:
    """``{"commit": <HEAD sha or None>, "dirty": <bool or None>}`` for the repository at ``root``."""
    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=30,
                                 check=True)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"commit": head, "dirty": None if status is None else bool(status)}
