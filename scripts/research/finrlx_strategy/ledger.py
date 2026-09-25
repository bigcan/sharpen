"""The trial ledger: every strategy variant evaluated in this mission, on any window, is one row.

The deflated Sharpe needs an honest trial count, including abandoned variants and scratch explorations, so
the ledger is append-only JSONL under results/ (gitignored) and a committed summary is regenerated from it.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

from .paths import RESULTS

LEDGER = RESULTS / "trial_ledger.jsonl"


def spec_hash(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def append(*, name: str, window: str, spec: dict, metrics: dict, kind: str = "grid", note: str = "",
           path: Path = LEDGER) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "name": name,
           "window": window, "kind": kind, "spec_hash": spec_hash(spec), "spec": spec, "metrics": metrics,
           "note": note}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return row


def read(path: Path = LEDGER) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def n_trials(path: Path = LEDGER, prior_family: int = 0) -> int:
    """Distinct strategy variants ever evaluated (by spec hash) plus a prior-family count carried in."""
    return len({r["spec_hash"] for r in read(path)}) + prior_family
