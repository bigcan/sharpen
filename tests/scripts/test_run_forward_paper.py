"""The daily forward-runner CLI end to end on real data (offline; TAILWIND Tier-2 X2).

The own-capital config (lead and financing on) is driven for two consecutive sessions into a
temporary state directory. Checks:
- the first run logs the next session's target and exits 0;
- the second settles that target at its close, logs the next one and scores the book (parity
  PASS, horizon unmet, exit 0);
- the certifying cache is byte-identical afterwards: offline mode reads, never fetches or writes
  (the 2026-09-29 overwrite was a check that meant only to read).

The challenge config adds drift wiring, but its gates file cannot be scored (paper_soak.performance
lacks keys; audit T6-05). That run must exit 1 with the error reported, parity and drift still
scored, and the book NOT flattened: a scoring failure is not a trading error.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "results" / "tailwind_v1"
CURVE = ROOT / "results" / "carry_falsification" / "yahoo_curve.parquet"
BASELINE = CACHE / "drift_baseline.json"

pytestmark = pytest.mark.skipif(
    not ((CACHE / "ohlcv_daily.parquet").exists() and CURVE.exists() and BASELINE.exists()),
    reason="TAILWIND caches / drift baseline absent (gitignored results/)")


def _digest(paths) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}


def _run(state: Path, as_of: str, config: str = "configs/tailwind_v1.yaml") -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/run_forward_paper.py", "--config", config, "--offline",
         "--as-of", as_of, "--state-dir", str(state)],
        cwd=ROOT, capture_output=True, text=True, timeout=600)


def test_two_consecutive_offline_sessions(tmp_path):
    guarded = [CACHE / n for n in ("ohlcv_daily.parquet", "ohlcv_daily_raw.parquet",
                                   "ohlcv_daily.manifest.json")] + [CURVE]
    before = _digest(guarded)

    first = _run(tmp_path, "2026-07-29")
    assert first.returncode == 0, first.stderr[-2000:]
    logged = [json.loads(x) for x in (tmp_path / "targets.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["fill_session"] for r in logged] == ["2026-07-30"]
    assert logged[0]["execution_stamp"]["decision_lead_bars"] == 1
    assert logged[0]["execution_stamp"]["financing"]["model"] == "tbill"

    second = _run(tmp_path, "2026-07-30")
    assert second.returncode == 0, second.stderr[-2000:]
    logged = [json.loads(x) for x in (tmp_path / "targets.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["fill_session"] for r in logged] == ["2026-07-30", "2026-07-31"]
    assert logged[1]["month_end_fill"] is True and logged[1]["prev_hash"] == logged[0]["hash"]
    fills = [json.loads(x) for x in (tmp_path / "fills.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [f["session"] for f in fills] == ["2026-07-30"] and fills[0]["kind"] == "target"
    verdict = json.loads((tmp_path / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["groups"]["parity"]["status"] == "PASS"
    assert verdict["forward"]["incremental_parity"]["weight_l1_drift_max"] <= 1e-12
    assert verdict["forward"]["action_drift"] is None          # no drift block in this config

    assert _digest(guarded) == before, "offline mode must never write a certifying cache"


def test_the_challenge_gates_fail_closed_without_flattening(tmp_path):
    assert _run(tmp_path, "2026-07-29", "configs/tailwind_v1_challenge.yaml").returncode == 0
    run = _run(tmp_path, "2026-07-30", "configs/tailwind_v1_challenge.yaml")
    assert run.returncode == 1
    verdict = json.loads((tmp_path / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["overall_status"] == "FAIL" and "not evaluable" in verdict["error"]
    assert verdict["forward"]["incremental_parity"]["weight_l1_drift_max"] <= 1e-12
    assert set(verdict["forward"]["action_drift"]) == {"momentum", "defensive"}
    logged = [json.loads(x) for x in (tmp_path / "targets.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["kind"] for r in logged] == ["target", "target"]  # no emergency flatten
