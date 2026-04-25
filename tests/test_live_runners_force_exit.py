"""Regression test for the S489/S498 PID-1 force-exit pattern in live runners.

Background (S489): when the engine logs `=== TRADING STOPPED ===`, non-daemon
helpers (wandb-core subprocess, twisted reactor, ccxt aiohttp pools) can keep
PID 1 alive. With `restart: unless-stopped`, docker won't restart a container
whose PID 1 is still up, so an IB-strategy hang during ibgateway restart left
gmgp1-gold "Up (unhealthy)" for 2h+ until manual intervention.

The fix wraps `asyncio.run(...)` in `try/except/finally` and calls
`os._exit(exit_code)` from the `finally` block. Originally shipped only in
`run_live_ib.py`. S498 generalizes the pattern to the other 3 runners so the
watchdog auto-restart path (`AUTO_RESTART_AFTER_N` consecutive sweeps in
`scripts/watchdog_docker.py`) actually has a stopped container to restart.

This test asserts the pattern is present in all 4 runners. If a future refactor
removes it, this test fires loud BEFORE deployment, not after a 2h alert loop.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNERS = [
    REPO_ROOT / "scripts" / "run_live.py",
    REPO_ROOT / "scripts" / "run_live_ctrader.py",
    REPO_ROOT / "scripts" / "run_live_dxtrade.py",
    REPO_ROOT / "scripts" / "run_live_ib.py",
]


def _has_os_exit_in_finally(tree: ast.AST) -> bool:
    """Return True iff some `try/finally` contains `os._exit(...)` in the finally block."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for stmt in node.finalbody:
            for sub in ast.walk(stmt):
                if not isinstance(sub, ast.Call):
                    continue
                func = sub.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "_exit"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "os"
                ):
                    return True
    return False


@pytest.mark.parametrize("runner_path", RUNNERS, ids=lambda p: p.name)
def test_runner_has_force_exit_in_finally(runner_path: Path):
    assert runner_path.exists(), f"Runner missing: {runner_path}"
    src = runner_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    assert _has_os_exit_in_finally(tree), (
        f"{runner_path.name}: missing `os._exit(...)` in a try/finally block. "
        f"This is the S489/S498 PID-1 force-exit pattern; without it, docker "
        f"`restart: unless-stopped` may not fire when wandb-core or twisted "
        f"reactor blocks the natural exit. See "
        f"`project_ibgateway_restart_orphan_netns.md`."
    )
