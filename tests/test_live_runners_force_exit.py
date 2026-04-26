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


# ---------------------------------------------------------------------------
# Risk-config pass-through parity (S498-cont 2026-04-26)
#
# Background: run_live.py used to drop max_net_short_exposure /
# max_gross_exposure / daily_turnover_limit / min_effective_bets when
# constructing CryptoRiskConfig, falling back to dataclass defaults
# regardless of YAML. This silently clipped sg1-btc's intended -1.0 net
# short to -0.5 and gmgp1-btc to a 1.5x daily turnover cap that the
# strategy hit ~10x/day. The cTrader/IB/DXtrade runners already passed
# these keys; only the crypto runner was inconsistent.
#
# This test asserts the union of risk keys is uniformly passed across all
# 4 runners. New live-impacting risk keys (added to CryptoRiskConfig) must
# be wired into every runner OR explicitly opted out below.
# ---------------------------------------------------------------------------

REQUIRED_RISK_KEYS = {
    "enabled",
    "max_drawdown_pct",
    "max_position_pct",
    "max_net_short_exposure",
    "min_effective_bets",
    "daily_turnover_limit",
    "static_peak",
    "eod_trailing_drawdown",
    "eod_hour_utc",
    "bar_interval_minutes",
}


def _crypto_risk_config_kwargs(tree: ast.AST) -> set[str] | None:
    """Return the set of keyword argument names passed to the first
    CryptoRiskConfig(...) call in `tree`, or None if no such call exists."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name == "CryptoRiskConfig":
            return {kw.arg for kw in node.keywords if kw.arg is not None}
    return None


@pytest.mark.parametrize("runner_path", RUNNERS, ids=lambda p: p.name)
def test_runner_risk_config_passthrough(runner_path: Path):
    src = runner_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    kwargs = _crypto_risk_config_kwargs(tree)
    assert kwargs is not None, (
        f"{runner_path.name}: no CryptoRiskConfig(...) construction found. "
        f"All live runners must build their risk manager from a CryptoRiskConfig "
        f"sourced from the YAML risk: block."
    )
    missing = REQUIRED_RISK_KEYS - kwargs
    assert not missing, (
        f"{runner_path.name}: missing pass-through for risk keys {sorted(missing)}. "
        f"Add `<key>=risk_cfg.get('<key>', <default>)` to the CryptoRiskConfig(...) "
        f"call. Without it, YAML overrides are silently ignored — see "
        f"`project_runner_risk_config_passthrough.md` for the sg1-btc / gmgp1-btc "
        f"incident on 2026-04-26."
    )
