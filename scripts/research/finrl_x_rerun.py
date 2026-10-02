"""Rerun the FinRL-X audit from a fresh clone: FinRL-X's own code, free Yahoo data, CPU only, no keys.

    python scripts/research/finrl_x_rerun.py            # 3 growth lists, ~10 min on 3+ cores (~30 min on 1)
    python scripts/research/finrl_x_rerun.py --full     # all 37 lists + the pre-registered verdict (~6 CPU-hours)

What it does (study overview: studies/finrl-x/README.md):
  1. clones AI4Finance's FinRL-Trading at commit 4409abe9 into .cache/finrl-x/FinRL-Trading (or $FINRLX_ROOT) with
     its own virtual environment. None of its files are modified.
  2. regenerates the growth lists from the published 2017-12-29 pool and seed, and checks the draw fingerprint.
  3. downloads one dividend-adjusted Yahoo snapshot with FinRL-X's own downloader (only ``auto_adjust`` flipped).
  4. runs FinRL-X's Adaptive Rotation backtest once per list, with only the growth symbols and the output paths
     changed in its config.
  5. re-prices each run with trading costs (finrl_x_rotation_repro.py). Each run must first match the table
     FinRL-X itself printed, else this exits non-zero.

Quick lists: ``mag7`` (as published), ``pit_top7`` (the 7 largest on 2017-12-29), ``rand_02`` (the best of the 35
random lists). Outputs: results/finrl_x/rerun/<list>/. Finished lists are skipped on a second run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling research modules

import finrl_x_hindsight_runs as hr  # noqa: E402

REPO_URL = "https://github.com/AI4Finance-Foundation/FinRL-Trading"
DEFAULT_CLONE = ROOT / ".cache" / "finrl-x" / "FinRL-Trading"
OUT = ROOT / "results" / "finrl_x" / "rerun"
# what the Adaptive Rotation runner and the downloader import (FinRL-X's requirements.txt does not install cleanly)
PACKAGES = ("pyyaml", "pandas", "numpy", "scipy", "yfinance", "pandas-market-calendars", "python-dotenv", "matplotlib",
            "pydantic")
IMPORTS = "import yaml, pandas, numpy, scipy, yfinance, pandas_market_calendars, dotenv, matplotlib, pydantic"
QUICK_ARMS = ("mag7", "pit_top7", "rand_02")
COST_GRID = "0,2,3,5,10"
LABELS = {"mag7": "Magnificent 7 (as published)", "pit_top7": "7 largest on 2017-12-29"}

logger = logging.getLogger("finrl_x_rerun")


def rerun_layout(clone: Path) -> hr.Layout:
    return hr.Layout(clone=clone, out=OUT, runs=OUT, data_dir=OUT / "prices")


def _git(clone: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(clone), *args], check=True, capture_output=True, text=True).stdout.strip()


def ensure_clone(clone: Path) -> None:
    """FinRL-Trading at the audited commit; an existing checkout already there is left as it is."""
    if not (clone / ".git").exists():
        clone.parent.mkdir(parents=True, exist_ok=True)
        logger.info("cloning %s -> %s", REPO_URL, clone)
        subprocess.run(["git", "clone", "--quiet", REPO_URL, str(clone)], check=True)
    if _git(clone, "rev-parse", "HEAD") != hr.FINRLX_COMMIT:
        subprocess.run(["git", "-C", str(clone), "checkout", "--quiet", "--detach", hr.FINRLX_COMMIT], check=True)
    head = _git(clone, "rev-parse", "HEAD")
    if head != hr.FINRLX_COMMIT:
        raise RuntimeError(f"{clone} is at {head}, not {hr.FINRLX_COMMIT}")


def ensure_venv(clone: Path) -> None:
    """The clone's own virtual environment with only what the Adaptive Rotation needs."""
    if not hr.venv_python(clone).exists():
        logger.info("creating %s", clone / ".venv")
        subprocess.run([sys.executable, "-m", "venv", str(clone / ".venv")], check=True)
    py = str(hr.venv_python(clone))
    if subprocess.run([py, "-c", IMPORTS], capture_output=True).returncode != 0:
        logger.info("installing into FinRL-X's environment: %s", " ".join(PACKAGES))
        subprocess.run([py, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *PACKAGES], check=True)


def check_recorded_pool(arm_map: dict[str, list[str]], pool_json: Path) -> None:
    """On a machine that holds the original pool.json, the regenerated lists must equal the recorded ones."""
    if pool_json.exists() and hr.recorded_arms(pool_json) != arm_map:
        raise RuntimeError(f"regenerated lists disagree with {pool_json}")


def select_arms(full: bool) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """(the lists to run, all 37 lists)."""
    every = hr.published_arms()
    if every["rand_02"] != ["ACN", "AMZN", "AVGO", "GOOGL", "MCD", "NVDA", "ORCL"]:
        raise RuntimeError(f"rand_02 regenerated as {every['rand_02']}")
    check_recorded_pool(every, hr.OUT / "pool.json")
    return (every if full else {a: every[a] for a in QUICK_ARMS}), every


def ensure_data(lay: hr.Layout, every: dict[str, list[str]]) -> None:
    """One snapshot for the union of all 37 lists, so a later --full reuses the quick run's prices."""
    done = lay.data_dir / ".complete"
    if done.exists():
        logger.info("prices: reusing %s", lay.data_dir)
        return
    logger.info("downloading dividend-adjusted Yahoo prices with FinRL-X's downloader")
    hr.stage_data(lay, every, sorted({s for g in every.values() for s in g}))
    done.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), encoding="utf-8")


def load_reports(lay: hr.Layout, arm_map: dict[str, list[str]]) -> dict[str, dict]:
    reports = {}
    for arm in arm_map:
        path = lay.runs / arm / "repro.json"
        if not path.exists():
            raise RuntimeError(f"{arm}: no scored result; see {lay.runs / arm / 'finrlx_stdout.log'}")
        reports[arm] = json.loads(path.read_text())
    return reports


def _pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def _row(label: str, r: dict) -> str:
    grid, qqq = r["cost_grid"], r["benchmarks"]["QQQ"]["ann_return"]
    per_year = " / ".join(_pct(grid[b]["ann_return"]) for b in ("0", hr.PRIMARY_BPS, hr.PAPER_BPS))
    parity = "PASS" if not r["parity_problems"] else "FAIL"
    return (f"{label:<30}{grid['0']['cumulative_x']:>6.2f}x   {per_year:<27}"
            f"{100 * (grid[hr.PRIMARY_BPS]['ann_return'] - qqq):>+7.2f} pts   {parity}")


def render(reports: dict[str, dict], arm_map: dict[str, list[str]]) -> list[str]:
    """The printed table; ``reports`` maps arm -> finrl_x_rotation_repro.py output."""
    first = next(iter(reports.values()))
    lines = [f"FinRL-X Adaptive Rotation @ {hr.FINRLX_COMMIT[:8]}, {hr.START} -> {hr.END}, "
             "dividend-adjusted Yahoo closes, costs per side",
             "",
             f"{'Growth list':<30}{'Growth':>7}   {'Per year 0 / 2 / 10 bps':<27}{'vs QQQ, 2 bps':>11}   Parity"]
    for arm in arm_map:
        label = LABELS.get(arm, f"Best random list ({arm})" if len(arm_map) == len(QUICK_ARMS) else arm)
        lines.append(_row(label, reports[arm]))
    for b, m in first["benchmarks"].items():
        lines.append(f"{b + ', buy and hold':<30}{m['cumulative_x']:>6.2f}x   {_pct(m['ann_return'])}")
    mag7 = reports.get("mag7")
    if mag7:
        be = mag7["breakeven_vs_QQQ_bps"]
        grid = "  ".join(f"{b} bps {_pct(m['ann_return'])} / {m['sharpe_arith']:.2f}" for b, m in mag7["cost_grid"].items())
        lines += ["",
                  f"Magnificent 7: turnover {mag7['strategy_0bps']['annual_turnover']:.1f}x a year (buys and sells); "
                  f"break-even vs QQQ {be['ann_return']:.2f} bps per side on return, {be['sharpe_arith']:.2f} on Sharpe",
                  f"Magnificent 7 cost grid (per year / Sharpe): {grid}  "
                  f"QQQ {_pct(first['benchmarks']['QQQ']['ann_return'])} / {first['benchmarks']['QQQ']['sharpe_arith']:.2f}"]
    return lines


def render_verdict(verdict: dict) -> list[str]:
    """The --full summary: the random lists against QQQ and the pre-registered call."""
    key = f"d_cagr_{hr.PRIMARY_BPS}"
    rand = {a: r[key] for a, r in verdict["rows"].items() if a.startswith("rand_")}
    best, worst = max(rand, key=rand.get), min(rand, key=rand.get)
    return ["",
            f"Random lists beating QQQ at {hr.PRIMARY_BPS} bps: {sum(v > 0 for v in rand.values())} of {len(rand)}",
            f"Random lists vs QQQ at {hr.PRIMARY_BPS} bps, points a year: median {100 * np.median(list(rand.values())):+.2f}"
            f", best {best} {100 * rand[best]:+.2f}, worst {worst} {100 * rand[worst]:+.2f}",
            f"Magnificent 7 beats {sum(v < verdict['rows']['mag7'][key] for v in rand.values())} of {len(rand)} random lists",
            f"Verdict (rules frozen before the first run): {verdict['verdict']} "
            f"(return rule {verdict['primary']['call']}, Sharpe rule {verdict['sharpe_reading']['call']}, "
            f"at {hr.PAPER_BPS} bps {verdict['paper_cost_reading']['call']})"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true", help="all 37 growth lists and the pre-registered verdict")
    ap.add_argument("--jobs", type=int, help="backtests in parallel (default: one per list, up to the CPU count)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t0 = time.time()

    arm_map, every = select_arms(args.full)
    clone = Path(os.environ.get("FINRLX_ROOT", DEFAULT_CLONE)).resolve()
    lay = rerun_layout(clone)
    ensure_clone(clone)
    ensure_venv(clone)
    hr.stage_configs(lay, arm_map)
    ensure_data(lay, every)
    jobs = args.jobs or min(len(arm_map), os.cpu_count() or 1)
    todo = [a for a in arm_map if not hr.weights_file(lay, a).exists()]
    logger.info("running %d FinRL-X backtests, %d at a time (about 10 minutes each on one core)", len(todo), jobs)
    hr.stage_run(lay, arm_map, jobs)
    hr.stage_score(lay, arm_map, COST_GRID)

    reports = load_reports(lay, arm_map)
    lines = render(reports, arm_map)
    failed = [a for a, r in reports.items() if r["parity_problems"]]
    if not failed and args.full:
        lines += render_verdict(hr.stage_verdict(lay, arm_map))
    for line in lines:
        logger.info(line)
    logger.info("")
    logger.info("wall time %.1f min; outputs in %s", (time.time() - t0) / 60, OUT)
    if failed:
        logger.error("parity vs FinRL-X's printed table FAILED: %s", failed)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
