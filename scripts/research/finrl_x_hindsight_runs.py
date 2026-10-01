"""Run the pre-registered FinRL-X hindsight-universe test.

Pre-registration: docs/research/finrl_x_hindsight_universe_2026-09-25.md (frozen; sha256 in
results/finrl_x/hindsight/prereg.sha256). Arms come from results/finrl_x/hindsight/pool.json
(scripts/research/finrl_x_pit_universe.py): ``mag7`` (published group), ``pit_top7``, ``rand_00``..``rand_34``.

Stages (``--stage``, default ``all``):
  configs  FinRL-X's v1.2.1 config per arm with ONLY the growth-group symbols and the output paths changed
           (a structural diff asserts it)
  data     one dividend-adjusted Yahoo snapshot for every arm: deploy.sh's own downloader, extracted, with only
           auto_adjust flipped (directory created first -- deploy.sh would otherwise refill it price-only)
  run      FinRL-X's runner with the arguments deploy.sh passes, ``--jobs`` at a time; finished arms are skipped
  score    scripts/research/finrl_x_rotation_repro.py per arm (parity vs FinRL-X's printed table, cost grid, break-even)
  verdict  the pre-registered decision rules -> verdict.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CLONE = Path(os.environ.get("FINRLX_ROOT", "C:/FinRL/FinRL-Trading"))
VENV_PY = CLONE / ".venv" / "Scripts" / "python.exe"
BASE_CFG = CLONE / "src" / "strategies" / "AdaptiveRotationConf_v1.2.1.yaml"
RUNNER = "src/strategies/run_adaptive_rotation_strategy.py"
OUT = ROOT / "results" / "finrl_x" / "hindsight"
DATA_DIR = CLONE / "data" / "fmp_daily_adj_hindsight"
START, END, FREQ = "2018-01-07", "2025-10-24", "W-FRI"
GROUP_KEY = ("asset_groups", "group_a_growth_tech", "symbols")
PATH_KEYS = ("output_root", "state_dir", "audit_dir", "weights_dir")
MAG7 = ["AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL", "TSLA"]
PRIMARY_BPS, PAPER_BPS = "2", "10"
PCTL = 90

logger = logging.getLogger("finrl_x_hindsight_runs")


def arms() -> dict[str, list[str]]:
    pool = json.loads((OUT / "pool.json").read_text())
    out = {"mag7": MAG7, "pit_top7": pool["deterministic_top7"]}
    out.update({f"rand_{i:02d}": g for i, g in enumerate(pool["random_draws"])})
    return out


def flatten(d, prefix: tuple = ()) -> dict:
    if not isinstance(d, dict):
        return {prefix: d}
    out = {}
    for k, v in d.items():
        out.update(flatten(v, prefix + (k,)))
    return out


def check_config_diff(base: dict, cfg: dict) -> None:
    """Raise unless ``cfg`` differs from ``base`` only in the growth symbols and the output paths."""
    fb, fc = flatten(base), flatten(cfg)
    changed = {k for k in fb.keys() | fc.keys() if fb.get(k) != fc.get(k)}
    allowed = {GROUP_KEY} | {("paths", k) for k in PATH_KEYS}
    if not changed <= allowed:
        raise ValueError(f"config diff outside the pre-registered keys: {sorted(changed - allowed)}")


def arm_config(base: dict, symbols: list[str], run_dir: Path) -> dict:
    """Base config with only the growth symbols and output paths changed -- anything else raises."""
    cfg = copy.deepcopy(base)
    cfg["asset_groups"]["group_a_growth_tech"]["symbols"] = list(symbols)
    for k in PATH_KEYS:
        cfg["paths"][k] = (run_dir / "output" / k).as_posix()
    check_config_diff(base, cfg)
    return cfg


def stage_configs() -> None:
    base = yaml.safe_load(BASE_CFG.read_text(encoding="utf-8"))
    for arm, symbols in arms().items():
        run_dir = OUT / "runs" / arm
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(arm_config(base, symbols, run_dir), sort_keys=False),
                                             encoding="utf-8")
    logger.info("configs written for %d arms", len(arms()))


def extract_downloader(deploy_sh: str) -> str:
    """deploy.sh's first <<'PYEOF' block (its downloader) with auto_adjust flipped and nothing else."""
    m = re.search(r"<<'PYEOF'\n(.*?)\nPYEOF\n", deploy_sh, flags=re.S)
    if m is None:
        raise RuntimeError("no <<'PYEOF' downloader block in deploy.sh")
    code = m.group(1)
    if code.count("auto_adjust=False") != 1:
        raise RuntimeError("deploy.sh downloader no longer has exactly one auto_adjust=False")
    return code.replace("auto_adjust=False", "auto_adjust=True")


def stage_data() -> None:
    pool = json.loads((OUT / "pool.json").read_text())
    base = yaml.safe_load(BASE_CFG.read_text(encoding="utf-8"))
    union = arm_config(base, sorted(set(MAG7) | set(pool["pool"])), OUT / "runs" / "_download")
    cfg_path, dl_path = OUT / "download_config.yaml", OUT / "downloader_adjusted.py"
    cfg_path.write_text(yaml.safe_dump(union, sort_keys=False), encoding="utf-8")
    dl_path.write_text(extract_downloader((CLONE / "deploy.sh").read_text(encoding="utf-8")), encoding="utf-8")
    DATA_DIR.mkdir(parents=True, exist_ok=True)  # before deploy.sh's logic can ever see it missing
    subprocess.run([str(VENV_PY), str(dl_path), str(cfg_path), str(DATA_DIR)], cwd=CLONE, env=_env(), check=True)
    need = {s for g in arms().values() for s in g}
    missing = [s for s in sorted(need) if not (DATA_DIR / f"{s}_daily.csv").exists()]
    if missing:
        raise RuntimeError(f"price files missing after download: {missing}")
    logger.info("snapshot: %d files in %s", len(list(DATA_DIR.glob("*_daily.csv"))), DATA_DIR)


def _env() -> dict:
    return os.environ | {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                         "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                         "OPENBLAS_NUM_THREADS": "1"}


def weights_file(arm: str) -> Path:
    return OUT / "runs" / arm / "output" / "weights_dir" / f"ars_portfolio_weights_{START}_to_{END}.csv"


def run_arm(arm: str) -> tuple[str, int | str]:
    if weights_file(arm).exists():
        return arm, "cached"
    run_dir = OUT / "runs" / arm
    cmd = [str(VENV_PY), RUNNER, "--config", str(run_dir / "config.yaml"), "--data-dir", str(DATA_DIR),
           "--backtest", "--start", START, "--end", END, "--freq", FREQ]
    with open(run_dir / "finrlx_stdout.log", "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=CLONE, env=_env(), stdout=fh, stderr=subprocess.STDOUT).returncode
    return arm, rc if weights_file(arm).exists() else f"rc={rc}, no weights file"


def stage_run(jobs: int) -> None:
    todo = list(arms())
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for fut in as_completed([ex.submit(run_arm, a) for a in todo]):
            arm, status = fut.result()
            logger.info("run %-9s %s", arm, status)


def stage_score() -> None:
    for arm in arms():
        run_dir = OUT / "runs" / arm
        subprocess.run([sys.executable, str(ROOT / "scripts" / "research" / "finrl_x_rotation_repro.py"),
                        "--weights", str(weights_file(arm)), "--data-dir", str(DATA_DIR),
                        "--cost-grid", f"0,{PRIMARY_BPS},{PAPER_BPS}", "--their-log", str(run_dir / "finrlx_stdout.log"),
                        "--out", str(run_dir / "repro.json")], check=False, capture_output=True)
    logger.info("scored %d arms", len(arms()))


def decide(mag7: float, pit: float, rand: list[float]) -> dict:
    """Pre-registered rule on one excess-over-QQQ metric (see the frozen pre-registration)."""
    rand = np.asarray(rand, dtype=float)
    p90, med = float(np.percentile(rand, PCTL)), float(np.median(rand))
    if mag7 > p90 and med <= 0:
        call = "HINDSIGHT"
    elif med > 0 and pit > 0:
        call = "ROBUST"
    else:
        call = "INCONCLUSIVE"
    return {"call": call, "mag7": mag7, "pit_top7": pit, "rand_median": med, f"rand_p{PCTL}": p90,
            "rand_min": float(rand.min()), "rand_max": float(rand.max()), "rand_share_above_0": float((rand > 0).mean()),
            "mag7_share_of_rand_below": float((rand < mag7).mean()), "n_rand": int(rand.size)}


def stage_verdict() -> dict:
    rows = {}
    for arm in arms():
        r = json.loads((OUT / "runs" / arm / "repro.json").read_text())
        q = r["benchmarks"]["QQQ"]
        row = {"parity_ok": not r["parity_problems"], "breakeven_vs_QQQ_bps": r["breakeven_vs_QQQ_bps"]}
        for bps in (PRIMARY_BPS, PAPER_BPS):
            m = r["cost_grid"][bps]
            row[f"cagr_{bps}"], row[f"sharpe_{bps}"] = m["ann_return"], m["sharpe_arith"]
            row[f"d_cagr_{bps}"] = m["ann_return"] - q["ann_return"]
            row[f"d_sharpe_{bps}"] = m["sharpe_arith"] - q["sharpe_arith"]
        row["cagr_0"], row["qqq_cagr"], row["qqq_sharpe"] = r["cost_grid"]["0"]["ann_return"], q["ann_return"], q["sharpe_arith"]
        rows[arm] = row
    if not all(r["parity_ok"] for r in rows.values()):
        raise RuntimeError(f"parity failed: {[a for a, r in rows.items() if not r['parity_ok']]}")

    def rule(metric: str) -> dict:
        return decide(rows["mag7"][metric], rows["pit_top7"][metric],
                      [r[metric] for a, r in rows.items() if a.startswith("rand_")])

    cagr, sharpe = rule(f"d_cagr_{PRIMARY_BPS}"), rule(f"d_sharpe_{PRIMARY_BPS}")
    verdict = {
        "primary": cagr, "sharpe_reading": sharpe,
        "paper_cost_reading": rule(f"d_cagr_{PAPER_BPS}"),
        "verdict": cagr["call"] if cagr["call"] == sharpe["call"] else "INCONCLUSIVE",
        "rows": rows,
    }
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2))
    logger.info("primary (dCAGR vs QQQ @%s bps): %s", PRIMARY_BPS, cagr)
    logger.info("sharpe reading: %s", sharpe)
    logger.info("VERDICT: %s", verdict["verdict"])
    return verdict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=("configs", "data", "run", "score", "verdict", "all"), default="all")
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    stages = ("configs", "data", "run", "score", "verdict") if args.stage == "all" else (args.stage,)
    for s in stages:
        logger.info("== stage %s", s)
        if s == "configs":
            stage_configs()
        elif s == "data":
            stage_data()
        elif s == "run":
            stage_run(args.jobs)
        elif s == "score":
            stage_score()
        else:
            stage_verdict()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
