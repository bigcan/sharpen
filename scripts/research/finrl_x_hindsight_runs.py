"""Run the pre-registered FinRL-X hindsight-universe test.

Pre-registration: docs/research/finrl_x_hindsight_universe_2026-09-25.md (frozen; sha256 in
results/finrl_x/hindsight/prereg.sha256). Arms come from results/finrl_x/hindsight/pool.json
(scripts/research/finrl_x_pit_universe.py): ``mag7`` (published group), ``pit_top7``, ``rand_00``..``rand_34``.
Without pool.json (a fresh clone: results/ is not in the repo) the arms are regenerated from the published pool
and seed, and checked against the published draw fingerprint.

Stages (``--stage``, default ``all``):
  configs  FinRL-X's v1.2.1 config per arm with ONLY the growth-group symbols and the output paths changed
           (a structural diff asserts it)
  data     one dividend-adjusted Yahoo snapshot for every arm: deploy.sh's own downloader, extracted, with only
           auto_adjust flipped (directory created first -- deploy.sh would otherwise refill it price-only)
  run      FinRL-X's runner with the arguments deploy.sh passes, ``--jobs`` at a time; finished arms are skipped
  score    scripts/research/finrl_x_rotation_repro.py per arm (parity vs FinRL-X's printed table, cost grid, break-even)
  verdict  the pre-registered decision rules -> verdict.json

scripts/research/finrl_x_rerun.py drives the same stages from a fresh clone.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling research modules

from finrl_x_pit_universe import AS_OF, GROUP_SIZE, MAG7, N_DRAWS, SEED, draw_groups  # noqa: E402

FINRLX_COMMIT = "4409abe925c904e570be78ebfb5e77ac3491dff8"
RUNNER = "src/strategies/run_adaptive_rotation_strategy.py"
OUT = ROOT / "results" / "finrl_x" / "hindsight"
START, END, FREQ = "2018-01-07", "2025-10-24", "W-FRI"
GROUP_KEY = ("asset_groups", "group_a_growth_tech", "symbols")
PATH_KEYS = ("output_root", "state_dir", "audit_dir", "weights_dir")
PRIMARY_BPS, PAPER_BPS = "2", "10"
PCTL = 90
# The 2017-12-29 pool in market-cap rank order, as published in the hindsight doc (finrl_x_pit_universe.py
# built it from S&P 500 membership files and SEC share counts that are not in the repo).
POOL = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "V", "HD", "INTC", "ORCL", "CMCSA",
        "CSCO", "DIS", "MA", "IBM", "MCD", "NVDA", "AVGO", "NKE", "TXN", "ACN"]
DRAW_SPEC_SHA256 = "ebe2c118130ad1372b0a122dfe9828eb77877dc776180794552bf77215859d8f"

logger = logging.getLogger("finrl_x_hindsight_runs")


def venv_python(clone: Path) -> Path:
    """The clone's own interpreter: ``.venv/Scripts/python.exe`` (Windows) or ``.venv/bin/python``."""
    win, posix = clone / ".venv" / "Scripts" / "python.exe", clone / ".venv" / "bin" / "python"
    if win.exists() != posix.exists():
        return win if win.exists() else posix
    return win if os.name == "nt" else posix


@dataclass(frozen=True)
class Layout:
    """Where one campaign reads FinRL-X and writes its outputs."""

    clone: Path     # FinRL-Trading checkout with its own .venv
    out: Path       # download config, extracted downloader, verdict.json
    runs: Path      # <runs>/<arm>/{config.yaml, finrlx_stdout.log, output/, repro.json}
    data_dir: Path  # the shared price snapshot

    @property
    def venv_py(self) -> Path:
        return venv_python(self.clone)

    @property
    def base_cfg(self) -> Path:
        return self.clone / "src" / "strategies" / "AdaptiveRotationConf_v1.2.1.yaml"


def hindsight_layout() -> Layout:
    clone = Path(os.environ.get("FINRLX_ROOT", "C:/FinRL/FinRL-Trading"))
    return Layout(clone=clone, out=OUT, runs=OUT / "runs", data_dir=clone / "data" / "fmp_daily_adj_hindsight")


def draw_spec(pool: list[str]) -> dict:
    return {"as_of": AS_OF, "pool": sorted(pool), "k": GROUP_SIZE, "n": N_DRAWS, "seed": SEED}


def draw_spec_sha256(pool: list[str]) -> str:
    return hashlib.sha256(json.dumps(draw_spec(pool), sort_keys=True).encode()).hexdigest()


def published_arms() -> dict[str, list[str]]:
    """Every arm, regenerated from the published pool and seed; raises unless the draw fingerprint matches."""
    if draw_spec_sha256(POOL) != DRAW_SPEC_SHA256:
        raise RuntimeError("draw spec no longer hashes to the published fingerprint")
    out = {"mag7": list(MAG7), "pit_top7": POOL[:GROUP_SIZE]}
    out.update({f"rand_{i:02d}": g for i, g in enumerate(draw_groups(sorted(POOL), N_DRAWS, GROUP_SIZE, SEED))})
    return out


def recorded_arms(pool_json: Path) -> dict[str, list[str]]:
    pool = json.loads(pool_json.read_text())
    out = {"mag7": list(MAG7), "pit_top7": pool["deterministic_top7"]}
    out.update({f"rand_{i:02d}": g for i, g in enumerate(pool["random_draws"])})
    return out


def arms(pool_json: Path = OUT / "pool.json") -> dict[str, list[str]]:
    """The recorded arms when pool.json exists, else the regenerated ones."""
    return recorded_arms(pool_json) if pool_json.exists() else published_arms()


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


def stage_configs(lay: Layout, arm_map: dict[str, list[str]]) -> None:
    base = yaml.safe_load(lay.base_cfg.read_text(encoding="utf-8"))
    for arm, symbols in arm_map.items():
        run_dir = lay.runs / arm
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(arm_config(base, symbols, run_dir), sort_keys=False),
                                             encoding="utf-8")
    logger.info("configs written for %d arms", len(arm_map))


def extract_downloader(deploy_sh: str) -> str:
    """deploy.sh's first <<'PYEOF' block (its downloader) with auto_adjust flipped and nothing else."""
    m = re.search(r"<<'PYEOF'\n(.*?)\nPYEOF\n", deploy_sh, flags=re.S)
    if m is None:
        raise RuntimeError("no <<'PYEOF' downloader block in deploy.sh")
    code = m.group(1)
    if code.count("auto_adjust=False") != 1:
        raise RuntimeError("deploy.sh downloader no longer has exactly one auto_adjust=False")
    return code.replace("auto_adjust=False", "auto_adjust=True")


def stage_data(lay: Layout, arm_map: dict[str, list[str]], growth_union: list[str]) -> None:
    """One snapshot covering ``growth_union`` (every growth name any arm can hold) plus the config's other groups."""
    base = yaml.safe_load(lay.base_cfg.read_text(encoding="utf-8"))
    union = arm_config(base, sorted(growth_union), lay.runs / "_download")
    lay.out.mkdir(parents=True, exist_ok=True)
    cfg_path, dl_path = lay.out / "download_config.yaml", lay.out / "downloader_adjusted.py"
    cfg_path.write_text(yaml.safe_dump(union, sort_keys=False), encoding="utf-8")
    dl_path.write_text(extract_downloader((lay.clone / "deploy.sh").read_text(encoding="utf-8")), encoding="utf-8")
    lay.data_dir.mkdir(parents=True, exist_ok=True)  # before deploy.sh's logic can ever see it missing
    subprocess.run([str(lay.venv_py), str(dl_path), str(cfg_path), str(lay.data_dir)], cwd=lay.clone, env=_env(),
                   check=True)
    need = {s for g in arm_map.values() for s in g}
    missing = [s for s in sorted(need) if not (lay.data_dir / f"{s}_daily.csv").exists()]
    if missing:
        raise RuntimeError(f"price files missing after download: {missing}")
    logger.info("snapshot: %d files in %s", len(list(lay.data_dir.glob("*_daily.csv"))), lay.data_dir)


def _env() -> dict:
    return os.environ | {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                         "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                         "OPENBLAS_NUM_THREADS": "1"}


def weights_file(lay: Layout, arm: str) -> Path:
    return lay.runs / arm / "output" / "weights_dir" / f"ars_portfolio_weights_{START}_to_{END}.csv"


def run_arm(lay: Layout, arm: str) -> tuple[str, int | str]:
    if weights_file(lay, arm).exists():
        return arm, "cached"
    run_dir = lay.runs / arm
    cmd = [str(lay.venv_py), RUNNER, "--config", str(run_dir / "config.yaml"), "--data-dir", str(lay.data_dir),
           "--backtest", "--start", START, "--end", END, "--freq", FREQ]
    with open(run_dir / "finrlx_stdout.log", "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=lay.clone, env=_env(), stdout=fh, stderr=subprocess.STDOUT).returncode
    return arm, rc if weights_file(lay, arm).exists() else f"rc={rc}, no weights file"


def stage_run(lay: Layout, arm_map: dict[str, list[str]], jobs: int) -> None:
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for fut in as_completed([ex.submit(run_arm, lay, a) for a in arm_map]):
            arm, status = fut.result()
            logger.info("run %-9s %s", arm, status)


def stage_score(lay: Layout, arm_map: dict[str, list[str]], cost_grid: str = f"0,{PRIMARY_BPS},{PAPER_BPS}") -> None:
    for arm in arm_map:
        run_dir = lay.runs / arm
        subprocess.run([sys.executable, str(ROOT / "scripts" / "research" / "finrl_x_rotation_repro.py"),
                        "--weights", str(weights_file(lay, arm)), "--data-dir", str(lay.data_dir),
                        "--cost-grid", cost_grid, "--their-log", str(run_dir / "finrlx_stdout.log"),
                        "--out", str(run_dir / "repro.json")], check=False, capture_output=True)
    logger.info("scored %d arms", len(arm_map))


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


def load_rows(lay: Layout, arm_map: dict[str, list[str]]) -> dict[str, dict]:
    """Per-arm scored metrics; raises if any arm fails parity with FinRL-X's printed table."""
    rows = {}
    for arm in arm_map:
        r = json.loads((lay.runs / arm / "repro.json").read_text())
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
    return rows


def stage_verdict(lay: Layout, arm_map: dict[str, list[str]]) -> dict:
    rows = load_rows(lay, arm_map)

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
    (lay.out / "verdict.json").write_text(json.dumps(verdict, indent=2))
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
    lay, arm_map = hindsight_layout(), arms()
    pool_json = OUT / "pool.json"
    pool = json.loads(pool_json.read_text())["pool"] if pool_json.exists() else POOL
    stages = ("configs", "data", "run", "score", "verdict") if args.stage == "all" else (args.stage,)
    for s in stages:
        logger.info("== stage %s", s)
        if s == "configs":
            stage_configs(lay, arm_map)
        elif s == "data":
            stage_data(lay, arm_map, sorted(set(MAG7) | set(pool)))
        elif s == "run":
            stage_run(lay, arm_map, args.jobs)
        elif s == "score":
            stage_score(lay, arm_map)
        else:
            stage_verdict(lay, arm_map)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
