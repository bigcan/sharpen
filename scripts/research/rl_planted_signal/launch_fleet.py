"""Fleet entry point for the planted-signal ladder: one process per GPU slot, launched through the
standard ``scripts/deploy_bare_metal.py --script`` path (which passes ``--config`` plus deploy flags).

The config (``configs/rl_planted_signal/<slot>.yaml``) lists the synthetic markets and the arms for
this GPU. The launcher generates any missing market (deterministic: ``gen_synth.make(rho, n, seed)``),
runs every arm concurrently as ``run_arm.py`` subprocesses on the visible GPU, writes one log per
arm plus ``status.json``, and exits when all arms finish. Deploy flags it does not use
(``--run_name``, ``--hpo_storage``, ``--tags``) are accepted and recorded.
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

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from gen_synth import make  # noqa: E402

log = logging.getLogger("rl_planted_signal.launch_fleet")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_name", default=None)
    args, unknown = ap.parse_known_args()
    cfg = yaml.safe_load((REPO / args.config).read_text(encoding="utf-8"))

    out_root = REPO / cfg["out_root"]
    data_dir = out_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, rho in cfg["datasets"].items():
        pq = data_dir / f"{name}.parquet"
        if not pq.exists():
            tmp = pq.with_suffix(".tmp.parquet")
            make(float(rho), int(cfg["n_base_bars"]), int(cfg["data_seed"])).to_parquet(tmp, index=False)
            tmp.replace(pq)
            log.info("generated %s (rho=%s)", pq, rho)

    env = dict(os.environ, WANDB_MODE="disabled", WANDB_SILENT="true", PYTHONUNBUFFERED="1")
    threads = str(cfg.get("threads_per_arm", 8))
    env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads)
    procs = {}
    for arm in cfg["arms"]:
        cmd = [sys.executable, "-u", str(HERE / "run_arm.py"), arm["name"],
               str(data_dir / f"{arm['data']}.parquet"), str(out_root)]
        cmd += [f"{k}={v}" for k, v in (arm.get("overrides") or {}).items()]
        logf = open(out_root / f"{arm['name']}.log", "w", encoding="utf-8")
        procs[arm["name"]] = (subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=logf,
                                               stderr=subprocess.STDOUT), logf)
        log.info("launched %s pid=%d: %s", arm["name"], procs[arm["name"]][0].pid, " ".join(cmd[3:]))
        time.sleep(float(cfg.get("stagger_seconds", 30)))

    status = {"config": args.config, "run_name": args.run_name, "deploy_flags": unknown,
              "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "arms": {}}
    while True:
        done = True
        for name, (p, _f) in procs.items():
            rc = p.poll()
            status["arms"][name] = "running" if rc is None else f"exit {rc}"
            done &= rc is not None
        (out_root / "status.json").write_text(json.dumps(status, indent=2))
        if done:
            break
        time.sleep(60)
    for _p, f in procs.values():
        f.close()
    log.info("all arms finished: %s", status["arms"])
    return 0 if all(v == "exit 0" for v in status["arms"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
