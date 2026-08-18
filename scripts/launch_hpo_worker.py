"""Start ONE extra distributed-HPO worker on an already-provisioned instance.

WHY THIS EXISTS ALONGSIDE deploy_bare_metal.py. `deploy_bare_metal.py` re-zips, re-uploads
and re-installs the whole project on every invocation. That is correct for the FIRST worker
on an instance, but for the 2nd worker on the SAME instance it re-extracts the project
directory underneath a run that is already training, and it costs several minutes per
worker. This script assumes the instance is already set up (verify with the caller) and only
launches the process, mirroring deploy_bare_metal.py's launch environment exactly:
PATH, CUDA_VISIBLE_DEVICES, WANDB_API_KEY, ulimit, nohup, unbuffered python.

It also exists because wrapping deploy_bare_metal.py in a shell for-loop did NOT survive:
the launcher's python child was reaped mid-deploy, leaving SETUP_SUCCESS in the log and
nothing running on the GPU.

Usage:
    python scripts/launch_hpo_worker.py --instance gpuhub-1 --gpu 1 \
        --config configs/gmgp1_spx500_ls_hpo.yaml --trials 10 --tag spx500-ls-hpo-w1
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path

import paramiko
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/workspace/DeepScalper"


def main() -> int:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--trials", type=int, required=True)
    ap.add_argument("--tag", required=True, help="descriptive run-name prefix")
    ap.add_argument("--stage", default="hpo")
    args = ap.parse_args()

    # encoding is load-bearing on Windows: the default cp950 codec dies on the em-dash in
    # instances.json's note field ("2x RTX 4090 — formerly gpuhub-2").
    reg = json.loads((ROOT / "instances.json").read_text(encoding="utf-8"))
    inst = reg["instances"][args.instance]
    db_url = os.environ["DISTRIBUTED_HPO_DB_URL"]
    wandb_key = os.environ.get("WANDB_API_KEY", "")

    run_name = f"{args.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_file = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}.log"

    # NOTE ON --hpo_storage ORDERING: deploy_bare_metal.py injects a per-run sqlite
    # --hpo_storage of its own; here we pass Postgres directly and there is no competing
    # flag. Every worker for a variant must land on the SAME study_name (set in the config)
    # for the shared search to actually be shared.
    inner = (
        f"python -u scripts/run_full_pipeline.py"
        f" --config {shlex.quote(args.config)}"
        f" --agent sac"
        f" --stage {args.stage}"
        f" --trials {args.trials}"
        f" --run_name {shlex.quote(run_name)}"
        f" --hpo_storage {shlex.quote(db_url)}"
    )
    cmd = (
        f"cd {REMOTE} && export PATH=/root/miniconda3/bin:$PATH"
        f" && export CUDA_VISIBLE_DEVICES={args.gpu}"
        + (f" && export WANDB_API_KEY={shlex.quote(wandb_key)}" if wandb_key else "")
        + f" && (ulimit -n 65535 || true)"
        f" && nohup {inner} > {log_file} 2>&1 & echo LAUNCHED"
    )

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(inst["host"], port=int(inst["port"]), username="root",
                password=inst["password"], timeout=30)
    _in, out, err = ssh.exec_command(cmd)
    print(out.read().decode().strip() or "(no stdout)")
    e = err.read().decode().strip()
    if e:
        print("STDERR:", e[:400])
    print(f"instance={args.instance} gpu={args.gpu} run_name={run_name} log={log_file}")
    ssh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
