"""Queue funding-arb DSAC L1 multiseed launch after the SG-1 XAUUSD leverage
narrow re-HPO V2 finishes on gpuhub-2.

Polls the Optuna study `sg1_xauusd_leverage_narrow_hpo_v2_20260427_071420`
every 5 minutes. When all trials are COMPLETE/FAIL (no RUNNING), verifies
gpuhub-2:0 is GPU-idle (last-mile sanity), then execs `launch_l1_multiseed.py`
on gpuhub-2:0 with the funding-arb runner.

Usage:
    python scripts/queue_funding_arb_after_leverage.py

    # Override the upstream study name (e.g. for re-runs):
    python scripts/queue_funding_arb_after_leverage.py \\
        --upstream_study other_study_name

    # Dry-run (poll once + report state, no launch):
    python scripts/queue_funding_arb_after_leverage.py --dry_run

Environment:
    DISTRIBUTED_HPO_DB_URL must be set (Postgres URL — read from .env).
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] queue: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("queue")


DEFAULT_UPSTREAM_STUDY = "sg1_xauusd_leverage_narrow_hpo_v2_20260427_071420"
DEFAULT_TARGET_INSTANCE = "gpuhub-2"
DEFAULT_TARGET_GPU = "0"
DEFAULT_POLL_INTERVAL = 300  # 5 minutes — leverage HPO trials are ~1 hr each


def _load_env_db_url() -> str:
    """Read DISTRIBUTED_HPO_DB_URL from env or .env file."""
    url = os.environ.get("DISTRIBUTED_HPO_DB_URL")
    if url:
        return url
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("DISTRIBUTED_HPO_DB_URL="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("DISTRIBUTED_HPO_DB_URL not set and not in .env")


def check_upstream_complete(study_name: str, db_url: str) -> dict:
    """Return upstream HPO study status."""
    import optuna
    study = optuna.load_study(study_name=study_name, storage=db_url)
    states = {"complete": 0, "running": 0, "failed": 0, "waiting": 0, "pruned": 0}
    for t in study.trials:
        s = t.state.name.lower()
        states[s] = states.get(s, 0) + 1
    total = len(study.trials)
    is_done = states["running"] == 0 and states["waiting"] == 0 and total > 0
    return {
        "total": total,
        "states": states,
        "complete": is_done,
    }


def check_gpu_idle(instance_name: str, gpu_idx: int = 0) -> bool:
    """SSH to instance and check the GPU is idle (<5% util, <500 MiB used)."""
    import paramiko
    instances = json.loads(
        (PROJECT_ROOT / "instances.json").read_text(encoding="utf-8"),
    )["instances"]
    inst = instances[instance_name]
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(inst["host"], port=inst["port"], username="root",
                password=inst["password"], timeout=30)
    cmd = (
        f"nvidia-smi --query-gpu=utilization.gpu,memory.used "
        f"--format=csv,noheader,nounits -i {gpu_idx}"
    )
    _, out, _ = ssh.exec_command(cmd, timeout=15)
    line = out.read().decode().strip()
    ssh.close()
    try:
        util_str, mem_str = line.split(",")
        util = int(util_str.strip())
        mem_used = int(mem_str.strip())
        idle = util < 5 and mem_used < 500
        logger.info("  gpu check %s:%d util=%d%% mem_used=%dMiB idle=%s",
                    instance_name, gpu_idx, util, mem_used, idle)
        return idle
    except (ValueError, IndexError) as e:
        logger.warning("  failed to parse nvidia-smi output %r: %s", line, e)
        return False


def launch_funding_arb_l1(
    config: str,
    instance: str,
    gpu: str,
    seeds: str,
    run_name_prefix: str,
    script: str,
    log_path: Path,
) -> int:
    """Spawn launch_l1_multiseed in the background; return PID."""
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "launch_l1_multiseed.py"),
        "--config", config,
        "--instance", instance,
        "--gpu", gpu,
        "--seeds", seeds,
        "--concurrent", "1",
        "--run_name_prefix", run_name_prefix,
        "--script", script,
    ]
    logger.info("Launching: %s", " ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    return proc.pid


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--upstream_study", default=DEFAULT_UPSTREAM_STUDY,
                   help=f"Optuna study to gate on (default: {DEFAULT_UPSTREAM_STUDY})")
    p.add_argument("--instance", default=DEFAULT_TARGET_INSTANCE)
    p.add_argument("--gpu", default=DEFAULT_TARGET_GPU)
    p.add_argument("--config", default="configs/funding_arb_dsac_l1_multiseed.yaml")
    p.add_argument("--seeds", default="42,123,456,789,1024,1337,2025,3141,5150,9999")
    p.add_argument("--run_name_prefix", default="funding-arb-dsac-l1-multiseed")
    p.add_argument("--script", default="scripts/funding_arb_dsac_train_seed.py")
    p.add_argument("--launcher_log", default="logs/funding_arb_l1_launcher.log")
    p.add_argument("--poll_interval", type=int, default=DEFAULT_POLL_INTERVAL)
    p.add_argument("--dry_run", action="store_true",
                   help="Poll once and report state; do not launch.")
    args = p.parse_args()

    db_url = _load_env_db_url()

    logger.info("Queue armed:")
    logger.info("  upstream study:  %s", args.upstream_study)
    logger.info("  target:          %s:%s", args.instance, args.gpu)
    logger.info("  config:          %s", args.config)
    logger.info("  seeds:           %s", args.seeds)
    logger.info("  poll interval:   %ds", args.poll_interval)

    while True:
        try:
            status = check_upstream_complete(args.upstream_study, db_url)
        except Exception as e:
            logger.warning("Optuna poll failed (transient): %s — retrying", e)
            time.sleep(60)
            continue

        logger.info("upstream %s: total=%d states=%s complete=%s",
                    args.upstream_study,
                    status["total"], status["states"], status["complete"])

        if args.dry_run:
            logger.info("--dry_run set; exiting after first poll")
            return 0

        if status["complete"]:
            logger.info("Upstream HPO complete. Verifying GPU idle...")
            if check_gpu_idle(args.instance, int(args.gpu)):
                logger.info("GPU idle. Launching funding-arb L1 multiseed...")
                pid = launch_funding_arb_l1(
                    config=args.config,
                    instance=args.instance,
                    gpu=args.gpu,
                    seeds=args.seeds,
                    run_name_prefix=args.run_name_prefix,
                    script=args.script,
                    log_path=Path(args.launcher_log),
                )
                logger.info("Launcher spawned (pid=%d). Log: %s", pid, args.launcher_log)
                logger.info("Queue script exiting cleanly.")
                return 0
            else:
                logger.info(
                    "Upstream study marked complete but GPU still busy — "
                    "leverage workers may be in graceful shutdown. Re-poll in 60s.",
                )
                time.sleep(60)
                continue

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    sys.exit(main())
