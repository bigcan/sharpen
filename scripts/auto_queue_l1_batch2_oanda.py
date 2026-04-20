#!/usr/bin/env python
"""Auto-queue batch 2 of SG-1 XAUUSD L1 multiseed OANDA redo (S488 Path A).

Same pattern as `auto_queue_l1_batch2.py` but targets the OANDA-data config
`configs/sg1_xauusd_ftmo_rehpo_l1_multiseed_oanda.yaml` and uses log-file
prefix `run_l1_oanda_batch2_seed*.log`.

Polls gpuhub-2 every POLL_INTERVAL. When all batch-1 seeds (1337, 2025,
3141, 5150, 9999) have exited their `-batch1-` processes, fires batch 2
(42, 123, 456, 789, 1024).

Run in background:
    python scripts/auto_queue_l1_batch2_oanda.py \
        2>&1 | tee C:/tmp/auto_queue_l1_batch2_oanda.log
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time

SSH_PREFIX = ("ssh", "-p", "<SSH_PORT>", "-o", "StrictHostKeyChecking=no",
              "root@<GPU_HOST>")

BATCH1_SEEDS = [1337, 2025, 3141, 5150, 9999]
BATCH2_SEEDS = [42, 123, 456, 789, 1024]
POLL_INTERVAL = 300  # 5 min
MAX_POLL_HOURS = 6

CONFIG_PATH = "configs/sg1_xauusd_ftmo_rehpo_l1_multiseed_oanda.yaml"
RUN_NAME_PREFIX_BATCH1 = "sg1-xauusd-oanda-l1-rehpo-batch1"
RUN_NAME_PREFIX_BATCH2 = "sg1-xauusd-oanda-l1-rehpo-batch2"
LOG_PREFIX_BATCH2 = "run_l1_oanda_batch2_seed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] auto_queue_oanda - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def ssh(remote_cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*SSH_PREFIX, remote_cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def count_batch1_running() -> int:
    """Count batch-1 OANDA l1-multiseed python processes still alive."""
    try:
        res = ssh(
            f"pgrep -af 'run_full_pipeline.*{RUN_NAME_PREFIX_BATCH1}' | wc -l"
        )
    except subprocess.TimeoutExpired:
        log.warning("ssh poll timed out — treating as unknown")
        return -1
    if res.returncode != 0:
        log.warning("poll ssh exit=%d stderr=%s", res.returncode, res.stderr[:200])
        return -1
    try:
        return int(res.stdout.strip())
    except ValueError:
        log.warning("unparseable count: %r", res.stdout)
        return -1


def launch_batch2() -> int:
    seeds_str = " ".join(str(s) for s in BATCH2_SEEDS)
    remote_cmd = (
        "cd /workspace/DeepScalper && "
        "set -a && source .env && set +a && "
        "TS=$(date +%Y%m%d_%H%M%S) && "
        f"for SEED in {seeds_str}; do "
        "TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_oanda_seed$SEED "
        "CUDA_VISIBLE_DEVICES=0 "
        "nohup /root/miniconda3/bin/python -u scripts/run_full_pipeline.py "
        f"--config {CONFIG_PATH} "
        f"--run_name {RUN_NAME_PREFIX_BATCH2}-seed${{SEED}}_${{TS}} "
        "--seed $SEED "
        f"> {LOG_PREFIX_BATCH2}${{SEED}}.log 2>&1 & "
        "echo \"launched seed $SEED pid $!\"; "
        "sleep 2; "
        "done; "
        "echo ===; "
        f"pgrep -af run_full_pipeline.*{RUN_NAME_PREFIX_BATCH2} | head"
    )
    log.info("Launching batch 2: seeds %s", BATCH2_SEEDS)
    try:
        res = ssh(remote_cmd, timeout=180)
    except subprocess.TimeoutExpired:
        log.error("launch ssh timed out")
        return 124
    log.info("stdout:\n%s", res.stdout)
    if res.returncode != 0:
        log.error("launch FAILED exit=%d stderr:\n%s", res.returncode, res.stderr)
    return res.returncode


def main() -> int:
    start = time.time()
    max_seconds = MAX_POLL_HOURS * 3600
    log.info("auto-queue batch2 (OANDA): polling every %ds (max %dh)",
             POLL_INTERVAL, MAX_POLL_HOURS)
    log.info("config: %s", CONFIG_PATH)
    log.info("batch1 seeds: %s (prefix %s)", BATCH1_SEEDS, RUN_NAME_PREFIX_BATCH1)
    log.info("batch2 seeds: %s (prefix %s)", BATCH2_SEEDS, RUN_NAME_PREFIX_BATCH2)

    while True:
        elapsed = time.time() - start
        if elapsed > max_seconds:
            log.error("poll ceiling %dh exceeded — aborting without launching",
                      MAX_POLL_HOURS)
            return 2

        n = count_batch1_running()
        # pgrep -af matches its own subshell's argv (which contains the pattern),
        # so when all training exits, n=1 (just self-match), not 0. Stop at n<=1.
        alive = max(0, n - 1) if n >= 0 else -1
        if n >= 0 and alive == 0:
            log.info("batch 1 complete — firing batch 2")
            break
        if n == -1:
            log.warning("poll error; retrying in %ds", POLL_INTERVAL)
        else:
            log.info("batch 1 alive: %d/5 (elapsed %.1fm)", alive, elapsed / 60.0)
        time.sleep(POLL_INTERVAL)

    rc = launch_batch2()
    log.info("batch 2 launch returncode=%d", rc)
    if rc == 0:
        log.info("batch 2 fired. Monitor via: ssh gpuhub-2 'ls -la "
                 f"/workspace/DeepScalper/{LOG_PREFIX_BATCH2}*.log'")
    return rc


if __name__ == "__main__":
    sys.exit(main())
