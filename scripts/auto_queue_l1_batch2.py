#!/usr/bin/env python
"""Auto-queue batch 2 of SG-1 XAUUSD L1 multiseed after batch 1 finishes.

Polls gpuhub-2 every POLL_INTERVAL seconds. When all batch-1 seeds (1337,
2025, 3141, 5150, 9999) have exited, fires batch 2 (seeds 42, 123, 456, 789,
1024) via SSH fan-out with staggered launches and isolated torch.compile
caches per seed (defensive against S488 cache race).

Run in background — this is a long poller:
    python scripts/auto_queue_l1_batch2.py \
        2>&1 | tee C:/tmp/auto_queue_l1_batch2.log
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] auto_queue - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def ssh(remote_cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*SSH_PREFIX, remote_cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def count_batch1_running() -> int:
    """Count batch-1 l1-multiseed python processes still alive on remote."""
    try:
        res = ssh(
            "pgrep -af 'run_full_pipeline.*sg1-xauusd-l1-multiseed-rehpo-seed' "
            "| grep -v 'batch2' | wc -l"
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
    # Per-seed isolated torch.compile cache + 2s stagger — defensive against
    # the S488 multiprocessing/cache race that killed 5 of 10 on 10 concurrent.
    remote_cmd = (
        "cd /workspace/DeepScalper && "
        "set -a && source .env && set +a && "
        "TS=$(date +%Y%m%d_%H%M%S) && "
        f"for SEED in {seeds_str}; do "
        "TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_seed$SEED "
        "CUDA_VISIBLE_DEVICES=0 "
        "nohup /root/miniconda3/bin/python -u scripts/run_full_pipeline.py "
        "--config configs/sg1_xauusd_ftmo_rehpo_l1_multiseed.yaml "
        "--run_name sg1-xauusd-l1-multiseed-rehpo-batch2-seed${SEED}_${TS} "
        "--seed $SEED "
        "> run_l1_batch2_seed${SEED}.log 2>&1 & "
        "echo \"launched seed $SEED pid $!\"; "
        "sleep 2; "
        "done; "
        "echo ===; "
        "pgrep -af run_full_pipeline.*batch2 | head"
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
    log.info("auto-queue batch2: polling every %ds (max %dh)",
             POLL_INTERVAL, MAX_POLL_HOURS)
    log.info("batch1 seeds: %s", BATCH1_SEEDS)
    log.info("batch2 seeds: %s", BATCH2_SEEDS)

    while True:
        elapsed = time.time() - start
        if elapsed > max_seconds:
            log.error("poll ceiling %dh exceeded — aborting without launching",
                      MAX_POLL_HOURS)
            return 2

        n = count_batch1_running()
        if n == 0:
            log.info("batch 1 complete — firing batch 2")
            break
        if n == -1:
            log.warning("poll error; retrying in %ds", POLL_INTERVAL)
        else:
            log.info("batch 1 alive: %d/5 (elapsed %.1fm)",
                     n, elapsed / 60.0)
        time.sleep(POLL_INTERVAL)

    rc = launch_batch2()
    log.info("batch 2 launch returncode=%d", rc)
    if rc == 0:
        log.info("batch 2 fired. Monitor via: ssh gpuhub-2 'ls -la "
                 "/workspace/DeepScalper/run_l1_batch2_seed*.log'")
    return rc


if __name__ == "__main__":
    sys.exit(main())
