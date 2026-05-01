#!/usr/bin/env python
"""Launch GMGP1 XAUUSD L1 multiseed OANDA redo (S493-cont, Path B).

10 seeds split into two sequential batches of 5 per the gpuhub-2 concurrency
rule (`feedback_gpuhub2_max_5_concurrent_sac.md`): batch-of-5 is safe, but
spawning onto a running fleet is cascade-fatal. So fire batch 1, poll until
all 5 exit naturally, then fire batch 2.

Mirrors `scripts/auto_queue_l1_batch2_oanda.py` (SG-1 variant) but:
  - Target config: `configs/gmgp1_xauusd_ftmo_rehpo_l1_multiseed_oanda.yaml`
  - Fires both batches in one script (single PID to track)
  - Run-name prefixes: gmgp1-xauusd-oanda-l1-rehpo-batch{1,2}
  - Log prefixes:     run_gmgp1_l1_oanda_batch{1,2}_seed<SEED>.log

Run in background:
    python scripts/launch_gmgp1_xauusd_l1_oanda.py \
        2>&1 | tee C:/tmp/launch_gmgp1_xauusd_l1_oanda.log
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time

# gpuhub-2 (non-HPO), port <SSH_PORT>
SSH_PREFIX = ("ssh", "-p", "<SSH_PORT>", "-o", "StrictHostKeyChecking=no",
              "root@<GPU_HOST>")

BATCH1_SEEDS = [1337, 2025, 3141, 5150, 9999]
BATCH2_SEEDS = [42, 123, 456, 789, 1024]
POLL_INTERVAL = 300   # 5 min
MAX_POLL_HOURS = 6

CONFIG_PATH = "configs/gmgp1_xauusd_ftmo_rehpo_l1_multiseed_oanda.yaml"
RUN_NAME_PREFIX_BATCH1 = "gmgp1-xauusd-oanda-l1-rehpo-batch1"
RUN_NAME_PREFIX_BATCH2 = "gmgp1-xauusd-oanda-l1-rehpo-batch2"
LOG_PREFIX_BATCH1 = "run_gmgp1_l1_oanda_batch1_seed"
LOG_PREFIX_BATCH2 = "run_gmgp1_l1_oanda_batch2_seed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] gmgp1_l1_oanda - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def ssh(remote_cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*SSH_PREFIX, remote_cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def _launch_batch(seeds: list[int], run_prefix: str, log_prefix: str) -> int:
    seeds_str = " ".join(str(s) for s in seeds)
    remote_cmd = (
        "cd /workspace/DeepScalper && "
        "set -a && source .env && set +a && "
        "TS=$(date +%Y%m%d_%H%M%S) && "
        f"for SEED in {seeds_str}; do "
        "TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_gmgp1_oanda_seed$SEED "
        "CUDA_VISIBLE_DEVICES=0 "
        "nohup /root/miniconda3/bin/python -u scripts/run_full_pipeline.py "
        f"--config {CONFIG_PATH} "
        f"--run_name {run_prefix}-seed${{SEED}}_${{TS}} "
        "--seed $SEED "
        f"> {log_prefix}${{SEED}}.log 2>&1 & "
        "echo \"launched seed $SEED pid $!\"; "
        "sleep 2; "
        "done; "
        "echo ===; "
        f"pgrep -af 'run_full_pipeline.*{run_prefix}' | head"
    )
    log.info("Launching batch: seeds=%s prefix=%s", seeds, run_prefix)
    try:
        res = ssh(remote_cmd, timeout=180)
    except subprocess.TimeoutExpired:
        log.error("launch ssh timed out")
        return 124
    log.info("stdout:\n%s", res.stdout)
    if res.returncode != 0:
        log.error("launch FAILED exit=%d stderr:\n%s", res.returncode, res.stderr)
    return res.returncode


def _count_running(run_prefix: str) -> int:
    """Count `run_full_pipeline` processes matching prefix still alive."""
    try:
        res = ssh(
            f"pgrep -af 'run_full_pipeline.*{run_prefix}' | wc -l"
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


def _wait_for_exit(run_prefix: str, expected: int) -> bool:
    """Poll until no processes matching `run_prefix` remain."""
    start = time.time()
    max_seconds = MAX_POLL_HOURS * 3600
    while True:
        elapsed = time.time() - start
        if elapsed > max_seconds:
            log.error("poll ceiling %dh exceeded for prefix=%s",
                      MAX_POLL_HOURS, run_prefix)
            return False
        n = _count_running(run_prefix)
        # pgrep -af matches its own subshell; n=1 means only the poller itself.
        alive = max(0, n - 1) if n >= 0 else -1
        if n >= 0 and alive == 0:
            log.info("all %s processes exited (elapsed %.1fm)",
                     run_prefix, elapsed / 60.0)
            return True
        if n == -1:
            log.warning("poll error; retrying in %ds", POLL_INTERVAL)
        else:
            log.info("%s alive: %d/%d (elapsed %.1fm)",
                     run_prefix, alive, expected, elapsed / 60.0)
        time.sleep(POLL_INTERVAL)


def main() -> int:
    log.info("GMGP1-XAUUSD L1-OANDA redo (Path B, S493-cont)")
    log.info("config: %s", CONFIG_PATH)
    log.info("batch1 seeds: %s (prefix %s)", BATCH1_SEEDS, RUN_NAME_PREFIX_BATCH1)
    log.info("batch2 seeds: %s (prefix %s)", BATCH2_SEEDS, RUN_NAME_PREFIX_BATCH2)

    rc = _launch_batch(BATCH1_SEEDS, RUN_NAME_PREFIX_BATCH1, LOG_PREFIX_BATCH1)
    if rc != 0:
        log.error("batch 1 launch failed rc=%d — aborting before batch 2", rc)
        return rc

    log.info("batch 1 fired; polling for exit before batch 2...")
    if not _wait_for_exit(RUN_NAME_PREFIX_BATCH1, len(BATCH1_SEEDS)):
        log.error("batch 1 did not complete in %dh — aborting batch 2",
                  MAX_POLL_HOURS)
        return 2

    log.info("firing batch 2")
    rc = _launch_batch(BATCH2_SEEDS, RUN_NAME_PREFIX_BATCH2, LOG_PREFIX_BATCH2)
    if rc != 0:
        log.error("batch 2 launch failed rc=%d", rc)
        return rc

    log.info("batch 2 fired. Monitor via: ssh -p <SSH_PORT> gpuhub-2 "
             "'ls -la /workspace/DeepScalper/%s*.log'", LOG_PREFIX_BATCH2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
