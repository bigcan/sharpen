#!/usr/bin/env python
"""Auto-queue the 6 backfill seeds of SG-1 BTC L1 multiseed after the 4
surviving seeds finish.

Context (S494-cont/S495): SG-1 BTC L1 multiseed launched 2026-04-23 08:42 UTC
on gpuhub-2:0 with N=10 planned seeds, parent WandB run `ighx368o`. A
scheduling conflict with the GMGP1-XAUUSD OANDA WF (both on gpuhub-2:0
simultaneously, ~260 env workers) crashed 6/10 of the L1 seeds within minutes
of XAUUSD WF launching. Survivors: [42, 123, 789, 9999].

Backfill: [456, 1024, 1337, 2025, 3141, 5150]. Re-runs under their own
per-seed WandB runs (no parent attach — batch1 parent `ighx368o` will be
`finished` by the time this fires). Aggregation via
`results/sg1_btc_l1_aggregate.json` reconciliation reads all seeds by tag.

Polls gpuhub-2 every POLL_INTERVAL seconds. When all batch-1 l1-multiseed
procs have exited (excluding any with `backfill` in the cmdline), fires the
backfill. Staggered 2s launches + isolated torch.compile caches per seed
(defensive against S488 multiprocessing/cache race).

Run in background:
    python scripts/auto_queue_sg1_btc_l1_backfill.py \
        2>&1 | tee C:/tmp/auto_queue_sg1_btc_l1_backfill.log
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time

# gpuhub-2 — same SSH host, different port from gpuhub-1
SSH_PREFIX = ("ssh", "-p", "<SSH_PORT>", "-o", "StrictHostKeyChecking=no",
              "root@<GPU_HOST>")

BATCH1_ALIVE_SEEDS = [42, 123, 789, 9999]         # the 4 survivors
BACKFILL_SEEDS = [456, 1024, 1337, 2025, 3141, 5150]
POLL_INTERVAL = 300   # 5 min
MAX_POLL_HOURS = 6

# Match batch1 cmdline but exclude backfill launches (which we're about to fire)
BATCH1_PGREP = (
    "pgrep -af 'run_full_pipeline.*sg1_btc_velotrade_l1_multiseed' "
    "| grep -v backfill "
    "| wc -l"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] auto_queue_btc - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def ssh(remote_cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*SSH_PREFIX, remote_cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def count_batch1_running() -> int:
    """Count batch-1 SG-1 BTC l1-multiseed python procs still alive."""
    try:
        res = ssh(BATCH1_PGREP)
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


def launch_backfill() -> int:
    seeds_str = " ".join(str(s) for s in BACKFILL_SEEDS)
    remote_cmd = (
        "cd /workspace/DeepScalper && "
        "set -a && source .env && set +a && "
        "TS=$(date +%Y%m%d_%H%M%S) && "
        f"for SEED in {seeds_str}; do "
        "TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_sg1_btc_backfill_seed$SEED "
        "CUDA_VISIBLE_DEVICES=0 "
        "nohup /root/miniconda3/bin/python -u scripts/run_full_pipeline.py "
        "--config configs/sg1_btc_velotrade_l1_multiseed.yaml "
        "--run_name sg1-btc-velotrade-l1-multiseed-backfill-seed${SEED}_${TS} "
        "--seed $SEED "
        "--tags sg1-btc-velotrade-l1-multiseed-backfill "
        "sg1-btc-velotrade-l1-multiseed-backfill-seed${SEED}_${TS} "
        "gpuhub rtx4090 "
        "> run_sg1_btc_l1_backfill_seed${SEED}.log 2>&1 & "
        "echo \"launched seed $SEED pid $!\"; "
        "sleep 2; "
        "done; "
        "echo ===; "
        "pgrep -af 'run_full_pipeline.*backfill' | head"
    )
    log.info("Launching backfill: seeds %s", BACKFILL_SEEDS)
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
    log.info("auto-queue SG-1 BTC L1 backfill: polling every %ds (max %dh)",
             POLL_INTERVAL, MAX_POLL_HOURS)
    log.info("batch1 alive seeds: %s", BATCH1_ALIVE_SEEDS)
    log.info("backfill seeds: %s", BACKFILL_SEEDS)

    while True:
        elapsed = time.time() - start
        if elapsed > max_seconds:
            log.error("poll ceiling %dh exceeded — aborting without launching",
                      MAX_POLL_HOURS)
            return 2

        n = count_batch1_running()
        if n == 0:
            log.info("batch 1 complete — firing backfill")
            break
        if n == -1:
            log.warning("poll error; retrying in %ds", POLL_INTERVAL)
        else:
            log.info("batch 1 alive: %d (expected up to %d) — elapsed %.1fm",
                     n, len(BATCH1_ALIVE_SEEDS) * 2, elapsed / 60.0)
        time.sleep(POLL_INTERVAL)

    rc = launch_backfill()
    log.info("backfill launch returncode=%d", rc)
    if rc == 0:
        log.info("backfill fired. Monitor via: ssh gpuhub-2 "
                 "'ls -la /workspace/DeepScalper/run_sg1_btc_l1_backfill_seed*.log'")
    return rc


if __name__ == "__main__":
    sys.exit(main())
