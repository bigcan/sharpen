#!/usr/bin/env python
"""Launch GMGP1 BTC Velotrade L1 multiseed (S494 re-HPO trial #48 HPs).

10 seeds split across gpuhub-1's two 4090s — 5 per GPU (respects the
`feedback_gpuhub2_max_5_concurrent_sac.md` 5-concurrent cap, which is
per-GPU not per-host). All 10 launched simultaneously because each GPU
sees only 5 concurrent SACs. Per-seed TORCHINDUCTOR_CACHE_DIR isolation
prevents the torch.compile/multiprocessing race that bit S488 batch 1.

gpuhub-2 is running the concurrent GMGP1-XAUUSD-WF (3 seeds), so per the
S488 non-HPO-on-gpuhub-2 rule we'd normally use gpuhub-2 for this L1 —
but gpuhub-2 is booked and gpuhub-1 is idle post-HPO. User-approved
deviation (S494) since the rule's spirit is "don't saturate the HPO
fleet" and gpuhub-1 has no pending HPO.

Config: configs/gmgp1_btc_velotrade_rehpo_l1_multiseed.yaml
Run-name prefix: gmgp1-btc-velotrade-l1-rehpo
Log prefix:      run_gmgp1_btc_l1_rehpo_seed<SEED>.log

Run in background:
    python scripts/launch_gmgp1_btc_l1_rehpo.py \
        2>&1 | tee C:/tmp/launch_gmgp1_btc_l1_rehpo.log
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time

# gpuhub-1 (2x RTX 4090), port <SSH_PORT>
SSH_PREFIX = ("ssh", "-p", "<SSH_PORT>", "-o", "StrictHostKeyChecking=no",
              "root@<GPU_HOST>")

GPU0_SEEDS = [1337, 2025, 3141, 5150, 9999]
GPU1_SEEDS = [42, 123, 456, 789, 1024]
POLL_INTERVAL = 300   # 5 min
MAX_POLL_HOURS = 4

CONFIG_PATH = "configs/gmgp1_btc_velotrade_rehpo_l1_multiseed.yaml"
RUN_NAME_PREFIX = "gmgp1-btc-velotrade-l1-rehpo"
LOG_PREFIX = "run_gmgp1_btc_l1_rehpo_seed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] gmgp1_btc_l1 - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def ssh(remote_cmd: str, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*SSH_PREFIX, remote_cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def _launch_all() -> int:
    gpu0 = " ".join(str(s) for s in GPU0_SEEDS)
    gpu1 = " ".join(str(s) for s in GPU1_SEEDS)
    remote_cmd = (
        "cd /workspace/DeepScalper && "
        "set -a && source .env && set +a && "
        "TS=$(date +%Y%m%d_%H%M%S) && "
        f"for SEED in {gpu0}; do "
        "TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_gmgp1_btc_seed$SEED "
        "CUDA_VISIBLE_DEVICES=0 "
        "nohup /root/miniconda3/bin/python -u scripts/run_full_pipeline.py "
        f"--config {CONFIG_PATH} "
        f"--run_name {RUN_NAME_PREFIX}-seed${{SEED}}_${{TS}} "
        "--seed $SEED "
        f"> {LOG_PREFIX}${{SEED}}.log 2>&1 & "
        "echo \"[gpu0] launched seed $SEED pid $!\"; "
        "sleep 2; "
        "done; "
        f"for SEED in {gpu1}; do "
        "TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_gmgp1_btc_seed$SEED "
        "CUDA_VISIBLE_DEVICES=1 "
        "nohup /root/miniconda3/bin/python -u scripts/run_full_pipeline.py "
        f"--config {CONFIG_PATH} "
        f"--run_name {RUN_NAME_PREFIX}-seed${{SEED}}_${{TS}} "
        "--seed $SEED "
        f"> {LOG_PREFIX}${{SEED}}.log 2>&1 & "
        "echo \"[gpu1] launched seed $SEED pid $!\"; "
        "sleep 2; "
        "done; "
        "echo ===; "
        f"pgrep -af 'run_full_pipeline.*{RUN_NAME_PREFIX}' | wc -l"
    )
    log.info("Launching 10 seeds: gpu0=%s gpu1=%s", GPU0_SEEDS, GPU1_SEEDS)
    try:
        res = ssh(remote_cmd, timeout=240)
    except subprocess.TimeoutExpired:
        log.error("launch ssh timed out")
        return 124
    log.info("stdout:\n%s", res.stdout)
    if res.returncode != 0:
        log.error("launch FAILED exit=%d stderr:\n%s", res.returncode, res.stderr)
    return res.returncode


def _count_running() -> int:
    try:
        res = ssh(
            f"pgrep -af 'run_full_pipeline.*{RUN_NAME_PREFIX}' | wc -l",
            timeout=60,
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


def _wait_for_exit() -> bool:
    start = time.time()
    max_seconds = MAX_POLL_HOURS * 3600
    while True:
        elapsed = time.time() - start
        if elapsed > max_seconds:
            log.error("poll ceiling %dh exceeded", MAX_POLL_HOURS)
            return False
        n = _count_running()
        # pgrep -af matches its own subshell/ssh; account for it.
        alive = max(0, n - 1) if n >= 0 else -1
        if n >= 0 and alive == 0:
            log.info("all seeds exited (elapsed %.1fm)", elapsed / 60.0)
            return True
        if n == -1:
            log.warning("poll error; retrying in %ds", POLL_INTERVAL)
        else:
            log.info("alive: %d/10 (elapsed %.1fm)", alive, elapsed / 60.0)
        time.sleep(POLL_INTERVAL)


def main() -> int:
    log.info("GMGP1-BTC Velotrade L1 multiseed (S494 re-HPO trial #48)")
    log.info("config: %s", CONFIG_PATH)
    log.info("host: gpuhub-1 (2x RTX 4090), gpu0 seeds=%s gpu1 seeds=%s",
             GPU0_SEEDS, GPU1_SEEDS)

    rc = _launch_all()
    if rc != 0:
        log.error("launch failed rc=%d", rc)
        return rc

    log.info("fired; polling for exit...")
    if not _wait_for_exit():
        log.error("did not complete in %dh", MAX_POLL_HOURS)
        return 2

    log.info("all 10 seeds complete. Pull checkpoints via "
             "`python scripts/auto_collect_checkpoints.py --hours 6`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
