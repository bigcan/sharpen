#!/usr/bin/env python
"""Launch GMGP1 XAUUSD Stage 3 WF on OANDA, 3 concurrent seeds (S493-cont, Path B).

3 seeds * 8 folds * 500K steps on gpuhub-2 GPU 0. One batch of 3 concurrent
SAC trainers is well under the 5-concurrent cap
(`feedback_gpuhub2_max_5_concurrent_sac.md`) and each run walks its 8 folds
sequentially, so total wall time is ~8 folds * ~2.5h/fold = ~20h.

Each WF run produces 8 fold checkpoint dirs matching the ensemble config's
  checkpoint_pattern: checkpoints/WF_seed{seed}_fold_{fold:02d}_*/checkpoint_final.pth

One-shot launch (no batching/polling required — single batch of 3).
"""
from __future__ import annotations

import logging
import subprocess
import sys

SSH_PREFIX = ("ssh", "-p", "<SSH_PORT>", "-o", "StrictHostKeyChecking=no",
              "root@<GPU_HOST>")

WF_SEEDS = [42, 3141, 2025]
CONFIG_PATH = "configs/gmgp1_xauusd_ftmo_rehpo_wf_multiseed_oanda.yaml"
LOG_PREFIX = "run_gmgp1_wf_oanda_seed"
RUN_NAME_PREFIX_TMPL = "gmgp1-xauusd-oanda-wf-seed{seed}"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] gmgp1_wf_oanda - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def ssh(remote_cmd: str, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*SSH_PREFIX, remote_cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def launch() -> int:
    seeds_str = " ".join(str(s) for s in WF_SEEDS)
    remote_cmd = (
        "cd /workspace/DeepScalper && "
        "set -a && source .env && set +a && "
        f"for SEED in {seeds_str}; do "
        "TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_gmgp1_wf_seed$SEED "
        "CUDA_VISIBLE_DEVICES=0 "
        "nohup /root/miniconda3/bin/python -u scripts/run_walk_forward.py "
        f"--config {CONFIG_PATH} "
        "--seed $SEED "
        f"--run_name_prefix gmgp1-xauusd-oanda-wf-seed${{SEED}} "
        f"> {LOG_PREFIX}${{SEED}}.log 2>&1 & "
        "echo \"launched WF seed $SEED pid $!\"; "
        "sleep 3; "
        "done; "
        "echo ===; "
        "pgrep -af 'run_walk_forward.*gmgp1-xauusd-oanda-wf' | head"
    )
    log.info("Launching WF: seeds=%s config=%s", WF_SEEDS, CONFIG_PATH)
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
    log.info("GMGP1-XAUUSD Stage 3 WF on OANDA (Path B, S493-cont)")
    log.info("3 seeds * 8 folds * 500K steps (~20h wall, ~60 GPU-hours)")
    rc = launch()
    if rc == 0:
        log.info("WF fired. Monitor via:")
        log.info("  ssh -p <SSH_PORT> gpuhub-2 'tail -f /workspace/DeepScalper/%s42.log'", LOG_PREFIX)
        log.info("  ssh -p <SSH_PORT> gpuhub-2 \"pgrep -af 'run_walk_forward.*gmgp1-xauusd-oanda-wf' | wc -l\"")
    return rc


if __name__ == "__main__":
    sys.exit(main())
