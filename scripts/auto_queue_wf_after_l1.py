#!/usr/bin/env python
"""Auto-chain SG-1 XAUUSD FTMO re-HPO WF Stage 3 after L1 OANDA batch 2.

S488 Path A. Overrides pre-committed "pause-and-eyeball" gate per user
direction (2026-04-20, going to bed, no eyeball). **Fires WF only if L1
gate passes**; otherwise exits with decision log and no WF launched.

Flow (all on gpuhub-2:gpu0 per fleet rule; HPO→gpuhub-1, non-HPO→gpuhub-2):
  1. Poll for batch-2 OANDA L1 processes every POLL_INTERVAL.
  2. When all 5 batch-2 seeds exit, sleep GRACE_SECS for WandB finalization.
  3. Pull `backtest_test/profit_factor` from WandB for all 10 L1 seeds
     (batch1 + batch2, tag-filtered).
  4. Evaluate L1 gate:
        - all 10 seeds reported PF
        - all PF > 1.0 (profitable)
        - median PF >= gates.l1_pf_floor (1.2)
        - CV(PF) <= gates.l1_pf_cv_max (0.30)
        - CV(PF) NOT in gates.l1_pf_cv_ambiguous [0.22, 0.38]
          (ambiguous = escalate to N=20, do not auto-chain)
  5. If pass: pick top-3 seeds by test PF, fan out WF concurrently on
     gpuhub-2:gpu0 (3 parallel < 5 cap). If fail: log reason, exit.

Run in background (no interactive sessions needed after launch):
    nohup python scripts/auto_queue_wf_after_l1.py \
        > C:/tmp/auto_queue_wf_after_l1.log 2>&1 &
"""
from __future__ import annotations

import logging
import statistics
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SSH_PREFIX = ("ssh", "-p", "<SSH_PORT>", "-o", "StrictHostKeyChecking=no",
              "root@<GPU_HOST>")

# Seeds that were launched on batch 1 and batch 2 of the OANDA L1 redo.
BATCH1_SEEDS = [1337, 2025, 3141, 5150, 9999]
BATCH2_SEEDS = [42, 123, 456, 789, 1024]
ALL_L1_SEEDS = BATCH1_SEEDS + BATCH2_SEEDS

# WF config + WandB run-name prefixes.
L1_NAME_PREFIX_BATCH1 = "sg1-xauusd-oanda-l1-rehpo-batch1"
L1_NAME_PREFIX_BATCH2 = "sg1-xauusd-oanda-l1-rehpo-batch2"
# Must see batch-2 alive before we can see it "complete" — guards against
# starting WF before batch-2 auto-queue has even fired.
MIN_BATCH2_ALIVE_SEEN = 3
L1_UPSTREAM_TAG = "upstream:sg1_xauusd_ftmo_rehpo_20260419"  # L1 config tag
WF_CONFIG = "configs/sg1_xauusd_ftmo_rehpo_wf_oanda.yaml"
WF_NAME_PREFIX = "sg1-xauusd-wf-rehpo-oanda"
WF_LOG_PREFIX = "run_wf_oanda_seed"

# L1 gate (mirror of configs/sg1_xauusd_ftmo_rehpo_l1_multiseed_oanda.yaml
# gates block — duplicated here so gate-eval is self-contained).
L1_PF_FLOOR = 1.2
L1_PF_CV_MAX = 0.30
L1_PF_CV_AMBIG_LO = 0.22
L1_PF_CV_AMBIG_HI = 0.38

# Polling.
POLL_INTERVAL = 300       # 5 min
MAX_POLL_HOURS = 24       # batch 2 ETA ~00:45 UTC+8 + 24h headroom
GRACE_SECS = 120          # let WandB uploads settle after process exit

# WandB filter fallback if env var missing.
WANDB_ENTITY = "bigcan-chiwin-technology"
WANDB_PROJECT = "FinRL-Pro-DS"

DECISION_LOG = Path("C:/tmp/auto_queue_wf_decision.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] wf_chain - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Remote poll
# ---------------------------------------------------------------------------

def ssh(remote_cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*SSH_PREFIX, remote_cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def count_batch2_running() -> int:
    """Count batch-2 OANDA l1-multiseed processes alive on gpuhub-2."""
    try:
        res = ssh(
            f"pgrep -af 'run_full_pipeline.*{L1_NAME_PREFIX_BATCH2}' | wc -l"
        )
    except subprocess.TimeoutExpired:
        log.warning("ssh poll timed out")
        return -1
    if res.returncode != 0:
        log.warning("poll exit=%d stderr=%s", res.returncode, res.stderr[:200])
        return -1
    try:
        return int(res.stdout.strip())
    except ValueError:
        log.warning("unparseable count: %r", res.stdout)
        return -1


# ---------------------------------------------------------------------------
# L1 gate eval via WandB
# ---------------------------------------------------------------------------

def collect_l1_pfs() -> dict[int, float | None]:
    """Pull test PF for each of the 10 L1 OANDA seeds from WandB.

    Match strategy: tag upstream:... is shared by both L1 batches. Walk the
    last ~200 runs under that tag, pick the one whose run-name contains
    `-seed<N>_` for each N in ALL_L1_SEEDS. Most-recent wins if duplicate.
    """
    import wandb  # deferred — prevents CLI latency on gate-only runs
    api = wandb.Api()
    filters = {"tags": {"$in": [L1_UPSTREAM_TAG]}}
    runs = api.runs(
        f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters=filters, order="-created_at", per_page=50,
    )

    by_seed: dict[int, float | None] = {s: None for s in ALL_L1_SEEDS}
    for run in runs:
        name = run.name or ""
        for seed in ALL_L1_SEEDS:
            if by_seed[seed] is not None:
                continue
            if f"-seed{seed}_" not in name:
                continue
            # Finished-only; partial runs skew medians.
            if run.state not in ("finished",):
                log.info("skip seed=%d run=%s state=%s", seed, name, run.state)
                continue
            summary = run.summary._json_dict
            pf = (summary.get("backtest_test/profit_factor")
                  or summary.get("backtest/test_profit_factor"))
            if pf is None:
                log.warning("seed=%d run=%s has no test PF in summary", seed, name)
                continue
            by_seed[seed] = float(pf)
            log.info("seed=%d test_pf=%.4f (%s)", seed, pf, name)

    return by_seed


def eval_l1_gate(pfs: dict[int, float | None]) -> tuple[bool, str, list[int]]:
    """Return (pass, reason, top3_seeds).

    Gate mirrors configs/sg1_xauusd_ftmo_rehpo_l1_multiseed_oanda.yaml.
    """
    missing = [s for s, p in pfs.items() if p is None]
    if missing:
        return False, f"missing PF for seeds {missing}", []

    vals = list(pfs.values())
    unprofitable = [s for s, p in pfs.items() if p <= 1.0]
    if unprofitable:
        return False, f"unprofitable seeds {unprofitable} (PF<=1.0)", []

    med = statistics.median(vals)
    if med < L1_PF_FLOOR:
        return False, f"median PF {med:.3f} < floor {L1_PF_FLOOR}", []

    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    cv = (std / mean) if mean > 0 else float("inf")
    if cv > L1_PF_CV_MAX:
        return False, f"CV(PF) {cv:.4f} > max {L1_PF_CV_MAX}", []
    if L1_PF_CV_AMBIG_LO <= cv <= L1_PF_CV_AMBIG_HI:
        return (False,
                f"CV(PF) {cv:.4f} in ambiguous range "
                f"[{L1_PF_CV_AMBIG_LO}, {L1_PF_CV_AMBIG_HI}] — escalate "
                "to N=20 manually, do not auto-chain",
                [])

    top3 = sorted(pfs.keys(), key=lambda s: pfs[s], reverse=True)[:3]
    reason = (f"PASS median_pf={med:.3f} mean={mean:.3f} cv={cv:.4f} "
              f"min={min(vals):.3f} top3={top3}")
    return True, reason, top3


# ---------------------------------------------------------------------------
# WF launch
# ---------------------------------------------------------------------------

def launch_wf_for_seeds(seeds: list[int]) -> int:
    """Fan out 3 WF runs concurrently on gpuhub-2:gpu0 via nohup+SSH.

    Each seed gets its own TORCHINDUCTOR_CACHE_DIR to avoid the
    torch.compile/multiprocessing race that killed L1 batch-1 attempt 1.
    """
    seeds_str = " ".join(str(s) for s in seeds)
    remote_cmd = (
        "cd /workspace/DeepScalper && "
        "set -a && source .env && set +a && "
        "TS=$(date +%Y%m%d_%H%M%S) && "
        f"for SEED in {seeds_str}; do "
        "TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_wf_seed$SEED "
        "CUDA_VISIBLE_DEVICES=0 "
        "nohup /root/miniconda3/bin/python -u scripts/run_walk_forward.py "
        f"--config {WF_CONFIG} "
        f"--run_name_prefix {WF_NAME_PREFIX}-seed${{SEED}} "
        "--seed $SEED "
        f"> {WF_LOG_PREFIX}${{SEED}}.log 2>&1 & "
        "echo \"launched WF seed $SEED pid $!\"; "
        "sleep 3; "
        "done; "
        "echo ===; "
        f"pgrep -af run_walk_forward.*{WF_NAME_PREFIX} | head"
    )
    log.info("Launching WF for seeds %s on gpuhub-2:gpu0", seeds)
    try:
        res = ssh(remote_cmd, timeout=240)
    except subprocess.TimeoutExpired:
        log.error("launch ssh timed out")
        return 124
    log.info("stdout:\n%s", res.stdout)
    if res.returncode != 0:
        log.error("launch FAILED exit=%d stderr:\n%s", res.returncode, res.stderr)
    return res.returncode


def write_decision_log(passed: bool, reason: str, pfs: dict[int, float | None],
                       top3: list[int]) -> None:
    DECISION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with DECISION_LOG.open("a", encoding="utf-8") as f:
        f.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S %Z')} ===\n")
        f.write(f"L1 gate: {'PASS' if passed else 'FAIL'}\n")
        f.write(f"reason: {reason}\n")
        f.write("per-seed test PF:\n")
        for s in ALL_L1_SEEDS:
            v = pfs.get(s)
            f.write(f"  seed {s:>5}: {v:.4f}\n" if v is not None
                    else f"  seed {s:>5}: MISSING\n")
        if top3:
            f.write(f"top-3 -> WF: {top3}\n")
        f.write("\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    log.info("auto-chain WF after L1 OANDA batch 2")
    log.info("config: %s", WF_CONFIG)
    log.info("L1 seeds: %s", ALL_L1_SEEDS)
    log.info("poll: every %ds (max %dh)", POLL_INTERVAL, MAX_POLL_HOURS)
    log.info("decision log: %s", DECISION_LOG)

    start = time.time()
    max_seconds = MAX_POLL_HOURS * 3600
    # Two-state gate: (1) observe batch 2 alive at least once (proof it has
    # launched), then (2) wait for it to drop to 0. Without this, we'd
    # mistake "not-yet-started" for "already-done" and fire WF on no data.
    peak_alive = 0

    while True:
        elapsed = time.time() - start
        if elapsed > max_seconds:
            log.error("poll ceiling %dh exceeded — aborting", MAX_POLL_HOURS)
            return 2

        n = count_batch2_running()
        # pgrep -af self-matches the subshell argv; real process count is n-1.
        alive = max(0, n - 1) if n >= 0 else -1
        if alive > peak_alive:
            peak_alive = alive

        if n == -1:
            log.warning("poll error; retrying in %ds", POLL_INTERVAL)
        elif peak_alive < MIN_BATCH2_ALIVE_SEEN:
            log.info("batch 2 not yet launched (alive=%d, peak=%d, need peak>=%d) "
                     "(elapsed %.1fm)", alive, peak_alive, MIN_BATCH2_ALIVE_SEEN,
                     elapsed / 60.0)
        elif alive == 0:
            log.info("batch 2 complete (peak was %d) — grace-sleeping %ds",
                     peak_alive, GRACE_SECS)
            time.sleep(GRACE_SECS)
            break
        else:
            log.info("batch 2 alive: %d/5 (peak %d, elapsed %.1fm)",
                     alive, peak_alive, elapsed / 60.0)
        time.sleep(POLL_INTERVAL)

    # Phase 2: pull L1 PFs + evaluate gate.
    log.info("collecting L1 test PFs from WandB")
    pfs = collect_l1_pfs()
    passed, reason, top3 = eval_l1_gate(pfs)
    log.info("L1 gate: %s - %s", "PASS" if passed else "FAIL", reason)
    write_decision_log(passed, reason, pfs, top3)

    if not passed:
        log.warning("L1 gate FAIL - NOT launching WF. See %s", DECISION_LOG)
        return 3

    # Phase 3: launch WF for top-3 seeds.
    rc = launch_wf_for_seeds(top3)
    if rc == 0:
        log.info("WF launched for top-3 seeds %s. Monitor:\n"
                 "  ssh gpuhub-2 'ls -la /workspace/DeepScalper/%s*.log'",
                 top3, WF_LOG_PREFIX)
    return rc


if __name__ == "__main__":
    sys.exit(main())
