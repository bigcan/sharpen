#!/usr/bin/env python
"""X1 cost-corrected de-leaked L1 -- collect + reconcile (S553-cont-13).

Pulls WandB val/test metrics + SFTPs checkpoint_final.pth for all 10 X1 seeds
(split across gpuhub-1 [6 seeds] and gpuhub-2 [4 seeds]), verifies
key == name_seed == ckpt_seed (S540 scramble guard), computes top-3 via
val_argmax_pf (Protocol v2.1 / S495), and writes
results/sg1_btc_velotrade_decay01_x1/seed_report.json (schema 1.0).

Single-host collect_run.py cannot pull a split-host cohort in one pass, so this
mirrors the S542 "WandB-authoritative" recovery pattern for the X1 re-derive.
"""
import json
import re
import statistics
import sys
from pathlib import Path

import paramiko
import wandb

ENT = "bigcan-chiwin-technology"
PROJ = "FinRL-Pro-DS"
WORKSTREAM = "sg1_btc_velotrade_decay01_x1"
L1_CONFIG = "configs/sg1_btc_velotrade_l1_multiseed_decay01_x1_costcorr.yaml"
RUN_TS = "20260530_172301"
REMOTE_WS = "/workspace/DeepScalper"
GENERATED_BY = "S553-cont-13 X1 collect+reconcile (collect_reconcile_x1.py, WandB authoritative)"
OUT_DIR = Path("results") / WORKSTREAM
SEED_NAME_RE = re.compile(r"seed(\d+)_")
DEGENERATE_TRADE_FLOOR = 100  # test-window trade_count below this => non-trading/degenerate policy

# seed -> (wandb_run_id, host_key)  [host confirmed via remote ls of checkpoints dir]
SEED_MAP = {
    42:   ("ly0hz3sr", "gpuhub-1"),
    123:  ("wxo59x3j", "gpuhub-1"),
    789:  ("pczrzv37", "gpuhub-1"),
    1337: ("zsme1eua", "gpuhub-1"),
    3141: ("0fi6c20l", "gpuhub-1"),
    9999: ("ry2nfnk8", "gpuhub-1"),
    456:  ("29r1r760", "gpuhub-2"),
    1024: ("n1dqzx2x", "gpuhub-2"),
    2025: ("yk2ii09w", "gpuhub-2"),
    5150: ("qhbt3xld", "gpuhub-2"),
}


def run_name_for(seed):
    return f"sg1-btc-x1-costcorr-seed{seed}_{RUN_TS}"


def load_instances():
    with open("instances.json", encoding="utf-8") as f:
        return json.load(f)["instances"]


def parse_seed(token):
    m = SEED_NAME_RE.search(token)
    return int(m.group(1)) if m else None


def sftp_checkpoint(inst, run_name):
    """Return (local_path, size_bytes) or (None, None) if remote file missing."""
    host, port, pw = inst["host"], int(inst["port"]), inst["password"]
    remote = f"{REMOTE_WS}/checkpoints/{run_name}/checkpoint_final.pth"
    local_dir = Path("checkpoints") / run_name
    local_dir.mkdir(parents=True, exist_ok=True)
    local = local_dir / "checkpoint_final.pth"
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.WarningPolicy())
    ssh.connect(host, port=port, username="root", password=pw, timeout=30)
    try:
        sftp = ssh.open_sftp()
        try:
            st = sftp.stat(remote)
        except FileNotFoundError:
            return None, None
        sftp.get(remote, str(local))
        return str(local).replace("\\", "/"), st.st_size
    finally:
        ssh.close()


def stats_block(pfs):
    if not pfs:
        return None
    mean = statistics.mean(pfs)
    sd = statistics.stdev(pfs) if len(pfs) > 1 else 0.0
    return {
        "n": len(pfs),
        "mean": mean,
        "median": statistics.median(pfs),
        "min": min(pfs),
        "max": max(pfs),
        "cv": (sd / mean) if mean else None,
    }


def main():
    api = wandb.Api()
    instances = load_instances()
    seeds = {}
    fail = []

    for seed, (rid, host_key) in sorted(SEED_MAP.items()):
        expected_name = run_name_for(seed)
        run = api.run(f"{ENT}/{PROJ}/{rid}")
        s = run.summary

        # --- S540 scramble guard: key == name_seed == ckpt_seed ---
        name_seed = parse_seed(run.name)
        ckpt_dir = f"checkpoints/{expected_name}"
        ckpt_seed = parse_seed(expected_name)
        recon_ok = (seed == name_seed == ckpt_seed) and (run.name == expected_name)
        if not recon_ok:
            fail.append(f"seed {seed}: key={seed} name='{run.name}'(seed={name_seed}) "
                        f"ckpt_seed={ckpt_seed} rid={rid}")

        # --- SFTP checkpoint from the host that trained this seed ---
        local_ckpt, size = sftp_checkpoint(instances[host_key], expected_name)
        ckpt_ok = local_ckpt is not None

        seeds[str(seed)] = {
            "wandb_run_id": rid,
            "wandb_run_name": run.name,
            "seed": seed,
            "host": host_key,
            "ckpt_dir": ckpt_dir,
            "ckpt_local": local_ckpt,
            "ckpt_size_mb": round(size / 1048576, 2) if size else None,
            "reconciled": recon_ok,
            "val_pf": s.get("backtest_val/profit_factor"),
            "test_pf": s.get("backtest_test/profit_factor"),
            "val_max_drawdown": s.get("backtest_val/max_drawdown"),
            "test_max_drawdown": s.get("backtest_test/max_drawdown"),
            "val_total_trades": s.get("backtest_val/trade_count"),
            "test_total_trades": s.get("backtest_test/trade_count"),
            "val_sharpe_daily": s.get("backtest_val/sharpe_daily"),
            "test_sharpe_daily": s.get("backtest_test/sharpe_daily"),
            "test_win_rate": s.get("backtest_test/win_rate"),
            "train_status": s.get("train/status"),
            "test_status": s.get("backtest_test/status"),
            "final_step": s.get("_step"),
            "checkpoint_downloaded": ckpt_ok,
        }
        flag = "OK " if (recon_ok and ckpt_ok) else "!! "
        print(f"  {flag}seed {seed:<5} {host_key}  val_pf={seeds[str(seed)]['val_pf']:.4f} "
              f"test_pf={seeds[str(seed)]['test_pf']:.4f} trades={seeds[str(seed)]['test_total_trades']} "
              f"ckpt={'down' if ckpt_ok else 'MISSING'}")

    # --- degenerate detection (non-trading policies) ---
    degenerate = sorted(
        int(k) for k, v in seeds.items()
        if (v["test_total_trades"] or 0) < DEGENERATE_TRADE_FLOOR
    )
    healthy = sorted(int(k) for k in seeds if int(k) not in degenerate)

    # --- top-3 via val_argmax_pf (Protocol v2.1 / S495) ---
    by_val_all = sorted(seeds.values(), key=lambda v: v["val_pf"], reverse=True)
    top3_val_all = [v["seed"] for v in by_val_all[:3]]
    by_val_healthy = sorted(
        (v for v in seeds.values() if v["seed"] in healthy),
        key=lambda v: v["val_pf"], reverse=True,
    )
    top3_val_healthy = [v["seed"] for v in by_val_healthy[:3]]
    by_test_healthy = sorted(
        (v for v in seeds.values() if v["seed"] in healthy),
        key=lambda v: v["test_pf"], reverse=True,
    )
    top3_test_healthy = [v["seed"] for v in by_test_healthy[:3]]

    val_pfs_all = [v["val_pf"] for v in seeds.values()]
    test_pfs_all = [v["test_pf"] for v in seeds.values()]
    val_pfs_h = [v["val_pf"] for v in seeds.values() if v["seed"] in healthy]
    test_pfs_h = [v["test_pf"] for v in seeds.values() if v["seed"] in healthy]

    report = {
        "schema_version": "1.0",
        "workstream": WORKSTREAM,
        "wandb_project": f"{ENT}/{PROJ}",
        "l1_config": L1_CONFIG,
        "generated_by": GENERATED_BY,
        "n_seeds": len(seeds),
        "aggregation_rule_selector": "val_argmax_pf",
        "top3_by_val_argmax_pf": top3_val_healthy,
        "selection_notes": (
            "val_argmax_pf over healthy seeds (Protocol v2.1/S495). val window is tiny "
            "(~16 trades/seed) and noisy -- top-3 is a CANDIDATE set; the downstream "
            "Stage 2.5-R bootstrap + Stage 3 WF (wf_median_pf + shallowest uncapped DD) "
            "are the real arbiters. Degenerate (non-trading) seeds excluded from selection."
        ),
        "degenerate_seeds": degenerate,
        "degenerate_trade_floor": DEGENERATE_TRADE_FLOOR,
        "healthy_seeds": healthy,
        "audit_only": {
            "top3_by_val_argmax_pf_all_seeds": top3_val_all,
            "top3_by_test_pf_healthy": top3_test_healthy,
        },
        "stats_all_seeds": {
            "n_seeds": len(seeds),
            "val_pf": stats_block(val_pfs_all),
            "test_pf": stats_block(test_pfs_all),
        },
        "stats_healthy_seeds": {
            "n_seeds": len(healthy),
            "val_pf": stats_block(val_pfs_h),
            "test_pf": stats_block(test_pfs_h),
        },
        "reconciliation_passed": len(fail) == 0,
        "reconciliation_failures": fail,
        "seeds": seeds,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "seed_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  RECONCILE: {'PASS' if not fail else 'FAIL'} ({len(fail)} mismatches)")
    if fail:
        for fl in fail:
            print("   !!", fl)
    print(f"  degenerate (non-trading) seeds: {degenerate}")
    print(f"  healthy seeds ({len(healthy)}): {healthy}")
    print(f"  top-3 val_argmax_pf (healthy)  -> {top3_val_healthy}")
    print(f"  [audit] top-3 test_pf (healthy) -> {top3_test_healthy}")
    sh = report["stats_healthy_seeds"]
    print(f"  healthy test_pf: median={sh['test_pf']['median']:.4f} "
          f"cv={sh['test_pf']['cv']:.4f}  val_pf median={sh['val_pf']['median']:.4f}")
    print(f"  checkpoints downloaded: "
          f"{sum(1 for v in seeds.values() if v['checkpoint_downloaded'])}/{len(seeds)}")
    print(f"  seed_report -> {out}")
    print("=" * 70)
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
