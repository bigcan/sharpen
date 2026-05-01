#!/usr/bin/env python3
"""GMGP1-XAUUSD OANDA WF Stage 2.5 ensemble-confirm wrapper (Protocol v2.1 S495).

Per-fold val-argmax-PF rule selection (S495 amendment) across the 8-fold OANDA
WF. Wraps scripts/sg1_xauusd_ensemble_eval.py::run_stage_2_5_val_selection so
each fold picks its own aggregation rule from val-split PFs, then evaluates
G1..G5 against configs/gmgp1_xauusd_ensemble.gates.yaml.

STAGED, not committed — produced for the 2026-04-24 OANDA WF ensemble-confirm.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from finrl_pro_ds.data.splitter import RollingWindowSplitter  # noqa: E402
from scripts import sg1_xauusd_ensemble_eval as sg1  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gmgp1-oanda-wf-stage-2-5")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_ckpts(pattern: str, seeds: List[int], fold_idx: int) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for s in seeds:
        pat = pattern.format(seed=s, fold=fold_idx)
        pat = pat.replace("{fold:02d}", f"{fold_idx:02d}")
        matches = sorted(glob.glob(pat))
        if not matches:
            raise FileNotFoundError(
                f"seed={s} fold={fold_idx}: no checkpoint match for {pat}"
            )
        if len(matches) > 1:
            log.info(
                f"seed={s} fold={fold_idx}: {len(matches)} matches; using latest "
                f"({matches[-1]})"
            )
        out[s] = matches[-1]
    return out


def _fold_config(base_cfg: dict, fold: dict) -> dict:
    """Return a config copy with data.{train_end,val_start,val_end,test_start,test_end}_date set."""
    c = copy.deepcopy(base_cfg)
    c.setdefault("data", {})
    c["data"]["train_end_date"] = fold["val"].start[:10]
    c["data"]["val_start_date"] = fold["val"].start[:10]
    c["data"]["val_end_date"] = fold["val"].end[:10]
    c["data"]["test_start_date"] = fold["test"].start[:10]
    c["data"]["test_end_date"] = fold["test"].end[:10]
    # Also ensure env uses a reasonable episode_length (backtest mode sets 0)
    return c


def _evaluate_gates(
    per_fold_verdicts: List[dict],
    gates_cfg: dict,
    seeds: List[int],
    agg_rule_per_fold: List[Optional[str]],
    solo_pf_per_fold: List[Dict[int, float]],
    ens_pf_per_fold: List[Optional[float]],
    ens_metrics_per_fold: List[dict],
) -> dict:
    """Apply G1..G5 gates to WF aggregate of per-fold verdicts."""
    gates = gates_cfg.get("gates", {})
    n_ok = sum(1 for v in per_fold_verdicts if v is not None)
    out = {
        "overall": "PENDING",
        "folds_completed": n_ok,
        "folds_total": len(per_fold_verdicts),
        "folds_expected": gates_cfg.get("wf_folds", 8),
        "gates": {},
    }
    if n_ok == 0:
        out["overall"] = "NO_DATA"
        return out

    # G1: solo baseline — per-fold, >=seeds_pass_min of seeds have PF>=pf_floor
    g1 = gates["g1_solo_baseline"]
    pf_floor = float(g1["pf_floor"])
    seeds_pass_min = int(g1["seeds_pass_min"])
    g1_min_folds = int(g1["min_folds_pass"])
    g1_passing = 0
    for spf in solo_pf_per_fold:
        if not spf:
            continue
        passes = sum(1 for s in seeds if (spf.get(s) or 0) >= pf_floor)
        if passes >= seeds_pass_min:
            g1_passing += 1
    out["gates"]["G1_solo_baseline"] = {
        "pass": g1_passing >= g1_min_folds,
        "folds_meeting": g1_passing,
        "min_required": g1_min_folds,
        "pf_floor": pf_floor,
        "seeds_pass_min": seeds_pass_min,
    }

    # G2: per-fold ens_PF >= ens_ratio_min * best_solo_PF
    g2 = gates["g2_no_regression"]
    ratio_min = float(g2["ens_ratio_min"])
    g2_min_folds = int(g2["min_folds_pass"])
    g2_passing = 0
    for spf, epf in zip(solo_pf_per_fold, ens_pf_per_fold):
        if not spf or epf is None:
            continue
        best_solo = max(spf.values()) if spf else None
        if best_solo is None or best_solo <= 0:
            continue
        if epf >= ratio_min * best_solo:
            g2_passing += 1
    out["gates"]["G2_no_regression"] = {
        "pass": g2_passing >= g2_min_folds,
        "folds_meeting": g2_passing,
        "min_required": g2_min_folds,
        "ratio_min": ratio_min,
    }

    # G3: uplift on medians
    g3 = gates["g3_uplift"]
    uplift_min = float(g3["uplift_ratio_min"])
    ens_values = [e for e in ens_pf_per_fold if e is not None]
    best_solo_values = []
    for spf in solo_pf_per_fold:
        if spf:
            vals = list(spf.values())
            if vals:
                best_solo_values.append(max(vals))
    med_ens = float(np.median(ens_values)) if ens_values else 0.0
    med_best = float(np.median(best_solo_values)) if best_solo_values else 0.0
    uplift = med_ens / med_best if med_best > 0 else 0.0
    out["gates"]["G3_uplift"] = {
        "pass": uplift >= uplift_min,
        "median_ens_pf": med_ens,
        "median_best_solo_pf": med_best,
        "uplift": uplift,
        "uplift_min": uplift_min,
    }

    # G4: FTMO compliance on ensemble trajectory per fold
    g4 = gates["g4_ftmo_compliance"]
    dd_buf_min = float(g4["daily_dd_buffer_pp_min"])
    tr_buf_min = float(g4["trailing_dd_buffer_pp_min"])
    g4_min_folds = int(g4["min_folds_pass"])
    g4_passing = 0
    for em in ens_metrics_per_fold:
        if not em:
            continue
        dd_buf = em.get("daily_dd_buffer_pp")
        tr_buf = em.get("trailing_dd_buffer_pp")
        comp = em.get("ftmo_compliance_pass")
        if dd_buf is None or tr_buf is None:
            continue
        if dd_buf >= dd_buf_min and tr_buf >= tr_buf_min and comp:
            g4_passing += 1
    out["gates"]["G4_ftmo_compliance"] = {
        "pass": g4_passing >= g4_min_folds,
        "folds_meeting": g4_passing,
        "min_required": g4_min_folds,
        "dd_buffer_pp_min": dd_buf_min,
        "trailing_dd_buffer_pp_min": tr_buf_min,
    }

    # G5: CV(ens_PF) across folds <= cv_pf_max
    g5 = gates["g5_dd_stability"]
    cv_max = float(g5["cv_pf_max"])
    arr = np.array(ens_values)
    if arr.size >= 2 and arr.mean() > 0:
        cv = float(arr.std() / arr.mean())
    else:
        cv = None
    out["gates"]["G5_dd_stability"] = {
        "pass": (cv is not None and cv <= cv_max),
        "cv_pf": cv,
        "cv_max": cv_max,
    }

    all_pass = all(g.get("pass") for g in out["gates"].values())
    out["overall"] = "PASS" if all_pass else "FAIL"
    out["decision"] = (
        gates_cfg.get("decision", {}).get("all_pass_action")
        if all_pass
        else gates_cfg.get("decision", {}).get("any_fail_action")
    )
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(
    wf_config_path: str,
    gates_path: str,
    output_dir: str,
    device: str = "cpu",
    only_fold: Optional[int] = None,
) -> dict:
    with open(wf_config_path, encoding="utf-8") as f:
        wf_cfg = yaml.safe_load(f)
    with open(gates_path, encoding="utf-8") as f:
        gates_cfg = yaml.safe_load(f)

    ens = wf_cfg.get("ensemble", {})
    seeds: List[int] = list(ens.get("seeds", [42, 3141, 2025]))
    pattern: str = ens.get(
        "checkpoint_pattern",
        "checkpoints/WF_seed{seed}_fold_{fold:02d}_*/checkpoint_final.pth",
    )
    seed_pfs_cfg = {int(k): float(v) for k, v in (ens.get("seed_pfs") or {}).items()}

    split_cfg = wf_cfg.get("splitter", {})
    data_cfg = wf_cfg.get("data", {})
    splitter = RollingWindowSplitter(
        train_months=split_cfg.get("train_months", 6),
        val_months=split_cfg.get("val_months", 1),
        test_months=split_cfg.get("test_months", 1),
        step_months=split_cfg.get("step_months", 1),
        buffer_days=split_cfg.get("buffer_days", 0),
    )
    folds = splitter.split(
        start_date=data_cfg.get("start_date", "2025-01-06"),
        end_date=data_cfg.get("end_date", "2026-04-15"),
    )
    log.info(f"Generated {len(folds)} WF folds from splitter")

    root_out = Path(output_dir)
    root_out.mkdir(parents=True, exist_ok=True)

    per_fold_verdicts: List[Optional[dict]] = []
    agg_rule_per_fold: List[Optional[str]] = []
    solo_pf_per_fold: List[Dict[int, float]] = []
    ens_pf_per_fold: List[Optional[float]] = []
    ens_metrics_per_fold: List[dict] = []

    for i, fold in enumerate(folds):
        if only_fold is not None and i != only_fold:
            per_fold_verdicts.append(None)
            agg_rule_per_fold.append(None)
            solo_pf_per_fold.append({})
            ens_pf_per_fold.append(None)
            ens_metrics_per_fold.append({})
            continue

        fold_id = f"fold_{i:02d}"
        fold_out = root_out / fold_id
        fold_out.mkdir(parents=True, exist_ok=True)
        log.info(
            f"=== {fold_id} val {fold['val'].start[:10]}->{fold['val'].end[:10]} "
            f"test {fold['test'].start[:10]}->{fold['test'].end[:10]} ==="
        )

        try:
            ckpt_paths = _resolve_ckpts(pattern, seeds, i)
        except FileNotFoundError as e:
            log.warning(f"[{fold_id}] skipped — {e}")
            per_fold_verdicts.append(None)
            agg_rule_per_fold.append(None)
            solo_pf_per_fold.append({})
            ens_pf_per_fold.append(None)
            ens_metrics_per_fold.append({})
            continue

        fold_cfg = _fold_config(wf_cfg, fold)
        agents = sg1._load_agents_from_paths(fold_cfg, ckpt_paths, device)

        # Build seed_pfs map for this fold — we reuse the upstream L1-OANDA pf
        # weights (per-ckpt weight doesn't vary across folds in practice).
        seed_pfs = {s: seed_pfs_cfg.get(s, 1.0) for s in seeds}

        try:
            verdict = sg1.run_stage_2_5_val_selection(
                config=fold_cfg,
                agents=agents,
                seed_pfs=seed_pfs,
                out_dir=fold_out,
                device=device,
                buffer_fn=sg1.ftmo_buffers,
                workstream_label=f"gmgp1_xauusd_oanda_wf_{fold_id}",
                seed_checkpoints=ckpt_paths,
                bundle_version="v1",
            )
        except Exception as e:
            log.error(f"[{fold_id}] ensemble eval failed: {e}")
            per_fold_verdicts.append({"error": str(e)})
            agg_rule_per_fold.append(None)
            solo_pf_per_fold.append({})
            ens_pf_per_fold.append(None)
            ens_metrics_per_fold.append({})
            del agents
            continue

        # Pull fold summary
        per_fold_verdicts.append(verdict)
        chosen_rule = verdict["chosen_rule"]
        agg_rule_per_fold.append(chosen_rule)
        solo_pfs = {int(k): float(v) for k, v in verdict["test_solo_pfs"].items()}
        solo_pf_per_fold.append(solo_pfs)
        ens_pf_per_fold.append(float(verdict["chosen_rule_test_pf"]))

        # Load ensemble metrics for G4
        ens_metrics_path = fold_out / "test" / f"{chosen_rule}_metrics.json"
        if ens_metrics_path.exists():
            with open(ens_metrics_path, encoding="utf-8") as f:
                ens_metrics = json.load(f)
            ens_metrics_per_fold.append(ens_metrics)
        else:
            ens_metrics_per_fold.append({})

        del agents

    # Aggregate G1..G5
    gates_verdict = _evaluate_gates(
        per_fold_verdicts,
        gates_cfg,
        seeds,
        agg_rule_per_fold,
        solo_pf_per_fold,
        ens_pf_per_fold,
        ens_metrics_per_fold,
    )

    # Compose overall verdict (Stage 2.5-aggregate)
    ens_pfs_clean = [e for e in ens_pf_per_fold if e is not None]
    median_uplift = None
    uplifts_per_fold = []
    for spf, epf in zip(solo_pf_per_fold, ens_pf_per_fold):
        if spf and epf is not None:
            best_solo = max(spf.values())
            if best_solo > 0:
                uplifts_per_fold.append(epf / best_solo)
    if uplifts_per_fold:
        median_uplift = float(np.median(uplifts_per_fold))

    # Classify Stage 2.5 overall PROMOTE / AMBIGUOUS / SOLO_BEST via median uplift
    uplift_promote = float(gates_cfg["gates"]["g3_uplift"]["uplift_ratio_min"])
    # Ambiguous band taken from gates.yaml ensemble_ambiguous_min when present
    uplift_amb_lo = float(gates_cfg["gates"].get("ensemble_ambiguous_min", 1.05))

    if gates_verdict["overall"] == "PASS":
        overall_verdict = "PROMOTE"
    elif median_uplift is not None and median_uplift >= uplift_promote:
        overall_verdict = "PROMOTE_GATES_FAIL"  # Uplift strong but another gate tripped
    elif median_uplift is not None and median_uplift >= uplift_amb_lo:
        overall_verdict = "AMBIGUOUS_RERUN"
    else:
        overall_verdict = "SOLO_BEST"

    # Winning deploy candidate
    if overall_verdict == "PROMOTE":
        # Per-fold chosen rules may differ → surface the per-fold rule counts.
        rule_counts = {}
        for r in agg_rule_per_fold:
            if r:
                rule_counts[r] = rule_counts.get(r, 0) + 1
        deploy_candidate = {
            "type": "ensemble",
            "per_fold_chosen_rules": agg_rule_per_fold,
            "rule_counts": rule_counts,
            "seeds": seeds,
        }
    else:
        # Solo-best by WF-median PF
        seed_wf_median = {}
        for s in seeds:
            vals = [
                spf.get(s) for spf in solo_pf_per_fold if spf and spf.get(s) is not None
            ]
            if vals:
                seed_wf_median[s] = float(np.median(vals))
        best_solo = (
            max(seed_wf_median, key=lambda s: seed_wf_median[s])
            if seed_wf_median
            else None
        )
        deploy_candidate = {
            "type": "solo",
            "seed": best_solo,
            "wf_median_pf": seed_wf_median.get(best_solo) if best_solo else None,
            "per_seed_wf_median_pf": seed_wf_median,
        }

    summary = {
        "protocol": "v2.1_stage_2_5_wf_val_selection + v2.2_backfill",
        "workstream": "gmgp1_xauusd_oanda_wf",
        "wf_config": wf_config_path,
        "gates_config": gates_path,
        "seeds": seeds,
        "n_folds": len([v for v in per_fold_verdicts if v is not None and "error" not in v]),
        "per_fold_chosen_rule": agg_rule_per_fold,
        "per_fold_uplift": uplifts_per_fold,
        "per_fold_solo_pf": solo_pf_per_fold,
        "per_fold_ens_pf": ens_pf_per_fold,
        "median_uplift": median_uplift,
        "ens_pf_mean": float(np.mean(ens_pfs_clean)) if ens_pfs_clean else None,
        "ens_pf_cv": (
            float(np.std(ens_pfs_clean) / np.mean(ens_pfs_clean))
            if ens_pfs_clean and np.mean(ens_pfs_clean) > 0
            else None
        ),
        "gate_verdict": gates_verdict,
        "overall_verdict": overall_verdict,
        "deploy_candidate": deploy_candidate,
    }
    (root_out / "verdict.json").write_text(json.dumps(summary, indent=2, default=str))

    # Flat CSV
    rows = []
    for i, (spf, epf, rule) in enumerate(
        zip(solo_pf_per_fold, ens_pf_per_fold, agg_rule_per_fold)
    ):
        row = {
            "fold": i,
            "chosen_rule": rule,
            "ens_pf": epf,
            **{f"solo_{s}_pf": spf.get(s) if spf else None for s in seeds},
        }
        if spf:
            row["best_solo_pf"] = max(spf.values())
            row["uplift"] = epf / row["best_solo_pf"] if row["best_solo_pf"] and epf else None
        rows.append(row)
    pd.DataFrame(rows).to_csv(root_out / "all_folds_summary.csv", index=False)

    print("\n========== GMGP1 OANDA WF STAGE 2.5 VERDICT ==========")
    print(json.dumps(summary["gate_verdict"], indent=2, default=str))
    print(f"\nOVERALL: {overall_verdict}  median_uplift={median_uplift}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--wf_config",
        default="configs/gmgp1_xauusd_ftmo_rehpo_wf_multiseed_oanda.yaml",
    )
    ap.add_argument(
        "--gates_file",
        default="configs/gmgp1_xauusd_ensemble.gates.yaml",
    )
    ap.add_argument(
        "--output_dir",
        default="results/gmgp1_xauusd_oanda_wf_ensemble",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only_fold", type=int, default=None, help="Run a single fold (0..7)")
    args = ap.parse_args()

    run(
        wf_config_path=args.wf_config,
        gates_path=args.gates_file,
        output_dir=args.output_dir,
        device=args.device,
        only_fold=args.only_fold,
    )


if __name__ == "__main__":
    main()
