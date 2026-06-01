#!/usr/bin/env python3
"""GMGP1-XAUUSD OANDA WF fold-7 Stage 2.5 ensemble-confirm (single-fold).

Protocol v2.1/v2.2 Stage 2.5 val-split rule selection (S495), scoped to the
freshest WF fold (fold 7, the paper-deploy candidate). This is a SINGLE-FOLD
application of the multi-fold WF-gate framework in
`scripts/sg1_xauusd_ensemble_eval.py` (run_wf_ensemble) — strict 8/8 fold
clauses in the gates YAML collapse to 1/1 (pass iff the single fold passes).

Does the minimum work to complete a half-done eval:
  * Reads existing val metric JSONs (solos + ens_mean + ens_median) from
    `results/gmgp1_xauusd_oanda_wf_ensemble/fold_07/val/`.
  * Runs `ens_agreement` + `ens_pf_weighted` on val if their metric JSONs
    are missing.
  * Picks the val-argmax-PF winner across the 7 candidate rules
    (solos + 4 ensembles).
  * Runs the chosen rule + 3 solos + the other 3 ensembles on the fold-7
    test window (so all-4 ensembles + 3 solos test trajectories exist,
    per caller brief).
  * Applies the S495 val-argmax-PF decision:
        uplift = test_PF(chosen) / max(test_PF(solos))       if chosen ∈ ensembles
        uplift = 1.0                                         if chosen ∈ solos
        >= ensemble_uplift_min         -> PROMOTE
        >= ensemble_ambiguous_min      -> AMBIGUOUS_RERUN
        else                           -> SOLO_BEST
  * Evaluates G1..G5 from `configs/gmgp1_xauusd_ensemble.gates.yaml` in
    single-fold semantics (G4 strict 1/1).
  * Writes `verdict.json` next to val/ and test/.

CPU-only by default (`--device cpu`). Set `CUDA_VISIBLE_DEVICES=""` if a
GPU is visible but should not be used.
"""
from __future__ import annotations
import argparse
import copy
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts import sg1_xauusd_ensemble_eval as sg1  # noqa: E402
from scripts.sg1_arm_gate_backtest import compute_gate_metrics  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gmgp1-oanda-wf-fold-07-eval")


# Fold 7 dates derived from the RollingWindowSplitter on start_date=2025-01-06
# with train/val/test = 6/1/1 months, step=1 month (per
# configs/gmgp1_xauusd_ftmo_rehpo_wf_multiseed_oanda.yaml).
FOLD_DATES = {
    "train_start_date": "2025-08-06",
    "train_end_date":   "2026-02-06",
    "val_start_date":   "2026-02-06",
    "val_end_date":     "2026-03-06",
    "test_start_date":  "2026-03-06",
    "test_end_date":    "2026-04-06",
}

SEEDS: List[int] = [42, 3141, 2025]

# L1-OANDA seed PFs (from configs/gmgp1_xauusd_ftmo_rehpo_wf_multiseed_oanda.yaml
# ensemble.seed_pfs) — weights for ens_pf_weighted.
SEED_PFS: Dict[int, float] = {42: 2.389, 3141: 2.261, 2025: 2.182}

CHECKPOINT_PATTERN = "checkpoints/WF_seed{seed}_fold_07_*/checkpoint_final.pth"


def _resolve_fold7_checkpoints() -> Dict[int, str]:
    out: Dict[int, str] = {}
    for s in SEEDS:
        matches = sorted(glob.glob(CHECKPOINT_PATTERN.format(seed=s)))
        if not matches:
            raise FileNotFoundError(
                f"seed {s}: no checkpoint match for {CHECKPOINT_PATTERN.format(seed=s)}"
            )
        out[s] = matches[-1]  # latest lexicographically
        log.info(f"seed {s}: {out[s]}")
    return out


def _override_window(config: dict, start: str, end: str, norm_cutoff: str) -> dict:
    c = copy.deepcopy(config)
    c.setdefault("data", {})
    c["data"]["test_start_date"] = start[:10]
    c["data"]["test_end_date"] = end[:10]
    c["data"]["val_end_date"] = norm_cutoff[:10]
    return c


def _ensemble_rules_factory(seed_pfs: Dict[int, float]) -> List[Tuple[str, Callable]]:
    return [
        ("ens_mean",        sg1._agg_mean),
        ("ens_median",      sg1._agg_median),
        ("ens_agreement",   sg1._agg_agreement),
        ("ens_pf_weighted", sg1._make_pf_weighted(seed_pfs)),
    ]


def _solo_rules(seeds: List[int]) -> List[Tuple[str, Callable]]:
    return [(f"solo_{s}", sg1._agg_solo(s)) for s in seeds]


def _load_or_run_val(
    config: dict,
    agents: Dict[int, object],
    rule_name: str,
    rule_fn: Callable,
    val_dir: Path,
    device: str,
) -> dict:
    """Read metrics JSON if it exists; else run the rule on val and persist it."""
    m_path = val_dir / f"{rule_name}_metrics.json"
    if m_path.exists():
        log.info(f"[val/{rule_name}] reusing existing {m_path.name}")
        return json.loads(m_path.read_text())
    log.info(f"[val/{rule_name}] missing — running val eval")
    df = sg1.run_rule(config, agents, rule_name, rule_fn, device, val_dir)
    m = compute_gate_metrics(df, rule_name)
    m.update(sg1.ftmo_buffers(m))
    m_path.write_text(json.dumps(m, indent=2, default=str))
    return m


def _run_test(
    config: dict,
    agents: Dict[int, object],
    rule_name: str,
    rule_fn: Callable,
    test_dir: Path,
    device: str,
) -> Tuple[dict, pd.DataFrame]:
    m_path = test_dir / f"{rule_name}_metrics.json"
    traj_path = test_dir / f"{rule_name}_trajectory.parquet"
    if m_path.exists() and traj_path.exists():
        log.info(f"[test/{rule_name}] reusing existing outputs")
        return json.loads(m_path.read_text()), pd.read_parquet(traj_path)
    log.info(f"[test/{rule_name}] running")
    df = sg1.run_rule(config, agents, rule_name, rule_fn, device, test_dir)
    m = compute_gate_metrics(df, rule_name)
    m.update(sg1.ftmo_buffers(m))
    m_path.write_text(json.dumps(m, indent=2, default=str))
    return m, df


def _evaluate_gates_single_fold(
    val_metrics: Dict[str, dict],
    test_metrics: Dict[str, dict],
    chosen_rule: str,
    gates_cfg: dict,
) -> dict:
    """Single-fold application of gates G1..G5.

    Strict 8/8 clauses collapse to 1/1 (the single fold must pass).
    G3 uplift is the val-argmax-chosen-rule test_PF over best solo test_PF
    (median-of-1 == single value).
    """
    gates = gates_cfg.get("gates") or {}
    out: dict = {}

    # G1 solo baseline: per-fold >= seeds_pass_min of {42,3141,2025} satisfy
    # pf_floor. In single-fold semantics: this fold must pass.
    g1 = gates["g1_solo_baseline"]
    pf_floor = float(g1["pf_floor"])
    seeds_pass_min = int(g1["seeds_pass_min"])
    g1_passes = sum(
        1 for s in SEEDS
        if (test_metrics.get(f"solo_{s}", {}).get("pf_bar") or 0.0) >= pf_floor
    )
    out["G1_solo_baseline"] = {
        "pass": g1_passes >= seeds_pass_min,
        "folds_meeting": 1 if g1_passes >= seeds_pass_min else 0,
        "min_required": 1,
        "seeds_meeting_in_fold": g1_passes,
        "seeds_pass_min": seeds_pass_min,
        "pf_floor": pf_floor,
    }

    # G2 no regression: per-fold ens_PF >= ens_ratio_min * best_solo_PF.
    g2 = gates["g2_no_regression"]
    ens_ratio_min = float(g2["ens_ratio_min"])
    solo_test_pfs = {s: test_metrics[f"solo_{s}"]["pf_bar"] for s in SEEDS}
    best_solo_pf = max(solo_test_pfs.values())
    ens_for_g2 = test_metrics.get(chosen_rule, {}).get("pf_bar")
    if chosen_rule.startswith("solo_"):
        # No ensemble regression to evaluate — by definition uplift == 1.0 and
        # no ensemble was picked. Treat G2 as PASS (nothing to regress).
        g2_pass = True
        g2_ratio = None
    else:
        g2_ratio = (ens_for_g2 / best_solo_pf) if best_solo_pf > 0 else 0.0
        g2_pass = g2_ratio >= ens_ratio_min
    out["G2_no_regression"] = {
        "pass": g2_pass,
        "folds_meeting": 1 if g2_pass else 0,
        "min_required": 1,
        "chosen_rule": chosen_rule,
        "chosen_rule_test_pf": ens_for_g2,
        "best_solo_test_pf": best_solo_pf,
        "ratio": g2_ratio,
        "ratio_min": ens_ratio_min,
    }

    # G3 uplift: median-of-1 == single value.
    g3 = gates["g3_uplift"]
    uplift_min = float(g3["uplift_ratio_min"])
    if chosen_rule.startswith("solo_"):
        uplift = 1.0
    else:
        uplift = (ens_for_g2 / best_solo_pf) if best_solo_pf > 0 else 0.0
    out["G3_uplift"] = {
        "pass": uplift >= uplift_min,
        "median_ens_pf": ens_for_g2,
        "median_best_solo_pf": best_solo_pf,
        "uplift": uplift,
        "uplift_min": uplift_min,
        "note": "single-fold: median-of-1 == point value",
    }

    # G4 FTMO compliance on chosen (ensemble or solo) trajectory.
    g4 = gates["g4_ftmo_compliance"]
    dd_buf_min = float(g4["daily_dd_buffer_pp_min"])
    tr_buf_min = float(g4["trailing_dd_buffer_pp_min"])
    em = test_metrics.get(chosen_rule, {})
    dd_buf = em.get("daily_dd_buffer_pp")
    tr_buf = em.get("trailing_dd_buffer_pp")
    comp = em.get("ftmo_compliance_pass")
    g4_pass = (
        dd_buf is not None and tr_buf is not None
        and dd_buf >= dd_buf_min and tr_buf >= tr_buf_min and bool(comp)
    )
    out["G4_ftmo_compliance"] = {
        "pass": g4_pass,
        "folds_meeting": 1 if g4_pass else 0,
        "min_required": 1,
        "chosen_rule": chosen_rule,
        "daily_dd_buffer_pp": dd_buf,
        "trailing_dd_buffer_pp": tr_buf,
        "compliance_pass": comp,
        "daily_dd_buffer_pp_min": dd_buf_min,
        "trailing_dd_buffer_pp_min": tr_buf_min,
    }

    # G5 DD stability: CV of ensemble PF across folds. Undefined on 1 fold.
    g5 = gates["g5_dd_stability"]
    out["G5_dd_stability"] = {
        "pass": True,
        "cv_pf": None,
        "cv_max": float(g5["cv_pf_max"]),
        "note": (
            "single-fold Stage 2.5 scope: cross-fold CV undefined; "
            "evaluated in the full 8-fold WF run (separate artifact)."
        ),
    }
    return out


def main() -> None:
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
        "--out_dir",
        default="results/gmgp1_xauusd_oanda_wf_ensemble/fold_07",
    )
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    with open(args.wf_config, encoding="utf-8") as f:
        wf_cfg = yaml.safe_load(f)
    with open(args.gates_file, encoding="utf-8") as f:
        gates_cfg = yaml.safe_load(f)

    # Inject fold-7 dates so _prep_backtest_config sees them. val/test configs
    # then override test_start/end_date + val_end_date (= norm cutoff).
    wf_cfg.setdefault("data", {})
    for k, v in FOLD_DATES.items():
        wf_cfg["data"][k] = v

    out_dir = Path(args.out_dir)
    val_dir = out_dir / "val"
    test_dir = out_dir / "test"
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Resolve + load agents (needed for any missing val or test rule).
    ckpt_paths = _resolve_fold7_checkpoints()
    agents = sg1._load_agents_from_paths(wf_cfg, ckpt_paths, args.device)

    solo_rules = _solo_rules(SEEDS)
    ens_rules = _ensemble_rules_factory(SEED_PFS)

    # --- Phase 1: val — complete missing rules, reuse existing ones ---
    val_cfg = _override_window(
        wf_cfg,
        FOLD_DATES["val_start_date"],
        FOLD_DATES["val_end_date"],
        norm_cutoff=FOLD_DATES["train_end_date"],
    )
    val_metrics: Dict[str, dict] = {}
    for rname, rfn in solo_rules + ens_rules:
        val_metrics[rname] = _load_or_run_val(val_cfg, agents, rname, rfn, val_dir, args.device)

    # --- Phase 2: pick chosen rule by val-argmax-PF over all 7 rules ---
    all_rule_names = [r[0] for r in solo_rules + ens_rules]
    val_pf_by_rule = {r: float(val_metrics[r]["pf_bar"]) for r in all_rule_names}
    chosen_rule = max(val_pf_by_rule, key=lambda r: val_pf_by_rule[r])
    log.info(f"[val-select] val PFs = {val_pf_by_rule}")
    log.info(f"[val-select] chosen_rule = {chosen_rule}  "
             f"(val PF {val_pf_by_rule[chosen_rule]:.4f})")

    # --- Phase 3: run all 4 ensembles + 3 solos on test ---
    test_cfg = _override_window(
        wf_cfg,
        FOLD_DATES["test_start_date"],
        FOLD_DATES["test_end_date"],
        norm_cutoff=FOLD_DATES["val_end_date"],
    )
    test_metrics: Dict[str, dict] = {}
    test_trajs: Dict[str, pd.DataFrame] = {}
    for rname, rfn in solo_rules + ens_rules:
        m, df = _run_test(test_cfg, agents, rname, rfn, test_dir, args.device)
        test_metrics[rname] = m
        test_trajs[rname] = df

    # --- Phase 4: S495 uplift + gates ---
    gates_block = gates_cfg.get("gates") or {}
    uplift_promote = float(gates_block.get("ensemble_uplift_min", 1.10))
    uplift_ambiguous = float(gates_block.get("ensemble_ambiguous_min", 1.05))

    solo_test_pfs = {s: float(test_metrics[f"solo_{s}"]["pf_bar"]) for s in SEEDS}
    best_solo_seed = max(solo_test_pfs, key=lambda s: solo_test_pfs[s])
    best_solo_test_pf = solo_test_pfs[best_solo_seed]
    chosen_test_pf = float(test_metrics[chosen_rule]["pf_bar"])

    if chosen_rule.startswith("solo_"):
        uplift = 1.0
        decision_source = "val_argmax_picked_solo"
    else:
        uplift = (chosen_test_pf / best_solo_test_pf) if best_solo_test_pf > 0 else 0.0
        decision_source = "val_argmax_picked_ensemble"

    if chosen_rule.startswith("solo_"):
        decision = "SOLO_BEST"
        reason = (
            f"val-argmax chose {chosen_rule} (val PF {val_pf_by_rule[chosen_rule]:.4f}); "
            f"no ensemble promotion — test uplift = 1.0 by definition."
        )
    elif uplift >= uplift_promote:
        decision = "PROMOTE"
        reason = (
            f"ensemble {chosen_rule} test uplift {uplift:.4f}x >= "
            f"{uplift_promote:.2f} PROMOTE threshold"
        )
    elif uplift >= uplift_ambiguous:
        decision = "AMBIGUOUS_RERUN"
        reason = (
            f"ensemble {chosen_rule} test uplift {uplift:.4f}x in "
            f"[{uplift_ambiguous:.2f}, {uplift_promote:.2f}) ambiguous band"
        )
    else:
        decision = "SOLO_BEST"
        reason = (
            f"ensemble {chosen_rule} test uplift {uplift:.4f}x < "
            f"{uplift_ambiguous:.2f} ambiguous floor — fall back to best solo"
        )

    gates_result = _evaluate_gates_single_fold(val_metrics, test_metrics, chosen_rule, gates_cfg)

    # deploy_candidate
    if decision == "PROMOTE" and all(g["pass"] for g in gates_result.values()):
        deploy_candidate = {
            "type": "ensemble",
            "rule": chosen_rule,
            "seeds": SEEDS,
            "checkpoints": ckpt_paths,
        }
    else:
        deploy_candidate = {
            "type": "solo",
            "seed": int(best_solo_seed),
            "checkpoint": ckpt_paths[best_solo_seed],
            "test_pf": best_solo_test_pf,
        }

    verdict = {
        "workstream": "gmgp1_xauusd_oanda_wf_fold_07",
        "protocol": "v2.1_v2.2_stage_2_5_val_selection_single_fold",
        "scope_note": (
            "Single-fold Stage 2.5 on the paper-deploy candidate (fold 7). "
            "Strict 8/8 WF gate clauses collapse to 1/1; G5 cross-fold CV "
            "evaluated in the full 8-fold WF run (separate artifact)."
        ),
        "fold_index": 7,
        "val_window": f"{FOLD_DATES['val_start_date']} -> {FOLD_DATES['val_end_date']}",
        "test_window": f"{FOLD_DATES['test_start_date']} -> {FOLD_DATES['test_end_date']}",
        "seeds": SEEDS,
        "checkpoints": ckpt_paths,
        "val_pf_by_rule": val_pf_by_rule,
        "chosen_rule": chosen_rule,
        "chosen_rule_val_pf": val_pf_by_rule[chosen_rule],
        "chosen_rule_test_pf": chosen_test_pf,
        "test_solo_pfs": solo_test_pfs,
        "best_solo_seed_on_test": int(best_solo_seed),
        "best_solo_test_pf": best_solo_test_pf,
        "test_ensemble_pfs": {
            r[0]: float(test_metrics[r[0]]["pf_bar"]) for r in ens_rules
        },
        "uplift": uplift,
        "uplift_promote_threshold": uplift_promote,
        "uplift_ambiguous_threshold": uplift_ambiguous,
        "decision": decision,
        "decision_source": decision_source,
        "reason": reason,
        "gates": gates_result,
        "gates_overall": all(g["pass"] for g in gates_result.values()),
        "deploy_candidate": deploy_candidate,
    }

    verdict_path = out_dir / "verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2, default=str))
    log.info(f"wrote {verdict_path}")

    # Flat summary CSV for quick audit (val + test side-by-side).
    rows = []
    for rname in all_rule_names:
        v = val_metrics.get(rname, {})
        t = test_metrics.get(rname, {})
        rows.append({
            "rule": rname,
            "val_pf": v.get("pf_bar"),
            "val_return_pct": v.get("total_return_pct"),
            "val_trail_dd_pct": v.get("trailing_max_drawdown_pct"),
            "test_pf": t.get("pf_bar"),
            "test_return_pct": t.get("total_return_pct"),
            "test_trail_dd_pct": t.get("trailing_max_drawdown_pct"),
            "test_trail_buf_pp": t.get("trailing_dd_buffer_pp"),
            "test_daily_buf_pp": t.get("daily_dd_buffer_pp"),
            "test_ftmo_ok": t.get("ftmo_compliance_pass"),
        })
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)

    print("\n========== GMGP1-XAUUSD OANDA WF fold 7 Stage 2.5 VERDICT ==========")
    print(json.dumps({
        "val_pf_by_rule": val_pf_by_rule,
        "chosen_rule": chosen_rule,
        "chosen_rule_test_pf": chosen_test_pf,
        "test_solo_pfs": solo_test_pfs,
        "uplift": uplift,
        "decision": decision,
        "gates": {k: v["pass"] for k, v in gates_result.items()},
        "deploy_candidate": deploy_candidate,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
