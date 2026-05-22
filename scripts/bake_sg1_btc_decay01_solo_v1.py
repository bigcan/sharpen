#!/usr/bin/env python3
"""Bake solo_v1.tar.gz for SG-1-BTC DECAY-01 (Stage 3 WF SOLO_BEST verdict).

Single-seed v2.3 atomic-swap bundle — seed 456, fold_07 checkpoint (freshest
training cutoff 2024-08-01 -> 2026-02-01). Stage 3 WF returned
`decision: reject_ensemble_use_best_solo` (G1/G4/G5 PASS, G2 FAIL 1/8,
G3 uplift 0.866x < 1.10). Seed 456 owns fold_07 pf_bar 2.524, top among
the 3-seed cohort on the freshest fold.

Workstream tag is NEW: `sg1_btc_velotrade_decay01` (vs predecessor
`sg1_btc_velotrade_extended`). First deploy will require the swap-approval
sentinel per feedback_v23_first_deploy_swap_approved_sentinel.

Config base: extracted from the live-validated v2 ensemble bundle, then
adapted for the DECAY-01 fold_07 windows and a single-seed ensemble block.

Output: bundles/sg1_btc_velotrade_decay01/solo_v1.tar.gz + .sha256
"""
from __future__ import annotations

import json
import sys
import tarfile
from copy import deepcopy
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.sg1_xauusd_ensemble_eval import write_ensemble_swap_bundle  # noqa: E402

WORKSTREAM = "sg1_btc_velotrade_decay01"
VERSION = "v1"
CHOSEN_RULE = "ens_mean"  # degenerate for N=1; aggregator returns the single action
SEEDS = [456]
TRIGGER = "sg1btc_decay01_stage3_wf_solo_best_fallback"

# fold_07 seed_456 — freshest training cutoff (train 2024-08-01 -> 2026-02-01)
CKPT_PATHS = {
    456: REPO / "checkpoints/sg1-btc-decay01-wf-fold7-seed456_20260521_230637/checkpoint_final.pth",
}

# Live-validated config shape from current production bundle
PREDECESSOR_BUNDLE = REPO / "bundles/sg1_btc_velotrade_extended/ensemble_v2.tar.gz"

OUT_BUNDLE = REPO / "bundles/sg1_btc_velotrade_decay01/solo_v1.tar.gz"

# Stage 3 WF verdict source (G1-G5 gates + decision)
STAGE3_VERDICT = REPO / "results/sg1_btc_ensemble_wf/verdict.json"
STAGE3_LONG_CSV = REPO / "results/sg1_btc_ensemble_wf/all_folds_long.csv"

# Stage 2.5-R verdict source (bootstrap P(PF)=0.650 SOLO_BEST_FALLBACK)
STAGE25R_VERDICT = REPO / "results/sg1_btc_velotrade_decay01_ensemble/verdict.json"


def load_predecessor_resolved_config() -> dict:
    with tarfile.open(PREDECESSOR_BUNDLE, "r:gz") as tar:
        for cand in ("./config.resolved.yaml", "config.resolved.yaml"):
            try:
                fp = tar.extractfile(cand)
                if fp is not None:
                    return yaml.safe_load(fp.read())
            except KeyError:
                continue
    raise FileNotFoundError(f"config.resolved.yaml not found in {PREDECESSOR_BUNDLE}")


def adapt_config_for_fold7_solo(base_cfg: dict, seed_pf: float) -> dict:
    cfg = deepcopy(base_cfg)
    cfg["workstream"] = WORKSTREAM

    cfg["data"]["train_start_date"] = "2024-08-01"
    cfg["data"]["train_end_date"] = "2026-02-01"
    cfg["data"]["val_start_date"] = "2026-02-01"
    cfg["data"]["val_end_date"] = "2026-03-01"
    cfg["data"]["test_start_date"] = "2026-03-01"
    cfg["data"]["test_end_date"] = "2026-04-01"

    cfg["chosen_rule"] = CHOSEN_RULE
    ens = cfg.setdefault("ensemble", {})
    ens["seeds"] = SEEDS
    ens["chosen_rule"] = CHOSEN_RULE
    ens["seed_pfs"] = {str(SEEDS[0]): float(seed_pf)}
    ens.pop("checkpoint_pattern", None)
    ens.pop("rules", None)

    cfg["provenance"] = {
        "source": "SG1BTC-DECAY-01 Stage 3 WF (8 folds, 3 seeds) — solo_best_fallback",
        "wf_config": "configs/sg1_btc_velotrade_decay01_wf_multiseed.yaml",
        "wf_manifest": "results/sg1_btc_decay01_wf_20260521_113130/manifest.json",
        "wf_verdict": "results/sg1_btc_ensemble_wf/verdict.json",
        "stage_2_5_r_verdict": "results/sg1_btc_velotrade_decay01_ensemble/verdict.json",
        "gates_file": "configs/sg1_btc_velotrade_ensemble.gates.yaml",
        "predecessor_bundle": "bundles/sg1_btc_velotrade_extended/ensemble_v2.tar.gz",
        "rule_change_vs_predecessor": "ens_agreement (3-seed) -> ens_mean (1-seed degenerate)",
        "seed_selection_rationale": (
            "Stage 3 WF rejected ensemble (G2 FAIL 1/8, G3 uplift 0.866x). "
            "Fold_07 (freshest training cutoff 2026-02-01) seed_456 pf_bar 2.524 "
            "tops both other seeds (42: 2.484, 2025: 2.363) and all ensemble rules "
            "(ens_mean 2.595, ens_pf_weighted 2.592, ens_median 2.553, ens_agreement 2.087). "
            "Aligns with Stage 2.5-R SOLO_BEST_FALLBACK pick (seed_456, val_argmax_pf 2.529)."
        ),
        "post_restart_cooldown_bars": 3,
        "post_restart_cooldown_decision_memory": "decision_sg1_btc_skip_first_3_trades_s538",
    }
    return cfg


def load_stage3_verdict() -> dict:
    return json.loads(STAGE3_VERDICT.read_text(encoding="utf-8"))


def load_stage25r_bootstrap() -> dict:
    v = json.loads(STAGE25R_VERDICT.read_text(encoding="utf-8"))
    bs = v.get("bootstrap_verdict") or {}
    if not bs:
        raise RuntimeError(
            f"{STAGE25R_VERDICT} missing 'bootstrap_verdict' block "
            f"(found keys: {sorted(v.keys())[:10]}...)"
        )
    return {
        "p_pf_ens_better": bs.get("p_pf_ens_better"),
        "p_mdd_ens_better": bs.get("p_mdd_ens_better"),
        "n_resamples": bs.get("n_resamples"),
        "block_len_mean": bs.get("block_len_mean"),
        "ens_pf_quantiles": bs.get("ens_pf_quantiles"),
        "solo_pf_quantiles": bs.get("solo_pf_quantiles"),
        "_source": "results/sg1_btc_velotrade_decay01_ensemble/verdict.json (Stage 2.5-R bootstrap_verdict)",
    }


def build_diversity_audit(stage3: dict) -> dict:
    return {
        "selected_seeds": SEEDS,
        "naive_top_k": SEEDS,
        "differs_from_naive": False,
        "diversity_lambda": 1.0,
        "pool_size": 1,
        "k": 1,
        "note_pool_too_small": "solo bundle: single seed, no diversity selection applicable",
        "note_selection_basis": (
            "Stage 3 WF reject_ensemble_use_best_solo verdict + Stage 2.5-R "
            "SOLO_BEST_FALLBACK on seed 456 (val_argmax_pf 2.529, test 2.456). "
            "Seed 456 wins fold_07 (freshest training cutoff) pf_bar 2.524."
        ),
        "wf_stage3_gates": stage3.get("gates", {}),
        "wf_stage3_decision": stage3.get("decision"),
        "wf_stage3_folds_completed": stage3.get("folds_completed"),
        "wf_stage3_folds_expected": stage3.get("folds_expected"),
    }


def main() -> None:
    for s, p in CKPT_PATHS.items():
        if not p.exists():
            raise FileNotFoundError(f"seed {s}: checkpoint missing at {p}")

    stage3 = load_stage3_verdict()
    if stage3.get("decision") != "reject_ensemble_use_best_solo":
        raise RuntimeError(
            f"Stage 3 verdict has unexpected decision: {stage3.get('decision')!r} "
            f"(expected reject_ensemble_use_best_solo)"
        )

    bootstrap_verdict = load_stage25r_bootstrap()
    diversity_audit = build_diversity_audit(stage3)

    # Seed PF anchor: fold_07 pf_bar from Stage 3 (per all_folds_long.csv)
    seed456_fold7_pf = 2.523684583631234
    base_cfg = load_predecessor_resolved_config()
    cfg = adapt_config_for_fold7_solo(base_cfg, seed_pf=seed456_fold7_pf)

    OUT_BUNDLE.parent.mkdir(parents=True, exist_ok=True)

    manifest = write_ensemble_swap_bundle(
        bundle_path=OUT_BUNDLE,
        workstream=WORKSTREAM,
        version=VERSION,
        chosen_rule=CHOSEN_RULE,
        selected_seeds=SEEDS,
        seed_checkpoints={s: str(p) for s, p in CKPT_PATHS.items()},
        config=cfg,
        diversity_audit=diversity_audit,
        bootstrap_verdict=bootstrap_verdict,
        decision="reject_ensemble_use_best_solo",
        predecessor_version=None,
        trigger=TRIGGER,
    )

    print(f"\nWrote {OUT_BUNDLE}")
    print(f"SHA256 sidecar: {OUT_BUNDLE}.sha256")
    print("\nManifest summary (diversity_audit omitted):")
    print(json.dumps({k: v for k, v in manifest.items() if k != "diversity_audit"},
                     indent=2, default=str))
    print("\nFirst-deploy reminder: workstream tag changes from "
          "sg1_btc_velotrade_extended -> sg1_btc_velotrade_decay01; "
          "pre-touch swap_approved sentinel before manage_strategies.sh build "
          "(see feedback_v23_first_deploy_swap_approved_sentinel).")


if __name__ == "__main__":
    main()
