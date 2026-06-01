#!/usr/bin/env python3
"""Bake ensemble_v1.tar.gz for GMGP1-XAUUSD extended-window (Tier B2).

Stage 3 WF 8-fold landed PROMOTE on bootstrap PRIMARY (P(PF)=1.000,
P(MDD)=0.986). Legacy G3 uplift FAIL by 0.0015 (1.0985 < 1.10) — audit-only per
Protocol v2.5. Bundle wraps fold_07 checkpoints (training data through
2026-02-28, freshest fold) for top-3 seeds [789, 2025, 1024].

Predecessor: solo seed-42 fold-07 OANDA L1 (live since 2026-04-24); this is
the first ensemble graduation for gmgp1-xauusd — requires `swap_approved`
sentinel pre-touch per S538-cont-4 (gmgp1-gold ensemble v1 LIVE) lesson.

Output: results/gmgp1_xauusd_extended_wf_ensemble/ensemble_v1.tar.gz
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.sg1_xauusd_ensemble_eval import write_ensemble_swap_bundle  # noqa: E402


WORKSTREAM = "gmgp1_xauusd_extended"
VERSION = "v1"
CHOSEN_RULE = "ens_pf_weighted"
SEEDS = [789, 2025, 1024]
TRIGGER = "extended_window_wf_8fold_promote_bootstrap_primary"

CKPT_PATTERN = str(
    PROJECT_ROOT
    / "checkpoints"
    / "gmgp1-xauusd-ext-wf-fold7-seed{seed}_20260527_224418"
    / "checkpoint_final.pth"
)

WF_CONFIG_PATH = PROJECT_ROOT / "configs" / "gmgp1_xauusd_wf_extended.yaml"
GATES_PATH = PROJECT_ROOT / "configs" / "gmgp1_xauusd_extended_ensemble.gates.yaml"
RESULTS_DIR = PROJECT_ROOT / "results" / "gmgp1_xauusd_extended_wf_ensemble"
VERDICT_PATH = RESULTS_DIR / "verdict.json"


def build_resolved_config() -> dict:
    """Materialize fold_07 test window into the WF config for live-engine consumption."""
    with WF_CONFIG_PATH.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = deepcopy(cfg)
    # fold_07: train 2024-08-01 -> 2026-02-01, val 2026-02-01 -> 03-01,
    #          test 2026-03-01 -> 04-01
    cfg.setdefault("data", {})
    cfg["data"]["train_start_date"] = "2024-08-01"
    cfg["data"]["train_end_date"] = "2026-01-31"
    cfg["data"]["val_start_date"] = "2026-02-01"
    cfg["data"]["val_end_date"] = "2026-02-28"
    cfg["data"]["test_start_date"] = "2026-03-01"
    cfg["data"]["test_end_date"] = "2026-03-31"
    cfg["chosen_rule"] = CHOSEN_RULE
    cfg.setdefault("ensemble", {})
    cfg["ensemble"]["chosen_rule"] = CHOSEN_RULE
    cfg["ensemble"]["seeds"] = SEEDS
    cfg["provenance"] = {
        "source": "Tier B2 extended-window 8-fold WF (fold_07 checkpoints)",
        "wf_config": "configs/gmgp1_xauusd_wf_extended.yaml",
        "gates_file": "configs/gmgp1_xauusd_extended_ensemble.gates.yaml",
        "predecessor": "solo seed-42 fold-07 OANDA L1 (live 2026-04-24)",
        "graduation": "first ensemble for gmgp1-xauusd — swap_approved sentinel required",
        "training_cutoff": "2026-02-28",
    }
    return cfg


def main() -> None:
    cfg = build_resolved_config()

    seed_ckpts = {s: CKPT_PATTERN.format(seed=s) for s in SEEDS}
    for s, p in seed_ckpts.items():
        if not Path(p).exists():
            raise FileNotFoundError(f"seed {s}: {p} missing")

    with VERDICT_PATH.open() as f:
        verdict = json.load(f)
    wf_bs = verdict.get("wf_bootstrap")
    if not wf_bs:
        raise RuntimeError(
            f"verdict.json at {VERDICT_PATH} has no wf_bootstrap block — "
            f"run scripts/wf_bootstrap_fu3.py first."
        )
    bs = wf_bs["bootstrap"]
    bootstrap_verdict = {
        "p_pf_ens_better": bs["p_pf_ens_better"],
        "p_mdd_ens_better": bs["p_mdd_ens_better"],
        "n_resamples": bs["n_resamples"],
        "block_len_mean": bs["block_len_mean"],
        "n_bars": bs["n_bars"],
        "ens_pf_quantiles": bs["ens_pf_quantiles"],
        "solo_pf_quantiles": bs["solo_pf_quantiles"],
        "ens_mdd_quantiles": bs["ens_mdd_quantiles"],
        "solo_mdd_quantiles": bs["solo_mdd_quantiles"],
        "ens_rule_paired": wf_bs["ens_rule"],
        "solo_seed_paired": wf_bs["solo_seed_paired"],
    }

    diversity_audit = {
        "selected_seeds": SEEDS,
        "naive_top_k": SEEDS,
        "differs_from_naive": False,
        "diversity_lambda": 1.0,
        "pool_size": len(SEEDS),
        "k": 3,
        "note_pool_too_small": "diversity-audit-only: pool size == K, no reselection possible",
        "note_inherited_from_l1": (
            "Top-3 seed selection inherited from L1 N=10 extended val-argmax-PF "
            "(seeds 789/2025/1024). Stage 3 WF confirmed temporal robustness "
            "(G1/G2/G4/G5 PASS, G3 legacy uplift 1.099 < 1.10 = noise-blind point gate; "
            "bootstrap PRIMARY P(PF)=1.000, P(MDD)=0.986 PROMOTE)."
        ),
    }

    bundle_path = RESULTS_DIR / f"ensemble_{VERSION}.tar.gz"
    manifest = write_ensemble_swap_bundle(
        bundle_path=bundle_path,
        workstream=WORKSTREAM,
        version=VERSION,
        chosen_rule=CHOSEN_RULE,
        selected_seeds=SEEDS,
        seed_checkpoints=seed_ckpts,
        config=cfg,
        diversity_audit=diversity_audit,
        bootstrap_verdict=bootstrap_verdict,
        decision="PROMOTE",
        predecessor_version=None,
        trigger=TRIGGER,
    )

    print(f"\nWrote {bundle_path}")
    print(f"SHA256 sidecar: {bundle_path}.sha256")
    print("\nManifest summary:")
    print(json.dumps({k: v for k, v in manifest.items() if k != "diversity_audit"},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
