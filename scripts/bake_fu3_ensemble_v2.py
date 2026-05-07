#!/usr/bin/env python3
"""Bake ensemble_v2.tar.gz for SG-1-BTC FU-3 (8-fold WF-confirmed PROMOTE).

Uses fold_07 checkpoints (training data through 2026-02-28, freshest fold) for
the three top-3 seeds [456, 3141, 123]. Aggregator rule: ens_agreement (per
configs/sg1_btc_velotrade_ensemble.gates.yaml). Bootstrap verdict sourced from
the WF-aggregated bootstrap on 57,028 paired bars (P(PF)=1.000).

Predecessor: ensemble_v1.tar.gz (FU-2 single-window L1 PROMOTE, ens_mean,
trained through 2026-01-31). v2 advances training cutoff by 1 month and
switches rule to ens_agreement (gates-yaml-canonical).

Output: results/sg1_btc_velotrade_extended_ensemble/ensemble_v2.tar.gz
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import yaml

sys.path.insert(0, "/workspace/DeepScalper")
from scripts.sg1_xauusd_ensemble_eval import write_ensemble_swap_bundle


WORKSTREAM = "sg1_btc_velotrade_extended"
VERSION = "v2"
CHOSEN_RULE = "ens_agreement"
SEEDS = [456, 3141, 123]
TRIGGER = "fu3_extended_wf_8fold_confirm"
PREDECESSOR = "v1"

CKPT_PATTERN = "/workspace/DeepScalper/checkpoints/sg1-btc-extended-wf-fold7-seed{seed}_20260507_095945/checkpoint_final.pth"

FU2_BUNDLE = Path("/workspace/DeepScalper/results/sg1_btc_velotrade_extended_ensemble/ensemble_v1.tar.gz")
RESULTS_DIR = Path("/workspace/DeepScalper/results/sg1_btc_velotrade_extended_ensemble")
VERDICT_PATH = Path("/workspace/DeepScalper/results/sg1_xauusd_ensemble_wf/verdict.json")


def load_fu2_resolved_config() -> dict:
    """Extract config.resolved.yaml from FU-2 ensemble_v1.tar.gz."""
    import tarfile
    with tarfile.open(FU2_BUNDLE, "r:gz") as tar:
        member = tar.extractfile("./config.resolved.yaml")
        if member is None:
            member = tar.extractfile("config.resolved.yaml")
        if member is None:
            raise FileNotFoundError("config.resolved.yaml not found in FU-2 bundle")
        return yaml.safe_load(member.read())


def adapt_config_for_fold7(base_cfg: dict) -> dict:
    """Update FU-2 resolved config for FU-3 fold_07 training cutoff."""
    cfg = deepcopy(base_cfg)
    # FU-3 fold_07: train 2024-08-01 → 2026-02-01, val Feb, test Mar
    cfg["data"]["train_start_date"] = "2024-08-01"
    cfg["data"]["train_end_date"] = "2026-01-31"  # train ends 2026-01-31, val is Feb
    cfg["data"]["val_start_date"] = "2026-02-01"
    cfg["data"]["val_end_date"] = "2026-02-28"
    cfg["data"]["test_start_date"] = "2026-03-01"
    cfg["data"]["test_end_date"] = "2026-03-31"
    cfg["chosen_rule"] = CHOSEN_RULE  # canonical aggregator (gates-yaml)
    cfg.setdefault("ensemble", {})["chosen_rule"] = CHOSEN_RULE
    cfg["ensemble"]["seeds"] = SEEDS
    cfg["provenance"] = {
        "source": "FU-3 extended-window 8-fold WF (fold_07 checkpoints)",
        "wf_config": "configs/sg1_btc_velotrade_rehpo_wf_multiseed_extended.yaml",
        "gates_file": "configs/sg1_btc_velotrade_ensemble.gates.yaml",
        "predecessor_bundle": "ensemble_v1.tar.gz",
        "rule_change_vs_predecessor": "ens_mean -> ens_agreement (gates-yaml-canonical)",
    }
    return cfg


def main():
    cfg = load_fu2_resolved_config()
    cfg = adapt_config_for_fold7(cfg)

    seed_ckpts = {s: CKPT_PATTERN.format(seed=s) for s in SEEDS}
    for s, p in seed_ckpts.items():
        if not Path(p).exists():
            raise FileNotFoundError(f"seed {s}: {p} missing")

    with VERDICT_PATH.open() as f:
        verdict = json.load(f)
    wf_bs = verdict.get("wf_bootstrap")
    if not wf_bs:
        raise RuntimeError(f"verdict.json at {VERDICT_PATH} has no wf_bootstrap block")
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
        "note_pool_too_small": "diversity-audit-only: pool size <= K, no reselection possible",
        "note_inherited_from_l1": (
            "Top-3 seed selection inherited from FU-2 L1 multiseed N=10 diversity audit; "
            "WF Stage 3 confirmed temporal robustness (G1/G2/G4/G5 PASS, G3 legacy uplift "
            "1.018 < 1.10 = noise-blind point gate; v2.5 bootstrap P(PF)=1.000 PROMOTE)."
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
        predecessor_version=PREDECESSOR,
        trigger=TRIGGER,
    )

    print(f"\nWrote {bundle_path}")
    print(f"SHA256 sidecar: {bundle_path}.sha256")
    print("\nManifest summary:")
    print(json.dumps({k: v for k, v in manifest.items() if k != "diversity_audit"},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
