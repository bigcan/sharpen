#!/usr/bin/env python3
"""Recompute Stage 2.5 verdicts from cached trajectory parquets.

Reproduces the v2.3 block-bootstrap verdict from cached val/test trajectories
without paying the 25-min agent re-eval cost. Used to retroactively apply the
S498-cont gates-file overlay fix to verdicts that ran before the structural
fix landed (GMGP1-BTC pre-v2.3, SG-1-BTC v2.3 but bootstrap_decision
LEGACY_GATE_DEFER because L1 config lacked the bootstrap gate keys).

Inputs:
  --workstream <name>          subdirectory under results/ (e.g. sg1_btc_velotrade)
  --gates-file <path>          standalone <ws>_ensemble.gates.yaml
  --legacy-uplift-promote N    used only to fall back to legacy decision when
                                bootstrap is insufficient (default 1.10)
  --legacy-uplift-ambiguous N  legacy ambiguous floor (default 1.05)

Outputs:
  results/<workstream>_ensemble/verdict_v2_3.json   recomputed verdict
  Original verdict.json untouched (audit trail).

Limitations:
  - Recomputes bootstrap on the chosen rule (from existing verdict.json) vs
    best_solo_on_test. Cannot re-pick chosen_rule because pre-v2.3 verdicts
    only cached the chosen-rule trajectory in test/, not all 4 ensembles.
  - Recomputes diversity_audit only when all 3 solo trajectories are present
    in test/ (true for both BTC workstreams).

This is offline-only — does NOT load agents, does NOT touch live containers.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.sg1_xauusd_ensemble_eval import (  # noqa: E402
    _per_bar_returns,
    _resolve_bootstrap_decision,
    block_bootstrap_pf_mdd,
    compute_action_correlation_matrix,
    select_diverse_top_k,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("recompute-2_5")


def _load_trajectory(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    return pd.read_parquet(parquet_path)


def recompute(
    workstream: str,
    gates_file: Path,
    legacy_promote: float,
    legacy_ambiguous: float,
    out_dir_override: Path | None = None,
) -> dict:
    out_dir = (
        Path(out_dir_override)
        if out_dir_override is not None
        else Path(f"results/{workstream}_ensemble")
    )
    if not out_dir.exists():
        raise FileNotFoundError(f"results dir missing: {out_dir}")

    verdict_path = out_dir / "verdict.json"
    if not verdict_path.exists():
        raise FileNotFoundError(f"prior verdict missing: {verdict_path}")
    prior = json.loads(verdict_path.read_text())

    chosen_rule = prior["chosen_rule"]
    best_solo_seed = prior["best_solo_seed_on_test"]
    seeds = sorted(int(s) for s in prior["seeds"])

    log.info(
        f"workstream={workstream} chosen_rule={chosen_rule} "
        f"best_solo_seed={best_solo_seed} seeds={seeds}"
    )

    # --- load gates (standalone wins on overlap with verdict-baked legacy) ---
    with open(gates_file, encoding="utf-8") as f:
        standalone = yaml.safe_load(f) or {}
    gates = dict(standalone.get("gates") or {})
    if "ensemble_uplift_min" not in gates:
        gates["ensemble_uplift_min"] = legacy_promote
    if "ensemble_ambiguous_min" not in gates:
        gates["ensemble_ambiguous_min"] = legacy_ambiguous
    log.info(
        f"loaded gates: P(PF)≥{gates.get('ensemble_bootstrap_p_pf_promote')} "
        f"P(MDD)≥{gates.get('ensemble_bootstrap_p_mdd_promote')} "
        f"legacy_promote={gates['ensemble_uplift_min']}"
    )

    # --- bootstrap: prefer cached if present, else recompute from trajectories ---
    test_dir = out_dir / "test"
    bs_n = int(gates.get("ensemble_bootstrap_resamples", 10000))
    bs_block = gates.get("ensemble_bootstrap_block_len", None)
    bs_block_f = float(bs_block) if bs_block is not None else None

    has_cached_bootstrap = (
        "bootstrap_verdict" in prior
        and prior["bootstrap_verdict"].get("p_pf_ens_better") is not None
    )
    if has_cached_bootstrap:
        log.info("[bootstrap] reusing cached bootstrap_verdict from prior verdict.json")
        bootstrap_verdict = prior["bootstrap_verdict"]
    else:
        # No cache → must recompute from cached test/ trajectories.
        chosen_traj = _load_trajectory(test_dir / f"{chosen_rule}_trajectory.parquet")
        solo_traj = _load_trajectory(test_dir / f"solo_{best_solo_seed}_trajectory.parquet")
        chosen_returns = _per_bar_returns(chosen_traj)
        solo_returns = _per_bar_returns(solo_traj)
        if chosen_returns.size != solo_returns.size:
            n_common = min(chosen_returns.size, solo_returns.size)
            log.warning(
                f"trajectory length mismatch: chosen={chosen_returns.size} "
                f"solo={solo_returns.size}; truncating to {n_common}"
            )
            chosen_returns = chosen_returns[:n_common]
            solo_returns = solo_returns[:n_common]
        log.info(
            f"[bootstrap] running stationary block bootstrap n_resamples={bs_n} "
            f"block_len={bs_block_f or 'auto-sqrt(n)'} on n={chosen_returns.size} bars"
        )
        bootstrap_verdict = block_bootstrap_pf_mdd(
            ens_returns=chosen_returns,
            solo_returns=solo_returns,
            n_resamples=bs_n,
            block_len_mean=bs_block_f,
        )

    legacy_uplift = prior.get("uplift")
    bs_resolved = _resolve_bootstrap_decision(bootstrap_verdict, gates)
    bs_decision = bs_resolved["decision"]
    bs_reason = bs_resolved["reason"]
    bs_primary_basis = bs_resolved["primary_basis"]
    bs_thresholds = bs_resolved["thresholds_used"]
    log.info(
        f"[bootstrap] P(PF)={bootstrap_verdict.get('p_pf_ens_better')} "
        f"P(MDD)={bootstrap_verdict.get('p_mdd_ens_better')} → {bs_decision} "
        f"(primary_basis={bs_primary_basis})"
    )

    # --- diversity audit (only if all 3 solo trajectories present) ---
    diversity_audit = None
    solo_paths = {s: test_dir / f"solo_{s}_trajectory.parquet" for s in seeds}
    if all(p.exists() for p in solo_paths.values()):
        trajectories = {f"solo_{s}": _load_trajectory(p) for s, p in solo_paths.items()}
        seed_pfs = {s: float(prior["test_solo_pfs"][str(s)]) for s in seeds}
        corr_matrix, corr_seed_order = compute_action_correlation_matrix(
            trajectories, seeds
        )
        diversity_k = int(gates.get("ensemble_top_k", min(3, len(seed_pfs))))
        diversity_lambda = float(gates.get("ensemble_diversity_lambda", 1.0))
        diversity_audit = select_diverse_top_k(
            seed_pfs=seed_pfs,
            corr_matrix=corr_matrix,
            seed_order=corr_seed_order,
            k=diversity_k,
            diversity_lambda=diversity_lambda,
        )
        diversity_audit["correlation_matrix"] = [
            [None if pd.isna(v) else float(v) for v in row] for row in corr_matrix
        ]
        diversity_audit["correlation_seed_order"] = corr_seed_order
        diversity_audit["pool_size"] = len(seed_pfs)
        diversity_audit["k"] = diversity_k
        diversity_audit["note_pool_too_small"] = (
            "diversity-audit-only: pool size <= K, no reselection possible"
            if len(seed_pfs) <= diversity_k
            else None
        )

    # --- compose verdict (Protocol v2.5 schema; legacy keys preserved) ---
    if bs_primary_basis == "legacy_uplift":
        decision = prior.get("legacy_decision") or prior.get("decision")
        decision_source = "legacy_uplift_v2.1"
    else:
        decision = bs_decision
        decision_source = "bootstrap_v2.5_recomputed"

    legacy_uplift_block = {
        "uplift_ratio": legacy_uplift,
        "promote_threshold": float(gates["ensemble_uplift_min"]),
        "ambiguous_band": [
            float(gates["ensemble_ambiguous_min"]),
            float(gates["ensemble_uplift_min"]),
        ],
        "would_have_decided": prior.get("legacy_decision") or prior.get("decision"),
    }

    verdict = {
        "schema_version": "2.5",
        "workstream": prior.get("workstream", workstream),
        "protocol": "v2.5_stage_2_5_bootstrap_primary_recomputed_s526",
        "supersedes": [
            prior.get("protocol", "<unknown>"),
            "verdict.json (kept as audit trail)",
        ],
        "recomputed_from_cache": {
            "gates_file": str(gates_file),
            "test_chosen_parquet": str(test_dir / f"{chosen_rule}_trajectory.parquet"),
            "test_solo_parquet": str(test_dir / f"solo_{best_solo_seed}_trajectory.parquet"),
            "bootstrap_reused": "bootstrap_verdict" in prior
            and prior["bootstrap_verdict"].get("p_pf_ens_better") is not None,
        },
        "val_window": prior.get("val_window"),
        "test_window": prior.get("test_window"),
        "seeds": seeds,
        "val_pf_by_rule": prior.get("val_pf_by_rule"),
        "chosen_rule": chosen_rule,
        "chosen_rule_val_pf": prior.get("chosen_rule_val_pf"),
        "chosen_rule_test_pf": prior.get("chosen_rule_test_pf"),
        "test_solo_pfs": prior.get("test_solo_pfs"),
        "best_solo_seed_on_test": best_solo_seed,
        "best_solo_test_pf": prior.get("best_solo_test_pf"),
        # v2.5 PRIMARY: bootstrap block.
        "bootstrap": {
            "p_pf_ens_better": bootstrap_verdict.get("p_pf_ens_better"),
            "p_mdd_ens_better": bootstrap_verdict.get("p_mdd_ens_better"),
            "block_len_used": bootstrap_verdict.get("block_len_mean"),
            "resamples": bootstrap_verdict.get("n_resamples"),
            "thresholds_used": bs_thresholds,
            "reason": bs_reason,
        },
        "bootstrap_verdict": bootstrap_verdict,
        "bootstrap_decision": bs_decision,
        "bootstrap_reason": bs_reason,
        # v2.5 SECONDARY: legacy uplift demoted to audit metadata.
        "legacy_uplift": legacy_uplift_block,
        "uplift": legacy_uplift,
        "uplift_promote_threshold": float(gates["ensemble_uplift_min"]),
        "uplift_ambiguous_threshold": float(gates["ensemble_ambiguous_min"]),
        "legacy_decision": prior.get("legacy_decision") or prior.get("decision"),
        "diversity_audit": diversity_audit,
        "decision": decision,
        "primary_basis": bs_primary_basis,
        "decision_source": decision_source,
        "reason": (
            f"[bootstrap v2.5] {bs_reason}"
            if decision_source == "bootstrap_v2.5_recomputed"
            else f"[legacy uplift] {prior.get('reason', 'unknown')}"
        ),
    }

    out_path = out_dir / "verdict_v2_5.json"
    out_path.write_text(json.dumps(verdict, indent=2, default=str))
    log.info(f"wrote {out_path}")
    return verdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--workstream",
        required=True,
        help="results/<workstream>_ensemble subdirectory name "
        "(e.g. sg1_btc_velotrade, gmgp1_btc)",
    )
    ap.add_argument(
        "--gates-file",
        required=True,
        help="Path to <workstream>_ensemble.gates.yaml",
    )
    ap.add_argument("--legacy-uplift-promote", type=float, default=1.10)
    ap.add_argument("--legacy-uplift-ambiguous", type=float, default=1.05)
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Override results/<workstream>_ensemble path (for non-standard suffixes "
        "like results/gmgp1_xauusd_ensemble_oanda).",
    )
    args = ap.parse_args()

    verdict = recompute(
        workstream=args.workstream,
        gates_file=Path(args.gates_file),
        legacy_promote=args.legacy_uplift_promote,
        legacy_ambiguous=args.legacy_uplift_ambiguous,
        out_dir_override=Path(args.out_dir) if args.out_dir else None,
    )
    print("\n========== RECOMPUTED VERDICT ==========")
    print(
        json.dumps(
            {
                "workstream": verdict["workstream"],
                "decision": verdict["decision"],
                "decision_source": verdict["decision_source"],
                "bootstrap_decision": verdict["bootstrap_decision"],
                "bootstrap_reason": verdict["bootstrap_reason"],
                "p_pf_ens_better": verdict["bootstrap_verdict"].get("p_pf_ens_better"),
                "p_mdd_ens_better": verdict["bootstrap_verdict"].get("p_mdd_ens_better"),
                "legacy_decision": verdict["legacy_decision"],
                "uplift": verdict["uplift"],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
