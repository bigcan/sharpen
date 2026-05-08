#!/usr/bin/env python3
"""GMGP1-Gold steady-state Stage 2.5-R ensemble-confirm (Protocol v2.5).

Gold variant of the val-split ensemble-confirm eval. Delegates to
`sg1.run_stage_2_5_val_selection()` — this file supplies the GMGP1-Gold-specific
seed/checkpoint map, seed PFs (from Stage 2 N=10 verdict), and a research-tier
buffer function (no prop-firm DD cap; gmgp1-gold paper container is internal-
validation IB MGC, not on FTMO/Velotrade).

Protocol v2.5 (per `decision_protocol_v25_bootstrap_primary.md`):
  Phase 0: diversity-aware top-K selection (audit-only when pool_size == K)
  Phase 1: run all 4 ensemble rules + N solos on val window
  Phase 2: argmax(val_PF) across ensembles → chosen_rule
  Phase 3: run chosen_rule + N solos on test window
  Phase 4a: legacy uplift gate (audit-only in v2.5)
  Phase 4b: stationary block bootstrap on per-bar returns →
            P(PF_ens > PF_best_solo), P(MDD_ens better than MDD_best_solo)
  Phase 4d: PROMOTE iff P_PF >= 0.90 AND P_MDD >= 0.90; PROMOTE_DD_ONLY iff
            only MDD passes; AMBIGUOUS_BOOT in [0.75, 0.90) band; else
            SOLO_BEST_FALLBACK
  Phase 4f: v2.4.1 §11 replay-buffer purge (no-op when --no_collect was used)

Active seed set — Stage 2.5-R post Stage 2 N=10 PASS (S536-cont, 2026-05-08):
  Top-3 by val_argmax_pf rule from results/gmgp1_gold_steadystate_l1_multiseed/
  seed_report.json:

    seed   wandb       val_PF   test_PF   source
    ----   --------    ------   -------   --------
     456   movqd2ur    2.8575   2.4871    rerun (S536-cont, sequential)
    1024   o27a9yvj    2.8322   2.4470    local-backtest of remote ckpt
    2025   0vb8qf8s    2.8237   2.5801    local-backtest of remote ckpt

Internal-validation flavor (not prop-firm):
  - max_drawdown_pct: 0.30 (research-tier; live MGC has no FTMO/Velotrade cap yet)
  - g4_research_dd_compliance gate uses 5pp buffer vs 30% (very loose; trips
    only on structurally broken aggregation)
  - No daily_loss gate
  - When/if gmgp1-gold graduates to MFFU Pro / Phidias Swing per
    project_gmgp1_mgc_retrain_queued_s477.md, fork this script + gates yaml.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict

import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts import sg1_xauusd_ensemble_eval as sg1  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gmgp1-gold-ensemble-eval")


SEED_CHECKPOINTS: Dict[int, str] = {
    456:  "checkpoints/gmgp1-gold-steadystate-l1-multiseed-recovered/seed456/checkpoint_final.pth",
    1024: "checkpoints/gmgp1-gold-steadystate-l1-multiseed-recovered/seed1024/checkpoint_final.pth",
    2025: "checkpoints/gmgp1-gold-steadystate-l1-multiseed-recovered/seed2025/checkpoint_final.pth",
}
SEED_PFS: Dict[int, float] = {
    # Test PFs from Stage 2 N=10 (results/gmgp1_gold_steadystate_l1_multiseed/seed_report.json).
    # Used by `_make_pf_weighted` in the ensemble; ens_mean / ens_median /
    # ens_agreement are unaffected. The val-argmax-pf rule selector uses val
    # PFs computed live in Phase 1, not these.
    456:  2.4871,
    1024: 2.4470,
    2025: 2.5801,
}


def research_buffers(metrics: dict) -> dict:
    """Research-tier buffer metrics for internal-validation gmgp1-gold.

    No prop-firm cap; gmgp1-gold IB MGC paper container is sim-to-live
    validation only. Buffer is computed against env's research-tier
    `max_drawdown_pct: 0.30` (30%). With Stage 2 test DDs uniformly < -1.5%,
    buffer is always ~28pp+ — the gate is a structural sanity check, not a
    binding constraint.
    """
    trail = metrics.get("trailing_max_drawdown_pct")
    return {
        "trailing_dd_buffer_pp": (30.0 + trail) if trail is not None else None,
        "daily_dd_buffer_pp": None,                   # N/A: no daily-loss gate
    }


def _load_agents(config: dict, device: str) -> Dict[int, object]:
    agents = {}
    for seed, ckpt in SEED_CHECKPOINTS.items():
        if not os.path.exists(ckpt):
            raise FileNotFoundError(ckpt)
        if os.path.getsize(ckpt) == 0:
            raise RuntimeError(
                f"seed {seed}: checkpoint {ckpt} is 0 bytes — "
                f"likely a crashed-init stub. Use the recovered-seed path."
            )
        agent = sg1._build_sac_agent(config, device)
        agent.load(ckpt)
        agents[seed] = agent
        log.info(f"seed {seed}: loaded {ckpt}")
    return agents


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/gmgp1_gold_steadystate_ensemble.yaml",
        help="Stage 2.5-R config (must contain data dates, gates.ensemble_*, "
             "and ensemble.gates_file overlay for v2.5 bootstrap thresholds).",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_dir = Path("results/gmgp1_gold_ensemble")
    out_dir.mkdir(parents=True, exist_ok=True)

    agents = _load_agents(config, args.device)

    verdict = sg1.run_stage_2_5_val_selection(
        config=config,
        agents=agents,
        seed_pfs=SEED_PFS,
        out_dir=out_dir,
        device=args.device,
        buffer_fn=research_buffers,
        workstream_label="gmgp1_gold_steadystate_a2_l1",
        seed_checkpoints=SEED_CHECKPOINTS,
        bundle_version="v1",                          # first ensemble bundle for gmgp1-gold
        predecessor_version=None,
        trigger="L1_N10_PASS_S536_cont",
    )

    print("\n========== GMGP1 GOLD ENSEMBLE-CONFIRM (Protocol v2.5) ==========")
    print(json.dumps(verdict, indent=2, default=str))


if __name__ == "__main__":
    main()
