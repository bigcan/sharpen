#!/usr/bin/env python3
"""SG-1-XAUUSD Volume Study v2 Phase 2 Stage 2.5-R (retrain ensemble-confirm).

Thin wrapper around `sg1.run_stage_2_5_val_selection()`. Delegates the
phase-1-val / phase-2-pick / phase-3-test / bootstrap-PRIMARY logic; only
this file owns the seed→checkpoint mapping + val PFs for ens_pf_weighted.

Top-3 by val_argmax_pf = [2025, 42, 9999] from
results/sg1_xauusd_volume_study_v2_phase2/seed_report.json (corrected
S540-cont; see feedback_seed_report_key_scramble_s540.md). Run after the
sanity check `key == name_seed == ckpt_seed` for all 10 entries.

Predecessor: SG-1-XAUUSD ensemble v1 (S488 OANDA L1 N=10, seeds [42, 2025,
3141] trained through 2026-02-06, live since 2026-04-28 with cTrader paper).
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
log = logging.getLogger("sg1-xauusd-vs-v2-phase2-stage25r")


SEED_CHECKPOINTS: Dict[int, str] = {
    2025: "checkpoints/sg1-xauusd-volume-study-v2-phase2-b2-seed2025_20260519_002617/checkpoint_final.pth",
    42:   "checkpoints/sg1-xauusd-volume-study-v2-phase2-b1-seed42_20260510_161414/checkpoint_final.pth",
    9999: "checkpoints/sg1-xauusd-volume-study-v2-phase2-b2-seed9999_20260519_002617/checkpoint_final.pth",
}

# val PFs from results/sg1_xauusd_volume_study_v2_phase2/seed_report.json
# (post key-scramble fix). Used to weight ens_pf_weighted aggregation.
SEED_PFS: Dict[int, float] = {
    2025: 2.1731,
    42:   2.1323,
    9999: 2.0858,
}


def _load_agents(config: dict, device: str) -> Dict[int, object]:
    agents = {}
    for seed, ckpt in SEED_CHECKPOINTS.items():
        if not os.path.exists(ckpt):
            raise FileNotFoundError(ckpt)
        agent = sg1._build_sac_agent(config, device)
        agent.load(ckpt)
        agents[seed] = agent
        log.info(f"seed {seed}: loaded {ckpt}")
    return agents


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/sg1_xauusd_volume_study_v2_phase2_l1_multiseed.yaml",
        help="Phase 2 L1 multiseed config (drives val/test windows + bootstrap gates)",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_dir = Path("results/sg1_xauusd_volume_study_v2_phase2_stage25r")
    out_dir.mkdir(parents=True, exist_ok=True)

    agents = _load_agents(config, args.device)

    verdict = sg1.run_stage_2_5_val_selection(
        config=config,
        agents=agents,
        seed_pfs=SEED_PFS,
        out_dir=out_dir,
        device=args.device,
        buffer_fn=sg1.ftmo_buffers,
        workstream_label="sg1_xauusd_volume_study_v2_phase2",
        seed_checkpoints=SEED_CHECKPOINTS,
        bundle_version="v2",
        predecessor_version="v1",
        trigger="volume_study_v2_phase2_retrain",
    )

    print("\n========== SG-1-XAUUSD VS-v2 PHASE 2 STAGE 2.5-R VERDICT ==========")
    print(json.dumps(verdict, indent=2, default=str))


if __name__ == "__main__":
    main()
