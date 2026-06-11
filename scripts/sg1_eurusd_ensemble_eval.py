#!/usr/bin/env python3
"""SG-1 EURUSD The5ers Hyper Growth — Stage 2.5-R ensemble-confirm.

Delegates to `sg1_xauusd_ensemble_eval.run_stage_2_5_val_selection()` —
the canonical Stage 2.5-R block-bootstrap evaluator (Protocol v2.5,
S526 decision_protocol_v25_bootstrap_primary).

Seeds: top-3 by L1 test PF (verdict
`results/sg1_eurusd_l1_verdict.json`, S549-cont 2026-05-23):
  - seed 9999  test PF 1.7248
  - seed 2025  test PF 1.6927
  - seed 3141  test PF 1.6466

Buffer fn = hyper_growth_buffers (Hyper Growth caps: 3% daily / 6% trailing).
Gates yaml: configs/sg1_eurusd_hyper_growth_ensemble.gates.yaml.
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
log = logging.getLogger("sg1-eurusd-ensemble-eval")


# Top-3 by L1 test PF (results/sg1_eurusd_l1_verdict.json, S549-cont).
SEED_CHECKPOINTS: Dict[int, str] = {
    9999: "checkpoints/sg1-eurusd-hg-l1-trial18-seed9999_20260523_120312/checkpoint_final.pth",
    2025: "checkpoints/sg1-eurusd-hg-l1-trial18-seed2025_20260523_120312/checkpoint_final.pth",
    3141: "checkpoints/sg1-eurusd-hg-l1-trial18-seed3141_20260523_120312/checkpoint_final.pth",
}
SEED_PFS: Dict[int, float] = {
    9999: 1.7248,
    2025: 1.6927,
    3141: 1.6466,
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
        default="configs/sg1_eurusd_hyper_growth_l1_multiseed.yaml",
        help="L1 multiseed config (defines data splits, env, network)",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_dir = Path("results/sg1_eurusd_hg_ensemble")
    out_dir.mkdir(parents=True, exist_ok=True)

    agents = _load_agents(config, args.device)

    verdict = sg1.run_stage_2_5_val_selection(
        config=config,
        agents=agents,
        seed_pfs=SEED_PFS,
        out_dir=out_dir,
        device=args.device,
        buffer_fn=sg1.hyper_growth_buffers,
        workstream_label="sg1_eurusd_hyper_growth_20260521_l1",
        seed_checkpoints=SEED_CHECKPOINTS,
        bundle_version="v1",
    )

    print("\n========== SG-1 EURUSD HYPER GROWTH ENSEMBLE-CONFIRM (v2.5) ==========")
    print(json.dumps(verdict, indent=2, default=str))


if __name__ == "__main__":
    main()
