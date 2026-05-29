#!/usr/bin/env python3
"""GMGP1-EURUSD The5ers Hyper Growth — Stage 2.5-R ensemble-confirm.

Delegates to `sg1_xauusd_ensemble_eval.run_stage_2_5_val_selection()` —
the canonical Stage 2.5-R block-bootstrap evaluator (Protocol v2.5,
S526 decision_protocol_v25_bootstrap_primary).

Seeds: top-3 by L1 val_argmax_pf rule (S495) from
`results/gmgp1_eurusd_l1_seed_report.json`, S551-cont-9 2026-05-29:
  - seed 1024 val PF 2.7641 / test PF 2.5535
  - seed 456  val PF 2.7542 / test PF 2.6775
  - seed 5150 val PF 2.7396 / test PF 2.4965

Buffer fn = hyper_growth_buffers (HG caps: 3% daily / 6% trailing).
Gates yaml: configs/gmgp1_eurusd_oanda_ensemble.gates.yaml.
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
log = logging.getLogger("gmgp1-eurusd-ensemble-eval")


# Top-3 by L1 val_argmax_pf (S495) from results/gmgp1_eurusd_l1_seed_report.json.
SEED_CHECKPOINTS: Dict[int, str] = {
    1024: "checkpoints/gmgp1-eurusd-l1-seed1024_20260529_085349/checkpoint_final.pth",
    456:  "checkpoints/gmgp1-eurusd-l1-seed456_20260529_085349/checkpoint_final.pth",
    5150: "checkpoints/gmgp1-eurusd-l1-seed5150_20260529_085349/checkpoint_final.pth",
}
SEED_PFS: Dict[int, float] = {
    1024: 2.5535,  # test PF
    456:  2.6775,
    5150: 2.4965,
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
        default="configs/gmgp1_eurusd_oanda_l1_multiseed.yaml",
        help="L1 multiseed config (defines data splits, env, network)",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_dir = Path("results/gmgp1_eurusd_hg_ensemble")
    out_dir.mkdir(parents=True, exist_ok=True)

    agents = _load_agents(config, args.device)

    verdict = sg1.run_stage_2_5_val_selection(
        config=config,
        agents=agents,
        seed_pfs=SEED_PFS,
        out_dir=out_dir,
        device=args.device,
        buffer_fn=sg1.hyper_growth_buffers,
        workstream_label="gmgp1_eurusd_hyper_growth_l1",
        seed_checkpoints=SEED_CHECKPOINTS,
        bundle_version="v1",
    )

    print("\n========== GMGP1-EURUSD ENSEMBLE-CONFIRM (Protocol v2.5) ==========")
    print(json.dumps(verdict, indent=2, default=str))


if __name__ == "__main__":
    main()
