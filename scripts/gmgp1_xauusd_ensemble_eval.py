#!/usr/bin/env python3
"""GMGP1-XAUUSD L1 re-HPO ensemble-confirm (Stage 2.5, Path A/CME, Protocol v2 S495).

Delegates to `sg1.run_stage_2_5_val_selection()`. See that function's docstring
and `decision_ensemble_val_selection_s495.md` for protocol details.

Top-3 distinct-checkpoint seeds from GMGP1 XAUUSD L1 re-HPO (2026-04-22).
Batch 4 (seeds 123/1337/3141) shared one checkpoint dir due to a
`deploy_bare_metal`/`run_full_pipeline` race — only ONE set of weights
is persisted on disk, so we substitute the next-best distinct seed (1024)
for the third slot.
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
log = logging.getLogger("gmgp1-ensemble-eval")


SEED_CHECKPOINTS: Dict[int, str] = {
    789:  "checkpoints/gmgp1-xauusd-ftmo-rehpo-l1-multiseed_20260422_135945/checkpoint_final.pth",
    2025: "checkpoints/gmgp1-xauusd-ftmo-rehpo-l1-multiseed_20260422_184946/checkpoint_final.pth",
    1024: "checkpoints/gmgp1-xauusd-ftmo-rehpo-l1-multiseed_20260422_140313/checkpoint_final.pth",
}
SEED_PFS: Dict[int, float] = {
    789:  2.467,
    2025: 2.185,
    1024: 1.959,
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
        default="configs/gmgp1_xauusd_ftmo_rehpo_l1_multiseed.yaml",
        help="L1 multiseed config",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_dir = Path("results/gmgp1_xauusd_ensemble")
    out_dir.mkdir(parents=True, exist_ok=True)

    agents = _load_agents(config, args.device)

    verdict = sg1.run_stage_2_5_val_selection(
        config=config,
        agents=agents,
        seed_pfs=SEED_PFS,
        out_dir=out_dir,
        device=args.device,
        buffer_fn=sg1.ftmo_buffers,
        workstream_label="gmgp1_xauusd_ftmo_rehpo_20260421_l1",
        seed_checkpoints=SEED_CHECKPOINTS,
        bundle_version="v1",
    )

    print("\n========== GMGP1 XAUUSD ENSEMBLE-CONFIRM (Protocol v2.3) ==========")
    print(json.dumps(verdict, indent=2, default=str))


if __name__ == "__main__":
    main()
