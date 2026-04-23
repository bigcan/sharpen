#!/usr/bin/env python3
"""GMGP1-XAUUSD L1-OANDA ensemble-confirm (Stage 2.5, Path B, Protocol v2 S495).

OANDA variant — delegates to `sg1.run_stage_2_5_val_selection()`. See that
function's docstring and `decision_ensemble_val_selection_s495.md` for the
val-argmax-PF selection protocol.

Path B re-ranking (S493-cont 2026-04-23): OANDA top-3 by test PF =
[42, 3141, 2025]. CME top-3 was [789, 2025, 1024] — only 2025 overlapped.
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
log = logging.getLogger("gmgp1-ensemble-eval-oanda")


SEED_CHECKPOINTS: Dict[int, str] = {
    42:   "checkpoints/gmgp1-xauusd-oanda-l1-rehpo-batch2-seed42_20260423_005221/checkpoint_final.pth",
    3141: "checkpoints/gmgp1-xauusd-oanda-l1-rehpo-batch1-seed3141_20260422_223124/checkpoint_final.pth",
    2025: "checkpoints/gmgp1-xauusd-oanda-l1-rehpo-batch1-seed2025_20260422_223124/checkpoint_final.pth",
}
SEED_PFS: Dict[int, float] = {
    42:   2.389,
    3141: 2.261,
    2025: 2.182,
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
        default="configs/gmgp1_xauusd_ftmo_rehpo_l1_multiseed_oanda.yaml",
        help="L1-OANDA multiseed config",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_dir = Path("results/gmgp1_xauusd_ensemble_oanda")
    out_dir.mkdir(parents=True, exist_ok=True)

    agents = _load_agents(config, args.device)

    verdict = sg1.run_stage_2_5_val_selection(
        config=config,
        agents=agents,
        seed_pfs=SEED_PFS,
        out_dir=out_dir,
        device=args.device,
        buffer_fn=sg1.ftmo_buffers,
        workstream_label="gmgp1_xauusd_ftmo_rehpo_20260421_l1_oanda",
    )

    print("\n========== GMGP1 OANDA ENSEMBLE-CONFIRM (Protocol v2 S495) ==========")
    print(json.dumps(verdict, indent=2, default=str))


if __name__ == "__main__":
    main()
