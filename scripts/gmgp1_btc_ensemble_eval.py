#!/usr/bin/env python3
"""GMGP1-BTC Velotrade L1 ensemble-confirm (Stage 2.5, Path A, Protocol v2 S495).

BTC variant of the val-split ensemble-confirm eval. Delegates to
`sg1.run_stage_2_5_val_selection()` — this file supplies the BTC-specific
seed/checkpoint map, seed PFs, buffer function (Velotrade, no daily-loss gate),
and output dir. Protocol logic (val-argmax-pf rule selection + test-window
uplift gate) lives in the shared helper.

Protocol (per S495 `decision_ensemble_val_selection_s495.md`, supersedes S493):
  Phase 1: run all 4 ensemble rules + 3 solos on L1 **val** window
  Phase 2: argmax(val_PF) across ensembles → chosen_rule (no test peek)
  Phase 3: run chosen_rule + 3 solos on L1 **test** window
  Phase 4: uplift = chosen_rule_test_PF / best_solo_test_PF
           >= gates.ensemble_uplift_min (1.10) → PROMOTE
           >= gates.ensemble_ambiguous_min (1.05) → AMBIGUOUS_RERUN
           else → SOLO_BEST_FALLBACK

Top-3 seeds from `results/gmgp1_btc_l1_rehpo_aggregate.json`
(2026-04-23 S494-cont, trial-#48 HPs, N=10 Path A Bitfinex):
  seed 456  PF=2.13823  (solo-recovery checkpoint at ..._102150/ — the
                         _080059/ dir is a 0-byte stub from the init crash)
  seed 1337 PF=2.08359
  seed 123  PF=2.02130

Velotrade 2-Step gate (from `gmgp1_btc_velotrade_rehpo_l1_multiseed.yaml`):
  max_trailing_drawdown_pct: 0.08, max_daily_loss_pct: 0.0 (disabled).
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
log = logging.getLogger("gmgp1-btc-ensemble-eval")


SEED_CHECKPOINTS: Dict[int, str] = {
    456:  "checkpoints/gmgp1-btc-velotrade-l1-rehpo-seed456_20260423_102150/checkpoint_final.pth",
    1337: "checkpoints/gmgp1-btc-velotrade-l1-rehpo-seed1337_20260423_080059/checkpoint_final.pth",
    123:  "checkpoints/gmgp1-btc-velotrade-l1-rehpo-seed123_20260423_080059/checkpoint_final.pth",
}
SEED_PFS: Dict[int, float] = {
    456:  2.13823,
    1337: 2.08359,
    123:  2.02130,
}


def velotrade_buffers(metrics: dict) -> dict:
    """Velotrade 2-Step buffer metrics. Daily-loss gate is DISABLED in Step 1,
    so only trailing-DD buffer is reported. Policy limit 8% is a Velotrade
    rule-constant (not a per-workstream tunable)."""
    trail = metrics.get("trailing_max_drawdown_pct")
    return {
        "trailing_dd_buffer_pp": (8.0 + trail) if trail is not None else None,
        "daily_dd_buffer_pp": None,
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
        default="configs/gmgp1_btc_velotrade_rehpo_l1_multiseed.yaml",
        help="L1 multiseed config (must contain data.train_end_date, "
             "val_start/end_date, test_start/end_date, gates.ensemble_uplift_min)",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_dir = Path("results/gmgp1_btc_ensemble")
    out_dir.mkdir(parents=True, exist_ok=True)

    agents = _load_agents(config, args.device)

    verdict = sg1.run_stage_2_5_val_selection(
        config=config,
        agents=agents,
        seed_pfs=SEED_PFS,
        out_dir=out_dir,
        device=args.device,
        buffer_fn=velotrade_buffers,
        workstream_label="gmgp1_btc_velotrade_rehpo_20260421_l1",
    )

    print("\n========== GMGP1 BTC ENSEMBLE-CONFIRM (Protocol v2 S495) ==========")
    print(json.dumps(verdict, indent=2, default=str))


if __name__ == "__main__":
    main()
