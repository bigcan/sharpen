#!/usr/bin/env python3
"""GMGP1-BTC Velotrade L1 ensemble-confirm (Stage 2.5-R, Protocol v2.3 + v2.4.1).

BTC variant of the val-split ensemble-confirm eval. Delegates to
`sg1.run_stage_2_5_val_selection()` — this file supplies the BTC-specific
seed/checkpoint map, seed PFs, buffer function (Velotrade, no daily-loss gate),
and output dir. Protocol logic (val-argmax-pf rule selection + v2.3 bootstrap
gate + v2.4.1 §11 replay-buffer purge) lives in the shared helper.

Protocol (per `decision_protocol_v2_mandatory.md` + Protocol v2.3):
  Phase 1: run all 4 ensemble rules + N solos on L1 **val** window
  Phase 2: argmax(val_PF) across ensembles → chosen_rule (no test peek)
  Phase 3: run chosen_rule + N solos on L1 **test** window
  Phase 4a: legacy uplift gate (back-compat, sanity check)
  Phase 4b: stationary block bootstrap on per-bar returns →
            P(PF_ens > PF_best_solo), P(MDD_ens better than MDD_best_solo)
  Phase 4d: PROMOTE iff P_PF ≥ 0.90 AND P_MDD ≥ 0.90;
            PROMOTE_DD_ONLY iff only MDD passes;
            else SOLO_BEST_FALLBACK
  Phase 4f: v2.4.1 §11 replay-buffer purge for non-selected seeds (graceful
            no-op when --no_collect was used and replay.pkl lives on remote)

Active seed set — Stage 2.5-R for A_long × 2M (S522, 2026-05-03):
  Triggered by L1 retrain at A_long × 2M (data-window study STRONG_GO winner;
  see `decision_volume_axis_v2_2m_ceiling.md`). Test PFs from data-window
  study `unified_20260502/per_seed.json`.

  seed   run_id     test_PF   val_PF
  ----   --------   -------   ------
  123    ba0zxoot   2.4422    2.3035
  456    drwd4fuk   2.4225    2.3328
  789    m4bjcriq   2.4695    2.3722
  1024   ro7ptpll   2.3670    2.3334
  2026   hgxtabl5   2.5084    2.3800

Cross-venue caveat:
  The 5 checkpoints were trained on Bybit BTC (only source with 22mo of bars).
  Live container deploys on Binance demo. This is a known sim-to-live gap
  (`project_sim_to_live_gap_audit_s470.md`); Stage 2.5-R verdict establishes
  Bybit-OOS rule selection only. Bitfinex (prior eval venue) lacks 2024 bars;
  re-evaling there would force budget back to A_base and defeat the data-window
  study finding.

Velotrade 2-Step gate (Velotrade buffer fn, daily-loss disabled):
  max_trailing_drawdown_pct: 0.08, max_daily_loss_pct: 0.0.
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
    123:  "checkpoints/gmgp1-dwin-long-2m-seed123_20260502_063041/checkpoint_final.pth",
    456:  "checkpoints/gmgp1-dwin-long-2m-seed456_20260502_063041/checkpoint_final.pth",
    789:  "checkpoints/gmgp1-dwin-long-2m-seed789_20260502_065129/checkpoint_final.pth",
    1024: "checkpoints/gmgp1-dwin-long-2m-seed1024_20260502_070147/checkpoint_final.pth",
    2026: "checkpoints/gmgp1-dwin-long-2m-seed2026_20260502_071220/checkpoint_final.pth",
}
SEED_PFS: Dict[int, float] = {
    # L1 test PF from data-window study unified_20260502/per_seed.json.
    # Used by `_make_pf_weighted` in the ensemble; ens_mean / ens_median /
    # ens_agreement are unaffected. The val-argmax-pf rule selector uses
    # val PFs computed live in Phase 1, not these.
    123:  2.4422,
    456:  2.4225,
    789:  2.4695,
    1024: 2.3670,
    2026: 2.5084,
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
        default="configs/gmgp1_btc_along_2m_ensemble.yaml",
        help="Stage 2.5-R config (must contain data.train_end_date, "
             "val_start/end_date, test_start/end_date, gates.ensemble_uplift_min, "
             "and ensemble.gates_file overlay for v2.3 bootstrap thresholds). "
             "Default targets A_long × 2M Bybit-trained checkpoints (S522).",
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
        # v2.3: pass checkpoints so the swap bundle writes on PROMOTE.
        # Bundle version starts at v1; Stage 2.5-R reruns bump to v2/v3/...
        seed_checkpoints=SEED_CHECKPOINTS,
        bundle_version="v2",                           # v1 was 2026-04-23 velotrade re-HPO
        predecessor_version="v1",
        trigger="L1_retrain_along_2m_S522",
    )

    print("\n========== GMGP1 BTC ENSEMBLE-CONFIRM (Protocol v2.3 + v2.4.1) ==========")
    print(json.dumps(verdict, indent=2, default=str))


if __name__ == "__main__":
    main()
