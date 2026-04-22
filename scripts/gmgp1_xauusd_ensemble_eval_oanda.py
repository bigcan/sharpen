#!/usr/bin/env python3
"""GMGP1-XAUUSD L1-OANDA ensemble confirmation test (Stage 2.5, Path B).

OANDA variant of `scripts/gmgp1_xauusd_ensemble_eval.py`. Reuses the SG-1
aggregation + backtest infrastructure; only the seed-to-checkpoint map, the
per-seed test PFs, and the default L1 config differ.

Protocol v2 Stage 2.5 mandatory rule (per S493 `decision_ensemble_mandatory_stage_2_5.md`):
  ens_agreement PF / best_solo PF >= 1.10 on L1 test window -> PASS,
  Stage 3 WF then trains on the top-3 OANDA seeds with ens_agreement aggregation.

Path B re-ranking (S493-cont 2026-04-23): GMGP1-XAUUSD L1-OANDA top-3 by test
PF = [42, 3141, 2025]. CME top-3 was [789, 2025, 1024] — only 2025 overlapped,
so WF on CME-picked seeds would have missed the two strongest OANDA seeds.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts import sg1_xauusd_ensemble_eval as sg1  # noqa: E402
from scripts.sg1_arm_gate_backtest import (  # noqa: E402
    _build_sac_agent,
    _prep_backtest_config,
    compute_gate_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gmgp1-ensemble-eval-oanda")


# Top-3 distinct-checkpoint seeds from GMGP1 XAUUSD L1-OANDA (2026-04-23).
# PFs sourced from backtest_test/profit_factor in remote training logs.
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
BEST_SOLO_SEED = 42
BEST_SOLO_PF = SEED_PFS[BEST_SOLO_SEED]
PROTOCOL_UPLIFT_MIN = 1.10


def _load_agents(config: dict, device: str) -> Dict[int, object]:
    agents = {}
    for seed, ckpt in SEED_CHECKPOINTS.items():
        if not os.path.exists(ckpt):
            raise FileNotFoundError(ckpt)
        agent = _build_sac_agent(config, device)
        agent.load(ckpt)
        agents[seed] = agent
        log.info(f"seed {seed}: loaded {ckpt}")
    return agents


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/gmgp1_xauusd_ftmo_rehpo_l1_multiseed_oanda.yaml",
        help="L1-OANDA multiseed config driving test_start/end_date",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--rules",
        nargs="*",
        default=None,
        help="Subset of rule names to run (default: all). Names: "
             "solo_42, solo_3141, solo_2025, ens_mean, ens_median, "
             "ens_agreement, ens_pf_weighted.",
    )
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_dir = Path("results/gmgp1_xauusd_ensemble_oanda")
    out_dir.mkdir(parents=True, exist_ok=True)

    agents = _load_agents(config, args.device)

    rules: List[tuple] = (
        [(f"solo_{s}", sg1._agg_solo(s)) for s in SEED_CHECKPOINTS]
        + [
            ("ens_mean",      sg1._agg_mean),
            ("ens_median",    sg1._agg_median),
            ("ens_agreement", sg1._agg_agreement),
            ("ens_pf_weighted", sg1._make_pf_weighted(SEED_PFS)),
        ]
    )

    selected = set(args.rules) if args.rules else None
    summary = []
    for rule_name, rule_fn in rules:
        if selected is not None and rule_name not in selected:
            continue
        df = sg1.run_rule(config, agents, rule_name, rule_fn, args.device, out_dir)
        m = compute_gate_metrics(df, rule_name)
        m.update(sg1.ftmo_buffers(m))
        (out_dir / f"{rule_name}_metrics.json").write_text(
            json.dumps(m, indent=2, default=str)
        )
        summary.append(m)
        log.info(
            f"[{rule_name}] PF={m['pf_bar']:.3f} ret={m['total_return_pct']:.2f}%  "
            f"trailing_DD={m['trailing_max_drawdown_pct']:.2f}%  "
            f"intraday_DD={m.get('worst_intraday_daily_dd_pct')}  "
            f"trades={m['trade_count']}  FTMO={m['ftmo_compliance_pass']}"
        )

    sdf = pd.DataFrame(summary)
    sdf.to_csv(out_dir / "summary.csv", index=False)
    sdf.to_markdown(out_dir / "summary.md", index=False)

    by_rule = {m["label"]: m for m in summary}
    ens_agreement_pf = by_rule.get("ens_agreement", {}).get("pf_bar")
    ens_mean_pf = by_rule.get("ens_mean", {}).get("pf_bar")
    ens_median_pf = by_rule.get("ens_median", {}).get("pf_bar")
    ens_pfw_pf = by_rule.get("ens_pf_weighted", {}).get("pf_bar")

    best_solo_pf_bar = max(
        by_rule.get(f"solo_{s}", {}).get("pf_bar", 0) for s in SEED_CHECKPOINTS
    )

    def _uplift(pf):
        if pf is None or best_solo_pf_bar <= 0:
            return None
        return pf / best_solo_pf_bar

    verdict = {
        "study": "gmgp1_xauusd_ftmo_rehpo_20260421_l1_oanda",
        "path": "B",
        "top3_seeds": sorted(SEED_CHECKPOINTS),
        "best_solo_seed": BEST_SOLO_SEED,
        "best_solo_pf_reported": BEST_SOLO_PF,
        "best_solo_pf_bar_eval": best_solo_pf_bar,
        "ens_mean_pf": ens_mean_pf,
        "ens_median_pf": ens_median_pf,
        "ens_agreement_pf": ens_agreement_pf,
        "ens_pf_weighted_pf": ens_pfw_pf,
        "uplift_ens_agreement_over_best_solo": _uplift(ens_agreement_pf),
        "protocol_promote_threshold": PROTOCOL_UPLIFT_MIN,
    }
    if ens_agreement_pf is not None and best_solo_pf_bar > 0:
        uplift = ens_agreement_pf / best_solo_pf_bar
        if uplift >= PROTOCOL_UPLIFT_MIN:
            verdict["protocol_decision"] = "PROMOTE_STAGE_2_5_CONFIRMED_ON_OANDA"
            verdict["protocol_reason"] = (
                f"ens_agreement uplift {uplift:.4f}x >= {PROTOCOL_UPLIFT_MIN:.2f} threshold"
            )
        elif uplift >= 1.05:
            verdict["protocol_decision"] = "AMBIGUOUS_RERUN"
            verdict["protocol_reason"] = (
                f"ens_agreement uplift {uplift:.4f}x in ambiguous 1.05-1.10 range"
            )
        else:
            verdict["protocol_decision"] = "SOLO_BEST_FALLBACK"
            verdict["protocol_reason"] = (
                f"ens_agreement uplift {uplift:.4f}x below 1.05 floor"
            )

    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str))

    print("\n========== GMGP1 OANDA ENSEMBLE SUMMARY ==========")
    print(
        sdf[
            [
                "label", "pf_bar", "total_return_pct", "trailing_max_drawdown_pct",
                "worst_intraday_daily_dd_pct", "trade_count", "active_days",
                "max_single_day_share", "ftmo_compliance_pass",
                "daily_dd_buffer_pp", "trailing_dd_buffer_pp",
            ]
        ].to_string(index=False)
    )
    print("\n========== PROTOCOL v2 STAGE 2.5 (OANDA CONFIRM) ==========")
    print(json.dumps(verdict, indent=2, default=str))


if __name__ == "__main__":
    main()
