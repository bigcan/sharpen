#!/usr/bin/env python
"""Multi-sleeve (ETF momentum+rates ⊕ BTC options-VRP) paper-executor validation.

Drives the N-sleeve fund-of-funds paper executor
(:class:`finrl_pro_ds.paper.PortfolioExecutor`) END-TO-END on real data and emits the
pre-registered ``paper_soak`` verdict (now incl. the realized-returns diversification gate
+ the VRP short-vol tail monitor). Generalizes ``run_cross_asset_paper_validation.py`` to
combine the ETF/IB account (momentum+rates, weight-combined, UNCHANGED) with the Deribit
VRP account (a return stream) at the capital-allocation level (MS-ADR-1).

With ``sleeves.vrp.enabled: false`` (the default) this reproduces the 2-sleeve verdict
byte-for-byte (MS-ADR-6); enable it only via a combined-book Tier-2 audit + operator
go-ahead before any capital (no live paper soak is running — fleet halted 2026-06-01).

Flow:
  load_two_sleeve_data → build_two_sleeve_arrays                 # ETF account bundle
  (+ deribit_options_loader → build_panels  if vrp enabled)      # VRP account panel
    → PortfolioExecutor.run()                    # batch tautology, parity-0 sanity baseline
    → PortfolioExecutor.run_independent_recompute()  # forward-path, catches calendar/look-ahead
    → compare(live, oracle)                      # the 4 paper_soak.parity metrics
    → evaluate_paper_soak_gates(...) → verdict (serialized JSON)

NOT a capital promotion: computes + serializes the verdict only. Thresholds come entirely
from ``configs/cross_asset_momentum.gates.yaml`` (paper_soak block).

Usage:
  python scripts/run_multi_sleeve_paper_validation.py
  python scripts/run_multi_sleeve_paper_validation.py --config configs/live_multi_sleeve_paper.yaml --enable_vrp
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from finrl_pro_ds.crypto.data import deribit_options_loader as dol
from finrl_pro_ds.crypto.data import options_array_builder as oab
from finrl_pro_ds.data.cross_asset_loader import (
    build_two_sleeve_arrays,
    load_two_sleeve_data,
)
from finrl_pro_ds.paper import (
    PortfolioExecutor,
    evaluate_paper_soak_gates,
    serialize_verdict,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("multi_sleeve_paper_validation")


def _parity_dict(p) -> dict:
    return {
        "weight_l1_drift_max": float(p.weight_l1_drift_max),
        "daily_return_te_bps_max": float(p.daily_return_te_bps_max),
        "missed_rebalances": int(p.missed_rebalances),
        "cost_drift_ratio": float(p.cost_drift_ratio),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/live_multi_sleeve_paper.yaml")
    ap.add_argument("--gates", default=None, help="gates yaml (default: ensemble.gates_file in config)")
    ap.add_argument("--out", default="results/cross_asset_paper/multi_sleeve_paper_verdict.json")
    ap.add_argument("--force_refetch", action="store_true")
    ap.add_argument("--require_fresh", action="store_true",
                    help="freshness-gate the ETF loader cache (scheduled live mode)")
    ap.add_argument("--enable_vrp", action="store_true",
                    help="force sleeves.vrp.enabled=true for this run (override the config flag)")
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.enable_vrp:
        config.setdefault("sleeves", {}).setdefault("vrp", {})["enabled"] = True
    gates_path = args.gates or config["ensemble"]["gates_file"]
    gates_cfg = yaml.safe_load(Path(gates_path).read_text(encoding="utf-8"))

    vrp_cfg = dict(config.get("sleeves", {}).get("vrp", {}))
    vrp_on = bool(vrp_cfg.get("enabled", False))

    log.info("loading ETF two-sleeve data (union=%d assets)...", config["universe"]["n_assets"])
    data = load_two_sleeve_data(config, force_refetch=args.force_refetch,
                                require_fresh=args.require_fresh)
    start_ts, end_ts = data["close"].index[0], data["close"].index[-1]
    log.info("ETF data %s -> %s (%d bars)", start_ts.date(), end_ts.date(), len(data["close"]))
    bundle = build_two_sleeve_arrays(data, start_ts, end_ts)

    data_bundle: dict = {"etf": bundle}
    if vrp_on:
        asset = str(vrp_cfg.get("asset", "BTC"))
        log.info("loading VRP panel (%s, Deribit account)...", asset)
        panel = oab.build_panels(dol.load({"universe": {"assets": [asset]}}))
        data_bundle["vrp"] = {"panel": panel}
        log.info("VRP panel %s -> %s (%d bars)",
                 panel.dates[0].date(), panel.dates[-1].date(), len(panel.timestamps))
    else:
        log.info("VRP sleeve DISABLED (MS-ADR-6) — combined book == 2-sleeve ETF book")

    ex = PortfolioExecutor(config)
    log.info("sleeves: %s", [s.name for s in ex.sleeves])

    # 1) BATCH tautology baseline — parity ~0 by construction; proves the wiring is intact.
    live_batch, oracle_batch = ex.run(data_bundle)
    parity_batch = ex.compare(live_batch, oracle_batch)
    log.info("BATCH         : l1_drift=%.3e te_bps=%.4f missed=%d cost_ratio=%.4f",
             parity_batch.weight_l1_drift_max, parity_batch.daily_return_te_bps_max,
             parity_batch.missed_rebalances, parity_batch.cost_drift_ratio)

    # 2) LOAD-BEARING forward-path — independent recompute per sleeve on a growing window.
    live_fwd, oracle_fwd = ex.run_independent_recompute(data_bundle)
    parity_fwd = ex.compare(live_fwd, oracle_fwd)
    log.info("FORWARD-RECOMP: l1_drift=%.3e te_bps=%.4f missed=%d cost_ratio=%.4f",
             parity_fwd.weight_l1_drift_max, parity_fwd.daily_return_te_bps_max,
             parity_fwd.missed_rebalances, parity_fwd.cost_drift_ratio)

    # 3) Pre-registered paper_soak verdict on the forward-path render.
    verdict = evaluate_paper_soak_gates(live_fwd, parity_fwd, gates_cfg)
    corr_full = live_fwd.corr_to_spy(window=None)
    verdict["validation_meta"] = {
        "config": str(args.config),
        "gates": str(gates_path),
        "vrp_enabled": vrp_on,
        "sleeves": [s.name for s in ex.sleeves],
        "combined_start": (str(pd.to_datetime(int(live_fwd.timestamps[0]), unit="s").date())
                           if live_fwd.n_steps else None),
        "combined_end": (str(pd.to_datetime(int(live_fwd.timestamps[-1]), unit="s").date())
                         if live_fwd.n_steps else None),
        "n_steps": int(live_fwd.n_steps),
        "corr_to_spy_full_sample": (None if corr_full is None else float(corr_full)),
        "batch_parity_baseline": _parity_dict(parity_batch),
        "forward_parity": _parity_dict(parity_fwd),
        "note": "HISTORICAL forward-path validation, NOT a live soak. Capital promotion "
                "still requires the COMBINED-book Tier-2 deep-lifecycle audit + operator go-ahead.",
    }
    out_path = serialize_verdict(verdict, args.out)

    log.info("=" * 72)
    log.info("paper_soak overall_status = %s", verdict["overall_status"])
    for g, body in verdict["groups"].items():
        log.info("  %-12s %-28s %s", g, body["status"], body["severity"])
    log.info("verdict -> %s", out_path)
    log.info("=" * 72)
    print(json.dumps({"overall_status": verdict["overall_status"],
                      "groups": {g: b["status"] for g, b in verdict["groups"].items()},
                      "summary": verdict["summary"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
