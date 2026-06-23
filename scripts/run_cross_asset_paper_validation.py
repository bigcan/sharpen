#!/usr/bin/env python
"""Cross-asset 2-sleeve paper-executor validation (rung-1 forward-path parity + soak gates).

Exercises the momentum + rates-carry fund-of-funds paper executor
(:class:`finrl_pro_ds.paper.TwoSleeveExecutor`) END-TO-END on real data and emits the
pre-registered ``paper_soak`` verdict. This is the runner for the **load-bearing
forward-path check** (``run_independent_recompute``) that the rung-1 Tier-2 audit named
as the step-4 gate before any paper-CAPITAL promotion — and the validation the parked
prop-firm CMDP build waits behind (``.agent/artifacts/prop_firm_cmdp_architecture.md``).

Flow:
  load_two_sleeve_data(config) -> build_two_sleeve_arrays (full available range)
    -> TwoSleeveExecutor.run()                     # batch tautology, parity-0 sanity baseline
    -> TwoSleeveExecutor.run_independent_recompute()  # forward-path, catches calendar/look-ahead
    -> compare(live, oracle)                       # the 4 paper_soak.parity metrics
    -> evaluate_paper_soak_gates(live, parity, gates) -> verdict (serialized JSON)

NOT a capital promotion: it computes + serializes the verdict only. Thresholds come
entirely from ``configs/cross_asset_momentum.gates.yaml`` (paper_soak block).

A FAIL on either the OHLCV or the curve data manifest ABORTS before any verdict is computed
(fail-closed, P1-01/P10-03); WARN is surfaced into ``validation_meta``. The freshness gate is
ON by default (live mode); pass ``--backtest`` for a reproducible historical window.

Usage:
  python scripts/run_cross_asset_paper_validation.py                      # live: freshness-gated
  python scripts/run_cross_asset_paper_validation.py --backtest           # reproducible window
  python scripts/run_cross_asset_paper_validation.py --config configs/live_cross_asset_paper.yaml --force_refetch
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Mapping

import yaml

from finrl_pro_ds.data.cross_asset_loader import (
    build_two_sleeve_arrays,
    load_two_sleeve_data,
)
from finrl_pro_ds.paper import (
    TwoSleeveExecutor,
    evaluate_paper_soak_gates,
    serialize_verdict,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("paper_validation")


def evaluate_data_integrity(manifest: Mapping, curve_manifest: Mapping | None) -> dict:
    """Fail-CLOSED data gate (P1-01/P10-03): the EARNED manifest status(es) decide whether the
    paper verdict may even be computed. A FAIL dataset — a stale-print P&L share over the
    gmgp1-gold threshold, or a grossly-corrupt curve leg — ABORTS the run; WARN is surfaced
    into the verdict, never silenced. Returns ``{ok, fails, warnings, statuses}``."""
    statuses = {"ohlcv": manifest.get("status"),
                "curve": (curve_manifest or {}).get("status")}
    fails = [f"{name} manifest status=FAIL" for name, st in statuses.items() if st == "FAIL"]
    warnings = [f"{name} manifest status=WARN" for name, st in statuses.items() if st == "WARN"]
    ss = manifest.get("stale_scan") or {}
    if ss.get("flagged_tickers"):
        warnings.append(f"ohlcv stale-flagged tickers: {ss['flagged_tickers']}")
    if curve_manifest and curve_manifest.get("calendar_desync_days"):
        warnings.append(f"curve date_max {curve_manifest.get('date_max')} is "
                        f"{curve_manifest['calendar_desync_days']}d behind ETF date_max")
    return {"ok": not fails, "fails": fails, "warnings": warnings, "statuses": statuses}


def _parity_dict(p) -> dict:
    """Compact, JSON-safe view of a ParityReport (built from known attributes so this does
    not depend on an internal serialization method name)."""
    return {
        "weight_l1_drift_max": float(p.weight_l1_drift_max),
        "daily_return_te_bps_max": float(p.daily_return_te_bps_max),
        "missed_rebalances": int(p.missed_rebalances),
        "cost_drift_ratio": float(p.cost_drift_ratio),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/live_cross_asset_paper.yaml")
    ap.add_argument(
        "--gates", default=None, help="gates yaml (default: ensemble.gates_file in config)"
    )
    ap.add_argument(
        "--out", default="results/cross_asset_paper/paper_validation_verdict.json"
    )
    ap.add_argument("--force_refetch", action="store_true")
    ap.add_argument("--backtest", action="store_true",
                    help="reproducible historical window: DISABLE the freshness gate. Default "
                         "is live/non-backtest mode, which freshness-gates the loader + curve "
                         "caches so they refetch when they don't reach today (P10-04).")
    args = ap.parse_args()

    # require_fresh defaults TRUE (live/scheduled mode): a bare run freshness-gates the cache.
    # An explicit --backtest opts into the reproducible historical window (no refetch).
    require_fresh = not args.backtest

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    gates_path = args.gates or config["ensemble"]["gates_file"]
    gates_cfg = yaml.safe_load(Path(gates_path).read_text(encoding="utf-8"))

    n_union = config["universe"]["n_assets"]
    log.info("loading two-sleeve data (union=%d assets, require_fresh=%s)...",
             n_union, require_fresh)
    data = load_two_sleeve_data(config, force_refetch=args.force_refetch,
                                require_fresh=require_fresh)
    start_ts = data["close"].index[0]
    end_ts = data["close"].index[-1]
    log.info(
        "data range %s -> %s (%d bars, %d union assets)",
        start_ts.date(), end_ts.date(), len(data["close"]), len(data["union_assets"]),
    )

    # Fail-CLOSED data gate BEFORE any verdict is computed (P1-01/P10-03): a FAIL manifest
    # (the gmgp1-gold stale-print class, or a corrupt curve leg) must abort, not drive a verdict.
    integrity = evaluate_data_integrity(data["manifest"], data.get("curve_manifest"))
    for w in integrity["warnings"]:
        log.warning("data-integrity WARN: %s", w)
    if not integrity["ok"]:
        log.error("ABORT — fail-closed data gate: %s", "; ".join(integrity["fails"]))
        return 2

    bundle = build_two_sleeve_arrays(data, start_ts, end_ts)

    ex = TwoSleeveExecutor(config)

    # 1) BATCH tautology baseline — parity ~0 by construction; proves the wiring is intact.
    live_batch, oracle_batch = ex.run(bundle)
    parity_batch = ex.compare(live_batch, oracle_batch)
    log.info(
        "BATCH         : l1_drift=%.3e te_bps=%.4f missed=%d cost_ratio=%.4f",
        parity_batch.weight_l1_drift_max, parity_batch.daily_return_te_bps_max,
        parity_batch.missed_rebalances, parity_batch.cost_drift_ratio,
    )

    # 2) LOAD-BEARING forward-path — independent recompute per sleeve on a growing window.
    #    A calendar / truncation / look-ahead bug in EITHER sleeve moves weight_l1_drift.
    live_fwd, oracle_fwd = ex.run_independent_recompute(bundle)
    parity_fwd = ex.compare(live_fwd, oracle_fwd)
    log.info(
        "FORWARD-RECOMP: l1_drift=%.3e te_bps=%.4f missed=%d cost_ratio=%.4f",
        parity_fwd.weight_l1_drift_max, parity_fwd.daily_return_te_bps_max,
        parity_fwd.missed_rebalances, parity_fwd.cost_drift_ratio,
    )

    # 3) Pre-registered paper_soak verdict on the forward-path render.
    verdict = evaluate_paper_soak_gates(live_fwd, parity_fwd, gates_cfg)
    # Full-sample (life-of-strategy) corr-to-SPY for context: the drift gate trips on the
    # trailing corr_window_days window (regime-drift monitor), but a reviewer should also see
    # the structural number — they diverge when the sleeve drifts within a recent regime.
    corr_full = live_fwd.corr_to_spy(window=None)
    corr_window = int(gates_cfg["paper_soak"]["drift"].get("corr_window_days", 252))
    curve_manifest = data.get("curve_manifest") or {}
    verdict["validation_meta"] = {
        "config": str(args.config),
        "gates": str(gates_path),
        "data_start": str(start_ts.date()),
        "data_end": str(end_ts.date()),
        "n_bars": int(len(data["close"])),
        "n_union_assets": int(len(data["union_assets"])),
        "corr_to_spy_full_sample": (None if corr_full is None else float(corr_full)),
        "corr_to_spy_gate_window_days": corr_window,
        "batch_parity_baseline": _parity_dict(parity_batch),
        "forward_parity": _parity_dict(parity_fwd),
        # Data-integrity provenance (P1-01/P1-03/P10-04): the EARNED statuses + freshness mode
        # this verdict ran under, so a reader can see whether the gate even had teeth.
        "require_fresh": require_fresh,
        "data_manifest_status": data["manifest"].get("status"),
        "curve_manifest_status": curve_manifest.get("status"),
        "cache_date_max": data["manifest"].get("date_max"),
        "curve_date_max": curve_manifest.get("date_max"),
        "stale_scan": data["manifest"].get("stale_scan"),
        "data_warnings": integrity["warnings"],
        "note": "HISTORICAL forward-path validation, NOT a live soak. Capital promotion "
                "still requires a Tier-2 deep-lifecycle audit + operator go-ahead.",
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
