"""Turnover-lever reachability probe for the cross-asset RL allocator (v1.1).

Pre-flight for the v1.1 cost round (`configs/cross_asset_momentum_v11.yaml`).
The v1 Stage-3 WF proved the RL has REAL gross alpha (median frictionless net
Sharpe 0.80 vs linear core 0.34) and loses ALL of it to transaction cost
(median net 0.135, cost_gap 0.41 vs the 0.15 gate) => ship_linear_core.

v1.1 answers that with three execution levers. This probe asks the cheap
question FIRST (the execution-overlay lesson: *is this gate reachable?*): on the
REAL substrate, does throttling turnover convert gross into net, or does it
destroy the gross edge at the same rate it saves cost?

Method (CPU, no training): drive the SAME env over the SAME 14 WF test windows
with a known weight path, across the lever grid `no_trade_band x
rebalance_interval`, net and frictionless. Two conviction paths:
  * daily  — raw daily `trend_conviction`, the HIGH-turnover analog of the RL's
             cost-killed path (this is the one the levers must rescue)
  * monthly— the frozen linear core's cadence = the gate baseline (reference)

Read: if throttling the daily path only walks it back to the monthly core's net
Sharpe, execution levers alone cannot deliver the gate's +0.10 uplift, and the
v1.1 run is a training-budget spend against an unreachable bar. If it overshoots
the core, the lever mechanism has headroom and the run is justified.

This probe is NOT a substitute for the run: the RL retrains under the levers and
follows a different path. It bounds the MECHANISM, not the policy.

Usage:
    python scripts/research/allocator_turnover_lever_probe.py \
        --config configs/cross_asset_momentum_v11.yaml \
        --out results/cross_asset_allocator/lever_probe.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sharpen.data import cross_asset_loader as loader  # noqa: E402
from sharpen.envs import allocator_factory as factory  # noqa: E402
from scripts.cross_asset_pipeline import build_wf_schedule, load_config  # noqa: E402

logger = logging.getLogger("lever_probe")

# Lever grid. Bands/intervals span the v1.1 HPO search space
# (no_trade_band [0.01,0.15], rebalance_interval {1,5,21}) plus the OFF cell.
BANDS = [0.0, 0.01, 0.03, 0.05, 0.10, 0.15]
INTERVALS = [1, 5, 21]

FRICTIONLESS = {"taker_fee": 0.0, "slippage_base_bps": 0.0, "slippage_impact_bps": 0.0}


def _drive_conviction(arrays, config, conv, overrides):
    """Drive the allocator env with a held-conviction series under `overrides`.

    Deliberately NOT `factory.drive_with_conviction` — that routes through
    `_linear_core_drive`, which forces the execution levers OFF (correct for the
    gate baseline, useless for measuring them)."""
    env = factory.make_allocator_env(arrays, config, overrides=overrides, eval_mode=True)
    metrics, _, _ = factory._drive(env, lambda k, obs: conv[k])
    return metrics


def probe_window(arrays, config) -> list[dict]:
    conv_daily = np.asarray(arrays["conviction_ary"], dtype=np.float64)
    conv_monthly = factory.monthly_rebal_conviction(arrays["timestamps"], conv_daily)
    rows = []
    for path_name, conv in (("daily", conv_daily), ("monthly", conv_monthly)):
        for band in BANDS:
            for interval in INTERVALS:
                lev = {"no_trade_band": band, "rebalance_interval": interval}
                net = _drive_conviction(arrays, config, conv, lev)
                fric = _drive_conviction(arrays, config, conv, {**lev, **FRICTIONLESS})
                rows.append({
                    "path": path_name, "no_trade_band": band, "rebalance_interval": interval,
                    "net_sharpe": net["net_sharpe"],
                    "frictionless_sharpe": fric["net_sharpe"],
                    "cost_gap": fric["net_sharpe"] - net["net_sharpe"],
                    "turnover_ann": net["turnover_ann"],
                    "max_drawdown": net["max_drawdown"],
                })
    return rows


def honest_oos(data, config, windows, lookbacks, norm_window) -> dict:
    """Train-selected, test-evaluated throttle on the FROZEN LINEAR CORE.

    The grid sweep above is best-cell-on-test — a selection, not a result. The
    linear core is deterministic (no training, no seed), so the honest version is
    cheap and exact: pick the (band, interval) cell on each window's TRAIN+VAL
    span, then apply that one cell to the untouched TEST span. No RL, no GPU, and
    no run-to-run noise — the pipeline's per-window Sharpe sd of ~1.75 does not
    exist on this path.

    Baseline is the same core with levers OFF (`evaluate_linear_core`'s conditions).
    """
    rows = []
    for w in windows:
        sel_arrays = loader.build_allocator_arrays(
            data["signals"], data["close"], data["volume"], data["assets"],
            w["train_start"], w["val_end"], lookbacks=lookbacks, norm_window=norm_window)
        test_arrays = loader.build_allocator_arrays(
            data["signals"], data["close"], data["volume"], data["assets"],
            w["test_start"], w["test_end"], lookbacks=lookbacks, norm_window=norm_window)

        def core_conv(a):
            return factory.monthly_rebal_conviction(
                a["timestamps"], np.asarray(a["conviction_ary"], dtype=np.float64))

        best, best_sharpe = None, -np.inf
        for band in BANDS:
            for interval in INTERVALS:
                m = _drive_conviction(sel_arrays, config, core_conv(sel_arrays),
                                      {"no_trade_band": band, "rebalance_interval": interval})
                if m["net_sharpe"] > best_sharpe:
                    best_sharpe, best = m["net_sharpe"], (band, interval)

        band, interval = best
        test_lev = _drive_conviction(test_arrays, config, core_conv(test_arrays),
                                     {"no_trade_band": band, "rebalance_interval": interval})
        test_off = _drive_conviction(test_arrays, config, core_conv(test_arrays),
                                     {"no_trade_band": 0.0, "rebalance_interval": 1})
        rows.append({
            "window": w["window"],
            "selected_band": band, "selected_interval": interval,
            "selection_sharpe": best_sharpe,
            "test_throttled_sharpe": test_lev["net_sharpe"],
            "test_baseline_sharpe": test_off["net_sharpe"],
            "test_uplift": test_lev["net_sharpe"] - test_off["net_sharpe"],
            "test_throttled_return": test_lev["total_return"],
            "test_baseline_return": test_off["total_return"],
            "test_throttled_turnover": test_lev["turnover_ann"],
            "test_baseline_turnover": test_off["turnover_ann"],
        })
        logger.info("window %d: selected band=%.2f interval=%d -> test %+.3f vs %+.3f",
                    w["window"], band, interval,
                    test_lev["net_sharpe"], test_off["net_sharpe"])
    up = np.array([r["test_uplift"] for r in rows])
    return {
        "per_window": rows,
        "median_uplift": float(np.median(up)),
        "mean_uplift": float(up.mean()),
        "n_windows_positive": int((up > 0).sum()),
        "n_windows": len(rows),
        "median_throttled": float(np.median([r["test_throttled_sharpe"] for r in rows])),
        "median_baseline": float(np.median([r["test_baseline_sharpe"] for r in rows])),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cross_asset_momentum_v11.yaml")
    ap.add_argument("--out", default="results/cross_asset_allocator/lever_probe.json")
    ap.add_argument("--max-windows", type=int, default=0, help="0 = all")
    ap.add_argument("--honest-oos", action="store_true",
                    help="train-selected/test-evaluated throttle on the linear core")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = load_config(args.config)
    data = loader.load_cross_asset_data(config)
    lookbacks = data["lookbacks"]
    norm_window = int(config.get("features", {}).get("norm_window", 252))
    windows = build_wf_schedule(data["close"].index, config["walk_forward"])
    if args.max_windows:
        windows = windows[: args.max_windows]

    if args.honest_oos:
        res = honest_oos(data, config, windows, lookbacks, norm_window)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("")
        print("===== HONEST OOS: throttle selected on train+val, applied to test =====")
        print(pd.DataFrame(res["per_window"]).to_string(
            index=False, float_format=lambda v: f"{v:8.3f}"))
        print("")
        print(f"median test uplift : {res['median_uplift']:+.4f}")
        print(f"mean   test uplift : {res['mean_uplift']:+.4f}")
        print(f"windows positive   : {res['n_windows_positive']}/{res['n_windows']}")
        print(f"median throttled   : {res['median_throttled']:+.4f}"
              f"   median baseline: {res['median_baseline']:+.4f}")
        print("")
        print(f"wrote {out}")
        return 0

    all_rows = []
    for w in windows:
        arrays = loader.build_allocator_arrays(
            data["signals"], data["close"], data["volume"], data["assets"],
            w["test_start"], w["test_end"], lookbacks=lookbacks, norm_window=norm_window)
        for r in probe_window(arrays, config):
            r["window"] = w["window"]
            all_rows.append(r)
        logger.info("window %d done (%s -> %s)", w["window"],
                    w["test_start"].date(), w["test_end"].date())

    df = pd.DataFrame(all_rows)
    agg = (df.groupby(["path", "no_trade_band", "rebalance_interval"])
             .agg(median_net_sharpe=("net_sharpe", "median"),
                  median_frictionless=("frictionless_sharpe", "median"),
                  median_cost_gap=("cost_gap", "median"),
                  median_turnover_ann=("turnover_ann", "median"),
                  n_windows=("net_sharpe", "size"))
             .reset_index())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": args.config,
        "n_windows": len(windows),
        "grid": {"no_trade_band": BANDS, "rebalance_interval": INTERVALS},
        "per_window": all_rows,
        "aggregate": agg.to_dict(orient="records"),
    }, indent=1), encoding="utf-8")

    pd.set_option("display.width", 200)
    for path_name in ("daily", "monthly"):
        print(f"\n===== conviction path: {path_name} (median over {len(windows)} WF test windows) =====")
        print(agg[agg["path"] == path_name].to_string(index=False,
              float_format=lambda v: f"{v:8.3f}"))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
