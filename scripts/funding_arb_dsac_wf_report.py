"""Funding-Arb DSAC walk-forward aggregator + leverage-grid stress sub-report
(Protocol v2 Stage 3).

Reads per-window manifests + checkpoints produced by `launch_funding_arb_wf.py`,
computes the WF gate verdicts (G1..G4 per Velotrade gates yaml), and runs a
leverage-grid stress replay (L ∈ {1×, 2×, 3×, 5×, 7.5×, 10×}) on each window's
test split. Output: `results/funding_arb_dsac_wf/wf_report_<ts>.json` plus a
human-readable summary printed to stdout.

Leverage replay mechanism:
  - Load DSAC agent from window's checkpoint
  - Build test env with `max_gross_exposure` overridden to leverage L
  - At each step, scale agent action by L (clip to [-1, 1]) before env.step
  - Compute trailing-DD against EOD-resetting high-water mark (Velotrade rule)

Usage:
    # Local (after SFTP'ing all 8 windows' checkpoints + manifests):
    python scripts/funding_arb_dsac_wf_report.py \
        --config configs/funding_arb_dsac_l1_multiseed.yaml \
        --gates configs/funding_arb_dsac_velotrade_wf.gates.yaml \
        --wf_root checkpoints/funding_arb_dsac_wf_<ts> \
        --windows 0,1,2,3,4,5,6,7
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.funding_arb_hpo_runner import (  # noqa: E402
    _create_eval_env,
    _DSACModelWrapper,
    _make_dsac_agent,
)
from scripts.funding_arb_dsac_train_seed import (  # noqa: E402
    _build_arrays_for_window,
    _extract_hps,
    _select_window,
)
from scripts.funding_arb_runner import prepare_data  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("wf_report")


# --- leverage replay -------------------------------------------------------

def _eod_trailing_dd(portfolio_values: list[float], timestamps: list[pd.Timestamp]) -> float:
    """Compute worst trailing drawdown against EOD-resetting high-water mark.

    Velotrade trailing DD rule: equity peak resets at each 00:00 UTC rollover.
    Within-day DD is measured against the higher of (start-of-day equity,
    intraday peak so far). Returns max trailing DD as fraction (0.06 = 6%).
    """
    if not portfolio_values or len(portfolio_values) < 2:
        return 0.0
    pv = np.asarray(portfolio_values, dtype=np.float64)
    if len(timestamps) != len(pv):
        return float(1.0 - pv.min() / max(pv.max(), 1e-10))  # fallback: peak DD
    days = pd.Series([pd.Timestamp(t).floor("D") for t in timestamps])
    worst_dd = 0.0
    high_water = pv[0]
    current_day = days.iloc[0]
    for i in range(len(pv)):
        if days.iloc[i] != current_day:
            current_day = days.iloc[i]
            high_water = pv[i]              # EOD reset to start-of-day
        else:
            high_water = max(high_water, pv[i])
        dd = 1.0 - pv[i] / max(high_water, 1e-10)
        worst_dd = max(worst_dd, dd)
    return float(worst_dd)


def _replay_with_leverage(model: _DSACModelWrapper, env, leverage: float) -> dict:
    """Run a single eval pass scaling actions by leverage. Caller has already
    overridden env.max_gross_exposure to match L (so the env's normalize-to-cap
    won't undo the scaling)."""
    obs, _ = env.reset()
    portfolio_values = [env.initial_capital]
    timestamps: list = []
    total_funding = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        scaled = np.clip(action * leverage, -1.0, 1.0)
        obs, _r, terminated, truncated, info = env.step(scaled)
        portfolio_values.append(info["portfolio_value"])
        # env.timestamps is int64 UTC epoch seconds; step_idx points to the
        # bar just processed after env.step (incremented inside step before logic).
        ts_epoch = int(env.timestamps[env.step_idx])
        timestamps.append(pd.Timestamp(ts_epoch, unit="s", tz="UTC"))
        total_funding = info["total_funding_earned"]
        done = terminated or truncated

    pv = np.array(portfolio_values)
    rets = np.diff(pv) / np.maximum(pv[:-1], 1e-10)
    total_return = pv[-1] / pv[0] - 1.0
    sharpe = 0.0
    if len(rets) > 1 and np.std(rets, ddof=1) > 1e-10:
        sharpe = float(np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(8760))
    peak_dd = float((1.0 - pv / np.maximum.accumulate(pv)).max())
    eod_dd = _eod_trailing_dd(list(pv), timestamps) if timestamps else peak_dd
    return {
        "leverage": leverage,
        "total_return": float(total_return),
        "sharpe": sharpe,
        "peak_drawdown": peak_dd,
        "eod_trailing_drawdown": eod_dd,
        "final_value": float(pv[-1]),
        "total_funding_earned": float(total_funding),
        "cumulative_fees": float(env.cumulative_fees),
        "funding_vs_costs_ratio": float(total_funding / (env.cumulative_fees + 1e-10)),
        "n_steps": len(rets),
    }


# --- gate verdicts ---------------------------------------------------------

def _verdict_g1(per_window: list[dict], gates: dict) -> dict:
    """G1: per-window profitability."""
    cfg = gates["gates"]["g1_per_window_profitability"]
    threshold = cfg["min_test_return_pct"]
    pass_count = sum(1 for w in per_window if w["test"]["total_return"] * 100 > threshold)
    return {"name": "g1_per_window_profitability", "threshold": threshold,
            "passed_windows": pass_count, "total": len(per_window),
            "min_required": cfg["folds_pass_min"],
            "pass": pass_count >= cfg["folds_pass_min"]}


def _verdict_g2(per_window: list[dict], gates: dict) -> dict:
    """G2: PF floor (per-window + median)."""
    cfg = gates["gates"]["g2_pf_floor"]
    pfs = [w["test"]["funding_vs_costs_ratio"] for w in per_window]
    pass_count = sum(1 for p in pfs if p >= cfg["pf_floor_per_window"])
    median_pf = float(np.median(pfs))
    return {"name": "g2_pf_floor",
            "pf_floor_per_window": cfg["pf_floor_per_window"],
            "pf_floor_median": cfg["pf_floor_median"],
            "passed_windows": pass_count, "total": len(per_window),
            "min_required": cfg["folds_pass_min"], "median_pf": median_pf,
            "pass": pass_count >= cfg["folds_pass_min"] and median_pf >= cfg["pf_floor_median"]}


def _verdict_g3(per_window: list[dict], gates: dict) -> dict:
    """G3: max drawdown per window (at training leverage)."""
    cfg = gates["gates"]["g3_max_drawdown"]
    threshold = cfg["mdd_max_per_window_pct"] / 100.0
    pass_count = sum(1 for w in per_window if w["test"]["max_drawdown"] <= threshold)
    return {"name": "g3_max_drawdown",
            "mdd_max_per_window_pct": cfg["mdd_max_per_window_pct"],
            "passed_windows": pass_count, "total": len(per_window),
            "min_required": cfg["folds_pass_min"],
            "pass": pass_count >= cfg["folds_pass_min"]}


def _verdict_leverage_grid(stress: dict, gates: dict) -> dict:
    """Operational leverage = max L such that worst_window EOD trailing DD +
    buffer ≤ velotrade cap. Reported per-aggregation (worst, median, p95)."""
    cfg = gates["gates"]["stress_leverage_grid"]
    cap = cfg["velotrade_cap_pct"] / 100.0
    buf = cfg["velotrade_buffer_pp"] / 100.0
    target = cap - buf

    out = {}
    for agg in cfg["aggregations"]:
        max_safe_l = None
        per_l = []
        for L_str, per_window_list in stress.items():
            L = float(L_str)
            dds = [w["eod_trailing_drawdown"] for w in per_window_list]
            if not dds:
                continue
            if agg == "worst_window":
                stat = max(dds)
            elif agg == "median_window":
                stat = float(np.median(dds))
            elif agg == "p95_window":
                stat = float(np.percentile(dds, 95))
            else:
                continue
            per_l.append({"leverage": L, agg: stat, "safe": stat <= target})
            if stat <= target and (max_safe_l is None or L > max_safe_l):
                max_safe_l = L
        out[agg] = {"per_leverage": per_l, "max_safe_leverage": max_safe_l,
                    "target_dd_max": target}
    return {"name": "stress_leverage_grid",
            "velotrade_cap_pct": cfg["velotrade_cap_pct"],
            "velotrade_buffer_pp": cfg["velotrade_buffer_pp"],
            "by_aggregation": out}


# --- main pipeline ---------------------------------------------------------

def _load_window_manifest(wf_root: Path, window_index: int) -> tuple[Path, dict]:
    """Find the per-window subdir and return (ckpt_path, manifest dict)."""
    candidates = sorted(wf_root.glob(f"window{window_index:02d}_*"))
    if not candidates:
        # Fallback: name pattern from the launcher = "<prefix>-window{NN}_<ts>"
        candidates = sorted(wf_root.glob(f"*window{window_index:02d}_*"))
    if not candidates:
        raise FileNotFoundError(f"window {window_index}: no subdir under {wf_root}")
    if len(candidates) > 1:
        log.warning("window %d: %d subdirs match, using latest %s",
                    window_index, len(candidates), candidates[-1].name)
    sub = candidates[-1]
    manifest_path = sub / "manifest.json"
    ckpt_path = sub / "checkpoint_final.pth"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"window {window_index}: missing manifest at {manifest_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"window {window_index}: missing checkpoint at {ckpt_path}")
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)
    return ckpt_path, m


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path,
                   help="Same L1 multiseed config used for WF training")
    p.add_argument("--gates", required=True, type=Path,
                   help="Velotrade WF gates yaml")
    p.add_argument("--wf_root", required=True, type=Path,
                   help="Directory containing per-window subdirs (each with "
                        "checkpoint_final.pth + manifest.json)")
    p.add_argument("--windows", default=None,
                   help="Comma-separated window indices (default: read range from gates)")
    p.add_argument("--out_dir", default=str(PROJECT_ROOT / "results" / "funding_arb_dsac_wf"))
    p.add_argument("--skip_leverage_grid", action="store_true",
                   help="Skip the leverage-grid stress replay (G1-G3 only)")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(args.gates, encoding="utf-8") as f:
        gates = yaml.safe_load(f)

    if args.windows:
        windows = [int(w) for w in args.windows.split(",") if w.strip()]
    elif "window_indices" in gates:
        windows = list(gates["window_indices"])
    else:
        rng = gates.get("window_index_range", [0, 7])
        windows = list(range(rng[0], rng[1] + 1))

    log.info("Aggregating %d windows from %s", len(windows), args.wf_root)

    # --- Load manifests ---
    per_window: list[dict] = []
    ckpt_paths: dict[int, Path] = {}
    for w in windows:
        ckpt, manifest = _load_window_manifest(args.wf_root, w)
        ckpt_paths[w] = ckpt
        per_window.append({
            "window": w,
            "window_dates": manifest.get("window_dates", {}),
            "val": manifest["val_metrics"],
            "test": manifest["test_metrics"],
            "train_seconds": manifest.get("train_seconds"),
            "ckpt": str(ckpt),
        })
        t = manifest["test_metrics"]
        log.info("  window %d: test PF=%.3f Sharpe=%.2f Ret=%+.2f%% DD=%.2f%%",
                 w, t["funding_vs_costs_ratio"], t["sharpe"],
                 t["total_return"] * 100, t["max_drawdown"] * 100)

    # --- G1, G2, G3 verdicts (from manifest metrics) ---
    g1 = _verdict_g1(per_window, gates)
    g2 = _verdict_g2(per_window, gates)
    g3 = _verdict_g3(per_window, gates)
    g4 = {"name": "g4_pf_xcheck", "pass": True,
          "note": "Funding-arb has no mid/close split; PF computed from a single perp price source. Gate auto-PASS."}

    # --- Leverage-grid stress replay (per-window per-leverage) ---
    stress: dict[str, list[dict]] = {}
    if not args.skip_leverage_grid:
        log.info("Preparing data for leverage-grid replay (this may use cache)...")
        data = prepare_data(config)
        if not data["walk_forward"]["passed"]:
            log.error("Walk-forward coverage check FAILED")
            return 2
        agent_params, dsac_params = _extract_hps(config)

        leverage_factors = gates["gates"]["stress_leverage_grid"]["leverage_factors"]
        for L in leverage_factors:
            stress[str(L)] = []
            for w in windows:
                window_dates = _select_window(data, w)
                _train, _val, test_arr = _build_arrays_for_window(
                    data, config["universe"]["assets"], window_dates,
                )
                # Override max_gross_exposure for replay so the env doesn't
                # normalize-to-cap and undo our action scaling.
                cfg_replay = {**config}
                cfg_replay["environment"] = dict(config["environment"])
                cfg_replay["environment"]["max_gross_exposure"] = float(L)

                env = _create_eval_env(test_arr, cfg_replay)
                agent = _make_dsac_agent(env, config, agent_params, dsac_params)
                agent.load(str(ckpt_paths[w]))
                model = _DSACModelWrapper(agent)

                metrics = _replay_with_leverage(model, env, leverage=L)
                metrics["window"] = w
                stress[str(L)].append(metrics)
                log.info("  window=%d L=%.1f×: ret=%+.2f%% trailing_dd=%.2f%% peak_dd=%.2f%%",
                         w, L, metrics["total_return"] * 100,
                         metrics["eod_trailing_drawdown"] * 100,
                         metrics["peak_drawdown"] * 100)

    # --- Verdict ---
    verdicts = {
        "g1": g1, "g2": g2, "g3": g3, "g4": g4,
        "stress_leverage_grid": (
            _verdict_leverage_grid(stress, gates) if stress
            else {"name": "stress_leverage_grid", "skipped": True}
        ),
    }
    hard_gates_pass = all(verdicts[g]["pass"] for g in ("g1", "g2", "g3", "g4"))
    decision = (
        gates["decision"]["all_pass_action"] if hard_gates_pass
        else gates["decision"]["any_fail_action"]
    )

    # Pretty print
    print("\n" + "=" * 78)
    print(f"Funding-Arb DSAC Stage 3 WF Report — {len(windows)} windows")
    print("=" * 78)
    for v in [g1, g2, g3, g4]:
        verdict_str = "PASS" if v["pass"] else "FAIL"
        extras = []
        if "passed_windows" in v:
            extras.append(f"{v['passed_windows']}/{v['total']} windows")
        if "median_pf" in v:
            extras.append(f"median_pf={v['median_pf']:.3f}")
        print(f"  {verdict_str:5} {v['name']:<35} {' | '.join(extras)}")
    if stress:
        print()
        print("Leverage-grid stress (EOD trailing DD, target ≤ %.2f%%):"
              % ((gates["gates"]["stress_leverage_grid"]["velotrade_cap_pct"] -
                  gates["gates"]["stress_leverage_grid"]["velotrade_buffer_pp"])))
        for agg, info in verdicts["stress_leverage_grid"]["by_aggregation"].items():
            row = " | ".join(f"L={d['leverage']}× {agg}={d[agg]*100:.2f}% "
                             f"{'OK' if d['safe'] else 'BREACH'}"
                             for d in info["per_leverage"])
            max_l = info["max_safe_leverage"]
            print(f"  [{agg}] max_safe_L={max_l}× | {row}")
    print()
    print(f"Decision: {decision}")
    print("=" * 78)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"wf_report_{timestamp}.json"
    out_path.write_text(json.dumps({
        "stage": 3,
        "workstream": "funding_arb_dsac",
        "config": str(args.config),
        "gates": str(args.gates),
        "wf_root": str(args.wf_root),
        "windows": windows,
        "per_window": per_window,
        "verdicts": verdicts,
        "decision": decision,
        "stress": stress,
        "timestamp": datetime.now().isoformat(),
    }, indent=2, default=str))
    log.info("Wrote %s", out_path)
    return 0 if hard_gates_pass else 1


if __name__ == "__main__":
    sys.exit(main())
