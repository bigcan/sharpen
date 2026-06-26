#!/usr/bin/env python
"""C1.4 — OOS sleeve-combiner selection (the ``combiner_beats_static`` deploy decision).

Sweeps the dynamic combiner's (λ=``tilt_strength``, ``perf_window``, ``perf_metric``) grid and
decides whether the DYNAMIC perf-tilt combiner beats the FROZEN static inverse-vol combiner
OOS by the pre-registered ``gates.combiner_beats_static.min_uplift_vs_baseline`` (ADR-C1-4),
else ``ship_static_rp``. Reuses :class:`~finrl_pro_ds.paper.TwoSleeveExecutor` end-to-end on
the same two-sleeve data as ``run_cross_asset_paper_validation.py``.

**Honest selection (anti-overfit):** the grid is selected on an IN-SAMPLE front split and the
reported uplift is measured on a HELD-OUT OOS tail — both static and dynamic are run on the
SAME tail, so the comparison is fair and the best-of-grid optimism does not leak into the
verdict. The single-render OOS point is a SCREEN, not a distribution: the *binding* deploy
verdict requires Component-2 CPCV (distribution-valued OOS) + a Tier-2 deep lifecycle audit.

**NOT a capital promotion.** This computes + serializes an ADVISORY verdict only; capital stays
BLOCKED (DSR 0.918 < 0.95). At N=2 sleeves the gate is EXPECTED to often say ``ship_static_rp``
— the cross-sectional tilt is near-degenerate — which is the honest, correct outcome and the
whole point of the inverted-emphasis thesis. Reading this verdict as a paper-promotion gate
triggers the Tier-2 audit per CLAUDE.md.

Usage:
  python scripts/run_combiner_selection.py                       # live: freshness-gated
  python scripts/run_combiner_selection.py --backtest            # reproducible window
  python scripts/run_combiner_selection.py --split_frac 0.7 --metric sharpe
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml

from finrl_pro_ds.paper import TwoSleeveExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("combiner_selection")

ANN = 252
# Default sweep grid (cost-aware: kept small — the filter, not idea supply, is the bottleneck).
_DEFAULT_TILTS = (0.5, 1.0, 2.0, 3.0, 5.0)
_DEFAULT_PERF_WINDOWS = (63, 126, 252)


def _net_sharpe(step_returns: np.ndarray) -> float:
    """Net annualized Sharpe of a combined-book step-return series (ddof=1, √252) — the same
    convention as ``allocator_factory._metrics`` / the crypto eval. The book is post-cost
    (the replay applies fills/fees), so this is NET."""
    r = np.asarray(step_returns, dtype=np.float64)
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ANN)) if sd > 1e-12 else 0.0


def _with_combiner(config: Mapping, block: dict) -> dict:
    out = dict(config)
    out["sleeve_combiner"] = block
    return out


def _static_cfg(config: Mapping) -> dict:
    return _with_combiner(config, {"mode": "inverse_vol"})


def _dynamic_cfg(config: Mapping, *, tilt_strength: float, perf_window: int,
                 perf_metric: str, tilt_clip: float = 1.5) -> dict:
    return _with_combiner(config, {
        "mode": "dynamic", "tilt_strength": float(tilt_strength),
        "perf_window": int(perf_window), "perf_min_periods": min(63, int(perf_window)),
        "perf_metric": str(perf_metric), "tilt_clip": float(tilt_clip)})


def _slice_bundle(bundle: Mapping, lo: int, hi: int) -> dict:
    """Slice every time-indexed array of a two-sleeve bundle to rows ``[lo:hi]`` (``assets``
    lists pass through). Both static and dynamic runs use the same slice, so the held-out
    comparison is fair even though trailing-vol warmup restarts at ``lo``."""
    def _sl(d: Mapping) -> dict:
        return {k: (list(v) if k == "assets" else np.asarray(v)[lo:hi]) for k, v in d.items()}
    return {s: _sl(bundle[s]) for s in bundle}


def _run_net_sharpe(config: Mapping, bundle: Mapping) -> float:
    live, _ = TwoSleeveExecutor(config).run(bundle)
    return _net_sharpe(live.step_returns)


def select_combiner(
    bundle: Mapping,
    config: Mapping,
    gates_cfg: Mapping,
    *,
    split_frac: float = 0.7,
    tilts: Sequence[float] = _DEFAULT_TILTS,
    perf_windows: Sequence[int] = _DEFAULT_PERF_WINDOWS,
    perf_metric: str = "sharpe",
) -> dict:
    """Select the dynamic combiner on the in-sample front, evaluate the ``combiner_beats_static``
    gate on the held-out OOS tail. Returns the advisory verdict dict (JSON-safe)."""
    gate = dict(gates_cfg.get("gates", {}).get("combiner_beats_static", {}))
    if not gate:
        raise KeyError("gates.combiner_beats_static missing — add the ADR-C1-4 gate "
                       "(configs/cross_asset_momentum.gates.yaml)")
    if "min_uplift_vs_baseline" not in gate:
        # gates-not-hardcoded: the numeric bar MUST come from the gates file, never a default.
        raise KeyError("gates.combiner_beats_static.min_uplift_vs_baseline missing — the "
                       "deploy bar must be pre-registered in the gates file, not defaulted")
    min_uplift = float(gate["min_uplift_vs_baseline"])
    else_action = str(gate.get("else", "ship_static_rp"))     # safe fail-action default (static)

    T = len(np.asarray(bundle["union"]["timestamps"]))
    split = int(T * float(split_frac))
    is_bundle = _slice_bundle(bundle, 0, split)     # in-sample (selection)
    oos_bundle = _slice_bundle(bundle, split, T)    # held-out (evaluation)

    # 1) select λ/perf_window on IN-SAMPLE.
    is_results = []
    for pw in perf_windows:
        for lam in tilts:
            cfg = _dynamic_cfg(config, tilt_strength=lam, perf_window=pw, perf_metric=perf_metric)
            is_results.append({"tilt_strength": float(lam), "perf_window": int(pw),
                               "perf_metric": perf_metric,
                               "is_net_sharpe": _run_net_sharpe(cfg, is_bundle)})
    best = max(is_results, key=lambda d: d["is_net_sharpe"])

    # 2) evaluate the SELECTED config + the static baseline on the HELD-OUT OOS tail.
    static_oos = _run_net_sharpe(_static_cfg(config), oos_bundle)
    sel_cfg = _dynamic_cfg(config, tilt_strength=best["tilt_strength"],
                           perf_window=best["perf_window"], perf_metric=perf_metric)
    dyn_oos = _run_net_sharpe(sel_cfg, oos_bundle)
    uplift = dyn_oos - static_oos
    decision = "ship_dynamic" if uplift >= min_uplift else else_action

    return {
        "gate": "combiner_beats_static",
        "metric": "oos_net_sharpe",
        "selection_method": "in_sample_grid_then_held_out_oos",
        "advisory": True,                       # NOT a capital promotion (see module docstring)
        "binding_requires": "C2 CPCV distribution + Tier-2 deep lifecycle audit",
        "split_frac": float(split_frac),
        "n_bars_total": int(T),
        "n_bars_in_sample": int(split),
        "n_bars_oos": int(T - split),
        "selected": {k: best[k] for k in ("tilt_strength", "perf_window", "perf_metric")},
        "selected_is_net_sharpe": float(best["is_net_sharpe"]),
        "static_inverse_vol_oos_net_sharpe": float(static_oos),
        "dynamic_oos_net_sharpe": float(dyn_oos),
        "uplift_vs_static": float(uplift),
        "min_uplift_vs_baseline": float(min_uplift),
        "decision": decision,
        "in_sample_grid": is_results,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/live_cross_asset_paper.yaml")
    ap.add_argument("--gates", default=None, help="gates yaml (default: ensemble.gates_file)")
    ap.add_argument("--out", default="results/cross_asset_paper/combiner_selection_verdict.json")
    ap.add_argument("--split_frac", type=float, default=0.7)
    ap.add_argument("--metric", default="sharpe", choices=("sharpe", "sortino"))
    ap.add_argument("--force_refetch", action="store_true")
    ap.add_argument("--backtest", action="store_true",
                    help="reproducible historical window (disable the freshness gate)")
    args = ap.parse_args()

    # Imported here so the pure select_combiner() (and its unit test) need no data layer.
    from finrl_pro_ds.data.cross_asset_loader import (
        build_two_sleeve_arrays,
        load_two_sleeve_data,
    )

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    gates_path = args.gates or config["ensemble"]["gates_file"]
    gates_cfg = yaml.safe_load(Path(gates_path).read_text(encoding="utf-8"))

    require_fresh = not args.backtest
    data = load_two_sleeve_data(config, force_refetch=args.force_refetch,
                                require_fresh=require_fresh)
    start_ts, end_ts = data["close"].index[0], data["close"].index[-1]
    log.info("data range %s -> %s (%d bars)", start_ts.date(), end_ts.date(), len(data["close"]))
    bundle = build_two_sleeve_arrays(data, start_ts, end_ts)

    verdict = select_combiner(bundle, config, gates_cfg,
                              split_frac=args.split_frac, perf_metric=args.metric)
    verdict["validation_meta"] = {
        "config": str(args.config), "gates": str(gates_path),
        "data_start": str(start_ts.date()), "data_end": str(end_ts.date()),
        "require_fresh": require_fresh,
    }
    log.warning("ADVISORY verdict (NOT a capital promotion): decision=%s  uplift=%.3f  "
                "(static_oos=%.3f -> dynamic_oos=%.3f)  binding verdict needs CPCV + Tier-2",
                verdict["decision"], verdict["uplift_vs_static"],
                verdict["static_inverse_vol_oos_net_sharpe"], verdict["dynamic_oos_net_sharpe"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
