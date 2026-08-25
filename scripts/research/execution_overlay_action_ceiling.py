#!/usr/bin/env python
"""What is the BEST uplift any policy could achieve in the overlay's action space?

The question this answers should be asked BEFORE training an agent against a gate, not after.
The RL execution overlay was built (June 2026), debugged through two real defects, and run for
ten GPU-seeds before anyone measured whether its ``execution_beats_baseline`` floor was
reachable at all. It was not: the entire action space spans +1.704 bps of uplift against a
2.0 bps floor (randd_log S553-cont-168).

Method — no training, no GPU. ``ExecutionSchedulerEnv``'s control is a scalar urgency
multiplier ``m`` on the closed-loop TWAP slice, and the realised schedule is monotone in it, so
sweeping CONSTANT actions brackets every policy a state-dependent agent could learn:

    m = 0            full pause; the forced phi=1 at h=H-1 dumps the parent in one bar
    m = 1            exactly the TWAP baseline the gate scores against (a = 0)
    m -> large       phi = 1 at bar 0, i.e. SNAP -- what the existing snap executor already does

If net-IS is monotone decreasing in ``m`` the optimum is snap, the overlay's premise (trade
slower to save impact) is inverted on that book, and no agent can pass. That is exactly what a
book with ~1e-5 participation gives you: no impact to schedule around, so shortfall is pure
timing drift and any delay is a pure loss.

    python scripts/research/execution_overlay_action_ceiling.py \
        --config configs/execution_overlay_sac_seedcheck.yaml

Exits 1 when the gate floor exceeds the achievable ceiling -- i.e. the gate is unreachable by
construction, which is a config/design defect, not a training outcome.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

log = logging.getLogger("action_ceiling")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/execution_overlay_sac_seedcheck.yaml")
    ap.add_argument("--train_frac", type=float, default=0.7,
                    help="must match the runner's, so the window is the one the gate scores")
    ap.add_argument("--snap_urgency", type=float, default=10.0,
                    help="urgency_max override large enough that phi=1 at bar 0 (=> snap)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from finrl_pro_ds.data.cross_asset_loader import (
        build_two_sleeve_arrays,
        load_two_sleeve_data,
    )
    from finrl_pro_ds.envs.execution_overlay_factory import (
        drive_execution_episodes,
        make_execution_env,
    )
    from finrl_pro_ds.paper import TwoSleeveExecutor
    from scripts.execution_overlay_runner import (
        effective_gates,
        load_overlay_config,
        temporal_split_bundle,
    )

    cfg = load_overlay_config(args.config)
    gates = effective_gates(cfg, None)
    ebb = dict(gates.get("execution_beats_baseline", {}) or {})
    floor = float(ebb.get("min_uplift_bps", 2.0))

    data = load_two_sleeve_data(cfg, force_refetch=False)
    bundle = build_two_sleeve_arrays(data, data["close"].index[0], data["close"].index[-1])
    _, detail = TwoSleeveExecutor(cfg).sim_oracle(bundle)
    _, _, test_bundle, test_target, _ = temporal_split_bundle(
        bundle, detail["combined_w"], args.train_frac)

    apply_pf = bool(cfg.get("prop_firm", {}).get("augment_obs", False))
    ov = dict(cfg.get("execution_overlay", {}) or {})
    u_min, u_max = float(ov.get("urgency_min", 0.0)), float(ov.get("urgency_max", 2.0))

    def drive(action: float, urgency_max: float) -> dict:
        env = make_execution_env(test_bundle, cfg, target_weights=test_target, eval_mode=True,
                                 apply_prop_firm=apply_pf,
                                 overrides={"urgency_max": urgency_max})
        return drive_execution_episodes(
            env, lambda _k, _o, a=action: np.array([a], dtype=np.float32))

    baseline = drive(0.0, u_max)["net_is_bps"]          # a=0 => m=1 => the gate's TWAP baseline
    log.info("baseline (a=0, exact TWAP): net-IS %.3f bps", baseline)
    log.info("%12s %10s %16s", "m (urgency)", "net_IS", "uplift vs base")

    rows = []
    for a in (-1.0, -0.5, 0.0, 0.5, 1.0):
        m = u_min + (a + 1.0) * 0.5 * (u_max - u_min)
        r = drive(a, u_max)
        rows.append((m, baseline - r["net_is_bps"]))
        log.info("%12.2f %10.3f %+16.3f", m, r["net_is_bps"], baseline - r["net_is_bps"])

    snap = drive(1.0, args.snap_urgency)["net_is_bps"]
    ceiling = baseline - snap
    log.info("%12s %10.3f %+16.3f   <- snap (phi=1 at bar 0)", "snap", snap, ceiling)

    monotone = all(rows[i][1] <= rows[i + 1][1] + 1e-9 for i in range(len(rows) - 1))
    log.info("")
    log.info("uplift monotone increasing in urgency : %s", monotone)
    log.info("in-action-space max uplift            : %+.3f bps", rows[-1][1])
    log.info("absolute ceiling (snap)               : %+.3f bps", ceiling)
    log.info("gate floor (execution_beats_baseline) : %+.3f bps", floor)

    if ceiling < floor:
        log.error("")
        log.error("UNREACHABLE: the gate floor (%.3f) exceeds the best uplift ANY policy in this "
                  "action space can achieve (%.3f = %.0f%% of it). No training outcome can pass; "
                  "this is a design/config defect, not a result.", floor, ceiling,
                  ceiling / floor * 100)
        if monotone:
            log.error("Monotone in urgency => the optimum is SNAP, which the existing snap "
                      "executor already does. The overlay's premise (trade slower to save "
                      "impact) is inverted on this book -- check participation: near-zero "
                      "participation means there is no impact to schedule around.")
        return 1
    log.info("reachable: ceiling clears the floor with %.3f bps to spare", ceiling - floor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
