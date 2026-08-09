"""The power guard is calibrated on a FRESH LORD++ account; production runs on a depleted one.

WHY (S553-cont-153). The substrate-power stamp — the number that decides whether a substrate may be
mined at all, and the whole basis of the `us_equity` "adequately powered" claim — comes from the MDE
curve in `crucible_calibration.py`. That curve scores every planted panel at
``fresh_lord_level(corrected)``, and says so:

    "The LORD++ level is a FRESH account's first level ... a power curve is one hypothesis at a time,
     so the stream has not yet decayed."

Production is not one hypothesis at a time. The orchestrator charges one LORD++ test per PRE-REGISTERED
spec, the account PERSISTS across ticks, and `_tick_lord_level` hands `evolve` the TIGHTEST level of
the batch. On `us_equity` a single 8-spec tick has already moved the next level from the fresh
0.021874 (z 2.02, so `t_min` 2.33 binds) to **0.000650** (z 3.21, so the LORD++ leg binds and `t_min`
is slack). A second 40-spec batch would ask for z 4.08.

So the guard's MDE is measured at a threshold production does not use, the mismatch is one-directional
(levels only decay over a barren stream), and it grows with every test the substrate has ever run. If
that materially moves detection, then "the only adequately-powered substrate" is a claim with a
shelf-life, and mining it MORE makes it LESS able to detect anything — the opposite of the standing
plan.

WHAT THIS MEASURES. The E2 planted-signal power curve, at the `us_equity` holdout size, at several
LORD++ levels: the fresh level the guard assumes, the live level after 8 tests, and the batch-tightest
levels a 20/40-spec round would run at. Same planted machinery, same scorer, same everything else — the
level is the only thing that varies. Reported as detection rate vs plant strength, and as the
interpolated MDE (the smallest realized ΔSR detected at >= `--power` rate).

Pre-registered read: if MDE at the live level is within ~5% of MDE at the fresh level, the calibration
convention is harmless and this is a documentation fix. If it is materially worse, the power guard is
anti-conservative by construction and the `us_equity` GO needs restating at the true threshold.

Usage:
    python scripts/research/crucible_lord_depletion_power.py --t 4930 --holdout-frac 0.35
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from finrl_pro_ds.crucible.corrected_contract import (  # noqa: E402
    CorrectedConfig,
    corrected_contract_fitness,
    fresh_lord_level,
)
from finrl_pro_ds.crucible.orchestrator.fdr import OnlineFDR  # noqa: E402
from finrl_pro_ds.crucible.orchestrator.substrate import _power_holdout_bars  # noqa: E402
from finrl_pro_ds.signals.generation.evolve import _combined_book, _overlay_returns  # noqa: E402
from research.crucible_calibration import (  # noqa: E402
    _panel_ts,
    _planted_base_sleeves,
    _planted_panel,
    load_calib,
)

log = logging.getLogger("lord_depletion")


def batch_tightest_level(cc: CorrectedConfig, *, n_tests_already: int, batch: int) -> float:
    """The level `_tick_lord_level` would hand `evolve` for a batch of `batch` fresh specs on an
    account that has already run `n_tests_already` barren tests. Mirrors the orchestrator exactly:
    simulate the whole batch on a COPY assuming no discovery, take the MINIMUM."""
    acct = OnlineFDR(alpha=cc.fdr_alpha, w0=cc.fdr_w0)
    for _ in range(max(0, int(n_tests_already))):
        acct.observe(is_discovery=False)
    levels = []
    for _ in range(max(1, int(batch))):
        levels.append(acct.next_level())
        acct.observe(is_discovery=False)
    return float(min(levels))


def power_at_level(cc_calib, corrected: CorrectedConfig, *, t: int, n: int, hb: int,
                   betas: list[float], n_seeds: int, cost_bps: float,
                   lord_level: float) -> list[dict]:
    """E2 planted-signal detection at a FIXED LORD++ level. Identical to the corrected branch of
    `crucible_calibration._e2_power_curve` except that `lord_level` is an argument."""
    curve: list[dict] = []
    for beta in betas:
        detections = 0
        deltas: list[float] = []
        t_pass = lord_pass = 0
        for k in range(n_seeds):
            panel, s = _planted_panel(t, n, seed=5000 + k)
            base = _planted_base_sleeves(s, beta=beta, seed=5000 + k)
            ts = _panel_ts(panel)
            base_book = _combined_book(base, ts, cc_calib.fit_cfg)
            out = _overlay_returns("macro:plant", panel, base_book, cost_bps=cost_bps)
            if out is None:
                continue
            cand, _turn = out
            cand_ho = cand[t - hb:]
            base_ho = {k2: np.asarray(v)[t - hb:] for k2, v in base.items()}
            cr = corrected_contract_fitness(cand_ho, base_ho, ts[t - hb:], cc_calib.fit_cfg,
                                            corrected, lord_level=lord_level)
            detections += int(cr.passes_corrected)
            t_pass += int(cr.t_pass)
            lord_pass += int(cr.lord_pass)
            if np.isfinite(cr.delta_sr):
                deltas.append(float(cr.delta_sr))
        m = max(1, len(deltas))
        curve.append({"beta": beta, "power": detections / max(1, n_seeds),
                      "mean_realized_delta_sr": float(np.mean(deltas)) if deltas else float("nan"),
                      "t_leg_rate": t_pass / max(1, n_seeds),
                      "lord_leg_rate": lord_pass / max(1, n_seeds), "n_seeds": n_seeds,
                      "n_delta": m})
    return curve


def mde_from_curve(curve: list[dict], target_power: float) -> float:
    """Smallest realized ΔSR whose detection rate reaches `target_power`, linearly interpolated on
    (delta, power) between the bracketing points. +inf when the curve never reaches it — the same
    "unmeasured / undetectable" sentinel convention the power module uses."""
    pts = [(c["mean_realized_delta_sr"], c["power"]) for c in curve
           if np.isfinite(c["mean_realized_delta_sr"])]
    pts.sort()
    for i, (d, p) in enumerate(pts):
        if p >= target_power:
            if i == 0:
                return float(d)
            d0, p0 = pts[i - 1]
            if p == p0:
                return float(d)
            return float(d0 + (d - d0) * (target_power - p0) / (p - p0))
    return float("inf")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=int, default=4930, help="panel bars (us_equity: 4930)")
    ap.add_argument("--n", type=int, default=8, help="planted-panel width (calibration convention)")
    ap.add_argument("--holdout-frac", type=float, default=0.35)
    ap.add_argument("--n-seeds", type=int, default=60)
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--betas", default="0.0,0.002,0.004,0.006,0.008,0.012,0.016")
    ap.add_argument("--tests-already", type=int, default=8,
                    help="LORD++ tests already charged on the substrate (us_equity: 8)")
    ap.add_argument("--corrected-config",
                    default="configs/us_equity_corrected_contract.gates.yaml")
    ap.add_argument("--config", default="configs/us_equity_signal_eval.gates.yaml")
    ap.add_argument("--out", default="results/crucible_lord_depletion")
    args = ap.parse_args()

    cc_calib = load_calib(ROOT / "configs" / "crucible_calibration.gates.yaml",
                          ROOT / args.config)
    corrected = CorrectedConfig.from_yaml(ROOT / args.corrected_config)
    betas = [float(x) for x in args.betas.split(",")]
    hb = _power_holdout_bars(args.t, args.holdout_frac)

    levels = {
        "fresh (what the power guard calibrates at)": fresh_lord_level(corrected),
        f"live after {args.tests_already} tests, next single":
            batch_tightest_level(corrected, n_tests_already=args.tests_already, batch=1),
        f"live after {args.tests_already}, batch of 8":
            batch_tightest_level(corrected, n_tests_already=args.tests_already, batch=8),
        f"live after {args.tests_already}, batch of 40":
            batch_tightest_level(corrected, n_tests_already=args.tests_already, batch=40),
    }
    log.info("holdout bars %d (t=%d, frac=%.2f); %d seeds x %d betas per level",
             hb, args.t, args.holdout_frac, args.n_seeds, len(betas))

    results = {}
    for label, lvl in levels.items():
        log.info("--- LORD++ level %.6g  (%s)", lvl, label)
        curve = power_at_level(cc_calib, corrected, t=args.t, n=args.n, hb=hb, betas=betas,
                               n_seeds=args.n_seeds, cost_bps=cc_calib.ek["cost_bps"],
                               lord_level=lvl)
        mde = mde_from_curve(curve, args.power)
        results[label] = {"lord_level": lvl, "mde_delta_sr": mde, "curve": curve}
        for c in curve:
            log.info("  beta %.4f  power %.2f  realized dSR %+.3f  (t-leg %.2f, lord-leg %.2f)",
                     c["beta"], c["power"], c["mean_realized_delta_sr"], c["t_leg_rate"],
                     c["lord_leg_rate"])
        log.info("  => MDE at power %.2f: %.4f", args.power, mde)

    base_mde = results[next(iter(levels))]["mde_delta_sr"]
    print("\n=== MDE by LORD++ level (planted overlay, us_equity holdout size) ===")
    print(f"{'level':<44}{'p-thresh':>11}{'MDE dSR':>10}{'vs fresh':>10}")
    for label, r in results.items():
        ratio = (r["mde_delta_sr"] / base_mde) if np.isfinite(base_mde) and base_mde > 0 else float("nan")
        print(f"{label:<44}{r['lord_level']:>11.2e}{r['mde_delta_sr']:>10.3f}{ratio:>9.2f}x")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "lord_depletion_power.json").write_text(json.dumps(
        {"t": args.t, "holdout_bars": hb, "holdout_frac": args.holdout_frac,
         "n_seeds": args.n_seeds, "target_power": args.power,
         "tests_already_charged": args.tests_already, "results": results}, indent=2),
        encoding="utf-8")
    print(f"\nwrote {out / 'lord_depletion_power.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
