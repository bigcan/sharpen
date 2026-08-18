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
from finrl_pro_ds.crucible.orchestrator.substrate import (  # noqa: E402
    _power_holdout_bars,
    interp_mde,
)
from finrl_pro_ds.signals.generation.evolve import _combined_book, _overlay_returns  # noqa: E402
from research.crucible_calibration import (  # noqa: E402
    _panel_ts,
    _planted_base_sleeves,
    _planted_panel,
    _planted_xsec_panel,
    _proxy_base_sleeves,
    _xsec_candidate_returns,
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


def xsec_pvalues(cc_calib, corrected: CorrectedConfig, *, t: int, n: int, betas: list[float],
                 n_seeds: int, cost_bps: float, holdout_frac: float, hold_horizon: int,
                 min_names: int) -> list[dict]:
    """Score the CROSS-SECTIONAL planted fixture ONCE per draw and return the per-draw components.

    Identical to `crucible_calibration._xsec_power_curve` — same planted per-name characteristic, same
    oracle rank-L/S book, same holdout window, same scorer — except that it does not collapse to a
    detection rate. The LORD++ level enters `corrected_contract_fitness` ONLY through
    `lord_pass = p_value <= lord_level`; every other leg, and the p-value itself, is level-independent.
    So one scoring pass supports every level, and re-thresholding is EXACT rather than an
    approximation — which also removes seed noise from the between-level comparison (the levels are
    compared on the SAME draws)."""
    out: list[dict] = []
    hb = _power_holdout_bars(t, holdout_frac)
    for beta in betas:
        for k in range(n_seeds):
            panel, x = _planted_xsec_panel(t, n, seed=7000 + k, beta=beta)
            base = _proxy_base_sleeves(panel, hold=hold_horizon)
            ts = _panel_ts(panel)
            cand, _turn = _xsec_candidate_returns(x, panel, hold_horizon=hold_horizon,
                                                  cost_bps=cost_bps, min_names=min_names)
            base_ho = {k2: np.asarray(v)[t - hb:] for k2, v in base.items()}
            # Score at lord_level = 1.0, which makes `lord_pass` unconditionally True for any valid
            # p-value, so `passes_corrected` IS the conjunction of every OTHER leg. Deriving it that
            # way rather than re-ANDing the leg flags by hand keeps this exact if the contract ever
            # grows a leg — `exposure_pass` (the cont-151 market-beta guard) is already a leg this
            # script never names, and it is inert here only because `market_returns` is None.
            cr = corrected_contract_fitness(cand[t - hb:], base_ho, ts[t - hb:], cc_calib.fit_cfg,
                                            corrected, lord_level=1.0)
            out.append({
                "beta": beta, "seed": 7000 + k, "p_value": float(cr.p_value),
                "delta_sr": float(cr.delta_sr),
                "non_lord_pass": bool(cr.passes_corrected),   # every leg but LORD++ (see above)
                "t_pass": bool(cr.t_pass),
            })
    return out


def xsec_curve_at_level(draws: list[dict], betas: list[float], lord_level: float) -> list[dict]:
    """Collapse pre-scored draws to a `mde_sweep`-shaped power curve at one LORD++ level."""
    curve: list[dict] = []
    for beta in betas:
        rows = [d for d in draws if d["beta"] == beta]
        det = sum(1 for d in rows
                  if d["non_lord_pass"] and np.isfinite(d["p_value"]) and d["p_value"] <= lord_level)
        deltas = [d["delta_sr"] for d in rows if np.isfinite(d["delta_sr"])]
        curve.append({"beta": beta, "power": det / max(1, len(rows)),
                      "mean_realized_delta_sr": float(np.mean(deltas)) if deltas else float("nan"),
                      "t_leg_rate": sum(d["t_pass"] for d in rows) / max(1, len(rows)),
                      "n_seeds": len(rows)})
    return curve


def first_point_mde(curve: list[dict], target_power: float) -> float | None:
    """The shipped `_mde_from_curve` convention: the FIRST point in beta order whose power reaches the
    target, reported as its mean realized ΔSR. Deliberately NOT the interpolating variant used for the
    overlay path below — this feeds `interp_mde`, which must receive rows built the same way the
    shipped sweep built them or the comparison is against a different estimator."""
    for p in curve:
        if float(p["power"]) >= target_power:
            return float(p["mean_realized_delta_sr"])
    return None


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


def run_cross_sectional(args, corrected: CorrectedConfig, levels: dict[str, float]) -> dict:
    """Re-measure the depletion ratio on the CROSS-SECTIONAL curve — the one `us_equity` is actually
    gated by — and read the implied MDE through the guard's OWN accessor (`interp_mde`).

    The overlay measurement is not transportable here: `crucible-v7.0` exists precisely because the two
    paths have different power laws (an overlay sees T observations, a cross-sectional book sees T x N).
    So this reproduces the SHIPPED xsec sweep's own parameters (holdout_frac / hold_horizon /
    ls_min_names / beta grid / n_seeds from the artifact, NOT us_equity's), re-derives its rows at each
    LORD++ level, and then feeds them back through `_pooled_points` + `interp_mde` at the substrate's
    real holdout depth. Anything less than that is a number that looks like the gate's but is not.

    Only the two grid depths BRACKETING the substrate's holdout are re-measured; `interp_mde` anchors
    the interpolation on exactly those two, so deeper/shallower rows cannot move the answer."""
    shipped = json.loads((ROOT / args.xsec_sweep).read_text(encoding="utf-8"))
    sw = shipped["mde_sweep"]
    cc_calib = load_calib(ROOT / "configs" / "crucible_calibration.gates.yaml",
                          ROOT / args.config)

    # The shipped beta grid QUANTIZES the answer: `first_point_mde` reports the realized ΔSR at the
    # first grid beta reaching target power, and the steps are wide (n=100 at holdout 1011 jumps
    # 1.398 -> 2.001 between adjacent betas). A level change that moves the crossing by less than one
    # step reads as NO change; one that crosses a step reads as a jump. `--xsec-betas` refines the grid
    # for EVERY level identically, so the ratio is estimated at higher resolution — at the cost of the
    # shipped grid's exact reproduction of the stamp, which is why the default keeps the shipped grid.
    betas = ([float(x) for x in args.xsec_betas.split(",")] if args.xsec_betas
             else [float(b) for b in sw["beta_grid"]])
    n_grid = [int(x) for x in sw["n_grid"]]
    t_grid = sorted(int(x) for x in sw["t_grid"])
    hf, target = float(sw["holdout_frac"]), float(sw["power_target"])
    hb_target = _power_holdout_bars(args.t, args.holdout_frac)

    # the two measured depths the guard interpolates between at `hb_target`
    depths = [t for t in t_grid if _power_holdout_bars(t, hf) <= hb_target][-1:] + \
             [t for t in t_grid if _power_holdout_bars(t, hf) > hb_target][:1]
    log.info("substrate holdout %d bars; re-measuring the bracketing sweep depths T=%s "
             "(holdout %s) x n=%s", hb_target, depths,
             [_power_holdout_bars(t, hf) for t in depths], n_grid)
    log.info("shipped sweep params: holdout_frac=%.2f hold=%d min_names=%d betas=%d seeds=%d",
             hf, sw["hold_horizon"], sw["ls_min_names"], len(betas), sw["n_seeds"])

    n_seeds = int(args.xsec_seeds or sw["n_seeds"])
    draws: dict[tuple[int, int], list[dict]] = {}
    for t in depths:
        for n in n_grid:
            log.info("  scoring T=%d n=%d (%d betas x %d seeds)...", t, n, len(betas), n_seeds)
            draws[(t, n)] = xsec_pvalues(
                cc_calib, corrected, t=t, n=n, betas=betas, n_seeds=n_seeds,
                cost_bps=cc_calib.ek["cost_bps"], holdout_frac=hf,
                hold_horizon=int(sw["hold_horizon"]), min_names=int(sw["ls_min_names"]))

    out: dict = {}
    for label, lvl in levels.items():
        rows = []
        for (t, n), d in draws.items():
            curve = xsec_curve_at_level(d, betas, lvl)
            rows.append({"t": t, "n": n, "holdout_bars": _power_holdout_bars(t, hf),
                         "mde_realized_delta_sr": first_point_mde(curve, target),
                         "curve": curve})
        # read the implied MDE the way the GUARD reads it: pool worst-across-n, interpolate in 1/sqrt(h)
        implied, mode = interp_mde(hb_target, {"mde_sweep": {"rows": rows}})
        out[label] = {"lord_level": lvl, "implied_mde_delta_sr": implied, "interp_mode": mode,
                      "rows": [{k: v for k, v in r.items() if k != "curve"} for r in rows],
                      "curves": {f"t{r['t']}_n{r['n']}": r["curve"] for r in rows}}
        log.info("LORD++ %.3e (%s): implied MDE at holdout %d = %.4f (%s)",
                 lvl, label, hb_target, implied, mode)
    return {"path": "cross_sectional", "substrate_holdout_bars": hb_target,
            "sweep_source": args.xsec_sweep, "n_seeds": n_seeds, "depths": depths,
            "n_grid": n_grid, "results": out}


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
    ap.add_argument("--path", choices=("overlay", "cross_sectional"), default="overlay",
                    help="which gate's power curve to measure; us_equity MINES cross_sectional")
    # The path the GUARD reads (`power_guard.calibration_sweep_path_corrected_xsec` in
    # configs/us_equity_power.gates.yaml) — not the `crucible_calibration/` sibling. The two differ
    # only by the cont-150 deep-grid extension (t 64512/129024) and carry BYTE-IDENTICAL rows at the
    # depths that anchor us_equity, so both interpolate to 1.3120527546583958 at holdout 1726; the
    # gate's own file is used anyway, because "read the number through the gate's accessor" is exactly
    # the rule this whole line of work exists to enforce.
    ap.add_argument("--xsec-sweep",
                    default=("results/crucible_calibration_union/"
                             "calibration_xsec_mde_sweep_corrected.json"),
                    help="the shipped cross-sectional MDE surface whose rows are re-derived")
    ap.add_argument("--xsec-seeds", type=int, default=0,
                    help="override the shipped sweep's n_seeds (0 = use it)")
    ap.add_argument("--xsec-betas", default="",
                    help="comma-separated beta grid override; refines the quantized MDE estimate "
                         "(empty = the shipped sweep's own grid, which reproduces the stamp exactly)")
    ap.add_argument("--xsec-out", default="lord_depletion_power_xsec.json")
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

    if args.path == "cross_sectional":
        payload = run_cross_sectional(args, corrected, levels)
        base = next(iter(payload["results"].values()))["implied_mde_delta_sr"]
        print("\n=== IMPLIED MDE by LORD++ level — CROSS-SECTIONAL path, read via interp_mde ===")
        print(f"{'level':<44}{'p-thresh':>11}{'implied MDE':>13}{'vs fresh':>10}")
        for label, r in payload["results"].items():
            m = r["implied_mde_delta_sr"]
            ratio = (m / base) if np.isfinite(base) and base > 0 and np.isfinite(m) else float("nan")
            print(f"{label:<44}{r['lord_level']:>11.2e}{m:>13.3f}{ratio:>9.2f}x")
        out = ROOT / args.out
        out.mkdir(parents=True, exist_ok=True)
        (out / args.xsec_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {out / args.xsec_out}")
        return 0

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
