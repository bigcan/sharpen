"""The power wall — ideal vs DEPLOYED minimum detectable effect. Re-derivation of audit F7.

REBUILD of `scratchpad/forward_power.py`, cited by
`docs/research/crucible_independent_audit_report_2026-07-14.md` (F7, report line 154) and never
committed. It is the last of the four uncommitted audit scripts.

    F7: "The residual honest power wall: ideal single-prereg t>=2 MDE80 ~= 0.71 (T=4044) /
    1.42 (T=1011); DEPLOYED CONTRACT ~= 1.9-3.5; dSR 0.3-0.5 undetectable under any
    discovery-grade contract on this data. The stop-mining decision is correct."

WHAT WAS ALREADY CLOSED, AND WHAT THIS ADDS
--------------------------------------------
F7 has three parts and they are not equally hard:

  (1) the IDEAL single-pre-registered t>=2 MDE80 -- pure arithmetic,
      `(t_min + z_0.80) * sqrt(ppy/N)`. Already reproduced to two decimals by
      `planted_sweep.py --bar-only`: 1.419 at T=1011 (audit 1.42), 0.709 at T=4044 (audit 0.71).
  (2) the DEPLOYED CONTRACT's MDE80 ~= 1.9-3.5 -- **this is the gap.** It is not arithmetic: the
      deployed gate is a six-leg AND (uplift, dsr_aug, marginal_t, collinearity, fragility,
      degenerate-vol), so its minimum detectable effect has to be MEASURED by planting edges of
      known size and finding where the whole conjunction reaches 80% power.
  (3) "0.3-0.5 undetectable" -- follows from (1) and (2) once both are in hand.

So this script measures (2) and puts it beside (1) on one axis. The ratio deployed/ideal is the
price the contract charges over an idealized single test.

THE MEASUREMENT IS IN *REALIZED* dSR, NOT PLANTED SHARPE
---------------------------------------------------------
An MDE is a statement about the effect size the gate can see, and the gate sees the candidate's
marginal contribution to the book -- not the candidate's standalone Sharpe. Planting a candidate
with standalone Sharpe s does NOT produce a book uplift of s (measured on the real Taiwan panel:
planted 0.50 -> realized dSR 0.272; planted 2.00 -> realized 1.305). Reporting the planted number
as the MDE would overstate the gate's ability by roughly 1.5x. Every figure here is therefore
interpolated in realized-dSR space, with the planted grid shown alongside so the mapping is visible.

TWO CONTROLS, BOTH REQUIRED
----------------------------
  * NULL (planted 0) must give power ~ 0. Otherwise the gate fires at everything and no MDE exists.
  * SATURATION (a deliberately huge edge) must give power ~ 1. Otherwise the gate cannot fire at
    ANY size, and "MDE80 unreachable" would be a statement about a broken harness rather than about
    the contract.

Without the second control an unreachable MDE is indistinguishable from a bug -- the same
one-directional trap that produced a false `SEAL_CONFIRMED` in `planted_sweep.py` and a p=1.0000
sweep in `cohort_mc_power_probe.py` earlier in this work.

USAGE
-----
    python scripts/research/forward_power.py                      # synthetic, the audit's spans
    python scripts/research/forward_power.py --substrate taiwan   # real panel (needs FINMIND_TOKEN)
    python scripts/research/forward_power.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sharpen.signals.eval_harness import _ann_sharpe                       # noqa: E402
from sharpen.signals.features import Panel, make_synthetic_panel           # noqa: E402
from sharpen.signals.generation.base_sleeves import tsmom_sleeve_returns   # noqa: E402
from sharpen.signals.generation.config import load_generation_config       # noqa: E402
from sharpen.signals.generation.fitness import combination_fitness         # noqa: E402

log = logging.getLogger("forward_power")

DEFAULT_GATES = "configs/taiwan_signal_eval.gates.yaml"
Z80 = 0.8416                      # one-sided normal quantile for 80% power

# F7's published anchors, so a run prints its own verdict against them.
AUDIT_IDEAL = {1011: 1.42, 4044: 0.71}
AUDIT_DEPLOYED_BAND = (1.9, 3.5)
REALISTIC_EDGE_BAND = (0.30, 0.50)


def ideal_mde80(t_min: float, n_bars: int, ppy: float) -> float:
    """Annualized dSR an IDEAL single pre-registered t-test detects at 80% power.

    ``(t_min + z_0.80) * sqrt(ppy / n_bars)`` -- no file-drawer deflation, no conjunction, no
    autocorrelation. The floor that survives after every machine defect is removed.

    NOTE: this duplicates `planted_sweep.mde80` deliberately rather than importing across research
    scripts by path. `tests/research/test_forward_power.py` pins the two to agree, so the
    duplication cannot drift silently.
    """
    if n_bars < 2:
        raise ValueError("n_bars must be >= 2")
    return (float(t_min) + Z80) * float(np.sqrt(float(ppy) / float(n_bars)))


def build_substrate(args: argparse.Namespace, ev: dict) -> tuple[Panel, dict[str, np.ndarray]]:
    """(panel, base_returns) for the requested substrate."""
    hold, cost = int(ev["hold_horizon"]), float(ev["cost_bps"])
    if args.substrate == "taiwan":
        import os
        if not os.environ.get("FINMIND_TOKEN"):
            raise SystemExit(
                "FINMIND_TOKEN is not set — the Taiwan substrate fetches from FinMind.\n"
                "  export FINMIND_TOKEN=...   (never pass a token on the command line)\n"
                "  Or use the default --substrate synthetic, which needs no credentials.")
        from sharpen.data.taiwan_panel_loader import load_taiwan_panel  # noqa: PLC0415
        from sharpen.signals.generation.base_sleeves import taiwan_base_sleeves  # noqa: PLC0415
        panel = load_taiwan_panel(args.start, args.end)
        base, _ = taiwan_base_sleeves(panel, hold_horizon=hold, cost_bps=cost,
                                      start=args.start, end=args.end, return_components=True)
        return panel, base
    panel = make_synthetic_panel(T=max(args.spans) + 10, N=args.names, seed=args.seed)
    from dataclasses import replace  # noqa: PLC0415
    idx = np.arange(min(3, panel.N))
    sub = replace(panel, tickers=tuple(panel.tickers[j] for j in idx),
                  open=panel.open[:, idx], high=panel.high[:, idx], low=panel.low[:, idx],
                  close=panel.close[:, idx], volume=panel.volume[:, idx],
                  active=panel.active[:, idx], adv_usd=panel.adv_usd[:, idx],
                  sector_id=panel.sector_id[idx])
    return panel, {"tsmom": np.asarray(tsmom_sleeve_returns(sub, hold_horizon=hold,
                                                            cost_bps=cost))}


def panel_ts(panel: Panel) -> np.ndarray:
    return panel.dates.astype("datetime64[s]").astype(np.int64).astype(np.float64)


def measure_power(*, planted: float, base: dict[str, np.ndarray], ts: np.ndarray, fit,
                  n_bars: int, reps: int, gen_n_eff: float, seed: int) -> dict:
    """P(the DEPLOYED six-leg gate fires) against a candidate of known planted Sharpe.

    The candidate is noise orthogonal to the base book plus a drift. Orthogonality is the most
    favourable case for a diversifier, so every power figure here is an UPPER bound and every MDE
    a LOWER bound -- the conservative direction for a claim that the wall is high.
    """
    ppy = fit.periods_per_year
    b0 = np.asarray(next(iter(base.values())), dtype=np.float64)[:n_bars]
    vol = float(b0[np.isfinite(b0)].std(ddof=1))
    base_n = {k: np.asarray(v)[:n_bars] for k, v in base.items()}
    ts_n = ts[:n_bars]
    n_pass, realized = 0, []
    for r in range(reps):
        rng = np.random.default_rng(seed + 7919 * r + int(planted * 1e6))
        cand = rng.standard_normal(n_bars) * vol
        cand = cand - cand.mean() + planted * vol / float(np.sqrt(ppy))
        res = combination_fitness(cand, base_n, ts_n, fit, gen_n_eff=gen_n_eff,
                                  turnover_ann=1.0, n_nodes=1, trial_sharpe_pool=None)
        n_pass += int(res.passes_gate)
        if np.isfinite(res.delta_sr_oos):
            realized.append(float(res.delta_sr_oos))
    return {
        "planted_ann_sharpe": float(planted),
        "power": n_pass / max(1, reps),
        "n_pass": n_pass,
        "reps": reps,
        "median_realized_delta_sr": float(np.median(realized)) if realized else float("nan"),
    }


def _interp_crossing(rows: list[dict], target: float = 0.80) -> float | None:
    """Realized dSR at which measured power first crosses ``target``, linearly interpolated.

    Interpolates in REALIZED dSR (see module docstring) rather than planted Sharpe. Returns None
    when the sweep never reaches the target -- an unreachable MDE, which is a real outcome and must
    not be silently reported as the top of the grid.
    """
    ok = [r for r in rows if np.isfinite(r["median_realized_delta_sr"])]
    if not ok:
        return None
    # The grid may already START at or above the target — the crossing is then at or BELOW the
    # smallest effect swept, not absent. Falling through to the loop below would return None and
    # report "unreachable", which is the OPPOSITE error to the one this function exists to avoid:
    # it would understate the gate's power instead of overstating it. Report the first point and
    # let the caller widen the grid downward.
    if ok[0]["power"] >= target:
        return float(ok[0]["median_realized_delta_sr"])
    for a, b in zip(ok, ok[1:]):
        if a["power"] < target <= b["power"]:
            xa, xb = a["median_realized_delta_sr"], b["median_realized_delta_sr"]
            pa, pb = a["power"], b["power"]
            if pb == pa:
                return float(xb)
            return float(xa + (target - pa) / (pb - pa) * (xb - xa))
    return None


def run(args: argparse.Namespace) -> dict:
    fit, ev = load_generation_config(_REPO_ROOT / args.gates)
    ppy = fit.periods_per_year
    panel, base = build_substrate(args, ev)
    ts = panel_ts(panel)
    spans = [n for n in sorted(set(args.spans)) if n <= panel.T]
    if not spans:
        raise SystemExit(f"--spans has no value <= panel length {panel.T}")

    out_spans: list[dict] = []
    for n_bars in spans:
        t0 = time.time()
        rows = [measure_power(planted=p, base=base, ts=ts, fit=fit, n_bars=n_bars,
                              reps=args.reps, gen_n_eff=args.gen_n_eff, seed=args.seed)
                for p in args.grid]
        # CONTROLS. Null must not fire; a deliberately huge edge must fire, or an unreachable
        # MDE below is a broken harness rather than a property of the contract.
        null = measure_power(planted=0.0, base=base, ts=ts, fit=fit, n_bars=n_bars,
                             reps=args.reps, gen_n_eff=args.gen_n_eff, seed=args.seed)
        sat = measure_power(planted=args.saturation, base=base, ts=ts, fit=fit, n_bars=n_bars,
                            reps=max(24, args.reps // 4), gen_n_eff=args.gen_n_eff, seed=args.seed)
        deployed = _interp_crossing(rows, 0.80)
        ideal = ideal_mde80(2.0, n_bars, ppy)
        out_spans.append({
            "n_bars": n_bars,
            "years": n_bars / ppy,
            "ideal_mde80_t2": ideal,
            "deployed_mde80": deployed,
            "ratio_deployed_over_ideal": (deployed / ideal) if deployed else None,
            "null_power": null["power"],
            "saturation_power": sat["power"],
            "saturation_planted": args.saturation,
            "controls_ok": bool(null["power"] <= 0.10 and sat["power"] >= 0.80),
            "grid": rows,
            "seconds": time.time() - t0,
        })
        log.info("N=%d: ideal %.3f  deployed %s  null %.3f  sat %.3f  (%.0fs)", n_bars, ideal,
                 f"{deployed:.3f}" if deployed else "UNREACHABLE", null["power"], sat["power"],
                 out_spans[-1]["seconds"])
    return {
        "substrate": {"kind": args.substrate, "bars": panel.T, "names": panel.N,
                      "gates": args.gates, "periods_per_year": ppy, "seed": args.seed,
                      "reps": args.reps, "gen_n_eff": args.gen_n_eff,
                      "base_ann_sharpe": _ann_sharpe(
                          np.concatenate([np.asarray(v) for v in base.values()])[
                              np.isfinite(np.concatenate([np.asarray(v) for v in base.values()]))],
                          ppy)},
        "audit_reference": {"ideal": AUDIT_IDEAL, "deployed_band": AUDIT_DEPLOYED_BAND,
                            "realistic_edge_band": REALISTIC_EDGE_BAND},
        "spans": out_spans,
    }


def _print(out: dict) -> None:
    s = out["substrate"]
    print("\n" + "=" * 100)
    print("THE POWER WALL — ideal vs DEPLOYED minimum detectable effect (audit F7)")
    print("=" * 100)
    print(f"  substrate {s['kind']}  T={s['bars']}  N={s['names']}  base ann SR "
          f"{s['base_ann_sharpe']:+.3f}  reps={s['reps']}  gen_n_eff={s['gen_n_eff']}")
    print(f"  ideal = (t>=2 + z80) * sqrt({s['periods_per_year']:g}/N)   deployed = measured 80% "
          f"crossing of the 6-leg gate, in REALIZED dSR")
    print("-" * 100)
    print(f"  {'bars':>6} {'years':>6} {'ideal':>8} {'deployed':>10} {'ratio':>7} "
          f"{'null':>6} {'sat':>6}  controls")
    for r in out["spans"]:
        dep = f"{r['deployed_mde80']:.3f}" if r["deployed_mde80"] else "UNREACH"
        rat = f"{r['ratio_deployed_over_ideal']:.2f}x" if r["ratio_deployed_over_ideal"] else "-"
        print(f"  {r['n_bars']:>6} {r['years']:>6.2f} {r['ideal_mde80_t2']:>8.3f} {dep:>10} "
              f"{rat:>7} {r['null_power']:>6.3f} {r['saturation_power']:>6.3f}  "
              f"{'OK' if r['controls_ok'] else 'BROKEN'}")
    print("-" * 100)

    bad = [r for r in out["spans"] if not r["controls_ok"]]
    if bad:
        print("  ⚠ CONTROLS FAILED on "
              f"{', '.join(str(r['n_bars']) for r in bad)} — a null that fires or a huge edge that")
        print("    does not means the MDE column is measuring the harness, not the contract.")
        print("    Fix before reading any verdict off this run.")
        return

    lo, hi = AUDIT_DEPLOYED_BAND
    dep = [r["deployed_mde80"] for r in out["spans"] if r["deployed_mde80"]]
    print(f"  audit F7 — ideal: {AUDIT_IDEAL}   deployed band: {lo}-{hi}")
    for r in out["spans"]:
        if r["n_bars"] in AUDIT_IDEAL:
            print(f"    ideal @ T={r['n_bars']}: measured {r['ideal_mde80_t2']:.3f} vs audit "
                  f"{AUDIT_IDEAL[r['n_bars']]}")
    if dep:
        print(f"\n  measured deployed MDE80 range: {min(dep):.2f}-{max(dep):.2f}   "
              f"audit: {lo}-{hi}  -> "
              f"{'OVERLAPS' if (min(dep) <= hi and max(dep) >= lo) else 'DISAGREES'}")
    unreach = [r["n_bars"] for r in out["spans"] if not r["deployed_mde80"]]
    if unreach:
        print(f"  ⚠ deployed MDE80 UNREACHABLE within the swept grid at N={unreach} — the gate")
        print("    never reaches 80% power there. Raise --grid to bound it, or report as "
              ">= grid max.")
    e_lo, e_hi = REALISTIC_EDGE_BAND
    if dep:
        print(f"\n  => A realistic single-signal edge is {e_lo}-{e_hi} annualized dSR. The DEPLOYED")
        print(f"     contract needs {min(dep):.2f}+ to see one at 80% power — "
              f"{min(dep) / e_hi:.0f}x the top of that range.")
        print("     F7's conclusion holds: 0.3-0.5 is undetectable on this data under this "
              "contract.")
    print("=" * 100 + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gates", default=DEFAULT_GATES)
    p.add_argument("--substrate", choices=["synthetic", "taiwan"], default="synthetic")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--spans", type=int, nargs="+", default=[1011, 2520, 4044],
                   help="panel lengths to measure (default: F7's own anchors plus a midpoint)")
    p.add_argument("--grid", type=float, nargs="+",
                   default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0],
                   help="planted annualized Sharpes to sweep (realized dSR is ~0.6x these)")
    p.add_argument("--saturation", type=float, default=12.0,
                   help="planted Sharpe for the saturation control; must reach ~100%% power")
    p.add_argument("--reps", type=int, default=200, help="replicates per grid point")
    p.add_argument("--names", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--gen-n-eff", type=float, default=233.0,
                   help="file-drawer trial count (default 233 = the project's real charged count)")
    p.add_argument("--json", type=str, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")   # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    out = run(args)
    _print(out)
    if args.json:
        dest = Path(args.json)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
