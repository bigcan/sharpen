"""Crucible INTRADAY power probe (Stage-0 Part A) — does a higher-FREQUENCY substrate detect a
realistic marginal edge AFTER the microstructure autocorrelation haircut, or does effective-N
collapse eat the raw-count gain?

Pre-registration: docs/research/crucible_intraday_stage0_preregistration_2026-07-13.md (Part A).
Frozen grids: configs/crucible_calibration_intraday.gates.yaml.

DESIGN v2 (after run-1 tripped its own TA-3). Run-1 measured MDE = "realized ΔSR at the 80%-power
crossing" on a full (T_raw x H) grid; a too-coarse low-beta grid PINNED the crossing to one grid
point, so MDE didn't resolve vs N_eff and the sqrt((2H-1)/T) extrapolation was untrustworthy. The
tripwire caught it. Fixed design:

  1. Measure the funnel's MDE-vs-N_eff curve DIRECTLY at H=1 (phi=0, IID) with a FINE beta grid.
  2. Autocorrelation enters ONLY via effective sample size: N_eff = holdout/(2H-1) (AR(1) VIF,
     Math-verified). Validation cells at H=12 must land on the H=1 curve at matched N_eff.
  3. Read the full-panel frontier by INTERPOLATION: at ~4.86M bars the tradeable holdings land
     INSIDE the measured N_eff span (H=60 -> ~10.2k, H=300 -> ~2.0k, both in [500,16000]), so the
     A1 gate is an interpolation off the measured curve, not a fragile extrapolation.

It REUSES the shipped calibration harness's REAL gate path (`_planted_panel`, `_overlay_returns`,
`_combined_book`, `combination_fitness`) — no statistic re-implemented, funnel `gates_hash`
(CRU-1 519158fa1450) untouched. The ONLY new ingredient is an AR(1)-persistent base-sleeve PnL
generator (`_planted_base_sleeves_ar`), a drop-in for the harness's IID `_planted_base_sleeves`
that reduces to it BIT-FOR-BIT at H=1 (tripwire TA-1).

Usage (repo root):
    python scripts/research/crucible_intraday_power.py
    python scripts/research/crucible_intraday_power.py --quick   # tiny grid smoke test
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the REAL harness gate path — nothing re-implemented.
from scripts.research.crucible_calibration import (  # noqa: E402
    DEFAULT_FUNNEL_GATES,
    _mde_from_curve,
    _panel_ts,
    _planted_base_sleeves,
    _planted_panel,
    load_calib,
)
from finrl_pro_ds.signals.generation.evolve import _overlay_returns  # noqa: E402
from finrl_pro_ds.signals.generation.fitness import _combined_book, combination_fitness  # noqa: E402

log = logging.getLogger("crucible_intraday_power")

DEFAULT_INTRADAY_GATES = ROOT / "configs" / "crucible_calibration_intraday.gates.yaml"
BARS_PER_YEAR = 245 * 3240  # 5-sec bars in a Taiwan trading year (session 09:00-13:30)


# ------------------------------------------------------------------ AR(1) persistence
def _ar1_unit(z: np.ndarray, hold_bars: int) -> np.ndarray:
    """Unit-variance AR(1) with phi = 1 - 1/H: x[t] = phi*x[t-1] + sqrt(1-phi^2)*z[t]. Steady-state
    Var(x)=1, lag-1 autocorr=phi. H=1 -> phi=0 -> x = z (identity), so the caller reduces to the
    shipped IID harness EXACTLY. Vectorized via scipy.signal.lfilter (C-level)."""
    phi = 1.0 - 1.0 / float(hold_bars)
    if phi <= 0.0:
        return z
    from scipy.signal import lfilter  # type: ignore[import-untyped]
    c = math.sqrt(1.0 - phi * phi)
    return lfilter([c], [1.0, -phi], z).astype(np.float64)


def _planted_base_sleeves_ar(s: np.ndarray, *, beta: float, seed: int, hold_bars: int) -> dict:
    """EXACT copy of crucible_calibration._planted_base_sleeves with the IID PnL noise replaced by a
    unit-variance AR(1) of the same marginal variance. The signal (beta*s_std) and drift are unchanged,
    so the TRUE effect size at a given beta is identical — only the noise autocorrelation (hence the
    Sharpe-estimator variance / detection power) changes. hold_bars=1 -> identical to the original."""
    rng = np.random.default_rng(seed + 7919)
    t = s.size
    s_lag = np.concatenate([[0.0], s[:-1]])
    s_std = (s_lag - s_lag.mean()) / (s_lag.std() + 1e-12)
    tsmom = 0.0004 + beta * s_std + 0.020 * _ar1_unit(rng.standard_normal(t), hold_bars)
    carry = 0.0003 + 0.008 * _ar1_unit(rng.standard_normal(t), hold_bars)
    return {"tsmom": tsmom.astype(np.float64), "rates_carry": carry.astype(np.float64)}


def _assert_ta1_iid_reduction() -> None:
    """TA-1: at H=1 the AR generator must equal the shipped IID generator bit-for-bit (same seed)."""
    _, s = _planted_panel(400, 6, seed=123)
    a = _planted_base_sleeves_ar(s, beta=0.01, seed=123, hold_bars=1)
    b = _planted_base_sleeves(s, beta=0.01, seed=123)
    for k in ("tsmom", "rates_carry"):
        if not np.array_equal(a[k], b[k]):
            raise SystemExit(f"[TA-1] ABORT — AR generator at H=1 does not reduce to the shipped IID "
                             f"generator on '{k}' (max|diff|={np.max(np.abs(a[k]-b[k])):.2e}).")
    log.info("[TA-1] IID reduction PASS: AR(H=1) == shipped IID generator bit-for-bit")


# ------------------------------------------------------------------ power curve (real gate)
def _power_curve(cc, *, t: int, n: int, hold_bars: int, betas: list[float], n_seeds: int,
                 gen_n_eff: float, cost_bps: float) -> list[dict[str, Any]]:
    """The E2 power curve at (T=t, holding=hold_bars): for each beta, score the planted overlay through
    the REAL combination_fitness gate over n_seeds AR-persistent panels. Mirrors the harness's
    _e2_power_curve exactly, but with the AR base sleeves."""
    curve: list[dict[str, Any]] = []
    for beta in betas:
        detections = 0
        realized: list[float] = []
        for k in range(n_seeds):
            panel, s = _planted_panel(t, n, seed=5000 + k)
            base = _planted_base_sleeves_ar(s, beta=beta, seed=5000 + k, hold_bars=hold_bars)
            ts = _panel_ts(panel)
            base_book = _combined_book(base, ts, cc.fit_cfg)
            out = _overlay_returns("macro:plant", panel, base_book, cost_bps=cost_bps)
            if out is None:
                continue
            cand, turnover = out
            res = combination_fitness(cand, base, ts, cc.fit_cfg, gen_n_eff=gen_n_eff,
                                      turnover_ann=turnover, n_nodes=1)
            if res.passes_gate:
                detections += 1
            if np.isfinite(res.delta_sr_oos):
                realized.append(float(res.delta_sr_oos))
        curve.append({
            "beta": beta, "power": detections / max(1, n_seeds), "detections": detections,
            "n_seeds": n_seeds,
            "mean_realized_delta_sr": float(np.mean(realized)) if realized else float("nan"),
        })
    return curve


def _eval_cell(cc, *, t: int, hb: int, n: int, betas: list[float], n_seeds: int, gen_n_eff: float,
               cost_bps: float, holdout_frac: float, power_target: float) -> dict[str, Any]:
    """Score one (T_raw, H) cell -> its MDE (realized marginal ΔSR at power_target) and N_eff."""
    curve = _power_curve(cc, t=t, n=n, hold_bars=hb, betas=betas, n_seeds=n_seeds,
                         gen_n_eff=gen_n_eff, cost_bps=cost_bps)
    mde = _mde_from_curve(curve, power_target)
    holdout = int(round(t * holdout_frac))
    return {
        "t_raw": t, "hold_bars": hb, "holdout_raw": holdout, "vif": 2 * hb - 1,
        "n_eff": holdout / (2 * hb - 1),
        "mde_realized_delta_sr": (float(mde["mean_realized_delta_sr"]) if mde else None),
        "mde_beta": (mde["beta"] if mde else None),
        "max_power": max((c["power"] for c in curve), default=0.0),
        "curve": curve,
    }


# ------------------------------------------------------------------ MDE(N_eff) interpolation
def _interp_mde(points: list[tuple[float, float]], target_neff: float) -> tuple[float, str]:
    """Interpolate MDE at target N_eff in LOG-LOG space off the measured H=1 curve (MDE ~ power law
    of N_eff). Returns (mde, mode) where mode in {interp, extrap_low, extrap_high}. Requires >=2
    detected points, ascending N_eff."""
    pts = sorted(points)
    xs = [math.log(ne) for ne, _ in pts]
    ys = [math.log(m) for _, m in pts]
    x = math.log(target_neff)
    if x <= xs[0]:
        sl = (ys[1] - ys[0]) / (xs[1] - xs[0])
        return math.exp(ys[0] + sl * (x - xs[0])), "extrap_low"
    if x >= xs[-1]:
        sl = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return math.exp(ys[-1] + sl * (x - xs[-1])), "extrap_high"
    for i in range(1, len(xs)):
        if x <= xs[i]:
            sl = (ys[i] - ys[i - 1]) / (xs[i] - xs[i - 1])
            return math.exp(ys[i - 1] + sl * (x - xs[i - 1])), "interp"
    return math.exp(ys[-1]), "interp"  # unreachable


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="Crucible intraday power probe (Stage-0 Part A): the "
                                             "funnel MDE-vs-N_eff curve + the intraday frontier.")
    ap.add_argument("--intraday-config", default=str(DEFAULT_INTRADAY_GATES))
    ap.add_argument("--config", default=str(DEFAULT_FUNNEL_GATES),
                    help="funnel gates YAML supplying the REAL FitnessConfig thresholds.")
    ap.add_argument("--out", default=str(ROOT / "results" / "crucible_intraday_power"))
    ap.add_argument("--quick", action="store_true", help="tiny grid smoke test, NOT a verdict.")
    args = ap.parse_args()

    _assert_ta1_iid_reduction()  # tripwire before any sweep (fatal SystemExit on failure)
    ta1_ok = True

    raw = yaml.safe_load(Path(args.intraday_config).read_text(encoding="utf-8"))
    ip = raw["intraday_power"]
    cc = load_calib(Path(args.intraday_config), Path(args.config))
    n = int(raw["substrate"]["n"])
    cost_bps = float(cc.ek["cost_bps"])
    holdout_frac = float(cc.ek["holdout_frac"])
    gen_n_eff = float(ip["gate_probe_gen_n_eff"])
    power_target = float(ip["power_target"])
    ceiling = float(ip["mde_ceiling"])
    full_panel = float(ip["full_panel_bars"])
    hold_cap = int(ip["tradeable_hold_bars_max"])
    frontier_h = [int(h) for h in ip["frontier_h_grid"]]

    if args.quick:
        primary_t = [2000, 8000, 32000]
        validation = [[46000, 12]]
        betas, n_seeds = [0.0, 0.0008, 0.002, 0.005, 0.013], 6
    else:
        primary_t = [int(x) for x in ip["primary_h1_t_grid"]]
        validation = [[int(a), int(b)] for a, b in ip["validation_cells"]]
        betas = [float(b) for b in ip["beta_grid"]]
        n_seeds = int(ip["n_seeds"])

    common: dict[str, Any] = dict(n=n, betas=betas, n_seeds=n_seeds, gen_n_eff=gen_n_eff,
                                  cost_bps=cost_bps, holdout_frac=holdout_frac,
                                  power_target=power_target)

    log.info("PRIMARY H=1 MDE-vs-N_eff: T in %s (%d betas x %d seeds)", primary_t, len(betas), n_seeds)
    primary: list[dict[str, Any]] = []
    for t in primary_t:
        cell = _eval_cell(cc, t=t, hb=1, **common)
        primary.append(cell)
        log.info("  H=1 T=%d (N_eff=%.0f): MDE=%s @beta=%s", t, cell["n_eff"],
                 cell["mde_realized_delta_sr"], cell["mde_beta"])

    log.info("VALIDATION H>1 (autocorr-acts-via-N_eff): cells %s", validation)
    valid: list[dict[str, Any]] = []
    for t, hb in validation:
        cell = _eval_cell(cc, t=t, hb=hb, **common)
        valid.append(cell)
        log.info("  H=%d T=%d (N_eff=%.0f): MDE=%s @beta=%s", hb, t, cell["n_eff"],
                 cell["mde_realized_delta_sr"], cell["mde_beta"])

    # ---- measured H=1 curve (N_eff, MDE) for detected cells
    pts = [(c["n_eff"], c["mde_realized_delta_sr"]) for c in primary
           if c["mde_realized_delta_sr"] is not None]
    enough = len(pts) >= 2

    # TA-2: MDE decreasing as N_eff increases (more data -> smaller detectable edge), slack for MC.
    ordered = [m for _, m in sorted(pts)]
    ta2_ok = enough and all(ordered[i] >= ordered[i + 1] - 0.20 for i in range(len(ordered) - 1))

    # TA-3': each validation (H>1) cell lands on the H=1 curve at its N_eff (autocorr acts via N_eff)
    ta3_details: list[dict[str, Any]] = []
    ta3_ok = enough
    for c in valid:
        if c["mde_realized_delta_sr"] is None:
            ta3_details.append({"cell": [c["t_raw"], c["hold_bars"]], "status": "undetected"})
            continue
        pred, mode = _interp_mde(pts, c["n_eff"])
        ratio = c["mde_realized_delta_sr"] / pred if pred > 0 else float("inf")
        ok = 0.6 <= ratio <= 1.7   # within ~1.6x either way (MC + interp slack)
        ta3_ok = ta3_ok and ok
        ta3_details.append({"cell": [c["t_raw"], c["hold_bars"]], "n_eff": c["n_eff"],
                            "measured_mde": c["mde_realized_delta_sr"], "h1_curve_mde": pred,
                            "ratio": ratio, "on_curve": ok, "interp_mode": mode})

    # ---- frontier: full-panel implied MDE at each holding H, by interpolation off the H=1 curve
    holdout_full = full_panel * holdout_frac
    frontier: list[dict[str, Any]] = []
    for hb in frontier_h:
        neff_full = holdout_full / (2 * hb - 1)
        if enough:
            mde_full, mode = _interp_mde(pts, neff_full)
        else:
            mde_full, mode = float("nan"), "n/a"
        frontier.append({"hold_bars": hb, "hold_minutes": hb * 5 / 60.0,
                         "n_eff_full_panel": neff_full, "full_panel_implied_mde": mde_full,
                         "interp_mode": mode, "clears_ceiling": bool(mde_full <= ceiling)})

    tradeable_ok = any(f["clears_ceiling"] for f in frontier if f["hold_bars"] <= hold_cap)
    a1_pass = bool(ta1_ok and ta2_ok and ta3_ok and enough and tradeable_ok)

    report = {
        "harness_version": raw.get("harness_version"), "quick": args.quick,
        "ts": datetime.now(timezone.utc).isoformat(), "design": "v2-measured-curve",
        "funnel_gates": str(args.config), "gate_probe_gen_n_eff": gen_n_eff,
        "power_target": power_target, "holdout_frac": holdout_frac, "cost_bps": cost_bps,
        "mde_ceiling": ceiling, "full_panel_bars": full_panel, "tradeable_hold_bars_max": hold_cap,
        "primary_h1": primary, "validation": valid, "measured_curve_points": pts,
        "frontier": frontier,
        "tripwires": {"TA1_iid_reduction": ta1_ok, "TA2_monotone_neff": ta2_ok,
                      "TA3_autocorr_via_neff": ta3_ok, "TA3_detail": ta3_details},
        "A1_verdict": "PASS" if a1_pass else ("INCONCLUSIVE" if not (ta2_ok and ta3_ok) else "FAIL"),
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "intraday_power_quick" if args.quick else "intraday_power"
    out_path = out_dir / f"{stem}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n============== CRUCIBLE INTRADAY POWER v2 (Stage-0 Part A) ==============")
    print("H=1 measured MDE-vs-N_eff (funnel detection floor at 80% power):")
    print(f"   {'N_eff':>8} {'T_raw':>9} {'MDE_dSR':>9} {'@beta':>8}")
    for c in primary:
        m = c["mde_realized_delta_sr"]
        print(f"   {c['n_eff']:>8.0f} {c['t_raw']:>9} {('%.2f' % m) if m is not None else 'undet':>9} "
              f"{c['mde_beta'] if c['mde_beta'] is not None else '-':>8}")
    print("validation (H>1 should land on the H=1 curve at matched N_eff):")
    for d in ta3_details:
        if d.get("status") == "undetected":
            print(f"   cell {d['cell']}: undetected")
        else:
            print(f"   cell {d['cell']} N_eff={d['n_eff']:.0f}: measured {d['measured_mde']:.2f} vs "
                  f"H1-curve {d['h1_curve_mde']:.2f} (ratio {d['ratio']:.2f}, on_curve={d['on_curve']})")
    print(f"\nFull-panel ({full_panel/1e6:.1f}M bars) implied MDE by holding H:")
    print(f"   {'H(bars)':>8} {'hold':>7} {'N_eff':>8} {'MDE':>7} {'clears<=%.2f':>12} {'mode':>12}"
          % ceiling)
    for f in frontier:
        print(f"   {f['hold_bars']:>8} {f['hold_minutes']:>5.1f}m {f['n_eff_full_panel']:>8.0f} "
              f"{f['full_panel_implied_mde']:>7.2f} {str(f['clears_ceiling']):>12} {f['interp_mode']:>12}")
    print(f"\ntripwires: TA-1={ta1_ok} TA-2(monotone)={ta2_ok} TA-3(autocorr-via-N_eff)={ta3_ok}")
    print(f"A1 VERDICT: {report['A1_verdict']}   (PASS = tradeable H<={hold_cap} bars clears "
          f"MDE<={ceiling} on the ~{full_panel/1e6:.1f}M-bar 2019+ panel)")
    print(f"report -> {out_path}")
    print("========================================================================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
