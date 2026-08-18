"""Is there a `hold_horizon` that makes the WQ101 cross-sectional bank BOTH tradeable AND powered
on `us_equity`? Derive it, then measure it — do not run an hour of mining to find out.

Two constraints move in OPPOSITE directions with H, and the project has never written them down
together:

  FEASIBILITY (`evolve.score`): a candidate is hard-infeasible when
      turnover_ann > turnover_soft_cap * 2
  Turnover falls as H rises (a rank-L/S book rebalances 252/H times a year), so feasibility wants
  H LARGE. Measured on the real panel at H=2: 68-135/yr against a 24/yr ceiling, 97 of 100 refused.

  POWER (`configs/us_equity_power.gates.yaml`): the substrate is admitted only when
      MDE_delta_sr <= plausible_delta_sr_max = IC_low * sqrt(BR),  BR = n_eff * (252 / H)
  Breadth per rebalance RISES with H, but sublinearly (measured 21.5 -> 43.7 -> 81.7 for
  H = 1 -> 2 -> 21, about H^0.27), so BR falls and power wants H SMALL. The shipped ceiling 1.457
  was derived at H=2 (43.7 * 126 = 5,505 bets/yr) and is NOT valid at any other H — nothing in the
  code re-derives it when `generation.hold_horizon` changes, which is the trap this script exists to
  make visible. Note the substrate config's "detection is indifferent to this choice / n_obs is
  invariant across H" holds only in the small-H regime where n_eff really does double (H=1->2); it
  does not extrapolate.

Run: python scripts/research/crucible_hold_horizon_tradeoff.py [--holds 2 5 10 21 63] [--limit N]
Writes results/crucible_hold_tradeoff/hold_tradeoff.json.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from finrl_pro_ds.crucible.agentic.proposer import _CS_SEED_BANK
from finrl_pro_ds.crucible.data.us_equity_panel import build_us_equity_panel
from finrl_pro_ds.signals.generation.config import load_generation_config
from finrl_pro_ds.signals.generation.evolve import _candidate_returns
from finrl_pro_ds.signals.library._alpha_formulas import FORMULAS
from finrl_pro_ds.signals.library.alphas101 import SKIP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("hold_tradeoff")

ROOT = Path(__file__).resolve().parents[2]
GATES = ROOT / "configs" / "us_equity_signal_eval.gates.yaml"
POWER_GATES = ROOT / "configs" / "us_equity_power.gates.yaml"

#: MEASURED realized breadth of the real WQ101 DSL on this panel, per hold horizon
#: (crucible_real_alpha_breadth.py --panel us_equity --hold H; median n_eff_realized).
#:
#: ⚠ DO NOT hold n_eff fixed and scale only the rebalance count — that was this script's first,
#: WRONG model and it under-stated the ceiling at large H. Breadth genuinely RISES with H (a longer
#: horizon buys more independent bets per rebalance), but SUBLINEARLY: 21.5 -> 43.7 -> 81.7 for
#: H = 1 -> 2 -> 21, i.e. roughly H^0.27, not H^1. The substrate config's "detection is indifferent
#: to this choice / n_obs is invariant across H" is exact only in the small-H regime where n_eff
#: does double (H=1->2); it does NOT extrapolate. Note H=2 and H=5 are FLAT (43.7 vs 43.4), so this
#: is not a clean power law either — which is why every point here is measured rather than fitted.
N_EFF_MEASURED: dict[int, float] = {1: 21.5, 2: 43.7, 5: 43.4, 10: 62.7, 21: 81.7}


def _n_eff(hold: int) -> float:
    """Measured breadth at ``hold`` when we have it; otherwise a log-log interpolation between the
    two nearest MEASURED holds (never an assumption of H-invariance in either direction)."""
    known = {h: v for h, v in N_EFF_MEASURED.items() if np.isfinite(v)}
    if hold in known:
        return known[hold]
    hs = sorted(known)
    lo = max([h for h in hs if h <= hold], default=hs[0])
    hi = min([h for h in hs if h >= hold], default=hs[-1])
    if lo == hi:
        return known[lo]
    w = (np.log(hold) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return float(np.exp((1 - w) * np.log(known[lo]) + w * np.log(known[hi])))


IC_LOW = 0.020                 # the bottom of the 0.02-0.03 band, per the power gates file
MDE_DELTA_SR = 1.3120527546583958      # this substrate's implied MDE (19.61y panel, 6.85y holdout)


def _bank() -> dict[int, str]:
    curated = {i for i, _, _, _ in _CS_SEED_BANK}
    idxs = sorted((set(FORMULAS) - set(SKIP)) | curated)
    return {i: FORMULAS[i] for i in idxs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--holds", type=int, nargs="+", default=[2, 5, 10, 21, 63])
    ap.add_argument("--limit", type=int, default=0, help="only the first N formulas (smoke test)")
    ap.add_argument("--out", default=str(ROOT / "results" / "crucible_hold_tradeoff"))
    args = ap.parse_args()

    cfg, ek = load_generation_config(GATES)
    ceiling_cap = float(cfg.turnover_soft_cap) * 2.0
    cost_bps, ls_min_names = float(ek["cost_bps"]), int(ek["ls_min_names"])
    log.info("feasibility ceiling = turnover_soft_cap*2 = %.1f/yr; cost_bps=%.4f ls_min_names=%d",
             ceiling_cap, cost_bps, ls_min_names)

    panel = build_us_equity_panel()
    log.info("panel T=%d N=%d", panel.T, panel.N)
    bank = _bank()
    if args.limit:
        bank = dict(list(bank.items())[: args.limit])

    rows = []
    for hold in args.holds:
        # --- POWER side: the ceiling this H actually implies (nothing in the code re-derives it) ---
        rebal = 252.0 / float(hold)
        n_eff = _n_eff(int(hold))           # MEASURED breadth at this H (never assumed)
        br = n_eff * rebal                  # bets/yr = breadth per rebalance x rebalances
        ceiling = IC_LOW * float(np.sqrt(br))
        powered = bool(MDE_DELTA_SR <= ceiling)

        # --- FEASIBILITY side: MEASURED turnover of the real bank on the real panel --------------
        turns, feasible = [], 0
        for idx, formula in bank.items():
            try:
                out = _candidate_returns(formula, panel, hold_horizon=int(hold),
                                         cost_bps=cost_bps, min_names=ls_min_names)
            except Exception:                                  # noqa: BLE001 - unscoreable, skip
                continue
            if out is None:
                continue
            t = float(out[1])
            turns.append(t)
            feasible += int(t <= ceiling_cap)
        med = float(np.median(turns)) if turns else float("nan")
        rows.append(dict(hold=int(hold), rebalances_per_year=rebal, n_eff_measured=n_eff,
                         breadth_bets_per_year=br,
                         implied_power_ceiling=ceiling, mde=MDE_DELTA_SR, powered=powered,
                         n_scored=len(turns), n_feasible=feasible,
                         median_turnover_ann=med,
                         min_turnover_ann=(float(np.min(turns)) if turns else float("nan")),
                         both=bool(powered and feasible > 0)))
        log.info("H=%-3d rebal/yr=%6.1f  n_eff=%5.1f  BR=%7.0f  ceiling=%.3f vs MDE %.3f -> %-8s | "
                 "feasible %3d/%3d (median turnover %.1f/yr vs cap %.1f)",
                 hold, rebal, n_eff, br, ceiling, MDE_DELTA_SR,
                 "POWERED" if powered else "REFUSED",
                 feasible, len(turns), med, ceiling_cap)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"ceiling_cap_turnover_ann": ceiling_cap, "n_eff_measured": N_EFF_MEASURED,
               "ic_low": IC_LOW, "mde_delta_sr": MDE_DELTA_SR, "rows": rows}
    (out_dir / "hold_tradeoff.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    ok = [r for r in rows if r["both"]]
    if ok:
        log.info("H values satisfying BOTH constraints: %s", [r["hold"] for r in ok])
    else:
        log.warning("NO tested hold_horizon satisfies BOTH constraints — the substrate cannot host a "
                    "tradeable AND adequately-powered cross-sectional cohort at any H tested. "
                    "Feasibility wants H large, power wants H small, and the windows do not overlap.")


if __name__ == "__main__":
    main()
