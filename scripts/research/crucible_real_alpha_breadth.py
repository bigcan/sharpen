"""What breadth do REAL alphas actually realize on a 300-name equity panel? (S553-cont-152)

The two prior probes this session bracket the answer but neither is decisive:

  * `crucible_equity_breadth_measure.py` — participation ratio of the RETURN correlation matrix:
    9.41 raw / 30.09 demeaned at K=300. This is the statistic the entire Crucible power analysis
    uses (it is where 5.01 for the FX panel came from).
  * `crucible_breadth_convention_probe.py` — planted a signal and inverted IR = IC*sqrt(BR); got an
    implied breadth of ~224, i.e. ~0.75 x N. That probe is NOT a refutation of the 9.41/30.09 pair
    so much as a demonstration that the planted signal was the wrong SHAPE: it was built from the
    ranks of each name's realized forward return plus independent noise, so it carried genuine and
    INDEPENDENT forecasting power on every name's idiosyncratic move. That manufactures N
    independent bets, and the experiment duly returned BR ~ N.

The lesson generalizes and it is the point of this script: **breadth is a joint property of the
substrate AND the signal, not of the return correlation matrix alone.** A return-covariance
participation ratio only predicts BR if the signal's cross-sectional bets inherit the returns' factor
structure. A signal that bets idiosyncratically has far more breadth; a signal that is really one
factor in disguise has far less. So "n_eff = 5.01, therefore the substrate cannot be mined" was
measuring the substrate while the binding quantity is the substrate-plus-DSL pair.

This script measures the pair that is actually on the table: the shipped WorldQuant-101 library
(the Crucible's own DSL) on the 300-name PIT US equity panel. For each alpha it computes the realized
cross-sectional IC and the realized annualized IR of the shipped dollar-neutral rank L/S book, then
inverts the law for the breadth that alpha actually achieved:

    n_eff_realized = (IR / IC)^2 / rebalances_per_year

The median over the library is the number Option A's power calculation should be using. No planting,
no assumption about factor structure — the real DSL on the real panel.

IN-SAMPLE BY CONSTRUCTION, AND THAT IS FINE: this measures BREADTH (a second-moment property of how
a signal spreads its bets), not alpha. No verdict is read off the ICs or IRs here, and none should
be. See CRU-2 -- this script reads no ledger and produces no candidate.

Usage:
    python scripts/research/crucible_real_alpha_breadth.py
    python scripts/research/crucible_real_alpha_breadth.py --k 300 --hold 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from finrl_pro_ds.crucible.data.us_equity_panel import build_us_equity_panel  # noqa: E402
from finrl_pro_ds.signals.eval_harness import _ls_weights, compute_scores  # noqa: E402
from finrl_pro_ds.signals.library.alphas101 import SIGNALS as ALPHAS  # noqa: E402

MEMBERS = Path(r"C:\tmp\sp500_pit_members.csv")
OUT = ROOT / "results" / "signal_eval" / "crucible_equity_breadth"

NEU = ("winsor", "zscore")     # sector unknown for former members; size omitted to stay generic
MIN_ABS_IC = 0.004             # below this the (IR/IC)^2 inversion is numerically meaningless
N_EFF_RAW = 9.41
N_EFF_DEMEANED = 30.09


def evaluate(scores: np.ndarray, rets: np.ndarray, univ: np.ndarray, hold: int) -> dict | None:
    """Realized IC and L/S book IR for one alpha. Signal at t -> return over (t, t+hold]."""
    T = scores.shape[0]
    pnl, ics, icr = [], [], []
    for t in range(0, T - hold, hold):
        act = univ[t] & univ[t + hold]
        if act.sum() < 30:
            continue
        s = scores[t]
        fwd = np.nansum(rets[t + 1:t + hold + 1], axis=0)
        m = act & np.isfinite(s) & np.isfinite(fwd)
        if m.sum() < 30:
            continue
        sv, yv = s[m], fwd[m]
        if sv.std() == 0 or yv.std() == 0:
            continue
        yd = yv - yv.mean()                       # the dollar-neutral book's actual exposure
        ics.append(float(np.corrcoef(sv, yd)[0, 1]))
        icr.append(float(spearmanr(sv, yd).statistic))
        row = np.full(scores.shape[1], np.nan)
        row[m] = sv
        w = _ls_weights(row, m)
        if np.any(w):
            pnl.append(float(w @ np.nan_to_num(fwd)))
    if len(pnl) < 200:
        return None
    pnl = np.asarray(pnl)
    if pnl.std() == 0:
        return None
    rpy = 252.0 / hold
    return {"ic": float(np.mean(ics)), "ic_rank": float(np.mean(icr)),
            "ir": float(pnl.mean() / pnl.std() * np.sqrt(rpy)), "n_obs": len(pnl)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=300)
    ap.add_argument("--hold", type=int, default=1)
    args = ap.parse_args()

    # Measure on the SHIPPED builder, not a local reconstruction: breadth is the number the whole
    # substrate is justified by, so it must come from the exact panel the miner will see (shifted
    # ADV, PIT membership, top-K mask) rather than a lookalike built here.
    panel = build_us_equity_panel(top_k=args.k)
    univ = panel.active.astype(bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        rets = np.diff(np.log(panel.close), axis=0, prepend=np.nan)
    rpy = 252.0 / args.hold
    print(f"panel {panel.T}x{panel.N} | universe {univ.sum(axis=1).mean():.0f} names/day | "
          f"H={args.hold} ({rpy:.0f} rebal/yr) | {len(ALPHAS)} alphas\n")

    rows = []
    for i, sig in enumerate(ALPHAS):
        name = sig.spec.name
        try:
            sc = compute_scores(sig, panel, NEU)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {name}: {exc!r}"[:90])
            continue
        r = evaluate(np.asarray(sc, dtype=np.float64), rets, univ, args.hold)
        if r is None:
            continue
        if abs(r["ic"]) < MIN_ABS_IC:
            continue
        if np.sign(r["ic"]) != np.sign(r["ir"]):
            r["sign_mismatch"] = True
        r["n_eff_realized"] = float((r["ir"] / r["ic"]) ** 2 / rpy)
        r["name"] = name
        rows.append(r)
        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(ALPHAS)} ({len(rows)} usable)")

    if not rows:
        print("[FATAL] no alpha cleared the |IC| floor")
        return 1

    rows.sort(key=lambda r: -abs(r["ic"]))
    print(f"\n{'alpha':<12}{'IC':>9}{'IC rank':>10}{'IR':>9}{'n_eff realized':>16}")
    for r in rows[:15]:
        print(f"{r['name']:<12}{r['ic']:>9.4f}{r['ic_rank']:>10.4f}{r['ir']:>9.3f}"
              f"{r['n_eff_realized']:>16.1f}")

    n_eff = np.array([r["n_eff_realized"] for r in rows])
    med = float(np.median(n_eff))
    q1, q3 = (float(np.percentile(n_eff, q)) for q in (25, 75))
    mism = sum(1 for r in rows if r.get("sign_mismatch"))
    print(f"\nusable alphas: {len(rows)}/{len(ALPHAS)}  (sign mismatches: {mism})")
    print(f"n_eff REALIZED by the real DSL: median {med:.1f}  IQR [{q1:.1f}, {q3:.1f}]")
    print(f"  vs RAW return-corr PR      {N_EFF_RAW:>7.2f}  -> {med / N_EFF_RAW:.1f}x")
    print(f"  vs DEMEANED return-corr PR {N_EFF_DEMEANED:>7.2f}  -> {med / N_EFF_DEMEANED:.1f}x")

    print(f"\nIC needed at the REALIZED breadth (3.17/sqrt(n_eff x {rpy:.0f} x years)):")
    for yrs in (5.0, 8.0, 12.0):
        print(f"   holdout {yrs:>4.0f}y : {3.17 / np.sqrt(med * rpy * yrs):.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"real_alpha_breadth_h{args.hold}.json").write_text(json.dumps(
        {"k": args.k, "hold": args.hold, "neutralization": list(NEU),
         "n_eff_realized_median": med, "iqr": [q1, q3], "n_usable": len(rows),
         "sign_mismatches": mism, "rows": rows}, indent=2))
    print(f"\nwrote {OUT / f'real_alpha_breadth_h{args.hold}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
