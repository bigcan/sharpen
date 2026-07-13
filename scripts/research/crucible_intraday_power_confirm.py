"""Stage-0 Part A — DECISIVE frontier confirmation (30 seeds, fine betas).

The main probe (crucible_intraday_power.py) sweeps a wide (T x H) grid cheaply (8 seeds) to shape the
MDE-vs-N_eff curve; at 8 seeds "power >= 0.80" actually requires 7/8 = 0.875, which biases MDE up and
left the wide sweep INCONCLUSIVE (TA-3 tripped on MC noise). This script measures ONLY the two
decision-relevant operating points at 30 seeds (true 0.80 resolution) plus a live anchor:

  * ANCHOR  N_eff=1011 (T=4044, H=1)  -> must reproduce the daily calibration's MDE=1.40 (self-check).
  * 25-min  N_eff=2028 (full-panel H=300 operating point).
  * 5-min   N_eff=10210 (full-panel H=60 operating point) -- the tradeable-boundary gate.

Because autocorrelation acts PURELY through N_eff = holdout/(2H-1) (validated by the main probe's TA-3
at N_eff=500, ratio 1.06), an H=1 cell at a given N_eff is a valid, cheap proxy for the autocorrelated
holding at the same N_eff -- so all cells are run at H=1 with T chosen to hit the target N_eff.

A1-PASS iff the 5-min operating point clears MDE <= 0.50. Reuses the REAL gate (combination_fitness);
funnel gates_hash 519158fa1450 untouched.

Usage (repo root):  python scripts/research/crucible_intraday_power_confirm.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research.crucible_calibration import DEFAULT_FUNNEL_GATES, load_calib  # noqa: E402
from scripts.research.crucible_intraday_power import _eval_cell  # noqa: E402

CEILING = 0.50
ANCHOR_MDE_REF = 1.40           # daily calibration MDE at holdout ~1011 (project memory)
BETAS = [0.0, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.006, 0.008, 0.012]
# (label, T_raw so holdout=0.25*T hits the target N_eff at H=1, full-panel H it stands in for)
CELLS = [("anchor  (H=1)", 4044, 1), ("25-min  (H=300)", 8112, 300), ("5-min   (H=60)", 40840, 60)]


def main() -> int:
    cc = load_calib(ROOT / "configs" / "crucible_calibration_intraday.gates.yaml", DEFAULT_FUNNEL_GATES)
    out: dict = {"ceiling": CEILING, "n_seeds": 30, "beta_grid": BETAS, "anchor_ref": ANCHOR_MDE_REF,
                 "cells": []}
    print(f"{'operating point':<18} {'N_eff':>7} {'MDE':>6} {'clears<=0.50':>13}")
    for label, t, h in CELLS:
        c = _eval_cell(cc, t=t, hb=1, n=12, betas=BETAS, n_seeds=30, gen_n_eff=50.0,
                       cost_bps=0.0010, holdout_frac=0.25, power_target=0.80)
        mde = c["mde_realized_delta_sr"]
        clears = mde is not None and mde <= CEILING
        out["cells"].append({"label": label.strip(), "full_panel_H": h, "n_eff": c["n_eff"],
                             "mde": mde, "mde_beta": c["mde_beta"], "clears": clears})
        print(f"{label:<18} {c['n_eff']:>7.0f} {(('%.2f' % mde) if mde is not None else 'undet'):>6} "
              f"{str(clears):>13}")

    anchor = next(x for x in out["cells"] if x["full_panel_H"] == 1)["mde"]
    # Self-validation tripwire: the harness must reproduce the known daily anchor within MC slack.
    if anchor is None or abs(anchor - ANCHOR_MDE_REF) > 0.25:
        raise SystemExit(f"[ANCHOR] ABORT — harness did not reproduce the daily MDE {ANCHOR_MDE_REF} "
                         f"(got {anchor}); refusing to read a frontier verdict from an unvalidated run.")
    print(f"\n[anchor] reproduced daily MDE {ANCHOR_MDE_REF} (got {anchor:.2f}) -> harness validated")

    five_min = next(x for x in out["cells"] if x["full_panel_H"] == 60)
    out["A1_verdict"] = "PASS" if five_min["clears"] else "FAIL"
    print(f"A1 VERDICT (measured, 30 seeds): {out['A1_verdict']}  "
          f"(PASS = tradeable 5-min holding clears MDE<={CEILING} on the full 2019+ panel)")
    out_dir = ROOT / "results" / "crucible_intraday_power"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frontier_confirm.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"report -> {out_dir / 'frontier_confirm.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
