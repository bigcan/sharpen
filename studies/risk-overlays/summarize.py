"""Summarise the risk-overlay replay of the GMGP1-BTC study by rule family.

    python studies/gmgp1/rerun.py                 # writes studies/gmgp1/out/overlay/grid_results.csv
    python studies/risk-overlays/summarize.py     # prints the tables in this study's README

Reads only the rerun output (20 runs x 54 arms: the no-overlay baseline plus 53 overlay arms).
"Median" is the median over the 20 runs (5 seeds x 4 test months) for one arm.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
GRID = HERE.parents[0] / "gmgp1" / "out" / "overlay" / "grid_results.csv"
if len(sys.argv) > 1:
    GRID = Path(sys.argv[1])
if not GRID.exists():
    sys.exit(f"Missing {GRID}. Run: python studies/gmgp1/rerun.py")

FAMILIES = [  # (prefix, plain name)
    ("SL_price", "Stop-loss on price"),
    ("SL_equity", "Stop-loss on equity"),
    ("SL_", "Stop-loss with cooldown"),
    ("TP_price", "Take-profit on price"),
    ("TP_equity", "Take-profit on equity"),
    ("TIMESTOP", "Time stop"),
    ("DAILYLOSS", "Daily loss limit"),
    ("VOLTARGET", "Volatility target"),
    ("DDTHROTTLE", "Drawdown brake"),
    ("CONTROL_delever", "Control: just trade smaller"),
]


def family(arm: str) -> str:
    for prefix, name in FAMILIES:
        if arm.startswith(prefix):
            return name
    return "Baseline (no rule)" if arm == "baseline" else arm


g = pd.read_csv(GRID)
assert g["arm"].nunique() == 54 and len(g) == 1080, "expected 54 arms x 20 runs"
per_arm = g.groupby("arm").agg(
    pf=("pf", "median"), ret=("ret", "median"), mdd=("mdd", "median"),
    expo=("expo", "median"), trades=("trades", "median"), cells_pf1=("pf", lambda x: int((x >= 1).sum())),
)
per_arm["family"] = [family(a) for a in per_arm.index]
base = per_arm.loc["baseline"]
ov = per_arm.drop(index="baseline")
cells = g[g["arm"] != "baseline"]

print(f"Overlay arms: {len(ov)}; arms with median PF >= 1.0: {(ov.pf >= 1).sum()}")
print(f"Baseline median: PF {base.pf:.4f}, return {base.ret:+.1%}, max drawdown {base.mdd:.1%}, "
      f"exposure {base.expo:.2f}, trades {base.trades:.0f}")
print(f"Run-and-arm cells with PF >= 1.0: {(cells.pf >= 1).sum()} of {len(cells)}; "
      f"from test month (fold) 2: {((cells.pf >= 1) & (cells.fold == 2)).sum()}\n")

order = [n for _, n in FAMILIES]
fam = ov.groupby("family").agg(
    arms=("pf", "size"), best_pf=("pf", "max"), worst_pf=("pf", "min"),
    beat_baseline=("pf", lambda x: int((x > base.pf).sum())), cells_pf1=("cells_pf1", "sum"),
).reindex(order)
print("| Rule family | Arms | Best median PF | Worst median PF | Arms above baseline | Runs with PF >= 1 |")
print("|---|---|---|---|---|---|")
for name, r in fam.iterrows():
    print(f"| {name} | {int(r.arms)} | {r.best_pf:.4f} | {r.worst_pf:.4f} | {int(r.beat_baseline)} of {int(r.arms)} | {int(r.cells_pf1)} |")

print("\nTop 5 arms by median PF:")
print("| Arm | Median PF | Median return | Median max drawdown | Median exposure | Median trades |")
print("|---|---|---|---|---|---|")
for arm, r in ov.sort_values("pf", ascending=False).head(5).iterrows():
    print(f"| {arm} | {r.pf:.4f} | {r.ret:+.1%} | {r.mdd:.1%} | {r.expo:.2f} | {r.trades:.0f} |")
print(f"| baseline (no rule) | {base.pf:.4f} | {base.ret:+.1%} | {base.mdd:.1%} | {base.expo:.2f} | {base.trades:.0f} |")
