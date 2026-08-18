"""Collect gmgp1-spx500 L1-multiseed (and WF / OOS) results from WandB into a comparison.

READS `backtest_val/*` and `backtest_test/*`, which is where run_full_pipeline actually
publishes backtest metrics — NOT the `Profit_Factor_Daily` / `Exposure_Frac` keys emitted by
analytics/wandb_evaluator.py. Those are a different reporting path that this pipeline does
not invoke, so exposure is unavailable here and `trade_count / steps` is reported instead as
an activity proxy. Stated explicitly because an absent metric must not be mistaken for a
measured zero.

Every strategy number is shown against BUY-AND-HOLD on the identical window. A profit factor
above 1.0 means the book made money; it says nothing about whether holding the index would
have made more, and on this workstream that distinction is the entire verdict.

Usage:
    python scripts/collect_spx500_l1.py [--tag l1] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import wandb

ENTITY_PROJECT = "bigcan-chiwin-technology/FinRL-Pro-DS"

# Buy-and-hold total return on each window, measured from the same parquet
# (scripts/analyze_spx500_ab.py companion; see randd_log S554 for the table).
BH = {"val": ("2024-01-02..2024-06-28", 0.1461),
      "test": ("2024-07-01..2024-12-31", 0.0766)}

FIELDS = ["profit_factor", "sharpe", "sortino", "total_return", "max_drawdown",
          "trade_count", "win_rate", "steps"]


def collect(tag: str):
    api = wandb.Api()
    rows = {"ls": [], "lo": []}
    for r in api.runs(ENTITY_PROJECT, order="-created_at"):
        n = r.name or ""
        if f"-{tag}-" not in n or not n.startswith("spx500-"):
            continue
        if r.state != "finished":
            continue
        variant = "ls" if "-ls-" in n else "lo" if "-lo-" in n else None
        if variant is None:
            continue
        s = dict(r.summary)
        row = {"run": n, "state": r.state}
        for split in ("val", "test"):
            for f in FIELDS:
                row[f"{split}_{f}"] = s.get(f"backtest_{split}/{f}")
        rows[variant].append(row)
    return rows


def _stat(vals, fn):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return fn(vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="l1")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows = collect(args.tag)
    print("=" * 92)
    print(f"GMGP1-SPX500 — stage '{args.tag}' results vs BUY-AND-HOLD")
    print("=" * 92)

    for variant, label in (("ls", "A / long-short"), ("lo", "B / long-only")):
        rs = rows[variant]
        print(f"\n### {label}  (n={len(rs)} finished)")
        if not rs:
            print("   none finished yet")
            continue
        print(f"   {'run':<34}{'val_PF':>8}{'val_ret':>10}{'test_PF':>9}{'test_ret':>10}"
              f"{'test_DD':>9}{'trades':>8}")
        for r in sorted(rs, key=lambda x: x["run"]):
            def f(v, pct=False):
                if not isinstance(v, (int, float)):
                    return "n/a"
                return f"{v:+.2%}" if pct else f"{v:.4f}"
            print(f"   {r['run'][:34]:<34}{f(r['val_profit_factor']):>8}"
                  f"{f(r['val_total_return'], True):>10}{f(r['test_profit_factor']):>9}"
                  f"{f(r['test_total_return'], True):>10}"
                  f"{f(r['test_max_drawdown'], True):>9}"
                  f"{(str(r['test_trade_count']) if r['test_trade_count'] else 'n/a'):>8}")
        for split in ("val", "test"):
            pfs = [r[f"{split}_profit_factor"] for r in rs]
            rets = [r[f"{split}_total_return"] for r in rs]
            win, bh = BH[split]
            med_pf, med_ret = _stat(pfs, st.median), _stat(rets, st.median)
            line = f"   {split.upper():<5} median PF={med_pf:.4f}" if med_pf else f"   {split.upper()} n/a"
            if med_ret is not None:
                beat = sum(1 for v in rets if isinstance(v, (int, float)) and v > bh)
                line += (f"  median return={med_ret:+.2%}  vs BUY-AND-HOLD {bh:+.2%} "
                         f"({win})  seeds beating B&H: {beat}/{len(rets)}")
            print(line)
            if med_pf and len(pfs) > 1:
                cv = _stat(pfs, st.pstdev) / med_pf if med_pf else None
                if cv is not None:
                    print(f"         PF CV across seeds = {cv:.3f} "
                          f"(protocol gate l1_pf_cv_max = 0.30)")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
