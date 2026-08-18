"""Collect the gmgp1-spx500 walk-forward MULTISEED stage and test whether the S555
regime pattern survives multiseeding.

S555 ran each walk-forward fold at a SINGLE seed, while its own Stage 2 measured a
long/short seed spread of 18.7pp across 5 seeds — wider than most fold-to-fold gaps in
the table that the regime story was read off. So the fold ranking was directional only.
This reads the N-seed re-run and reports, per (variant, fold), the MEDIAN alongside the
seed spread, so a fold difference can be compared against the noise it has to clear.

Reads `backtest_test/*` — the keys run_full_pipeline actually publishes. `Exposure_Frac`
is NOT on this path (it lives in analytics/wandb_evaluator.py, a different reporting
path), so activity is reported as trade_count only; an absent metric is printed as n/a
and never as a measured zero.

Usage:
    python scripts/collect_spx500_wf_multiseed.py [--stamp 20260818_132822] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

import wandb

ENTITY_PROJECT = "bigcan-chiwin-technology/FinRL-Pro-DS"

# Buy-and-hold total return over each fold's TEST window, measured from the same parquet
# (scripts/analyze_spx500_ab.py; randd_log S555). The regime label is what the index did,
# and it is the ordering variable the whole S555 finding rests on.
FOLDS = {
    "1": ("2025-01-02..2025-05-30", 0.0024, "flat"),
    "2": ("2025-06-02..2025-10-31", 0.1620, "strong bull"),
    "3": ("2025-11-03..2026-03-31", -0.0460, "bear"),
    "4": ("2026-04-01..2026-08-14", 0.1880, "strong bull"),
}

# The single-seed numbers the multiseed run has to confirm or overturn.
S555 = {("ls", "1"): 0.01655, ("lo", "1"): 0.03397,
        ("ls", "2"): -0.04427, ("lo", "2"): -0.01627,
        ("ls", "3"): -0.04538, ("lo", "3"): -0.05176,
        ("ls", "4"): -0.01549, ("lo", "4"): 0.02305}

FIELDS = ["profit_factor", "sharpe", "total_return", "max_drawdown", "trade_count"]
NAME_RE = re.compile(r"^spx500-(ls|lo)-wfms-f(\d)-s(\d+)_(\d{8}_\d{6})$")


def collect(stamp: str | None):
    api = wandb.Api()
    cells: dict[tuple[str, str], list[dict]] = {}
    other_states: list[str] = []
    for r in api.runs(ENTITY_PROJECT, order="-created_at"):
        m = NAME_RE.match(r.name or "")
        if not m:
            continue
        variant, fold, seed, run_stamp = m.groups()
        if stamp and run_stamp != stamp:
            continue
        if r.state != "finished":
            other_states.append(f"{r.name} [{r.state}]")
            continue
        s = dict(r.summary)
        row = {"seed": int(seed), "run": r.name}
        for f in FIELDS:
            row[f] = s.get(f"backtest_test/{f}")
        cells.setdefault((variant, fold), []).append(row)
    for v in cells.values():
        v.sort(key=lambda x: x["seed"])
    return cells, other_states


def _num(vals):
    return [v for v in vals if isinstance(v, (int, float))]


def summarize(rows, key):
    vals = _num([r[key] for r in rows])
    if not vals:
        return None
    return {"n": len(vals), "median": st.median(vals), "min": min(vals),
            "max": max(vals), "spread": max(vals) - min(vals)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", default=None,
                    help="restrict to one launch stamp (recommended; excludes reruns)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cells, others = collect(args.stamp)
    if not cells:
        print("no finished wfms runs found"
              + (f" for stamp {args.stamp}" if args.stamp else ""))
        if others:
            print("unfinished:", ", ".join(others))
        return 1

    print("=" * 108)
    print("GMGP1-SPX500 walk-forward MULTISEED — per-fold MEDIAN with seed spread")
    print("=" * 108)

    out = {}
    for variant, label in (("ls", "A / long-short"), ("lo", "B / long-only")):
        print(f"\n### {label}")
        print(f"   {'fold':<6}{'regime':<13}{'B&H':>9}{'n':>4}{'med ret':>10}{'spread':>9}"
              f"{'min':>9}{'max':>9}{'med PF':>9}{'med DD':>9}{'S555':>9}{'beat':>6}")
        for fold in ("1", "2", "3", "4"):
            rows = cells.get((variant, fold), [])
            win, bh, regime = FOLDS[fold]
            if not rows:
                print(f"   f{fold:<5}{regime:<13}{bh:>+8.2%}{0:>4}   (no finished runs)")
                continue
            ret = summarize(rows, "total_return")
            pf = summarize(rows, "profit_factor")
            dd = summarize(rows, "max_drawdown")
            beat = sum(1 for r in rows
                       if isinstance(r["total_return"], (int, float)) and r["total_return"] > bh)
            s555 = S555[(variant, fold)]
            print(f"   f{fold:<5}{regime:<13}{bh:>+8.2%}{ret['n']:>4}{ret['median']:>+10.2%}"
                  f"{ret['spread']:>9.2%}{ret['min']:>+9.2%}{ret['max']:>+9.2%}"
                  f"{(pf['median'] if pf else float('nan')):>9.4f}"
                  f"{(dd['median'] if dd else float('nan')):>+9.2%}"
                  f"{s555:>+9.2%}{beat:>3}/{ret['n']}")
            out[f"{variant}_f{fold}"] = {
                "regime": regime, "window": win, "buy_hold": bh, "s555_single_seed": s555,
                "return": ret, "profit_factor": pf, "max_drawdown": dd,
                "seeds_beating_bh": beat,
                "per_seed": {r["seed"]: r["total_return"] for r in rows},
            }

    # The question the stage exists to answer: is the fold ORDERING robust, or is it
    # inside the seed noise? Compare the gap between adjacent folds (ranked by median)
    # against the wider of the two folds' own seed spreads.
    print("\n" + "=" * 108)
    print("IS THE FOLD ORDERING SEPARABLE FROM SEED NOISE?")
    print("=" * 108)
    for variant, label in (("ls", "long-short"), ("lo", "long-only")):
        ranked = sorted(
            ((f, out[f"{variant}_f{f}"]) for f in ("1", "2", "3", "4")
             if f"{variant}_f{f}" in out),
            key=lambda kv: kv[1]["return"]["median"], reverse=True)
        if len(ranked) < 2:
            continue
        print(f"\n{label}: fold order by median return = "
              + " > ".join(f"f{f}({d['regime']})" for f, d in ranked))
        for (fa, da), (fb, db) in zip(ranked, ranked[1:]):
            gap = da["return"]["median"] - db["return"]["median"]
            noise = max(da["return"]["spread"], db["return"]["spread"])
            verdict = "SEPARATED" if gap > noise else "INSIDE NOISE"
            print(f"   f{fa} vs f{fb}: gap {gap:>7.2%} vs worse seed spread {noise:>7.2%}"
                  f"  -> {verdict}")

    if others:
        print(f"\n{len(others)} run(s) not finished: " + ", ".join(others[:12]))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
