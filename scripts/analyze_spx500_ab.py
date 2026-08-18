"""Head-to-head report for the gmgp1-spx500 long/short vs long-only A/B (Stage 1 HPO).

WHAT THIS IS AND IS NOT. It summarises a hyperparameter SEARCH, not a strategy evaluation.
HPO trials are not iid draws: TPE steers later trials toward earlier winners, so the trials
within an arm are a dependent trajectory. Rank tests across arms are therefore reported as
DESCRIPTIVE effect sizes with the non-independence stated, never as significance tests of
"variant B is better" — that claim requires the L1 multiseed stage, where independent seeds
give genuinely exchangeable samples.

Two axes are reported side by side on purpose. Profit factor is the HPO objective (BUG-01),
but it is SCALE-FREE: a book taking tiny high-hit-rate positions scores well while barely
moving capital. Total return is what actually pays. On this study those two disagree at the
top of the long-only arm, and reporting only the objective would invert the verdict.

Usage:
    python scripts/analyze_spx500_ab.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as stats
from pathlib import Path

import optuna
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"A / long-short": "gmgp1_spx500_ls_hpo_20260815",
        "B / long-only": "gmgp1_spx500_lo_hpo_20260815"}
KILL = -900.0          # sentinel floor: values <= this are gate kills, not scores


def summarize(trials):
    done = [t for t in trials if t.value is not None]
    killed = [t for t in done if t.value <= KILL]
    scored = [t for t in done if t.value > KILL]
    pf = [t.value for t in scored]
    ret = [t.user_attrs.get("total_return") for t in scored
           if t.user_attrs.get("total_return") is not None]
    hurdle = next((t.user_attrs.get("buy_hold_return") for t in scored
                   if t.user_attrs.get("buy_hold_return") is not None), None)
    beat = [t for t in scored if t.user_attrs.get("beat_buy_hold") is True]
    return {
        "n_completed": len(done), "n_killed": len(killed), "n_scored": len(scored),
        "pf_max": max(pf) if pf else None,
        "pf_median": stats.median(pf) if pf else None,
        "pf_min": min(pf) if pf else None,
        "pf_frac_above_1": (sum(1 for v in pf if v > 1.0) / len(pf)) if pf else None,
        "ret_max": max(ret) if ret else None,
        "ret_median": stats.median(ret) if ret else None,
        "ret_min": min(ret) if ret else None,
        "hurdle": hurdle,
        "n_beat_hurdle": len(beat),
        "argmax_pf": _trial_row(max(scored, key=lambda t: t.value)) if scored else None,
        "argmax_ret": _trial_row(max(
            [t for t in scored if t.user_attrs.get("total_return") is not None],
            key=lambda t: t.user_attrs["total_return"])) if ret else None,
    }


def _trial_row(t):
    return {"number": t.number, "pf": t.value,
            "total_return": t.user_attrs.get("total_return"),
            "beat_buy_hold": t.user_attrs.get("beat_buy_hold"),
            "params": {k: (round(v, 6) if isinstance(v, float) else v)
                       for k, v in t.params.items()}}


def main() -> int:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    storage = optuna.storages.RDBStorage(os.environ["DISTRIBUTED_HPO_DB_URL"])
    out = {}
    for label, name in ARMS.items():
        out[label] = summarize(optuna.load_study(study_name=name, storage=storage).trials)

    print("=" * 78)
    print("GMGP1-SPX500 — Stage 1 HPO head-to-head (search summary, NOT a strategy verdict)")
    print("=" * 78)
    hdr = f"{'metric':<26}" + "".join(f"{k:>25}" for k in out)
    print(hdr)
    for key, fmt in [("n_completed", "{}"), ("n_killed", "{}"), ("n_scored", "{}"),
                     ("pf_max", "{:.4f}"), ("pf_median", "{:.4f}"), ("pf_min", "{:.4f}"),
                     ("pf_frac_above_1", "{:.1%}"),
                     ("ret_max", "{:+.4%}"), ("ret_median", "{:+.4%}"),
                     ("ret_min", "{:+.4%}"), ("n_beat_hurdle", "{}")]:
        row = f"{key:<26}"
        for label in out:
            v = out[label][key]
            row += f"{(fmt.format(v) if v is not None else 'n/a'):>25}"
        print(row)

    h = next((o["hurdle"] for o in out.values() if o["hurdle"]), None)
    if h:
        print(f"\nbuy-and-hold hurdle (same env, same episodic protocol): {h:+.4%}")

    for label, o in out.items():
        print(f"\n--- {label} ---")
        for tag in ("argmax_pf", "argmax_ret"):
            r = o[tag]
            if not r:
                continue
            print(f"  {tag:<11} trial {r['number']:>3}  PF={r['pf']:.4f}  "
                  f"return={r['total_return']:+.4%}  beat_hurdle={r['beat_buy_hold']}")
        if o["argmax_pf"] and o["argmax_ret"] and o["argmax_pf"]["number"] != o["argmax_ret"]["number"]:
            print("  NOTE: the PF-best and return-best trials DIFFER — selecting on the HPO "
                  "objective does not select the most profitable model in this arm.")

    total_beat = sum(o["n_beat_hurdle"] for o in out.values())
    total_scored = sum(o["n_scored"] for o in out.values())
    print(f"\nVERDICT INPUT: {total_beat}/{total_scored} scored trials beat buy-and-hold.")
    if total_beat == 0:
        print("  => No hyperparameter setting in EITHER arm cleared the passive benchmark.")
        print("     Read as NEGATIVE on deployability. The A/B still ranks the two variants,")
        print("     but it ranks them below doing nothing.")
    print("\nCAVEAT: HPO trials are a TPE-steered trajectory, not iid samples. Differences")
    print("between arms here are descriptive. Inference needs the L1 multiseed stage.")

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
