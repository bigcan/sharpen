"""Check the ATL x Jev report's headline numbers against the saved result files in this repo.

    python studies/atl-jev/verify.py

CPU only, standard library only, no keys, no data download (a few seconds). It reads the saved screening results
in results/atl_jev/ (written by scripts/research/atl_jev_evaluate.py and the Jev scoring run on 2026-09-24) and
asserts every figure the report and the video quote. Exit code 0 means 0 mismatches.

What this does NOT do: it does not re-score the filings (that needs a Jev API key and costs money) and it does
not rebuild the price panel. It re-derives the numbers from what those runs saved.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "atl_jev"

fails: list[str] = []


def load(rel: str) -> dict:
    return json.loads((RES / rel).read_text(encoding="utf-8"))


def check(label: str, got, want, tol: float | None = None) -> None:
    if tol is None:
        ok = got == want
    else:
        ok = abs(got - want) <= tol
    mark = "OK  " if ok else "FAIL"
    print(f"{mark} {label}: {got!r} (expected {want!r}{'' if tol is None else f' +/- {tol}'})")
    if not ok:
        fails.append(label)


p1, p2, p3, scr = (load(f"phase3/{n}.json") for n in ("p1", "p2", "p3", "screening"))
scores = load("phase3/scores_screening_manifest.json")
corpus = load("phase2/corpus_screening_manifest.json")
ph0 = load("phase0/phase0_summary.json")

cards = {c["name"]: c for c in p1["scorecard"]["cards"]}

print("== Screening (P1 funnel, 2012-2024)")
check("signals tested", len(cards), 5)
check("signals promoted (PROMISING)", p1["promising"], ["jev-surprise-63"])
check("multiplicity charged (trials)", p1["scorecard"]["n_multiplicity"], 8)

s = cards["jev-surprise-63"]
ic5 = s["gross"]["by_horizon"]["5"]
check("jev-surprise-63 verdict", s["verdict"], "PROMISING")
check("5-day IC", ic5["ic_mean"], 0.0090, 0.00005)
check("IC t-statistic (funnel)", ic5["ic_tstat"], 3.21, 0.005)
check("deflated Sharpe", s["deflation"]["dsr"], 0.946, 0.0005)
check("FDR q", s["deflation"]["fdr_q"], 0.025, 0.0005)

cap = s["capturability"]["by_cost"]
check("Sharpe before costs", cap["frictionless"]["net_sharpe"], 0.02, 0.005)
check("Sharpe after standard costs", cap["standard"]["net_sharpe"], -0.28, 0.005)
check("Sharpe after harsh costs", cap["harsh"]["net_sharpe"], -0.73, 0.005)
check("turnovers per year", cap["frictionless"]["turnover_ann"], 8.1, 0.05)
check("Sharpe needed to break even at standard costs ('cost wall')", s["capturability"]["cost_wall"], 0.30, 0.005)
check("share of resampled train/test paths positive", s["cpcv"]["frac_paths_positive"], 0.47, 0.005)
check("all four subperiods positive (IC-IR)", all(x > 0 for x in s["robustness"]["subperiod_ic_ir"]), True)
check("verdict caveat: cost-blocked", any("cost-blocked" in c for c in s["caveats"]), True)

print("== The other four signals (logged, not promoted)")
want = {
    "jev-toneinfl-63": (0.0058, 2.28, 0.764, 0.10, -0.04),
    "jev-comp-63": (0.0072, 2.16, 0.723, 0.10, -0.13),
    "jev-comp-21": (0.0032, 0.71, 0.251, 0.45, -0.12),
    "jev-quality-63": (0.0015, 0.66, 0.178, 0.45, -0.15),
}
for name, (ic, t, dsr, q, sr) in want.items():
    c = cards[name]
    h = c["gross"]["by_horizon"][str(c["gross"]["primary_horizon"])]
    check(f"{name} verdict", c["verdict"], "LOGGED")
    check(f"{name} IC", h["ic_mean"], ic, 0.00006)
    check(f"{name} IC t", h["ic_tstat"], t, 0.006)
    check(f"{name} DSR", c["deflation"]["dsr"], dsr, 0.0006)
    check(f"{name} FDR q", c["deflation"]["fdr_q"], q, 0.006)
    check(f"{name} Sharpe before costs", c["capturability"]["frictionless_sharpe"], sr, 0.006)

print("== P2 placebo and P3 baselines")
sp2 = p2["signals"]["jev-surprise-63"]
check("placebo shuffles", sp2["permutations"], 200)
check("placebos at or above the real IC", sp2["placebo_ge_real"], 0)
check("placebo p", sp2["p"], 0.005, 0.0001)
check("P2 pass", sp2["pass"], True)
sp3 = p3["signals"]["jev-surprise-63"]
check("P3 Newey-West t", sp3["t"], 3.18, 0.005)
check("P3 bar", sp3["min_t"], 2.0)
check("P3 pass", sp3["pass"], True)
check("decision", scr["verdict"], "PROCEED to P4")
check("K3 (stop rule) fired", scr["k3_fired"], False)

print("== The run")
st = scores["stats"]
check("filings scored", st["filings"], 24676)
check("filings with text", st["scored"], 24654)
check("answers from Jev", st["questions"], 140614)
check("one model version served", list(st["served_models"]), ["jev-1.13.0"])
check("Jev scoring cost (USD)", scores["jev_usage"]["cost_usd"], 6.57, 0.005)
check("companies", corpus["n_filers"], 584)
check("8-Ks listed for index members", corpus["stats"]["listed"], 98119)
check("corpus window", corpus["stamp"]["rows"], ["2012-01-01", "2024-12-31"])
check("questionnaire hash (frozen before scoring)", scores["questionnaire"]["hash"], "3fc01888e31f")

print("== Phase 0 probes")
ident = ph0["C3_identification"]["summary"]["masked"]
check("masked filings: filer picked from 10 candidates", ident["company_top1"], 0.935, 0.0005)
check("masked filings: year picked", ident["year_top1"], 0.495, 0.0005)
reg = ph0["C1_knowledge_cutoff"]["exploratory_regime_memory_by_year"]
check("regime memory 2020 (monthly correlation)", reg["2020"]["corr_implied_vs_realized"], 0.692, 0.0005)
check("regime memory 2026 (monthly correlation)", reg["2026"]["corr_implied_vs_realized"], -0.719, 0.0005)
check("clean window starts", ph0["C1_knowledge_cutoff"]["adopted"]["clean_window_from"], "2025-02")

print()
if fails:
    print(f"{len(fails)} mismatch(es): {fails}")
    sys.exit(1)
print("0 mismatches")
