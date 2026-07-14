"""Crucible Cross-Market Pooling — Stage-0 Part A (power feasibility).

Pre-registration: docs/research/crucible_crossmarket_pooling_stage0_preregistration_2026-07-13.md

THE make-or-break question: given the funnel's MEASURED per-market MDE anchor (1.403 at holdout 1011,
the E1/E2 calibration), how many INDEPENDENT gold-positioning markets `k` must a fixed/random-effects
pool combine to drag the pooled MDE below the 0.50 realistic-alpha ceiling — and do that many exist?

Pooling law (equicorrelated-mean variance, Math-verified inline + MC self-checked):
    Var(mean of k unit-var estimators, pairwise corr rho) = (1 + (k-1)*rho) / k
    MDE_pool = m * sqrt((1 + (k-1)*rho) / k)
    k_needed(rho) = (1 - rho) / (f - rho),  f = (c/m)**2,  valid iff rho < f
    correlation floor: MDE_pool -> m*sqrt(rho) as k->inf  =>  rho >= f can NEVER reach c

Analytic + Monte-Carlo only. No funnel gate byte changes; gates_hash 519158fa1450 untouched.
Run: python scripts/research/crucible_crossmarket_power.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SWEEP_JSON = ROOT / "results" / "crucible_calibration" / "calibration_mde_sweep.json"
OUT_JSON = ROOT / "results" / "crucible_crossmarket_power" / "crossmarket_power.json"

SEED = 20260713           # frozen in the pre-registration
CEILING = 0.50            # E2 realistic-alpha ceiling (project_crucible_calibration_e1e2_s553)
ANCHOR_HOLDOUT = 1011     # deepest real daily substrate
ANCHOR_MDE = 1.403        # expected per-market MDE at that holdout (TC-2 target)
RHO_BAND = (0.0, 0.05, 0.10, 0.127)         # optimistic -> correlation floor
K_AVAILABLE = {"base (US COT + TW T86)": 2, "generous (related metals/venues)": 4}
M_SINGLE_SENS = (1.10, 1.20)                # pre-registered single-test MDE sensitivities (decision aid)


# ---- shipped interpolation law (copied byte-for-byte from orchestrator/substrate.interp_mde) --------
def interp_mde(holdout_bars: int, sweep: dict) -> float:
    """MDE at a holdout, interpolated in x=1/sqrt(N) between the sweep's grid points (the exact law the
    live power guard uses). Used ONLY to reproduce the 1.403 anchor (TC-2) — no new statistics."""
    rows = sorted(sweep["mde_sweep"]["rows"], key=lambda r: r["holdout_bars"])
    pts = [(int(r["holdout_bars"]), float(r["mde_realized_delta_sr"])) for r in rows]
    h = int(holdout_bars)
    for hi, mi in pts:
        if hi == h:
            return mi
    h_lo, m_lo = pts[0]
    h_hi, m_hi = pts[-1]
    if h < h_lo:
        return m_lo * (h_lo / h) ** 0.5
    if h > h_hi:
        return m_hi * (h_hi / h) ** 0.5
    for (a_h, a_m), (b_h, b_m) in zip(pts, pts[1:]):
        if a_h <= h <= b_h:
            x, xa, xb = h ** -0.5, a_h ** -0.5, b_h ** -0.5
            return a_m + (b_m - a_m) * (x - xa) / (xb - xa)
    return m_hi


# ---- pooling law -------------------------------------------------------------------------------------
def pooled_var_factor(k: int, rho: float) -> float:
    """Var(equal-weight mean of k unit-variance, pairwise-rho estimators) = (1 + (k-1)rho)/k."""
    return (1.0 + (k - 1) * rho) / k


def mde_pool(m: float, k: int, rho: float) -> float:
    return m * math.sqrt(pooled_var_factor(k, rho))


def k_needed(m: float, rho: float, c: float = CEILING) -> float:
    """Markets needed to reach ceiling c. inf when rho >= f (correlation floor dominates)."""
    f = (c / m) ** 2
    if rho >= f:
        return math.inf
    return (1.0 - rho) / (f - rho)


def _fmt_k(v: float) -> str:
    return "inf" if math.isinf(v) else f"{v:.1f}"


# ---- tripwires ---------------------------------------------------------------------------------------
def tc1_montecarlo(rng: np.random.Generator, n: int = 300_000) -> dict:
    """TC-1: empirical Var(mean of k equicorrelated unit-var estimators) matches the closed form."""
    worst = 0.0
    rows = []
    for k in (2, 3, 4, 8):
        for rho in (0.0, 0.05, 0.10, 0.20):
            cov = (1 - rho) * np.eye(k) + rho * np.ones((k, k))
            L = np.linalg.cholesky(cov)
            z = rng.standard_normal((k, n))
            x = L @ z                                   # k x n, each row ~N(0,1), pairwise corr rho
            emp = float(x.mean(axis=0).var())
            closed = pooled_var_factor(k, rho)
            rel = abs(emp - closed) / closed
            worst = max(worst, rel)
            rows.append({"k": k, "rho": rho, "empirical": emp, "closed": closed, "rel_err": rel})
    return {"pass": worst < 0.01, "worst_rel_err": worst, "rows": rows}


def tc2_anchor(sweep: dict) -> dict:
    got = interp_mde(ANCHOR_HOLDOUT, sweep)
    return {"pass": abs(got - ANCHOR_MDE) < 1e-3, "got": got, "expected": ANCHOR_MDE}


def tc3_limits() -> dict:
    k1 = mde_pool(ANCHOR_MDE, 1, 0.0)                   # k=1 -> unchanged
    rho1 = [mde_pool(ANCHOR_MDE, k, 1.0) for k in (2, 5, 20)]  # rho=1 -> unchanged for all k
    ok = abs(k1 - ANCHOR_MDE) < 1e-12 and all(abs(v - ANCHOR_MDE) < 1e-9 for v in rho1)
    return {"pass": ok, "k1": k1, "rho1": rho1, "m": ANCHOR_MDE}


def main() -> None:
    sweep = json.loads(SWEEP_JSON.read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)

    tw = {"tc1_montecarlo": tc1_montecarlo(rng), "tc2_anchor": tc2_anchor(sweep), "tc3_limits": tc3_limits()}
    all_green = all(t["pass"] for t in tw.values())

    print("=" * 78)
    print("Crucible Cross-Market Pooling — Stage-0 Part A (power feasibility)")
    print("=" * 78)
    print("\nTRIPWIRES (must all pass before reading the verdict):")
    print(f"  TC-1 MC vs closed-form pooling var : {'PASS' if tw['tc1_montecarlo']['pass'] else 'FAIL'}"
          f"  (worst rel-err {tw['tc1_montecarlo']['worst_rel_err']:.4%})")
    print(f"  TC-2 anchor reproduces MDE 1.403   : {'PASS' if tw['tc2_anchor']['pass'] else 'FAIL'}"
          f"  (got {tw['tc2_anchor']['got']:.4f} at holdout {ANCHOR_HOLDOUT})")
    print(f"  TC-3 degenerate limits k=1 / rho=1 : {'PASS' if tw['tc3_limits']['pass'] else 'FAIL'}")
    if not all_green:
        print("\nTRIPWIRE FAIL — verdict NOT read (per pre-registration). Fix before interpreting.")
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps({"tripwires": tw, "verdict": "TRIPWIRE_FAIL"}, indent=2))
        return

    m = ANCHOR_MDE
    f = (CEILING / m) ** 2
    print(f"\nAnchor per-market MDE m = {m}  |  ceiling c = {CEILING}  |  f = (c/m)^2 = {f:.4f}")
    print(f"Correlation floor: pooling can EVER reach {CEILING} only if rho < {f:.3f} "
          f"(else MDE_pool -> m*sqrt(rho) >= {m*math.sqrt(f):.2f}).")

    print("\nMarkets needed to reach ceiling  k_needed(rho)  [gate uses measured m=1.403]:")
    print(f"  {'rho':>6} | {'m=1.403':>10} | {'m=1.20':>10} | {'m=1.10':>10}")
    frontier = []
    for rho in RHO_BAND:
        kn = [k_needed(mm, rho) for mm in (1.403, 1.20, 1.10)]
        frontier.append({"rho": rho, "k_needed_m1403": kn[0], "k_needed_m120": kn[1], "k_needed_m110": kn[2]})
        print(f"  {rho:>6.3f} | {_fmt_k(kn[0]):>10} | {_fmt_k(kn[1]):>10} | {_fmt_k(kn[2]):>10}")

    print("\nPooled MDE at the AVAILABLE market counts (measured m=1.403):")
    achieved = []
    for label, k in K_AVAILABLE.items():
        row = {"label": label, "k": k}
        cells = []
        for rho in RHO_BAND:
            mp = mde_pool(m, k, rho)
            row[f"mde_rho_{rho}"] = mp
            cells.append(f"rho={rho}: {mp:.2f}")
        achieved.append(row)
        print(f"  k={k:>2} ({label}):  " + "  ".join(cells))

    # Verdict: A-PASS iff k_needed at the realistic rho band <= k_available (base=2). Optimistic rho=0.
    k_avail_base = K_AVAILABLE["base (US COT + TW T86)"]
    kn0 = k_needed(m, 0.0)                              # most optimistic: independent markets
    a_pass = kn0 <= k_avail_base
    verdict = "A-PASS" if a_pass else "A-FAIL"
    print("\n" + "=" * 78)
    print(f"VERDICT: {verdict}")
    print(f"  Even at the optimistic rho=0 (independent markets), pooling needs k>={math.ceil(kn0)} "
          f"markets to reach {CEILING}; only k={k_avail_base} same-gold-mechanism markets exist "
          f"(US COT + TW T86).")
    if not a_pass:
        print(f"  For any rho>=0.127 pooling can NEVER reach {CEILING} (correlation floor). The single-test "
              f"sensitivity (m=1.10) still needs k>={math.ceil(k_needed(1.10,0.0))} at rho=0.")
        print("  => Third power lever CLOSED. Only the not-recommended less-deflated gate remains. "
              "Redirect: TSMOM -> paper. Part B (US data pull) NOT run per pre-registration.")
    print("=" * 78)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "seed": SEED, "anchor_mde": m, "ceiling": CEILING, "f": f,
        "k_available": K_AVAILABLE, "tripwires": tw,
        "k_needed_frontier": frontier, "pooled_mde_at_available": achieved,
        "verdict": verdict, "k_needed_rho0": kn0,
    }, indent=2))
    print(f"\nWrote {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
