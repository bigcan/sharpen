"""Crucible F5 — the MATCHED-NULL Hall-of-Fame ceiling (independent-audit finding F4).

WHY THIS EXISTS (S553-cont-139, anatomy-paper build plan §4 figure F5)
----------------------------------------------------------------------
The Crucible record's "best discoveries" (train-split HoF ΔSR up to 0.99 / 0.71 / 0.66; marginal_t
≤ 2.12) were read as evidence the search found *something*. The independent audit
(``docs/research/crucible_independent_audit_report_2026-07-14.md`` §2 "(A)" / finding F4) showed they
are **the noise ceiling**: running the SHIPPED ``evolve`` + gate on PURE NOISE at the real search
budget (pop 200 × gens 40) produces a Hall-of-Fame whose MAX ΔSR reaches **0.80–0.94** and whose MAX
marginal_t reaches **~2.93** — at or ABOVE the real record's maxima. The record is exactly what
no-signal looks like *through this instrument*; nothing survives the binding gate (0 PROMISING).

This script rebuilds the auditor's gone ``scratchpad/phase2/null_grid_sim.py`` as a reproducible
experiment. It measures the per-search HoF maxima across noise seeds (the ceiling distribution) and
prints them against the real-record markers.

KEY SETUP CHOICES (faithful to the audit)
  * base book rescaled to ~0 train Sharpe — the un-sign-sealed regime (matching the real Taiwan train
    book, SR ≈ −0.007). A strongly positive base book would sign-seal marginal_t (F1/F3) and the noise
    ceiling in t would not appear; the record's positive-t maxima came from a ~0-Sharpe base era.
  * real substrate length (T≈2782, Taiwan-like) and n≈18 (cross_asset-like): the t-ceiling scales
    √T, so a short panel understates it (probe: t 1.84 @ T=1200 → ~2.8 @ T=2782).
  * cross_sectional candidate type — the OHLCV DSL search that produced the "best OOS ΔSR 0.66" record.

Zero funnel/gate bytes touched (CRU-1 safe): reads ``evolve`` / ``combination_fitness`` as shipped.
Writes the report AFTER EACH SEED (long run; a kill keeps completed seeds).

Usage (from repo root):
    python scripts/research/crucible_matched_null.py                 # real budget (slow, ~30-45 min)
    python scripts/research/crucible_matched_null.py --quick         # tiny budget smoke test
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.research.crucible_calibration as cal  # noqa: E402  (reuse the E1 pure-noise substrate)
from sharpen.signals.generation.evolve import evolve  # noqa: E402
from sharpen.signals.generation.fitness import FitnessConfig  # noqa: E402
from sharpen.signals.library._alpha_formulas import FORMULAS  # noqa: E402

log = logging.getLogger("crucible_matched_null")

# The real record's maxima (independent audit §2 "(A)" / F4, verified against the ledger). These are
# CROSS-CHECK markers only — the ledger is proprietary/absent, so they appear here as constants, not
# reproduced. The figure's own empirical content is the synthetic noise ceiling below.
REAL_RECORD_DELTA_SR_MAXES = (0.99, 0.71, 0.66)
REAL_RECORD_MARGINAL_T_MAX = 2.12

# Real search budget (audit §2 "(A)"): pop 200 × gens 40. --quick shrinks it for a smoke test only.
_POP_SIZE = 200
_N_GENERATIONS = 40
_T = 2782          # Taiwan-like train length (t-ceiling scales √T)
_N = 18            # cross_asset-like cross-section
_N_SEEDS = 6


def _zero_sharpe_base(t: int, *, seed: int) -> dict[str, np.ndarray]:
    """Two ~0-mean sleeves — a base book at ~0 train Sharpe (matches the real Taiwan train era, the
    un-sign-sealed regime in which marginal_t can reach a positive noise ceiling)."""
    rng = np.random.default_rng(seed)
    return {"tsmom": 0.010 * rng.standard_normal(t),
            "rates_carry": 0.008 * rng.standard_normal(t)}


def _finite_max(xs: list[float]) -> float:
    xs = [x for x in xs if np.isfinite(x)]
    return max(xs) if xs else float("nan")


def _summary(rows: list[dict[str, Any]], budget: dict) -> dict:
    dsr = [r["hof_max_delta_sr"] for r in rows if np.isfinite(r["hof_max_delta_sr"])]
    mt = [r["hof_max_marginal_t"] for r in rows if np.isfinite(r["hof_max_marginal_t"])]
    ceil = {
        "max_delta_sr": (max(dsr) if dsr else float("nan")),
        "p90_delta_sr": (float(np.quantile(dsr, 0.90)) if dsr else float("nan")),
        "median_delta_sr": (float(np.median(dsr)) if dsr else float("nan")),
        "max_marginal_t": (max(mt) if mt else float("nan")),
        "median_marginal_t": (float(np.median(mt)) if mt else float("nan")),
        "total_promising": int(sum(r["n_promising"] for r in rows)),
    }
    rec_dsr = max(REAL_RECORD_DELTA_SR_MAXES)
    return {
        "budget": budget,
        "n_seeds_done": len(rows),
        "rows": rows,
        "noise_ceiling": ceil,
        "real_record": {"delta_sr_maxes": list(REAL_RECORD_DELTA_SR_MAXES),
                        "marginal_t_max": REAL_RECORD_MARGINAL_T_MAX},
        # The claim: the real record's maxima sit AT OR BELOW the pure-noise ceiling (with a small
        # tolerance on ΔSR, which is "at" the ceiling; marginal_t is clearly below it).
        "record_at_or_below_ceiling": {
            "delta_sr": bool(np.isfinite(ceil["max_delta_sr"]) and rec_dsr <= ceil["max_delta_sr"] + 0.05),
            "marginal_t": bool(np.isfinite(ceil["max_marginal_t"])
                               and REAL_RECORD_MARGINAL_T_MAX <= ceil["max_marginal_t"]),
        },
        "note": ("noise HoF ceiling from the shipped evolve+gate on pure noise; 0 PROMISING = nothing "
                 "survives the binding gate. Real-record maxima are cross-check markers, not reproduced."),
    }


def run_matched_null(cfg: FitnessConfig, *, t: int, n: int, pop_size: int, n_generations: int,
                     n_seeds: int, out_path: Path | None = None) -> dict:
    """Run the shipped cross_sectional evolve on ``n_seeds`` pure-noise panels at the given budget and
    record the per-search Hall-of-Fame MAX ΔSR and marginal_t (the noise ceiling). Writes the running
    summary to ``out_path`` after each seed when given (the run is long)."""
    seed_formulas = [FORMULAS[i] for i in cal._CS_SEED_NUMS]
    budget = {"pop_size": pop_size, "n_generations": n_generations, "t": t, "n": n,
              "candidate_type": "cross_sectional", "base_book": "~0 train Sharpe (un-sign-sealed)"}
    log.info("matched-null: %d noise seeds x evolve(pop=%d, gen=%d) on T=%d, n=%d",
             n_seeds, pop_size, n_generations, t, n)
    rows: list[dict[str, Any]] = []
    for k in range(n_seeds):
        panel = cal._noise_panel(t, n, seed=2000 + k, n_feature_slots=4)
        ts = cal._panel_ts(panel)
        base = _zero_sharpe_base(t, seed=2000 + k)
        t0 = time.time()
        rep = evolve(seed_formulas, panel, base, ts, cfg, candidate_type="cross_sectional",
                     rng_seed=7, pop_size=pop_size, n_generations=n_generations, ls_min_names=6,
                     elite_frac=0.30, cost_bps=0.0010, holdout_frac=0.25, holdout_embargo=21,
                     hold_horizon=21)
        dt = time.time() - t0
        hof = [c for c in rep.hall_of_fame if c.result is not None]
        row = {
            "seed": 2000 + k,
            "gen_n_total": int(rep.gen_n_total),
            "hof_max_delta_sr": _finite_max([float(c.result.delta_sr_oos) for c in hof]),
            "hof_max_marginal_t": _finite_max([float(c.result.marginal_t) for c in hof]),
            "n_promising": len(rep.promising),
            "elapsed_s": round(dt, 1),
        }
        rows.append(row)
        log.info("  seed %d: gen_n=%d  HoFmax dSR=%.3f t=%.3f  promising=%d  (%.0fs)",
                 row["seed"], row["gen_n_total"], row["hof_max_delta_sr"],
                 row["hof_max_marginal_t"], row["n_promising"], dt)
        if out_path is not None:
            _write(out_path, _summary(rows, budget))
    return _summary(rows, budget)


def _write(out_path: Path, summary: dict) -> None:
    report = {
        "artifact": "crucible_matched_null_F5",
        "ts": datetime.now(timezone.utc).isoformat(),
        "audit_ref": "independent audit F4 (2026-07-14 §2 '(A)'); anatomy-paper figure F5",
        **summary,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="Crucible F5: matched-null Hall-of-Fame ceiling (audit F4)")
    ap.add_argument("--pop-size", type=int, default=_POP_SIZE)
    ap.add_argument("--n-generations", type=int, default=_N_GENERATIONS)
    ap.add_argument("--t", type=int, default=_T)
    ap.add_argument("--n", type=int, default=_N)
    ap.add_argument("--n-seeds", type=int, default=_N_SEEDS)
    ap.add_argument("--embargo", type=int, default=10)
    ap.add_argument("--quick", action="store_true",
                    help="tiny budget/short panel — smoke test the harness, NOT the ceiling.")
    ap.add_argument("--out", default=str(ROOT / "results" / "crucible_matched_null"))
    args = ap.parse_args()

    pop = 20 if args.quick else args.pop_size
    gen = 3 if args.quick else args.n_generations
    t = 700 if args.quick else args.t
    n = 10 if args.quick else args.n
    n_seeds = 2 if args.quick else args.n_seeds

    cfg = FitnessConfig(embargo=args.embargo)
    out_dir = Path(args.out)
    stem = "matched_null_quick" if args.quick else "matched_null"
    out_path = out_dir / f"{stem}.json"
    summary = run_matched_null(cfg, t=t, n=n, pop_size=pop, n_generations=gen,
                               n_seeds=n_seeds, out_path=out_path)

    ceil = summary["noise_ceiling"]
    rc = summary["record_at_or_below_ceiling"]
    print("\n================ CRUCIBLE F5: MATCHED-NULL HALL-OF-FAME CEILING ================")
    print(f"budget: evolve(pop={pop}, gen={gen}) on {n_seeds} pure-noise panels, T={t}, n={n}, "
          f"base ~0 Sharpe")
    print(f"   {'seed':>6} {'gen_n':>7} {'HoFmax_dSR':>11} {'HoFmax_t':>9} {'promising':>10} {'sec':>6}")
    for r in summary["rows"]:
        print(f"   {r['seed']:>6} {r['gen_n_total']:>7} {r['hof_max_delta_sr']:>11.3f} "
              f"{r['hof_max_marginal_t']:>9.3f} {r['n_promising']:>10} {r['elapsed_s']:>6.0f}")
    print(f"NOISE CEILING : max dSR={ceil['max_delta_sr']:.3f} (p90 {ceil['p90_delta_sr']:.3f}, "
          f"median {ceil['median_delta_sr']:.3f}) | max t={ceil['max_marginal_t']:.3f} "
          f"(median {ceil['median_marginal_t']:.3f}) | total PROMISING={ceil['total_promising']}")
    print(f"REAL RECORD   : dSR maxes={REAL_RECORD_DELTA_SR_MAXES} | t max={REAL_RECORD_MARGINAL_T_MAX}")
    print(f"RECORD <= NOISE CEILING?  dSR: {rc['delta_sr']}   marginal_t: {rc['marginal_t']}")
    print(f"report -> {out_path}")
    print("===============================================================================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
