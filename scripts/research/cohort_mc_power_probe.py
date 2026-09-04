"""Narrow power probe for the cohort MC null — the one cell that carries the R&D thesis.

The root-cause investigation (`docs/research/crucible_zero_alpha_root_cause_2026-08-09.md` §5b
Finding 4) reports that the selection-aware MC null is the only gate in the system with real power:

    | scenario            | hit rate | pass rate | med p  | real admitted |
    | NULL (0 real)       | 0.00     | 0.00      | 0.4985 | 0.0           |
    | 6/40 real, IR 0.30  | 0.15     | 0.25      | 0.1673 | 5.0           |

and on that rests the strategic inversion the project's forward plan is built on — "power is LINEAR
in hit rate and only sqrt in everything else, so the binding constraint is the HYPOTHESIS BANK."

That table's script (`mc_full.py`) was never committed. Its full re-derivation is expensive (the doc
records 29,869 s on 14 parallel workers), and it CANNOT change the "(A) is unclaimable" verdict,
because the record contains exactly ONE cohort verdict lifetime (§5e) — a power figure cannot fix a
denominator of 1. So this probe deliberately does NOT re-run the whole grid. It re-derives the single
decisive cell (6/40 real at IR 0.30, published 25%) plus the NULL calibration cell, which is what
would actually change the forward R&D decision.

B=49 is defensible here on the doc's own evidence: it reports the B=49 pilot and the B=1000 full run
producing IDENTICAL pass rates (0.00/0.25/0.50/0.75 in both), i.e. the replicate count was not
driving the result. `--n-reps` raises it if you want the production setting.

TWO CONFIGURATIONS, and the difference matters
-----------------------------------------------
The published table was measured at ``K<=20, rho<=0.10`` — NOT the shipped gates. The doc says so
explicitly: "max_cohort_size: 12 / max_pairwise_corr: 0.35 remain at shipped values … what is enabled
today is the WEAKER setting (ensemble multiplier 1.57x vs 2.77x). Quote the measured power against
the configuration it was measured at, not against the shipped one."

So `--config measured` tests the CLAIM and `--config shipped` tests what PRODUCTION would actually
get. Reporting only one of them would repeat the error the doc warns about.

THE NULL CELL IS THE CALIBRATION CHECK, NOT DECORATION
-------------------------------------------------------
A power probe that only measures the alternative cannot distinguish "the gate has power" from "the
gate fires at everything". The null cell must return median p ~ 0.5 and a pass rate ~ alpha; if it
does not, the harness is wrong and the power number means nothing. This is the same two-directional
discipline that caught the `permute_panel` defect in `null_grid_sim.py`.

USAGE
-----
    python scripts/research/cohort_mc_power_probe.py --seeds 24
    python scripts/research/cohort_mc_power_probe.py --config shipped --seeds 24
    python scripts/research/cohort_mc_power_probe.py --n-reps 199 --seeds 32 --json out.json

Self-contained: synthetic streams only, no network, no credentials. The gate under test is the
shipped `cohort_mc.mc_null_pvalue` with the shipped `greedy_decorrelated_admission` inside it.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sharpen.signals.generation.cohort import CohortConfig                  # noqa: E402
from sharpen.signals.generation.cohort_mc import mc_null_pvalue             # noqa: E402
from sharpen.signals.generation.config import (                             # noqa: E402
    load_cohort_config,
    load_generation_config,
)

log = logging.getLogger("cohort_mc_power_probe")

DEFAULT_GATES = "configs/taiwan_signal_eval.gates.yaml"
DEFAULT_COHORT_GATES = "configs/crucible_cohort.gates.yaml"

# Published values this probe is testing (root-cause doc §5b Finding 4, B=1000 / 12 seeds).
PUBLISHED = {
    "null":  {"pass_rate": 0.00, "med_p": 0.4985, "real_admitted": 0.0},
    "power": {"pass_rate": 0.25, "med_p": 0.1673, "real_admitted": 5.0},
}
# The configuration the table was MEASURED at — not the shipped one (see module docstring).
MEASURED_K, MEASURED_RHO = 20, 0.10


def build_cell(*, t: int, pool_size: int, n_real: int, ir: float, base_sharpe: float,
               ppy: float, rng: np.random.Generator
               ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, set[str]]:
    """One synthetic cell: (base_returns, pool_returns, timestamps, names_of_real_signals).

    Planted candidates are mutually independent Gaussians with a drift set so each carries annualized
    Sharpe ``ir`` and is orthogonal to the base in expectation. This mirrors the published setup and
    inherits its stated caveat verbatim: *real* alphas correlate with each other and with the base, so
    they realize a LOWER effective K than independent draws — meaning any power measured here is an
    UPPER bound on what the same gate would deliver on real candidates.
    """
    ts = (np.datetime64("2015-01-05") + np.arange(t) * np.timedelta64(1, "D")) \
        .astype("datetime64[s]").astype(np.int64).astype(np.float64)
    vol = 0.01                                                   # ~15.9% annualized, a normal book
    base = rng.standard_normal(t) * vol + base_sharpe * vol / float(np.sqrt(ppy))
    pool: dict[str, np.ndarray] = {}
    real: set[str] = set()
    for i in range(pool_size):
        nm = f"c{i:02d}"
        # DO NOT demean. Candidates are raw draws whose REALIZED mean varies at random, exactly as a
        # real pool's does — that dispersion is what the admission ranks on, and the whole point of a
        # selection-aware null is to price the selection it induces.
        #
        # A first version forced each candidate to exactly zero sample mean (and then added the drift
        # for planted ones). That makes every standalone Sharpe identically 0.0 in the OBSERVED panel,
        # so the ranking is degenerate and `t_obs` is systematically the smallest value in the
        # comparison, while the bootstrap replicates — resampled, hence with random nonzero means —
        # retain real dispersion. Result: p = 1.0000 on EVERY seed, null and alternative alike. The
        # null calibration cell caught it (median p 1.0000 against the required ~0.50); the power
        # number alone would have looked like a clean refutation of the published 0.25.
        x = rng.standard_normal(t) * vol
        if i < n_real:
            x = x + ir * vol / float(np.sqrt(ppy))               # planted edge, orthogonal to base
            real.add(nm)
        pool[nm] = x
    return {"base": base}, pool, ts, real


def run_cell(label: str, *, args: argparse.Namespace, ccfg: CohortConfig, fcfg,
             n_real: int, ir: float, alpha: float) -> dict:
    """Run one cell across seeds through the SHIPPED MC null and summarize."""
    rows: list[dict] = []
    for s in range(args.seeds):
        rng = np.random.default_rng(args.seed + 7919 * s)
        base, pool, ts, real = build_cell(
            t=args.bars, pool_size=args.pool_size, n_real=n_real, ir=ir,
            base_sharpe=args.base_sharpe, ppy=fcfg.periods_per_year, rng=rng)
        t0 = time.time()
        res = mc_null_pvalue(pool, base, ts, ccfg, fcfg, n_reps=args.n_reps,
                             alpha_cohort=alpha, seed=args.seed + s, resample_base=True)
        rows.append({
            "seed": args.seed + s,
            "p_value": res.p_value,
            "passes": bool(res.passes_mc),
            "t_obs": res.t_obs,
            "n_members": len(res.members_obs),
            "n_real_admitted": sum(1 for m in res.members_obs if m in real),
            "n_valid_reps": res.n_valid_reps,
            "block_length": res.block_length,
            "seconds": time.time() - t0,
        })
        log.info("%s seed %d: p=%.4f pass=%s members=%d real_admitted=%d (%.0fs)",
                 label, args.seed + s, res.p_value, res.passes_mc, len(res.members_obs),
                 rows[-1]["n_real_admitted"], rows[-1]["seconds"])
    ps = [r["p_value"] for r in rows]
    n_pass = sum(1 for r in rows if r["passes"])
    # One-sided exact binomial against the published rate — the directional question.
    binom_p = None
    if n_real > 0 and rows:
        try:
            from scipy import stats  # noqa: PLC0415
            binom_p = float(stats.binomtest(
                n_pass, len(rows), PUBLISHED["power"]["pass_rate"], alternative="less").pvalue)
        except Exception:                                        # noqa: BLE001 - scipy optional
            binom_p = None
    return {
        "binom_p_less": binom_p,
        "cell": label,
        "n_real_planted": n_real,
        "ir": ir,
        "n_seeds": len(rows),
        "pass_rate": n_pass / max(1, len(rows)),
        "n_pass": n_pass,
        "median_p": float(np.median(ps)) if ps else float("nan"),
        "mean_real_admitted": float(np.mean([r["n_real_admitted"] for r in rows])) if rows else 0.0,
        "mean_members": float(np.mean([r["n_members"] for r in rows])) if rows else 0.0,
        "seconds": float(np.sum([r["seconds"] for r in rows])),
        "rows": rows,
    }


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — reported because a pass rate over ~24 seeds is a WIDE estimate.

    Quoting a bare 0.25 from 12 seeds (3/12) hides a 95% interval of roughly [0.09, 0.53]; without
    the interval a probe can look like it confirms or refutes a published number when it does
    neither.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _print(out: dict) -> None:
    cfg = out["config"]
    print("\n" + "=" * 100)
    print("COHORT MC-NULL POWER PROBE — the one cell that carries the R&D thesis")
    print("=" * 100)
    print(f"  config={cfg['name']}  K<={cfg['max_cohort_size']}  rho<={cfg['max_pairwise_corr']}  "
          f"B={cfg['n_reps']}  alpha={cfg['alpha_cohort']}  seeds={cfg['seeds']}")
    print(f"  panel {cfg['bars']} bars (~{cfg['bars'] / 252:.1f}y)  pool {cfg['pool_size']}  "
          f"base ann SR {cfg['base_sharpe']:+.2f}")
    if cfg["name"] == "shipped":
        print("  NOTE: the published table was measured at K<=20/rho<=0.10, NOT these shipped")
        print("        values — this run measures what PRODUCTION gets, not the published claim.")
    print("-" * 100)
    print(f"  {'cell':<22} {'pass rate':>12} {'95% CI':>16} {'med p':>9} {'real adm':>9} "
          f"{'members':>8}  published")
    for c in out["cells"]:
        lo, hi = wilson_ci(c["n_pass"], c["n_seeds"])
        pub = PUBLISHED["null" if c["n_real_planted"] == 0 else "power"]
        print(f"  {c['cell']:<22} {c['pass_rate']:>11.3f} "
              f"{'[' + f'{lo:.2f}' + ',' + f'{hi:.2f}' + ']':>16} {c['median_p']:>9.4f} "
              f"{c['mean_real_admitted']:>9.1f} {c['mean_members']:>8.1f}  "
              f"pass {pub['pass_rate']:.2f} / p {pub['med_p']:.3f}")
    print("-" * 100)

    null = next((c for c in out["cells"] if c["n_real_planted"] == 0), None)
    pwr = next((c for c in out["cells"] if c["n_real_planted"] > 0), None)
    if null is not None:
        ok = abs(null["median_p"] - 0.5) <= 0.15 and null["pass_rate"] <= 0.20
        print(f"  CALIBRATION (null cell): median p {null['median_p']:.4f} (want ~0.50), "
              f"pass rate {null['pass_rate']:.3f} (want <= alpha)  -> "
              f"{'OK' if ok else 'BROKEN'}")
        if not ok:
            print("    The null is miscalibrated, so the power number below means NOTHING.")
            print("    Fix the harness before reading any verdict off this run.")
            return
    if pwr is not None:
        lo, hi = wilson_ci(pwr["n_pass"], pwr["n_seeds"])
        claim = PUBLISHED["power"]["pass_rate"]
        print(f"\n  POWER at 6/40 real, IR 0.30: measured {pwr['pass_rate']:.3f} "
              f"95% CI [{lo:.2f},{hi:.2f}]  vs published {claim:.2f}")
        # BOTH instruments, because they can disagree and picking one after seeing the data is how
        # a result gets oversold. The two-sided Wilson interval answers "what rates are compatible
        # with this sample"; the ONE-SIDED binomial answers the question actually being asked —
        # "is the true power BELOW the published figure" — and is the stricter, more apposite test.
        p_less = pwr.get("binom_p_less")
        if p_less is not None:
            print(f"  one-sided binomial P(X <= {pwr['n_pass']} | p={claim:.2f}) = {p_less:.4f}")
        print(f"  interval containment: {'contains' if lo <= claim <= hi else 'excludes'} {claim:.2f}"
              f"   |   directional test: "
              f"{'BELOW' if (p_less is not None and p_less < 0.05) else 'not separable'}")
        # Multiplicity: this probe is normally run at BOTH configurations, so a nominal 0.05 is two
        # tests. Quoting an uncorrected p in a repo whose entire subject is multiplicity discipline
        # would be indefensible.
        if p_less is not None:
            bonf = 0.05 / 2.0
            if p_less < bonf:
                print(f"  => BELOW the published {claim:.2f}, and it SURVIVES Bonferroni over the "
                      f"2 configs (p {p_less:.4f} < {bonf:.3f}).")
                print("     The gate has LESS power than the forward R&D plan assumes.")
            elif p_less < 0.05:
                print(f"  => SUGGESTIVE that the true power is below {claim:.2f} (p {p_less:.4f}), "
                      f"but it does NOT survive")
                print(f"     Bonferroni over the 2 configs ({bonf:.3f}). Do not report this as a "
                      f"refutation on its own.")
            else:
                print(f"  => Not separable from the published {claim:.2f} at this seed count.")
        if lo <= 0.05:
            print("  ⚠ The interval's lower bound also reaches ~alpha, so 'the gate has real power'")
            print("    and 'the gate fires at its false-positive rate' are both live readings here.")
    print("=" * 100 + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gates", default=DEFAULT_GATES)
    p.add_argument("--cohort-gates", default=DEFAULT_COHORT_GATES)
    p.add_argument("--config", choices=["measured", "shipped"], default="measured",
                   help="'measured' = K<=20/rho<=0.10, the configuration the published table was "
                        "measured at (tests the CLAIM). 'shipped' = the gates file's own values "
                        "(tests what PRODUCTION gets).")
    p.add_argument("--bars", type=int, default=2520, help="panel length (default 2520 ~ 10y)")
    p.add_argument("--pool-size", type=int, default=40)
    p.add_argument("--n-real", type=int, default=6, help="planted real signals (default 6 => 15%%)")
    p.add_argument("--ir", type=float, default=0.30, help="annualized Sharpe of each planted signal")
    p.add_argument("--base-sharpe", type=float, default=0.50)
    p.add_argument("--n-reps", type=int, default=49,
                   help="bootstrap replicates B (default 49; the doc records B=49 and B=1000 giving "
                        "IDENTICAL pass rates)")
    p.add_argument("--seeds", type=int, default=24, help="independent trials per cell")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--skip-null", action="store_true",
                   help="skip the null calibration cell (NOT recommended — it is what makes the "
                        "power number readable)")
    p.add_argument("--json", type=str, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")   # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    fcfg, _ = load_generation_config(_REPO_ROOT / args.gates)
    ccfg_shipped, mc_kwargs = load_cohort_config(_REPO_ROOT / args.gates,
                                                 _REPO_ROOT / args.cohort_gates)
    assert isinstance(ccfg_shipped, CohortConfig)
    alpha = float(mc_kwargs.get("alpha_cohort", 0.05))

    ccfg = (ccfg_shipped if args.config == "shipped"
            else replace(ccfg_shipped, max_cohort_size=MEASURED_K,
                         max_pairwise_corr=MEASURED_RHO))

    cells = []
    if not args.skip_null:
        cells.append(run_cell("NULL (0 real)", args=args, ccfg=ccfg, fcfg=fcfg,
                              n_real=0, ir=0.0, alpha=alpha))
    cells.append(run_cell(f"{args.n_real}/{args.pool_size} real, IR {args.ir:.2f}",
                          args=args, ccfg=ccfg, fcfg=fcfg,
                          n_real=args.n_real, ir=args.ir, alpha=alpha))

    out = {
        "config": {
            "name": args.config, "max_cohort_size": ccfg.max_cohort_size,
            "max_pairwise_corr": ccfg.max_pairwise_corr, "min_cohort_size": ccfg.min_cohort_size,
            "n_reps": args.n_reps, "alpha_cohort": alpha, "seeds": args.seeds,
            "bars": args.bars, "pool_size": args.pool_size, "base_sharpe": args.base_sharpe,
            "gates": args.gates, "cohort_gates": args.cohort_gates,
        },
        "published_reference": PUBLISHED,
        "cells": cells,
    }
    _print(out)
    if args.json:
        dest = Path(args.json)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
