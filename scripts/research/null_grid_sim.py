"""Matched-null search + planted-power grid — re-derivation of independent-audit finding F4.

REBUILD of ``scratchpad/phase2/null_grid_sim.py``, cited by
``docs/research/crucible_independent_audit_report_2026-07-14.md`` (F4, report line 151) and never
committed. F4 is the finding that carries the audit's headline verdict, so with the project heading
for open-source release it needs to be re-derivable rather than taken on trust — especially since
the one other scratchpad experiment that could be rechecked (``planted_sweep.py``, F1/F2) did NOT
reproduce on the full real substrate (see ``planted_oracle_reproduction_2026-09-02.md``).

THE CLAIM BEING RE-DERIVED
--------------------------
    F4: "The record is uninformative and (A) unclaimable: matched-null searches at the real budget
    yield HoF dSR 0.80-0.94, t <= 2.93 (pure noise) >= real record maxima; joint gate power ~ 7e-4
    at true dSR 0.5; E[discoveries | all 233 charged were true 0.5-edges] ~ 0.16-0.4. Bounded-(A)
    only at dSR >= ~1.2 in the narrow expressible class."

and from the audit's verdict section, the comparison it turns on:

    "Matched-null searches (shipped `evolve` + gate, real budget pop 200 x gens 40, base book
    rescaled to the measured real train Sharpe) produce hall-of-fame dSR 0.80-0.94 and marginal_t
    up to 2.93 under pure noise. The real record's maxima (dSR 0.99 / 0.71 / 0.66; t <= 2.12) sit
    at or below the noise ceiling."

Four separable sub-claims, and this script measures each:

  (a) NULL CEILING   -- run the shipped search at the real budget on a panel with NO signal; record
                        the hall-of-fame's best dSR and marginal_t. If those match or exceed the
                        real record's best, the record carries no evidence of signal.
  (b) GATE POWER     -- plant candidates of known strength and measure P(the full 6-leg gate fires).
  (c) E[DISCOVERIES] -- (b) x the 233 hypotheses the project actually charged.
  (d) BOUNDED-(A)    -- the effect size at which power becomes non-trivial, i.e. the ONLY size the
                        record can speak about.

WHY THE NULL IS A PERMUTATION, NOT SYNTHETIC NOISE
---------------------------------------------------
"Matched" is doing real work here. The null must differ from the real substrate in EXACTLY one
respect — the presence of predictable structure — or the comparison measures the wrong thing (the
lesson from the synthetic arm of ``planted_sweep``, where a random-walk book turned out far more
timeable than the real one and inverted the verdict).

So the null is built by permuting the real panel in RETURN space: per-bar cross-sectional return
vectors are kept intact and reordered in time, and each bar's open/high/low/volume are carried along
as ratios to its own close. That preserves the cross-sectional covariance, the per-name marginal
return distribution, the OHLC geometry, and the panel's calendar — and destroys every temporal
relationship a formula could exploit. Any hall-of-fame result on it is selection, by construction.

NOTHING IS RE-IMPLEMENTED. The search is ``signals.generation.evolve.evolve`` at the gates YAML's
own budget; the gate is ``combination_fitness``; the seeds are the production ``_CS_SEED_BANK``.

USAGE
-----
    python scripts/research/null_grid_sim.py --mode null  --replicates 5
    python scripts/research/null_grid_sim.py --mode power --replicates 40
    python scripts/research/null_grid_sim.py --substrate taiwan --mode both --json out.json

``--substrate taiwan`` needs FINMIND_TOKEN (the panel is cached after the first fetch); the
synthetic substrate needs nothing and is the default.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sharpen.crucible.agentic.proposer import _CS_SEED_BANK                 # noqa: E402
from sharpen.signals.eval_harness import _ann_sharpe                       # noqa: E402
from sharpen.signals.features import Panel, make_synthetic_panel           # noqa: E402
from sharpen.signals.generation.base_sleeves import tsmom_sleeve_returns   # noqa: E402
from sharpen.signals.generation.config import load_generation_config       # noqa: E402
from sharpen.signals.generation.evolve import evolve                       # noqa: E402
from sharpen.signals.generation.fitness import (                           # noqa: E402
    FitnessConfig,
    combination_fitness,
)
from sharpen.signals.library._alpha_formulas import FORMULAS               # noqa: E402
from sharpen.signals.library.alphas101 import SKIP                         # noqa: E402

log = logging.getLogger("null_grid_sim")

DEFAULT_GATES = "configs/taiwan_signal_eval.gates.yaml"

# The audit's published comparison points, so a run prints its own verdict against them rather than
# leaving the reader to fetch the report. Report line 37 / line 151.
AUDIT_NULL_HOF_DSR = (0.80, 0.94)
AUDIT_NULL_MAX_T = 2.93
AUDIT_REAL_RECORD_DSR = (0.99, 0.71, 0.66)
AUDIT_REAL_RECORD_MAX_T = 2.12
AUDIT_POWER_AT_DSR_050 = 7e-4
AUDIT_E_DISCOVERIES = (0.16, 0.40)
AUDIT_CHARGED_HYPOTHESES = 233
AUDIT_BOUNDED_A_DSR = 1.2


def panel_ts(panel: Panel) -> np.ndarray:
    """Epoch-seconds float64 — the combiner's expected unit (``crucible_orchestrator._panel_ts``)."""
    return panel.dates.astype("datetime64[s]").astype(np.int64).astype(np.float64)


def seed_formulas() -> list[str]:
    """The production cross-sectional seed bank — the same 8 WQ101 formulas the miner starts from."""
    return [FORMULAS[i] for i, _, _, _ in _CS_SEED_BANK if i not in SKIP]


# --------------------------------------------------------------------------------------------
# The matched null
# --------------------------------------------------------------------------------------------
def permute_panel(panel: Panel, rng: np.random.Generator) -> Panel:
    """A matched null: the same panel with its TIME ORDER destroyed and everything else preserved.

    Works in return space so the result is a realistic price series rather than a permutation of
    price LEVELS (which would inject enormous artificial jumps and change what a formula sees):

      1. per-bar cross-sectional simple returns ``r[t] = close[t]/close[t-1] - 1``;
      2. permute the ROWS of ``r`` — the cross-sectional vector at each bar stays intact, so the
         panel's contemporaneous covariance is untouched, while every lead/lag relation dies;
      3. rebuild closes by compounding the permuted returns from the real first bar;
      4. carry each permuted bar's own open/high/low as RATIOS to its close, so OHLC stays
         internally consistent (an OHLC violation would be culled upstream and silently thin the
         search), and permute volume/adv with the same index.

    What survives: cross-sectional covariance, per-name marginal return distribution, OHLC
    geometry, the calendar, the active mask, sector ids. What dies: momentum, reversal,
    autocorrelation, and every other temporal structure a DSL formula could express.
    """
    close = np.asarray(panel.close, dtype=np.float64)
    T, N = close.shape
    if T < 3:
        raise ValueError("panel too short to permute")
    with np.errstate(divide="ignore", invalid="ignore"):
        r = close[1:] / close[:-1] - 1.0                       # (T-1, N)
        ratio_o = np.asarray(panel.open, dtype=np.float64) / close
        ratio_h = np.asarray(panel.high, dtype=np.float64) / close
        ratio_l = np.asarray(panel.low, dtype=np.float64) / close

    # STAGGERED LISTING. The panel's names do NOT all start at bar 0 — on the Taiwan cross-section
    # only 3 of 10 ETFs exist at 2010-01-04, the rest list as late as 2021. An earlier version of
    # this function rebuilt every name as `close[0] * cumprod(1+r)`, which propagates the bar-0 NaN
    # of every late lister across its ENTIRE series: the null panel came out with 3 usable names
    # and ZERO bars reaching `ls_min_names`, so no candidate could form a book and every null
    # ceiling measured on it was an artifact. Each name is therefore anchored at its OWN first
    # finite close and compounded only across its OWN live window.
    #
    # Cross-sectional co-movement still survives because ONE global permutation drives every name:
    # for name j the live indices are reordered by the restriction of that permutation to j's live
    # set, so names live at the same bars receive the SAME relative reordering.
    perm = rng.permutation(T - 1)
    rank = np.empty(T - 1, dtype=np.int64)
    rank[perm] = np.arange(T - 1)                              # position of each bar under perm

    new_close = np.full_like(close, np.nan)
    src = np.zeros((T, N), dtype=np.int64)                     # which real bar each cell copies OHLC from
    for j in range(N):
        live = np.flatnonzero(np.isfinite(close[:, j]))
        if live.size == 0:
            continue
        f = int(live[0])
        new_close[f, j] = close[f, j]
        ridx = np.flatnonzero(np.isfinite(r[:, j]))             # return rows this name actually has
        if ridx.size == 0:
            continue
        order = ridx[np.argsort(rank[ridx])]                    # induced order from the global perm
        rr = r[order, j]
        pos = ridx + 1                                          # the close rows those returns fill
        new_close[pos, j] = close[f, j] * np.cumprod(1.0 + rr)
        src[pos, j] = order + 1
        src[f, j] = f

    def _pick(ratio: np.ndarray) -> np.ndarray:
        out = np.take_along_axis(ratio, src, axis=0)
        return np.where(np.isfinite(out), out, 1.0)

    new_open = new_close * _pick(ratio_o)
    new_high = np.maximum(new_close * _pick(ratio_h), np.maximum(new_open, new_close))
    new_low = np.minimum(new_close * _pick(ratio_l), np.minimum(new_open, new_close))

    from dataclasses import replace
    # Liveness stays as it really was: a name is tradeable exactly where it has a price. Carrying
    # the ORIGINAL active mask (rather than a permuted one) keeps the panel's listing structure
    # contiguous and realistic, which is what `ls_min_names` and the cull paths react to.
    return replace(panel, open=new_open, high=new_high, low=new_low, close=new_close,
                   volume=np.take_along_axis(np.asarray(panel.volume), src, axis=0),
                   adv_usd=np.take_along_axis(np.asarray(panel.adv_usd), src, axis=0),
                   active=np.asarray(panel.active) & np.isfinite(new_close),
                   feature_slots={})


def retarget_sharpe(rets: np.ndarray, target: float, ppy: float) -> np.ndarray:
    """Location-shift a return stream to a target annualized Sharpe (vol and path shape untouched).

    The audit rescales the null's base book "to the measured real train Sharpe" — the base book's
    own level matters because ``dsr_aug`` deflates the WHOLE augmented book and the uplift leg is
    measured against it (audit F2/S-2), so a null run against a book at a different Sharpe would
    not be matched where it counts.
    """
    r = np.asarray(rets, dtype=np.float64)
    m = r[np.isfinite(r)]
    if m.size < 2:
        return r
    sd = float(m.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0.0:
        return r
    return r + (target * sd / float(np.sqrt(ppy)) - float(m.mean()))


# --------------------------------------------------------------------------------------------
# Substrates
# --------------------------------------------------------------------------------------------
def build_substrate(args: argparse.Namespace, ev: dict, ppy: float
                    ) -> tuple[Panel, dict[str, np.ndarray]]:
    """(panel, base_returns) for the requested substrate, base rescaled to ``--base-sharpe``."""
    hold, cost = int(ev["hold_horizon"]), float(ev["cost_bps"])
    if args.substrate == "taiwan":
        import os
        if not os.environ.get("FINMIND_TOKEN"):
            raise SystemExit(
                "FINMIND_TOKEN is not set — the Taiwan substrate fetches from FinMind.\n"
                "  export FINMIND_TOKEN=...   (never pass a token on the command line)\n"
                "  Or use the default --substrate synthetic, which needs no credentials.")
        from sharpen.data.taiwan_panel_loader import load_taiwan_panel  # noqa: PLC0415
        from sharpen.signals.generation.base_sleeves import taiwan_base_sleeves  # noqa: PLC0415
        panel = load_taiwan_panel(args.start, args.end)
        base, _ = taiwan_base_sleeves(panel, hold_horizon=hold, cost_bps=cost,
                                      start=args.start, end=args.end, return_components=True)
    else:
        panel = make_synthetic_panel(T=args.bars, N=args.names, seed=args.seed)
        from dataclasses import replace  # noqa: PLC0415
        idx = np.arange(min(3, panel.N))
        sub = replace(panel, tickers=tuple(panel.tickers[j] for j in idx),
                      open=panel.open[:, idx], high=panel.high[:, idx], low=panel.low[:, idx],
                      close=panel.close[:, idx], volume=panel.volume[:, idx],
                      active=panel.active[:, idx], adv_usd=panel.adv_usd[:, idx],
                      sector_id=panel.sector_id[idx])
        base = {"tsmom": np.asarray(tsmom_sleeve_returns(sub, hold_horizon=hold, cost_bps=cost))}
    if args.max_bars is not None and args.max_bars < panel.T:
        # Panel LENGTH is a first-class axis here, not a convenience. A best-of-N search on fewer
        # bars overfits further: the hall-of-fame maximum any search reaches under pure noise grows
        # as the sample shrinks. The companion finding for F1/F2 turned out to be exactly this
        # (planted_oracle_reproduction_2026-09-02.md), so an F4 gap must be tested against it
        # before any other explanation is entertained.
        n = int(args.max_bars)
        panel = panel.truncated(n - 1)
        base = {k: np.asarray(v)[:n] for k, v in base.items()}
    if args.base_sharpe is not None:
        base = {k: retarget_sharpe(v, args.base_sharpe, ppy) for k, v in base.items()}
    return panel, base


# --------------------------------------------------------------------------------------------
# (a) The null ceiling
# --------------------------------------------------------------------------------------------
def run_null_search(panel: Panel, base: dict, fit: FitnessConfig, ev: dict,
                    args: argparse.Namespace) -> list[dict]:
    """Run the SHIPPED search at the real budget on permuted (signal-free) panels.

    Each replicate reports the hall-of-fame's best ``delta_sr_oos`` and ``marginal_t`` — the
    ceiling a best-of-N search reaches on pure noise. That ceiling is the yardstick the real
    record's maxima must clear to carry any information.
    """
    # KNOWN SIMPLIFICATION, stated because it moves the result in a knowable direction. On the
    # Taiwan substrate the REAL base book is TX/TE/TF futures — a universe DISJOINT from the 10-ETF
    # candidate cross-section. Here the null's base book is rebuilt from the first three names of
    # the permuted panel, so base and candidates share three names. Any resulting collinearity can
    # only make the `not_redundant` leg bite MORE often, which LOWERS the null ceiling. That is the
    # conservative direction for the claim under test: it makes F4 harder to confirm, never easier.
    ppy = fit.periods_per_year
    rows: list[dict] = []
    seeds = seed_formulas()
    for rep in range(args.replicates):
        rng = np.random.default_rng(args.seed + 1000 * rep)
        null_panel = permute_panel(panel, rng)
        # The base book is recomputed ON the null panel, then rescaled: its own Sharpe is a
        # matched nuisance parameter (audit F2), not something the null should accidentally change.
        hold, cost = int(ev["hold_horizon"]), float(ev["cost_bps"])
        from dataclasses import replace  # noqa: PLC0415
        idx = np.arange(min(3, null_panel.N))
        nsub = replace(null_panel, tickers=tuple(null_panel.tickers[j] for j in idx),
                       open=null_panel.open[:, idx], high=null_panel.high[:, idx],
                       low=null_panel.low[:, idx], close=null_panel.close[:, idx],
                       volume=null_panel.volume[:, idx], active=null_panel.active[:, idx],
                       adv_usd=null_panel.adv_usd[:, idx], sector_id=null_panel.sector_id[idx])
        nbase = {"tsmom": np.asarray(tsmom_sleeve_returns(nsub, hold_horizon=hold, cost_bps=cost))}
        tgt = args.base_sharpe if args.base_sharpe is not None else _measured_base_sharpe(base, ppy)
        nbase = {k: retarget_sharpe(v, tgt, ppy) for k, v in nbase.items()}

        t0 = time.time()
        rep_out = evolve(
            seeds, null_panel, nbase, panel_ts(null_panel), fit,
            rng_seed=int(ev["rng_seed"]) + rep, pop_size=args.pop_size,
            n_generations=args.n_generations, hold_horizon=hold, cost_bps=cost,
            ls_min_names=int(ev["ls_min_names"]), holdout_frac=float(ev["holdout_frac"]),
            holdout_embargo=int(ev["holdout_embargo"]), elite_frac=float(ev["elite_frac"]))
        dt = time.time() - t0

        scored = [c for c in rep_out.hall_of_fame if c.result is not None]
        best_dsr = max((c.result.delta_sr_oos for c in scored
                        if np.isfinite(c.result.delta_sr_oos)), default=float("nan"))
        best_t = max((c.result.marginal_t for c in scored
                      if np.isfinite(c.result.marginal_t)), default=float("nan"))
        row = {
            "replicate": rep,
            "seconds": dt,
            "gen_n_total": rep_out.gen_n_total,
            "hof_size": len(rep_out.hall_of_fame),
            "best_delta_sr_oos": float(best_dsr),
            "best_marginal_t": float(best_t),
            "n_promising": len(rep_out.promising),
            "n_holdout_tested": rep_out.n_holdout_tested,
            "null_base_ann_sharpe": _measured_base_sharpe(nbase, ppy),
        }
        # VACUITY GUARD. A cross-sectional book needs `ls_min_names` active names; the Taiwan panel
        # does not reach 6 until bar 1405 (2015-09-07) because the ETFs list in stages. Run the
        # search inside that dead zone and EVERY candidate books a flat return, so the hall-of-fame
        # "best dSR" is exactly 0.000 — a number that looks like a measured ceiling and is actually
        # "nothing was tested". Measured here at --max-bars 500 and 1000: dSR 0.000 across all
        # replicates. This is the same shape as the audit's own vacuous `promising=0` (F4/S-4), so
        # it is flagged on the row rather than left to be read as evidence.
        n_tradeable = int((np.isfinite(np.asarray(null_panel.close))
                           & np.asarray(null_panel.active)).sum(axis=1).max(initial=0))
        row["max_active_names"] = n_tradeable
        row["vacuous"] = bool(row["best_delta_sr_oos"] == 0.0
                              or n_tradeable < int(ev["ls_min_names"]))
        if row["vacuous"]:
            log.warning("null rep %d is VACUOUS: max active names %d (need %d), best dSR %.3f — "
                        "no candidate could form a book; this is NOT a measured ceiling",
                        rep, n_tradeable, int(ev["ls_min_names"]), row["best_delta_sr_oos"])
        rows.append(row)
        log.info("null rep %d/%d: best dSR %.3f  best t %.3f  N=%d  (%.0fs)",
                 rep + 1, args.replicates, row["best_delta_sr_oos"], row["best_marginal_t"],
                 row["gen_n_total"], dt)
    return rows


def _measured_base_sharpe(base: dict, ppy: float) -> float:
    v = np.concatenate([np.asarray(x, dtype=np.float64) for x in base.values()])
    v = v[np.isfinite(v)]
    return _ann_sharpe(v, ppy) if v.size > 1 else float("nan")


# --------------------------------------------------------------------------------------------
# (b)-(d) Planted power
# --------------------------------------------------------------------------------------------
def run_power_grid(panel: Panel, base: dict, fit: FitnessConfig, args: argparse.Namespace
                   ) -> list[dict]:
    """P(the 6-leg gate fires) against a candidate of KNOWN strength — F4's power leg.

    The candidate is a stream orthogonal to the base book (independent noise) with a drift set so
    its own annualized Sharpe equals the planted ``delta``. Both the planted delta and the REALIZED
    CPCV ``delta_sr_oos`` are reported, so the mapping between "an alpha this good" and "the uplift
    the gate actually sees" is explicit rather than assumed.

    Scoring uses the SHIPPED ``combination_fitness`` at the real file-drawer count, which is what
    makes this a JOINT-gate power (all six legs) rather than a single-leg t-test.
    """
    ppy = fit.periods_per_year
    ts = panel_ts(panel)
    b0 = np.asarray(next(iter(base.values())), dtype=np.float64)
    vol = float(b0[np.isfinite(b0)].std(ddof=1))
    rows: list[dict] = []
    for delta in args.deltas:
        n_pass = 0
        realized: list[float] = []
        tstats: list[float] = []
        for rep in range(args.replicates):
            rng = np.random.default_rng(args.seed + 7919 * rep + int(delta * 1e6))
            noise = rng.standard_normal(panel.T) * vol
            cand = noise - noise.mean() + delta * vol / float(np.sqrt(ppy))
            res = combination_fitness(cand, base, ts, fit, gen_n_eff=args.gen_n_eff,
                                      turnover_ann=1.0, n_nodes=1, trial_sharpe_pool=None)
            n_pass += int(res.passes_gate)
            if np.isfinite(res.delta_sr_oos):
                realized.append(float(res.delta_sr_oos))
            if np.isfinite(res.marginal_t):
                tstats.append(float(res.marginal_t))
        power = n_pass / max(1, args.replicates)
        rows.append({
            "planted_ann_sharpe": float(delta),
            "power": power,
            "n_pass": n_pass,
            "n_replicates": args.replicates,
            "median_realized_delta_sr": float(np.median(realized)) if realized else float("nan"),
            "median_marginal_t": float(np.median(tstats)) if tstats else float("nan"),
            "expected_discoveries_at_233": power * AUDIT_CHARGED_HYPOTHESES,
        })
        log.info("power at planted SR %.2f: %d/%d = %.4f (realized dSR med %.3f)",
                 delta, n_pass, args.replicates, power, rows[-1]["median_realized_delta_sr"])
    return rows


# --------------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------------
def _print_null(rows: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("(a) NULL CEILING — what the SHIPPED search reaches on a signal-free panel")
    print("=" * 100)
    print(f"  {'rep':>4} {'gen_N':>7} {'best dSR':>10} {'best marg_t':>12} {'promising':>10} "
          f"{'secs':>7}  note")
    for r in rows:
        print(f"  {r['replicate']:>4} {r['gen_n_total']:>7} {r['best_delta_sr_oos']:>10.3f} "
              f"{r['best_marginal_t']:>12.3f} {r['n_promising']:>10} {r['seconds']:>7.0f}"
              f"  {'VACUOUS (no tradeable cross-section)' if r.get('vacuous') else ''}")
    live = [r for r in rows if not r.get("vacuous")]
    if not live:
        print("\n  EVERY replicate is VACUOUS — no candidate could form a book on this panel")
        print("  (fewer than ls_min_names active names). This run measures NOTHING; it is NOT")
        print("  a null ceiling of zero. Widen the window or use a panel with a live")
        print("  cross-section before reading any verdict off it.")
        return
    rows = live
    dsr = [r["best_delta_sr_oos"] for r in rows if np.isfinite(r["best_delta_sr_oos"])]
    tt = [r["best_marginal_t"] for r in rows if np.isfinite(r["best_marginal_t"])]
    if not dsr:
        print("\n  no finite results")
        return
    print(f"\n  measured null ceiling : dSR {min(dsr):.3f}-{max(dsr):.3f}   "
          f"marginal_t max {max(tt):.3f}")
    print(f"  audit F4 reported     : dSR {AUDIT_NULL_HOF_DSR[0]:.2f}-{AUDIT_NULL_HOF_DSR[1]:.2f}"
          f"   marginal_t max {AUDIT_NULL_MAX_T:.2f}")
    print(f"  real record maxima    : dSR {'/'.join(f'{x:.2f}' for x in AUDIT_REAL_RECORD_DSR)}"
          f"   marginal_t <= {AUDIT_REAL_RECORD_MAX_T:.2f}")
    # PER-AXIS verdict. An earlier version required the noise maximum to exceed the LARGEST record
    # dSR (0.99) on top of the t axis, and called anything less a refutation. That bar is stricter
    # than F4's own claim: the audit's published null ceiling tops out at 0.94, which does not
    # exceed 0.99 either. F4 asserts the record's maxima sit INSIDE the noise distribution, not
    # strictly under its maximum — so the axes are reported separately and `marginal_t` carries the
    # verdict, because it is the binding gate leg (the one that decides PROMISING).
    t_covers = max(tt) >= AUDIT_REAL_RECORD_MAX_T
    n_dsr_covered = sum(1 for x in AUDIT_REAL_RECORD_DSR if max(dsr) >= x)
    print(f"\n  marginal_t (BINDING leg): noise {max(tt):.3f} vs record {AUDIT_REAL_RECORD_MAX_T:.2f}"
          f"  -> noise {'EXCEEDS' if t_covers else 'below'} the record's best")
    print(f"  delta_SR                : noise {max(dsr):.3f} covers "
          f"{n_dsr_covered}/{len(AUDIT_REAL_RECORD_DSR)} of the record's values "
          f"({', '.join(f'{x:.2f}' for x in AUDIT_REAL_RECORD_DSR)})")
    if t_covers:
        print("\n  => F4's core inference REPRODUCES. A search of this size, on data containing no")
        print("     signal at all, reaches a HIGHER marginal_t than anything the real record ever")
        print("     produced — so the record's best results carry no evidence of signal.")
    else:
        print("\n  => F4's core inference does NOT reproduce at this configuration: noise stayed")
        print("     below the record's best marginal_t. Re-check before citing F4.")


def _print_power(rows: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("(b-d) JOINT-GATE POWER — P(all six legs fire) against a planted edge")
    print("=" * 100)
    print(f"  {'planted SR':>11} {'realized dSR':>13} {'med marg_t':>11} {'power':>9} "
          f"{'E[disc]@233':>12}")
    for r in rows:
        print(f"  {r['planted_ann_sharpe']:>11.2f} {r['median_realized_delta_sr']:>13.3f} "
              f"{r['median_marginal_t']:>11.3f} {r['power']:>9.4f} "
              f"{r['expected_discoveries_at_233']:>12.3f}")
    near = min(rows, key=lambda r: abs(r["planted_ann_sharpe"] - 0.5), default=None)
    if near is not None:
        print(f"\n  at planted SR 0.50: power {near['power']:.4f} "
              f"(audit F4: ~{AUDIT_POWER_AT_DSR_050:.0e}), "
              f"E[discoveries|233] {near['expected_discoveries_at_233']:.3f} "
              f"(audit: {AUDIT_E_DISCOVERIES[0]}-{AUDIT_E_DISCOVERIES[1]})")
    reach = [r for r in rows if r["power"] >= 0.5]
    if reach:
        print(f"  power first reaches 50% at planted SR {min(r['planted_ann_sharpe'] for r in reach):.2f}"
              f"   (audit's bounded-(A) threshold: dSR >= ~{AUDIT_BOUNDED_A_DSR})")
    else:
        print(f"  power never reaches 50% anywhere in the swept grid "
              f"(audit's bounded-(A) threshold: dSR >= ~{AUDIT_BOUNDED_A_DSR})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gates", default=DEFAULT_GATES)
    p.add_argument("--mode", choices=["null", "power", "both"], default="both")
    p.add_argument("--substrate", choices=["synthetic", "taiwan"], default="synthetic")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--bars", type=int, default=1200)
    p.add_argument("--names", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--replicates", type=int, default=5,
                   help="null: independent permuted panels. power: draws per planted level.")
    p.add_argument("--pop-size", type=int, default=None,
                   help="search population (default: the gates YAML's pop_size — the real budget)")
    p.add_argument("--n-generations", type=int, default=None,
                   help="search generations (default: the gates YAML's n_generations)")
    p.add_argument("--max-bars", type=int, default=None,
                   help="truncate the panel to its first N bars. Panel length is the axis that "
                        "explained the F1/F2 discrepancy, so it is exposed here too: a best-of-N "
                        "search on fewer bars reaches a HIGHER hall-of-fame maximum under pure "
                        "noise")
    p.add_argument("--base-sharpe", type=float, default=None,
                   help="rescale the base book to this annualized Sharpe (default: as measured)")
    p.add_argument("--deltas", type=float, nargs="+",
                   default=[0.25, 0.50, 0.75, 1.00, 1.20, 1.50, 2.00],
                   help="planted annualized Sharpes for the power grid")
    p.add_argument("--gen-n-eff", type=float, default=233.0,
                   help="file-drawer trial count for deflation (default 233 = the hypotheses the "
                        "project actually charged, so power is measured at the REAL multiplicity)")
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

    fit, ev = load_generation_config(_REPO_ROOT / args.gates)
    if args.pop_size is None:
        args.pop_size = int(ev["pop_size"])
    if args.n_generations is None:
        args.n_generations = int(ev["n_generations"])

    panel, base = build_substrate(args, ev, fit.periods_per_year)
    log.info("substrate %s: T=%d N=%d  base ann SR %.3f  budget pop=%d x gens=%d",
             args.substrate, panel.T, panel.N, _measured_base_sharpe(base, fit.periods_per_year),
             args.pop_size, args.n_generations)

    out: dict = {
        "substrate": {"kind": args.substrate, "bars": panel.T, "names": panel.N,
                      "gates": args.gates, "seed": args.seed,
                      "base_ann_sharpe": _measured_base_sharpe(base, fit.periods_per_year),
                      "pop_size": args.pop_size, "n_generations": args.n_generations},
        "audit_reference": {
            "null_hof_dsr": AUDIT_NULL_HOF_DSR, "null_max_t": AUDIT_NULL_MAX_T,
            "real_record_dsr": AUDIT_REAL_RECORD_DSR, "real_record_max_t": AUDIT_REAL_RECORD_MAX_T,
            "power_at_dsr_050": AUDIT_POWER_AT_DSR_050,
            "e_discoveries": AUDIT_E_DISCOVERIES, "bounded_a_dsr": AUDIT_BOUNDED_A_DSR,
            "charged_hypotheses": AUDIT_CHARGED_HYPOTHESES},
    }
    if args.mode in ("null", "both"):
        out["null"] = run_null_search(panel, base, fit, ev, args)
        _print_null(out["null"])
    if args.mode in ("power", "both"):
        out["power"] = run_power_grid(panel, base, fit, args)
        _print_power(out["power"])
    print("\n" + "=" * 100 + "\n")

    if args.json:
        dest = Path(args.json)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
