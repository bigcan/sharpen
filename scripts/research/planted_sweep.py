"""Planted-oracle sweep — where does the funnel's PROMISING bar sit, and can foresight clear it?

REBUILD of the script cited by the independent external audit
(``docs/research/crucible_independent_audit_report_2026-07-14.md``, findings F1/F2, verdict §1)
as ``scratchpad/planted_sweep.py``. The original lived in the auditor's scratchpad and was never
committed, so the audit's most load-bearing experiment — the one deciding whether ``0 PROMISING``
is a fact about markets or a fact about the machine — was not reproducible from this repo.

THE AUDIT'S CLAIM
-----------------
    "a planted perfect weekly-hold timing oracle — holdout annualized Sharpe 1.80 — fails 0/5 at
    every skill level from p=0.52 to p=1.00. Only a daily perfect-foresight oracle (SR 12.6)
    passes."  … "t>=3 on ~1011 holdout bars requires the marginal stream to carry annualized
    Sharpe ~= 1.50; the perfect weekly oracle reached 2.92 on train — still below 3.0."

WHAT THIS SCRIPT REPRODUCES, AND WHAT IT DOES NOT
--------------------------------------------------
Read this before quoting any number it prints, and before citing audit F1/F2 anywhere.

REPRODUCED EXACTLY — the power arithmetic (no data needed, ``--bar-only``):

  * ``marginal_t = SR_pp(marg) * sqrt(N_eff)``, so ``hlz_t_min`` 3.0 on an N-bar holdout demands an
    annualized marginal Sharpe of ``3 * sqrt(252/N)``. At the measured 1022-bar Taiwan holdout that
    is **1.490**; the audit says "~= 1.50".
  * Audit F7's power wall, to two decimals: ideal single-prereg t>=2 MDE80 is **1.419** at T=1011
    (audit: 1.42) and **0.709** at T=4044 (audit: 0.71).

REPRODUCED — the substrate itself (``--substrate taiwan``, measured 2026-09-02):

  * Panel 4064 bars, holdout 1022 (audit: "~1011").
  * Taiwan base book annualized Sharpe ~0 on train (measured −0.155; audit −0.007) and strongly
    positive on holdout (measured **+1.333**; audit **+1.364**). The era-dependence F2 describes is
    real and reproduces closely.

**NOT REPRODUCED — the headline oracle result.** The audit reports the perfect weekly oracle failing
**0/5** with ``marginal_t`` **2.92** on train. Measured here on the same substrate, same shipped
funnel, same gates, with the candidate scored at the most GENEROUS settings:

  ===========================  ==========  ============  =============  ==========
  window                       train bars  train marg_t  holdout marg_t verdict
  ===========================  ==========  ============  =============  ==========
  2010-01-01 .. latest              3042        8.608          3.800    SEAL_ABSENT
  2010-01-01 .. 2022-12-31          2379        8.048          4.149    SEAL_ABSENT
  2018-01-01 .. 2020-12-31           529        2.590          1.533    SEAL_CONFIRMED
  ===========================  ==========  ============  =============  ==========

On the full panel and on the audit's own stated 2010->2022 era the perfect weekly oracle **PASSES**
the gate comfortably. It only fails on a much SHORTER window — and there the numbers land close to
the audit's published pair (train t **2.590** vs 2.92; marginal ann SR **1.788** vs its holdout
"1.80").

Two candidate explanations were tested and ELIMINATED: the oracle carrying forward magnitude vs
direction only (``--oracle-signal``; marg_t 3.800 vs 3.699 on the real holdout — no material
difference), and the base book's Sharpe regime (swept; moves t by <0.7).

THE EXPLANATION, MEASURED (``--length-sweep``). The disagreement is not about the substrate, the
oracle, or the code — it is about SAMPLE SIZE, and the sweep quantifies it. ``marginal_t`` grows as
``sqrt(N)`` while the bar it must clear is a CONSTANT 3.0, so "does perfect foresight pass?" has no
substrate-free answer. On the real Taiwan panel, planting the perfect weekly oracle at increasing
window lengths:

  ======  =======  ========  ========  ========  =====
    bars    years    margSR    req.SR    marg_t   PASS
  ======  =======  ========  ========  ========  =====
     380     1.51     1.932     2.443     2.372     no
     500     1.98     2.321     2.130     3.270    YES
    1000     3.97     3.034     1.506     5.928    YES
    2000     7.94     2.631     1.065     7.412    YES
    4000    15.87     2.362     0.753     9.410    YES
  ======  =======  ========  ========  ========  =====

**The oracle crosses the bar at N ~= 442-462 bars (1.75-1.83 years)** — stable across seeds and
across ``--oracle-signal magnitude|sign``. The achievable marginal Sharpe is roughly FLAT in N
(1.93-3.03, a property of the substrate); what moves is the REQUIRED Sharpe, falling 2.443 -> 0.753.
So the crossing is driven purely by how many bars are scored.

The Taiwan panel holds **4085** bars. The audit's 2.92 therefore corresponds to roughly **a tenth**
of the available data — it lands inside the narrow rejecting band below ~1.8 years. On anything
approaching the full panel, the gate does NOT reject perfect weekly foresight.

Why the original run might have been short is not established and its script was never committed,
so it cannot be inspected. One live mechanism was found and fixed during this rebuild:
``fetch_and_clean``/``fetch_and_clean_taiwan`` reused their cache on an asset superset plus
``_cache_covers_end`` — which tests the window END only. Nothing checked the requested START, and
with ``end=None`` the freshness leg returned True unconditionally, so a cache built from a narrower
window was served for a wider request while logging "using cached clean OHLCV"; the cache-hit path
also returned the whole cached range rather than the requested window. Measured before the fix: a
2018-2020 cache answered a 2010-01-01 request with **713 bars instead of 4085**, silently. Both
halves are now fixed in ``sharpen/data/`` (``_cache_covers_start`` + ``_clip_window``, mutation-
tested in ``tests/data/test_cross_asset_loader.py``), and ``build_taiwan_substrate`` additionally
REFUSES a short panel rather than mis-measure. Whether the audit hit this is UNKNOWN.

CONSEQUENCE FOR THE RECORD. The audit's verdict that hypothesis (A) — "the market has no alpha" — is
UNCLAIMABLE does NOT depend on the oracle experiment: it rests independently on F7's power wall,
which reproduces exactly. But the specific, widely-quoted sentence *"the machine rejects even a
perfect weekly-foresight oracle"* does not reproduce on the full real substrate and **should not be
published as-is**. Cite the power wall, not the oracle.

NOTHING IS RE-IMPLEMENTED. The gate is ``fitness.combination_fitness``; the candidate is booked by
``evolve._overlay_returns``; thresholds load from a gates YAML via ``config.load_generation_config``
(never hardcoded — CLAUDE.md). The oracle enters as a Panel ``feature_slot``, i.e. through the same
CR-9 terminal route a real alt-data overlay uses, so the shipped code cannot tell it apart from a
mined candidate. The only difference is that its series is computed from the future.

WHY AN OVERLAY AND NOT A CROSS-SECTIONAL BOOK
----------------------------------------------
The audit's oracle is a *timing* oracle, and the overlay path is the one that collapses a formula
to a per-day scalar timing the whole base book (``evolve._overlay_returns``, audit S-3). A
cross-sectional rank-L/S oracle is a different and far stronger object — on a 40-name synthetic
panel perfect weekly cross-sectional foresight reaches SR ~38 and sails through every leg. That
arm is available via ``--candidate cross_sectional`` but it is NOT the audit's experiment; using it
to argue about F1/F2 would be a category error.

TWO-DIRECTIONAL BY CONSTRUCTION
-------------------------------
This project has been burned by one-directional tests: a check that can never pass survives a test
that only looks for failure (``feedback_mutation_check_false_survivor_crlf``; the cohort
short-circuit a unit test asserted as a *requirement*). A sweep that only showed "the oracle fails"
could not distinguish a sealed gate from a broken harness. So every run carries both controls and
the verdict is REFUSED unless both behave:

  * POSITIVE — a daily perfect-foresight timing oracle. Must PASS, else the harness is miswired.
  * NEGATIVE — a pure-noise timing series. Must FAIL, else the gate is not discriminating.

EXIT CODES (so this is a tripwire, not just a demo)
----------------------------------------------------
  0  SEAL CONFIRMED — controls behaved; the perfect weekly oracle was REJECTED.
  1  SEAL ABSENT — controls behaved; the perfect weekly oracle PASSED. Expected on the synthetic
     default. On a real substrate this would mean F1/F2 no longer reproduces and every document
     citing it needs re-checking.
  2  HARNESS INVALID — a control misbehaved. The run says nothing about the gate.

USAGE
-----
    python scripts/research/planted_sweep.py --bar-only          # the arithmetic, instantly
    python scripts/research/planted_sweep.py                     # synthetic ladder + controls
    python scripts/research/planted_sweep.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sharpen.signals.eval_harness import _ann_sharpe, _ls_weights          # noqa: E402
from sharpen.signals.features import Panel, make_synthetic_panel           # noqa: E402
from sharpen.signals.generation.base_sleeves import tsmom_sleeve_returns   # noqa: E402
from sharpen.signals.generation.config import load_generation_config       # noqa: E402
from sharpen.signals.generation.evolve import (                            # noqa: E402
    _overlay_ctx,
    _overlay_returns,
    _slice_components,
    _split,
)
from sharpen.signals.generation.fitness import (                           # noqa: E402
    FitnessConfig,
    FitnessResult,
    _ar1_effective_n,
    _combined_book,
    combination_fitness,
)

log = logging.getLogger("planted_sweep")

DEFAULT_GATES = "configs/taiwan_signal_eval.gates.yaml"
ORACLE_SLOT = "oracle:timing"
# "0/5 at every skill level from p=0.52 to p=1.00" — p is the DIRECTIONAL HIT RATE, so 0.50 is a
# coin flip and 1.00 is omniscience. The audit reported five levels; these are they.
SKILL_LADDER: tuple[float, ...] = (0.52, 0.60, 0.75, 0.90, 1.00)


def panel_ts(panel: Panel) -> np.ndarray:
    """Timestamps in the combiner's units — epoch SECONDS as float64.

    Transcribed from ``crucible_orchestrator._panel_ts``: ``dynamic_sleeve_alphas`` reads these as
    ``pd.to_datetime(ts, unit="s")``, so passing raw ``datetime64[ns]`` overflows.
    """
    return panel.dates.astype("datetime64[s]").astype(np.int64).astype(np.float64)


# --------------------------------------------------------------------------------------------
# The bar — exact arithmetic, the part that reproduces without any data
# --------------------------------------------------------------------------------------------
def required_marginal_ann_sharpe(t_min: float, n_bars: int, ppy: float) -> float:
    """Annualized Sharpe the marginal stream must carry to clear the ``hlz_t_min`` leg.

    ``fitness.combination_fitness`` computes ``marginal_t = SR_pp(marg) * sqrt(_ar1_effective_n)``
    and requires ``>= cfg.hlz_t_min``. Inverting at the ideal (zero-autocorrelation) case where
    ``N_eff == N`` gives ``SR_ann = t_min * sqrt(ppy / N)``. Autocorrelation only shrinks
    ``N_eff``, so this is a LOWER BOUND on what the gate actually demands.
    """
    if n_bars < 2:
        raise ValueError("n_bars must be >= 2")
    return float(t_min) * float(np.sqrt(float(ppy) / float(n_bars)))


def mde80(t_min: float, n_bars: int, ppy: float) -> float:
    """Annualized ΔSR detectable at 80% power by an IDEAL single pre-registered t-test.

    ``(t_min + z_0.80) * sqrt(ppy / n_bars)`` — the significance threshold plus the one-sided 80%
    power term. This is the audit's finding F7 ("the residual honest power wall"), and it is the
    floor BELOW the deployed contract: it assumes one pre-registered hypothesis, no file-drawer
    deflation, and no autocorrelation. Reproduces F7's published values exactly at ``t_min=2``:
    T=4044 -> 0.709 (F7: 0.71), T=1011 -> 1.419 (F7: 1.42).
    """
    return (float(t_min) + 0.8416) * float(np.sqrt(float(ppy) / float(n_bars)))


def bar_table(fit: FitnessConfig, spans: tuple[int, ...], *, ideal_t: float = 2.0) -> list[dict]:
    """The bar across holdout lengths — the audit's 1011-bar Taiwan row is the anchor.

    Two columns, because the audit makes two separate points. ``required_marginal_ann_sharpe`` is
    what the SHIPPED gate demands (``hlz_t_min``, no power margin — finding F1). ``mde80_ideal`` is
    what an idealized single pre-registered test could detect at 80% power (finding F7) — the
    irreducible floor that survives even after every machine defect is removed.
    """
    return [{"n_bars": n,
             "years": n / fit.periods_per_year,
             "required_marginal_ann_sharpe": required_marginal_ann_sharpe(
                 fit.hlz_t_min, n, fit.periods_per_year),
             "mde80_ideal": mde80(ideal_t, n, fit.periods_per_year),
             "ideal_t": ideal_t}
            for n in spans]


# --------------------------------------------------------------------------------------------
# Planted candidates
# --------------------------------------------------------------------------------------------
def timing_series(base_book: np.ndarray, *, horizon: int, skill: float,
                  rng: np.random.Generator, signal: str = "magnitude") -> np.ndarray:
    """``(T,)`` weekly-HELD perfect-foresight timing signal on ``base_book``, at hit rate ``skill``.

    At each rebalance bar ``t0`` the signal is the base book's realized return over the coming
    ``horizon`` bars, held constant across the block — the audit's "weekly-HOLD timing oracle".
    The sign is kept with probability ``skill`` and flipped otherwise, so ``skill=1.0`` is perfect
    foresight and ``skill=0.5`` is a coin flip. Magnitude survives the flip, which makes the
    planted candidate STRONGER than a sign-only signal — deliberately, since the finding of
    interest is a rejection.

    This is a LOOK-AHEAD series by design. It exists only to be planted; it must never reach a
    research path that scores real candidates.
    """
    if not 0.0 <= skill <= 1.0:
        raise ValueError(f"skill must be in [0,1]; got {skill}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1; got {horizon}")
    if signal not in ("magnitude", "sign"):
        raise ValueError(f"signal must be 'magnitude' or 'sign'; got {signal!r}")
    bb = np.asarray(base_book, dtype=np.float64)
    T = bb.shape[0]
    cum = np.nancumsum(np.nan_to_num(bb, nan=0.0))
    g = np.full(T, np.nan, dtype=np.float64)
    for t0 in range(0, T - horizon, horizon):
        fwd = float(cum[t0 + horizon] - cum[t0])
        if signal == "sign":
            fwd = float(np.sign(fwd))
        g[t0:t0 + horizon] = fwd if rng.random() < skill else -fwd
    return g


def noise_series(n: int, *, rng: np.random.Generator) -> np.ndarray:
    """``(T,)`` pure-noise timing signal — the NEGATIVE control."""
    return rng.standard_normal(n)


def book_overlay(g: np.ndarray, panel: Panel, ctx: tuple, *, cost_bps: float
                 ) -> tuple[np.ndarray, float]:
    """Book a ``(T,)`` timing series through the UNMODIFIED shipped overlay path.

    The series is attached as a CR-9 feature slot, so ``_overlay_returns`` resolves it as an
    ordinary DSL terminal and applies the production formula
    ``cand[t] = m[t-1]*b_gross[t] - |m[t-1]|*c_base[t] - cost_bps*|dm[t]|*G[t]`` with
    ``m = tanh(causal expanding z-score)``.

    ``ctx`` is ``evolve._overlay_ctx(...)`` — the base book plus its F14 gross/cost/gross-exposure
    decomposition. On the REAL substrate those components are present, so the overlay pays the
    embedded base cost and its own rescaling at the book's TRUE gross (both monotone-stricter). On
    synthetic they are None, the unit-gross cost-free fallback ``_overlay_returns`` documents as
    exactly correct for planted sleeves and bit-identical to the pre-F14 formula.
    """
    base_book, base_gross, base_cost, gross_exp = ctx
    p2 = replace(panel, feature_slots={**panel.feature_slots, ORACLE_SLOT: g})
    out = _overlay_returns(ORACLE_SLOT, p2, base_book, cost_bps=cost_bps,
                           base_gross=base_gross, base_cost=base_cost, gross_exposure=gross_exp)
    if out is None:
        raise RuntimeError("overlay path culled the planted series (degenerate) — harness bug")
    return out


def book_cross_sectional(scores: np.ndarray, panel: Panel, *, hold_horizon: int, cost_bps: float,
                         min_names: int) -> tuple[np.ndarray, float]:
    """Book ``(T,N)`` scores as a rank-L/S sleeve — transcribes ``evolve._candidate_returns``.

    NOT the audit's experiment (see the module docstring); available for contrast only.
    """
    fwd1 = panel.forward_returns(1)
    rets = np.full(panel.T, np.nan)
    turns: list[float] = []
    w = np.zeros(panel.N)
    for t in range(panel.T - 1):
        rebal = t % hold_horizon == 0
        if rebal:
            w_new = _ls_weights(scores[t], panel.active[t], min_names=min_names)
            turns.append(float(np.abs(w_new - w).sum()))
            w = w_new
        rets[t] = float(np.nansum(w * fwd1[t]) - (cost_bps * turns[-1] if rebal and turns else 0.0))
    return rets, (float(np.mean(turns) * (252.0 / hold_horizon)) if turns else 0.0)


def cross_sectional_scores(panel: Panel, *, horizon: int, skill: float,
                           rng: np.random.Generator) -> np.ndarray:
    """``(T,N)`` cross-sectional foresight scores (``--candidate cross_sectional`` only)."""
    fwd = panel.forward_returns(horizon)
    return np.where(rng.random(fwd.shape) < skill, 1.0, -1.0) * fwd


# --------------------------------------------------------------------------------------------
# Base book
# --------------------------------------------------------------------------------------------
def _retarget_sharpe(rets: np.ndarray, target: float, ppy: float) -> np.ndarray:
    """Location-shift ``rets`` to a target annualized Sharpe (vol and path shape untouched)."""
    r = np.asarray(rets, dtype=np.float64)
    m = r[np.isfinite(r)]
    if m.size < 2:
        return r
    sd = float(m.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0.0:
        return r
    return r + (target * sd / float(np.sqrt(ppy)) - float(m.mean()))


def build_base_book(panel: Panel, *, n_sleeves: int, base_names: int, hold_horizon: int,
                    cost_bps: float, base_sharpe: float | None, ppy: float) -> dict[str, np.ndarray]:
    """Base sleeves via the shipped production TSMOM booking, on disjoint sub-universes.

    ``base_names`` defaults to 3 to match production geometry. The base sleeve is vol-targeted
    (``cross_asset_signals.baseline_weight``: ~10% per-asset, leverage cap 2.0), so its GROSS
    scales with universe size — give it 40 names and it runs at gross ~18 against a candidate's
    gross 1, and the F14 degenerate-vol cull (``cand_vol >= 0.10 * min_base_vol``) then rejects
    every candidate INCLUDING perfect foresight. That is an artifact of an over-wide base book,
    not a property of the gate: the real Taiwan base is TX/TE/TF, three futures.
    """
    if n_sleeves < 1:
        raise ValueError("n_sleeves must be >= 1")
    if base_names < 1 or base_names * n_sleeves > panel.N:
        raise ValueError(f"need 1 <= base_names*n_sleeves <= panel.N ({panel.N}); "
                         f"got {base_names}*{n_sleeves}")
    book: dict[str, np.ndarray] = {}
    for i in range(n_sleeves):
        idx = np.arange(i * base_names, (i + 1) * base_names)
        sub = Panel(
            dates=panel.dates, tickers=tuple(panel.tickers[j] for j in idx),
            open=panel.open[:, idx], high=panel.high[:, idx], low=panel.low[:, idx],
            close=panel.close[:, idx], volume=panel.volume[:, idx],
            active=panel.active[:, idx], adv_usd=panel.adv_usd[:, idx],
            sector_id=panel.sector_id[idx], meta=dict(panel.meta),
        )
        rets = np.asarray(tsmom_sleeve_returns(sub, hold_horizon=hold_horizon, cost_bps=cost_bps),
                          dtype=np.float64)
        book[f"tsmom_{i}"] = rets if base_sharpe is None else _retarget_sharpe(rets, base_sharpe, ppy)
    return book


# --------------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------------
def _legs(res: FitnessResult, fit: FitnessConfig) -> dict[str, bool]:
    """The individual AND-legs of ``passes_gate``, so a rejection names its binding constraint."""
    return {
        "uplift": bool(np.isfinite(res.delta_sr_oos)
                       and res.delta_sr_oos >= fit.min_combination_uplift),
        "dsr": bool(np.isfinite(res.dsr_aug) and res.dsr_aug >= fit.promising_dsr),
        "marginal_t": bool(res.cand_hlz_pass),
        "not_redundant": bool(np.isfinite(res.max_base_corr_obs)
                              and res.max_base_corr_obs <= fit.max_base_corr),
        "not_fragile": bool(np.isfinite(res.delta_sr_median)
                            and res.delta_sr_median >= fit.delta_median_min
                            and np.isfinite(res.frac_paths_positive)
                            and res.frac_paths_positive >= fit.frac_positive_min),
        "not_degenerate": bool(res.not_degenerate),
    }


def score_arm(label: str, cand: np.ndarray, turnover_ann: float, base: dict[str, np.ndarray],
              panel: Panel, fit: FitnessConfig, *, gen_n_eff: float, n_nodes: int) -> dict:
    """Push a planted candidate through the UNMODIFIED shipped gate and describe the outcome.

    ``gen_n_eff``/``n_nodes`` arrive at their most GENEROUS values (minimum file-drawer deflation,
    minimum complexity penalty) and ``trial_sharpe_pool`` is None — the fallback path production
    actually decided on (audit S-4: the certified pool-based holdout gate never executed). Every
    benefit of the doubt goes to the oracle, so a rejection is unambiguous.
    """
    res = combination_fitness(cand, base, panel_ts(panel), fit, gen_n_eff=gen_n_eff,
                              turnover_ann=turnover_ann, n_nodes=n_nodes, trial_sharpe_pool=None)
    legs = _legs(res, fit)
    cf = cand[np.isfinite(cand)]
    # Recover the marginal stream's annualized Sharpe from the shipped t-stat and the same
    # AR(1)-effective N the gate used, so "achieved" and "required" are on one scale.
    b_base = _combined_book(base, panel_ts(panel), fit)
    b_aug = _combined_book({**base, "_candidate": cand}, panel_ts(panel), fit)
    marg = (b_aug - b_base)
    marg = marg[np.isfinite(marg)]
    n_eff = float(_ar1_effective_n(marg)) if marg.size > 1 else float("nan")
    marg_ann_sr = (float(res.marginal_t) * float(np.sqrt(fit.periods_per_year / n_eff))
                   if np.isfinite(res.marginal_t) and n_eff > 0 else float("nan"))
    return {
        "arm": label,
        "cand_ann_sharpe": _ann_sharpe(cf, fit.periods_per_year) if cf.size > 1 else float("nan"),
        "delta_sr_oos": res.delta_sr_oos,
        "dsr_aug": res.dsr_aug,
        "marginal_t": res.marginal_t,
        "marginal_n_eff": n_eff,
        "marginal_ann_sharpe_achieved": marg_ann_sr,
        "max_base_corr": res.max_base_corr_obs,
        "turnover_ann": turnover_ann,
        # evolve.py:496 culls a candidate BEFORE scoring when turnover exceeds 2x the soft cap.
        # Reported because an arm can clear the GATE yet be unreachable in production.
        "evolve_hard_infeasible": bool(turnover_ann > fit.turnover_soft_cap * 2.0),
        "legs": legs,
        "failed_legs": sorted(k for k, v in legs.items() if not v),
        "passes_gate": bool(res.passes_gate),
    }


# --------------------------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------------------------
def build_taiwan_substrate(args: argparse.Namespace, ev: dict) -> tuple[Panel, dict, dict]:
    """The REAL Taiwan substrate the audit ran on — 10 TAIEX ETFs, TX/TE/TF futures TSMOM base.

    Both halves come from the production loaders, so this is the same panel and the same base book
    ``crucible_orchestrator``'s ``panel == "taiwan"`` branch builds:

      * ``load_taiwan_panel`` — the 10-ETF cross-section from ``configs/taiwan_cross_asset.yaml``,
        FinMind-fetched, DATA-CLEAN'd, with the gmgp1-gold stale-print gate (a ``FAIL`` status
        REFUSES to build rather than returning a quietly bad panel).
      * ``taiwan_base_sleeves`` — TSMOM on TX/TE/TF, ``return_components=True`` for the F14
        overlay-cost decomposition.

    The alt-data feature-slot bridge the orchestrator also runs is deliberately SKIPPED: the only
    slot this script needs is the planted oracle, and skipping it avoids a large number of
    connector round-trips against a 600 req/hour free-tier quota for series nothing here reads.

    Requires ``FINMIND_TOKEN`` in the environment. This script never accepts a token as an
    argument and never logs one — the loaders read the env var themselves.
    """
    import os

    if not os.environ.get("FINMIND_TOKEN"):
        raise SystemExit(
            "FINMIND_TOKEN is not set.\n"
            "  The Taiwan substrate fetches from FinMind; the loaders read the token from the\n"
            "  environment. Set it in your shell and re-run:\n"
            "      export FINMIND_TOKEN=...       # do NOT pass it on the command line\n"
            "  Free tier is 600 requests/hour. The panel is cached under data/raw/taiwan_panel,\n"
            "  so only the first run pays the fetch.\n"
            "  Without a token, run the synthetic substrate (--substrate synthetic) or the\n"
            "  data-free arithmetic (--bar-only).")

    from sharpen.data.taiwan_panel_loader import DEFAULT_CACHE_DIR, load_taiwan_panel  # noqa: PLC0415
    from sharpen.signals.generation.base_sleeves import taiwan_base_sleeves  # noqa: PLC0415

    if args.refresh_cache:
        # Regenerable cache only (raw + clean parquet + manifest), and only on explicit opt-in.
        for name in ("ohlcv_daily.parquet", "ohlcv_daily_raw.parquet",
                     "ohlcv_daily.manifest.json"):
            f = Path(DEFAULT_CACHE_DIR) / name
            if f.exists():
                log.warning("--refresh-cache: removing %s", f)
                f.unlink()

    log.info("loading real Taiwan panel %s..%s (cached after first fetch)", args.start, args.end)
    panel = load_taiwan_panel(args.start, args.end)

    # START-COVERAGE GUARD. `fetch_and_clean_taiwan` reuses its cache on an asset superset plus
    # `_cache_covers_end` — which, as its own docstring says, tests `date_max` ONLY. Nothing checks
    # the requested START, and with `end=None` the freshness leg returns True unconditionally, so a
    # cache built from a NARROWER window is served for a WIDER request and the caller silently gets
    # a truncated panel with a cheerful "using cached clean OHLCV" log line. Measured here: a cache
    # built at --start 2018-01-01 answered a --start 2010-01-01 request with 713 bars instead of
    # ~4000, and every downstream number moved without a single warning.
    #
    # This is the shipped loader's behaviour, not this script's, and the orchestrator's taiwan
    # branch calls the same function the same way. Rather than mis-measure quietly, refuse.
    want = np.datetime64(str(args.start)[:10])
    got = panel.dates.min().astype("datetime64[D]")
    slack = np.timedelta64(int(args.start_slack_days), "D")
    if got > want + slack:
        raise SystemExit(
            f"Taiwan panel starts {np.datetime_as_string(got, unit='D')} but --start asked for "
            f"{np.datetime_as_string(want, unit='D')} "
            f"({int((got - want) / np.timedelta64(1, 'D'))} days late).\n"
            "  This is the cache-reuse defect described at build_taiwan_substrate: the loader's\n"
            "  cache-hit test covers the window END only, so a narrower cached window is served\n"
            "  for a wider request without warning.\n"
            "  Re-run with --refresh-cache to refetch, or widen --start-slack-days if a late\n"
            "  panel start is genuinely expected (e.g. an ETF that had not listed yet).")
    base, components = taiwan_base_sleeves(
        panel, hold_horizon=int(ev["hold_horizon"]), cost_bps=float(ev["cost_bps"]),
        start=args.start, end=args.end, return_components=True)
    log.info("taiwan panel: T=%d N=%d  %s..%s", panel.T, panel.N,
             np.datetime_as_string(panel.dates.min(), unit="D"),
             np.datetime_as_string(panel.dates.max(), unit="D"))
    return panel, base, components


def run_length_sweep(args: argparse.Namespace) -> dict:
    """How does the perfect weekly oracle's ``marginal_t`` move with SAMPLE SIZE?

    This exists because a single number could not settle the disagreement with the audit. The gate
    statistic is ``marginal_t = SR_pp(marg) * sqrt(N_eff)``: it grows as sqrt(N) while the bar it
    must clear is a CONSTANT 3.0. So "does perfect foresight pass?" has no substrate-free answer —
    it depends entirely on how many bars you score, and the honest output is a curve with the
    crossing marked, not a verdict.

    Each window is the FIRST ``n`` bars, so every longer window is a strict superset of the shorter
    and the only thing varying is sample size (plus whatever era the extra bars add — the base
    book's own Sharpe is reported per row so that confound stays visible rather than hidden).
    """
    fit, ev = load_generation_config(_REPO_ROOT / args.gates)
    ppy = fit.periods_per_year
    cost_bps = float(ev["cost_bps"])
    base_hold = int(ev["hold_horizon"])
    rng = np.random.default_rng(args.seed)

    if args.substrate == "taiwan":
        panel, base, components = build_taiwan_substrate(args, ev)
        kind = "taiwan"
    else:
        panel = make_synthetic_panel(T=args.bars, N=args.names, seed=args.seed)
        base = build_base_book(panel, n_sleeves=args.base_sleeves, base_names=args.base_names,
                               hold_horizon=base_hold, cost_bps=cost_bps,
                               base_sharpe=args.base_sharpe, ppy=ppy)
        components, kind = None, "synthetic"

    lengths = sorted({n for n in args.length_grid if 260 <= n <= panel.T})
    if not lengths:
        raise SystemExit(f"--length-grid has no value in [260, {panel.T}] for this panel")

    rows: list[dict] = []
    for n in lengths:
        sub = panel.truncated(n - 1)
        base_n = {k: np.asarray(v)[:n] for k, v in base.items()}
        comp_n = _slice_components(components, n)
        ts_n = panel_ts(sub)
        ctx = _overlay_ctx(base_n, comp_n, ts_n, fit)
        bb = ctx[0]
        # A timing overlay can only tilt a book that actually takes positions. The Taiwan TSMOM
        # base uses a 252-bar lookback plus warmup, so below ~273 bars the book is IDENTICALLY
        # FLAT (measured: nonzero base returns = 0 at N=260, 107 at N=380). The oracle then has
        # nothing to time, `g` is constant, and `_overlay_returns` culls it — correctly. Record
        # the row as skipped with the reason rather than crashing or, worse, reporting a
        # meaningless t-stat for a window where the experiment cannot be posed.
        n_active = int((np.nan_to_num(np.asarray(bb), nan=0.0) != 0.0).sum())
        if n_active < int(args.min_active_base_bars):
            log.warning("N=%d skipped: base book has %d non-flat bars (< %d) — TSMOM warmup "
                        "incomplete, nothing for a timing overlay to tilt",
                        n, n_active, int(args.min_active_base_bars))
            rows.append({"n_bars": n, "years": n / ppy, "skipped": True,
                         "reason": f"base book flat ({n_active} active bars) — TSMOM warmup",
                         "n_active_base_bars": n_active,
                         "marginal_t": float("nan"),
                         "marginal_ann_sharpe_achieved": float("nan"),
                         "base_book_ann_sharpe": float("nan"),
                         "required_marginal_ann_sharpe": required_marginal_ann_sharpe(
                             fit.hlz_t_min, n, ppy),
                         "passes_gate": False})
            continue
        g = timing_series(bb, horizon=args.weekly_hold, skill=1.0, rng=rng,
                          signal=args.oracle_signal)
        cand, turn = book_overlay(g, sub, ctx, cost_bps=cost_bps)
        row = score_arm(f"perfect_weekly_N{n}", cand, turn, base_n, sub, fit,
                        gen_n_eff=args.gen_n_eff, n_nodes=args.n_nodes)
        bfin = np.asarray(bb)[np.isfinite(bb)]
        row.update({
            "n_bars": n,
            "years": n / ppy,
            "base_book_ann_sharpe": _ann_sharpe(bfin, ppy) if bfin.size > 1 else float("nan"),
            "required_marginal_ann_sharpe": required_marginal_ann_sharpe(fit.hlz_t_min, n, ppy),
        })
        row["skipped"] = False
        row["n_active_base_bars"] = n_active
        rows.append(row)
        log.info("N=%d marg_t=%.3f pass=%s", n, row["marginal_t"], row["passes_gate"])

    # Where does the curve cross the constant bar? Linear interpolation in sqrt(N), which is the
    # scale marginal_t is (approximately) linear in.
    crossing = None
    scored = [r for r in rows if not r.get("skipped")]
    for a, b in zip(scored, scored[1:]):
        ta, tb = a["marginal_t"], b["marginal_t"]
        if np.isfinite(ta) and np.isfinite(tb) and (ta < fit.hlz_t_min <= tb):
            xa, xb = np.sqrt(a["n_bars"]), np.sqrt(b["n_bars"])
            frac = (fit.hlz_t_min - ta) / (tb - ta) if tb != ta else 0.0
            crossing = float((xa + frac * (xb - xa)) ** 2)
            break

    return {
        "mode": "length_sweep",
        "substrate": {"kind": kind, "bars": panel.T, "names": panel.N,
                      "gates": args.gates, "periods_per_year": ppy,
                      "candidate_type": args.candidate, "oracle_signal": args.oracle_signal},
        "hlz_t_min": fit.hlz_t_min,
        "crossing_n_bars": crossing,
        "rows": rows,
    }


def _print_length_sweep(out: dict) -> None:
    ppy = out["substrate"]["periods_per_year"]
    print("\n" + "=" * 96)
    print("LENGTH SWEEP - perfect weekly oracle's marginal_t vs SAMPLE SIZE")
    print("=" * 96)
    print(f"  substrate {out['substrate']['kind']}  "
          f"oracle=perfect weekly ({out['substrate']['oracle_signal']})  "
          f"bar = marginal_t >= {out['hlz_t_min']} (a CONSTANT)")
    print(f"\n  {'bars':>6} {'years':>6} {'baseSR':>8} {'margSR':>8} {'req.SR':>8} "
          f"{'marg_t':>8}  {'PASS':>5}")
    for r in out["rows"]:
        if r.get("skipped"):
            print(f"  {r['n_bars']:>6} {r['years']:>6.2f} {'-':>8} {'-':>8} "
                  f"{r['required_marginal_ann_sharpe']:>8.3f} {'-':>8}  {'skip':>5}"
                  f"   ({r['reason']})")
            continue
        print(f"  {r['n_bars']:>6} {r['years']:>6.2f} {r['base_book_ann_sharpe']:>8.3f} "
              f"{r['marginal_ann_sharpe_achieved']:>8.3f} "
              f"{r['required_marginal_ann_sharpe']:>8.3f} {r['marginal_t']:>8.3f}  "
              f"{'YES' if r['passes_gate'] else 'no':>5}")
    c = out["crossing_n_bars"]
    print()
    if c is None:
        print("  No crossing inside the swept range - the oracle is on one side of the bar")
        print("  throughout. Widen --length-grid to locate it.")
    else:
        print(f"  => the oracle crosses marginal_t = {out['hlz_t_min']} at roughly "
              f"N ≈ {c:.0f} bars ({c / ppy:.2f}y).")
        print("    BELOW that the gate rejects perfect weekly foresight; ABOVE it, it does not.")
        print("    The audit's reported 2.92 sits on the REJECTING side of this curve, i.e. at a")
        print("    sample size well under the full panel - which is the discrepancy, stated as a")
        print("    measurement rather than a guess.")
    print("=" * 96 + "\n")


def decide(*, perfect_weekly_passes: bool, positive_passes: bool, negative_passes: bool
           ) -> tuple[str, int]:
    """The verdict rule, isolated so all three branches are unit-testable.

    Control validity is checked FIRST and dominates: a run whose positive control failed or whose
    negative control passed says nothing about the gate, whatever the ladder did. Extracted because
    on every synthetic substrate tried the oracle clears the bar, so ``SEAL_CONFIRMED`` is not
    reachable from the shipped default and would otherwise be untested code — exactly the
    one-directional blind spot this project keeps rediscovering.
    """
    if not positive_passes or negative_passes:
        return "HARNESS_INVALID", 2
    return ("SEAL_ABSENT", 1) if perfect_weekly_passes else ("SEAL_CONFIRMED", 0)


def _eval_split(*, split: str, panel: Panel, base: dict, components: dict | None,
                fit: FitnessConfig, args: argparse.Namespace, cost_bps: float, min_names: int,
                rng: np.random.Generator) -> dict:
    """Run the full ladder + both controls on one split, mirroring the production evaluation.

    ``train`` reproduces ``evolve``'s train step: panel and base truncated to the pre-embargo rows.

    ``holdout`` reproduces ``evolve``'s BINDING step, and the ordering matters. The candidate is
    built on the FULL timeline and only then sliced to the holdout rows (``evolve.py:602-624``),
    so trailing-window operators warm up from causal past history. For the overlay path this is
    not cosmetic: ``m = tanh(causal expanding z-score)``, so booking on a fresh holdout slice
    would restart the warm-up and score a different signal.
    """
    T = panel.T
    if split == "full":
        sub, base_sub, comp_sub, sl = panel, base, components, slice(0, T)
    elif split == "train":
        tr, _ = _split(panel, float(args.holdout_frac), int(args.holdout_embargo))
        n = tr.T
        sub, base_sub = tr, {k: np.asarray(v)[:n] for k, v in base.items()}
        comp_sub, sl = _slice_components(components, n), slice(0, n)
    elif split == "holdout":
        _, ho = _split(panel, float(args.holdout_frac), int(args.holdout_embargo))
        n = ho.T
        sub, base_sub, comp_sub = panel, base, components      # build on FULL, score on the tail
        sl = slice(T - n, T)
    else:
        raise ValueError(f"unknown split {split!r}")

    ts_sub = panel_ts(sub)
    ctx = (_overlay_ctx({k: np.asarray(v) for k, v in base_sub.items()}, comp_sub, ts_sub, fit)
           if args.candidate == "overlay" else None)
    bb = ctx[0] if ctx is not None else _combined_book(base_sub, ts_sub, fit)

    if split == "holdout":
        score_panel = replace(panel, dates=panel.dates[sl], open=panel.open[sl],
                              high=panel.high[sl], low=panel.low[sl], close=panel.close[sl],
                              volume=panel.volume[sl], active=panel.active[sl],
                              adv_usd=panel.adv_usd[sl], feature_slots=panel._sliced_slots(sl))
        base_score = {k: np.asarray(v)[sl] for k, v in base.items()}
    else:
        score_panel, base_score = sub, base_sub

    def _arm(label: str, *, horizon: int, skill: float | None) -> dict:
        if args.candidate == "overlay":
            g = (noise_series(sub.T, rng=rng) if skill is None
                 else timing_series(bb, horizon=horizon, skill=skill, rng=rng,
                                    signal=args.oracle_signal))
            cand, turn = book_overlay(g, sub, ctx, cost_bps=cost_bps)
        else:
            sc = (rng.standard_normal((sub.T, sub.N)) if skill is None
                  else cross_sectional_scores(sub, horizon=horizon, skill=skill, rng=rng))
            cand, turn = book_cross_sectional(sc, sub, hold_horizon=horizon,
                                              cost_bps=cost_bps, min_names=min_names)
        if split == "holdout":
            cand = cand[sl]                     # holdout rows, warmed up from train history
        row = score_arm(label, cand, turn, base_score, score_panel, fit,
                        gen_n_eff=args.gen_n_eff, n_nodes=args.n_nodes)
        row["hold_horizon"] = horizon
        return row

    ladder = [_arm(f"weekly_oracle_p{p:.2f}", horizon=args.weekly_hold, skill=p)
              for p in args.skills]
    positive = _arm("POSITIVE_daily_perfect_foresight", horizon=1, skill=1.0)
    negative = _arm("NEGATIVE_pure_noise", horizon=args.weekly_hold, skill=None)
    perfect = next(r for r in ladder if r["arm"].endswith("p1.00"))
    verdict, code = decide(perfect_weekly_passes=bool(perfect["passes_gate"]),
                           positive_passes=bool(positive["passes_gate"]),
                           negative_passes=bool(negative["passes_gate"]))

    bfin = np.asarray(bb)[sl] if split == "holdout" else np.asarray(bb)
    bfin = bfin[np.isfinite(bfin)]
    n_scored = int(bfin.size)
    return {
        "split": split,
        "verdict": verdict,
        "exit_code": code,
        "harness_valid": bool(positive["passes_gate"]) and not negative["passes_gate"],
        "n_bars_scored": n_scored,
        "base_book_ann_sharpe": (_ann_sharpe(bfin, fit.periods_per_year) if n_scored > 1
                                 else float("nan")),
        "required_marginal_ann_sharpe": required_marginal_ann_sharpe(
            fit.hlz_t_min, max(2, n_scored), fit.periods_per_year),
        "n_ladder_passed": sum(1 for r in ladder if r["passes_gate"]),
        "n_ladder": len(ladder),
        "ladder": ladder,
        "positive_control": positive,
        "negative_control": negative,
    }


def run_sweep(args: argparse.Namespace) -> dict:
    fit, ev = load_generation_config(_REPO_ROOT / args.gates)
    ppy = fit.periods_per_year
    cost_bps = float(ev["cost_bps"])
    min_names = int(ev["ls_min_names"])
    base_hold = int(ev["hold_horizon"])
    rng = np.random.default_rng(args.seed)

    if args.substrate == "taiwan":
        panel, base, components = build_taiwan_substrate(args, ev)
        meta = {"kind": "taiwan", "start": args.start, "end": args.end,
                "tickers": list(panel.tickers),
                "date_min": np.datetime_as_string(panel.dates.min(), unit="D"),
                "date_max": np.datetime_as_string(panel.dates.max(), unit="D")}
        note = ("REAL Taiwan substrate — the panel and base book the audit ran on. This run CAN "
                "confirm or refute audit F1/F2; the holdout split carries the binding verdict.")
    else:
        panel = make_synthetic_panel(T=args.bars, N=args.names, seed=args.seed)
        base = build_base_book(panel, n_sleeves=args.base_sleeves, base_names=args.base_names,
                               hold_horizon=base_hold, cost_bps=cost_bps,
                               base_sharpe=args.base_sharpe, ppy=ppy)
        components = None
        meta = {"kind": "synthetic", "seed": args.seed, "base_sleeves": args.base_sleeves,
                "base_names": args.base_names, "base_sharpe_target": args.base_sharpe}
        note = ("Synthetic substrate. The BAR reproduces exactly; the audit's Taiwan magnitudes "
                "(oracle SR 1.80 / t 2.92) do NOT - a synthetic base book is more timeable than "
                "the real one. Do not cite this run as confirmation of audit F1/F2.")

    if args.split == "auto":
        splits = ["full"] if args.substrate == "synthetic" else ["train", "holdout"]
    elif args.split == "both":
        splits = ["train", "holdout"]
    else:
        splits = [args.split]
    results = [_eval_split(split=s, panel=panel, base=base, components=components, fit=fit,
                           args=args, cost_bps=cost_bps, min_names=min_names, rng=rng)
               for s in splits]

    # The BINDING split decides the exit code: holdout when evaluated (evolve's binding gate),
    # else the single split that was run.
    binding = next((r for r in results if r["split"] == "holdout"), results[-1])
    return {
        "verdict": binding["verdict"],
        "exit_code": binding["exit_code"],
        "binding_split": binding["split"],
        "harness_valid": binding["harness_valid"],
        "substrate_can_confirm_audit": args.substrate == "taiwan",
        "note": note,
        "bar": {
            "hlz_t_min": fit.hlz_t_min,
            "formula": "required_marginal_ann_sharpe = hlz_t_min * sqrt(periods_per_year / n_bars)",
            "across_spans": bar_table(fit, tuple(args.bar_spans)),
        },
        "substrate": {
            **meta, "bars": panel.T, "names": panel.N, "candidate_type": args.candidate,
            "gates": args.gates, "periods_per_year": ppy, "cost_bps": cost_bps,
            "min_names": min_names, "base_hold_horizon": base_hold,
            "holdout_frac": args.holdout_frac, "holdout_embargo": args.holdout_embargo,
            "base_sleeve_ann_sharpe": {k: _ann_sharpe(np.asarray(v)[np.isfinite(np.asarray(v))], ppy)
                                       for k, v in base.items()},
        },
        "thresholds": {
            "hlz_t_min": fit.hlz_t_min, "promising_dsr": fit.promising_dsr,
            "min_combination_uplift": fit.min_combination_uplift,
            "max_base_corr": fit.max_base_corr, "delta_median_min": fit.delta_median_min,
            "frac_positive_min": fit.frac_positive_min,
        },
        "scoring": {"gen_n_eff": args.gen_n_eff, "n_nodes": args.n_nodes,
                    "trial_sharpe_pool": None},
        "splits": results,
        "fitness_config": asdict(fit),
    }


def _print_bar(fit: FitnessConfig, spans: tuple[int, ...]) -> None:
    print("\n" + "=" * 92)
    print("THE BAR - what the marginal-t leg demands (exact arithmetic, no data required)")
    print("=" * 92)
    print(f"  marginal_t = SR_pp(b_aug - b_base) * sqrt(N_eff)   must be >= {fit.hlz_t_min}")
    print(f"  => required annualized marginal Sharpe = {fit.hlz_t_min} * sqrt("
          f"{fit.periods_per_year:g}/N)\n")
    rows = bar_table(fit, spans)
    print(f"  {'holdout bars':>13} {'years':>7} {'shipped gate needs':>20} "
          f"{'ideal t>=2 MDE80':>18}")
    for row in rows:
        print(f"  {row['n_bars']:>13} {row['years']:>7.2f} "
              f"{row['required_marginal_ann_sharpe']:>20.3f} {row['mde80_ideal']:>18.3f}")
    print("\n  Audit anchors, both reproduced by the arithmetic above:")
    print("    F1  Taiwan holdout ~1011 bars -> 1.498   (audit: 'annualized Sharpe ~= 1.50')")
    print("    F7  ideal t>=2 MDE80, T=1011 -> 1.419    (audit: 1.42)")
    print("    F7  ideal t>=2 MDE80, T=4044 -> 0.709    (audit: 0.71)")
    print("\n  Both tighten only as 1/sqrt(N). Sixteen years of daily data still cannot detect a")
    print("  0.3-0.5 annualized dSR edge, which is the realistic size of a real single signal.")
    print("=" * 92 + "\n")


def _print(out: dict) -> None:
    s = out["substrate"]
    print("\n" + "=" * 108)
    print("PLANTED-ORACLE SWEEP - can foresight clear the shipped PROMISING gate?")
    print("=" * 108)
    if s["kind"] == "taiwan":
        print(f"substrate : REAL Taiwan  T={s['bars']}  N={s['names']}  "
              f"{s['date_min']}..{s['date_max']}  candidate={s['candidate_type']}")
        print(f"universe  : {', '.join(s['tickers'])}")
    else:
        print(f"substrate : synthetic  T={s['bars']}  N={s['names']}  seed={s['seed']}  "
              f"candidate={s['candidate_type']}")
    print(f"gates     : {s['gates']}  (t_min={out['thresholds']['hlz_t_min']}, "
          f"dsr={out['thresholds']['promising_dsr']}, "
          f"uplift={out['thresholds']['min_combination_uplift']})")
    print(f"base book : {', '.join(f'{k}={v:+.3f}' for k, v in s['base_sleeve_ann_sharpe'].items())}"
          f"  (ann SR)")
    print(f"scoring   : gen_n_eff={out['scoring']['gen_n_eff']} n_nodes={out['scoring']['n_nodes']}"
          f" pool=None  (most generous to the candidate)")

    for r in out["splits"]:
        print("\n" + "-" * 108)
        print(f"SPLIT: {r['split'].upper()}   {r['n_bars_scored']} scored bars "
              f"({r['n_bars_scored'] / s['periods_per_year']:.2f}y)   "
              f"base book ann SR {r['base_book_ann_sharpe']:+.3f}")
        print(f"  bar: marginal ann SR must reach {r['required_marginal_ann_sharpe']:.3f}")
        print("-" * 108)
        print(f"{'arm':<36} {'candSR':>8} {'margSR':>8} {'marg_t':>8} {'dSR':>7} {'dsr':>6} "
              f"{'PASS':>5}  binding")
        for a in r["ladder"] + [r["positive_control"], r["negative_control"]]:
            print(f"{a['arm']:<36} {a['cand_ann_sharpe']:>8.3f} "
                  f"{a['marginal_ann_sharpe_achieved']:>8.3f} {a['marginal_t']:>8.3f} "
                  f"{a['delta_sr_oos']:>7.3f} {a['dsr_aug']:>6.3f} "
                  f"{'YES' if a['passes_gate'] else 'no':>5}  "
                  f"{','.join(a['failed_legs']) if a['failed_legs'] else '-'}")
        print(f"  ladder {r['n_ladder_passed']}/{r['n_ladder']} passed"
              f"   positive: {'PASS' if r['positive_control']['passes_gate'] else 'FAIL'}"
              f"   negative: {'PASS' if r['negative_control']['passes_gate'] else 'FAIL'}"
              f"   -> {r['verdict']}")

    print("\n" + "=" * 108)
    print(f"VERDICT ({out['binding_split']} split is binding): {out['verdict']}")
    if out["verdict"] == "SEAL_CONFIRMED":
        print("  The gate rejected a perfect weekly-foresight oracle: its pass bar sits ABOVE")
        print("  perfect knowledge of the coming week, so every realizable strategy sits below it.")
        if out["substrate_can_confirm_audit"]:
            print("  On the REAL substrate this CONFIRMS audit F1/F2.")
    elif out["verdict"] == "SEAL_ABSENT":
        print("  The perfect weekly oracle PASSED on this substrate.")
        if out["substrate_can_confirm_audit"]:
            print("  On the REAL substrate this REFUTES audit F1/F2 as stated - re-check every")
            print("  document citing it before relying on it.")
        else:
            print("  EXPECTED on the synthetic default; NOT a refutation of F1/F2 and NOT a")
            print("  confirmation. Reproducing that verdict needs --substrate taiwan.")
    else:
        print("  A control misbehaved - this run says NOTHING about the gate.")
        for r in out["splits"]:
            if not r["positive_control"]["passes_gate"]:
                print(f"    [{r['split']}] positive control failed on: "
                      f"{','.join(r['positive_control']['failed_legs'])}")
            if r["negative_control"]["passes_gate"]:
                print(f"    [{r['split']}] negative control PASSED - gate not discriminating.")
    print(f"\n  {out['note']}")
    print("=" * 108 + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gates", default=DEFAULT_GATES,
                   help=f"gates YAML supplying every threshold (default: {DEFAULT_GATES})")
    p.add_argument("--bar-only", action="store_true",
                   help="print the arithmetic bar and exit (no simulation, instant)")
    p.add_argument("--bar-spans", type=int, nargs="+",
                   default=[252, 504, 1011, 2500, 4044, 10000],
                   help="holdout lengths for the bar table (default includes Taiwan's 1011)")
    p.add_argument("--substrate", choices=["synthetic", "taiwan"], default="synthetic",
                   help="'synthetic' (default) is self-contained and reproduces the BAR but not "
                        "the audit's magnitudes. 'taiwan' is the REAL panel the audit ran on (10 "
                        "TAIEX ETFs + TX/TE/TF base) and is the only substrate whose verdict can "
                        "confirm or refute F1/F2; it needs FINMIND_TOKEN in the environment")
    p.add_argument("--split", choices=["auto", "full", "train", "holdout", "both"], default="auto",
                   help="which split(s) to evaluate. 'auto' = full for synthetic, train+holdout "
                        "for taiwan (the audit reports both: SR 1.80 holdout, t 2.92 train). The "
                        "holdout is evolve's BINDING gate and decides the exit code")
    p.add_argument("--start", default="2010-01-01",
                   help="taiwan substrate start date (default 2010-01-01, the audit's train start)")
    p.add_argument("--end", default=None, help="taiwan substrate end date (default: latest)")
    p.add_argument("--refresh-cache", action="store_true",
                   help="delete the cached Taiwan OHLCV parquet/manifest and refetch. Needed when "
                        "widening --start, because the loader's cache-hit test covers the window "
                        "END only and will otherwise serve a narrower cached window silently")
    p.add_argument("--start-slack-days", type=int, default=10,
                   help="how many days after --start the panel may legitimately begin before the "
                        "start-coverage guard refuses (default 10; a listing gap is the usual "
                        "legitimate reason)")
    p.add_argument("--candidate", choices=["overlay", "cross_sectional"], default="overlay",
                   help="overlay = the audit's timing oracle (default); cross_sectional is a "
                        "different and much stronger object — see the module docstring")
    p.add_argument("--bars", type=int, default=2500, help="panel length T (default 2500)")
    p.add_argument("--names", type=int, default=40, help="cross-section width N (default 40)")
    p.add_argument("--seed", type=int, default=7, help="RNG seed (default 7)")
    p.add_argument("--base-sleeves", type=int, default=1,
                   help="base-book sleeve count (default 1 = the Taiwan TSMOM-only book)")
    p.add_argument("--base-names", type=int, default=3,
                   help="instruments per base sleeve (default 3 = TX/TE/TF geometry). Widening "
                        "this raises base gross and can trip the F14 degenerate-vol cull on every "
                        "candidate — see build_base_book")
    p.add_argument("--base-sharpe", type=float, default=None,
                   help="re-target each base sleeve to this annualized Sharpe (default: as "
                        "measured; the audit's Taiwan train book sat at -0.007)")
    p.add_argument("--oracle-signal", choices=["magnitude", "sign"], default="magnitude",
                   help="what the planted timing series carries: the base book's forward block "
                        "RETURN ('magnitude', default) or only its DIRECTION ('sign'). Measured "
                        "on the real Taiwan holdout the two are near-identical (marg_t 3.800 vs "
                        "3.699), so this is not the axis that explains any gap with the audit")
    p.add_argument("--weekly-hold", type=int, default=5,
                   help="oracle hold horizon in bars (default 5 = the audit's weekly oracle)")
    p.add_argument("--skills", type=float, nargs="+", default=list(SKILL_LADDER),
                   help="directional hit rates to sweep (default 0.52 0.60 0.75 0.90 1.00)")
    p.add_argument("--gen-n-eff", type=float, default=2.0,
                   help="file-drawer trial count for deflation (default 2 = most generous)")
    p.add_argument("--n-nodes", type=int, default=1,
                   help="candidate AST size for the complexity penalty (default 1 = most generous)")
    p.add_argument("--holdout-frac", type=float, default=None,
                   help="embargoed holdout fraction (default: the gates YAML's holdout_frac)")
    p.add_argument("--holdout-embargo", type=int, default=None,
                   help="embargo days between train and holdout (default: the gates YAML's)")
    p.add_argument("--length-sweep", action="store_true",
                   help="sweep the perfect weekly oracle's marginal_t against SAMPLE SIZE and "
                        "report where it crosses the constant t>=3 bar. This is the measurement "
                        "that explains why 'does perfect foresight pass?' has no single answer")
    p.add_argument("--min-active-base-bars", type=int, default=60,
                   help="a length-sweep window whose base book has fewer non-flat bars than this "
                        "is SKIPPED: a timing overlay cannot tilt a book that never takes a "
                        "position, so the experiment is not posable there (default 60)")
    p.add_argument("--length-grid", type=int, nargs="+",
                   default=[260, 380, 500, 700, 1000, 1400, 2000, 2800, 4000],
                   help="bar counts for --length-sweep (clipped to the panel)")
    p.add_argument("--json", type=str, default=None, help="write the full result record here")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    # Windows consoles here default to cp950, which cannot encode the arrows/approx signs in the
    # report and kills the run at the LAST print — after all the work is done. Force UTF-8 and
    # degrade unencodable characters instead of raising.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")   # type: ignore[union-attr]
        except (AttributeError, OSError):                            # already-wrapped / redirected
            pass

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    fit, _ev = load_generation_config(_REPO_ROOT / args.gates)
    # Split geometry defaults to the gates YAML (never hardcoded — CLAUDE.md).
    if args.holdout_frac is None:
        args.holdout_frac = float(_ev["holdout_frac"])
    if args.holdout_embargo is None:
        args.holdout_embargo = int(_ev["holdout_embargo"])
    _print_bar(fit, tuple(args.bar_spans))
    if args.bar_only:
        return 0

    if 1.00 not in args.skills:
        p.error("--skills must include 1.00 (the perfect-foresight arm decides the verdict)")

    if args.length_sweep:
        out = run_length_sweep(args)
        _print_length_sweep(out)
        if args.json:
            dest = Path(args.json)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
            print(f"wrote {dest}")
        return 0

    out = run_sweep(args)
    _print(out)
    if args.json:
        dest = Path(args.json)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(f"wrote {dest}")
    return int(out["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
