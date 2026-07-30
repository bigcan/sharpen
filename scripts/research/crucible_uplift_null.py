"""Crucible U7 — calibrate the ``uplift_min`` floor against its OWN null (audit §5 U7 / RC-9).

WHY THIS EXISTS
---------------
``configs/crucible_corrected_contract.gates.yaml`` ships ``guards.uplift_min = 0.10`` — a number that
was inherited from the funnel's ``min_combination_uplift`` and **never measured against a null**. It
passed 170/170 lifetime candidates (audit §1.1), i.e. it is not a constraint at all. That was tolerable
while the two sealed legs (``marginal_t``, ``dsr_aug``) did the rejecting. The corrected contract
(``crucible-v6.0``, DEFAULT since ``v8.0``) DROPS both seals, so ``uplift_min`` is now the only
ECONOMIC leg standing between a statistically-significant-but-worthless tilt and a PROMISING verdict —
and the audit made calibrating it a precondition on the corrected contract going live
(§5 U7, "gate on this before U1 goes live"). This script is that measurement.

WHAT IS MEASURED
----------------
The decision statistic guarded by ``uplift_min`` is ``CorrectedResult.delta_sr`` — the FULL-PANEL
annualized ΔSR between the augmented and base combined books, on the embargoed holdout rows
(``corrected_contract.py``; note the YAML comment calling it "mean ΔSR over CPCV paths" is stale — that
is ``delta_median``'s neighbourhood, not this leg). So the quantity to calibrate against is the
distribution of ``delta_sr`` for candidates WITH NO EDGE, scored on the REAL base books at the REAL
holdout length — the audit's phrasing exactly ("measure the ΔSR null distribution on the real base
books").

TWO INDEPENDENT NULLS (they must agree, or neither is trustworthy)
  * ``noise_dsl`` — the REAL base book (production/Taiwan sleeves, with F14 components) augmented by a
    candidate stream produced by the REAL DSL machinery run on a ``_realistic_noise_panel`` (GARCH-t
    fat tails + vol clustering + common factor, no planted edge). Realistic turnover, cost, and vol
    shape; zero edge by construction. This is E1's null candidate, re-pointed at a real base book.
  * ``rotate`` — the REAL candidate stream from the REAL panel, circularly ROTATED on the holdout
    window by a random offset and randomly sign-flipped. Preserves the stream's own volatility,
    serial correlation and cost drag EXACTLY; destroys only its contemporaneous alignment with the
    base book. The sign flip is what makes it a null of the UPLIFT statistic rather than of alignment
    alone: rotation alone preserves the stream's unconditional mean, and a stream with a positive mean
    raises a book's Sharpe whatever its alignment, so an un-flipped rotation null is centred on that
    mean instead of on zero. Flipping makes the draw as likely to help as to hurt — H0: "this stream
    contributes nothing systematically". Caveat: the wrap seam splices two non-adjacent bars once per
    draw (standard for a rotation null; immaterial at ~10³ bars).

The two nulls have different weaknesses (``noise_dsl`` inherits the synthetic panel's DGP; ``rotate``
inherits one real stream's marginal distribution) and no shared failure mode, so agreement is evidence.

OUTPUT
------
Per null and pooled: the ``delta_sr`` quantile ladder, the per-leg null pass rates of the corrected
contract, and — the actual deliverable — the ``uplift_min`` that puts the uplift leg at a chosen
one-sided null quantile, plus the resulting JOINT null pass rate (with a Clopper-Pearson upper bound)
at that floor. Nothing is auto-written into a gates file: the operator reads the report and edits the
YAML, per the CLAUDE.md gate rule.

Usage (from repo root):
    python scripts/research/crucible_uplift_null.py --quick          # smoke (few draws, ~1 min)
    python scripts/research/crucible_uplift_null.py                  # real measurement
    python scripts/research/crucible_uplift_null.py --panel taiwan   # the other real substrate
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.research.crucible_calibration as cal  # noqa: E402  (reuse the realistic-null substrate)
from finrl_pro_ds.crucible.corrected_contract import (  # noqa: E402
    CorrectedConfig,
    corrected_contract_fitness,
    fresh_lord_level,
)
from finrl_pro_ds.signals.features import Panel  # noqa: E402
from finrl_pro_ds.signals.generation.config import (  # noqa: E402
    load_generation_config,
    load_generation_meta,
)
from finrl_pro_ds.signals.generation.evolve import (  # noqa: E402
    _candidate_returns,
    _overlay_ctx,
    _overlay_returns,
    _split,
)
from finrl_pro_ds.signals.generation.grammar import available_terminals, parse  # noqa: E402

log = logging.getLogger("crucible_uplift_null")

DEFAULT_FUNNEL_GATES = ROOT / "configs" / "signal_eval.gates.yaml"
DEFAULT_CORRECTED_GATES = ROOT / "configs" / "crucible_corrected_contract.gates.yaml"
DEFAULT_OUT = ROOT / "results" / "crucible_uplift_null"

# Candidate banks. The cross-sectional bank is the same mechanism-diverse library subset E1 draws its
# null seeds from (`cal._CS_SEED_NUMS`); the overlay bank reads a macro feature slot. Under `noise_dsl`
# these run on the NOISE panel (no edge); under `rotate` they run on the REAL panel and are then
# rotated + sign-flipped (edge destroyed).
_CS_SEEDS = tuple(cal.FORMULAS[i] for i in cal._CS_SEED_NUMS)
_OVERLAY_SEEDS = cal._OVERLAY_NULL_SEEDS

# Quantile ladder reported for every null (one-sided upper tail — the uplift leg is one-sided).
_QUANTILES = (0.50, 0.75, 0.90, 0.95, 0.975, 0.99)


# --------------------------------------------------------------------------- real substrate loading
def _load_real_substrate(panel_name: str, ek: dict, cfg, *, start: str | None, end: str | None
                         ) -> tuple[Panel, dict[str, np.ndarray], dict, np.ndarray]:
    """(panel, base_returns_net, base_components, timestamps) for a REAL substrate — the same builders
    the orchestrator's real branch uses, with ``return_components=True`` so the overlay path charges the
    F14 cost correction exactly as production does."""
    if panel_name == "taiwan":
        from finrl_pro_ds.data.taiwan_panel_loader import load_taiwan_panel
        from finrl_pro_ds.signals.generation.base_sleeves import taiwan_base_sleeves
        panel = load_taiwan_panel(start, end)
        base, comps = taiwan_base_sleeves(
            panel, hold_horizon=ek["hold_horizon"], cost_bps=ek["cost_bps"],
            start=start, end=end, return_components=True)
    else:
        from finrl_pro_ds.data.cross_asset_panel_loader import load_cross_asset_panel
        from finrl_pro_ds.signals.generation.base_sleeves import production_base_sleeves
        panel = load_cross_asset_panel(
            start or "2007-01-01", end, config_path=ROOT / "configs" / "cross_asset_momentum.yaml")
        base, comps = production_base_sleeves(
            panel, hold_horizon=ek["hold_horizon"], cost_bps=ek["cost_bps"],
            start=start, end=end, return_components=True)
    return panel, base, comps, cal._panel_ts(panel)


def _leaves(formula: str) -> set[str]:
    """The named value-leaf terminals a formula references (``input`` payloads) — used to skip a seed
    whose feature slot is absent from a given panel instead of silently mining an all-NaN score."""
    out: set[str] = set()

    def walk(nd) -> None:
        if nd.op == "input" and isinstance(nd.payload, str):
            out.add(nd.payload)
        for ch in nd.children:
            walk(ch)

    walk(parse(formula))
    return out


# --------------------------------------------------------------------------- null candidate streams
def _holdout_geometry(panel: Panel, ek: dict) -> tuple[int, int]:
    """(n_hold, first_holdout_row) — the SAME split ``evolve`` takes, so the measurement is at the
    sample size the real decision is taken at (MDE ∝ 1/√n_hold, so this must not drift)."""
    _, hold = _split(panel, float(ek["holdout_frac"]), int(ek["holdout_embargo"]))
    return hold.T, panel.T - hold.T


def _rescale_base_sharpe(base: dict[str, np.ndarray], comps: dict, cfg,
                         target_sr: float) -> tuple[dict[str, np.ndarray], dict]:
    """H1 probe: shift each base sleeve's MEAN so the book runs at ``target_sr`` annualized, holding its
    volatility, serial structure and every other feature fixed. A pure location shift is the cleanest way
    to vary "how good is the base book" without varying anything else — the confound that made the three
    observed cells (base SR 0.045 / 0.496 / 1.364) uninterpretable, since substrate, sleeve count and
    panel all moved with it.

    The F14 components are shifted on the GROSS leg by the same amount, so ``net == gross - cost`` still
    holds exactly and the overlay path stays self-consistent."""
    import dataclasses
    out_base, out_comps = {}, {}
    for k, v in base.items():
        r = np.asarray(v, dtype=np.float64)
        m = np.isfinite(r)
        sd = float(r[m].std(ddof=1))
        want_mu = target_sr * sd / np.sqrt(cfg.periods_per_year)     # per-period mean for target ann SR
        shift = want_mu - float(r[m].mean())
        out_base[k] = r + shift
        c = comps.get(k)
        out_comps[k] = (c if c is None
                        else dataclasses.replace(c, gross=np.asarray(c.gross, dtype=np.float64) + shift,
                                                 net=out_base[k]))
    return out_base, out_comps


def _streams_noise_dsl(real_panel: Panel, real_base: dict[str, np.ndarray],
                       real_comps: dict, ts: np.ndarray, cfg, ek: dict, *,
                       n_panels: int, seed0: int, randomize_regime: bool = False
                       ) -> list[dict[str, Any]]:
    """Null-A streams: DSL candidates mined on a realistic NOISE panel of the real panel's shape,
    scored against the REAL base book. The overlay path tilts the REAL base book (that is what an
    overlay candidate is), so its null is "a timing signal built from noise"."""
    out: list[dict[str, Any]] = []
    ctx_full = _overlay_ctx(real_base, real_comps, ts, cfg)
    for k in range(n_panels):
        npanel = cal._realistic_noise_panel(real_panel.T, real_panel.N, seed=seed0 + k,
                                            n_feature_slots=4)
        if randomize_regime:
            npanel = cal.randomize_feature_slots(npanel, seed=seed0 + 90000 + k)
        for f in _CS_SEEDS:
            cr = _candidate_returns(f, npanel, hold_horizon=int(ek["hold_horizon"]),
                                    cost_bps=float(ek["cost_bps"]),
                                    min_names=int(ek["ls_min_names"]))
            if cr is not None:
                out.append({"stream": cr[0], "candidate_type": "cross_sectional",
                            "formula": f, "draw": k, "turnover_ann": float(cr[1])})
        bb, bg, bc, ge = ctx_full
        for f in _OVERLAY_SEEDS:
            # The timing signal comes from the NOISE panel's slots; the tilted book is the REAL one.
            cr = _overlay_returns(f, npanel, bb, cost_bps=float(ek["cost_bps"]),
                                  base_gross=bg, base_cost=bc, gross_exposure=ge)
            if cr is not None:
                out.append({"stream": cr[0], "candidate_type": "overlay", "formula": f, "draw": k,
                            "turnover_ann": float(cr[1])})
    return out


def _real_streams(real_panel: Panel, real_base: dict[str, np.ndarray], real_comps: dict,
                  ts: np.ndarray, cfg, ek: dict) -> list[dict[str, Any]]:
    """The REAL candidate streams (both types) on the real panel — the raw material Null-B rotates,
    and also the sanity marker "what does the un-nulled statistic look like here"."""
    out: list[dict[str, Any]] = []
    for f in _CS_SEEDS:
        cr = _candidate_returns(f, real_panel, hold_horizon=int(ek["hold_horizon"]),
                                cost_bps=float(ek["cost_bps"]), min_names=int(ek["ls_min_names"]))
        if cr is not None:
            out.append({"stream": cr[0], "candidate_type": "cross_sectional", "formula": f})
    bb, bg, bc, ge = _overlay_ctx(real_base, real_comps, ts, cfg)
    legal = set(available_terminals(real_panel))
    for f in _OVERLAY_SEEDS:
        if not _leaves(f) <= legal:       # slot absent on this real panel — skip rather than NaN-mine
            log.info("overlay seed %r skipped: terminals %s not on this panel", f,
                     sorted(_leaves(f) - legal))
            continue
        cr = _overlay_returns(f, real_panel, bb, cost_bps=float(ek["cost_bps"]),
                              base_gross=bg, base_cost=bc, gross_exposure=ge)
        if cr is not None:
            out.append({"stream": cr[0], "candidate_type": "overlay", "formula": f})
    return out


def _rotate_streams(real_streams: list[dict[str, Any]], *, n_hold: int, first_row: int,
                    n_draws: int, seed: int, mode: str,
                    min_shift_frac: float = 0.05) -> list[dict[str, Any]]:
    """Rotation nulls: each real HOLDOUT stream circularly rotated by a random offset, which destroys
    its contemporaneous alignment with the base book while preserving vol, serial correlation and cost
    drag exactly. Offsets are bounded away from 0 and ``n_hold`` so a draw is never ~the identity.

    ``mode`` picks WHICH null hypothesis the draw represents — they are NOT interchangeable:

      * ``demean`` — the holdout mean is REMOVED before rotating, so the stream has exactly zero own
        return. This is a genuine NO-EDGE stream with real-world shape: whatever ΔSR it still buys is
        chance covariance with the base book (variance reduction), which is exactly the nuisance the
        uplift floor must sit above. **This is the calibration-grade rotation null.**
      * ``signflip`` — the mean is KEPT and the sign is randomized. This models "the search fitted a
        direction on train that is uninformative on the holdout", but its MAGNITUDE is the real seed's
        own realized |mean|, so on library seeds that carry genuine edge the upper tail is inflated by
        that edge rather than by chance. Reported as a CONSERVATIVE stress reference, deliberately NOT
        folded into the recommended floor.
    """
    if mode not in ("demean", "signflip"):
        raise ValueError(f"mode must be demean|signflip; got {mode!r}")
    rng = np.random.default_rng(seed)
    lo = max(1, int(n_hold * min_shift_frac))
    out: list[dict[str, Any]] = []
    for rs in real_streams:
        ho = np.asarray(rs["stream"], dtype=np.float64)[first_row:first_row + n_hold]
        if mode == "demean":
            mu = float(np.nanmean(ho))
            ho = ho - mu
        for d in range(n_draws):
            shift = int(rng.integers(lo, max(lo + 1, n_hold - lo)))
            sign = 1.0 if (mode == "demean" or rng.random() < 0.5) else -1.0
            out.append({"stream_hold": sign * np.roll(ho, shift),
                        "candidate_type": rs["candidate_type"], "formula": rs["formula"],
                        "draw": d, "shift": shift, "sign": sign, "mode": mode})
    return out


# --------------------------------------------------------------------------- scoring + aggregation
def _score(streams: list[dict[str, Any]], *, base_ho: dict[str, np.ndarray], ts_ho: np.ndarray,
           cfg, cc: CorrectedConfig, lord_level: float, n_hold: int, first_row: int,
           prekeyed: bool = False) -> list[dict[str, Any]]:
    """Score each null stream through the CORRECTED contract on the holdout rows. ``prekeyed`` means
    the stream is already the holdout slice (Null-B); otherwise it is a full-panel stream to slice."""
    rows: list[dict[str, Any]] = []
    for i, s in enumerate(streams):
        cand = (np.asarray(s["stream_hold"], dtype=np.float64) if prekeyed
                else np.asarray(s["stream"], dtype=np.float64)[first_row:first_row + n_hold])
        if cand.size != n_hold or not np.isfinite(cand).any():
            continue
        try:
            r = corrected_contract_fitness(cand, base_ho, ts_ho, cfg, cc, lord_level=lord_level)
        except Exception as exc:                                    # noqa: BLE001 — skip, don't crash
            log.warning("null draw %d raised %r — skipped", i, exc)
            continue
        rows.append({
            "candidate_type": s["candidate_type"], "formula": s["formula"],
            "turnover_ann": float(s.get("turnover_ann", float("nan"))),
            "delta_sr": float(r.delta_sr), "corrected_t": float(r.corrected_t),
            "p_value": float(r.p_value), "delta_median": float(r.delta_median),
            "frac_positive": float(r.frac_positive), "max_base_corr": float(r.max_base_corr_obs),
            "t_pass": bool(r.t_pass), "lord_pass": bool(r.lord_pass),
            "uplift_pass": bool(r.uplift_pass), "fragility_pass": bool(r.fragility_pass),
            "collinearity_pass": bool(r.collinearity_pass),
            "passes_corrected": bool(r.passes_corrected)})
    return rows


def _joint_pass_at(rows: list[dict[str, Any]], uplift_floor: float) -> tuple[int, int]:
    """(k, n) — how many null draws clear the WHOLE corrected contract if ``uplift_min`` were set to
    ``uplift_floor``. The other four legs are re-used verbatim from the scored result; only the uplift
    leg is re-thresholded, which is exact because the legs are independent AND conditions."""
    n = len(rows)
    k = sum(1 for r in rows
            if r["t_pass"] and r["lord_pass"] and r["fragility_pass"] and r["collinearity_pass"]
            and np.isfinite(r["delta_sr"]) and r["delta_sr"] >= uplift_floor)
    return k, n


def _quantile_ci(d: np.ndarray, q: float, *, n_boot: int = 4000, alpha: float = 0.05,
                 seed: int = 7) -> tuple[float, float]:
    """Percentile-bootstrap CI for a QUANTILE of the null draws. Load-bearing for the decision this
    script feeds: a calibrated floor is only worth shipping if it is DISTINGUISHABLE from the value
    already in the gates file, and a point quantile estimate cannot answer that. Resamples draws with
    replacement (they are i.i.d. across panels/rotations by construction)."""
    if d.size < 20:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.quantile(rng.choice(d, size=(n_boot, d.size), replace=True), q, axis=1)
    return float(np.quantile(boots, alpha / 2.0)), float(np.quantile(boots, 1.0 - alpha / 2.0))


def _summarize(rows: list[dict[str, Any]], label: str, *, cc: CorrectedConfig,
               target_fpr: float, ci_alpha: float) -> dict[str, Any]:
    d = np.asarray([r["delta_sr"] for r in rows], dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"null": label, "n_draws": 0, "error": "no finite delta_sr draws"}
    qs = {f"q{int(q * 1000):04d}": float(np.quantile(d, q)) for q in _QUANTILES}
    # The floor that puts the UPLIFT LEG ALONE at the target one-sided null rate.
    floor = float(np.quantile(d, 1.0 - target_fpr))
    lo, hi = _quantile_ci(d, 1.0 - target_fpr, alpha=ci_alpha)
    k_shipped, n = _joint_pass_at(rows, cc.uplift_min)
    k_cal, _ = _joint_pass_at(rows, floor)
    legs = ("t_pass", "lord_pass", "uplift_pass", "fragility_pass", "collinearity_pass",
            "passes_corrected")
    return {
        "null": label,
        "n_draws": int(d.size),
        "delta_sr": {"mean": float(d.mean()), "sd": float(d.std(ddof=1)) if d.size > 1 else 0.0,
                     "min": float(d.min()), "max": float(d.max()), **qs},
        "per_leg_null_pass_rate": {lg: float(np.mean([r[lg] for r in rows])) for lg in legs},
        "uplift_min_shipped": cc.uplift_min,
        "uplift_min_calibrated": floor,
        "uplift_min_calibrated_ci95": [lo, hi],
        # The decision this feeds: only ship a new floor if the shipped one is OUTSIDE the CI, i.e. the
        # measurement can actually tell them apart. Otherwise the calibrated verdict is "the shipped
        # value IS the calibrated value" and churning the gates hash buys nothing but lost comparability.
        "shipped_inside_calibrated_ci": bool(np.isfinite(lo) and lo <= cc.uplift_min <= hi),
        "target_uplift_leg_fpr": target_fpr,
        "joint_null_pass_at_shipped": {
            "k": k_shipped, "n": n, "rate": k_shipped / max(1, n),
            "cp_upper95": cal.clopper_pearson_upper(k_shipped, max(1, n), ci_alpha)},
        "joint_null_pass_at_calibrated": {
            "k": k_cal, "n": n, "rate": k_cal / max(1, n),
            "cp_upper95": cal.clopper_pearson_upper(k_cal, max(1, n), ci_alpha)},
    }


def _independence_diagnostic(rows: list[dict[str, Any]], n_formulas: int) -> dict[str, Any]:
    """Is this null actually SAMPLING, or are its draws a few fixed patterns repeated?

    Compares the between-formula variance of the draw means against the mean within-formula variance
    across panels. A healthy null is dominated by panel-to-panel variation (ratio well under 1); a ratio
    >> 1 says the draws cluster by formula, i.e. the panel seed barely moves the statistic and the
    effective sample size is closer to the FORMULA count than the draw count.

    This exists because that is exactly how RC-11 was manufactured: both null-panel generators build the
    timing slot as a FIXED-period, FIXED-phase sinusoid, so four overlay seeds produced four tight
    clusters across 150 panels (ratio 11.8) and a q95 read off them looked like a null quantile but was
    "the second best of four fixed patterns". Nothing in the pipeline checked whether the null could vary
    — every guard checks a measurement's RESULT against a threshold, never the measurement's own
    dispersion. Reported for every null now, so a degenerate one is visible at a glance instead of after
    a HIGH-severity finding has propagated into six documents."""
    out: dict[str, Any] = {}
    for ct in sorted({r["candidate_type"] for r in rows}):
        d = np.asarray([r["delta_sr"] for r in rows if r["candidate_type"] == ct], dtype=float)
        d = d[np.isfinite(d)]
        k = max(1, int(n_formulas))
        if d.size < 2 * k:
            continue
        m = d[:(d.size // k) * k].reshape(-1, k)
        if m.shape[0] < 2:
            continue
        between = float(m.mean(axis=0).var(ddof=1))
        within = float(m.var(axis=0, ddof=1).mean())
        ratio = (between / within) if within > 0 else float("inf")
        out[ct] = {"n_draws": int(d.size), "n_formula_slots": k, "n_panels": int(m.shape[0]),
                   "between_formula_var": between, "within_formula_var": within,
                   "between_over_within": ratio,
                   "verdict": ("DEGENERATE — draws cluster by formula; the panel seed barely moves the "
                               "statistic, so the effective n is near the formula count, NOT the draw "
                               "count. Do not read a quantile off this." if ratio > 1.0 else
                               "healthy — panel-to-panel variation dominates")}
    return out


def _by_type(rows: list[dict[str, Any]], *, target_fpr: float) -> dict[str, Any]:
    """Per candidate_type quantiles — the two paths build ΔSR differently (a rank-L/S sleeve vs a tilt
    on the base book itself), so a single pooled floor must be read against the WORSE of the two."""
    out: dict[str, Any] = {}
    for ct in sorted({r["candidate_type"] for r in rows}):
        d = np.asarray([r["delta_sr"] for r in rows if r["candidate_type"] == ct])
        d = d[np.isfinite(d)]
        if d.size == 0:
            continue
        lo, hi = _quantile_ci(d, 1.0 - target_fpr)
        tv = np.asarray([r.get("turnover_ann", np.nan) for r in rows
                         if r["candidate_type"] == ct], dtype=float)
        tv = tv[np.isfinite(tv)]
        out[ct] = {"n": int(d.size), "mean": float(d.mean()),
                   "turnover_ann_mean": (float(tv.mean()) if tv.size else None),
                   "q0950": float(np.quantile(d, 0.95)),
                   "calibrated_floor": float(np.quantile(d, 1.0 - target_fpr)),
                   "calibrated_floor_ci95": [lo, hi],
                   "max": float(d.max()),
                   # Raw draws, so a later re-read can re-quantile / re-bootstrap without re-running the
                   # whole measurement (and so the numbers in the report are auditable, not just quoted).
                   "draws": [round(float(x), 6) for x in d]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="U7 — calibrate uplift_min against its own null")
    ap.add_argument("--panel", default="cross_asset", choices=("cross_asset", "taiwan"))
    ap.add_argument("--config", default=str(DEFAULT_FUNNEL_GATES),
                    help="funnel gates YAML (supplies FitnessConfig MECHANICS + evolve kwargs)")
    ap.add_argument("--corrected-config", default=str(DEFAULT_CORRECTED_GATES))
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--cost-bps", type=float, default=None,
                    help="override the funnel's cost_bps for this measurement. --cost-bps 0 isolates "
                         "the ALIGNMENT channel from the TURNOVER/COST channel in the overlay null.")
    ap.add_argument("--randomize-regime", action="store_true",
                    help="RC-11: give every null panel its own timing-slot shape (random period/phase/"
                         "trend/noise mix) instead of the generators' ONE fixed sin(2*pi*t/80). Without "
                         "this the overlay draws are ~4 clusters, not n independent samples.")
    ap.add_argument("--base-sharpe", type=float, default=None,
                    help="H1 probe: shift the base sleeves' means so the book runs at this annualized "
                         "Sharpe, holding vol/shape fixed. Isolates 'base-book quality' from substrate, "
                         "sleeve count and panel, which were all confounded in the observed cells.")
    ap.add_argument("--keep-sleeves", default=None,
                    help="comma-separated base-sleeve names to KEEP (e.g. 'tsmom'). Isolation knob for "
                         "the RC-11 mechanism test: the overlay null's spread differs ~200x between the "
                         "2-sleeve cross_asset book and the 1-sleeve Taiwan book, and sleeve COUNT and "
                         "base-book SHARPE are confounded across those two substrates. Dropping a sleeve "
                         "from cross_asset varies count while holding the substrate fixed.")
    ap.add_argument("--noise-panels", type=int, default=40,
                    help="Null-A: independent realistic-noise panels (x %d seeds each)" % (
                        len(_CS_SEEDS) + len(_OVERLAY_SEEDS)))
    ap.add_argument("--rotations", type=int, default=40,
                    help="Null-B: rotation+sign-flip draws per real stream")
    ap.add_argument("--target-fpr", type=float, default=0.05,
                    help="one-sided null rate the UPLIFT LEG alone should carry (default 0.05)")
    ap.add_argument("--ci-alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument("--quick", action="store_true", help="tiny budget smoke test")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.quick:
        args.noise_panels, args.rotations = 3, 4

    cfg, ek = load_generation_config(args.config)
    if args.cost_bps is not None:
        log.warning("COST OVERRIDE: cost_bps %.6f -> %.6f (mechanism probe, not a gate measurement)",
                    ek["cost_bps"], args.cost_bps)
        ek = {**ek, "cost_bps": float(args.cost_bps)}
    meta = load_generation_meta(args.config)
    cc = CorrectedConfig.from_yaml(args.corrected_config)
    lord_level = fresh_lord_level(cc)
    log.info("substrate=%s (config panel=%s) | corrected contract t_min=%.2f uplift_min=%.4f "
             "lord_level=%.5g", args.panel, meta["panel"], cc.t_min, cc.uplift_min, lord_level)

    panel, base, comps, ts = _load_real_substrate(args.panel, ek, cfg, start=args.start, end=args.end)
    if args.keep_sleeves:
        keep = {s.strip() for s in args.keep_sleeves.split(",") if s.strip()}
        missing = keep - set(base)
        if missing:
            raise SystemExit(f"--keep-sleeves names sleeves this substrate does not have: {sorted(missing)}"
                             f" (available: {sorted(base)})")
        base = {k: v for k, v in base.items() if k in keep}
        comps = {k: v for k, v in comps.items() if k in keep}
        log.warning("ISOLATION RUN: base book restricted to %s — this is a mechanism experiment, NOT the "
                    "production base book; do not read its floor as a gate recommendation", sorted(base))
    if args.base_sharpe is not None:
        base, comps = _rescale_base_sharpe(base, comps, cfg, args.base_sharpe)
        log.warning("H1 PROBE: base book mean-shifted to annualized SR %.3f — a mechanism experiment, "
                    "NOT the production base book; its floor is not a gate recommendation",
                    args.base_sharpe)
    n_hold, first_row = _holdout_geometry(panel, ek)
    base_ho = {k: np.asarray(v)[first_row:first_row + n_hold] for k, v in base.items()}
    ts_ho = np.asarray(ts)[first_row:first_row + n_hold]
    log.info("real panel T=%d N=%d | holdout rows %d..%d (n_hold=%d) | base sleeves %s",
             panel.T, panel.N, first_row, first_row + n_hold, n_hold, sorted(base))

    # ---- Null A: noise-DSL candidates vs the REAL base book
    log.info("Null-A (noise_dsl): %d noise panels x %d seeds", args.noise_panels,
             len(_CS_SEEDS) + len(_OVERLAY_SEEDS))
    a_streams = _streams_noise_dsl(panel, base, comps, ts, cfg, ek,
                                   n_panels=args.noise_panels, seed0=args.seed,
                                   randomize_regime=args.randomize_regime)
    rows_a = _score(a_streams, base_ho=base_ho, ts_ho=ts_ho, cfg=cfg, cc=cc,
                    lord_level=lord_level, n_hold=n_hold, first_row=first_row)
    log.info("Null-A scored %d draws", len(rows_a))

    # ---- Null B / C: rotations of the REAL streams (see _rotate_streams for why two modes)
    real_streams = _real_streams(panel, base, comps, ts, cfg, ek)
    log.info("rotation nulls: %d real streams x %d draws each", len(real_streams), args.rotations)
    rows_by_null: dict[str, list[dict[str, Any]]] = {"noise_dsl": rows_a}
    for i, mode in enumerate(("demean", "signflip")):
        st = _rotate_streams(real_streams, n_hold=n_hold, first_row=first_row,
                             n_draws=args.rotations, seed=args.seed + 1 + i, mode=mode)
        rows_by_null[f"rotate_{mode}"] = _score(
            st, base_ho=base_ho, ts_ho=ts_ho, cfg=cfg, cc=cc, lord_level=lord_level,
            n_hold=n_hold, first_row=first_row, prekeyed=True)
        log.info("rotate_%s scored %d draws", mode, len(rows_by_null[f"rotate_{mode}"]))

    # ---- the real (un-nulled) streams, as a marker only — NOT a verdict
    rows_real = _score(real_streams, base_ho=base_ho, ts_ho=ts_ho, cfg=cfg, cc=cc,
                       lord_level=lord_level, n_hold=n_hold, first_row=first_row)

    kw = {"cc": cc, "target_fpr": args.target_fpr, "ci_alpha": args.ci_alpha}
    # CALIBRATION-GRADE nulls (both are genuine no-edge streams) vs the conservative stress reference.
    calibration_nulls = ("noise_dsl", "rotate_demean")
    pooled = [r for k in calibration_nulls for r in rows_by_null[k]]
    summaries = [_summarize(rows_by_null[k], k, **kw) for k in rows_by_null]
    summaries.append(_summarize(pooled, "pooled_calibration_grade", **kw))
    floors = [s["uplift_min_calibrated"] for s in summaries
              if s.get("n_draws") and s["null"] in calibration_nulls]
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "U7 / RC-9 — calibrate corrected-contract guards.uplift_min against its own null",
        "substrate": {"panel": args.panel, "T": int(panel.T), "N": int(panel.N),
                      "n_hold": int(n_hold), "first_holdout_row": int(first_row),
                      "base_sleeves": sorted(base),
                      "holdout_frac": float(ek["holdout_frac"]),
                      "holdout_embargo": int(ek["holdout_embargo"])},
        "corrected_contract": {"gates": str(args.corrected_config), "t_min": cc.t_min,
                               "uplift_min_shipped": cc.uplift_min, "n_eff_mode": cc.n_eff_mode,
                               "lord_level": lord_level, "fdr_binding": cc.fdr_binding,
                               "offspring_policy": cc.offspring_policy},
        "budget": {"noise_panels": args.noise_panels, "rotations_per_stream": args.rotations,
                   "seed": args.seed, "quick": bool(args.quick)},
        "nulls": summaries,
        "calibration_grade_nulls": list(calibration_nulls),
        # RC-11 guard: a null that cannot vary produces a confident-looking quantile that is wrong.
        "independence_diagnostic": {k: _independence_diagnostic(v, len(_OVERLAY_SEEDS))
                                    for k, v in rows_by_null.items()},
        "by_candidate_type": {k: _by_type(v, target_fpr=args.target_fpr)
                              for k, v in rows_by_null.items()},
        # RECOMMENDATION = the STRICTEST calibrated floor across the CALIBRATION-GRADE nulls (the two
        # genuine no-edge constructions). Taking the max (not the mean) is deliberate: each null
        # under-represents some real uplift mechanism, and the floor's job is to be a floor.
        # `rotate_signflip` is excluded on purpose — see _rotate_streams.
        "recommended_uplift_min": (float(max(floors)) if floors else None),
        "real_streams_marker": {
            "n": len(rows_real),
            "delta_sr": sorted(round(r["delta_sr"], 4) for r in rows_real),
            "note": "the un-nulled library seeds on this substrate — a REFERENCE SCALE for reading "
                    "the calibrated floor, NOT a discovery claim (these are not pre-registered, are "
                    "scored on the same holdout, and carry no multiplicity control)"},
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sr_tag = ("" if args.base_sharpe is None
              else "_sr" + str(args.base_sharpe).replace(".", "p").replace("-", "neg"))
    tag = (f"{args.panel}{'_quick' if args.quick else ''}"
           f"{'_keep-' + args.keep_sleeves.replace(',', '-') if args.keep_sleeves else ''}"
           f"{'_randregime' if args.randomize_regime else ''}{sr_tag}"
           f"{'_cost' + str(args.cost_bps).replace('.', 'p') if args.cost_bps is not None else ''}")
    (out_dir / f"uplift_null_{tag}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    for s in summaries:
        if not s.get("n_draws"):
            continue
        log.info("[%s] n=%d  delta_sr mean %+.4f sd %.4f  q95 %+.4f  max %+.4f", s["null"],
                 s["n_draws"], s["delta_sr"]["mean"], s["delta_sr"]["sd"],
                 s["delta_sr"]["q0950"], s["delta_sr"]["max"])
        log.info("[%s] calibrated floor %.4f  CI95 [%.4f, %.4f]  shipped %.4f inside CI: %s",
                 s["null"], s["uplift_min_calibrated"], s["uplift_min_calibrated_ci95"][0],
                 s["uplift_min_calibrated_ci95"][1], s["uplift_min_shipped"],
                 s["shipped_inside_calibrated_ci"])
        log.info("[%s] uplift_min shipped %.4f -> joint null pass %d/%d (%.4f, CP95 %.4f) | "
                 "calibrated %.4f -> %d/%d (%.4f, CP95 %.4f)", s["null"], s["uplift_min_shipped"],
                 s["joint_null_pass_at_shipped"]["k"], s["joint_null_pass_at_shipped"]["n"],
                 s["joint_null_pass_at_shipped"]["rate"],
                 s["joint_null_pass_at_shipped"]["cp_upper95"], s["uplift_min_calibrated"],
                 s["joint_null_pass_at_calibrated"]["k"], s["joint_null_pass_at_calibrated"]["n"],
                 s["joint_null_pass_at_calibrated"]["rate"],
                 s["joint_null_pass_at_calibrated"]["cp_upper95"])
    log.info("RECOMMENDED uplift_min = %s (strictest calibrated floor across nulls) -> %s",
             report["recommended_uplift_min"], out_dir / f"uplift_null_{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
