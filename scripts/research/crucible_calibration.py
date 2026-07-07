"""Crucible funnel CALIBRATION harness — E1 (null false-positive rate) + E2 (planted-signal power).

WHY THIS EXISTS (S553-cont-120)
-------------------------------
Every Crucible verdict to date is "0 PROMISING". On its own that number is ambiguous: it could mean
"no alpha on this substrate" OR "the funnel cannot detect anything." This harness disambiguates them
and, read jointly, is the empirical operating characteristic of the deflation funnel:

  * E1 (null): on PURE-NOISE substrates, how often does the funnel emit a PROMISING survivor? A
    trustworthy funnel keeps this false-positive rate below a small ceiling.
  * E2 (power): on substrates with a KNOWN planted edge of controllable strength, how reliably does
    the funnel recover it? A trustworthy funnel's power -> 1 as the edge grows, and it detects a
    plausibly-small edge (a low minimum-detectable-effect).

Neither alone is sufficient (a reject-everything funnel aces E1; an accept-everything funnel aces
E2). The harness is GREEN only if BOTH hold at one operating point.

WHAT IS CALIBRATED (the real gate, not a copy)
----------------------------------------------
Drives the SHIPPED funnel: ``finrl_pro_ds.signals.generation.evolve.evolve(...)`` -> ``report.promising``,
whose binding decision is ``combination_fitness(...).passes_gate`` re-scored on the embargoed holdout
(a 5-condition AND: marginal uplift >= 0.10, deflated-Sharpe dsr_aug >= 0.90, HLZ marginal-t >= 3.0,
base-corr <= 0.70, CPCV path distribution not fragile). No statistic is re-implemented here.

Criteria live in ``configs/crucible_calibration.gates.yaml`` (never hardcoded — CLAUDE.md gate rule).
Math verification of the FPR/CI/power definitions: S553-cont-120 (findings M1-M4 in the gates file
header + the design note).

Usage (from repo root):
    python scripts/research/crucible_calibration.py --exp both
    python scripts/research/crucible_calibration.py --exp e1 --quick     # fast smoke test
    python scripts/research/crucible_calibration.py --exp e2 --out results/crucible_calibration
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finrl_pro_ds.signals.features import Panel  # noqa: E402
from finrl_pro_ds.signals.generation.config import load_generation_config  # noqa: E402
from finrl_pro_ds.signals.generation.evolve import _overlay_returns, evolve  # noqa: E402
from finrl_pro_ds.signals.generation.fitness import (  # noqa: E402
    FitnessConfig,
    _combined_book,
    combination_fitness,
)
from finrl_pro_ds.signals.library._alpha_formulas import FORMULAS  # noqa: E402

log = logging.getLogger("crucible_calibration")

DEFAULT_FUNNEL_GATES = ROOT / "configs" / "signal_eval.gates.yaml"
DEFAULT_CALIB_GATES = ROOT / "configs" / "crucible_calibration.gates.yaml"

# Cross-sectional seed bank: a small, mechanism-diverse subset of the shipped alpha library (the same
# source the evolve tests warm-start from). These are REAL DSL formulas — on a noise panel they must
# not survive (E1); they are the null bank.
_CS_SEED_NUMS = (1, 3, 4, 6, 9, 12)
# Overlay seeds read a named macro feature slot. On a null panel the slot does not predict the base
# book, so these are a valid null too.
_OVERLAY_NULL_SEEDS = (
    "macro:regime",
    "delta(macro:regime, 20)",
    "decay_linear(macro:regime, 10)",
    "delay(macro:regime, 5)",
)


# ------------------------------------------------------------------ statistics (M2)
def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Exact one-sided upper (1 - alpha) confidence bound on a binomial proportion — k successes in
    n trials. Chosen over Wilson (M2) because the null FPR is a rare event (often k=0), where Wilson
    under-covers in the tail. Closed form for k=0 is 1 - alpha**(1/n) (the rule of three); the general
    case is the Beta quantile Beta.ppf(1 - alpha, k + 1, n - k)."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    if k == 0:
        return float(1.0 - alpha ** (1.0 / n))
    from scipy.stats import beta as _beta  # type: ignore[import-untyped]
    return float(_beta.ppf(1.0 - alpha, k + 1, n - k))


def clopper_pearson_lower(k: int, n: int, alpha: float = 0.05) -> float:
    """Exact one-sided lower (1 - alpha) bound — for reporting a power estimate's floor."""
    if n <= 0:
        return 0.0
    if k <= 0:
        return 0.0
    if k >= n:
        return float(alpha ** (1.0 / n))
    from scipy.stats import beta as _beta  # type: ignore[import-untyped]
    return float(_beta.ppf(alpha, k, n - k + 1))


# ------------------------------------------------------------------ synthetic substrates
def _panel_ts(panel: Panel) -> np.ndarray:
    return panel.dates.astype("datetime64[s]").astype(np.int64).astype(np.float64)


def _noise_panel(t: int, n: int, *, seed: int, n_feature_slots: int) -> Panel:
    """Pure-noise OHLCV panel carrying ``macro:regime`` + independent ``macro:Xnn`` slots (drives the
    overlay path). No planted edge — mirrors ``crucible_orchestrator._synthetic_panel``. The base
    sleeves (built by ``_proxy_base_sleeves``) key off PRICE momentum/reversal, which is independent
    of the regime slot, so an overlay reading ``macro:regime`` here is a valid null."""
    rng = np.random.default_rng(seed)
    base = np.cumsum(0.01 * rng.standard_normal((t, n)), axis=0)
    close = np.exp(base + rng.uniform(3.0, 5.0, size=n))
    open_ = close * (1 + 0.001 * rng.standard_normal((t, n)))
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    vol = rng.uniform(1e6, 1e8, (t, n))
    dates = (np.datetime64("2010-01-04") + np.arange(t) * np.timedelta64(1, "D")
             ).astype("datetime64[ns]")
    regime = (np.sin(2.0 * np.pi * np.arange(t) / 80.0) + 0.2 * rng.standard_normal(t)
              ).astype(np.float64)
    slots = {"macro:regime": regime}
    for i in range(max(0, n_feature_slots - 1)):
        slots[f"macro:X{i:02d}"] = np.cumsum(0.05 * rng.standard_normal(t)).astype(np.float64)
    return Panel(dates, tuple(f"S{i:02d}" for i in range(n)), open_, high, low, close, vol,
                 np.ones((t, n), bool), close * vol, rng.integers(0, 4, size=n),
                 {"survivorship_free": True, "source": "synthetic_noise"}, feature_slots=slots)


def _proxy_base_sleeves(panel: Panel, *, hold: int) -> dict[str, np.ndarray]:
    """Inline TSMOM (252-1) + short reversal PROXY rank-L/S books — a base book for the marginal gate
    to improve upon. Derived from PRICE only (independent of the regime slot)."""
    from finrl_pro_ds.signals.eval_harness import _ls_weights

    c = panel.close
    fwd1 = panel.forward_returns(1)

    def book(score: np.ndarray) -> np.ndarray:
        out = np.full(panel.T, np.nan)
        w = np.zeros(panel.N)
        for t in range(panel.T - 1):
            if t % hold == 0:
                w = _ls_weights(score[t], panel.active[t], min_names=6)
            out[t] = float(np.nansum(w * fwd1[t]))
        return np.nan_to_num(out)

    mom = np.full_like(c, np.nan)
    mom[252:] = c[252:] / c[:-252] - 1.0
    rev = np.full_like(c, np.nan)
    rev[21:] = -(c[21:] / c[:-21] - 1.0)
    return {"tsmom": book(mom), "rates_carry": book(rev)}


def _planted_panel(t: int, n: int, *, seed: int) -> tuple[Panel, np.ndarray]:
    """Noise OHLCV panel carrying a ``macro:plant`` regime slot ``s`` (slow sinusoid + noise). Returns
    (panel, s). The base sleeves are built separately (``_planted_base_sleeves``) so the base book's
    FUTURE return depends on the LAGGED plant — the overlay that reads ``macro:plant`` then earns a
    genuine MARGINAL uplift by timing the base book. Strength is injected via the base, not here, so
    beta=0 collapses cleanly to a null."""
    rng = np.random.default_rng(seed)
    base = np.cumsum(0.01 * rng.standard_normal((t, n)), axis=0)
    close = np.exp(base + rng.uniform(3.0, 5.0, size=n))
    open_ = close * (1 + 0.001 * rng.standard_normal((t, n)))
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    vol = rng.uniform(1e6, 1e8, (t, n))
    dates = (np.datetime64("2010-01-04") + np.arange(t) * np.timedelta64(1, "D")
             ).astype("datetime64[ns]")
    s = (np.sin(2.0 * np.pi * np.arange(t) / 80.0) + 0.2 * rng.standard_normal(t)).astype(np.float64)
    return (Panel(dates, tuple(f"S{i:02d}" for i in range(n)), open_, high, low, close, vol,
                  np.ones((t, n), bool), close * vol, rng.integers(0, 4, size=n),
                  {"survivorship_free": True, "source": "synthetic_planted"},
                  feature_slots={"macro:plant": s}), s)


def _planted_base_sleeves(s: np.ndarray, *, beta: float, seed: int) -> dict[str, np.ndarray]:
    """Base book whose tsmom return at ``t`` depends on the LAGGED, standardized plant ``s[t-1]`` with
    strength ``beta``, plus a substantial idiosyncratic noise floor. An overlay reading ``macro:plant``
    times/amplifies this regime -> positive MARGINAL uplift. The predictor is CONTINUOUS (not sign()):
    a square-wave makes the edge unrealistically clean and jumps realized ΔSR straight past the
    interesting region (probe, S553-cont-120); the noise floor makes beta sweep realized ΔSR smoothly
    through the realistic 0.1-3 band so E2 can locate the funnel's minimum detectable effect. beta=0
    -> pure-noise base (null; realized ΔSR ~0, undetected)."""
    rng = np.random.default_rng(seed + 7919)
    t = s.size
    s_lag = np.concatenate([[0.0], s[:-1]])
    s_std = (s_lag - s_lag.mean()) / (s_lag.std() + 1e-12)
    tsmom = 0.0004 + beta * s_std + 0.020 * rng.standard_normal(t)
    carry = 0.0003 + 0.008 * rng.standard_normal(t)
    return {"tsmom": tsmom.astype(np.float64), "rates_carry": carry.astype(np.float64)}


# ------------------------------------------------------------------ config
@dataclass(frozen=True, slots=True)
class CalibConfig:
    raw: dict
    fit_cfg: FitnessConfig   # the real funnel thresholds being calibrated (promising_dsr etc.)
    ek: dict                 # evolve_kwargs merged with the calibration search budget

    @property
    def substrate(self) -> dict:
        return self.raw["substrate"]


def load_calib(calib_gates: Path, funnel_gates: Path) -> CalibConfig:
    raw = yaml.safe_load(Path(calib_gates).read_text(encoding="utf-8"))
    fit_cfg, ek_funnel = load_generation_config(funnel_gates)   # real FitnessConfig thresholds
    # Override ONLY the search-budget knobs from the calibration file (keeps the funnel THRESHOLDS —
    # promising_dsr, hlz_t_min, ... — exactly as shipped; we calibrate those, we don't move them).
    b = raw["evolve_budget"]
    ek = dict(ek_funnel)
    ek.update(rng_seed=int(b["rng_seed"]), pop_size=int(b["pop_size"]),
              n_generations=int(b["n_generations"]), ls_min_names=int(b["ls_min_names"]),
              elite_frac=float(b["elite_frac"]), cost_bps=float(b["cost_bps"]),
              holdout_frac=float(b["holdout_frac"]), holdout_embargo=int(b["holdout_embargo_days"]))
    ek["hold_horizon"] = int(raw["substrate"]["hold_horizon"])
    return CalibConfig(raw=raw, fit_cfg=fit_cfg, ek=ek)


# ------------------------------------------------------------------ E1: null FPR
@dataclass
class LegTally:
    uplift: int = 0
    dsr: int = 0
    marginal_t: int = 0
    not_redundant: int = 0
    not_fragile: int = 0
    all_pass: int = 0
    denom: int = 0

    def observe(self, r, fit) -> None:
        if r is None:
            return
        self.denom += 1
        if np.isfinite(r.delta_sr_oos) and r.delta_sr_oos >= fit.min_combination_uplift:
            self.uplift += 1
        if np.isfinite(r.dsr_aug) and r.dsr_aug >= fit.promising_dsr:
            self.dsr += 1
        if r.cand_hlz_pass:
            self.marginal_t += 1
        if np.isfinite(r.max_base_corr_obs) and r.max_base_corr_obs <= fit.max_base_corr:
            self.not_redundant += 1
        if (np.isfinite(r.delta_sr_median) and r.delta_sr_median >= fit.delta_median_min
                and np.isfinite(r.frac_paths_positive) and r.frac_paths_positive >= fit.frac_positive_min):
            self.not_fragile += 1
        if r.passes_gate:
            self.all_pass += 1

    def rates(self) -> dict:
        d = max(1, self.denom)
        return {k: getattr(self, k) / d for k in
                ("uplift", "dsr", "marginal_t", "not_redundant", "not_fragile", "all_pass")}


def run_e1(cc: CalibConfig, *, quick: bool) -> dict:
    sub = cc.substrate
    e1 = cc.raw["e1_null"]
    n_panels = 6 if quick else int(e1["n_panels"])
    alpha = float(e1["ci_alpha"])
    t, n, n_slots = int(sub["t"]), int(sub["n"]), int(sub["n_feature_slots"])
    cs_seeds = [FORMULAS[i] for i in _CS_SEED_NUMS]
    ov_seeds = list(_OVERLAY_NULL_SEEDS)

    promising_total = 0
    genomes_total = 0
    ticks_with_promising = 0
    holdout_evals = 0
    gen_n_effs: list[float] = []
    legs = LegTally()

    log.info("E1 null-calibration: %d noise panels x {cross_sectional[%d] + overlay[%d]} seeds",
             n_panels, len(cs_seeds), len(ov_seeds))
    for k in range(n_panels):
        panel = _noise_panel(t, n, seed=1000 + k, n_feature_slots=n_slots)
        base = _proxy_base_sleeves(panel, hold=cc.ek["hold_horizon"])
        ts = _panel_ts(panel)
        n_prom_this = 0
        for ct, seeds in (("cross_sectional", cs_seeds), ("overlay", ov_seeds)):
            rep = evolve(seeds, panel, base, ts, cc.fit_cfg, candidate_type=ct, **cc.ek)
            n_prom_this += len(rep.promising)
            genomes_total += int(rep.gen_n_total)
            holdout_evals += len(rep.holdout_validation)
            gen_n_effs.append(float(rep.gen_n_eff))
            for c in rep.hall_of_fame:
                legs.observe(c.result, cc.fit_cfg)
        promising_total += n_prom_this
        if n_prom_this > 0:
            ticks_with_promising += 1
        log.info("  panel %d/%d: promising=%d", k + 1, n_panels, n_prom_this)

    per_tick_fpr = ticks_with_promising / max(1, n_panels)
    per_cand_fpr = promising_total / max(1, genomes_total)
    tick_upper = clopper_pearson_upper(ticks_with_promising, n_panels, alpha)
    cand_upper = clopper_pearson_upper(promising_total, max(1, genomes_total), alpha)

    tick_ceiling = float(e1["null_tick_fpr_max"])
    cand_ceiling = float(e1["null_fpr_max"])
    e1_green = bool(tick_upper <= tick_ceiling and cand_upper <= cand_ceiling)
    return {
        "n_panels": n_panels,
        "seeds": {"cross_sectional": len(cs_seeds), "overlay": len(ov_seeds)},
        "promising_total": promising_total,
        "genomes_scored_total": genomes_total,
        "holdout_evals": holdout_evals,
        "mean_gen_n_eff": float(np.mean(gen_n_effs)) if gen_n_effs else float("nan"),
        "per_tick_fpr": per_tick_fpr,
        "per_tick_fpr_cp_upper95": tick_upper,
        "per_tick_fpr_ceiling": tick_ceiling,
        "per_candidate_fpr": per_cand_fpr,
        "per_candidate_fpr_cp_upper95": cand_upper,
        "per_candidate_fpr_ceiling": cand_ceiling,
        "per_leg_null_pass_rate": legs.rates(),   # M1 diagnostic: which leg binds under the null
        "per_leg_denom": legs.denom,
        "verdict": "GREEN" if e1_green else "RED",
    }


# ------------------------------------------------------------------ E2: planted-signal power
def _e2_power_curve(cc: CalibConfig, *, t: int, n: int, betas: list[float], n_seeds: int,
                    gen_n_eff: float, cost_bps: float, alpha: float = 0.05) -> list[dict[str, Any]]:
    """The E2 power curve at a FIXED panel length ``t``: for each plant strength ``beta``, score the
    planted overlay ``macro:plant`` through the real ``combination_fitness`` gate over ``n_seeds``
    panels and record power = P(passes_gate) + mean realized marginal ΔSR + per-leg pass-rates.
    Shared by ``run_e2`` (single T) and ``run_mde_sweep`` (T grid)."""
    curve: list[dict[str, Any]] = []
    for beta in betas:
        detections = 0
        realized_deltas: list[float] = []
        legs = LegTally()
        for k in range(n_seeds):
            panel, s = _planted_panel(t, n, seed=5000 + k)
            base = _planted_base_sleeves(s, beta=beta, seed=5000 + k)
            ts = _panel_ts(panel)
            base_book = _combined_book(base, ts, cc.fit_cfg)
            out = _overlay_returns("macro:plant", panel, base_book, cost_bps=cost_bps)
            if out is None:
                continue                                  # degenerate overlay (should not happen)
            cand, turnover = out
            res = combination_fitness(cand, base, ts, cc.fit_cfg, gen_n_eff=gen_n_eff,
                                      turnover_ann=turnover, n_nodes=1)
            legs.observe(res, cc.fit_cfg)
            if res.passes_gate:
                detections += 1
            if np.isfinite(res.delta_sr_oos):
                realized_deltas.append(float(res.delta_sr_oos))
        power = detections / max(1, n_seeds)
        mean_delta = float(np.mean(realized_deltas)) if realized_deltas else float("nan")
        curve.append({
            "beta": beta, "power": power, "detections": detections, "n_seeds": n_seeds,
            "power_cp_lower95": clopper_pearson_lower(detections, n_seeds, alpha),
            "mean_realized_delta_sr": mean_delta,
            "per_leg_pass_rate": legs.rates(),            # which leg blocks near the MDE
        })
    return curve


def _mde_from_curve(curve: list[dict[str, Any]], power_target: float) -> dict[str, Any] | None:
    """The MDE entry = the first point (ascending beta) whose power reaches ``power_target``."""
    return next((p for p in curve if float(p["power"]) >= power_target), None)


def run_e2(cc: CalibConfig, *, quick: bool) -> dict:
    """Planted-signal power via the DIRECT gate path: score the planted overlay ``macro:plant``
    through the real ``combination_fitness`` (the exact PROMISING gate) at a declared file-drawer N.
    This isolates the funnel's DETECTION POWER as a function of the realized marginal ΔSR — fast (no
    search), fully diagnostic (every leg readable), and it uses the real gate, not a copy. It scores
    on the FULL panel, so it is an UPPER bound on the deployed holdout-binding power (the holdout has
    ~holdout_frac of the bars, hence less power) — i.e. any under-power finding is conservative."""
    sub = cc.substrate
    e2 = cc.raw["e2_power"]
    n_seeds = 6 if quick else int(e2["n_seeds"])
    betas = [0.0, 0.008, 0.02] if quick else [float(b) for b in e2["beta_grid"]]
    gen_n_eff = float(e2["gate_probe_gen_n_eff"])
    t, n = int(sub["t"]), int(sub["n"])
    cost_bps = float(cc.ek["cost_bps"])

    log.info("E2 planted-power (direct gate, T=%d, gen_n_eff=%.0f): %d betas x %d seeds",
             t, gen_n_eff, len(betas), n_seeds)
    curve = _e2_power_curve(cc, t=t, n=n, betas=betas, n_seeds=n_seeds, gen_n_eff=gen_n_eff,
                            cost_bps=cost_bps)
    for p in curve:
        log.info("  beta=%.4f: power=%.2f (%d/%d)  realized_dSR=%.3f", p["beta"], p["power"],
                 p["detections"], p["n_seeds"], p["mean_realized_delta_sr"])

    power_target = float(e2["power_target"])
    power_min = float(e2["power_min"])
    mde_delta_cap = float(e2["mde_delta_sr_max"])
    # MDE = first point (ascending beta) reaching power_target; its realized marginal ΔSR must be
    # <= the cap for GREEN (the funnel detects a plausibly-small edge, not only a huge one).
    mde = _mde_from_curve(curve, power_target)
    max_power = max((float(p["power"]) for p in curve), default=0.0)
    beta0 = next((p for p in curve if float(p["beta"]) == 0.0), None)
    mde_delta = float(mde["mean_realized_delta_sr"]) if mde is not None else float("nan")
    e2_green = bool(
        mde is not None and max_power >= power_min
        and np.isfinite(mde_delta) and mde_delta <= mde_delta_cap)
    return {
        "t": t, "n_seeds": n_seeds, "beta_grid": betas, "gate_probe_gen_n_eff": gen_n_eff,
        "power_curve": curve, "power_target": power_target, "power_min": power_min,
        "max_power": max_power,
        "mde_beta": (mde["beta"] if mde is not None else None),
        "mde_realized_delta_sr": (mde_delta if mde is not None else None),
        "mde_delta_sr_cap": mde_delta_cap,
        "null_consistency_beta0_power": (beta0["power"] if beta0 else None),  # should ~match E1 FPR
        "note": ("full-panel gate power = UPPER bound on deployed holdout-binding power; MDE scales "
                 "~1/sqrt(T_holdout), so longer real substrates detect smaller edges"),
        "verdict": "GREEN" if e2_green else "RED",
    }


def run_mde_sweep(cc: CalibConfig, *, quick: bool) -> dict:
    """The MDE-vs-T curve: run the E2 power sweep at each substrate length in ``mde_sweep.t_grid``
    and report the funnel's detection floor (MDE in realized marginal ΔSR at ``power_target``) as a
    function of sample size. This is THE characterization of the funnel's power: because MDE scales
    ~1/sqrt(T_holdout), it shows how small an edge the funnel can see on the ACTUAL real substrates
    (cross_asset ~4044 bars, Taiwan ~2782) vs short ones — i.e. whether any past '0 PROMISING' was
    a real absence or just low power at that substrate's length."""
    sub = cc.substrate
    e2 = cc.raw["e2_power"]
    sw = cc.raw["mde_sweep"]
    n = int(sub["n"])
    gen_n_eff = float(e2["gate_probe_gen_n_eff"])
    cost_bps = float(cc.ek["cost_bps"])
    holdout_frac = float(cc.ek["holdout_frac"])
    power_target = float(e2["power_target"])
    n_seeds = 8 if quick else int(sw["n_seeds"])
    betas = [0.0, 0.008, 0.02, 0.04] if quick else [float(b) for b in sw["beta_grid"]]
    t_grid = [756, 2782] if quick else [int(x) for x in sw["t_grid"]]

    log.info("MDE-vs-T sweep: T in %s x %d betas x %d seeds", t_grid, len(betas), n_seeds)
    rows: list[dict[str, Any]] = []
    for t in t_grid:
        curve = _e2_power_curve(cc, t=t, n=n, betas=betas, n_seeds=n_seeds, gen_n_eff=gen_n_eff,
                                cost_bps=cost_bps)
        mde = _mde_from_curve(curve, power_target)
        row = {
            "t": t, "holdout_bars": int(round(t * holdout_frac)),
            "mde_realized_delta_sr": (float(mde["mean_realized_delta_sr"]) if mde else None),
            "mde_beta": (mde["beta"] if mde else None),
            "mde_power": (float(mde["power"]) if mde else None),
            "max_power": max((float(p["power"]) for p in curve), default=0.0),
            "curve": [{"beta": p["beta"], "power": p["power"],
                       "realized_delta_sr": p["mean_realized_delta_sr"]} for p in curve],
        }
        rows.append(row)
        log.info("  T=%d (holdout ~%d bars): MDE realized_dSR=%s @ power=%s",
                 t, row["holdout_bars"], row["mde_realized_delta_sr"], row["mde_power"])

    # Physical expectation: MDE FALLS as T grows (more data -> smaller detectable edge, ~1/sqrt(T)).
    mdes = [r["mde_realized_delta_sr"] for r in rows if r["mde_realized_delta_sr"] is not None]
    monotone = all(mdes[i] >= mdes[i + 1] - 0.5 for i in range(len(mdes) - 1))  # slack for MC noise
    return {
        "t_grid": t_grid, "n_seeds": n_seeds, "beta_grid": betas,
        "power_target": power_target, "gate_probe_gen_n_eff": gen_n_eff,
        "holdout_frac": holdout_frac, "rows": rows,
        "mde_falls_with_t": monotone,
        "note": ("MDE = realized marginal ΔSR (annualized) at power_target, full-panel gate (UPPER "
                 "bound on deployed holdout-binding power). Expect MDE ~ 1/sqrt(T_holdout)."),
    }


# ------------------------------------------------------------------ driver
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="Crucible funnel calibration: E1 null-FPR + E2 power")
    ap.add_argument("--exp", choices=("e1", "e2", "both", "mde_sweep"), default="both",
                    help="e1/e2/both = calibration verdicts; mde_sweep = the MDE-vs-T detection "
                         "floor across substrate lengths (the funnel's power characterization).")
    ap.add_argument("--calib-config", default=str(DEFAULT_CALIB_GATES))
    ap.add_argument("--config", default=str(DEFAULT_FUNNEL_GATES),
                    help="funnel gates YAML — its generation block supplies the REAL FitnessConfig "
                         "thresholds being calibrated (default signal_eval.gates.yaml).")
    ap.add_argument("--out", default=str(ROOT / "results" / "crucible_calibration"))
    ap.add_argument("--quick", action="store_true",
                    help="tiny K / short beta grid — smoke test the harness end-to-end, NOT a verdict.")
    args = ap.parse_args()

    cc = load_calib(Path(args.calib_config), Path(args.config))
    ts = datetime.now(timezone.utc).isoformat()
    report: dict = {
        "harness_version": cc.raw.get("harness_version"),
        "ts": ts, "quick": args.quick, "exp": args.exp,
        "funnel_gates": str(args.config),
        "funnel_thresholds": {
            "promising_dsr": cc.fit_cfg.promising_dsr, "hlz_t_min": cc.fit_cfg.hlz_t_min,
            "min_combination_uplift": cc.fit_cfg.min_combination_uplift,
            "max_base_corr": cc.fit_cfg.max_base_corr},
        "search_budget": {k: cc.ek[k] for k in ("pop_size", "n_generations", "rng_seed")},
    }
    if args.exp in ("e1", "both"):
        report["e1"] = run_e1(cc, quick=args.quick)
    if args.exp in ("e2", "both"):
        report["e2"] = run_e2(cc, quick=args.quick)
    if args.exp == "mde_sweep":
        report["mde_sweep"] = run_mde_sweep(cc, quick=args.quick)

    # Joint verdict (M4): GREEN only if BOTH pass (when both ran). The sweep is a characterization,
    # not a pass/fail, so it carries no joint verdict.
    verdicts = [report[e]["verdict"] for e in ("e1", "e2") if e in report]
    if cc.raw.get("joint", {}).get("require_both", True) and len(verdicts) == 2:
        report["joint_verdict"] = "GREEN" if all(v == "GREEN" for v in verdicts) else "RED"
    else:
        report["joint_verdict"] = verdicts[0] if len(verdicts) == 1 else "N/A"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "calibration_quick" if args.quick else "calibration"
    out_path = out_dir / f"{stem}_{args.exp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n==================== CRUCIBLE CALIBRATION ====================")
    if "e1" in report:
        e = report["e1"]
        print(f"E1 null-FPR   : per-tick={e['per_tick_fpr']:.3f} (CP<={e['per_tick_fpr_cp_upper95']:.3f} "
              f"vs {e['per_tick_fpr_ceiling']}) | per-cand={e['per_candidate_fpr']:.4f} "
              f"(CP<={e['per_candidate_fpr_cp_upper95']:.4f} vs {e['per_candidate_fpr_ceiling']}) "
              f"-> {e['verdict']}")
        print("   per-leg null pass-rate (which leg binds): "
              + ", ".join(f"{k}={v:.3f}" for k, v in e["per_leg_null_pass_rate"].items()))
    if "e2" in report:
        e = report["e2"]
        print(f"E2 power      : max_power={e['max_power']:.2f} | MDE beta={e['mde_beta']} "
              f"realized_dSR={e['mde_realized_delta_sr']} (cap {e['mde_delta_sr_cap']}) "
              f"| beta0 power={e['null_consistency_beta0_power']} -> {e['verdict']}")
    if "mde_sweep" in report:
        sw = report["mde_sweep"]
        print(f"MDE-vs-T (detection floor @ power>={sw['power_target']}; realized marginal dSR):")
        print(f"   {'T(bars)':>8} {'holdout':>8} {'MDE_dSR':>9} {'@power':>7}")
        for r in sw["rows"]:
            mde = r["mde_realized_delta_sr"]
            print(f"   {r['t']:>8} {r['holdout_bars']:>8} "
                  f"{('%.2f' % mde) if mde is not None else 'undetected':>9} "
                  f"{('%.2f' % r['mde_power']) if r['mde_power'] is not None else '-':>7}")
        print(f"   MDE falls with T (expected ~1/sqrt(T)): {sw['mde_falls_with_t']}")
    print(f"JOINT VERDICT : {report['joint_verdict']}")
    print(f"report -> {out_path}")
    print("==============================================================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
