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
    python scripts/research/crucible_calibration.py --exp e1 --null realistic  # F4 realistic null (1 pt)
    python scripts/research/crucible_calibration.py --exp e1_sensitivity # F4 robustness across null shapes
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

from finrl_pro_ds.crucible.corrected_contract import (  # noqa: E402
    CorrectedConfig,
    corrected_contract_fitness,
    fresh_lord_level,
)
from finrl_pro_ds.crucible.orchestrator.substrate import _power_holdout_bars  # noqa: E402
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
DEFAULT_CORRECTED_GATES = ROOT / "configs" / "crucible_corrected_contract.gates.yaml"

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


# ---------------------------------------------------------- realistic null (audit F14 / S553-cont-138)
def _garch_t_series(t: int, k: int, *, rng: np.random.Generator, df: float, alpha: float,
                    beta: float, uncond_var: float) -> np.ndarray:
    """``(t, k)`` zero-mean returns with FAT TAILS + VOL CLUSTERING, vectorized across the ``k`` series
    (sequential only in time). Innovations are standardized Student-t(``df``): ``z = T(df)·√((df−2)/df)``
    has unit variance (Var[T(df)] = df/(df−2)). Conditional variance follows GARCH(1,1)
    ``h_s = ω + α·a²_{s−1} + β·h_{s−1}`` with ``a_s = √h_s · z_s``; ``ω = uncond_var·(1−α−β)`` makes the
    unconditional variance exactly ``uncond_var`` (E[h] = ω/(1−α−β)). Requires ``df>2`` (finite variance)
    and ``α,β≥0, α+β<1`` (covariance-stationary)."""
    if df <= 2.0:
        raise ValueError(f"df must be > 2 for finite variance; got {df}")
    if alpha < 0.0 or beta < 0.0 or alpha + beta >= 1.0:
        raise ValueError(f"need alpha,beta>=0 and alpha+beta<1 (stationarity); got {alpha}, {beta}")
    z = rng.standard_t(df, size=(t, k)) * np.sqrt((df - 2.0) / df)     # standardized -> Var = 1
    omega = uncond_var * (1.0 - alpha - beta)
    h = np.empty((t, k), dtype=np.float64)
    a = np.empty((t, k), dtype=np.float64)
    h[0] = uncond_var
    a[0] = np.sqrt(h[0]) * z[0]
    for s in range(1, t):
        h[s] = omega + alpha * a[s - 1] ** 2 + beta * h[s - 1]
        a[s] = np.sqrt(h[s]) * z[s]
    return a


def _realistic_returns(t: int, n: int, *, rng: np.random.Generator, df: float = 5.0,
                       garch_alpha: float = 0.08, garch_beta: float = 0.90,
                       target_vol: float = 0.01, factor_share: float = 0.35,
                       load_mean: float = 1.0, load_sd: float = 0.3) -> np.ndarray:
    """``(t, n)`` NULL returns = one shared market factor + idiosyncratic, each a GARCH-t series:
    ``r[:, i] = load_i·f + idio[:, i]``. The shared ``f`` induces a positive average cross-sectional
    correlation (COMMON FACTOR ≈ ``factor_share`` — exactly, ≈0.33 at defaults, since the factor explains
    ``factor_share`` of AVERAGE variance but per-pair corr normalizes by per-asset σ). The variance budget
    splits each asset's total ≈ ``target_vol``² so the factor explains ≈ ``factor_share`` of it:
    ``mean_i(load_i²)·σ_f² = factor_share·target_vol²`` (using the REALIZED ``mean(loads²)``, which equals
    the theoretical ``load_mean²+load_sd²`` in expectation), ``σ_idio² = (1−factor_share)·target_vol²`` —
    so ``mean_i Var[r_i] = target_vol²`` exactly for the realized loads. Total vol matches the IID panel's
    0.01 increment scale, so the ONLY change vs the IID null is SHAPE (fat tails, clustering, correlation).
    There is NO planted edge — returns are independent of the overlay slots — so this stays a valid null."""
    tv2 = target_vol ** 2
    loads = rng.normal(load_mean, load_sd, size=n)
    e_load2 = float(np.mean(loads ** 2))                      # realized E[load²] for this panel
    fac_var = factor_share * tv2 / max(e_load2, 1e-12)
    idio_var = (1.0 - factor_share) * tv2
    f = _garch_t_series(t, 1, rng=rng, df=df, alpha=garch_alpha, beta=garch_beta,
                        uncond_var=fac_var)[:, 0]
    idio = _garch_t_series(t, n, rng=rng, df=df, alpha=garch_alpha, beta=garch_beta,
                           uncond_var=idio_var)
    return loads[None, :] * f[:, None] + idio


def _realistic_noise_panel(t: int, n: int, *, seed: int, n_feature_slots: int,
                           null_params: dict | None = None) -> Panel:
    """Realistic-null counterpart of ``_noise_panel``: identical OHLCV/vol/date/(null)-slot construction,
    but ``close`` compounds ``_realistic_returns`` (fat tails + vol clustering + common factor) instead of
    IID-Gaussian increments. Audit F14: today's E1 GREEN rides on the mis-specified marginal_t/dsr legs
    under an IID-Gaussian null; this panel re-measures that FPR under a realistic null. No planted edge.
    ``null_params`` (optional) overrides the ``_realistic_returns`` null-SHAPE knobs (``df``,
    ``factor_share``, ...) — the axis the E1 sensitivity sweep varies (S553-cont-138)."""
    rng = np.random.default_rng(seed)
    r = _realistic_returns(t, n, rng=rng, **(null_params or {}))
    close = np.exp(np.cumsum(r, axis=0) + rng.uniform(3.0, 5.0, size=n))
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
                 {"survivorship_free": True, "source": "synthetic_noise_realistic"},
                 feature_slots=slots)


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


# -------------------------------------------- planted CROSS-SECTIONAL substrate (breadth axis, V2)
# The planted fixture above plants a BROADCAST `macro:plant` slot, which only the overlay path can read
# (evolve draws cross_sectional leaves from OHLCV `INPUTS` only). So every MDE row this harness has ever
# produced characterizes the OVERLAY gate — a per-day timing scalar whose power comes from BARS alone —
# and the substrate-power guard has been applying that curve to cross_sectional substrates too.
#
# That matters because the two paths have DIFFERENT power laws. An overlay sees T observations; a
# cross-sectional rank-L/S book sees T x N, because averaging the spread across N names shrinks the
# book's noise. A per-name panel can therefore be adequately powered at a bar count where an overlay is
# hopeless — and the guard, which keys on `holdout_bars` alone, cannot see the difference.
#
# The plant here is a per-name CHARACTERISTIC: `x[t, i]` is a persistent standardized score whose LAGGED
# value shifts asset i's next return. A rank-L/S book on x then earns a genuine cross-sectional edge.
# The signal is put in `close` (via the returns it generates) rather than in a feature slot precisely so
# the cross_sectional path can reach it through OHLCV terminals.
def _planted_xsec_panel(t: int, n: int, *, seed: int, beta: float,
                        persistence: float = 0.94) -> tuple[Panel, np.ndarray]:
    """OHLCV panel whose forward returns depend on a LAGGED per-name characteristic. Returns
    ``(panel, x)`` with ``x`` shape ``(t, n)`` standardized cross-sectionally each day.

    ``r[s, i] = beta * x[s-1, i] + noise`` — strictly causal (the characteristic at ``s-1`` drives the
    return earned over ``s-1 -> s``), so a book formed on ``x`` at ``s-1`` captures it without look-ahead.
    ``x`` is an AR(1) with ``persistence`` so it is tradeable at a realistic holding period rather than
    being fresh noise each bar. ``beta=0`` collapses to a clean null (x independent of returns)."""
    rng = np.random.default_rng(seed)
    x = np.empty((t, n), dtype=np.float64)
    x[0] = rng.standard_normal(n)
    sd_innov = np.sqrt(max(1e-12, 1.0 - persistence ** 2))
    for s in range(1, t):
        x[s] = persistence * x[s - 1] + sd_innov * rng.standard_normal(n)
    x -= x.mean(axis=1, keepdims=True)                     # cross-sectionally demeaned each day
    sd = x.std(axis=1, keepdims=True)
    x = np.divide(x, np.where(sd > 0, sd, 1.0))            # ... and standardized
    r = 0.010 * rng.standard_normal((t, n))
    r[1:] += beta * x[:-1]                                 # LAGGED characteristic drives next return
    close = np.exp(np.cumsum(r, axis=0) + rng.uniform(3.0, 5.0, size=n))
    open_ = close * (1 + 0.001 * rng.standard_normal((t, n)))
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    vol = rng.uniform(1e6, 1e8, (t, n))
    dates = (np.datetime64("2010-01-04") + np.arange(t) * np.timedelta64(1, "D")
             ).astype("datetime64[ns]")
    return (Panel(dates, tuple(f"S{i:02d}" for i in range(n)), open_, high, low, close, vol,
                  np.ones((t, n), bool), close * vol, rng.integers(0, 4, size=n),
                  {"survivorship_free": True, "source": "synthetic_planted_xsec"},
                  feature_slots={}), x)


def _xsec_candidate_returns(x: np.ndarray, panel: Panel, *, hold_horizon: int, cost_bps: float,
                            min_names: int) -> tuple[np.ndarray, float]:
    """The ORACLE cross-sectional book on the planted characteristic — the same daily-marked,
    rebalance-and-hold rank-L/S construction ``evolve._candidate_returns`` builds from a DSL genome
    (:func:`finrl_pro_ds.signals.eval_harness._ls_weights`), fed the plant directly.

    Using the oracle rather than a mined genome is deliberate and mirrors how the OVERLAY curve is
    measured (it scores ``macro:plant`` itself, not a search result): the question is the GATE's
    detection floor at a given substrate size, so the signal must be handed over cleanly. Any real
    genome is weaker, so every MDE row here is a LOWER bound on the deployed detection floor — the
    conservative direction for a guard that refuses when MDE is too high."""
    from finrl_pro_ds.signals.eval_harness import _ls_weights

    fwd1 = panel.forward_returns(1)
    T = panel.T
    rets = np.full(T, np.nan)
    turns: list[float] = []
    w = np.zeros(panel.N)
    for s in range(T - 1):
        if s % hold_horizon == 0:
            w_new = _ls_weights(x[s], panel.active[s], min_names=min_names)
            turns.append(float(np.abs(w_new - w).sum()))
            w = w_new
        rets[s] = float(np.nansum(w * fwd1[s])
                        - cost_bps * (turns[-1] if s % hold_horizon == 0 and turns else 0.0))
    ppy = 252.0 / hold_horizon
    return rets, (float(np.mean(turns) * ppy) if turns else 0.0)


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


def run_e1(cc: CalibConfig, *, quick: bool, null_kind: str = "iid",
           null_params: dict | None = None, corrected: CorrectedConfig | None = None,
           n_panels_override: int | None = None) -> dict:
    """Null false-positive rate driven THROUGH the real search (``evolve``), not through a single-shot
    scorer — which is the only way the FILE-DRAWER effect enters.

    ``corrected`` (crucible-v6.0) runs the search under the corrected contract. This measurement is
    load-bearing for that bump: the shipped funnel's defence against search multiplicity was ``dsr_aug``
    deflating the augmented-book Sharpe by ``gen_n_eff`` (every genome ever scored). The corrected
    contract DROPS that leg, and its remaining multiplicity control — the LORD++ p-gate — charges one
    test per PRE-REGISTERED spec, while a GP search evaluates hundreds of evolved offspring that charge
    nothing. So the per-candidate E1 measured by ``crucible_corrected_contract.py`` (one hypothesis at a
    time) does NOT bound the FPR of the same contract inside a search. Run this before trusting it."""
    sub = cc.substrate
    e1 = cc.raw["e1_null"]
    n_panels = 6 if quick else int(e1["n_panels"])
    if n_panels_override:
        n_panels = int(n_panels_override)
    alpha = float(e1["ci_alpha"])
    t, n, n_slots = int(sub["t"]), int(sub["n"]), int(sub["n_feature_slots"])
    cs_seeds = [FORMULAS[i] for i in _CS_SEED_NUMS]
    ov_seeds = list(_OVERLAY_NULL_SEEDS)
    # F4 (audit F14): swap the IID-Gaussian null for a realistic one (fat tails + vol clustering +
    # common factor) to re-measure the funnel's FPR when the mis-specified marginal_t/dsr legs face
    # a null they were never calibrated against. OHLCV/slot construction is otherwise identical.
    panel_gen = _realistic_noise_panel if null_kind == "realistic" else _noise_panel

    promising_total = 0
    genomes_total = 0
    ticks_with_promising = 0
    holdout_evals = 0
    gen_n_effs: list[float] = []
    legs = LegTally()

    log.info("E1 null-calibration [%s null]: %d noise panels x {cross_sectional[%d] + overlay[%d]} seeds",
             null_kind, n_panels, len(cs_seeds), len(ov_seeds))
    # null_params only reshapes the realistic generator; the IID generator ignores it (frozen baseline).
    extra = {"null_params": null_params} if null_kind == "realistic" else {}
    for k in range(n_panels):
        panel = panel_gen(t, n, seed=1000 + k, n_feature_slots=n_slots, **extra)
        base = _proxy_base_sleeves(panel, hold=cc.ek["hold_horizon"])
        ts = _panel_ts(panel)
        n_prom_this = 0
        ckw = ({"contract": "corrected", "corrected_cfg": corrected} if corrected is not None else {})
        for ct, seeds in (("cross_sectional", cs_seeds), ("overlay", ov_seeds)):
            rep = evolve(seeds, panel, base, ts, cc.fit_cfg, candidate_type=ct, **cc.ek, **ckw)
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
        "null_kind": null_kind,
        "contract": ("corrected" if corrected is not None else "shipped"),
        "null_params": dict(null_params) if null_params else {},   # provenance for the shape sweep
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
        # M1 diagnostic: which leg binds under the null. NOTE this tally always reads the SHIPPED legs
        # off the TRAIN FitnessResult (`LegTally.observe`), because that is the only per-genome record
        # `evolve` surfaces. Under --contract corrected it therefore describes the legs that are NO
        # LONGER decision-bearing, and its `all_pass` is NOT the corrected FPR — the corrected decision
        # lives on the holdout, in `report.holdout_validation`. Flagged rather than dropped: the shipped
        # legs' null behaviour is still the useful contrast.
        "per_leg_null_pass_rate": legs.rates(),
        "per_leg_denom": legs.denom,
        "per_leg_is_shipped_train_legs": True,
        "verdict": "GREEN" if e1_green else "RED",
    }


# ------------------------------------------------- E1 sensitivity sweep (F4 robustness, cont-138)
# Null-shape knobs the sweep is allowed to vary (a subset of _realistic_returns' signature). Anything
# else in a point row (label, comments) is ignored; unknown numeric keys would be a config typo, so
# we whitelist explicitly rather than forward the whole row.
_NULL_SHAPE_KEYS = ("df", "factor_share", "garch_alpha", "garch_beta", "target_vol",
                    "load_mean", "load_sd")


def _with_e1_panels(cc: CalibConfig, n_panels: int) -> CalibConfig:
    """A shallow copy of ``cc`` whose ``e1_null.n_panels`` is overridden for a sweep sub-run — lets each
    sensitivity point use the (smaller) sweep budget without mutating the shared config. Gate THRESHOLDS
    (fit_cfg, ceilings) are untouched, so CRU-1 is safe."""
    raw = dict(cc.raw)
    raw["e1_null"] = {**cc.raw["e1_null"], "n_panels": int(n_panels)}
    return CalibConfig(raw=raw, fit_cfg=cc.fit_cfg, ek=cc.ek)


def run_e1_sensitivity(cc: CalibConfig, *, quick: bool) -> dict:
    """Sensitivity of the F4 realistic-null E1 verdict to the null-SHAPE knobs (``df`` = Student-t tail
    fatness; ``factor_share`` = cross-sectional common-factor share of variance). F4 measured E1 at ONE
    default point (df=5, factor_share=0.35); this re-runs E1 (realistic null) over the ``e1_sensitivity``
    grid and re-applies the UNCHANGED ``e1_null`` ceilings at each point. GREEN iff EVERY point clears
    them — the funnel's false-positive protection is a property of the GATE, not of the particular null
    shape. Introduces no new threshold (reuses ``null_fpr_max`` / ``null_tick_fpr_max``)."""
    sens = cc.raw["e1_sensitivity"]
    points = list(sens["points"])
    n_panels = 6 if quick else int(sens["n_panels"])
    cand_ceiling = float(cc.raw["e1_null"]["null_fpr_max"])
    tick_ceiling = float(cc.raw["e1_null"]["null_tick_fpr_max"])
    sub = _with_e1_panels(cc, n_panels)

    log.info("E1 sensitivity sweep: %d null-shape points x %d panels (ceilings cand<=%.3g, tick<=%.3g)",
             len(points), n_panels, cand_ceiling, tick_ceiling)
    rows: list[dict[str, Any]] = []
    for pt in points:
        label = str(pt.get("label", ""))
        null_params = {k: float(pt[k]) for k in _NULL_SHAPE_KEYS if k in pt}
        e1 = run_e1(sub, quick=quick, null_kind="realistic", null_params=null_params)
        rows.append({
            "label": label,
            "null_params": null_params,
            "promising_total": e1["promising_total"],
            "genomes_scored_total": e1["genomes_scored_total"],
            "per_candidate_fpr": e1["per_candidate_fpr"],
            "per_candidate_fpr_cp_upper95": e1["per_candidate_fpr_cp_upper95"],
            "per_tick_fpr": e1["per_tick_fpr"],
            "per_tick_fpr_cp_upper95": e1["per_tick_fpr_cp_upper95"],
            "per_leg_null_pass_rate": e1["per_leg_null_pass_rate"],
            "verdict": e1["verdict"],
        })
        log.info("  %-16s df=%.1f fs=%.2f: promising=%d, candFPR CP<=%.4f, tickFPR CP<=%.3f -> %s",
                 label, null_params.get("df", float("nan")), null_params.get("factor_share", float("nan")),
                 e1["promising_total"], e1["per_candidate_fpr_cp_upper95"],
                 e1["per_tick_fpr_cp_upper95"], e1["verdict"])

    all_green = all(r["verdict"] == "GREEN" for r in rows)
    worst_cand = max((r["per_candidate_fpr_cp_upper95"] for r in rows), default=float("nan"))
    worst_tick = max((r["per_tick_fpr_cp_upper95"] for r in rows), default=float("nan"))
    return {
        "n_panels": n_panels,
        "n_points": len(points),
        "cand_ceiling": cand_ceiling,
        "tick_ceiling": tick_ceiling,
        "rows": rows,
        "worst_per_candidate_fpr_cp_upper95": worst_cand,
        "worst_per_tick_fpr_cp_upper95": worst_tick,
        "all_points_green": all_green,
        "verdict": "GREEN" if all_green else "RED",
        "note": ("F4 robustness panel: E1 re-run under realistic nulls of varying tail-fatness (df) and "
                 "cross-sectional correlation (factor_share), reusing the frozen e1_null ceilings. GREEN "
                 "iff FP protection holds across ALL null shapes, not only the default (df=5, fs=0.35)."),
    }


# ------------------------------------------------------------------ E2: planted-signal power
def _e2_power_curve(cc: CalibConfig, *, t: int, n: int, betas: list[float], n_seeds: int,
                    gen_n_eff: float, cost_bps: float, alpha: float = 0.05,
                    corrected: CorrectedConfig | None = None,
                    holdout_frac: float | None = None) -> list[dict[str, Any]]:
    """The E2 power curve at a FIXED panel length ``t``: for each plant strength ``beta``, score the
    planted overlay ``macro:plant`` over ``n_seeds`` panels and record power + mean realized marginal
    ΔSR (+ per-leg pass-rates under the shipped gate). Shared by ``run_e2`` (single T) and
    ``run_mde_sweep`` (T grid).

    Two contracts, and they are scored on DIFFERENT WINDOWS on purpose:

    * ``corrected is None`` — the SHIPPED gate ``combination_fitness(...).passes_gate``, scored on the
      FULL panel. This is the frozen v5.0 convention and is left byte-stable: the shipped
      ``calibration_mde_sweep.json`` is the provenance the v5.0 power guard reads, so it must not move.
      Note the known mismatch it carries: full-panel power EXCEEDS the deployed holdout-binding power,
      so the recorded MDE is an UNDER-statement (documented in ``run_e2`` as "an upper bound on power").
      Harmless while every MDE sits far above the guard's ceiling; anti-conservative the moment one
      does not.
    * ``corrected is not None`` — the crucible-v6.0 contract ``corrected_contract_fitness(...)
      .passes_corrected``, scored on the LAST ``_power_holdout_bars(t, holdout_frac)`` rows. That is
      exactly the window ``evolve`` takes its binding decision on, and exactly the ``holdout_bars`` the
      power guard interpolates against — so the new sweep measures the gate's REAL N and cannot
      under-state MDE the way the full-panel convention does. The overlay is still computed on the full
      panel first, so trailing-window operators warm up from (causal, past) train history, mirroring
      ``evolve``'s own holdout re-score.

    The LORD++ level is a FRESH account's first level (``fresh_lord_level``), the same per-candidate
    convention ``crucible_corrected_contract.py`` uses: a power curve is one hypothesis at a time, so
    the stream has not yet decayed."""
    lord = fresh_lord_level(corrected) if corrected is not None else float("nan")
    hb = (_power_holdout_bars(t, float(holdout_frac if holdout_frac is not None else 0.25))
          if corrected is not None else t)
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
            if corrected is not None:
                # slice to the binding holdout window (candidate warmed up on the full panel above)
                cand_ho = cand[t - hb:]
                base_ho = {k2: np.asarray(v)[t - hb:] for k2, v in base.items()}
                cr = corrected_contract_fitness(cand_ho, base_ho, ts[t - hb:], cc.fit_cfg,
                                                corrected, lord_level=lord)
                if cr.passes_corrected:
                    detections += 1
                if np.isfinite(cr.delta_sr):
                    realized_deltas.append(float(cr.delta_sr))
                continue
            res = combination_fitness(cand, base, ts, cc.fit_cfg, gen_n_eff=gen_n_eff,
                                      turnover_ann=turnover, n_nodes=1)
            legs.observe(res, cc.fit_cfg)
            if res.passes_gate:
                detections += 1
            if np.isfinite(res.delta_sr_oos):
                realized_deltas.append(float(res.delta_sr_oos))
        power = detections / max(1, n_seeds)
        mean_delta = float(np.mean(realized_deltas)) if realized_deltas else float("nan")
        row: dict[str, Any] = {
            "beta": beta, "power": power, "detections": detections, "n_seeds": n_seeds,
            "power_cp_lower95": clopper_pearson_lower(detections, n_seeds, alpha),
            "mean_realized_delta_sr": mean_delta,
        }
        if corrected is None:
            row["per_leg_pass_rate"] = legs.rates()       # which leg blocks near the MDE
        else:
            row["scored_bars"] = int(hb)
            row["lord_level"] = float(lord)
        curve.append(row)
    return curve


def _xsec_power_curve(cc: CalibConfig, *, t: int, n: int, betas: list[float], n_seeds: int,
                      cost_bps: float, corrected: CorrectedConfig, holdout_frac: float,
                      hold_horizon: int, min_names: int,
                      alpha: float = 0.05) -> list[dict[str, Any]]:
    """The CROSS-SECTIONAL power curve at a fixed ``(t, n)``: plant a per-name characteristic, build the
    oracle rank-L/S book on it, and score its marginal contribution through the corrected contract on the
    BINDING holdout window — the cross-sectional analogue of :func:`_e2_power_curve`.

    ``n`` is the axis the overlay curve cannot see. A cross-sectional book averages its spread over ``n``
    names, so its noise falls with breadth while ``holdout_bars`` stays fixed; measuring along ``n`` is
    the whole point of this sweep."""
    lord = fresh_lord_level(corrected)
    hb = _power_holdout_bars(t, holdout_frac)
    curve: list[dict[str, Any]] = []
    for beta in betas:
        detections = 0
        realized: list[float] = []
        for k in range(n_seeds):
            panel, x = _planted_xsec_panel(t, n, seed=7000 + k, beta=beta)
            base = _proxy_base_sleeves(panel, hold=hold_horizon)
            ts = _panel_ts(panel)
            cand, _turn = _xsec_candidate_returns(x, panel, hold_horizon=hold_horizon,
                                                  cost_bps=cost_bps, min_names=min_names)
            cand_ho = cand[t - hb:]
            base_ho = {k2: np.asarray(v)[t - hb:] for k2, v in base.items()}
            cr = corrected_contract_fitness(cand_ho, base_ho, ts[t - hb:], cc.fit_cfg, corrected,
                                            lord_level=lord)
            if cr.passes_corrected:
                detections += 1
            if np.isfinite(cr.delta_sr):
                realized.append(float(cr.delta_sr))
        curve.append({
            "beta": beta, "power": detections / max(1, n_seeds), "detections": detections,
            "n_seeds": n_seeds,
            "power_cp_lower95": clopper_pearson_lower(detections, n_seeds, alpha),
            "mean_realized_delta_sr": float(np.mean(realized)) if realized else float("nan"),
            "scored_bars": int(hb), "lord_level": float(lord),
        })
    return curve


def run_xsec_mde_sweep(cc: CalibConfig, *, quick: bool, corrected: CorrectedConfig) -> dict:
    """The MDE SURFACE over (bars x breadth) for the CROSS-SECTIONAL path.

    Why this exists (V3, 2026-07-29). Every MDE row the harness produced before this was measured on the
    OVERLAY path, whose only power axis is bars — yet the substrate-power guard applied that one curve to
    every substrate. A per-name ``(T, N)`` panel carries roughly ``N x`` the effective observations of a
    broadcast one at the SAME ``holdout_bars``, so the guard was structurally unable to see the payoff of
    per-name data (audit U3). This sweep measures the missing axis so it can.

    Emitted in the same ``rows`` schema the guard interpolates, plus an ``n`` on every row and an
    ``n_grid``; the consumer selects the largest measured ``n`` that does not EXCEED the substrate's
    breadth (a wider panel has more power, so a lower-``n`` curve over-states MDE — the conservative
    direction) and refuses outright below the smallest measured ``n``."""
    e2 = cc.raw["e2_power"]
    sw = cc.raw["xsec_mde_sweep"]
    cost_bps = float(cc.ek["cost_bps"])
    holdout_frac = float(cc.ek["holdout_frac"])
    hold_horizon = int(cc.ek["hold_horizon"])
    min_names = int(sw.get("ls_min_names", cc.ek.get("ls_min_names", 6)))
    power_target = float(e2["power_target"])
    n_seeds = 8 if quick else int(sw["n_seeds"])
    betas = [0.0, 0.001, 0.004] if quick else [float(b) for b in sw["beta_grid"]]
    t_grid = [1512, 4044] if quick else [int(x) for x in sw["t_grid"]]
    n_grid = [12, 50] if quick else [int(x) for x in sw["n_grid"]]

    log.info("XSEC MDE surface [corrected]: T in %s x N in %s x %d betas x %d seeds",
             t_grid, n_grid, len(betas), n_seeds)
    rows: list[dict[str, Any]] = []
    for n in n_grid:
        for t in t_grid:
            curve = _xsec_power_curve(cc, t=t, n=n, betas=betas, n_seeds=n_seeds, cost_bps=cost_bps,
                                      corrected=corrected, holdout_frac=holdout_frac,
                                      hold_horizon=hold_horizon, min_names=min_names)
            mde = _mde_from_curve(curve, power_target)
            rows.append({
                "t": t, "n": n, "holdout_bars": _power_holdout_bars(t, holdout_frac),
                "mde_realized_delta_sr": (float(mde["mean_realized_delta_sr"]) if mde else None),
                "mde_beta": (mde["beta"] if mde else None),
                "mde_power": (float(mde["power"]) if mde else None),
                "max_power": max((float(p["power"]) for p in curve), default=0.0),
                "curve": [{"beta": p["beta"], "power": p["power"],
                           "realized_delta_sr": p["mean_realized_delta_sr"]} for p in curve],
            })
            log.info("  N=%-4d T=%-6d (holdout %d): MDE=%s @power=%s", n, t,
                     rows[-1]["holdout_bars"], rows[-1]["mde_realized_delta_sr"],
                     rows[-1]["mde_power"])
    return {
        "contract": "corrected", "candidate_type": "cross_sectional",
        "t_grid": t_grid, "n_grid": n_grid, "beta_grid": betas, "n_seeds": n_seeds,
        "power_target": power_target, "holdout_frac": holdout_frac,
        "hold_horizon": hold_horizon, "ls_min_names": min_names, "rows": rows,
        "corrected_thresholds": {"t_min": corrected.t_min, "n_eff_mode": corrected.n_eff_mode,
                                 "p_value_model": corrected.p_value_model,
                                 "fdr_binding": corrected.fdr_binding,
                                 "uplift_min": corrected.uplift_min},
        "note": ("MDE surface over bars x BREADTH for the CROSS-SECTIONAL path under the corrected "
                 "contract, scored on the binding holdout window. The candidate is the ORACLE book on "
                 "the planted characteristic, so each row is a LOWER bound on the deployed detection "
                 "floor (any mined genome is weaker) — the conservative direction for a refuse-gate. "
                 "Consumers must select the largest measured n NOT EXCEEDING the substrate's breadth."),
    }


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


def run_mde_sweep(cc: CalibConfig, *, quick: bool, corrected: CorrectedConfig | None = None) -> dict:
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

    contract = "corrected" if corrected is not None else "shipped"
    log.info("MDE-vs-T sweep [%s contract]: T in %s x %d betas x %d seeds",
             contract, t_grid, len(betas), n_seeds)
    rows: list[dict[str, Any]] = []
    for t in t_grid:
        curve = _e2_power_curve(cc, t=t, n=n, betas=betas, n_seeds=n_seeds, gen_n_eff=gen_n_eff,
                                cost_bps=cost_bps, corrected=corrected, holdout_frac=holdout_frac)
        mde = _mde_from_curve(curve, power_target)
        # ``holdout_bars`` is the key the power guard interpolates on, so it MUST be computed by the
        # orchestrator's own `_power_holdout_bars` (T - int(T*(1-frac))) — not a rounded product — or a
        # substrate would look up its MDE at a grid point one bar off and silently miss the `grid` mode.
        row = {
            "t": t, "holdout_bars": _power_holdout_bars(t, holdout_frac),
            "mde_realized_delta_sr": (float(mde["mean_realized_delta_sr"]) if mde else None),
            "mde_beta": (mde["beta"] if mde else None),
            "mde_power": (float(mde["power"]) if mde else None),
            "max_power": max((float(p["power"]) for p in curve), default=0.0),
            "curve": [{"beta": p["beta"], "power": p["power"],
                       "realized_delta_sr": p["mean_realized_delta_sr"]} for p in curve],
        }
        rows.append(row)
        log.info("  T=%d (holdout %d bars): MDE realized_dSR=%s @ power=%s",
                 t, row["holdout_bars"], row["mde_realized_delta_sr"], row["mde_power"])

    # Physical expectation: MDE FALLS as T grows (more data -> smaller detectable edge, ~1/sqrt(T)).
    mdes = [r["mde_realized_delta_sr"] for r in rows if r["mde_realized_delta_sr"] is not None]
    monotone = all(mdes[i] >= mdes[i + 1] - 0.5 for i in range(len(mdes) - 1))  # slack for MC noise
    note = ("MDE = realized marginal ΔSR (annualized) at power_target, full-panel SHIPPED gate (UPPER "
            "bound on deployed holdout-binding power). Expect MDE ~ 1/sqrt(T_holdout).")
    if corrected is not None:
        note = ("MDE = realized marginal ΔSR (annualized) at power_target under the crucible-v6.0 "
                "CORRECTED contract (JKM Sharpe-difference z >= t_min + binding LORD++), scored on the "
                "BINDING holdout window (holdout_bars), i.e. the gate's real N — not the full panel. "
                "Read by the substrate-power guard when the substrate's contract is 'corrected'.")
    return {
        "t_grid": t_grid, "n_seeds": n_seeds, "beta_grid": betas,
        "contract": contract,
        "power_target": power_target, "gate_probe_gen_n_eff": gen_n_eff,
        "holdout_frac": holdout_frac, "rows": rows,
        "mde_falls_with_t": monotone,
        "corrected_thresholds": (None if corrected is None else
                                 {"t_min": corrected.t_min, "n_eff_mode": corrected.n_eff_mode,
                                  "p_value_model": corrected.p_value_model,
                                  "fdr_binding": corrected.fdr_binding,
                                  "uplift_min": corrected.uplift_min}),
        "note": note,
    }


# ------------------------------------------------------------------ driver
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="Crucible funnel calibration: E1 null-FPR + E2 power")
    ap.add_argument("--exp", choices=("e1", "e2", "both", "mde_sweep", "xsec_mde_sweep",
                                      "e1_sensitivity"),
                    default="both",
                    help="e1/e2/both = calibration verdicts; mde_sweep = the MDE-vs-T detection "
                         "floor across substrate lengths; e1_sensitivity = re-run E1 under realistic "
                         "nulls of varying tail-fatness/correlation (F4 robustness sweep).")
    ap.add_argument("--calib-config", default=str(DEFAULT_CALIB_GATES))
    ap.add_argument("--config", default=str(DEFAULT_FUNNEL_GATES),
                    help="funnel gates YAML — its generation block supplies the REAL FitnessConfig "
                         "thresholds being calibrated (default signal_eval.gates.yaml).")
    ap.add_argument("--out", default=str(ROOT / "results" / "crucible_calibration"))
    ap.add_argument("--quick", action="store_true",
                    help="tiny K / short beta grid — smoke test the harness end-to-end, NOT a verdict.")
    ap.add_argument("--null", choices=("iid", "realistic"), default="iid",
                    help="E1 null generator: iid = Gaussian random walk (default, the frozen baseline); "
                         "realistic = fat tails + vol clustering + common factor (audit F4). E2/mde_sweep "
                         "use planted panels and are unaffected.")
    ap.add_argument("--contract", choices=("shipped", "corrected"), default="shipped",
                    help="which DECISION contract to characterize (applies to e1 and mde_sweep; e2 and "
                         "e1_sensitivity are shipped-only). shipped = the "
                         "historical full-panel combination_fitness gate (the frozen v5.0 curve the "
                         "power guard reads today). corrected = the crucible-v6.0 contract scored on "
                         "the BINDING holdout window; writes calibration_mde_sweep_corrected.json, "
                         "which the power guard reads for contract='corrected' substrates.")
    ap.add_argument("--corrected-config", default=str(DEFAULT_CORRECTED_GATES),
                    help="--contract corrected only: the corrected contract's thresholds YAML.")
    ap.add_argument("--e1-panels", type=int, default=None,
                    help="override e1_null.n_panels (EXPERIMENTAL DESIGN, not a gate — the ceilings in "
                         "the gates file are untouched). Raising it TIGHTENS the Clopper-Pearson bound "
                         "in both directions, so it can only make a true over-ceiling FPR easier to "
                         "detect; at n=60 the per-tick bound is 0.0487 even at k=0, leaving almost no "
                         "headroom under the 0.05 ceiling for a single false positive.")
    args = ap.parse_args()

    cc = load_calib(Path(args.calib_config), Path(args.config))
    corrected = (CorrectedConfig.from_yaml(args.corrected_config)
                 if args.contract == "corrected" else None)
    ts = datetime.now(timezone.utc).isoformat()
    report: dict = {
        "harness_version": cc.raw.get("harness_version"),
        "ts": ts, "quick": args.quick, "exp": args.exp, "null_kind": args.null,
        "funnel_gates": str(args.config),
        "funnel_thresholds": {
            "promising_dsr": cc.fit_cfg.promising_dsr, "hlz_t_min": cc.fit_cfg.hlz_t_min,
            "min_combination_uplift": cc.fit_cfg.min_combination_uplift,
            "max_base_corr": cc.fit_cfg.max_base_corr},
        "search_budget": {k: cc.ek[k] for k in ("pop_size", "n_generations", "rng_seed")},
    }
    if args.exp in ("e1", "both"):
        report["e1"] = run_e1(cc, quick=args.quick, null_kind=args.null, corrected=corrected,
                              n_panels_override=args.e1_panels)
    if args.exp in ("e2", "both"):
        report["e2"] = run_e2(cc, quick=args.quick)
    if args.exp == "mde_sweep":
        report["contract"] = args.contract
        report["mde_sweep"] = run_mde_sweep(cc, quick=args.quick, corrected=corrected)
    if args.exp == "xsec_mde_sweep":
        if corrected is None:
            ap.error("--exp xsec_mde_sweep requires --contract corrected (the cross-sectional surface "
                     "is only defined for the v6.0 contract; the shipped legs are sealed)")
        report["contract"] = args.contract
        # Emitted under the SAME `mde_sweep` key the power guard reads, so one consumer serves both
        # curves; `candidate_type` inside distinguishes them.
        report["mde_sweep"] = run_xsec_mde_sweep(cc, quick=args.quick, corrected=corrected)
    if args.exp == "e1_sensitivity":
        report["e1_sensitivity"] = run_e1_sensitivity(cc, quick=args.quick)

    # Joint verdict (M4): GREEN only if BOTH pass (when both ran). mde_sweep is a characterization (no
    # pass/fail -> N/A); e1_sensitivity is itself a pass/fail so it becomes the run's verdict.
    verdicts = [report[e]["verdict"] for e in ("e1", "e2") if e in report]
    if cc.raw.get("joint", {}).get("require_both", True) and len(verdicts) == 2:
        report["joint_verdict"] = "GREEN" if all(v == "GREEN" for v in verdicts) else "RED"
    elif "e1_sensitivity" in report:
        # A self-contained pass/fail (F4 robustness across null shapes); it IS the run's verdict.
        report["joint_verdict"] = report["e1_sensitivity"]["verdict"]
    else:
        report["joint_verdict"] = verdicts[0] if len(verdicts) == 1 else "N/A"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "calibration_quick" if args.quick else "calibration"
    null_sfx = "" if args.null == "iid" else f"_{args.null}"
    # The corrected sweep is a SEPARATE artifact — the shipped curve stays byte-stable because it is
    # the provenance the v5.0 power guard reads for shipped-contract substrates.
    contract_sfx = ("_corrected" if (args.exp in ("mde_sweep", "xsec_mde_sweep")
                                     and args.contract == "corrected") else "")
    out_path = out_dir / f"{stem}_{args.exp}{null_sfx}{contract_sfx}.json"
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
        is_xsec = sw.get("candidate_type") == "cross_sectional"
        label = "MDE surface (bars x BREADTH)" if is_xsec else "MDE-vs-T"
        print(f"{label} (detection floor @ power>={sw['power_target']}; realized marginal dSR):")
        head = f"   {'T(bars)':>8} {'holdout':>8} {'MDE_dSR':>9} {'@power':>7} {'@beta':>8}"
        print((f"   {'N':>5}" + head[3:]) if is_xsec else head)
        for r in sw["rows"]:
            mde = r["mde_realized_delta_sr"]
            body = (f"{r['t']:>8} {r['holdout_bars']:>8} "
                    f"{('%.2f' % mde) if mde is not None else 'undetected':>9} "
                    f"{('%.2f' % r['mde_power']) if r['mde_power'] is not None else '-':>7} "
                    f"{r['mde_beta'] if r['mde_beta'] is not None else '-':>8}")
            print(f"   {r['n']:>5} {body}" if is_xsec else f"   {body}")
        if "mde_falls_with_t" in sw:
            print(f"   MDE falls with T (expected ~1/sqrt(T)): {sw['mde_falls_with_t']}")
        if is_xsec:
            print("   NOTE: a row whose mde_beta is the grid's FIRST non-zero point is GRID-PINNED — "
                  "the true MDE is lower. Densify beta_grid and re-read before using it.")
    if "e1_sensitivity" in report:
        sw = report["e1_sensitivity"]
        print(f"E1 SENSITIVITY (F4 robustness; ceilings candFPR<={sw['cand_ceiling']}, "
              f"tickFPR<={sw['tick_ceiling']}; {sw['n_panels']} panels/point):")
        print(f"   {'point':<16} {'df':>5} {'fac_sh':>7} {'prom':>5} {'candFPR_CP':>11} "
              f"{'tickFPR_CP':>11} {'verdict':>8}")
        for r in sw["rows"]:
            p = r["null_params"]
            print(f"   {r['label']:<16} {p.get('df', float('nan')):>5.1f} "
                  f"{p.get('factor_share', float('nan')):>7.2f} {r['promising_total']:>5} "
                  f"{r['per_candidate_fpr_cp_upper95']:>11.4f} {r['per_tick_fpr_cp_upper95']:>11.3f} "
                  f"{r['verdict']:>8}")
        print(f"   worst candFPR CP-upper95 = {sw['worst_per_candidate_fpr_cp_upper95']:.4f} | "
              f"ALL-GREEN across {sw['n_points']} shapes = {sw['all_points_green']}")
    print(f"JOINT VERDICT : {report['joint_verdict']}")
    print(f"report -> {out_path}")
    print("==============================================================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
