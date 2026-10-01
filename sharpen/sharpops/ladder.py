"""Promotion-ladder statistics (docs/sharpops_promotion_standard.md).

Protocol v2 audit 2026-09-29 §6 replaced the bar-level profit-factor gates and the AND-of-units
rules with one pre-registered primary test per rung. This module holds those tests:

  * :func:`pooled_oos_sharpe` — daily net Sharpe pooled over NON-overlapping OOS folds, with
    its PSR (one-sided p against SR 0) and a block-bootstrap CI (items 3-4).
  * :func:`betting_eprocess` — an anytime-valid test of "mean daily net P&L <= 0" for a forward
    paper record (items 1, 9). Valid at every stopping time by Ville's inequality.
  * :func:`non_inferiority` — paired block bootstrap for the RL-overlay gate (item 8).
  * :func:`operating_characteristic` — false-pass rate and power of ANY gate on planted
    signals of known Sharpe (item 5). Gates are calibrated here, never on candidates.

All Sharpe inputs/outputs are annualized with ``periods_per_year`` unless named ``_pp``.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from sharpen.crypto.eval.statistics import block_bootstrap_sharpe_ci, probabilistic_sharpe_ratio


def _sharpe_ann(r: np.ndarray, ppy: int) -> float:
    sd = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    return float(np.mean(r) / sd * np.sqrt(ppy)) if sd > 0 else 0.0


# ------------------------------------------------------------------ pooled OOS (rung test)
def pooled_oos_sharpe(folds: Sequence[pd.Series], *, periods_per_year: int = 252,
                      block: int = 21, n_boot: int = 10_000, seed: int = 7) -> dict:
    """Pool the daily net returns of walk-forward OOS folds and test SR > 0 once.

    Folds must be date-indexed and NON-overlapping (an overlapping pool double-counts days and
    inflates precision): overlap raises. Returns the pooled annualized Sharpe, its PSR against
    0 (per-period form, periods_per_year=1 — the S-10 convention) and a block-bootstrap CI.
    Per-fold Sharpes are returned as diagnostics only."""
    if not folds:
        raise ValueError("no folds")
    clean = [f.dropna().sort_index() for f in folds]
    if any(len(f) == 0 for f in clean):
        raise ValueError("an OOS fold is empty")
    spans = sorted((f.index[0], f.index[-1]) for f in clean)
    for (a0, a1), (b0, _) in zip(spans, spans[1:]):
        if b0 <= a1:
            raise ValueError(f"OOS folds overlap ({a0.date()}..{a1.date()} and {b0.date()}..)")
    pooled = pd.concat(clean).sort_index()
    r = pooled.to_numpy(dtype=np.float64)
    if len(r) < 3:
        raise ValueError("pooled OOS has < 3 days")
    boot = block_bootstrap_sharpe_ci(r, block=block, n_boot=n_boot, periods_per_year=periods_per_year,
                                     seed=seed)
    return {
        "n_folds": len(clean), "n_days": int(len(r)),
        "sharpe_ann": _sharpe_ann(r, periods_per_year),
        "psr_vs_0": float(probabilistic_sharpe_ratio(r.tolist(), sr_benchmark=0.0, periods_per_year=1)),
        "ci95": None if boot is None else [float(boot["ci_low"]), float(boot["ci_high"])],
        "fold_sharpes_diagnostic": [_sharpe_ann(f.to_numpy(dtype=np.float64), periods_per_year)
                                    for f in clean],
    }


def rung_test_passes(psr: float | None, alpha: float) -> bool:
    """One-sided test at level alpha: PSR = P(SR > 0) must reach 1 - alpha. None fails closed."""
    return psr is not None and np.isfinite(psr) and psr >= 1.0 - alpha


# ------------------------------------------------------------------ forward sequential test
def betting_eprocess(daily_net: Sequence[float], *, bound: float, alpha: float,
                     lam_max: float = 0.5, return_path: bool = False) -> dict:
    """Anytime-valid test of H0: E[x_t | past] <= 0, for daily net returns bounded below by
    ``-bound`` (testing by betting; Waudby-Smith & Ramdas 2023).

    With y_t = x_t / bound and a PREDICTABLE bet lambda_t in [0, lam_max] (lam_max < 1), the
    wealth K_t = prod(1 + lambda_t * y_t) is a non-negative supermartingale under H0, so
    P(sup_t K_t >= 1/alpha) <= alpha (Ville) at every stopping time: the record can be checked
    daily without a multiplicity penalty.

    Fails closed:
      * a loss below -bound breaks the supermartingale (1 + lambda*y could go negative) and
        raises: the declared bound was wrong, so the test is void, never "passing";
      * a gain above +bound is CLIPPED to +bound, which only lowers wealth (conservative).
    lambda_t is the aGRAPA-style plug-in mean/(var + mean^2) of y_1..y_{t-1}, clipped to
    [0, lam_max]; it is 0 until two observations exist."""
    x = np.asarray(daily_net, dtype=np.float64)
    if bound <= 0 or not 0 < alpha < 1 or not 0 < lam_max < 1:
        raise ValueError("need bound > 0, 0 < alpha < 1, 0 < lam_max < 1")
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite daily return")
    if np.any(x < -bound):
        raise ValueError(f"a daily loss {x.min():.4f} is below -bound {-bound}: the e-process is void")
    y = np.minimum(x, bound) / bound
    wealth, path, lams = 1.0, [], []
    s1 = s2 = 0.0
    for t, yt in enumerate(y):
        if t >= 2:
            mu = s1 / t
            var = max(s2 / t - mu * mu, 0.0)
            lam = float(np.clip(mu / (var + mu * mu), 0.0, lam_max)) if (var + mu * mu) > 0 else 0.0
        else:
            lam = 0.0
        wealth *= 1.0 + lam * yt
        path.append(wealth)
        lams.append(lam)
        s1 += yt
        s2 += yt * yt
    path_arr = np.asarray(path)
    thr = 1.0 / alpha
    hit = np.flatnonzero(path_arr >= thr)
    out = {"n_days": int(len(x)), "e_value": float(path_arr[-1]) if len(x) else 1.0,
           "max_e_value": float(path_arr.max()) if len(x) else 1.0, "threshold": thr,
           "rejected_h0": bool(hit.size), "first_rejection_day": int(hit[0]) + 1 if hit.size else None,
           "n_clipped_gains": int(np.sum(x > bound))}
    if return_path:
        out["path"], out["bets"] = path, lams
    return out


# ------------------------------------------------------------------ RL overlay gate
def paired_block_bootstrap(a: np.ndarray, b: np.ndarray, stat: Callable[[np.ndarray], float], *,
                           block: int = 21, n_boot: int = 5_000, seed: int = 7) -> np.ndarray:
    """Distribution of stat(a) - stat(b) under a moving-block bootstrap that resamples the SAME
    block indices for both series (preserves their correlation and autocorrelation)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be aligned 1-D series")
    n = len(a)
    if n < 2 * block:
        raise ValueError(f"series of {n} < 2 blocks of {block}")
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    return np.array([stat(a[i]) - stat(b[i]) for i in idx])


def non_inferiority(overlay: np.ndarray, baseline: np.ndarray, *, margin: float, alpha: float,
                    periods_per_year: int = 252, block: int = 21, n_boot: int = 5_000,
                    seed: int = 7) -> dict:
    """RL overlay vs linear core on aligned daily net returns: non-inferior iff the one-sided
    (1 - alpha) lower bound of SR_overlay - SR_baseline exceeds -margin (annualized)."""
    if margin < 0 or not 0 < alpha < 1:
        raise ValueError("need margin >= 0 and 0 < alpha < 1")
    d = paired_block_bootstrap(overlay, baseline, lambda r: _sharpe_ann(r, periods_per_year),
                               block=block, n_boot=n_boot, seed=seed)
    lower = float(np.quantile(d, alpha))
    return {"delta_sharpe": _sharpe_ann(np.asarray(overlay, float), periods_per_year)
            - _sharpe_ann(np.asarray(baseline, float), periods_per_year),
            "lower_bound": lower, "margin": margin, "alpha": alpha,
            "non_inferior": bool(lower > -margin)}


def superiority(overlay: np.ndarray, baseline: np.ndarray, stat: Callable[[np.ndarray], float], *,
                alpha: float, lower_is_better: bool, block: int = 21, n_boot: int = 5_000,
                seed: int = 7) -> dict:
    """Superiority on a precisely measurable secondary (cost, turnover, drawdown): the one-sided
    (1 - alpha) bound of the paired difference must clear 0 in the better direction."""
    d = paired_block_bootstrap(overlay, baseline, stat, block=block, n_boot=n_boot, seed=seed)
    if lower_is_better:
        bound = float(np.quantile(d, 1.0 - alpha))
        ok = bound < 0.0
    else:
        bound = float(np.quantile(d, alpha))
        ok = bound > 0.0
    return {"delta": float(np.median(d)), "bound": bound, "superior": bool(ok),
            "lower_is_better": lower_is_better}


# ------------------------------------------------------------------ planted-signal calibration
def planted_returns(sharpe_ann: float, n_days: int, n_sims: int, *, vol_ann: float = 0.10,
                    ar1: float = 0.0, periods_per_year: int = 252, seed: int = 29) -> np.ndarray:
    """(n_sims, n_days) daily returns with TRUE annualized Sharpe ``sharpe_ann`` (Gaussian,
    optionally AR(1) with coefficient ``ar1``; the stationary variance is held at vol_ann)."""
    if not -1 < ar1 < 1:
        raise ValueError("|ar1| must be < 1")
    rng = np.random.default_rng(seed)
    sd = vol_ann / np.sqrt(periods_per_year)
    mu = sharpe_ann * sd / np.sqrt(periods_per_year)
    e = rng.standard_normal((n_sims, n_days)) * sd * np.sqrt(1 - ar1 * ar1)
    z = np.empty_like(e)
    z[:, 0] = rng.standard_normal(n_sims) * sd
    for t in range(1, n_days):
        z[:, t] = ar1 * z[:, t - 1] + e[:, t]
    return mu + z


def operating_characteristic(gate: Callable[[np.ndarray], bool], *, n_days: int,
                             sharpes: Sequence[float], n_sims: int = 500, target_sharpe: float,
                             **planted_kw) -> dict:
    """Pass rate of ``gate`` (daily returns -> bool) on planted signals at each true Sharpe.
    Reports the false-pass rate at SR 0 and the power at ``target_sharpe`` with Wilson CIs;
    0 and the target are always included. A gate is calibrated only through this."""
    from sharpen.sharpops.tripwires import wilson

    grid = sorted({0.0, float(target_sharpe), *map(float, sharpes)})
    kw = dict(planted_kw)
    base_seed = int(kw.pop("seed", 29))
    rows = {}
    for i, s in enumerate(grid):
        sims = planted_returns(s, n_days, n_sims, seed=base_seed + i, **kw)
        k = int(sum(bool(gate(r)) for r in sims))
        lo, hi = wilson(k, n_sims)
        rows[f"{s:g}"] = {"pass_rate": k / n_sims, "wilson95": [lo, hi], "k": k, "n": n_sims}
    return {"n_days": n_days, "grid": rows, "false_pass_at_sr0": rows["0"],
            "power_at_target": rows[f"{float(target_sharpe):g}"], "target_sharpe": float(target_sharpe)}
