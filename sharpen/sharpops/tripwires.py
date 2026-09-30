"""Artifact tripwires: hard gates at every promotion rung (Protocol v2 audit 2026-09-29 §6 item 2).

Every artifact on this project's record (the X2 coarse-bar leak, frictionless HPO, bid-ask
bounce, seeds that never reached the envs) would have tripped at least one of these. They ask
"is this number believable?", not "is it big enough?". Each returns ``{"pass": bool, ...}``;
thresholds come from ``configs/sharpops_ladder.gates.yaml`` via the caller, never from here.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from statistics import NormalDist

import numpy as np


def wilson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided Wilson score interval for k of n; (0, 1) when n == 0 (no information)."""
    if n <= 0:
        return 0.0, 1.0
    if not 0 <= k <= n:
        raise ValueError(f"k={k} outside [0, {n}]")
    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    p = k / n
    den = 1.0 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, ctr - hw), min(1.0, ctr + hw)


def _sr(r: np.ndarray, ppy: int) -> float:
    sd = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    return float(np.mean(r) / sd * np.sqrt(ppy)) if sd > 0 else 0.0


def plausibility_ceiling(sharpe_ann: float, ceiling: float) -> dict:
    """A net Sharpe above the ceiling is presumed an artifact until an audit clears it.
    Non-finite fails."""
    ok = bool(np.isfinite(sharpe_ann) and sharpe_ann <= ceiling)
    return {"pass": ok, "sharpe_ann": float(sharpe_ann), "ceiling": ceiling,
            "reason": None if ok else "above the plausibility ceiling: presumed artifact"}


def cost_sign(sharpe_frictionless: float, sharpe_net: float, *, tol: float = 1e-9) -> dict:
    """Costs can only lower a book's Sharpe. Net above frictionless means the cost leg has the
    wrong sign or was double-applied to the benchmark: an artifact, not an edge."""
    ok = bool(sharpe_net <= sharpe_frictionless + tol)
    return {"pass": ok, "sharpe_frictionless": float(sharpe_frictionless),
            "sharpe_net": float(sharpe_net), "gap": float(sharpe_frictionless - sharpe_net)}


def seed_tripwire(by_seed: Mapping[object, Sequence[float]],
                  rerun: tuple[Sequence[float], Sequence[float]] | None = None) -> dict:
    """Seeds must DIVERGE (different seeds, different paths) and a same-seed rerun must
    REPRODUCE. Identical paths across seeds is the SEED-01 class: the seed never reached the
    env, so a "multiseed" result is one sample repeated."""
    paths = [np.asarray(v, np.float64) for v in by_seed.values()]
    if len(paths) < 2:
        return {"pass": False, "reason": "need >= 2 seeds"}
    identical = [(i, j) for i in range(len(paths)) for j in range(i + 1, len(paths))
                 if paths[i].shape == paths[j].shape and np.array_equal(paths[i], paths[j])]
    out = {"n_seeds": len(paths), "identical_pairs": len(identical)}
    reproduces = None
    if rerun is not None:
        a, b = (np.asarray(x, np.float64) for x in rerun)
        reproduces = bool(a.shape == b.shape and np.allclose(a, b, rtol=0, atol=1e-12))
    out["same_seed_reproduces"] = reproduces
    out["pass"] = not identical and reproduces is not False
    return out


def selection_embargo(selection_windows: Sequence[tuple[str, str]],
                      test_windows: Sequence[tuple[str, str]], *, embargo_days: int = 0) -> dict:
    """No window used for SELECTION (HPO val/test, seed ranking, ensemble weights) may overlap
    a walk-forward TEST window, widened by ``embargo_days`` on each side. Windows are
    half-open [start, end). Confirmed breach 2026-09-30: SG-1-BTC's HPO val 2025-07-01..
    2025-10-01 contains WF folds 0-1's test windows (decay01 schedule)."""
    import pandas as pd

    pad = pd.Timedelta(days=int(embargo_days))
    hits = []
    for s0, s1 in selection_windows:
        a0, a1 = pd.Timestamp(s0), pd.Timestamp(s1)
        for k, (t0, t1) in enumerate(test_windows):
            b0, b1 = pd.Timestamp(t0) - pad, pd.Timestamp(t1) + pad
            if a0 < b1 and b0 < a1:
                hits.append({"selection": [str(a0.date()), str(a1.date())], "test_fold": k,
                             "test": [str(t0)[:10], str(t1)[:10]]})
    return {"pass": not hits, "overlaps": hits, "embargo_days": int(embargo_days)}


def shuffle_placebo(strategy: Callable[[np.ndarray], np.ndarray], returns: np.ndarray, *,
                    n_perm: int, max_null_sharpe: float, alpha: float,
                    periods_per_year: int = 252, seed: int = 11) -> dict:
    """Run the strategy on time-shuffled returns (no temporal structure left to exploit).

    Two readings, both required:
      * quiet: the median placebo Sharpe must stay <= ``max_null_sharpe``; a book that makes
        money on shuffled data is reading the future or mis-costing;
      * beats: the real Sharpe must exceed the placebo distribution at level ``alpha``
        (p = (1 + #null >= obs) / (1 + n_perm))."""
    r = np.asarray(returns, np.float64)
    obs = _sr(np.asarray(strategy(r), np.float64), periods_per_year)
    rng = np.random.default_rng(seed)
    null = np.array([_sr(np.asarray(strategy(rng.permutation(r)), np.float64), periods_per_year)
                     for _ in range(n_perm)])
    p = (1 + int(np.sum(null >= obs))) / (1 + n_perm)
    quiet = bool(np.median(null) <= max_null_sharpe)
    return {"pass": bool(quiet and p <= alpha), "quiet": quiet, "p_value": p,
            "observed_sharpe": obs, "null_median_sharpe": float(np.median(null)),
            "null_q95_sharpe": float(np.quantile(null, 0.95))}


def circular_shift_null(signal: np.ndarray, fwd_returns: np.ndarray, *, n_shifts: int,
                        min_shift: int, alpha: float, periods_per_year: int = 252,
                        seed: int = 13) -> dict:
    """Timing null for a rule: rotate the (causally aligned) position series against the
    returns it earns. A rotation keeps the signal's own distribution and autocorrelation and
    destroys only its timing, so a real edge must beat most rotations. pnl_t = signal_t * r_t."""
    s, f = np.asarray(signal, np.float64), np.asarray(fwd_returns, np.float64)
    if s.shape != f.shape or s.ndim != 1:
        raise ValueError("signal and returns must be aligned 1-D")
    n = len(s)
    if not 0 < min_shift < n // 2:
        raise ValueError("need 0 < min_shift < n/2")
    obs = _sr(s * f, periods_per_year)
    rng = np.random.default_rng(seed)
    ks = rng.integers(min_shift, n - min_shift + 1, size=n_shifts)
    null = np.array([_sr(np.roll(s, int(k)) * f, periods_per_year) for k in ks])
    p = (1 + int(np.sum(null >= obs))) / (1 + n_shifts)
    return {"pass": bool(p <= alpha), "p_value": p, "observed_sharpe": obs,
            "null_q95_sharpe": float(np.quantile(null, 0.95))}
