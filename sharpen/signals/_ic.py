"""Information-Coefficient primitives — the predictive-power core of the harness.

Ported (with provenance) from the proven falsification probes so the new harness has
ONE canonical home for cross-sectional rank-IC and the multiple-testing helpers:

  * ``cross_sectional_ic`` generalizes ``_xs_ic`` of
    ``scripts/research/cmgp1_x2_leak_ic_probe_v2.py:53`` (per-bar Spearman across
    assets -> mean IC + IC-IR).
  * ``spearman_ic`` generalizes ``_pooled_ic`` (same file, :45) / ``ic`` of
    ``scripts/research/gmgp1_btc_conviction_probe.py:97``.
  * ``one_sided_p`` / ``bh_fdr`` are lifted from
    ``scripts/research/r1_illiquidity_probe.py:403`` / ``:411``.
  * ``block_bootstrap_mean`` is the circular block bootstrap of a precomputed series'
    mean — the IC-series analogue of ``block_bootstrap_sharpe_ci`` in
    ``sharpen/crypto/eval/statistics.py``.

DESIGN NOTE (ADR-1): the daily cross-sectional IC series IS a return series, so its
per-period Sharpe equals IC-IR; deflation by batch size reuses
``statistics.deflated_sharpe_ratio`` verbatim (done in the harness, not here).

v1 keeps the historical probes untouched (copy-with-provenance, not re-point) to avoid
perturbing their committed verdict JSONs; re-pointing is a deferred cleanup.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass(frozen=True, slots=True)
class CrossSectionalIC:
    """Result of a cross-sectional IC evaluation.

    ``ic_series`` is the per-day Spearman IC (only valid days retained). ``ic_ir`` is the
    per-period information ratio ``ic_mean / ic_std`` (Grinold); ``ic_tstat`` is its
    significance ``ic_ir * sqrt(n_days)`` (== the quantity ``_xs_ic`` returned as its
    "ic_ir"). All are NaN when fewer than 2 valid days exist. ``kept_days`` are the integer
    row indices into the original ``(T, N)`` panel that survived the ``min_names``/finite
    gate and produced ``ic_series`` (1:1, same length) — used by ``effective_n_trials`` to
    date-align the IC series of different signals onto a common grid (C2.1).
    """

    ic_series: np.ndarray
    ic_mean: float
    ic_std: float
    ic_ir: float
    ic_tstat: float
    n_days: int
    n_obs_total: int
    kept_days: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))


def spearman_ic(
    x: np.ndarray, y: np.ndarray, *, min_obs: int = 30, min_unique: int = 3
) -> float:
    """Pooled Spearman rank-IC between two flat arrays.

    Returns NaN on degenerate input (fewer than ``min_obs`` finite pairs or fewer than
    ``min_unique`` distinct values on either side) — matching the guards in the source
    probes so a constant/near-constant feature never yields a spurious correlation.
    """
    xa = np.asarray(x, dtype=float).reshape(-1)
    ya = np.asarray(y, dtype=float).reshape(-1)
    m = np.isfinite(xa) & np.isfinite(ya)
    if int(m.sum()) < min_obs:
        return float("nan")
    xv, yv = xa[m], ya[m]
    if np.unique(xv).size < min_unique or np.unique(yv).size < min_unique:
        return float("nan")
    rho, _ = spearmanr(xv, yv)
    return float(rho) if np.isfinite(rho) else float("nan")


def cross_sectional_ic(
    signal: np.ndarray,
    fwd: np.ndarray,
    active: np.ndarray | None = None,
    *,
    min_names: int = 4,
) -> CrossSectionalIC:
    """Cross-sectional rank-IC: per-day Spearman across names, then mean + IC-IR.

    Parameters
    ----------
    signal, fwd : (T, N) arrays
        ``signal[t, i]`` is the (causal) score for name ``i`` at day ``t``; ``fwd[t, i]``
        the forward return label. Must be equal-shaped 2-D.
    active : (T, N) bool, optional
        Point-in-time membership / tradeable mask. Names that are inactive (or NaN) on a
        day are excluded from that day's rank-IC.
    min_names : int
        Minimum active names required to rank a day (else the day is skipped).

    Vectorized: per-row average-tie ranks (matching ``scipy.spearmanr``) + a nan-aware
    per-row Pearson — equivalent to a per-day ``spearmanr`` loop (parity-tested) but ~10x
    faster. A day with zero rank-variance on either side (constant signal/return) yields
    NaN and is dropped. Rank-IC is invariant to monotone transforms of ``fwd`` — neutralize
    the SIGNAL (done upstream), never the return.
    """
    s = np.asarray(signal, dtype=float)
    f = np.asarray(fwd, dtype=float)
    if s.ndim != 2 or s.shape != f.shape:
        raise ValueError(f"signal and fwd must be equal-shaped (T,N); got {s.shape}, {f.shape}")
    if active is None:
        act = np.ones(s.shape, dtype=bool)
    else:
        act = np.asarray(active, dtype=bool)
        if act.shape != s.shape:
            raise ValueError(f"active must match signal shape; got {act.shape} vs {s.shape}")

    valid = np.isfinite(s) & np.isfinite(f) & act           # (T, N) common valid set
    sv = np.where(valid, s, np.nan)
    fv = np.where(valid, f, np.nan)
    rs = pd.DataFrame(sv).rank(axis=1).to_numpy()           # average-tie ranks; NaN preserved
    rf = pd.DataFrame(fv).rank(axis=1).to_numpy()
    n = valid.sum(axis=1).astype(np.float64)                # (T,)
    with np.errstate(invalid="ignore", divide="ignore"):
        nn = np.where(n > 0, n, np.nan)
        ds = rs - (np.nansum(rs, axis=1) / nn)[:, None]
        df_ = rf - (np.nansum(rf, axis=1) / nn)[:, None]
        cov = np.nansum(ds * df_, axis=1)
        denom = np.sqrt(np.nansum(ds * ds, axis=1) * np.nansum(df_ * df_, axis=1))
        ic_row = np.where(denom > 0, cov / denom, np.nan)   # (T,) per-day Spearman

    keep = (n >= min_names) & np.isfinite(ic_row)
    kept_idx = np.flatnonzero(keep).astype(np.int64)         # panel-row indices behind ic_series
    arr = ic_row[keep].astype(np.float64)
    n_days = int(arr.size)
    n_obs = int(n[keep].sum())
    if n_days < 2:
        return CrossSectionalIC(arr, float("nan"), float("nan"), float("nan"),
                                float("nan"), n_days, n_obs, kept_idx)
    ic_mean = float(arr.mean())
    ic_std = float(arr.std(ddof=1))
    ic_ir = ic_mean / ic_std if ic_std > 0 else float("nan")
    ic_tstat = ic_ir * float(np.sqrt(n_days)) if np.isfinite(ic_ir) else float("nan")
    return CrossSectionalIC(arr, ic_mean, ic_std, ic_ir, ic_tstat, n_days, n_obs, kept_idx)


def one_sided_p(ic: float, se: float) -> float:
    """One-sided p-value that the (mean) IC exceeds zero, from IC and its SE.

    Lifted from ``r1_illiquidity_probe.one_sided_p`` — z = ic/se, upper-tail normal.
    """
    from math import erf, sqrt

    if not np.isfinite(ic) or not np.isfinite(se) or se <= 0:
        return 1.0
    z = ic / se
    return float(1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))))


def bh_fdr(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg false-discovery-rate q-values (lifted from r1 probe).

    The multiple-comparison control across a BATCH of candidate signals: each signal's
    one-sided IC p-value -> q-value; ``q <= fdr_q_max`` is the batch gate.
    """
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank_i in range(m - 1, -1, -1):
        i = order[rank_i]
        val = p[i] * m / (rank_i + 1)
        prev = min(prev, val)
        q[i] = prev
    return q.tolist()


def bhy_fdr(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg-Yekutieli FDR q-values — valid under ARBITRARY dependence (C2.3).

    Plain BH (:func:`bh_fdr`) controls the false-discovery rate only under independence or
    positive-regression dependence (PRDS). Yekutieli (2001) extends FDR control to *any*
    dependence structure by inflating every threshold by the harmonic factor
    ``c(m) = Σ_{i=1}^{m} 1/i`` (≈ ln m + γ_E):

        q_(i) = min_{j ≥ i} [ p_(j) · m · c(m) / j ]

    so ``bhy_fdr(p)[i] >= bh_fdr(p)[i]`` elementwise (strictly more conservative). This is
    the Harvey-Liu-Zhu recommended multiple-testing control for the cross-section of
    expected returns (the "HLZ haircut" hurdle, alongside the t>3 bar). Monotone, bounded
    to [0, 1]. Returns ``[]`` for an empty input.
    """
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    if m == 0:
        return []
    c_m = float(np.sum(1.0 / np.arange(1, m + 1)))          # harmonic dependence penalty
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank_i in range(m - 1, -1, -1):                     # walk high→low rank, running min
        i = order[rank_i]
        val = p[i] * m * c_m / (rank_i + 1)
        prev = min(prev, val)
        q[i] = min(prev, 1.0)
    return q.tolist()


def effective_n_trials(
    series_list: list[np.ndarray],
    days_list: list[np.ndarray],
    *,
    n_periods: int | None = None,
    min_overlap: int = 23,
) -> float:
    """Effective number of *independent* trials among correlated IC series (C2.1).

    The Deflated Sharpe order statistic ``E[max SR over N trials]`` uses ``N`` = the number
    of configurations searched. When the ``N`` candidates are correlated (e.g. a formulaic-
    alpha library), the raw count over-states the *independent* breadth of the search,
    inflating the deflation benchmark ``SR*`` and over-penalizing (Type-II — real alphas
    killed). This returns the **participation ratio** (effective rank) of the trial
    correlation matrix ``R``::

        N_eff = (Σ λ_i)² / Σ λ_i²        (λ = eigenvalues of the K×K trial-corr matrix)

    which equals ``K`` when the trials are mutually uncorrelated (λ ≡ 1) and tends to 1 when
    all identical (one λ = K, the rest 0) — a continuous relaxation of "number of
    correlation clusters" (López de Prado, *Detection of False Investment Strategies*).
    Clamped to ``[1, K]``; ``K < 2`` returns ``K``.

    The ``series_list[j]`` IC series is aligned on its ``days_list[j]`` panel-row indices
    into a common ``[0, n_periods)`` grid. Each pairwise Pearson correlation uses only days
    where BOTH series are present; a pair with fewer than ``min_overlap`` common days is
    treated as uncorrelated (``r = 0``) — the conservative side, counting toward MORE
    effective trials (a higher deflation bar). Pairwise-complete ``R`` may be slightly
    non-PSD, so negative eigenvalues are clipped to 0 before the ratio (standard
    regularization).
    """
    K = len(series_list)
    if K != len(days_list):
        raise ValueError("series_list and days_list must be equal length")
    if K < 2:
        return float(K)
    if n_periods is None:
        n_periods = 1 + max((int(np.asarray(d).max()) for d in days_list
                             if np.asarray(d).size), default=0)
    n_periods = max(int(n_periods), 1)

    mat = np.full((n_periods, K), np.nan, dtype=np.float64)
    for j, (s, d) in enumerate(zip(series_list, days_list)):
        sv = np.asarray(s, dtype=np.float64).reshape(-1)
        dv = np.asarray(d, dtype=np.int64).reshape(-1)
        if dv.size != sv.size:                              # defensive: align on shorter prefix
            mlen = min(dv.size, sv.size)
            sv, dv = sv[:mlen], dv[:mlen]
        ok = (dv >= 0) & (dv < n_periods) & np.isfinite(sv)
        mat[dv[ok], j] = sv[ok]

    corr = np.eye(K, dtype=np.float64)
    for i in range(K):
        xi = mat[:, i]
        for jj in range(i + 1, K):
            xj = mat[:, jj]
            both = np.isfinite(xi) & np.isfinite(xj)
            rij = 0.0
            if int(both.sum()) >= min_overlap:
                a, b = xi[both], xj[both]
                if a.std() > 0 and b.std() > 0:
                    r = float(np.corrcoef(a, b)[0, 1])
                    rij = r if np.isfinite(r) else 0.0
            corr[i, jj] = corr[jj, i] = rij

    lam = np.clip(np.linalg.eigvalsh(corr), 0.0, None)      # clip tiny non-PSD negatives
    denom = float((lam ** 2).sum())
    if denom <= 0.0:
        return float(K)
    n_eff = float(lam.sum() ** 2 / denom)
    return float(min(max(n_eff, 1.0), float(K)))


def block_bootstrap_mean(
    series,
    *,
    block: int = 21,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 7,
) -> dict | None:
    """Circular block-bootstrap CI for the MEAN of a (daily) series.

    Used for a CI on mean-IC that respects autocorrelation (the IC series is serially
    dependent). Mirrors the block construction of
    ``statistics.block_bootstrap_sharpe_ci`` but returns the mean and the bootstrap mass
    at or below zero. Returns ``None`` with fewer than ``block + 2`` finite observations.
    """
    d = np.asarray([float(v) for v in series], dtype=np.float64)
    d = d[np.isfinite(d)]
    t_n = len(d)
    if t_n < block + 2:
        return None
    rng = np.random.default_rng(seed)
    boots = max(1, n_boot)
    n_blocks = int(np.ceil(t_n / block))
    base = np.arange(block)
    means = np.empty(boots)
    # Vectorized circular block bootstrap (chunked to bound memory) — replaces the
    # per-element while-loop (7.3M rng.integers calls -> a handful of batched draws).
    chunk = 2000
    done = 0
    while done < boots:
        b = min(chunk, boots - done)
        starts = rng.integers(0, t_n, size=(b, n_blocks))
        idx = (starts[:, :, None] + base[None, None, :]) % t_n   # (b, n_blocks, block)
        idx = idx.reshape(b, -1)[:, :t_n]                        # (b, t_n) circular blocks
        means[done:done + b] = d[idx].mean(axis=1)
        done += b
    return {
        "ci_low": float(np.quantile(means, alpha / 2.0)),
        "ci_high": float(np.quantile(means, 1.0 - alpha / 2.0)),
        "p_le_0": float((means <= 0.0).mean()),
        "mean": float(d.mean()),
        "block": int(block),
        "n_boot": int(boots),
    }
