"""Panel: the aligned (T, N) cross-sectional equity data the harness operates on,
plus cross-sectional neutralization and OHLC sanity (DATA-CLEAN).

A ``Panel`` is populated by a ``PanelLoader`` (Sharadar in v1; see ``loaders.py``) or by
``make_synthetic_panel`` for tests / smoke runs. Neutralization (winsor -> z-score ->
sector/size demean) is applied to SIGNAL scores BEFORE the IC is computed, so an alpha's
measured predictive power is name-selection skill, not a hidden sector or size tilt
(ADR-4). Rank-IC is invariant to monotone transforms of the forward return, so the
RETURN is never neutralized here — only the signal is.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True, slots=True)
class Panel:
    """Aligned point-in-time cross-section. All price/volume arrays are ``(T, N)`` float64
    with NaN where a name is inactive; ``active`` is the PIT membership / tradeable mask.
    """

    dates: np.ndarray          # (T,) datetime64[ns], sorted unique
    tickers: tuple[str, ...]   # (N,)
    open: np.ndarray           # (T, N)
    high: np.ndarray           # (T, N)
    low: np.ndarray            # (T, N)
    close: np.ndarray          # (T, N)
    volume: np.ndarray         # (T, N)
    active: np.ndarray         # (T, N) bool — point-in-time membership/tradeable
    adv_usd: np.ndarray        # (T, N) trailing dollar ADV (cost & size proxy)
    sector_id: np.ndarray      # (N,) int (v1 current GICS; PIT at GO-gate)
    meta: dict                 # {survivorship_free: bool, source, universe_def, ...}

    @property
    def T(self) -> int:
        return int(self.dates.shape[0])

    @property
    def N(self) -> int:
        return len(self.tickers)

    def truncated(self, t: int) -> "Panel":
        """Return a Panel with rows ``[0..t]`` inclusive — used by the Tier-0 causality
        tripwire to verify ``compute(panel.truncated(t))[t] == compute(panel)[t]``."""
        if not 0 <= t < self.T:
            raise IndexError(f"t={t} out of range [0,{self.T})")
        s = slice(0, t + 1)
        return replace(
            self, dates=self.dates[s], open=self.open[s], high=self.high[s],
            low=self.low[s], close=self.close[s], volume=self.volume[s],
            active=self.active[s], adv_usd=self.adv_usd[s],
        )

    def forward_returns(self, h: int) -> np.ndarray:
        """``(T, N)`` close-to-close forward return ``close[t+h]/close[t] - 1``.

        NaN on the last ``h`` rows and wherever either endpoint is inactive/non-positive
        (LEAK-2: this is a LABEL, never a feature). Matches the gated construction in
        ``cmgp1_x2_leak_ic_probe_v2``.
        """
        if h < 1:
            raise ValueError(f"horizon must be >= 1; got {h}")
        c = self.close
        fwd = np.full(c.shape, np.nan, dtype=np.float64)
        if h < self.T:
            denom = np.where(c[:-h] > 0, c[:-h], np.nan)
            ratio = c[h:] / denom - 1.0
            both_active = self.active[:-h] & self.active[h:]
            ratio = np.where(both_active, ratio, np.nan)
            fwd[:-h] = ratio
        return fwd


def ohlc_violations(panel: Panel) -> dict:
    """Count OHLC-sanity violations over active, finite bars (DATA-CLEAN, PF-XCHECK).

    Returns counts for: high below max(open,close), low above min(open,close), high<low,
    and non-positive prices. ``total == 0`` is the Tier-0 clean condition.
    """
    o, h, lo, c, act = panel.open, panel.high, panel.low, panel.close, panel.active
    fin = np.isfinite(o) & np.isfinite(h) & np.isfinite(lo) & np.isfinite(c) & act
    eps = 1e-9
    hi_bad = fin & (h < np.maximum(o, c) - eps)
    lo_bad = fin & (lo > np.minimum(o, c) + eps)
    hl_bad = fin & (h < lo - eps)
    pos_bad = fin & ~((o > 0) & (h > 0) & (lo > 0) & (c > 0))
    v = {
        "high_lt_max_oc": int(hi_bad.sum()),
        "low_gt_min_oc": int(lo_bad.sum()),
        "high_lt_low": int(hl_bad.sum()),
        "nonpositive_price": int(pos_bad.sum()),
    }
    v["total"] = sum(v.values())
    return v


def neutralize(
    scores: np.ndarray,
    panel: Panel,
    steps: tuple[str, ...] = ("winsor", "zscore", "sector"),
    *,
    winsor_pct: tuple[float, float] = (0.01, 0.99),
    min_names: int = 4,
) -> np.ndarray:
    """Per-day cross-sectional neutralization of a ``(T, N)`` signal.

    Pipeline (subset/order driven by ``steps``): winsorize to ``winsor_pct`` quantiles ->
    z-score across active names -> residualize on an intercept + sector dummies
    (``"sector"``) and/or a log-ADV size factor (``"size"``) via least squares, keeping the
    residual. Days with fewer than ``min_names`` active names are dropped (NaN row).
    Residualization is skipped on a day with too few degrees of freedom (k <= n_cols + 5)
    so thin days are not spuriously zeroed.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.shape != (panel.T, panel.N):
        raise ValueError(f"scores must be {(panel.T, panel.N)}; got {s.shape}")
    out = np.full(s.shape, np.nan, dtype=np.float64)

    for t in range(panel.T):
        row = s[t]
        mask = np.isfinite(row) & panel.active[t]
        k = int(mask.sum())
        if k < min_names:
            continue
        x = row[mask].astype(np.float64)

        if "winsor" in steps:
            lo_q, hi_q = np.quantile(x, winsor_pct[0]), np.quantile(x, winsor_pct[1])
            x = np.clip(x, lo_q, hi_q)
        if "zscore" in steps:
            sd = x.std()
            x = (x - x.mean()) / sd if sd > 0 else x - x.mean()

        cols: list[np.ndarray] = []
        if "sector" in steps:
            sec = panel.sector_id[mask]
            uniq = np.unique(sec)
            for u in uniq[1:]:  # drop first level — intercept carries it
                cols.append((sec == u).astype(np.float64))
        if "size" in steps:
            adv = panel.adv_usd[t][mask]
            sz = np.log(np.where(adv > 0, adv, np.nan))
            fin = np.isfinite(sz)
            if int(fin.sum()) >= 2:
                m_, sd_ = sz[fin].mean(), sz[fin].std()
                sz = np.where(fin, (sz - m_) / (sd_ if sd_ > 0 else 1.0), 0.0)
            else:
                sz = np.zeros(k)
            cols.append(sz)

        if cols:
            design = np.column_stack([np.ones(k), *cols])
            if k > design.shape[1] + 5:  # enough residual dof
                coef, *_ = np.linalg.lstsq(design, x, rcond=None)
                x = x - design @ coef

        out[t, mask] = x
    return out


def make_synthetic_panel(
    *,
    T: int = 400,
    N: int = 60,
    n_sectors: int = 6,
    seed: int = 0,
) -> Panel:
    """Construct a clean random-walk Panel for tests / smoke runs (OHLC-sane by
    construction, fully active, random sectors). Not for research — demo fixture only.
    """
    rng = np.random.default_rng(seed)
    rets = 0.01 * rng.standard_normal((T, N))
    logp = np.cumsum(rets, axis=0) + rng.uniform(3.0, 5.0, size=N)
    close = np.exp(logp)
    open_ = close * np.exp(0.001 * rng.standard_normal((T, N)))
    high = np.maximum(open_, close) * np.exp(np.abs(0.002 * rng.standard_normal((T, N))))
    low = np.minimum(open_, close) * np.exp(-np.abs(0.002 * rng.standard_normal((T, N))))
    volume = rng.uniform(1e5, 1e7, size=(T, N))
    adv_usd = close * volume
    active = np.ones((T, N), dtype=bool)
    sector_id = rng.integers(0, n_sectors, size=N)
    dates = (np.datetime64("2010-01-04")
             + np.arange(T) * np.timedelta64(1, "D")).astype("datetime64[ns]")
    tickers = tuple(f"SYN{i:03d}" for i in range(N))
    meta = {"survivorship_free": False, "source": "synthetic", "universe_def": "synthetic"}
    return Panel(dates, tickers, open_, high, low, close, volume, active,
                 adv_usd, sector_id, meta)
