"""Transaction-cost models and drawdown / profit-factor helpers (Tier-2 capturability).

``COST_MODELS`` is the one-way equity transaction-cost trio, generalized from
``scripts/research/xsec_momentum_falsification.py:47`` (which used 2 bps / 10 bps for
liquid ETFs). Single-name equity is wider, so the defaults here are 10 bps (standard) /
25 bps (harsh) one-way; the live values are overridable by
``configs/signal_eval.gates.yaml`` so nothing is hardcoded at a decision boundary.

Sharpe / Sortino / Calmar are NOT duplicated here — call them from
``finrl_pro_ds.crypto.eval.statistics`` with ``periods_per_year=252`` for equity daily.
"""
from __future__ import annotations

import numpy as np

# One-way fraction-of-notional cost models. Capturability (Tier 2) is SECONDARY to the
# gross-IC rank key, so a constant-bps model per profile is sufficient for v1 triage.
COST_MODELS: dict[str, float] = {
    "frictionless": 0.0,
    "standard": 0.0010,
    "harsh": 0.0025,
}


def max_drawdown(daily_returns) -> float:
    """Maximum peak-to-trough drawdown of an equity curve built from daily returns.

    Returns a non-positive float (0.0 if empty / no drawdown). Matches
    ``xsec_momentum_falsification.max_dd`` semantics.
    """
    r = np.asarray([float(v) for v in daily_returns], dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0.0
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    return float((eq / np.where(peak == 0.0, 1.0, peak) - 1.0).min())


def profit_factor(pnl) -> float:
    """Profit factor = sum(positive P&L) / |sum(negative P&L)|.

    Returns ``+inf`` when there are no losing observations (handled honestly by callers /
    gates). Matches ``pf`` / ``pf_bar`` across the probes.
    """
    a = np.asarray([float(v) for v in pnl], dtype=np.float64)
    a = a[np.isfinite(a)]
    up = a[a > 0.0].sum()
    dn = -a[a < 0.0].sum()
    return float(up / dn) if dn > 0 else float("inf")
