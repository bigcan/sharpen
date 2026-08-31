"""Rolling delta-hedged short-straddle VRP simulator — the library core.

Promoted VERBATIM from ``scripts/research/options_vrp_falsification.py``
(MS-ADR-9, ``.agent/artifacts/multi_sleeve_paper_executor_architecture.md``) so the
paper executor's ``VRPSleeve`` can import the *validated* sim without a ``scripts/``
dependency (the library must never import from ``scripts/``). **Logic is unchanged** —
a refactor-parity check pins ``results/options_vrp/verdict.json`` byte-identical
before/after the move.

Model A (the tradeable linear core the RL must beat — RL ship-linear binding,
uplift −0.088): a rolling, daily-delta-hedged **short ATM straddle**, priced/marked
off DVOL (30d constant-maturity IV), hedged with the perpetual, **net of** Deribit
option fees (min(0.03% underlying, 12.5% premium) per leg, open+close), an option
bid/ask in vol points, perp taker fee on each rehedge, and perp funding carry on the
hedge.

Honest band: size to net Sharpe 0.6–0.9 (BTC; ETH dropped), NOT the 1.06 best-case;
short-vol left tail real (skew −2.7, intraday DD ~8.4%, CVaR99 −1.47%/day). See
``docs/research/options_vrp_paper_promotion_2026-06-22.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sharpen.crypto.options_pricing import (
    straddle_delta,
    straddle_price,
    straddle_vega,
)

ANN = 365.0  # crypto 24/7


# ---------------------------------------------------------------------------
# Single-asset rolling delta-hedged short-straddle simulation
# ---------------------------------------------------------------------------
@dataclass
class SimConfig:
    # Defensible primary cadence: sell a 30d ATM straddle, roll at 21 days held
    # (tenor 30d -> 9d, so we never trade the gamma-explosive final week), modest
    # 5%-of-equity premium sizing. Rolling a 30d option every ~7 days over-trades
    # (discards tenor already paid for in spread) and is shown as a cautionary
    # point in the turnover grid, NOT the primary.
    premium_frac: float = 0.05        # gross option premium sold per roll, as frac of equity
    roll_days: int = 21               # hold each straddle this many days, then roll
    entry_tenor_days: int = 30        # constant-maturity target
    initial_capital: float = 100_000.0
    option_fee_pct_underlying: float = 0.0003   # Deribit: 0.03% of underlying per option
    option_fee_cap_pct_premium: float = 0.125   # capped at 12.5% of the option premium
    perp_taker_fee: float = 0.0005
    option_spread_vol_pts: float = 1.0          # modelled bid/ask in vol points (round trip via vega)
    frictionless: bool = False                  # zero all costs (for the cost-gap probe)


def _option_fees(n: float, S: float, premium_total: float, cfg: SimConfig) -> float:
    """Deribit option fee for opening/closing ``n`` straddles (2 legs each)."""
    if cfg.frictionless:
        return 0.0
    fee_underlying = 2.0 * n * cfg.option_fee_pct_underlying * S
    fee_premium_cap = cfg.option_fee_cap_pct_premium * premium_total
    return min(fee_underlying, fee_premium_cap)


def _spread_cost(n: float, S: float, sigma: float, tau: float, cfg: SimConfig) -> float:
    """Half-spread cost (vol points -> USD via vega) for one side of the trade."""
    if cfg.frictionless:
        return 0.0
    vega = straddle_vega(S, S, sigma, tau)
    return n * vega * (cfg.option_spread_vol_pts / 100.0) * 0.5


def simulate_asset(spot, iv, funding, cfg: SimConfig):
    """Run the rolling short-straddle sim. Returns dict with daily PnL array etc.

    Arrays are 1-D, aligned, NaN-free for spot/iv. ``funding`` is the daily perp
    carry fraction.
    """
    T = len(spot)
    equity = cfg.initial_capital
    daily_pnl = np.zeros(T)
    eq_curve = np.full(T, cfg.initial_capital, float)

    # active straddle state
    n = 0.0            # number of short straddles (>=0; we are short)
    K = 0.0
    tau = 0.0
    q_prev = 0.0       # previous perp hedge qty (coin); long perp offsets short-straddle delta
    days_held = 0
    tau_step = 1.0 / ANN
    gross_premium_history = []
    vega_pct_history = []   # |net vega| / equity per held bar (short-vol tail metric)

    def open_straddle(t):
        nonlocal n, K, tau, equity, q_prev
        S, sigma = spot[t], iv[t]
        K = S
        tau = cfg.entry_tenor_days / ANN
        unit_prem = straddle_price(S, K, sigma, tau)
        gross = cfg.premium_frac * equity
        n = gross / unit_prem if unit_prem > 0 else 0.0
        prem_total = n * unit_prem
        gross_premium_history.append(prem_total)
        # opening costs (we RECEIVE premium; pay fee + half-spread)
        cost = _option_fees(n, S, prem_total, cfg) + _spread_cost(n, S, sigma, tau, cfg)
        # perp-hedge establishment / roll-jump taker fee: trade the hedge from its
        # prior level (q_prev — leftover hedge from the just-closed straddle, or 0 at
        # the initial open) to the new straddle's delta hedge. The in-loop rehedge fee
        # only covers intra-hold delta moves, so without this the establishment leg of
        # every roll was free (the omitted ~$1.1k/5.2y fee; V2-05/V4-01).
        q_new = n * straddle_delta(S, K, sigma, tau)
        if not cfg.frictionless:
            cost += cfg.perp_taker_fee * abs(q_new - q_prev) * S
        equity -= cost
        daily_pnl[t] -= cost
        # set initial hedge (long perp = +straddle_delta to neutralise short straddle)
        q_prev = q_new

    def close_straddle(t):
        nonlocal n, equity
        if n <= 0:
            return
        S, sigma = spot[t], iv[t]
        # FIX (V3-03): args were transposed — straddle_price(S, K, max(tau, tau_step),
        # sigma) passed tau into the sigma slot and sigma into the tau slot (plus a dead
        # `sigma if False else sigma` ternary). Inert at 21d (this V only feeds the
        # close-fee premium cap, which never binds), but a latent unit landmine.
        V = straddle_price(S, K, sigma, max(tau, tau_step))
        cost = _option_fees(n, S, n * V, cfg) + _spread_cost(n, S, sigma, max(tau, tau_step), cfg)
        equity -= cost
        daily_pnl[t] -= cost
        n = 0.0

    # find first valid bar
    t0 = 0
    while t0 < T and (not np.isfinite(spot[t0]) or not np.isfinite(iv[t0]) or iv[t0] <= 0):
        t0 += 1
    if t0 >= T - 2:
        return {"daily_pnl": daily_pnl, "eq_curve": eq_curve, "t0": t0,
                "gross_premium": np.array(gross_premium_history),
                "vega_pct": np.array(vega_pct_history)}

    open_straddle(t0)
    days_held = 0

    for t in range(t0, T - 1):
        S_t, sig_t = spot[t], iv[t]
        S_n, sig_n = spot[t + 1], iv[t + 1]
        if not (np.isfinite(S_n) and np.isfinite(sig_n) and sig_n > 0):
            S_n, sig_n = S_t, sig_t

        # 1) hedge for today from current greeks (delta-neutralise short straddle)
        q_t = n * straddle_delta(S_t, K, sig_t, max(tau, tau_step))
        # rehedge cost on the change in perp position
        rehedge_cost = 0.0 if cfg.frictionless else cfg.perp_taker_fee * abs(q_t - q_prev) * S_t
        # net-vega / equity at this bar (short-vol tail exposure; V5-03 gate metric)
        vega_pct_history.append(
            abs(n * straddle_vega(S_t, K, sig_t, max(tau, tau_step))) / max(equity, 1e-6))

        # 2) advance one day: option MTM (short => gain when value falls), hedge PnL, funding
        V_t = straddle_price(S_t, K, sig_t, max(tau, tau_step))
        tau_next = max(tau - tau_step, tau_step)
        V_n = straddle_price(S_n, K, sig_n, tau_next)
        option_pnl = n * (V_t - V_n)                       # short straddle
        hedge_pnl = q_t * (S_n - S_t)                      # long perp
        funding_pnl = 0.0 if cfg.frictionless else -q_t * S_t * funding[t]

        pnl = option_pnl + hedge_pnl - rehedge_cost + funding_pnl
        daily_pnl[t + 1] += pnl
        equity += pnl
        eq_curve[t + 1] = equity
        q_prev = q_t
        tau = tau_next
        days_held += 1

        # 3) roll if held long enough or tenor too short
        if days_held >= cfg.roll_days or tau <= (cfg.entry_tenor_days - cfg.roll_days - 1) / ANN:
            close_straddle(t + 1)
            if equity > 0:
                open_straddle(t + 1)
            days_held = 0
            eq_curve[t + 1] = equity

    return {"daily_pnl": daily_pnl, "eq_curve": eq_curve, "t0": t0,
            "gross_premium": np.array(gross_premium_history),
            "vega_pct": np.array(vega_pct_history)}


def returns_from_pnl(daily_pnl, eq_curve):
    """Daily fractional returns from a daily-PnL array + its equity curve (prior-bar
    equity as the base; <=0 prior equity → NaN). Public API (was the falsification
    script's ``_returns_from_pnl``; promoted for reuse by the paper ``VRPSleeve``)."""
    prev = np.concatenate([[eq_curve[0]], eq_curve[:-1]])
    prev = np.where(prev <= 0, np.nan, prev)
    return daily_pnl / prev
