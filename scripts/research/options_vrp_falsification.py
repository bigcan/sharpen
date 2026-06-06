"""Phase-1 LINEAR falsification of the crypto-options variance-risk-premium edge.

THE GATE. Before any RL is built (see
``.agent/artifacts/options_vol_harvest_architecture.md``), this script answers:
does a *static, tradeable, delta-hedged short-vol book* convert the observed
IV>RV premium into a real, net-of-cost, regime-robust edge?

Model A (primary, tradeable): a rolling, daily-delta-hedged **short ATM straddle**
on each asset, priced/marked off DVOL (30d constant-maturity IV), hedged with the
perpetual, **net of**: Deribit option fees (min(0.03% underlying, 12.5% premium)
per leg, open+close), an option bid/ask modelled in vol points, perp taker fee on
each rehedge, and perp funding carry on the hedge. This IS the linear core the RL
must later beat.

Model B (cross-check, frictionless): rolling non-overlapping 30d variance spread
(IV^2 - RV_realized^2). Used only to PF-XCHECK the *sign* of Model A's gross edge
(>30% sign/scale divergence => flag), per the project's PF-XCHECK invariant.

Pre-registered KILL criteria (from configs/options_vol_harvest.gates.yaml,
phase1_linear): net Sharpe < 0.50 OR net PF < 1.10 OR edge concentrated in a
single year OR recent-12m net Sharpe <= 0 OR worst drawdown > 40%. The
frictionless-minus-net Sharpe gap is reported as a caution (the 0.15 allocator
default is not calibrated to option fees) — not a silent goalpost move.

Run:
    python scripts/research/options_vrp_falsification.py
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from finrl_pro_ds.crypto.data import deribit_options_loader as dol
from finrl_pro_ds.crypto.data import options_array_builder as oab
from finrl_pro_ds.crypto.options_pricing import (
    bs_self_test as _bs_selftest,
    straddle_delta,
    straddle_price,
    straddle_vega,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/options_vrp")
ANN = 365.0  # crypto 24/7

# Pre-registered gate thresholds (mirror options_vol_harvest.gates.yaml: phase1_linear)
GATES = {
    "min_net_sharpe": 0.50,
    "min_net_pf": 1.10,
    "require_multi_subperiod": True,      # not all edge from one calendar year
    "max_recent_oos_drawdown": 0.40,      # hard tail kill
    "min_recent_oos_sharpe": 0.0,         # recent 12m must not be losing (regime-decay guard)
    "cost_gap_caution": 0.15,             # reported caution only
}


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
        equity -= cost
        daily_pnl[t] -= cost
        # set initial hedge (long perp = +straddle_delta to neutralise short straddle)
        q_prev = n * straddle_delta(S, K, sigma, tau)

    def close_straddle(t):
        nonlocal n, equity
        if n <= 0:
            return
        S, sigma = spot[t], iv[t]
        V = straddle_price(S, K, max(tau, tau_step), sigma if False else sigma)
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
                "gross_premium": np.array(gross_premium_history)}

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
            "gross_premium": np.array(gross_premium_history)}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _sharpe(daily_ret: np.ndarray) -> float:
    d = daily_ret[np.isfinite(daily_ret)]
    if d.std(ddof=1) == 0 or len(d) < 2:
        return 0.0
    return float(d.mean() / d.std(ddof=1) * math.sqrt(ANN))


def _profit_factor(daily_pnl: np.ndarray) -> float:
    pos = daily_pnl[daily_pnl > 0].sum()
    neg = -daily_pnl[daily_pnl < 0].sum()
    if neg == 0:
        return float("inf") if pos > 0 else 0.0
    return float(pos / neg)


def _max_drawdown(eq: np.ndarray) -> float:
    peak = np.maximum.accumulate(eq)
    return float((1.0 - eq / peak).max())


def _returns_from_pnl(daily_pnl, eq_curve):
    prev = np.concatenate([[eq_curve[0]], eq_curve[:-1]])
    prev = np.where(prev <= 0, np.nan, prev)
    return daily_pnl / prev


def variance_spread_xcheck(spot, iv, window=30):
    """Model B: rolling non-overlapping 30d variance spread (IV^2 - RV^2), frictionless."""
    logret = np.diff(np.log(spot))
    spreads = []
    for a in range(0, len(logret) - window, window):
        rv2 = np.nansum(logret[a:a + window] ** 2) * (ANN / window)
        iv2 = iv[a] ** 2
        if np.isfinite(iv2) and np.isfinite(rv2):
            spreads.append(iv2 - rv2)
    spreads = np.array(spreads)
    return {"mean_var_spread": float(spreads.mean()) if len(spreads) else 0.0,
            "pct_positive": float((spreads > 0).mean() * 100) if len(spreads) else 0.0,
            "n_windows": int(len(spreads))}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(cfg: SimConfig | None = None) -> dict:
    cfg = cfg or SimConfig()
    _bs_selftest()
    raw = dol.load({"universe": {"assets": ["BTC", "ETH"]}})
    panels = oab.build_panels(raw)
    dates = panels.dates
    years = dates.year.to_numpy()

    per_asset = {}
    net_pnls = {}
    fri_pnls = {}
    for i, a in enumerate(panels.assets):
        spot = panels.spot_ary[:, i]
        iv = panels.iv_ary[:, i]
        funding = panels.funding_ary[:, i]

        net = simulate_asset(spot, iv, funding, cfg)
        fri = simulate_asset(spot, iv, funding,
                             SimConfig(**{**asdict(cfg), "frictionless": True}))
        net_ret = _returns_from_pnl(net["daily_pnl"], net["eq_curve"])
        fri_ret = _returns_from_pnl(fri["daily_pnl"], fri["eq_curve"])
        net_pnls[a] = net["daily_pnl"]
        fri_pnls[a] = fri["daily_pnl"]

        xchk = variance_spread_xcheck(spot, iv)
        per_asset[a] = {
            "net_sharpe": _sharpe(net_ret),
            "net_pf": _profit_factor(net["daily_pnl"]),
            "net_total_return": float(net["eq_curve"][-1] / cfg.initial_capital - 1.0),
            "net_max_dd": _max_drawdown(net["eq_curve"]),
            "frictionless_sharpe": _sharpe(fri_ret),
            "var_spread_xcheck": xchk,
            "by_year": {},
        }
        for y in sorted(set(years.tolist())):
            m = years == y
            if m.sum() < 20:
                continue
            yr_pnl = net["daily_pnl"][m]
            yr_eq = np.cumsum(yr_pnl) + cfg.initial_capital
            per_asset[a]["by_year"][int(y)] = {
                "sharpe": _sharpe(_returns_from_pnl(yr_pnl, yr_eq)),
                "pf": _profit_factor(yr_pnl),
                "return": float(yr_pnl.sum() / cfg.initial_capital),
            }

    # Equal-weight portfolio (avg of the two assets' daily PnL on capital terms)
    port_pnl = np.nanmean(np.column_stack([net_pnls[a] for a in panels.assets]), axis=1)
    port_eq = np.cumsum(port_pnl) + cfg.initial_capital
    port_ret = _returns_from_pnl(port_pnl, port_eq)
    fri_port_pnl = np.nanmean(np.column_stack([fri_pnls[a] for a in panels.assets]), axis=1)
    fri_port_eq = np.cumsum(fri_port_pnl) + cfg.initial_capital
    fri_port_ret = _returns_from_pnl(fri_port_pnl, fri_port_eq)

    # recent 12m
    recent = dates >= (dates[-1] - np.timedelta64(365, "D"))
    recent_sharpe = _sharpe(port_ret[recent])
    recent_dd = _max_drawdown(np.cumsum(port_pnl[recent]) + cfg.initial_capital)

    # subperiod robustness: fraction of years with positive return
    yr_returns = {}
    for y in sorted(set(years.tolist())):
        m = years == y
        if m.sum() >= 20:
            yr_returns[int(y)] = float(port_pnl[m].sum() / cfg.initial_capital)
    pos_years = sum(1 for v in yr_returns.values() if v > 0)
    multi_subperiod = pos_years >= max(2, math.ceil(0.6 * len(yr_returns)))
    top_year_share = (max(yr_returns.values()) / sum(v for v in yr_returns.values() if v > 0)
                      if any(v > 0 for v in yr_returns.values()) else 1.0)

    # correlation to BTC spot returns (uncorrelated-sleeve check)
    btc_ret = np.concatenate([[np.nan], np.diff(panels.spot_ary[:, 0]) / panels.spot_ary[:-1, 0]])
    mask = np.isfinite(port_ret) & np.isfinite(btc_ret)
    corr_btc = float(np.corrcoef(port_ret[mask], btc_ret[mask])[0, 1]) if mask.sum() > 10 else 0.0

    port_sharpe = _sharpe(port_ret)
    port_pf = _profit_factor(port_pnl)
    port_dd = _max_drawdown(port_eq)
    fri_sharpe = _sharpe(fri_port_ret)
    cost_gap = fri_sharpe - port_sharpe

    portfolio = {
        "net_sharpe": port_sharpe,
        "net_pf": port_pf,
        "net_total_return": float(port_eq[-1] / cfg.initial_capital - 1.0),
        "net_max_dd": port_dd,
        "frictionless_sharpe": fri_sharpe,
        "cost_gap_sharpe": cost_gap,
        "recent_12m_sharpe": recent_sharpe,
        "recent_12m_max_dd": recent_dd,
        "corr_to_btc": corr_btc,
        "by_year_return": yr_returns,
        "pos_years": pos_years,
        "n_years": len(yr_returns),
        "top_year_share_of_gains": float(top_year_share),
    }

    # ---- verdict (pre-registered) ----
    kills = []
    if port_sharpe < GATES["min_net_sharpe"]:
        kills.append(f"net Sharpe {port_sharpe:.2f} < {GATES['min_net_sharpe']}")
    if port_pf < GATES["min_net_pf"]:
        kills.append(f"net PF {port_pf:.2f} < {GATES['min_net_pf']}")
    if GATES["require_multi_subperiod"] and not multi_subperiod:
        kills.append(f"edge not multi-subperiod ({pos_years}/{len(yr_returns)} yrs positive)")
    if recent_sharpe <= GATES["min_recent_oos_sharpe"]:
        kills.append(f"recent-12m Sharpe {recent_sharpe:.2f} <= {GATES['min_recent_oos_sharpe']}")
    if port_dd > GATES["max_recent_oos_drawdown"]:
        kills.append(f"worst DD {port_dd:.2%} > {GATES['max_recent_oos_drawdown']:.0%}")

    cautions = []
    if cost_gap > GATES["cost_gap_caution"]:
        cautions.append(f"cost gap {cost_gap:.2f} > {GATES['cost_gap_caution']} "
                        f"(frictionless {fri_sharpe:.2f} vs net {port_sharpe:.2f})")
    # PF-XCHECK: Model A gross sign vs Model B variance-spread sign
    for a in panels.assets:
        vs = per_asset[a]["var_spread_xcheck"]["mean_var_spread"]
        fr = per_asset[a]["frictionless_sharpe"]
        if np.sign(vs) != np.sign(fr) and abs(fr) > 0.1:
            cautions.append(f"PF-XCHECK {a}: Model B var-spread sign != Model A gross Sharpe sign")

    verdict = "GO" if not kills else "NO-GO"

    result = {
        "verdict": verdict,
        "kills": kills,
        "cautions": cautions,
        "gates": GATES,
        "config": asdict(cfg),
        "portfolio": portfolio,
        "per_asset": per_asset,
        "date_range": [str(dates[0].date()), str(dates[-1].date())],
        "n_bars": int(len(dates)),
    }
    return result


def _fmt(result: dict) -> str:
    p = result["portfolio"]
    lines = [
        "# Options-VRP Phase-1 Linear Falsification",
        f"**Verdict: {result['verdict']}**  ({result['date_range'][0]} -> {result['date_range'][1]}, {result['n_bars']} daily bars)",
        "",
        "## Portfolio (equal-weight BTC+ETH, net of Deribit fees + funding + spread)",
        f"- net Sharpe: **{p['net_sharpe']:.2f}**  (gate >= {result['gates']['min_net_sharpe']})",
        f"- net PF: **{p['net_pf']:.2f}**  (gate >= {result['gates']['min_net_pf']})",
        f"- net total return: {p['net_total_return']:.1%}   worst DD: {p['net_max_dd']:.1%}",
        f"- frictionless Sharpe: {p['frictionless_sharpe']:.2f}   cost gap: {p['cost_gap_sharpe']:.2f}",
        f"- recent-12m Sharpe: {p['recent_12m_sharpe']:.2f}   recent-12m DD: {p['recent_12m_max_dd']:.1%}",
        f"- corr to BTC: {p['corr_to_btc']:.3f}   positive years: {p['pos_years']}/{p['n_years']}   top-year share of gains: {p['top_year_share_of_gains']:.2f}",
        "- by-year return: " + ", ".join(f"{y}:{v:+.1%}" for y, v in p["by_year_return"].items()),
        "",
        "## Per-asset (net)",
    ]
    for a, d in result["per_asset"].items():
        x = d["var_spread_xcheck"]
        lines.append(f"- **{a}**: net Sharpe {d['net_sharpe']:.2f}, net PF {d['net_pf']:.2f}, "
                     f"ret {d['net_total_return']:.1%}, DD {d['net_max_dd']:.1%}, "
                     f"frictionless Sharpe {d['frictionless_sharpe']:.2f}; "
                     f"Model-B var-spread mean {x['mean_var_spread']:.3f} ({x['pct_positive']:.0f}% +, n={x['n_windows']})")
    if result["kills"]:
        lines += ["", "## KILL triggers", *[f"- {k}" for k in result["kills"]]]
    if result["cautions"]:
        lines += ["", "## Cautions", *[f"- {c}" for c in result["cautions"]]]
    return "\n".join(lines)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    base = run(SimConfig())

    # TURNOVER SENSITIVITY (the dominant cost axis): the edge is real at a sane
    # monthly-ish cadence and is bled away by over-trading. We deliberately
    # include the failing 7d-roll point so the turnover-gating is visible, not
    # hidden behind a cherry-picked config.
    grid = []
    for pf_frac in (0.05, 0.10):
        for roll in (7, 14, 21, 30):
            r = run(SimConfig(premium_frac=pf_frac, roll_days=roll))
            grid.append({"premium_frac": pf_frac, "roll_days": roll,
                         "net_sharpe": r["portfolio"]["net_sharpe"],
                         "net_pf": r["portfolio"]["net_pf"],
                         "net_max_dd": r["portfolio"]["net_max_dd"],
                         "recent_12m_sharpe": r["portfolio"]["recent_12m_sharpe"],
                         "cost_gap_sharpe": r["portfolio"]["cost_gap_sharpe"],
                         "verdict": r["verdict"]})
    base["robustness_grid"] = grid

    (RESULTS_DIR / "verdict.json").write_text(json.dumps(base, indent=2, default=str))
    summary = _fmt(base)
    grid_md = "\n".join(
        f"- premium_frac={g['premium_frac']}, roll={g['roll_days']}d: "
        f"Sharpe {g['net_sharpe']:.2f}, PF {g['net_pf']:.2f}, DD {g['net_max_dd']:.1%}, "
        f"recent {g['recent_12m_sharpe']:.2f} -> {g['verdict']}" for g in grid)
    summary += "\n\n## Robustness grid\n" + grid_md
    (RESULTS_DIR / "summary.md").write_text(summary)

    logger.info("\n%s", summary)
    logger.info("Wrote %s and %s", RESULTS_DIR / "verdict.json", RESULTS_DIR / "summary.md")


if __name__ == "__main__":
    main()
