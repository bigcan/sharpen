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

import hashlib
import json
import logging
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from finrl_pro_ds.crypto.data import deribit_options_loader as dol
from finrl_pro_ds.crypto.data import options_array_builder as oab
from finrl_pro_ds.crypto.options_pricing import bs_self_test as _bs_selftest, ncdf
from finrl_pro_ds.crypto.options_vrp_sim import (
    ANN,
    SimConfig,
    returns_from_pnl as _returns_from_pnl,
    simulate_asset,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/options_vrp")

# Gate thresholds live in configs/options_vol_harvest.gates.yaml — NEVER hardcoded
# here (CLAUDE.md invariant). The prior module-level dict *mirrored* but did not
# *read* the yaml, so a yaml edit silently never reached the gate (V1-11/V5-03).
# Resolved relative to this script so the working directory does not matter.
GATES_FILE = Path(__file__).resolve().parents[2] / "configs" / "options_vol_harvest.gates.yaml"


def load_gates(gates_file: Path | str = GATES_FILE) -> dict:
    """Read the pre-registered kill thresholds from the gates overlay: the
    ``phase1_linear`` criteria + the short-vol ``tail`` kill-switches (the latter
    were declared-but-not-wired before this — V5-03). The ``.get`` defaults are a
    last-resort for a key missing from the yaml, NOT an alternate source of truth."""
    g = yaml.safe_load(Path(gates_file).read_text(encoding="utf-8")).get("gates", {})
    p1 = g.get("phase1_linear", {})
    tail = g.get("tail", {})
    return {
        "min_net_sharpe": float(p1.get("min_net_sharpe", 0.50)),
        "min_net_pf": float(p1.get("min_net_pf", 1.10)),
        "require_multi_subperiod": bool(p1.get("require_multi_subperiod", True)),
        "max_recent_oos_drawdown": float(p1.get("max_recent_oos_drawdown", 0.40)),
        "min_recent_oos_sharpe": float(p1.get("min_recent_oos_sharpe", 0.0)),
        # short-vol tail kill-switches — now WIRED against the linear core (V5-03)
        "max_worst_window_dd_pct": float(tail.get("max_worst_window_dd_pct", 25.0)),
        "max_net_vega_per_100k": float(tail.get("max_net_vega_per_100k", 50000.0)),
        "cvar95_floor_pct": float(tail.get("cvar95_floor_pct", -8.0)),
        # reported caution only (not a pre-registered kill) — falsification-local
        "cost_gap_caution": 0.15,
    }


GATES = load_gates()


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


# ---------------------------------------------------------------------------
# Risk / tail / multiplicity stats (V6 — short-vol left tail; the honest-band caveat).
# A short-vol book has negative skew, so Sharpe overstates it — report Sortino, CVaR,
# the block-bootstrap CI and the multiplicity-deflated Sharpe alongside the headline.
# ---------------------------------------------------------------------------
def _block_bootstrap_sharpe_ci(daily_ret, block=21, n_boot=10_000, seed=7):
    """Circular block-bootstrap CI95 of the annualized Sharpe (block preserves the
    short-vol autocorrelation/clustering a naive iid bootstrap would destroy)."""
    d = daily_ret[np.isfinite(daily_ret)]
    if len(d) < block + 2:
        return None
    rng = np.random.default_rng(seed)
    Tn = len(d)
    sh = np.empty(n_boot)
    base = np.arange(block)
    for b in range(n_boot):
        idx = []
        while len(idx) < Tn:
            s0 = int(rng.integers(0, Tn))
            idx.extend(((s0 + base) % Tn).tolist())
        x = d[np.array(idx[:Tn])]
        sd = x.std(ddof=1)
        sh[b] = x.mean() / sd * math.sqrt(ANN) if sd > 0 else 0.0
    return {"ci95_low": float(np.quantile(sh, 0.025)),
            "ci95_high": float(np.quantile(sh, 0.975)),
            "p_sharpe_lt_0_5": float((sh < 0.5).mean()),
            "p_sharpe_lt_0": float((sh < 0).mean())}


def _risk_block(daily_ret, vega_pct=None, *, bootstrap=False):
    """Distribution / tail stats for a daily-return series (+ optional vega exposure)."""
    d = daily_ret[np.isfinite(daily_ret)]
    out = {"n_obs": int(len(d))}
    if len(d) >= 3:
        mu, sd = float(d.mean()), float(d.std(ddof=1))
        out["mean_daily"] = mu
        out["std_daily"] = sd
        out["skew"] = float(((d - mu) ** 3).mean() / sd ** 3) if sd > 0 else 0.0
        out["kurtosis"] = float(((d - mu) ** 4).mean() / sd ** 4) if sd > 0 else 0.0
        downside = d[d < 0]
        out["sortino"] = (float(mu / downside.std(ddof=1) * math.sqrt(ANN))
                          if len(downside) > 2 and downside.std(ddof=1) > 0 else float("nan"))
        v95 = float(np.quantile(d, 0.05))
        out["cvar95_pct_daily"] = float(d[d <= v95].mean()) * 100.0
        v99 = float(np.quantile(d, 0.01))
        out["cvar99_pct_daily"] = float(d[d <= v99].mean()) * 100.0
        out["worst_day_pct"] = float(d.min()) * 100.0
    if vega_pct is not None and len(vega_pct):
        out["max_net_vega_per_100k"] = float(np.nanmax(vega_pct) * 1e5)
        out["median_net_vega_per_100k"] = float(np.nanmedian(vega_pct) * 1e5)
    if bootstrap:
        out["bootstrap_sharpe"] = _block_bootstrap_sharpe_ci(daily_ret)
    return out


def _deflated_sharpe(observed_sr_daily, trial_sharpes_daily, n_obs, skew, kurt, n_trials):
    """Bailey & López de Prado deflated Sharpe: the probability the *true* SR>0 after
    correcting the observed daily SR for the ``n_trials`` configs graded (multiplicity)
    and the non-normal (skew/kurt) return shape. ``trial_sharpes_daily`` supplies the
    across-config SR variance (the deflation benchmark SR*)."""
    ts = np.asarray(trial_sharpes_daily, float)
    v_sr = float(np.var(ts, ddof=1)) if len(ts) > 1 else 0.0
    if v_sr <= 0 or n_obs < 3 or n_trials < 2:
        return None
    gamma = 0.5772156649015329  # Euler-Mascheroni

    def _qnorm(p):  # inverse standard-normal CDF via bisection on ncdf (no scipy dep)
        lo, hi = -10.0, 10.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if ncdf(mid) < p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    sr_star = math.sqrt(v_sr) * ((1 - gamma) * _qnorm(1 - 1.0 / n_trials)
                                 + gamma * _qnorm(1 - 1.0 / (n_trials * math.e)))
    denom = math.sqrt(max(1 - skew * observed_sr_daily
                          + (kurt - 1) / 4.0 * observed_sr_daily ** 2, 1e-12))
    dsr = ncdf((observed_sr_daily - sr_star) * math.sqrt(n_obs - 1) / denom)
    return {"sr_star_ann": float(sr_star * math.sqrt(ANN)), "dsr": float(dsr)}


def _data_fingerprint(cache_dir="data/processed/deribit") -> dict:
    """SHA256 (16-hex prefix) of each cached Deribit parquet — the data fingerprint a
    refresh would change (V1-07: results/ + data/ are gitignored, no DVC)."""
    out = {}
    p = Path(cache_dir)
    if p.exists():
        for f in sorted(p.glob("*.parquet")):
            out[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    return out


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
def run(cfg: SimConfig | None = None, *, bootstrap_stats: bool = True) -> dict:
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
            "risk_stats": _risk_block(net_ret, net["vega_pct"],
                                      bootstrap=(a == "BTC" and bootstrap_stats)),
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

    # short-vol tail stats + the kill-switch metrics (V5-03/V6) against the linear core
    port_risk = _risk_block(port_ret, bootstrap=bootstrap_stats)
    worst_asset_dd_pct = max(d["net_max_dd"] for d in per_asset.values()) * 100.0
    worst_vega_100k = max(d["risk_stats"].get("max_net_vega_per_100k", 0.0)
                          for d in per_asset.values())
    port_cvar95 = port_risk.get("cvar95_pct_daily", 0.0)

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
        "risk_stats": port_risk,
        "tail_metrics": {
            "worst_per_asset_dd_pct": worst_asset_dd_pct,
            "worst_per_asset_net_vega_per_100k": worst_vega_100k,
            "portfolio_cvar95_pct_daily": port_cvar95,
        },
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
    # short-vol tail kill-switches — declared in gates.yaml `tail:` but, until now,
    # never evaluated against the linear core (V5-03). Wired fail-loud; all pass on
    # today's clean data, but a refresh that pushed any past the floor now NO-GOs.
    if port_cvar95 < GATES["cvar95_floor_pct"]:
        kills.append(f"daily CVaR95 {port_cvar95:.2f}% < floor {GATES['cvar95_floor_pct']}%")
    if worst_vega_100k > GATES["max_net_vega_per_100k"]:
        kills.append(f"net vega/100k {worst_vega_100k:,.0f} > cap {GATES['max_net_vega_per_100k']:,.0f}")
    if worst_asset_dd_pct > GATES["max_worst_window_dd_pct"]:
        kills.append(f"worst per-asset DD {worst_asset_dd_pct:.1f}% > {GATES['max_worst_window_dd_pct']}%")

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
    hb = result.get("honest_band")
    if hb:
        btc_rs = result["per_asset"]["BTC"].get("risk_stats", {})
        dsr16 = (btc_rs.get("deflated_sharpe", {}) or {}).get("16") or {}
        boot = btc_rs.get("bootstrap_sharpe") or {}
        port_rs = result["portfolio"].get("risk_stats", {})
        band = hb.get("honest_expectation_band")
        if band:  # legacy deploy band (pre-2026-06-23)
            lo, hi = band
            head = (f"- **SIZE to the {lo:.1f}-{hi:.1f} honest band**, not the "
                    f"{hb['headline_best_case_btc_net_sharpe']:.2f} best-of-2-asset headline")
        else:  # de-contaminated: NO real-execution-validated band; sleeve BLOCKED
            head = (f"- **DO NOT DEPLOY — sleeve BLOCKED, clearance RESCINDED.** {hb.get('deploy_status', '')}. "
                    f"The 0.61 real-chain anchor was USDC-linear contamination; de-contaminated real-chain "
                    f"{hb.get('real_chain_status', 'NO-GO')}. Honest read: {hb.get('deflated_sharpe_read', '')}. "
                    f"DVOL-synthetic headline {hb['headline_best_case_btc_net_sharpe']:.2f} is UNCONFIRMED by real execution")
        lines += [
            "",
            "## Honest band & tail caveats (RE-DERIVED post de-contamination 2026-06-23)",
            head,
            f"- BTC Sortino {btc_rs.get('sortino', float('nan')):.2f} < Sharpe {result['per_asset']['BTC']['net_sharpe']:.2f} "
            f"(short-vol left tail), skew {btc_rs.get('skew', float('nan')):.2f}, "
            f"CVaR95 {btc_rs.get('cvar95_pct_daily', float('nan')):.2f}%/day, CVaR99 {btc_rs.get('cvar99_pct_daily', float('nan')):.2f}%/day",
            f"- deflated Sharpe (N=16) {dsr16.get('dsr', float('nan')):.2f} | "
            f"bootstrap Sharpe CI95 [{boot.get('ci95_low', float('nan')):.2f}, {boot.get('ci95_high', float('nan')):.2f}], "
            f"P(SR<0.5)={boot.get('p_sharpe_lt_0_5', float('nan')):.2f}",
            f"- intraday-trough DD ~{hb['intraday_trough_dd_pct_btc']:.1f}% vs close-basis {hb['close_basis_dd_pct_btc']:.1f}% (BTC); "
            f"portfolio Sortino {port_rs.get('sortino', float('nan')):.2f}",
        ]
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
            r = run(SimConfig(premium_frac=pf_frac, roll_days=roll), bootstrap_stats=False)
            grid.append({"premium_frac": pf_frac, "roll_days": roll,
                         "net_sharpe": r["portfolio"]["net_sharpe"],
                         "net_pf": r["portfolio"]["net_pf"],
                         "net_max_dd": r["portfolio"]["net_max_dd"],
                         "recent_12m_sharpe": r["portfolio"]["recent_12m_sharpe"],
                         "cost_gap_sharpe": r["portfolio"]["cost_gap_sharpe"],
                         "verdict": r["verdict"]})
    base["robustness_grid"] = grid

    # ---- multiplicity-deflated Sharpe (V6-03/V7-02) ----
    # The headline is the best of >=16 graded configs (8-cell turnover×sizing grid × 2
    # assets) + ~12 spread-grid cells. The across-config SR variance comes from the grid;
    # N escalates to fold in the asset + spread-grid multiplicity. The 1.06 barely clears
    # its own multiplicity-corrected hurdle — log DSR so paper out-perf != confirmation.
    grid_sharpes_daily = [g["net_sharpe"] / math.sqrt(ANN) for g in grid]
    for scope, blk, sr in (("BTC", base["per_asset"]["BTC"]["risk_stats"],
                            base["per_asset"]["BTC"]["net_sharpe"]),
                           ("portfolio", base["portfolio"]["risk_stats"],
                            base["portfolio"]["net_sharpe"])):
        blk["deflated_sharpe"] = {
            str(N): _deflated_sharpe(sr / math.sqrt(ANN), grid_sharpes_daily,
                                     blk.get("n_obs", 0), blk.get("skew", 0.0),
                                     blk.get("kurtosis", 3.0), N)
            for N in (8, 16, 28)}

    # ---- honest band (V1-03/V1-09/V6) — RE-DERIVED after the 2026-06-23 de-contamination ----
    # The prior 0.61 "real-chain anchor" was USDC-linear data contamination (the free Tardis
    # chain mixed USD-priced BTC_USDC rows into the coin-priced ATM argmin in 5/63 months, all
    # recent). De-contaminated (inverse-only), the real-chain is NO-GO: straddle -0.25, strangle
    # +0.05, instrument A/B 0/8 (results/options_vrp/skew_verdict.json + instrument_ab_verdict.json).
    # PATH A (this DVOL-synthetic core) is a SEPARATE, un-contaminated data path and still
    # reproduces ~1.06/0.77, but it has LOST its only real-price corroboration, so the honest read
    # is the multiplicity-deflated DSR, NOT a 0.6-0.9 deploy band. Sleeve BLOCKED, clearance RESCINDED.
    btc = base["per_asset"]["BTC"]
    base["honest_band"] = {
        "headline_best_case_btc_net_sharpe": btc["net_sharpe"],
        "portfolio_pre_registered_net_sharpe": base["portfolio"]["net_sharpe"],
        "real_chain_anchor_net_sharpe": None,             # VOID: was 0.61 = USDC-linear contamination
        "real_chain_clean_straddle_net_sharpe": -0.25,    # de-contaminated, NO-GO (skew_verdict.json)
        "real_chain_status": "NO-GO (de-contaminated 2026-06-23; inverse-only; straddle -0.25 / strangle +0.05 / A-B 0-8)",
        "honest_expectation_band": None,                  # no real-execution-validated band
        "deflated_sharpe_read": "DSR ~0.27-0.37 portfolio / ~0.51-0.61 BTC (N=16-28); bootstrap P(SR<0.5)=0.29",
        "deploy_status": "BLOCKED flag-off; revival needs PAID daily-chain 21d-roll real-chain re-validation (net Sharpe >=~0.5 AND >0 OOS), else retire",
        "best_case_ceiling": 1.10,
        "close_basis_dd_pct_btc": btc["net_max_dd"] * 100.0,
        "intraday_trough_dd_pct_btc": 8.42,               # reconstructed high/low (tail recompute; ~1.6x close)
        "note": ("DO NOT DEPLOY. The 0.61 real-chain anchor was USDC-linear contamination "
                 "(5/63 months, all recent); de-contaminated real-chain short-vol is NO-GO on "
                 "BTC+ETH across straddle/strangle/iron_fly/iron_condor. This DVOL-synthetic "
                 "1.06/0.77 is UNCONFIRMED by real execution and multiplicity-deflated to DSR "
                 "~0.3-0.5; left tail real (skew<0, Sortino<Sharpe, bootstrap CI95 spans <0.5); "
                 "intraday-trough DD ~8.4% vs close-basis. Paper out-performance is NOT confirmation."),
        "_provenance": "docs/research/options_vrp_decontamination_reaudit_2026-06-23.md",
    }

    # ---- provenance (V1-07): the data fingerprint a --refresh would change ----
    base["provenance"] = {
        "data_sha256": _data_fingerprint(),
        "gates_file": "configs/options_vol_harvest.gates.yaml",
        "gates_source": "yaml-loaded (not hardcoded)",
        "note": ("Commit verdict.json + this fingerprint together; the loader writes a "
                 ".bak before any --refresh overwrites the graded parquets."),
    }

    (RESULTS_DIR / "verdict.json").write_text(
        json.dumps(base, indent=2, default=str), encoding="utf-8")
    summary = _fmt(base)
    grid_md = "\n".join(
        f"- premium_frac={g['premium_frac']}, roll={g['roll_days']}d: "
        f"Sharpe {g['net_sharpe']:.2f}, PF {g['net_pf']:.2f}, DD {g['net_max_dd']:.1%}, "
        f"recent {g['recent_12m_sharpe']:.2f} -> {g['verdict']}" for g in grid)
    summary += "\n\n## Robustness grid\n" + grid_md
    (RESULTS_DIR / "summary.md").write_text(summary, encoding="utf-8")

    logger.info("\n%s", summary)
    logger.info("Wrote %s and %s", RESULTS_DIR / "verdict.json", RESULTS_DIR / "summary.md")


if __name__ == "__main__":
    main()
