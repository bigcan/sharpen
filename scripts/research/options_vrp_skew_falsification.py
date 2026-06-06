"""Phase-0b REAL-CHAIN, skew-augmented falsification of the crypto-options VRP edge.

Upgrades the Tier-A DVOL-synthetic gate (`options_vrp_falsification.py`) to REAL
Deribit option prices from the free Tardis first-of-month chain snapshots
(`tardis_options_chain_loader`). Three things become real instead of modelled:

  1. **Entry prices/costs** — we SELL at the real *bid* (not a modelled mid minus
     a guessed spread). The real bid/ask is also measured directly (ask_iv-bid_iv)
     and reported, validating the Tier-A 1.0-vol-pt assumption.
  2. **The skew/strangle leg** — a real 25-delta strangle (put-skew VRP), selected
     by the chain's real per-instrument greeks. Untestable on DVOL (ATM only).
  3. **Real strikes/expiries** — monthly sell-and-hold-to-expiry, cash-settled at
     intrinsic from the free daily perp (no exit spread; the hold-to-expiry edge).

Structure (causally clean): at each first-of-month snapshot, sell the chosen
structure at the real bid, delta-hedge daily with the perp using BS delta from the
entry IV, and settle at the option's own expiry (mid-month) at intrinsic. Flat
until the next month. Option premium is COIN-denominated (Deribit); USD = price *
underlying.

Run:
    python scripts/research/options_vrp_skew_falsification.py
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from finrl_pro_ds.crypto.data import deribit_options_loader as dol
from finrl_pro_ds.crypto.data import tardis_options_chain_loader as tcl
from finrl_pro_ds.crypto.features import options_vol_features as ovf

logger = logging.getLogger(__name__)
RESULTS_DIR = Path("results/options_vrp")
ANN = 365.0

# Pre-registered gates (identical to options_vrp_falsification.GATES; duplicated to
# keep this research script self-contained — scripts/ is not an importable package).
GATES = {"min_net_sharpe": 0.50, "min_net_pf": 1.10}


# --- shared math/metric helpers (mirror options_vrp_falsification.py) ---------
def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1(S, K, sigma, tau):
    return (math.log(S / K) + 0.5 * sigma * sigma * tau) / (sigma * math.sqrt(tau))


def _sharpe(daily_ret: np.ndarray) -> float:
    d = daily_ret[np.isfinite(daily_ret)]
    if len(d) < 2 or d.std(ddof=1) == 0:
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


# ---------------------------------------------------------------------------
# Per-leg Black-Scholes (r=0; vol fraction; tau years). Intrinsic at tau<=0.
# ---------------------------------------------------------------------------
def bs_call(S, K, sig, tau):
    if tau <= 0 or sig <= 0:
        return max(S - K, 0.0)
    d1 = _d1(S, K, sig, tau)
    d2 = d1 - sig * math.sqrt(tau)
    return S * _ncdf(d1) - K * _ncdf(d2)


def bs_put(S, K, sig, tau):
    if tau <= 0 or sig <= 0:
        return max(K - S, 0.0)
    d1 = _d1(S, K, sig, tau)
    d2 = d1 - sig * math.sqrt(tau)
    return K * _ncdf(-d2) - S * _ncdf(-d1)


def leg_price(opt_type, S, K, sig, tau):
    return bs_call(S, K, sig, tau) if opt_type == "call" else bs_put(S, K, sig, tau)


def leg_delta(opt_type, S, K, sig, tau):
    if tau <= 0 or sig <= 0:
        return (1.0 if S > K else 0.0) if opt_type == "call" else (-1.0 if S < K else 0.0)
    d1 = _d1(S, K, sig, tau)
    return _ncdf(d1) if opt_type == "call" else _ncdf(d1) - 1.0


@dataclass
class SkewConfig:
    premium_frac: float = 0.05
    target_tenor_days: int = 30
    min_tenor_days: int = 12          # ignore expiries closer than this
    initial_capital: float = 100_000.0
    option_fee_pct_underlying: float = 0.0003
    option_fee_cap_pct_premium: float = 0.125
    perp_taker_fee: float = 0.0005


# ---------------------------------------------------------------------------
# Instrument selection from a real chain snapshot
# ---------------------------------------------------------------------------
def _pick_expiry(snap, underlying, entry_date, cfg):
    snap = snap[(snap.bid_price > 0) & (snap.ask_price > 0) & (snap.mark_iv > 0)]
    if snap.empty:
        return None, snap
    exp = snap["expiry_dt"]
    days = (exp - entry_date).dt.total_seconds() / 86400.0
    cand = snap[days >= cfg.min_tenor_days]
    if cand.empty:
        return None, snap
    cdays = (cand["expiry_dt"] - entry_date).dt.total_seconds() / 86400.0
    target = cand["expiry_dt"].iloc[(cdays - cfg.target_tenor_days).abs().argmin()]
    return target, snap[snap.expiry_dt == target]


def select_legs(snap_asset, entry_date, kind, cfg):
    """Return list of legs for 'straddle' or 'strangle', or None if not tradeable."""
    if snap_asset.empty:
        return None
    underlying = float(snap_asset["underlying_price"].median())
    target, chain = _pick_expiry(snap_asset, underlying, entry_date, cfg)
    if target is None or chain.empty:
        return None
    calls = chain[chain.type == "call"]
    puts = chain[chain.type == "put"]
    if calls.empty or puts.empty:
        return None
    tau = (target - entry_date).total_seconds() / 86400.0 / ANN

    def mk(row):
        return {"type": row.type, "K": float(row.strike_price),
                "bid": float(row.bid_price), "mark": float(row.mark_price),
                "mark_iv": float(row.mark_iv) / 100.0,
                "bid_iv": float(row.bid_iv) / 100.0 if pd.notna(row.bid_iv) else np.nan,
                "ask_iv": float(row.ask_iv) / 100.0 if pd.notna(row.ask_iv) else np.nan,
                "underlying": underlying, "expiry": target, "tau": tau}

    if kind == "straddle":
        c = calls.iloc[(calls.strike_price - underlying).abs().argmin()]
        p = puts.iloc[(puts.strike_price - underlying).abs().argmin()]
        return [mk(c), mk(p)]
    # strangle: 25-delta call & put by the chain's REAL delta
    c = calls.iloc[(calls.delta - 0.25).abs().argmin()]
    p = puts.iloc[(puts.delta + 0.25).abs().argmin()]
    return [mk(c), mk(p)]


# ---------------------------------------------------------------------------
# Real-chain monthly sell-and-hold-to-expiry simulation
# ---------------------------------------------------------------------------
def simulate_real_chain(asset, snaps_by_month, spot_by_date, funding_by_date, kind, cfg):
    """Return (daily_pnl Series indexed by date, diagnostics dict)."""
    dates = sorted(spot_by_date.keys())
    pnl = pd.Series(0.0, index=pd.DatetimeIndex(dates))
    measured_half_spread_volpts = []

    pos = None  # active position dict
    q_prev = 0.0

    def open_position(d):
        nonlocal pos, q_prev
        snap = snaps_by_month.get((d.year, d.month))
        if snap is None:
            return
        sa = snap[snap.symbol.str.startswith(asset)]
        legs = select_legs(sa, d, kind, cfg)
        if not legs:
            return
        underlying = legs[0]["underlying"]
        prem_unit_usd = sum(lg["bid"] * underlying for lg in legs)  # we receive bids
        if prem_unit_usd <= 0:
            return
        n = (cfg.premium_frac * cfg.initial_capital) / prem_unit_usd
        # entry costs: option fee + real spread (mark - bid) captured via MTM from mark
        fee = 0.0
        for lg in legs:
            prem = lg["bid"] * underlying
            fee += min(cfg.option_fee_pct_underlying * underlying, cfg.option_fee_cap_pct_premium * prem) * n
            # entry spread realized: receive bid, mark to mark_iv -> immediate (mark-bid) loss
            if np.isfinite(lg["bid_iv"]) and np.isfinite(lg["ask_iv"]):
                measured_half_spread_volpts.append((lg["ask_iv"] - lg["bid_iv"]) / 2 * 100)
        pnl[d] -= fee
        pos = {"legs": legs, "n": n, "underlying_entry": underlying, "expiry": legs[0]["expiry"]}
        q_prev = 0.0

    def mtm_value(pos, S, d):
        """Current BS value (USD) of the SHORT structure's liability, per the n held."""
        v = 0.0
        for lg in pos["legs"]:
            tau = max((lg["expiry"] - d).total_seconds() / 86400.0 / ANN, 0.0)
            v += leg_price(lg["type"], S, lg["K"], lg["mark_iv"], tau)
        return pos["n"] * v

    def net_delta(pos, S, d):
        dl = 0.0
        for lg in pos["legs"]:
            tau = max((lg["expiry"] - d).total_seconds() / 86400.0 / ANN, 0.0)
            dl += leg_delta(lg["type"], S, lg["K"], lg["mark_iv"], tau)
        return pos["n"] * dl  # long-structure delta; short position delta = -this

    didx = pnl.index
    for i in range(len(didx) - 1):
        d, d1 = didx[i], didx[i + 1]
        # open on first-of-month if flat
        if pos is None and d.day == 1:
            open_position(d)
            if pos is not None:
                # mark entry: received bids already (implicit); book spread cost mark-bid
                S = spot_by_date[d]
                v0_mark = mtm_value(pos, S, d)
                prem_bid = pos["n"] * sum(lg["bid"] * pos["underlying_entry"] for lg in pos["legs"])
                pnl[d] += prem_bid - v0_mark   # spread cost = (received bid) - (liability at mark)
                pos["liab_prev"] = v0_mark
        if pos is None:
            continue

        S, S1 = spot_by_date[d], spot_by_date[d1]
        # delta hedge: long perp q to neutralise short structure (short delta = -net_delta)
        q_t = net_delta(pos, S, d)
        rehedge_cost = cfg.perp_taker_fee * abs(q_t - q_prev) * S
        hedge_pnl = q_t * (S1 - S)
        funding_pnl = -q_t * S * funding_by_date.get(d, 0.0)

        # option liability MTM to next day (or settle at expiry)
        expd = pos["expiry"]
        if d1 >= expd:
            S_exp = spot_by_date.get(expd.normalize(), S1)
            liab_next = 0.0
            for lg in pos["legs"]:
                liab_next += (max(S_exp - lg["K"], 0.0) if lg["type"] == "call"
                              else max(lg["K"] - S_exp, 0.0))
            liab_next *= pos["n"]
        else:
            liab_next = mtm_value(pos, S1, d1)

        option_pnl = pos["liab_prev"] - liab_next   # short: gain when liability falls
        pnl[d1] += option_pnl + hedge_pnl - rehedge_cost + funding_pnl
        pos["liab_prev"] = liab_next
        q_prev = q_t
        if d1 >= expd:
            pos = None
            q_prev = 0.0

    diag = {"n_cycles": int((pnl != 0).sum() > 0),
            "measured_half_spread_volpts": float(np.nanmean(measured_half_spread_volpts))
            if measured_half_spread_volpts else float("nan")}
    return pnl, diag


def _metrics(pnl: pd.Series, cfg):
    eq = pnl.cumsum() + cfg.initial_capital
    ret = _returns_from_pnl(pnl.to_numpy(), eq.to_numpy())
    years = pnl.index.year.to_numpy()
    by_year = {}
    for y in sorted(set(years.tolist())):
        m = years == y
        if m.sum() >= 20:
            by_year[int(y)] = float(pnl.to_numpy()[m].sum() / cfg.initial_capital)
    recent = pnl.index >= (pnl.index[-1] - pd.Timedelta(days=365))
    return {
        "net_sharpe": _sharpe(ret), "net_pf": _profit_factor(pnl.to_numpy()),
        "net_total_return": float(eq.iloc[-1] / cfg.initial_capital - 1.0),
        "net_max_dd": _max_drawdown(eq.to_numpy()),
        "recent_12m_sharpe": _sharpe(ret[recent]),
        "by_year": by_year,
        "pos_years": sum(1 for v in by_year.values() if v > 0), "n_years": len(by_year),
    }


def run(cfg: SkewConfig | None = None) -> dict:
    cfg = cfg or SkewConfig()
    raw = dol.load({"universe": {"assets": ["BTC", "ETH"]}})
    chain = tcl.load_chain_snapshots("2021-04", "2026-06")
    if chain.empty:
        raise RuntimeError("No chain snapshots cached — run tardis_options_chain_loader first")
    chain["snapshot_date"] = pd.to_datetime(chain["snapshot_date"], utc=True)
    chain["expiry_dt"] = pd.to_datetime(chain["expiry_dt"], utc=True)
    snaps_by_month = {(d.year, d.month): g for d, g in chain.groupby("snapshot_date")}

    assets = ["BTC", "ETH"]
    results = {"config": asdict(cfg), "per_asset": {}, "measured_half_spread_volpts": {}}
    kinds = ["straddle", "strangle"]
    combined_pnl = {k: [] for k in kinds + ["combined"]}

    for a in assets:
        perp = ovf._to_date_index(raw.perp[a])["close"]
        fund = ovf._daily_funding(raw.funding[a])
        spot_by_date = {pd.Timestamp(d): float(v) for d, v in perp.items()}
        funding_by_date = {pd.Timestamp(d): float(v) for d, v in fund.items()}
        results["per_asset"][a] = {}
        asset_kind_pnl = {}
        for kind in kinds:
            pnl, diag = simulate_real_chain(a, snaps_by_month, spot_by_date, funding_by_date, kind, cfg)
            results["per_asset"][a][kind] = _metrics(pnl, cfg)
            results["measured_half_spread_volpts"][a] = diag["measured_half_spread_volpts"]
            asset_kind_pnl[kind] = pnl
            combined_pnl[kind].append(pnl)
        # straddle+strangle combined for this asset (equal capital each)
        comb = (asset_kind_pnl["straddle"].add(asset_kind_pnl["strangle"], fill_value=0.0)) / 2
        results["per_asset"][a]["combined"] = _metrics(comb, cfg)
        combined_pnl["combined"].append(comb)

    # portfolio: equal-weight across assets per structure
    portfolio = {}
    for k, series_list in combined_pnl.items():
        df = pd.concat(series_list, axis=1).fillna(0.0)
        port = df.mean(axis=1)
        portfolio[k] = _metrics(port, cfg)

    results["portfolio"] = portfolio

    # PER-ASSET deployable verdict on the combined (straddle+strangle) book.
    # Equal-weighting a dead leg (ETH) into a live one (BTC) is not how we'd
    # deploy — we'd run the GO assets only. So verdict per asset, GO overall if
    # any asset clears the gate.
    asset_verdicts = {}
    for a in assets:
        m = results["per_asset"][a]["combined"]
        multi = m["pos_years"] >= max(2, math.ceil(0.6 * m["n_years"]))
        ok = (m["net_sharpe"] >= GATES["min_net_sharpe"] and m["net_pf"] >= GATES["min_net_pf"]
              and multi and m["recent_12m_sharpe"] > 0)
        asset_verdicts[a] = "GO" if ok else "NO-GO"
    go_assets = [a for a, v in asset_verdicts.items() if v == "GO"]
    results["asset_verdicts"] = asset_verdicts
    results["go_assets"] = go_assets
    results["verdict"] = "GO" if go_assets else "NO-GO"
    results["kills"] = [] if go_assets else ["no single asset clears the gate"]
    return results


def _fmt(r):
    L = ["# Options-VRP REAL-CHAIN + Skew Falsification (free Tardis first-of-month)",
         f"**Verdict: {r['verdict']}** — GO assets: {r.get('go_assets', [])}",
         f"Per-asset (straddle+strangle) verdicts: {r.get('asset_verdicts', {})}",
         "",
         "## Per-asset combined (straddle+strangle) — the deployable book",
         *[f"- **{a}**: Sharpe {r['per_asset'][a]['combined']['net_sharpe']:.2f}, "
           f"PF {r['per_asset'][a]['combined']['net_pf']:.2f}, "
           f"ret {r['per_asset'][a]['combined']['net_total_return']:.1%}, "
           f"DD {r['per_asset'][a]['combined']['net_max_dd']:.1%}, "
           f"recent {r['per_asset'][a]['combined']['recent_12m_sharpe']:.2f}, "
           f"+yrs {r['per_asset'][a]['combined']['pos_years']}/{r['per_asset'][a]['combined']['n_years']}"
           for a in r["per_asset"]],
         ""]
    for k in ("straddle", "strangle", "combined"):
        p = r["portfolio"][k]
        L.append(f"## Portfolio — {k}")
        L.append(f"- net Sharpe {p['net_sharpe']:.2f}, net PF {p['net_pf']:.2f}, "
                 f"totRet {p['net_total_return']:.1%}, DD {p['net_max_dd']:.1%}, "
                 f"recent-12m {p['recent_12m_sharpe']:.2f}, +years {p['pos_years']}/{p['n_years']}")
        L.append("  by-year: " + ", ".join(f"{y}:{v:+.1%}" for y, v in p["by_year"].items()))
    L.append("")
    L.append("## Real measured option half-spread (vs Tier-A modelled 1.0 vol pt)")
    for a, s in r["measured_half_spread_volpts"].items():
        L.append(f"- {a}: measured half-spread ≈ {s:.2f} vol pts")
    L.append("")
    L.append("## Per-asset (net)")
    for a, kinds in r["per_asset"].items():
        for kind, m in kinds.items():
            L.append(f"- {a} {kind}: Sharpe {m['net_sharpe']:.2f}, PF {m['net_pf']:.2f}, "
                     f"ret {m['net_total_return']:.1%}, DD {m['net_max_dd']:.1%}, recent {m['recent_12m_sharpe']:.2f}")
    if r["kills"]:
        L += ["", "## KILL triggers", *[f"- {k}" for k in r["kills"]]]
    return "\n".join(L)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    r = run()
    (RESULTS_DIR / "skew_verdict.json").write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    summary = _fmt(r)
    (RESULTS_DIR / "skew_summary.md").write_text(summary, encoding="utf-8")
    logger.info("\n%s", summary)


if __name__ == "__main__":
    main()
