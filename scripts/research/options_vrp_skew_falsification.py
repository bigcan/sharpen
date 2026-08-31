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
import yaml

from sharpen.crypto.data import deribit_options_loader as dol
from sharpen.crypto.data import tardis_options_chain_loader as tcl
from sharpen.crypto.eval.statistics import (
    block_bootstrap_sortino_ci,
    sortino_ratio,
)
from sharpen.crypto.features import options_vol_features as ovf
from sharpen.crypto.options_pricing import leg_delta, leg_price, leg_vega

logger = logging.getLogger(__name__)
RESULTS_DIR = Path("results/options_vrp")
ANN = 365.0

# Gate thresholds live in configs/options_vol_harvest.gates.yaml — NEVER hardcoded here
# (CLAUDE.md invariant). We read phase1_linear + the short-vol tail kills + the
# instrument_ab A/B block. Resolved relative to this script so the cwd doesn't matter.
GATES_FILE = Path(__file__).resolve().parents[2] / "configs" / "options_vol_harvest.gates.yaml"


def load_gates(gates_file: Path | str = GATES_FILE) -> dict:
    """Read the pre-registered phase-1 + tail + instrument_ab thresholds from the yaml.
    The ``.get`` defaults are a last-resort for a missing key, NOT a second source of
    truth (mirrors options_vrp_falsification.load_gates)."""
    g = yaml.safe_load(Path(gates_file).read_text(encoding="utf-8")).get("gates", {})
    p1 = g.get("phase1_linear", {})
    tail = g.get("tail", {})
    ab = g.get("instrument_ab", {})
    return {
        "min_net_sharpe": float(p1.get("min_net_sharpe", 0.50)),
        "min_net_pf": float(p1.get("min_net_pf", 1.10)),
        "require_multi_subperiod": bool(p1.get("require_multi_subperiod", True)),
        "max_recent_oos_drawdown": float(p1.get("max_recent_oos_drawdown", 0.40)),
        "min_recent_oos_sharpe": float(p1.get("min_recent_oos_sharpe", 0.0)),
        "max_worst_window_dd_pct": float(tail.get("max_worst_window_dd_pct", 25.0)),
        "max_net_vega_per_100k": float(tail.get("max_net_vega_per_100k", 50000.0)),
        "cvar95_floor_pct": float(tail.get("cvar95_floor_pct", -8.0)),
        "instruments": list(ab.get("instruments", ["straddle", "strangle", "iron_fly", "iron_condor"])),
        "wing_delta": float(ab.get("wing_delta", 0.10)),
        "require_pass_phase1": bool(ab.get("require_pass_phase1", True)),
        "select_by": str(ab.get("select_by", "sortino")),
        "min_sortino": float(ab.get("min_sortino", 0.50)),
        "sample_basis": str(ab.get("sample_basis", "tardis_first_of_month")),
    }


GATES = load_gates()


# --- shared math/metric helpers (mirror options_vrp_falsification.py) ---------
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


@dataclass
class SkewConfig:
    premium_frac: float = 0.05
    target_tenor_days: int = 30
    min_tenor_days: int = 12          # ignore expiries closer than this
    initial_capital: float = 100_000.0
    option_fee_pct_underlying: float = 0.0003
    option_fee_cap_pct_premium: float = 0.125
    perp_taker_fee: float = 0.0005
    wing_delta: float = 0.10          # |delta| of the long wings for iron_fly/iron_condor


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


SHORT_DELTA = 0.25  # the strangle / iron_condor short-leg target |delta|
STRUCTURES = ("straddle", "strangle", "iron_fly", "iron_condor")
# DATA-CLEAN toggle: trade only Deribit INVERSE (coin-settled) options. The free
# Tardis chain mixes USDC-LINEAR duplicates (USD-priced) from 2024-03; left in, they
# leak into the strike/delta argmin and (mis-scaled as coin) inject spurious P&L. Keep
# True for honest results — False only to reproduce the pre-fix contaminated anchor.
INVERSE_ONLY = True


def build_structure(snap_asset, entry_date, kind, cfg):
    """Signed legs for a kind in STRUCTURES, or None if not tradeable that month.

    Each leg carries ``side`` (-1 short = sell at bid, +1 long = buy at ask) and both
    ``bid`` and ``ask``. The shorts are ATM (straddle/iron_fly) or 25-delta
    (strangle/iron_condor); iron_fly/iron_condor add long OTM wings at ``cfg.wing_delta``
    selected by the chain's REAL per-instrument delta (same mechanism as the 25-delta
    short). straddle/strangle are all-short and BYTE-IDENTICAL to the prior
    ``select_legs`` (the refactor-parity anchor on skew_verdict.json).
    """
    if kind not in STRUCTURES:
        raise ValueError(f"unknown structure {kind!r}; expected one of {STRUCTURES}")
    if snap_asset.empty:
        return None
    # DATA-CLEAN: keep only Deribit INVERSE (coin-settled) instruments. From 2024-03
    # the free Tardis chain also carries USDC-LINEAR duplicates (BTC_USDC-...) quoted in
    # USD, not coin — mixing them breaks the "USD = price * underlying" convention. The
    # 10-delta wing grabbed the USD row and blew up; the ATM short also hit it in 5/63
    # months, mis-sizing n and injecting a spurious ~+5%/cycle that INFLATED the prior
    # 0.61 anchor. Inverse symbols carry no underscore; USDC/USDT-linear do.
    if INVERSE_ONLY:
        snap_asset = snap_asset[~snap_asset["symbol"].str.contains("_", regex=False)]
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

    def mk(row, side):
        return {"type": row.type, "K": float(row.strike_price), "side": int(side),
                "bid": float(row.bid_price), "ask": float(row.ask_price),
                "mark": float(row.mark_price),
                "mark_iv": float(row.mark_iv) / 100.0,
                "bid_iv": float(row.bid_iv) / 100.0 if pd.notna(row.bid_iv) else np.nan,
                "ask_iv": float(row.ask_iv) / 100.0 if pd.notna(row.ask_iv) else np.nan,
                "underlying": underlying, "expiry": target, "tau": tau}

    # SHORT legs: ATM-by-strike (straddle/iron_fly) or 25-delta (strangle/iron_condor).
    if kind in ("straddle", "iron_fly"):
        sc = calls.iloc[(calls.strike_price - underlying).abs().argmin()]
        sp = puts.iloc[(puts.strike_price - underlying).abs().argmin()]
    else:
        sc = calls.iloc[(calls.delta - SHORT_DELTA).abs().argmin()]
        sp = puts.iloc[(puts.delta + SHORT_DELTA).abs().argmin()]
    legs = [mk(sc, -1), mk(sp, -1)]
    if kind in ("straddle", "strangle"):
        return legs

    # IRON FLY / CONDOR: add long OTM wings that cap the tail. Wings strictly OTM of
    # the 25-delta shorts (so 0 < wing_delta < SHORT_DELTA) and strictly OUTSIDE the
    # short strikes — else this month's chain can't form the spread (return None).
    wd = cfg.wing_delta
    if not (0.0 < wd < SHORT_DELTA):
        raise ValueError(f"wing_delta must be in (0, {SHORT_DELTA}); got {wd}")
    wc = calls.iloc[(calls.delta - wd).abs().argmin()]
    wp = puts.iloc[(puts.delta + wd).abs().argmin()]
    if not (float(wc.strike_price) > float(sc.strike_price)
            and float(wp.strike_price) < float(sp.strike_price)):
        logger.debug("%s %s: no valid wings beyond shorts (call %s>%s, put %s<%s) — skip",
                     entry_date.date(), kind, wc.strike_price, sc.strike_price,
                     wp.strike_price, sp.strike_price)
        return None
    legs += [mk(wc, +1), mk(wp, +1)]
    return legs


# ---------------------------------------------------------------------------
# Real-chain monthly sell-and-hold-to-expiry simulation
# ---------------------------------------------------------------------------
def simulate_real_chain(asset, snaps_by_month, spot_by_date, funding_by_date, kind, cfg):
    """Return (daily_pnl Series indexed by date, diagnostics dict).

    SIGNED multi-leg accounting: every per-leg sum carries the leg's ``side`` so long
    wings (side=+1) net AGAINST the shorts (side=-1) on premium, MTM, delta, vega and
    intrinsic settlement. ``n`` is sized off the SHORT premium received (gross of
    wings) so the short vega exposure is comparable across structures and the wings
    then SHRINK |net vega| and CAP the settlement loss. For all-short structures
    (straddle/strangle) this reduces exactly to the prior unsigned engine.
    """
    dates = sorted(spot_by_date.keys())
    pnl = pd.Series(0.0, index=pd.DatetimeIndex(dates))
    measured_half_spread_volpts = []
    net_vega_per_100k = []   # |signed net vega| per $100k, daily, for the tail gate
    opens = []               # entry dates of opened cycles (the realized sample size)

    pos = None  # active position dict
    q_prev = 0.0

    def open_position(d):
        nonlocal pos, q_prev
        snap = snaps_by_month.get((d.year, d.month))
        if snap is None:
            return
        sa = snap[snap.symbol.str.startswith(asset)]
        legs = build_structure(sa, d, kind, cfg)
        if not legs:
            return
        underlying = legs[0]["underlying"]
        # size off the SHORT premium received (gross of wings) — keeps short vega
        # comparable across structures; identical to prior n for all-short kinds.
        short_prem_unit_usd = sum(lg["bid"] * underlying for lg in legs if lg["side"] == -1)
        if short_prem_unit_usd <= 0:
            return
        n = (cfg.premium_frac * cfg.initial_capital) / short_prem_unit_usd
        # entry costs: option fee (BOTH sides — 4-leg irons pay ~2x a straddle) +
        # the real spread, captured below via MTM from mark vs the price transacted.
        fee = 0.0
        for lg in legs:
            entry_price = lg["bid"] if lg["side"] == -1 else lg["ask"]
            prem = entry_price * underlying
            fee += min(cfg.option_fee_pct_underlying * underlying, cfg.option_fee_cap_pct_premium * prem) * n
            if np.isfinite(lg["bid_iv"]) and np.isfinite(lg["ask_iv"]):
                measured_half_spread_volpts.append((lg["ask_iv"] - lg["bid_iv"]) / 2 * 100)
        pnl[d] -= fee
        pos = {"legs": legs, "n": n, "underlying_entry": underlying, "expiry": legs[0]["expiry"]}
        q_prev = 0.0

    def opt_value(pos, S, d):
        """SIGNED mark-to-market USD value of the option book to us (long +, short -)."""
        v = 0.0
        for lg in pos["legs"]:
            tau = max((lg["expiry"] - d).total_seconds() / 86400.0 / ANN, 0.0)
            v += lg["side"] * leg_price(lg["type"], S, lg["K"], lg["mark_iv"], tau)
        return pos["n"] * v

    def position_delta(pos, S, d):
        """SIGNED option-book delta (long +, short -); the perp hedge holds -this."""
        dl = 0.0
        for lg in pos["legs"]:
            tau = max((lg["expiry"] - d).total_seconds() / 86400.0 / ANN, 0.0)
            dl += lg["side"] * leg_delta(lg["type"], S, lg["K"], lg["mark_iv"], tau)
        return pos["n"] * dl

    def net_vega_usd(pos, S, d):
        """SIGNED net vega (USD per 1.00 vol): long wings (+) shrink short vega (-)."""
        vg = 0.0
        for lg in pos["legs"]:
            tau = max((lg["expiry"] - d).total_seconds() / 86400.0 / ANN, 0.0)
            vg += lg["side"] * leg_vega(lg["type"], S, lg["K"], lg["mark_iv"], tau)
        return pos["n"] * vg

    didx = pnl.index
    for i in range(len(didx) - 1):
        d, d1 = didx[i], didx[i + 1]
        # open on first-of-month if flat
        if pos is None and d.day == 1:
            open_position(d)
            if pos is not None:
                # book the entry: net credit received (+bid shorts, -ask longs) plus the
                # signed mark value => the realized bid/ask spread cost is baked in.
                S = spot_by_date[d]
                ov0 = opt_value(pos, S, d)
                net_credit = pos["n"] * sum(
                    (-lg["side"]) * (lg["bid"] if lg["side"] == -1 else lg["ask"])
                    * pos["underlying_entry"] for lg in pos["legs"])
                pnl[d] += net_credit + ov0
                pos["opt_value_prev"] = ov0
                opens.append(d)
        if pos is None:
            continue

        S, S1 = spot_by_date[d], spot_by_date[d1]
        # delta hedge with the perp: hold q = -(signed option-book delta)
        q_t = -position_delta(pos, S, d)
        rehedge_cost = cfg.perp_taker_fee * abs(q_t - q_prev) * S
        hedge_pnl = q_t * (S1 - S)
        funding_pnl = -q_t * S * funding_by_date.get(d, 0.0)
        net_vega_per_100k.append(abs(net_vega_usd(pos, S, d)) / cfg.initial_capital * 1e5)

        # option book MTM to next day (or signed intrinsic settlement at expiry)
        expd = pos["expiry"]
        if d1 >= expd:
            S_exp = spot_by_date.get(expd.normalize(), S1)
            ov_next = pos["n"] * sum(
                lg["side"] * (max(S_exp - lg["K"], 0.0) if lg["type"] == "call"
                              else max(lg["K"] - S_exp, 0.0))
                for lg in pos["legs"])
        else:
            ov_next = opt_value(pos, S1, d1)

        option_pnl = ov_next - pos["opt_value_prev"]   # gain when the book's value rises
        pnl[d1] += option_pnl + hedge_pnl - rehedge_cost + funding_pnl
        pos["opt_value_prev"] = ov_next
        q_prev = q_t
        if d1 >= expd:
            pos = None
            q_prev = 0.0

    diag = {"n_cycles_opened": len(opens),
            "measured_half_spread_volpts": float(np.nanmean(measured_half_spread_volpts))
            if measured_half_spread_volpts else float("nan"),
            "max_net_vega_per_100k": float(np.max(net_vega_per_100k)) if net_vega_per_100k else 0.0,
            "median_net_vega_per_100k": float(np.median(net_vega_per_100k)) if net_vega_per_100k else 0.0}
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


# ---------------------------------------------------------------------------
# Instrument A/B: tail-adjusted (Sortino/CVaR) metrics + per-instrument gate + selection
# (.agent/artifacts/options_vrp_instrument_ab_spec.md). Each instrument clears
# phase1_linear + the short-vol tail on its OWN (no inheritance); the winner among
# passers is the highest-Sortino cell that also clears the Sortino floor + CVaR guard.
# ---------------------------------------------------------------------------
def _metrics_ab(pnl: pd.Series, diag: dict, cfg) -> dict:
    """``_metrics`` + the tail-adjusted stats the A/B selects on (Sortino primary, CVaR95
    guard) and the short-vol tail-gate inputs (worst-window DD, net vega, sample size)."""
    m = _metrics(pnl, cfg)
    eq = pnl.cumsum() + cfg.initial_capital
    ret = _returns_from_pnl(pnl.to_numpy(), eq.to_numpy())
    r = ret[np.isfinite(ret)]
    sortino = float(sortino_ratio(r.tolist(), periods_per_year=int(ANN))) if len(r) > 2 else 0.0
    if len(r) >= 3:
        v95 = float(np.quantile(r, 0.05))
        cvar95 = float(r[r <= v95].mean()) * 100.0
    else:
        cvar95 = 0.0
    m.update({
        "sortino": sortino,
        "cvar95_pct_daily": cvar95,
        "worst_window_dd_pct": m["net_max_dd"] * 100.0,
        "max_net_vega_per_100k": float(diag.get("max_net_vega_per_100k", 0.0)),
        "sample_n_months": int(diag.get("n_cycles_opened", 0)),
        "bootstrap_sortino_ci": block_bootstrap_sortino_ci(r.tolist(), periods_per_year=int(ANN)),
    })
    return m


def _instrument_verdict(m: dict, gates: dict) -> dict:
    """Score ONE (asset, instrument) cell against phase1_linear + tail, independently —
    a different payoff inherits nothing from the straddle (ADR-3)."""
    kills = []
    multi = m["pos_years"] >= max(2, math.ceil(0.6 * m["n_years"])) if m["n_years"] else False
    if m["net_sharpe"] < gates["min_net_sharpe"]:
        kills.append(f"net Sharpe {m['net_sharpe']:.2f} < {gates['min_net_sharpe']}")
    if m["net_pf"] < gates["min_net_pf"]:
        kills.append(f"net PF {m['net_pf']:.2f} < {gates['min_net_pf']}")
    if gates["require_multi_subperiod"] and not multi:
        kills.append(f"not multi-subperiod ({m['pos_years']}/{m['n_years']} yrs +)")
    if m["recent_12m_sharpe"] <= gates["min_recent_oos_sharpe"]:
        kills.append(f"recent-12m Sharpe {m['recent_12m_sharpe']:.2f} <= {gates['min_recent_oos_sharpe']}")
    if m["net_max_dd"] > gates["max_recent_oos_drawdown"]:
        kills.append(f"worst DD {m['net_max_dd']:.1%} > {gates['max_recent_oos_drawdown']:.0%}")
    if m["worst_window_dd_pct"] > gates["max_worst_window_dd_pct"]:
        kills.append(f"worst-window DD {m['worst_window_dd_pct']:.1f}% > {gates['max_worst_window_dd_pct']}%")
    if m["max_net_vega_per_100k"] > gates["max_net_vega_per_100k"]:
        kills.append(f"net vega/100k {m['max_net_vega_per_100k']:,.0f} > {gates['max_net_vega_per_100k']:,.0f}")
    if m["cvar95_pct_daily"] < gates["cvar95_floor_pct"]:
        kills.append(f"CVaR95 {m['cvar95_pct_daily']:.2f}% < floor {gates['cvar95_floor_pct']}%")
    return {"verdict": "GO" if not kills else "NO-GO", "kills": kills,
            "multi_subperiod": bool(multi)}


# Ranking-metric map for instrument_ab.select_by (the metric the winner is chosen on).
# Pre-registered default is "sortino" (ADR-4); the Sortino floor + CVaR95 guard apply
# regardless of which metric ranks, as pre-registered tail guards.
_SELECT_KEY = {"sortino": "sortino", "sharpe": "net_sharpe"}


def select_instrument(cells: list[dict], gates: dict) -> dict:
    """Among phase1+tail PASSERS (when ``require_pass_phase1``), pick the top cell by
    ``select_by`` that also clears the Sortino floor + the CVaR95 guard (ADR-4). No
    eligible cell ⇒ no winner (keep the incumbent)."""
    key = _SELECT_KEY.get(gates["select_by"])
    if key is None:
        raise ValueError(f"unsupported select_by {gates['select_by']!r}; "
                         f"expected one of {sorted(_SELECT_KEY)}")
    go_cells = [c for c in cells if c["verdict"]["verdict"] == "GO"]
    pool = go_cells if gates["require_pass_phase1"] else list(cells)
    eligible = [c for c in pool
                if c["metrics"]["sortino"] >= gates["min_sortino"]
                and c["metrics"]["cvar95_pct_daily"] >= gates["cvar95_floor_pct"]]
    ranked = sorted(eligible, key=lambda c: c["metrics"][key], reverse=True)
    win = ranked[0] if ranked else None
    return {
        "winner": ({"asset": win["asset"], "instrument": win["instrument"],
                    "sortino": win["metrics"]["sortino"], "net_sharpe": win["metrics"]["net_sharpe"],
                    "cvar95_pct_daily": win["metrics"]["cvar95_pct_daily"],
                    "bootstrap_sortino_ci": win["metrics"]["bootstrap_sortino_ci"]} if win else None),
        "ranked_by": gates["select_by"],
        "ranking": [{"asset": c["asset"], "instrument": c["instrument"],
                     "sortino": c["metrics"]["sortino"], "net_sharpe": c["metrics"]["net_sharpe"],
                     "cvar95_pct_daily": c["metrics"]["cvar95_pct_daily"]} for c in ranked],
        "n_go_cells": len(go_cells),
        "n_eligible_after_floors": len(eligible),
    }


def _json_safe(obj):
    """Recursively replace non-finite floats (NaN/±inf) with None so the verdict is
    STRICT JSON (json.dumps(..., allow_nan=False) would otherwise emit `Infinity`)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def run_instrument_ab(cfg: SkewConfig | None = None, gates: dict | None = None) -> dict:
    """Real-chain instrument bake-off: every structure in ``gates['instruments']`` scored
    per (asset, instrument) cell against phase1+tail, then a Sortino/CVaR selection."""
    gates = gates or GATES
    cfg = cfg or SkewConfig(wing_delta=gates["wing_delta"])
    raw = dol.load({"universe": {"assets": ["BTC", "ETH"]}})
    chain = tcl.load_chain_snapshots("2021-04", "2026-06")
    if chain.empty:
        raise RuntimeError("No chain snapshots cached — run tardis_options_chain_loader first")
    chain["snapshot_date"] = pd.to_datetime(chain["snapshot_date"], utc=True)
    chain["expiry_dt"] = pd.to_datetime(chain["expiry_dt"], utc=True)
    snaps_by_month = {(d.year, d.month): g for d, g in chain.groupby("snapshot_date")}

    assets = ["BTC", "ETH"]
    cells, per_instrument = [], {}
    for a in assets:
        perp = ovf._to_date_index(raw.perp[a])["close"]
        fund = ovf._daily_funding(raw.funding[a])
        spot_by_date = {pd.Timestamp(d): float(v) for d, v in perp.items()}
        funding_by_date = {pd.Timestamp(d): float(v) for d, v in fund.items()}
        per_instrument[a] = {}
        for kind in gates["instruments"]:
            pnl, diag = simulate_real_chain(a, snaps_by_month, spot_by_date, funding_by_date, kind, cfg)
            m = _metrics_ab(pnl, diag, cfg)
            v = _instrument_verdict(m, gates)
            per_instrument[a][kind] = {"metrics": m, "verdict": v}
            cells.append({"asset": a, "instrument": kind, "metrics": m, "verdict": v})

    selection = select_instrument(cells, gates)
    sample_months = max((c["metrics"]["sample_n_months"] for c in cells), default=0)
    gate_keys = ("min_net_sharpe", "min_net_pf", "require_multi_subperiod", "max_recent_oos_drawdown",
                 "min_recent_oos_sharpe", "max_worst_window_dd_pct", "max_net_vega_per_100k",
                 "cvar95_floor_pct", "instruments", "wing_delta", "require_pass_phase1",
                 "select_by", "min_sortino", "sample_basis")
    return _json_safe({
        "config": asdict(cfg),
        "gates": {k: gates[k] for k in gate_keys},
        "inverse_only": INVERSE_ONLY,
        "sample_n_months": int(sample_months),
        "per_instrument": per_instrument,
        "selection": selection,
        "verdict": "GO" if selection["winner"] else "NO-GO",
        "kills": [] if selection["winner"]
        else ["no instrument cleared phase1+tail AND the Sortino floor + CVaR95 guard"],
    })


def _fmt_ab(r: dict) -> str:
    L = ["# Options-VRP Instrument A/B Bake-off (real-chain, free Tardis first-of-month)",
         f"**Verdict: {r['verdict']}** — winner: {r['selection']['winner']}",
         f"sample: {r['sample_n_months']} monthly cycles | inverse_only={r['inverse_only']} | "
         f"select_by={r['selection']['ranked_by']}",
         "",
         "## Per (asset, instrument) — net, with the tail-adjusted selection metrics",
         "| asset | instrument | Sharpe | Sortino | PF | CVaR95% | wDD% | vega/100k | +yrs | verdict |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for a, kinds in r["per_instrument"].items():
        for kind, cell in kinds.items():
            m, v = cell["metrics"], cell["verdict"]
            L.append(f"| {a} | {kind} | {m['net_sharpe']:.2f} | {m['sortino']:.2f} | {m['net_pf']:.2f} | "
                     f"{m['cvar95_pct_daily']:.2f} | {m['worst_window_dd_pct']:.1f} | "
                     f"{m['max_net_vega_per_100k']:,.0f} | {m['pos_years']}/{m['n_years']} | {v['verdict']} |")
    L += ["", "## Selection (ranked by Sortino among phase1+tail passers, CVaR-guarded)",
          f"- GO cells: {r['selection']['n_go_cells']} | eligible after Sortino/CVaR floors: "
          f"{r['selection']['n_eligible_after_floors']}"]
    for c in r["selection"]["ranking"]:
        L.append(f"  - {c['asset']} {c['instrument']}: Sortino {c['sortino']:.2f}, "
                 f"Sharpe {c['net_sharpe']:.2f}, CVaR95 {c['cvar95_pct_daily']:.2f}%")
    if r["kills"]:
        L += ["", "## KILL", *[f"- {k}" for k in r["kills"]]]
    return "\n".join(L)


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
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Real-chain skew falsification + instrument A/B")
    ap.add_argument("--instrument-ab", action="store_true",
                    help="run the signed-leg instrument bake-off (straddle/strangle/iron_fly/iron_condor) "
                         "→ instrument_ab_verdict.json, instead of the default straddle+strangle skew gate")
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.instrument_ab:
        r = run_instrument_ab()
        (RESULTS_DIR / "instrument_ab_verdict.json").write_text(
            json.dumps(r, indent=2, default=str, allow_nan=False), encoding="utf-8")
        summary = _fmt_ab(r)
        (RESULTS_DIR / "instrument_ab_summary.md").write_text(summary, encoding="utf-8")
    else:
        r = run()
        (RESULTS_DIR / "skew_verdict.json").write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
        summary = _fmt(r)
        (RESULTS_DIR / "skew_summary.md").write_text(summary, encoding="utf-8")
    logger.info("\n%s", summary)


if __name__ == "__main__":
    main()
