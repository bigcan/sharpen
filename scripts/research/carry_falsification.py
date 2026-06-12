"""Carry falsification (Lever C, the 2nd factor) — pre-registered spec
`docs/research/carry_falsification_spec_2026-06-12.md` (committed BEFORE data).

The cheap, CPU-only, linear, net-of-cost falsification the redesign protocol
requires BEFORE wiring carry into the env or any RL: does a LINEAR carry book
(a) survive realistic costs across asset classes, AND (b) ADD to the validated
TSMOM momentum core (low correlation + a combined-portfolio Sharpe lift)?

Mirrors `xsec_momentum_falsification.py` construction/costs/metrics VERBATIM
(reused by import) so the carry book and momentum book are directly comparable.
Carry faces a STRICTER gate than momentum: standalone viability is necessary but
not sufficient — it must also be additive (Gate 2), with tail honesty (Gate 3).

Scored classes (spec v1): equity (div yield), rates (Treasury-curve carry+roll),
FX-G10 (short-rate differential). Commodity (futures-curve) + crypto (decayed) excluded.

DATA REALITY (recorded honestly): FRED is UNREACHABLE from this workstation
(timeouts/conn-closed on every series incl. DGS10) — the same geo/firewall
constraint R1 hit (api.binance blocked / Vision CDN open). So FX foreign short
rates can't be fetched here => this run is a DATA-LIMITED 2-class (rates+equity)
test and FX carry is the gating follow-up. US Treasury curve comes from Yahoo
indices (^IRX/^FVX/^TNX/^TYX, reachable). The FRED fetcher stays (defensive, fast
fail) so the run self-confirms the block and auto-includes FX on a workstation
that can reach FRED.

Causality (LEAK-2): carry signal at rebalance t uses yield/dividend data stamped
<= t; weights apply to returns from t+1 (T+1 execution lag, inherited from the
momentum backtest). A same-day-execution variant is the leak tripwire.
"""
from __future__ import annotations

import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "carry_falsification"
OUT.mkdir(parents=True, exist_ok=True)
CURVE_CACHE = OUT / "yahoo_curve.parquet"
PRICE_CACHE = OUT / "carry_prices.parquet"
DIV_CACHE = OUT / "div_yields.parquet"

# Reuse the momentum harness VERBATIM (same construction/costs/metrics) ----------
sys.path.insert(0, str(Path(__file__).resolve().parent))
import xsec_momentum_falsification as mom  # noqa: E402

ANN = mom.ANN
COST_MODELS = mom.COST_MODELS
START, END = mom.START, mom.END
VOL_WIN = mom.VOL_WIN

# --- Carry universe (tradeable ETF return side) ---
FX_ETF = {"FXE": "EUR", "FXY": "JPY", "FXB": "GBP",
          "FXA": "AUD", "FXF": "CHF", "FXC": "CAD"}            # each vs USD
RATES_ETF = {"SHY": "y2", "IEF": "10y", "TLT": "30y", "LQD": "10y"}  # ETF -> curve tenor
EQUITY_ETF = ["SPY", "QQQ", "IWM", "EFA", "EEM"]
CARRY_TICKERS = sorted(set(list(FX_ETF) + list(RATES_ETF) + EQUITY_ETF))

# US Treasury curve via Yahoo (reachable where FRED is blocked)
US_CURVE_YH = {"3m": "^IRX", "5y": "^FVX", "10y": "^TNX", "30y": "^TYX"}
# FX foreign 3M short-rate candidates (FRED OECD interbank) — used only if FRED reachable
FRED_FX = {
    "EUR": ["IR3TIB01EZM156N", "ECBDFR"], "JPY": ["IR3TIB01JPM156N", "IRSTCI01JPM156N"],
    "GBP": ["IR3TIB01GBM156N", "IUDSOIA"], "AUD": ["IR3TIB01AUM156N", "IRSTCB01AUM156N"],
    "CHF": ["IR3TIB01CHM156N", "IRSTCI01CHM156N"], "CAD": ["IR3TIB01CAM156N", "IRSTCB01CAM156N"],
}
_fred_resolved: dict[str, str] = {}


# ============================ data fetch ============================
def fred_reachable() -> bool:
    try:
        req = urllib.request.Request(
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10",
            headers={"User-Agent": "Mozilla/5.0"})
        urllib.request.urlopen(req, timeout=8).read()
        return True
    except Exception:
        return False


def _fred_one(series: str) -> pd.Series | None:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=12).read().decode()
        df = pd.read_csv(io.StringIO(raw))
        dc, vc = df.columns[0], df.columns[1]
        df[dc] = pd.to_datetime(df[dc], errors="coerce")
        df[vc] = pd.to_numeric(df[vc], errors="coerce")
        df = df.dropna()
        if len(df) < 24:
            return None
        s = df.set_index(dc)[vc].sort_index()
        med = pd.Series(s.index).diff().dt.days.median()
        if med and med > 20:               # monthly -> 1-month publication lag (causal)
            s = s.shift(1)
        didx = pd.date_range(s.index.min(), pd.Timestamp(END), freq="D")
        return s.reindex(didx).ffill()
    except Exception:
        return None


def get_fx_rate(ccy: str) -> pd.Series | None:
    for sid in FRED_FX[ccy]:
        s = _fred_one(sid)
        if s is not None:
            _fred_resolved[ccy] = sid
            return s
    _fred_resolved[ccy] = "NONE"
    return None


def get_yahoo_curve() -> dict[str, pd.Series]:
    if CURVE_CACHE.exists():
        df = pd.read_parquet(CURVE_CACHE)
        return {c: df[c].dropna() for c in df.columns}
    import yfinance as yf
    out = {}
    for ten, tk in US_CURVE_YH.items():
        h = yf.Ticker(tk).history(period="max")
        s = h["Close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        out[ten] = s.sort_index()
    # interpolate a 2y proxy between 3m (0.25y) and 5y
    irx, fvx = out["3m"], out["5y"]
    common = irx.index.intersection(fvx.index)
    out["y2"] = irx.reindex(common) + (2 - 0.25) / (5 - 0.25) * (
        fvx.reindex(common) - irx.reindex(common))
    out["10y"], out["30y"] = out["10y"], out["30y"]
    pd.DataFrame(out).to_parquet(CURVE_CACHE)
    return out


def get_prices() -> pd.DataFrame:
    if PRICE_CACHE.exists():
        return pd.read_parquet(PRICE_CACHE)
    import yfinance as yf
    raw = yf.download(CARRY_TICKERS + ["UUP"], start=START, end=END,
                      progress=False, auto_adjust=True)
    close = raw["Close"].copy().dropna(how="all").sort_index()
    close.to_parquet(PRICE_CACHE)
    return close


def get_div_yields(close: pd.DataFrame) -> pd.DataFrame:
    if DIV_CACHE.exists():
        return pd.read_parquet(DIV_CACHE)
    import yfinance as yf
    dy = pd.DataFrame(index=close.index, columns=EQUITY_ETF, dtype=float)
    for t in EQUITY_ETF:
        try:
            div = yf.Ticker(t).dividends
            if div is None or len(div) == 0:
                continue
            div.index = pd.to_datetime(div.index).tz_localize(None)
            daily_div = div.reindex(close.index).fillna(0.0)
            ttm = daily_div.rolling(252).sum()
            dy[t] = (ttm / close[t]).replace([np.inf, -np.inf], np.nan)
        except Exception:
            continue
    dy.to_parquet(DIV_CACHE)
    return dy


# ============================ carry signals ============================
def _sig(rebal, tickers):
    return pd.DataFrame(index=rebal, columns=tickers, dtype=float)


def _asof(s, dt):
    if s is None:
        return np.nan
    sub = s.loc[:dt]
    return float(sub.iloc[-1]) if len(sub) else np.nan


def _xs(row):
    r = row.dropna()
    if len(r) < 2 or r.std() == 0:
        return pd.Series(0.0, index=row.index)
    z = (r - r.mean()) / (r.std() + 1e-9)
    return (z / 2.0).clip(-1, 1).reindex(row.index).fillna(0.0)


def fx_carry_signal(rebal, fx_rates, us3m):
    sig = _sig(rebal, list(FX_ETF))
    for dt in rebal:
        usd = _asof(us3m, dt)
        diffs = {etf: (_asof(fx_rates.get(ccy), dt) - usd) for etf, ccy in FX_ETF.items()}
        sig.loc[dt] = _xs(pd.Series(diffs))
    return sig.fillna(0.0)


def rates_carry_signal(rebal, curve):
    """Per bond ETF carry+roll = (tenor yield - 3m financing), >0 when curve upward
    (positive carry -> long duration). TS sign+magnitude, tanh-squashed."""
    sig = _sig(rebal, list(RATES_ETF))
    for dt in rebal:
        fin = _asof(curve.get("3m"), dt)
        for etf, ten in RATES_ETF.items():
            c = _asof(curve.get(ten), dt) - fin
            sig.loc[dt, etf] = np.tanh(c / 1.5) if np.isfinite(c) else 0.0
    return sig.fillna(0.0)


def equity_carry_signal(rebal, dy, us3m):
    """Cross-sectional dividend-yield carry (div yield - rf); high-yield long."""
    sig = _sig(rebal, EQUITY_ETF)
    for dt in rebal:
        rf = _asof(us3m, dt) / 100.0       # pct -> fraction; constant across XS (cancels)
        row = {}
        for t in EQUITY_ETF:
            sub = dy[t].loc[:dt].dropna() if t in dy.columns else pd.Series(dtype=float)
            row[t] = (float(sub.iloc[-1]) - rf) if len(sub) else np.nan
        sig.loc[dt] = _xs(pd.Series(row))
    return sig.fillna(0.0)


# ============================ backtest w/ configurable lag ============================
def backtest_lag(weights_rebal, rets, lag):
    di = rets.index
    w_daily = weights_rebal.reindex(di).ffill().fillna(0.0)
    w_eff = w_daily.shift(lag).fillna(0.0)
    gross = (w_eff * rets).sum(axis=1)
    dw = weights_rebal.fillna(0.0).diff().abs().sum(axis=1)
    if len(weights_rebal):
        dw.iloc[0] = weights_rebal.iloc[0].abs().sum()
    years = (di[-1] - di[0]).days / 365.25
    turn = float(dw.sum() / years) if years > 0 else 0.0
    cost = {n: dw.reindex(di).fillna(0.0) * c for n, c in COST_MODELS.items()}
    return gross, cost, turn


def net_series(weights_rebal, rets, cm="standard_2bps", lag=1):
    g, c, _ = backtest_lag(weights_rebal, rets, lag)
    return (g - c[cm]).dropna()


# ============================ metrics ============================
def cvar95(d):
    x = d.dropna().to_numpy()
    if len(x) < 20:
        return 0.0
    q = np.quantile(x, 0.05)
    tail = x[x <= q]
    return float(tail.mean()) if len(tail) else 0.0


def subperiod_sharpes(d):
    B = {"2006-09": ("2006-01-01", "2009-12-31"), "2010-15": ("2010-01-01", "2015-12-31"),
         "2016-20": ("2016-01-01", "2020-12-31"), "2021-26": ("2021-01-01", "2026-12-31")}
    return {k: (round(mom.sharpe(d.loc[a:b]), 3) if len(d.loc[a:b]) > 60 else None)
            for k, (a, b) in B.items()}


def scale10(x):
    v = x.std() * np.sqrt(ANN)
    return x * (0.10 / v) if v > 0 else x


def main():
    close = get_prices()
    close = close[[t for t in CARRY_TICKERS + ["UUP"] if t in close.columns]]
    rets = close.pct_change()
    bench = close["SPY"].pct_change()
    print(f"Loaded {close.shape[1]} ETFs, {close.shape[0]} days, "
          f"{close.index.min().date()} -> {close.index.max().date()}")

    curve = get_yahoo_curve()
    dy = get_div_yields(close)
    us3m = curve.get("3m")

    fred_ok = fred_reachable()
    print(f"FRED reachable: {fred_ok}")
    fx_rates = {ccy: get_fx_rate(ccy) for ccy in set(FX_ETF.values())} if fred_ok else {}
    fx_available = fred_ok and any(v is not None for v in fx_rates.values())

    rebal = mom.last_trading_of_period(close.index, "monthly")
    rebal = rebal[rebal >= close.index[max(mom.LOOKBACKS) + mom.SKIP + VOL_WIN]]

    # ---- per-class carry books (only classes with data) ----
    classes = {}
    classes["rates"] = mom.vol_scaled_weights(
        rates_carry_signal(rebal, curve), rets[list(RATES_ETF)], rebal
    ).reindex(columns=close.columns).fillna(0.0)
    classes["equity"] = mom.vol_scaled_weights(
        equity_carry_signal(rebal, dy, us3m), rets[EQUITY_ETF], rebal
    ).reindex(columns=close.columns).fillna(0.0)
    if fx_available:
        classes["fx"] = mom.vol_scaled_weights(
            fx_carry_signal(rebal, fx_rates, us3m), rets[list(FX_ETF)], rebal
        ).reindex(columns=close.columns).fillna(0.0)
    unavailable = [] if fx_available else ["fx"]

    per_class = {c: mom.run_book(f"carry_{c}", w, rets, bench) for c, w in classes.items()}
    w_pool = sum(classes.values())
    pooled = mom.run_book("carry_pooled", w_pool, rets, bench)
    carry_net = pooled["_net_standard"].dropna()

    # ---- momentum baseline (reuse momentum harness on its own cached universe) ----
    mclose = mom.get_prices()
    mclose = mclose[[t for t in mom.ALL_TICKERS if t in mclose.columns]]
    mrets = mclose.pct_change()
    mbench = mclose["SPY"].pct_change()
    mrebal = mom.last_trading_of_period(mclose.index, "monthly")
    mrebal = mrebal[mrebal >= mclose.index[max(mom.LOOKBACKS) + mom.SKIP + VOL_WIN]]
    mw = mom.vol_scaled_weights(mom.tsmom_signal(mclose, mrebal), mrets, mrebal)
    mom_net = mom.run_book("TSMOM_pooled_monthly", mw, mrets, mbench)["_net_standard"].dropna()

    # ---- Gate 2 additivity ----
    common = carry_net.index.intersection(mom_net.index)
    cn, mn = carry_net.reindex(common), mom_net.reindex(common)
    corr_cm = float(cn.corr(mn))
    combined = 0.5 * scale10(cn) + 0.5 * scale10(mn)
    mom_only = scale10(mn)
    sh_carry, sh_mom = mom.sharpe(cn), mom.sharpe(mn)
    sh_comb, dd_comb, dd_mom = mom.sharpe(combined), mom.max_dd(combined), mom.max_dd(mom_only)

    # ---- Gate 1 ----
    net_sh = pooled["by_cost"]["standard_2bps"]["sharpe"]
    fric_sh = pooled["by_cost"]["frictionless"]["sharpe"]
    gap = round(fric_sh - net_sh, 3)
    cls_sh = {c: per_class[c]["by_cost"]["standard_2bps"]["sharpe"] for c in classes}
    classes_pos = sum(1 for s in cls_sh.values() if s > 0)
    g1 = (net_sh >= 0.40) and (classes_pos >= 3) and (gap <= 0.15)

    # ---- Gate 2 ----
    g2 = (corr_cm < 0.30) and (sh_comb >= 0.65) and (dd_comb >= dd_mom)  # dd negative

    # ---- Gate 3 ----
    sk = float(cn.skew())
    cv = round(cvar95(cn) * 100, 3)
    subp = subperiod_sharpes(cn)
    subp_pos = sum(1 for v in subp.values() if v is not None and v > 0)
    skew_flag = sk < -1.0

    # ---- leak tripwire ----
    causal = mom.sharpe(net_series(w_pool, rets, lag=1))
    look = mom.sharpe(net_series(w_pool, rets, lag=0))
    leak_gap = round(look - causal, 3)
    leak_ok = abs(leak_gap) < 0.10

    if unavailable:
        decision = "INCONCLUSIVE_DATA_BLOCKED"
    elif g1 and g2:
        decision = "GO"
    elif g1:
        decision = "CARRY_REDUNDANT"
    else:
        decision = "NO_GO"

    verdict = {
        "test": "carry falsification (linear, pre-RL, 2nd-factor)",
        "spec": "docs/research/carry_falsification_spec_2026-06-12.md",
        "decision": decision,
        "data_note": ("FRED unreachable from this workstation -> FX carry NOT run; "
                      "rates+equity scored via Yahoo curve+dividends. FX is the gating follow-up."
                      if unavailable else "all 3 classes scored"),
        "classes_scored": list(classes), "classes_unavailable": unavailable,
        "leak_tripwire_ok": leak_ok,
        "gate1_standalone": {
            "pass": bool(g1), "pooled_net_sharpe": net_sh, "pooled_frictionless_sharpe": fric_sh,
            "cost_gap": gap, "per_class_net_sharpe": cls_sh, "classes_positive": classes_pos,
            "rule": "pooled net SR>=0.40 AND >=3/3 classes net-positive AND cost_gap<=0.15"},
        "gate2_additivity": {
            "pass": bool(g2), "corr_carry_momentum": round(corr_cm, 3),
            "sharpe_carry": round(sh_carry, 3), "sharpe_momentum": round(sh_mom, 3),
            "sharpe_combined_riskparity": round(sh_comb, 3),
            "maxdd_combined": round(dd_comb, 4), "maxdd_momentum_only": round(dd_mom, 4),
            "rule": "corr<0.30 AND combined SR>=0.65 AND combined DD<=momentum-alone DD"},
        "gate3_tail_persistence": {
            "carry_skew": round(sk, 3), "carry_cvar95_pct": cv,
            "skew_flag_short_vol": bool(skew_flag), "subperiod_sharpes": subp,
            "subperiods_positive": subp_pos},
        "leak": {"causal_sharpe": round(causal, 3), "lookahead_sharpe": round(look, 3),
                 "gap": leak_gap},
        "fred_reachable": fred_ok, "fred_resolved": _fred_resolved,
        "carry_window": [str(carry_net.index.min().date()), str(carry_net.index.max().date())],
        "per_class_turnover": {c: per_class[c]["turnover_ann"] for c in classes},
    }
    results = {"params": {"universe": {"fx": list(FX_ETF), "rates": list(RATES_ETF),
                                       "equity": EQUITY_ETF}, "cost_models": COST_MODELS,
                          "rebalance": "monthly"},
               "books": {b: {"by_cost": (pooled if b == "carry_pooled"
                                         else per_class[b.replace("carry_", "")])["by_cost"]}
                         for b in ["carry_pooled"] + [f"carry_{c}" for c in classes]},
               "verdict": verdict}
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str))

    L = [f"# Carry Falsification — Verdict: **{decision}**", "",
         "> Session 553-cont-45, 2026-06-12. Spec `docs/research/carry_falsification_spec_2026-06-12.md` (pre-registered before data).",
         f"> Mirrors `xsec_momentum_falsification.py` verbatim. Carry window {verdict['carry_window'][0]} -> {verdict['carry_window'][1]}.",
         f"> DATA NOTE: {verdict['data_note']}", "",
         f"## Gate 1 — standalone viability: {'PASS' if g1 else 'FAIL/PARTIAL'}", "",
         "| pooled net SR | frictionless | cost gap | classes net-positive |",
         "|---|---|---|---|", f"| {net_sh:.3f} | {fric_sh:.3f} | {gap:.3f} | {classes_pos}/{len(classes)} |", "",
         "Per-class net Sharpe (2bps): " + ", ".join(f"{c}={cls_sh[c]:.3f}" for c in classes), "",
         f"## Gate 2 — additivity to momentum: {'PASS' if g2 else 'FAIL'}", "",
         "| corr(carry,mom) | SR carry | SR mom | SR combined | DD combined | DD mom-only |",
         "|---|---|---|---|---|---|",
         f"| {corr_cm:.3f} | {sh_carry:.3f} | {sh_mom:.3f} | {sh_comb:.3f} | {dd_comb*100:.2f}% | {dd_mom*100:.2f}% |", "",
         "## Gate 3 — tail + persistence", "",
         f"carry skew = {sk:.3f} ({'SHORT-VOL FLAG' if skew_flag else 'ok'}), CVaR95 = {cv:.3f}%/day; "
         f"subperiods positive = {subp_pos}/4 {json.dumps(subp)}", "",
         f"## Leak tripwire: {'OK' if leak_ok else 'FAIL'} — causal SR {causal:.3f} vs look-ahead {look:.3f} (gap {leak_gap})", "",
         f"## Decision: **{decision}**"]
    (OUT / "summary.md").write_text("\n".join(L))

    print("\n" + "#" * 64)
    print(f"classes scored={list(classes)} unavailable={unavailable}")
    print(f"GATE1 {'PASS' if g1 else 'FAIL/PARTIAL'} | pooled net SR={net_sh} fric={fric_sh} "
          f"gap={gap} classes+={classes_pos}/{len(classes)} {cls_sh}")
    print(f"GATE2 {'PASS' if g2 else 'FAIL'} | corr={corr_cm:.3f} SR carry={sh_carry:.3f} "
          f"mom={sh_mom:.3f} combined={sh_comb:.3f} DDcomb={dd_comb*100:.1f}% DDmom={dd_mom*100:.1f}%")
    print(f"GATE3 skew={sk:.3f} {'FLAG' if skew_flag else ''} CVaR95={cv}% subperiods+={subp_pos}/4 {subp}")
    print(f"LEAK {'OK' if leak_ok else 'FAIL'} causal={causal:.3f} look={look:.3f} gap={leak_gap}")
    print(f"DECISION: {decision}")
    print("#" * 64)
    return verdict


if __name__ == "__main__":
    main()
