"""Stage-0 falsification probe: ETF-vs-underlying-stock arbitrage (0050 vs TSMC 2330) (S553-cont-128).

Implements docs/research/etf_underlying_arb_preregistration_2026-07-12.md VERBATIM — gates frozen
before this ran. CPU-only, ~$0. Operator stance: fees deprioritized ⇒ judged on GROSS structure +
statistical validity; net + a per-leg harvestability decomposition reported for the record.

Both legs are Taiwan equities closing at 13:30 (synchronized auction) ⇒ no timing-mark artifact and
no cash-index staleness (the two failure modes that killed the ETF-ETF spread and the futures basis).

Usage:
    python scripts/research/etf_underlying_arb_probe.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research.futures_basis_arb_probe import (  # noqa: E402
    _boot_p05, _perm_pvalue, _sharpe,
)

ETF_PANEL = ROOT / "data" / "raw" / "taiwan_panel" / "ohlcv_daily.parquet"
STOCK_PANEL = ROOT / "data" / "taiwan_universe" / "taiwan_daily.parquet"

log = logging.getLogger("etf_underlying_arb_probe")

W_SET = (20, 40, 60)
Z_CAP = 2.0
COST_ETF = 0.0010          # 10 bps one-way (0050); record only
COST_STOCK = 0.0020        # 20 bps one-way (stock leg, 0.3% sell tax); record only
ANN = np.sqrt(252.0)
OOS_FRAC = 0.30
SHARPE_FLOOR = 0.50
TOP10 = ["2330", "2317", "2454", "2308", "2412", "2882", "2881", "2303", "3711", "2891"]


def _pf(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    g, loss = x[x > 0].sum(), -x[x < 0].sum()
    return float(g / loss) if loss > 0 else float("inf")


def _etf_close() -> pd.Series:
    e = pd.read_parquet(ETF_PANEL)
    e["date"] = pd.to_datetime(e["date"])
    s = e[e.ticker == "0050"].set_index("date")["close"].sort_index()
    return s[s > 0]


def _stock_close(ticker: str) -> pd.Series:
    d = pd.read_parquet(STOCK_PANEL)
    d["date"] = pd.to_datetime(d["date"])
    s = d[d.ticker.astype(str) == ticker].set_index("date")["close"].sort_index()
    return s[s > 0]


def _ew_basket_logclose() -> pd.Series:
    """Equal-weight top-10 basket log-level = mean over constituents of ln(price)."""
    d = pd.read_parquet(STOCK_PANEL)
    d["date"] = pd.to_datetime(d["date"])
    wide = d[d.ticker.astype(str).isin(TOP10)].pivot_table(
        index="date", columns="ticker", values="close", aggfunc="last")
    wide = wide[(wide > 0).all(axis=1)]
    return np.log(wide).mean(axis=1)


def _frame(logA: pd.Series, logB: pd.Series) -> pd.DataFrame:
    """Aligned frame with log-spread + forward close-to-close leg returns."""
    m = pd.concat([logA.rename("lA"), logB.rename("lB")], axis=1).dropna().sort_index()
    assert not m.isna().any().any(), "Gate D FAIL: NaN in used window"
    m["spread"] = m["lA"] - m["lB"]
    m["rA"] = m["lA"].shift(-1) - m["lA"]          # forward t->t+1
    m["rB"] = m["lB"].shift(-1) - m["lB"]
    m["spread_ret"] = m["rA"] - m["rB"]
    return m


def _cell(m: pd.DataFrame, W: int, *, lag: int = 0) -> dict:
    aa = pd.Series(m["spread"].values)
    z = ((aa - aa.rolling(W).mean()) / aa.rolling(W).std(ddof=1)).values
    pos = -np.clip(z, -Z_CAP, Z_CAP) / Z_CAP
    pos = np.where(np.isfinite(pos), pos, 0.0)
    sr = np.where(np.isfinite(m["spread_ret"].values), m["spread_ret"].values, 0.0)
    fwd = np.roll(sr, -lag)
    if lag > 0:
        fwd[-lag:] = 0.0
    elif lag < 0:
        fwd[:-lag] = 0.0
    gross = pos * fwd
    dpos = np.abs(np.diff(pos, prepend=0.0))
    net = gross - (COST_ETF + COST_STOCK) * dpos
    return {"gross": _sharpe(gross), "net": _sharpe(net), "net_pf": _pf(net),
            "turn": float(np.mean(dpos)), "gross_pnl": gross, "pos": pos, "fwd": fwd}


def _run(name: str, m: pd.DataFrame, *, primary: bool) -> dict:
    log.info("=== %s  n=%d  %s -> %s ===", name, len(m), m.index.min().date(), m.index.max().date())
    per = {W: _cell(m, W, lag=0) for W in W_SET}
    per1 = {W: _cell(m, W, lag=1) for W in W_SET}
    med = float(np.median([per[W]["gross"] for W in W_SET]))
    for W in W_SET:
        log.info("  W=%2d  lag0 gross=%+.3f net=%+.3f PF=%.3f turn=%.3f | lag+1 gross=%+.3f",
                 W, per[W]["gross"], per[W]["net"], per[W]["net_pf"], per[W]["turn"], per1[W]["gross"])
    med_W = min(W_SET, key=lambda W: abs(per[W]["gross"] - med))
    base = per[med_W]
    p05 = _boot_p05(base["gross_pnl"])
    perm_p = _perm_pvalue(base["pos"], base["fwd"], base["gross"])
    n = len(base["gross_pnl"])
    cut = int(n * (1 - OOS_FRAC))
    oos = _sharpe(base["gross_pnl"][cut:])
    samebar = _cell(m, med_W, lag=-1)["gross"]
    oracle = _sharpe(np.sign(base["fwd"]) * base["fwd"])
    # leg decomposition
    pos = base["pos"]
    rA = np.where(np.isfinite(m["rA"].values), m["rA"].values, 0.0)
    rB = np.where(np.isfinite(m["rB"].values), m["rB"].values, 0.0)
    etf_leg = _sharpe(pos * rA)
    stock_leg = _sharpe(pos * (-rB))
    log.info("  median-W gross=%+.3f (W=%d) | G2 p05=%+.3f | G3 OOS=%+.3f | net=%+.3f",
             med, med_W, p05, oos, base["net"])
    log.info("  leg-decomp: total=%+.2f  ETF-leg(pos*rA)=%+.2f  STOCK-leg(pos*-rB)=%+.2f  "
             "(both legs liquid/tradeable)", base["gross"], etf_leg, stock_leg)
    log.info("  harness tripwires: TW-1 gap(%+.3f-%+.3f)>0.30? %s | TW-4 oracle=%+.2f≫causal? %s "
             "|| significance: perm_p=%.4f (sig if <0.05)", base["gross"], samebar,
             base["gross"] - samebar > 0.30, oracle, oracle > base["gross"] + 2.0, perm_p)
    return {"med": med, "p05": p05, "perm_p": perm_p, "oos": oos, "net": base["net"],
            "primary": primary, "tw1": base["gross"] - samebar > 0.30,
            "tw4": oracle > base["gross"] + 2.0, "etf_leg": etf_leg, "stock_leg": stock_leg}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    log.info("ETF-vs-underlying Stage-0 probe (pre-reg etf_underlying_arb_preregistration_2026-07-12.md)")
    etf = _etf_close()
    p1 = _run("P1 0050/2330(TSMC)", _frame(np.log(etf), np.log(_stock_close("2330"))), primary=True)
    _run("P2 0050/EW-top10-basket", _frame(np.log(etf), _ew_basket_logclose()), primary=False)

    # harness tripwires (must pass or the probe is invalid) vs signal gates (a failed gate = NO-GO)
    harness_ok = p1["tw1"] and p1["tw4"]
    g1 = p1["med"] >= SHARPE_FLOOR
    g2 = (p1["p05"] > 0) and (p1["perm_p"] < 0.05)     # significant: bootstrap CI lo>0 AND perm-null
    g3 = p1["oos"] > 0
    go = g1 and g2 and g3
    log.info("=" * 78)
    if not harness_ok:
        log.info("VERDICT: INVALID — harness tripwire failed (TW1=%s TW4=%s).", p1["tw1"], p1["tw4"])
    else:
        log.info("VERDICT (P1): %s  [G1 gross>=0.50=%s (%.2f) | G2 significant(p05>0 & perm<0.05)=%s "
                 "(p05=%.2f perm=%.3f) | G3 OOS>0=%s (%.2f)]",
                 "VALID SIGNAL -> Stage-1" if go else "NO-GO",
                 g1, p1["med"], g2, p1["p05"], p1["perm_p"], g3, p1["oos"])
        log.info("  harvestability: ETF-leg=%+.2f STOCK-leg=%+.2f (both tradeable) | net@30bps=%+.2f",
                 p1["etf_leg"], p1["stock_leg"], p1["net"])
        log.info("  -> %s", "genuinely tradeable (no staleness/timing artifact); confirm cost realism"
                 if go else "ETF-vs-underlying daily reversion CLOSED; no grammar change")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
