"""T1 — TIME-SERIES momentum on country ETFs, as a candidate diversifying sleeve.

Pre-registration: `docs/research/country_tsmom_preregistration_2026-07-31.md` (committed with an
empty Results section BEFORE this ran).

Uses the house's validated machinery UNCHANGED — `tsmom_signal` / `vol_scaled_weights` / `backtest`
from `xsec_momentum_falsification`, and `portfolio_frontier.risk_parity` for the combine. No new
statistic, no tuned parameter: the constants are the ones already locked for the 0.60 book.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import portfolio_frontier as pf  # noqa: E402
import xsec_momentum_falsification as mom  # noqa: E402
TICKERS = ["DBB","DBE","DBP","GSG","UNG","USL","UGA","PALL","PPLT","BNO","CORN","CANE","SOYB","WEAT","NIB","FTGC","COMT"]  # pre-registered; EXCLUDES GLD,SLV,DBC,USO,DBA already in the book

log = logging.getLogger("commodity_tsmom")
OUT = ROOT / "results" / "commodity_tsmom"
COSTS = {"frictionless": 0.0, "etf_2bps": 0.0002, "etf_5bps": 0.0005, "etf_10bps": 0.0010}

# Pre-committed bars (pre-reg §2)
MIN_STANDALONE_SR = 0.30
MAX_CORR_TO_TSMOM = 0.60
MIN_COMBINED_SR = 0.66


def _sharpe(d: pd.Series) -> float:
    d = d.dropna()
    s = d.std()
    return float(d.mean() / s * np.sqrt(mom.ANN)) if s > 0 else 0.0


def _max_dd(d: pd.Series) -> float:
    eq = (1 + d.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import yfinance as yf
    raw = yf.download(TICKERS, start="1996-01-01", end="2026-07-31", progress=False,
                      auto_adjust=True)["Close"]
    close = raw.dropna(how="all").sort_index()
    keep = [t for t in TICKERS if t in close.columns and close[t].notna().sum() >= 750]
    close = close[keep]
    rets = close.pct_change()
    rebal = mom.last_trading_of_period(close.index, "monthly")
    warm = max(mom.LOOKBACKS) + mom.SKIP + mom.VOL_WIN
    rebal = rebal[rebal >= close.index[warm]]
    log.info("commodity panel: %d tickers, %d days (%s..%s), %d rebalances",
             len(keep), len(close), close.index[0].date(), close.index[-1].date(), len(rebal))

    # --- T1: the house's TSMOM, unchanged, on the country universe ---
    w = mom.vol_scaled_weights(mom.tsmom_signal(close, rebal), rets, rebal)
    gross, _cd, turn = mom.backtest(w, rets)
    dw = w.fillna(0.0).diff().abs().sum(axis=1)
    dw.iloc[0] = w.iloc[0].abs().sum()

    rows = {}
    nets = {}
    for name, bps in COSTS.items():
        net = (gross - dw.reindex(rets.index).fillna(0.0) * bps).dropna()
        nets[name] = net
        rows[name] = {"sharpe": round(_sharpe(net), 3),
                      "ann_ret_pct": round(float(net.mean() * mom.ANN * 100), 2),
                      "ann_vol_pct": round(float(net.std() * np.sqrt(mom.ANN) * 100), 2),
                      "max_dd_pct": round(_max_dd(net) * 100, 2)}
    t1 = nets["etf_5bps"]

    # --- the existing validated cross-asset TSMOM sleeve ---
    base = pf.build_momentum_net()
    idx = t1.index.intersection(base.index)
    corr = float(t1.reindex(idx).corr(base.reindex(idx)))

    # --- risk-parity combine, the same function TAILWIND uses ---
    combined, _cidx, (a_al, b_al) = pf.risk_parity([base.reindex(idx), t1.reindex(idx)])
    sr_base = _sharpe(base.reindex(idx))
    sr_t1 = _sharpe(t1.reindex(idx))
    sr_comb = _sharpe(combined)

    c1 = rows["etf_5bps"]["sharpe"] >= MIN_STANDALONE_SR
    c2 = corr <= MAX_CORR_TO_TSMOM
    c3 = sr_comb > MIN_COMBINED_SR
    verdict = "PASS" if (c1 and c2 and c3) else "NO-GO"

    out = {"sleeve": "commodity_breadth_tsmom", "n_tickers": len(keep),
           "turnover_ann": round(turn, 2), "by_cost": rows,
           "overlap_days": int(len(idx)),
           "standalone_sharpe_5bp": rows["etf_5bps"]["sharpe"],
           "corr_to_cross_asset_tsmom": round(corr, 3),
           "sharpe_cross_asset_tsmom_on_overlap": round(sr_base, 3),
           "sharpe_t1_on_overlap": round(sr_t1, 3),
           "combined_risk_parity_sharpe": round(sr_comb, 3),
           "combined_max_dd_pct": round(_max_dd(combined) * 100, 2),
           "bars": {"min_standalone_sr": MIN_STANDALONE_SR, "max_corr": MAX_CORR_TO_TSMOM,
                    "min_combined_sr": MIN_COMBINED_SR},
           "conditions": {"standalone_ge_0.30": bool(c1), "corr_le_0.60": bool(c2),
                          "combined_gt_0.66": bool(c3)},
           "verdict": verdict}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "commodity_tsmom_sleeve.json").write_text(json.dumps(out, indent=2))

    print("=" * 76)
    print("T3 — COMMODITY-breadth TIME-SERIES momentum as a diversifying sleeve")
    print("=" * 76)
    print(f"tickers {len(keep)}   turnover/yr {turn:.2f}")
    print(f"{'cost':<14}{'Sharpe':>8}{'ret%':>8}{'vol%':>8}{'maxDD%':>9}")
    for k, r in rows.items():
        print(f"{k:<14}{r['sharpe']:>8.3f}{r['ann_ret_pct']:>8.2f}"
              f"{r['ann_vol_pct']:>8.2f}{r['max_dd_pct']:>9.2f}")
    print(f"\noverlap with cross-asset TSMOM: {len(idx)} days")
    print(f"  cross-asset TSMOM Sharpe (overlap): {sr_base:.3f}")
    print(f"  T1 Sharpe (overlap):                {sr_t1:.3f}")
    print(f"  correlation:                        {corr:+.3f}")
    print(f"  risk-parity COMBINED Sharpe:        {sr_comb:.3f}   maxDD {_max_dd(combined)*100:.1f}%")
    print("\npre-committed bars:")
    print(f"  standalone >= {MIN_STANDALONE_SR}:  {rows['etf_5bps']['sharpe']:.3f}  {'PASS' if c1 else 'FAIL'}")
    print(f"  corr <= {MAX_CORR_TO_TSMOM}:          {corr:+.3f}  {'PASS' if c2 else 'FAIL'}")
    print(f"  combined > {MIN_COMBINED_SR}:       {sr_comb:.3f}  {'PASS' if c3 else 'FAIL'}")
    print(f"\nVERDICT: {verdict}")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
