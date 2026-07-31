"""Do the already-measured sleeves help TOGETHER, even though none helps alone?

Today's sleeve arc tested each candidate PAIRWISE against the existing cross-asset TSMOM book and
rejected all three:

    T1 country TSMOM    standalone 0.409  rho_base +0.693   pairwise combined 0.498
    T2 crypto TSMOM     standalone -0.067 rho_base -0.063   pairwise combined 0.425
    T3 commodity TSMOM  standalone 0.356  rho_base +0.485   pairwise combined 0.568

The admission rule derived from T3 — `s2 > s1*(sqrt(2+2*rho) - 1)` — is a TWO-asset result. It says
nothing about whether T1 and T3 are correlated to EACH OTHER. If their mutual correlation is low,
the three-sleeve optimum is not determined by the pairwise tests, and the arc's conclusion would be
incomplete rather than wrong.

This is a PORTFOLIO question on FIXED, already-pre-registered, already-measured return series. No new
signal is computed, nothing is searched, and no multiplicity is incurred: every sleeve was sealed and
run before this script existed. It simply asks what the measured covariance implies.

Reports, on the common overlap window:
  * the full correlation matrix,
  * equal-risk (the combine TAILWIND actually uses) for every subset,
  * the max-Sharpe ceiling for every subset (analytic, long-only-unconstrained),
and compares each to the existing book alone.
"""
from __future__ import annotations

import itertools
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
from country_momentum_eval import TICKERS as COUNTRY  # noqa: E402

log = logging.getLogger("multisleeve")
OUT = ROOT / "results" / "multisleeve_frontier"
CMDTY = ["DBB", "DBE", "DBP", "GSG", "UNG", "USL", "UGA", "PALL", "PPLT", "BNO",
         "CORN", "CANE", "SOYB", "WEAT", "NIB", "FTGC", "COMT"]


def _sharpe(d) -> float:
    d = pd.Series(d).dropna()
    s = d.std()
    return float(d.mean() / s * np.sqrt(mom.ANN)) if s > 0 else 0.0


def _tsmom_net(tickers, start, cost_bps):
    import yfinance as yf
    raw = yf.download(tickers, start=start, end="2026-07-31", progress=False,
                      auto_adjust=True)["Close"]
    close = raw.dropna(how="all").sort_index()
    keep = [t for t in tickers if t in close.columns and close[t].notna().sum() >= 750]
    close = close[keep]
    rets = close.pct_change()
    rebal = mom.last_trading_of_period(close.index, "monthly")
    rebal = rebal[rebal >= close.index[max(mom.LOOKBACKS) + mom.SKIP + mom.VOL_WIN]]
    w = mom.vol_scaled_weights(mom.tsmom_signal(close, rebal), rets, rebal)
    gross, _c, _t = mom.backtest(w, rets)
    dw = w.fillna(0.0).diff().abs().sum(axis=1)
    dw.iloc[0] = w.iloc[0].abs().sum()
    return (gross - dw.reindex(rets.index).fillna(0.0) * cost_bps).dropna()


def _crypto_net(cost_bps):
    df = pd.read_parquet(ROOT / "data" / "crypto_cache" / "silver_ohlcv.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["day"] = df["timestamp"].dt.normalize()
    close = df.groupby(["day", "ticker"])["close"].last().unstack("ticker").sort_index()
    close.index = close.index.tz_localize(None)
    rets = close.pct_change()
    rebal = mom.last_trading_of_period(close.index, "monthly")
    rebal = rebal[rebal >= close.index[max(mom.LOOKBACKS) + mom.SKIP + mom.VOL_WIN]]
    w = mom.vol_scaled_weights(mom.tsmom_signal(close, rebal), rets, rebal)
    gross, _c, _t = mom.backtest(w, rets)
    dw = w.fillna(0.0).diff().abs().sum(axis=1)
    dw.iloc[0] = w.iloc[0].abs().sum()
    return (gross - dw.reindex(rets.index).fillna(0.0) * cost_bps).dropna()


def _equal_risk(cols: pd.DataFrame) -> float:
    """What pf.risk_parity does: scale each sleeve to a common vol, then equal-weight."""
    scaled = [c / (c.std() or 1.0) for _, c in cols.items()]
    return _sharpe(sum(scaled) / len(scaled))


def _max_sharpe(cols: pd.DataFrame) -> float:
    """Analytic unconstrained max-Sharpe: sqrt(mu' Sigma^-1 mu), annualised."""
    mu = cols.mean().to_numpy()
    cov = cols.cov().to_numpy()
    try:
        inv = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        return float("nan")
    q = float(mu @ inv @ mu)
    return float(np.sqrt(max(q, 0.0)) * np.sqrt(mom.ANN))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log.info("rebuilding the four measured sleeves ...")
    sleeves = {
        "base_cross_asset": pf.build_momentum_net(),
        "T1_country": _tsmom_net(COUNTRY, "1996-01-01", 0.0005),
        "T3_commodity": _tsmom_net(CMDTY, "2006-01-01", 0.0005),
        "T2_crypto": _crypto_net(0.0010),
    }
    for k, v in sleeves.items():
        log.info("  %-18s %5d days  standalone SR %+.3f", k, len(v), _sharpe(v))

    df = pd.DataFrame(sleeves).dropna()
    log.info("common overlap: %d days (%s..%s)", len(df), df.index[0].date(), df.index[-1].date())
    corr = df.corr()
    base_sr = _sharpe(df["base_cross_asset"])

    rows = []
    names = [c for c in df.columns if c != "base_cross_asset"]
    for r in range(0, len(names) + 1):
        for combo in itertools.combinations(names, r):
            cols = ["base_cross_asset", *combo]
            sub = df[cols]
            rows.append({"sleeves": "+".join(["base", *[c.split('_')[0] for c in combo]]),
                         "n": len(cols),
                         "equal_risk_sr": round(_equal_risk(sub), 3),
                         "max_sharpe_sr": round(_max_sharpe(sub), 3)})
    rows.sort(key=lambda x: -x["max_sharpe_sr"])

    best_eq = max(rows, key=lambda x: x["equal_risk_sr"])
    best_mx = rows[0]
    out = {"overlap_days": int(len(df)),
           "window": [str(df.index[0].date()), str(df.index[-1].date())],
           "standalone_sharpes": {k: round(_sharpe(v), 3) for k, v in sleeves.items()},
           "correlation_matrix": corr.round(3).to_dict(),
           "base_alone_sr": round(base_sr, 3),
           "subsets": rows,
           "best_equal_risk": best_eq, "best_max_sharpe": best_mx,
           "note": ("Portfolio question on FIXED pre-registered sleeves. Max-Sharpe is an "
                    "unconstrained in-sample CEILING — it is not a deployable weighting and is "
                    "shown only to bound what any allocation could achieve.")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "multisleeve_frontier.json").write_text(json.dumps(out, indent=2))

    print("=" * 78)
    print("MULTI-SLEEVE FRONTIER — do the measured sleeves help TOGETHER?")
    print("=" * 78)
    print(f"overlap {len(df)} days ({df.index[0].date()}..{df.index[-1].date()})\n")
    print("standalone Sharpe on the overlap:")
    for k in df.columns:
        print(f"   {k:<18} {_sharpe(df[k]):+.3f}")
    print("\ncorrelation matrix:")
    print(corr.round(3).to_string())
    print(f"\n{'subset':<28}{'equal-risk':>12}{'max-Sharpe':>12}")
    for r in rows:
        print(f"{r['sleeves']:<28}{r['equal_risk_sr']:>12.3f}{r['max_sharpe_sr']:>12.3f}")
    print(f"\nbase alone: {base_sr:.3f}")
    print(f"best equal-risk: {best_eq['sleeves']} = {best_eq['equal_risk_sr']:.3f} "
          f"({best_eq['equal_risk_sr']-base_sr:+.3f})")
    print(f"best max-Sharpe CEILING: {best_mx['sleeves']} = {best_mx['max_sharpe_sr']:.3f} "
          f"({best_mx['max_sharpe_sr']-base_sr:+.3f})")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
