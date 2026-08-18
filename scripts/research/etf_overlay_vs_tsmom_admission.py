"""Does the 36m/36m ETF momentum overlay earn admission to the TAILWIND book?

Pre-registered follow-on to the ETF-outperformance study (S553-cont-161,
docs/research/etf_outperformance_research_2026-08-14.md). That study found the
ONLY horizon with ex-ante predictive power over the ETF cross-section is 12-36
months, and recommended testing it as a candidate overlay rather than a new
workstream. This is that test.

ADMISSION RULE (pre-existing, from the T1/T2/T3 sleeve arc):

    admit iff  s2 > s1 * (sqrt(2 + 2*rho) - 1)

where s1 = standalone Sharpe of the existing book, s2 = standalone Sharpe of the
candidate, rho = their correlation. This is the two-asset condition for an
equal-risk combination to beat the base book alone. The rule is applied here
exactly as it was to country/crypto/commodity TSMOM -- no re-derivation, no
threshold tuning.

CANDIDATE, two variants (both pre-registered before running):
  A. xsec_ls  -- rank equity ETFs by trailing 36m excess vs SPY, long top quintile
                 / short bottom quintile, equal-weighted, dollar-neutral, 36m hold.
  B. long_only -- long the top quintile only, minus SPY. This is what a
                 benchmark-relative long-only mandate can actually hold, and the
                 study measured its gross edge at just +1.49%/yr.

Overlapping cohorts (Jegadeesh-Titman): with a 36m hold and monthly formation,
1/36 of the book is refreshed each month and the sleeve return is the average
over active cohorts. Weights formed on date f are applied from the NEXT session
(LEAK-2).

CAVEAT carried forward from the parent study: the ETF universe is survivors-only
(yfinance retains nothing delisted), so the candidate's Sharpe is an UPPER bound.
Admission failing on an upper bound is a strong verdict; admission passing would
need a survivorship-corrected re-run before it meant anything.
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

log = logging.getLogger("etf_overlay_admission")

ETF_DIR = ROOT / "results" / "etf_outperformance"
OUT = ROOT / "results" / "etf_overlay_admission"

BENCH = "SPY"
N_Q = 5
LOOKBACK_M = 36
HOLD_M = 36
TRADING_DAYS = 252
# Same cost ladder as the base book (xsec_momentum_falsification.COST_MODELS).
COST_STANDARD = 0.0002
COST_HARSH = 0.0010

EQUITY_CATS = {
    "broad_us", "us_style", "us_midsmall", "sector_spdr", "sector_vanguard",
    "sector_ishares", "industry_thematic", "factor_smartbeta", "dividend",
    "growth_tech_core", "intl_developed", "intl_emerging", "global", "reit",
}


def sharpe(daily: pd.Series) -> float:
    d = pd.Series(daily).dropna()
    s = d.std()
    return float(d.mean() / s * np.sqrt(TRADING_DAYS)) if s > 0 else 0.0


def build_overlay(px: pd.DataFrame, tickers: list[str], long_only: bool,
                  cost_bps: float) -> pd.Series:
    """Overlapping-cohort quintile momentum sleeve -> daily net return series."""
    rets = px[tickers + [BENCH]].pct_change()
    idx = px.index
    lb = int(LOOKBACK_M / 12 * TRADING_DAYS)
    hz = int(HOLD_M / 12 * TRADING_DAYS)
    form_dates = mom.last_trading_of_period(idx, "monthly")

    cohorts: list[tuple[int, int, pd.Series]] = []  # (start_pos, end_pos, weights)
    for f in form_dates:
        pos = idx.searchsorted(f)
        if pos - lb < 0 or pos + 1 >= len(idx):
            continue
        t0, tf = idx[pos - lb], idx[pos]
        past = px.loc[tf, tickers] / px.loc[t0, tickers] - 1.0
        past_b = px.loc[tf, BENCH] / px.loc[t0, BENCH] - 1.0
        pe = (past - past_b).dropna()
        if len(pe) < 30:
            continue
        q = np.ceil(pe.rank(pct=True) * N_Q).clip(1, N_Q).astype(int)
        top, bot = pe.index[q == N_Q], pe.index[q == 1]
        if len(top) == 0 or (not long_only and len(bot) == 0):
            continue
        w = pd.Series(0.0, index=tickers + [BENCH])
        if long_only:
            # long top quintile, short the benchmark => benchmark-relative sleeve
            w[top] = 1.0 / len(top)
            w[BENCH] = -1.0
        else:
            w[top] = 0.5 / len(top)
            w[bot] = -0.5 / len(bot)
        # applied from the NEXT session (LEAK-2)
        cohorts.append((pos + 1, min(pos + 1 + hz, len(idx)), w))

    if not cohorts:
        raise RuntimeError("no cohorts formed")

    cols = tickers + [BENCH]
    W = pd.DataFrame(0.0, index=idx, columns=cols)
    active = pd.Series(0.0, index=idx)
    for s, e, w in cohorts:
        W.iloc[s:e] = W.iloc[s:e].add(w, axis=1)
        active.iloc[s:e] += 1.0
    W = W.div(active.replace(0.0, np.nan), axis=0).fillna(0.0)  # average over active cohorts

    gross = (W.shift(1) * rets[cols]).sum(axis=1)
    turn = W.diff().abs().sum(axis=1)
    net = gross - turn * cost_bps
    net = net[active > 0]
    net.attrs["turnover_ann"] = float(turn[active > 0].mean() * TRADING_DAYS)
    return net.dropna()


def admission(s1: float, s2: float, rho: float) -> tuple[float, bool]:
    thresh = s1 * (np.sqrt(2 + 2 * rho) - 1)
    return float(thresh), bool(s2 > thresh)


def equal_risk(a: pd.Series, b: pd.Series) -> float:
    df = pd.concat([a, b], axis=1).dropna()
    scaled = [c / (c.std() or 1.0) for _, c in df.items()]
    return sharpe(sum(scaled) / len(scaled))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    px = pd.read_parquet(ETF_DIR / "etf_prices_adj.parquet")
    meta = pd.read_parquet(ETF_DIR / "etf_universe_meta.parquet").set_index("ticker")
    eq = [t for t in px.columns
          if t in meta.index and meta.loc[t, "category"] in EQUITY_CATS and t != BENCH]
    log.info("candidate universe: %d equity ETFs", len(eq))

    log.info("rebuilding the validated base book (cross-asset TSMOM) ...")
    base = pf.build_momentum_net()
    log.info("  base: %d days, standalone SR %+.3f", len(base), sharpe(base))

    results = {}
    for variant, long_only in [("A_xsec_ls", False), ("B_long_only", True)]:
        for cost_name, cost in [("standard_2bps", COST_STANDARD), ("harsh_10bps", COST_HARSH)]:
            cand = build_overlay(px, eq, long_only=long_only, cost_bps=cost)
            df = pd.concat([base.rename("base"), cand.rename("cand")], axis=1).dropna()
            if len(df) < 500:
                log.warning("%s/%s: overlap only %d days, skipping", variant, cost_name, len(df))
                continue
            s1, s2 = sharpe(df.base), sharpe(df.cand)
            rho = float(df.base.corr(df.cand))
            thresh, admit = admission(s1, s2, rho)
            combined = equal_risk(df.base, df.cand)
            results[f"{variant}|{cost_name}"] = {
                "variant": variant, "cost": cost_name,
                "overlap_days": len(df),
                "window": f"{df.index[0].date()}..{df.index[-1].date()}",
                "s1_base": s1, "s2_candidate": s2, "rho": rho,
                "admission_threshold": thresh, "ADMIT": admit,
                "margin": s2 - thresh,
                "combined_equal_risk_sr": combined,
                "uplift_vs_base": combined - s1,
                "turnover_ann": cand.attrs.get("turnover_ann", float("nan")),
            }

    pd.set_option("display.width", 220, "display.max_columns", 30)
    tab = pd.DataFrame(results).T
    print("\n" + "=" * 110)
    print("ADMISSION TEST — 36m/36m ETF momentum overlay vs the validated TSMOM book")
    print("rule: admit iff s2 > s1 * (sqrt(2 + 2*rho) - 1)")
    print("=" * 110)
    print(tab[["window", "overlap_days", "s1_base", "s2_candidate", "rho",
               "admission_threshold", "margin", "ADMIT", "combined_equal_risk_sr",
               "uplift_vs_base", "turnover_ann"]].to_string(
        float_format=lambda x: f"{x:,.3f}"))

    (OUT / "admission.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT / 'admission.json'}")

    any_admit = any(v["ADMIT"] for v in results.values())
    print(f"\nVERDICT: {'ADMIT in >=1 cell' if any_admit else 'REJECT in every cell'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
