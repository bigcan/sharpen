"""Point-in-time growth universe for the FinRL-X hindsight-universe test.

FinRL-X's Adaptive Rotation hard-codes its growth group as AAPL, MSFT, NVDA, META, AMZN, GOOGL, TSLA
(2026's Magnificent 7) and backtests it from 2018. This script builds the universe an investor could
have drawn on 2017-12-29 instead:

  pool = the POOL_SIZE largest S&P 500 members on AS_OF by market cap, among 2017-era GICS
         Information Technology + Consumer Discretionary (the two sectors the Magnificent 7 sat in then).

Market cap = shares outstanding x the unadjusted close on AS_OF.
  - Shares: SEC XBRL frames, one request each. Primary = the latest cover-page
    ``dei:EntityCommonStockSharesOutstanding`` dated on or before AS_OF (cover pages are never restated, so
    the count is as filed). Fallbacks for multi-class filers that tag the cover count per class only: the
    2017-09-30 balance-sheet ``CommonStockSharesOutstanding`` (all classes), then CY2017Q3 diluted weighted
    shares. Registrants that changed CIK since 2017 are read under their 2017 CIK (``LEGACY_CIK``); Visa,
    which tags no all-class count, uses its own as-converted total (``SPECIAL_SHARES``).
  - Close: yfinance ``Close`` (split-adjusted, not dividend-adjusted) times every split ratio with an
    ex-date after AS_OF, i.e. the close as printed that day.
2017-era sectors come from today's GICS plus the documented reclassifications: Telecommunication
Services (T, VZ, TMUS, LUMN) was its own sector in 2017, and payments/processors moved from IT to
Financials or Industrials in 2023.

Writes results/finrl_x/hindsight/pool.json (+ pool.csv): every input, the ranking, the pool, the
deterministic top-7 and N_DRAWS seeded random 7-name groups, with a sha256 of the draw spec.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger("finrl_x_pit_universe")

AS_OF = "2017-12-29"
POOL_SIZE = 20
GROUP_SIZE = 7
N_DRAWS = 35
SEED = 20260925
FRAMES = {
    "dei_q3": "https://data.sec.gov/api/xbrl/frames/dei/EntityCommonStockSharesOutstanding/shares/CY2017Q3I.json",
    "dei_q4": "https://data.sec.gov/api/xbrl/frames/dei/EntityCommonStockSharesOutstanding/shares/CY2017Q4I.json",
    "bs_q3": "https://data.sec.gov/api/xbrl/frames/us-gaap/CommonStockSharesOutstanding/shares/CY2017Q3I.json",
    "diluted_q3": "https://data.sec.gov/api/xbrl/frames/us-gaap/WeightedAverageNumberOfDilutedSharesOutstanding/shares/CY2017Q3.json",
}
# Filers whose 2017 SEC registrant differs from today's (the frames are keyed by the 2017 filer).
# The matched entity name is logged so a wrong CIK is visible.
LEGACY_CIK = {"DIS": 1001039,   # TWDC Enterprises 18 (Walt Disney before the 2019 Fox deal holdco)
              "AVGO": 1649338,  # Broadcom Ltd (Singapore) before the 2018 redomicile
              "FOXA": 1308161, "FOX": 1308161}  # Twenty-First Century Fox before the 2019 spin
# Multi-class filers that tag no all-class count: take the filer's own as-converted total.
SPECIAL_SHARES = {
    # Visa 10-K 2017-11-17, Note 13 "As-converted class A common stock" table total (2,350M = 32 + 44
    # preferred + 1,818 A + 405 B@1.6483 + 51 C@4.0000), tagged ConversionOfStockSharesConverted1
    "V": (1403161, "us-gaap", "ConversionOfStockSharesConverted1"),
}
SECTORS_2017 = {"Information Technology", "Consumer Discretionary"}
# 2017-era GICS differs from today's for these names (current ticker -> 2017 sector)
GICS_2017_OVERRIDES = {
    # Telecommunication Services was a separate sector until 2018-09
    "T": "Telecommunication Services", "VZ": "Telecommunication Services",
    "TMUS": "Telecommunication Services", "LUMN": "Telecommunication Services",
    # payments / processors moved out of IT in 2023
    "V": "Information Technology", "MA": "Information Technology", "PYPL": "Information Technology",
    "FIS": "Information Technology", "FI": "Information Technology", "GPN": "Information Technology",
    "JKHY": "Information Technology", "ADP": "Information Technology", "PAYX": "Information Technology",
    "BR": "Information Technology",
}
# second listed class of a company already in the list (one ticker per company; GOOGL is the Mag-7's class)
SECONDARY_CLASSES = {"GOOG", "FOX", "NWS"}
REFINE_TOP = 40  # re-size the top of the frames screen from companyfacts; the pool cut sits at #20
MAX_AGE_DAYS = 200  # a share count must be dated within this many days before AS_OF
STALE_FLOOR = str((pd.Timestamp(AS_OF) - pd.Timedelta(days=MAX_AGE_DAYS)).date())
# end-2017 S&P 500 tickers renamed since (old -> current)
RENAMES = {"FB": "META", "PCLN": "BKNG", "FISV": "FI"}
MAG7 = ["AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL", "TSLA"]


def sector_2017(ticker: str, current: str) -> str:
    """2017-era GICS sector. Communication Services was created in 2018-09 from Telecommunication
    Services (overridden above) plus media / interactive-media names that sat in IT or Consumer
    Discretionary, so every other current Communication Services name is in scope for 2017."""
    if ticker in GICS_2017_OVERRIDES:
        return GICS_2017_OVERRIDES[ticker]
    if current == "Communication Services":
        return "IT or Consumer Discretionary (Communication Services since 2018)"
    return current


def in_scope_2017(sector: str) -> bool:
    return sector in SECTORS_2017 or sector.startswith("IT or Consumer Discretionary")


def members_on(date: str, path: Path) -> set[str]:
    snaps = pd.read_csv(path, parse_dates=["date"])
    row = snaps[snaps["date"] <= pd.Timestamp(date)].iloc[-1]
    return {RENAMES.get(t, t) for t in row["tickers"].split(",")}


def latest_shares(frames: dict[str, dict]) -> pd.DataFrame:
    """Per CIK, in preference order: the latest cover-page count dated <= AS_OF; the 2017-09-30 balance-sheet
    count (all classes); diluted weighted-average shares for CY2017Q3."""
    rows = []
    for key in ("dei_q3", "dei_q4"):
        for d in frames[key]["data"]:
            if d["end"] <= AS_OF:
                rows.append({"cik": d["cik"], "shares": d["val"], "shares_date": d["end"], "source": "dei_cover",
                             "entity": d["entityName"]})
    dei = (pd.DataFrame(rows).sort_values("shares_date").groupby("cik").tail(1).set_index("cik"))

    def frame(key: str, col: str) -> pd.DataFrame:
        return (pd.DataFrame([{"cik": d["cik"], col: d["val"], f"{col}_end": d["end"], f"{col}_entity": d["entityName"]}
                              for d in frames[key]["data"]]).drop_duplicates("cik").set_index("cik"))

    out = frame("diluted_q3", "diluted").join(frame("bs_q3", "bs"), how="outer").join(dei, how="outer")
    for col, source in (("bs", "balance_sheet_2017-09-30"), ("diluted", "diluted_wavg_cy2017q3")):
        missing = out["shares"].isna() & out[col].notna()
        out.loc[missing, "shares"] = out.loc[missing, col]
        out.loc[missing, "shares_date"] = out.loc[missing, f"{col}_end"]
        out.loc[missing, "source"] = source
        out.loc[missing, "entity"] = out.loc[missing, f"{col}_entity"]
    out["cover_vs_diluted"] = out["shares"] / out["diluted"]
    return out


def companyfacts_shares(transport, cik: int) -> dict | None:
    """PIT-precise share count from one filer's companyfacts: the latest fact dated AND filed on or before
    AS_OF, preferring the cover-page count, then the balance-sheet count, then quarterly diluted weighted
    shares. (The frames API assigns some facts to no frame at all -- Disney's 2017-11-15 cover count --
    so the frames pass is only a screen.)"""
    facts = transport(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json")
    for tax, tag, label, is_flow in (("dei", "EntityCommonStockSharesOutstanding", "cover", False),
                                     ("us-gaap", "CommonStockSharesOutstanding", "balance_sheet", False),
                                     ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding", "diluted_wavg", True)):
        obs = facts["facts"].get(tax, {}).get(tag, {}).get("units", {}).get("shares", [])
        # dated within MAX_AGE_DAYS: multi-class filers stop tagging an all-class cover count, leaving a years-old
        # "latest" fact (Comcast's is from 2010, before its 2017 split) -- fall through to the next source instead
        obs = [o for o in obs if STALE_FLOOR <= o["end"] <= AS_OF and o["filed"] <= AS_OF]
        if is_flow:
            obs = [o for o in obs if "start" in o and 80 <= (pd.Timestamp(o["end"]) - pd.Timestamp(o["start"])).days <= 100]
        if obs:
            o = max(obs, key=lambda x: (x["end"], x["filed"]))
            return {"shares": float(o["val"]), "shares_date": o["end"], "shares_filed": o["filed"],
                    "source": f"cf_{label} ({o.get('form')})", "entity": facts["entityName"]}
    return None


def special_shares(transport) -> dict[str, dict]:
    """SPECIAL_SHARES from companyfacts: the latest fact dated and filed on or before AS_OF."""
    out = {}
    for ticker, (cik, tax, tag) in SPECIAL_SHARES.items():
        facts = transport(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
        obs = [o for o in facts["facts"][tax][tag]["units"]["shares"] if o["end"] <= AS_OF and o["filed"] <= AS_OF]
        o = max(obs, key=lambda x: (x["end"], x["filed"]))
        out[ticker] = {"shares": float(o["val"]), "shares_date": o["end"], "shares_filed": o["filed"],
                       "entity": facts["entityName"], "source": f"cf_{tag} ({o.get('form')})"}
    return out


def unadjusted_close(tickers: list[str]) -> pd.DataFrame:
    """Close on AS_OF as printed that day: split-adjusted Close x all later split ratios."""
    import yfinance as yf

    raw = yf.download(tickers, start="2017-12-15", end=None, auto_adjust=False, actions=True,
                      progress=False, group_by="ticker", threads=True)
    rows = []
    for t in tickers:
        if t not in raw.columns.get_level_values(0):
            continue
        df = raw[t]
        close = df["Close"].loc[:AS_OF].dropna()
        if close.empty or close.index[-1] != pd.Timestamp(AS_OF):
            continue
        splits = df["Stock Splits"].loc[df.index > pd.Timestamp(AS_OF)]
        factor = float(np.prod(splits[splits > 0].to_numpy())) if (splits > 0).any() else 1.0
        rows.append({"ticker": t, "close_split_adj": float(close.iloc[-1]), "split_factor_after": factor,
                     "close_unadjusted": float(close.iloc[-1]) * factor})
    return pd.DataFrame(rows).set_index("ticker")


def draw_groups(pool: list[str], n: int, k: int, seed: int) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    seen, draws = set(), []
    while len(draws) < n:
        g = tuple(sorted(rng.choice(pool, size=k, replace=False).tolist()))
        if g not in seen:
            seen.add(g)
            draws.append(list(g))
    return draws


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "finrl_x" / "hindsight")
    ap.add_argument("--data-root", type=Path, default=Path("C:/FinRL/FinRL-Pro_DS/data"),
                    help="primary checkout data dir (gitignored, not present in worktrees)")
    ap.add_argument("--env-file", type=Path, default=Path("C:/FinRL/FinRL-Pro_DS/.env"),
                    help="holds SEC_EDGAR_UA")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # imported here so the draw helpers stay importable without the SEC client
    from dotenv import load_dotenv

    sys.path.insert(0, str(ROOT))
    from sharpen.data.fundamentals import _live_transport

    load_dotenv(args.env_file)
    transport = _live_transport(rate_per_sec=2.0)
    frames = {k: transport(u) for k, u in FRAMES.items()}
    logger.info("frames: %s", {k: len(v["data"]) for k, v in frames.items()})

    cons = pd.read_csv(args.data_root / "raw" / "equity_panel" / "sp500_constituents.csv")
    cons["ticker"] = cons["Symbol"].str.replace(".", "-", regex=False)
    cons["sector_2017"] = [sector_2017(t, s) for t, s in zip(cons["ticker"], cons["GICS Sector"])]
    members = members_on(AS_OF, args.data_root / "raw" / "equity_panel" / "sp500_pit_members.csv")
    cons["member_2017"] = cons["Symbol"].isin(members) | cons["ticker"].isin(members)
    gone = sorted(t for t in members if t not in set(cons["Symbol"]) | set(cons["ticker"]))
    logger.info("end-2017 members not in today's index (unpriceable, listed for review): %d -> %s",
                len(gone), " ".join(gone))

    cand = cons[cons["member_2017"] & cons["sector_2017"].map(in_scope_2017)].copy()
    cand = cand[~cand["ticker"].isin(SECONDARY_CLASSES)]
    cand["share_cik"] = [LEGACY_CIK.get(t, c) for t, c in zip(cand["ticker"], cand["CIK"])]
    cand = cand.join(latest_shares(frames), on="share_cik")
    cand = cand.join(unadjusted_close(cand["ticker"].tolist()), on="ticker")

    # frames are only a screen; the ranking that matters is re-sized from each filer's own companyfacts
    screen_cap = cand["shares"] * cand["close_unadjusted"]
    priced = cand["close_unadjusted"].notna()
    refine = priced & ((screen_cap.rank(ascending=False) <= REFINE_TOP) | screen_cap.isna())
    cand["refined"], cand["shares_filed"] = False, None
    for idx in cand.index[refine]:
        s = companyfacts_shares(transport, cand.at[idx, "share_cik"])
        if s:
            for k, v in s.items():
                cand.at[idx, k] = v
            cand.at[idx, "refined"] = True
    for t, s in special_shares(transport).items():
        idx = cand.index[cand["ticker"] == t]
        for k, v in s.items():
            cand.loc[idx, k] = v
        cand.loc[idx, "refined"] = True
    logger.info("refined %d of %d candidates from companyfacts (top %d by the frames screen + unsized)",
                int(cand["refined"].sum()), len(cand), REFINE_TOP)
    for t in sorted(set(LEGACY_CIK) | set(SPECIAL_SHARES)):
        row = cand[cand["ticker"] == t]
        if not row.empty:
            logger.info("  %-5s shares from CIK %s = %r via %s", t, row["share_cik"].iloc[0],
                        row["entity"].iloc[0], row["source"].iloc[0])
    cand["market_cap"] = cand["shares"] * cand["close_unadjusted"]
    ranked = cand.dropna(subset=["market_cap"]).sort_values("market_cap", ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    unpriced = cand[cand["market_cap"].isna()]["ticker"].tolist()
    if not ranked.head(POOL_SIZE)["refined"].all():
        raise RuntimeError("a pool member was sized only by the frames screen")

    pool = ranked["ticker"].head(POOL_SIZE).tolist()
    top7 = pool[:GROUP_SIZE]
    draw_spec = {"as_of": AS_OF, "pool": sorted(pool), "k": GROUP_SIZE, "n": N_DRAWS, "seed": SEED}
    draws = draw_groups(sorted(pool), N_DRAWS, GROUP_SIZE, SEED)
    cols = ["rank", "ticker", "Security", "sector_2017", "CIK", "share_cik", "entity", "shares", "shares_date",
            "shares_filed", "source", "refined", "cover_vs_diluted", "close_split_adj", "split_factor_after",
            "close_unadjusted", "market_cap"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ranked[cols].to_csv(args.out_dir / "pool_ranking.csv", index=False)
    out = {
        "rule": f"top {POOL_SIZE} S&P 500 members on {AS_OF} by market cap, 2017-era GICS {sorted(SECTORS_2017)}",
        "as_of": AS_OF, "sources": FRAMES, "gics_2017_overrides": GICS_2017_OVERRIDES, "renames": RENAMES,
        "n_candidates": int(len(cand)), "unpriced_candidates": unpriced, "gone_since_2017": gone,
        "pool": pool, "deterministic_top7": top7, "mag7_reference": MAG7,
        "mag7_ranks": {t: (int(ranked.loc[ranked["ticker"] == t, "rank"].iloc[0])
                           if (ranked["ticker"] == t).any() else None) for t in MAG7},
        "draw_spec": draw_spec, "draw_spec_sha256": hashlib.sha256(json.dumps(draw_spec, sort_keys=True).encode()).hexdigest(),
        "random_draws": draws,
        "top30": ranked[cols].head(30).to_dict(orient="records"),
    }
    (args.out_dir / "pool.json").write_text(json.dumps(out, indent=2, default=str))
    logger.info("top 25 by 2017-12-29 market cap ($B):")
    for r in ranked.head(25).itertuples():
        logger.info("  %2d %-6s %-24s %7.1f  %-40s filed %-10s  split x%g",
                    r.rank, r.ticker, r.sector_2017[:24], r.market_cap / 1e9, r.source, r.shares_filed,
                    r.split_factor_after)
    logger.info("Mag-7 ranks: %s", out["mag7_ranks"])
    logger.info("pool: %s", " ".join(pool))
    logger.info("deterministic top-7: %s", " ".join(top7))
    logger.info("unpriced candidates: %s", unpriced)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
