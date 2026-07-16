"""Taiwan small/mid-cap ALT-DATA ingestion via FinMind → tidy Parquet (probe step 1).

Step one of the pre-registered small/mid-cap alt-data probes
(``docs/research/taiwan_smallcap_altdata_probes_preregistration_2026-07-15.md``). Pulls the three
niche retail-behaviour channels the free daily price feed cannot supply, plus the common-stock
POOL prices needed to build the cap-rank 51-250 universe:

  - ``TaiwanStockMonthRevenue``          → month-revenue momentum   (P1)
  - ``TaiwanStockMarginPurchaseShortSale`` → margin / short crowding  (P2)
  - ``TaiwanStockHoldingSharesPer``      → 集保 big-holder concentration + total shares (P3 + cap-rank)
  - ``TaiwanStockPrice``                 → RAW daily bars for the pool (LEAK-2: unadjusted)
  - ``TaiwanStockInfo`` / ``TaiwanStockDelisting`` → pool enumeration + survivorship

WHY a separate script (not folded into ``fetch_taiwan_finmind``): that module is the single source
of the OHLCV contract; we IMPORT its verified HTTP + normalize + clean helpers here and add the
alt-data channels on top, so the price/clean path stays byte-identical and only the new datasets
live here.

THE ONE CORRECTNESS PROPERTY THAT MATTERS (LEAK-2, audit-critical): each alt-data row is stamped
with a causal ``avail_date`` — the first date on which the value is *publicly known* — computed HERE
so the lag lives in exactly one place and the panel builder cannot reintroduce a leak:
  - month-revenue: mandatory disclosure by the **10th of the following month** → ``avail_date`` =
    10th of ``revenue_month + 1`` (never the revenue month-end).
  - margin/short:  published after the close of day T → ``avail_date`` = **T + 1 business day**.
  - 集保 shareholding: TDCC as-of a Friday, released the following week → ``avail_date`` =
    ``date + 6 calendar days`` (conservative).
The panel builder then activates a signal only from the first trading day ``>= avail_date``.

Fail-loud, never silent: (a) every channel asserts its expected columns and HALTS with a clear diff
on schema drift (mirrors ``fetch_taiwan_finmind._normalize``) rather than mis-mapping; (b) a FinMind
account-TIER/permission block (HTTP 400 "level is register" — the free-tier lock seen on intraday in
June) HALTS with an actionable "needs a higher tier" message on the FIRST ticker, instead of writing
an empty parquet after 1000 identical skip-warnings; (c) HTTP 402 quota is surfaced explicitly.
``--selftest`` exercises the pure lag/parse/tier-detect helpers with NO token.

Usage:
    export FINMIND_TOKEN=...                       # Sponsor tier; env only, never committed
    python scripts/data/fetch_taiwan_fundamentals_finmind.py \
        --datasets month_revenue,margin_short,shareholding,prices \
        --pool auto --start 2005-01-01 --out data/taiwan_smallcap
    python scripts/data/fetch_taiwan_fundamentals_finmind.py --selftest   # offline, no token
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Single-source contract: reuse the verified FinMind HTTP + OHLCV normalize/clean from step 1.
from scripts.data.fetch_taiwan_finmind import (  # noqa: E402
    _STOCK_MAP,
    _clean_long,
    _finmind_get,
    _normalize,
)

log = logging.getLogger("fetch_taiwan_fundamentals")

BOARD_LOT = 1000  # 1 board lot (張) = 1000 shares. Big-holder tier = ">400 lots" = >400,000 shares.


# --------------------------------------------------------------------------- #
# Causal availability lags (LEAK-2) — pure, unit-tested by --selftest and pytest
# --------------------------------------------------------------------------- #
def month_revenue_avail_date(revenue_year: int, revenue_month: int) -> pd.Timestamp:
    """First public date for a month's revenue: the 10th of the FOLLOWING month (statutory deadline).

    April (month 4) revenue is disclosed by May 10 → avail 2000-05-10. December rolls to next
    January. Conservative (the deadline, not the earliest voluntary print) — can only DELAY a
    signal's activation, never advance it (no look-ahead).
    """
    y, m = int(revenue_year), int(revenue_month) + 1
    if m > 12:
        y, m = y + 1, 1
    return pd.Timestamp(year=y, month=m, day=10)


def _avail_plus_bdays(dates: pd.Series, n: int) -> pd.Series:
    """``date + n`` business days — margin/short balance known at the next session's open (T+1)."""
    return pd.to_datetime(dates) + pd.tseries.offsets.BDay(n)


def _avail_plus_days(dates: pd.Series, n: int) -> pd.Series:
    """``date + n`` calendar days — 集保 weekly as-of released the following week (conservative)."""
    return pd.to_datetime(dates) + pd.Timedelta(days=n)


def parse_holding_lower_bound(level: object) -> float | None:
    """Lower bound (in SHARES) of a 集保 ``HoldingSharesLevel``, or ``None`` for the total/unknown row.

    Handles both FinMind schemes:
      - string labels: ``"1-999"``, ``"1,000-5,000"``, ``"400,001-600,000"``,
        ``"more than 1,000,001"``, ``"total"`` / ``"合計"`` → lower bound = first integer, total → None.
      - integer level codes 1..15 (+ 16/17 total): mapped to the standard TDCC tier lower bounds.
    Returns the lower bound so the caller can sum ``percent`` over tiers with lower bound
    ``> 400*BOARD_LOT`` (the ">400 lots" big-holder definition, pre-registered).
    """
    if level is None:
        return None
    s = str(level).strip().lower()
    if s in {"total", "合計", "总计", "總計", ""}:
        return None
    # Standard TDCC 15-tier lower bounds (shares) when the level is a bare integer code 1..15.
    tdcc = {1: 1, 2: 1_000, 3: 5_001, 4: 10_001, 5: 15_001, 6: 20_001, 7: 30_001, 8: 40_001,
            9: 50_001, 10: 100_001, 11: 200_001, 12: 400_001, 13: 600_001, 14: 800_001,
            15: 1_000_001}
    if s.isdigit():
        code = int(s)
        if code in tdcc:
            return float(tdcc[code])
        return None  # 16/17 = total or an unknown code → treat as non-tier
    # Strip thousands-commas ("400,001"→"400001") FIRST, then treat any non-digit (the "-" range
    # separator, "more than ") as a token break so the LOWER bound is the first integer.
    t = s.replace(",", "")
    tokens = "".join(ch if ch.isdigit() else " " for ch in t).split()
    if not tokens:
        return None
    return float(int(tokens[0]))


def big_holder_percent(group: pd.DataFrame, threshold_lots: int = 400) -> float | None:
    """Sum of ``percent`` over 集保 tiers whose lower bound ``> threshold_lots`` board lots.

    ``group`` = all tier rows for one (stock, date). Returns the big-holder share of the register
    (a percent in [0,100]), or ``None`` if no tier row parses (loud-failure signal upstream).
    """
    thresh_shares = threshold_lots * BOARD_LOT
    tot = 0.0
    seen = False
    for _, r in group.iterrows():
        lb = parse_holding_lower_bound(r.get("HoldingSharesLevel"))
        if lb is None:
            continue
        if lb > thresh_shares:
            pct = pd.to_numeric(r.get("percent"), errors="coerce")
            if pd.notna(pct):
                tot += float(pct)
                seen = True
    return tot if seen else None


def _holding_total_shares(group: pd.DataFrame) -> float | None:
    """Total registered shares for one (stock, date) from the 集保 'total' (合計) tier ``unit``."""
    for _, r in group.iterrows():
        if parse_holding_lower_bound(r.get("HoldingSharesLevel")) is None:
            lvl = str(r.get("HoldingSharesLevel")).strip().lower()
            if lvl in {"total", "合計", "總計", "总计"} or lvl in {"16", "17"}:
                u = pd.to_numeric(r.get("unit"), errors="coerce")
                if pd.notna(u) and u > 0:
                    return float(u)
    return None


def _assert_cols(df: pd.DataFrame, need: set[str], dataset: str, sid: str) -> None:
    missing = need - set(df.columns)
    if missing:
        raise RuntimeError(
            f"{dataset}/{sid}: FinMind frame missing {sorted(missing)} "
            f"(got {list(df.columns)}) — dataset contract drifted; fix the mapping, do not mis-map.")


# FinMind account-tier / permission block markers. A tier block is SYSTEMATIC (it fails the whole
# dataset, so it hits the very first ticker) — distinct from a transient timeout or a single-id
# dataset miss. FinMind signals it as HTTP 400 "level is register" (the free-tier lock seen on
# intraday in June) or a payload permission message. Detecting it lets the fetcher HALT loudly with
# an actionable message instead of writing an empty parquet after 1000 identical skip-warnings.
_TIER_BLOCK_MARKERS = ("level is register", "level is ", "permission", "upgrade", "sponsor", "backer",
                       "無權限", "權限不足", "訂閱")


def _is_tier_block(err: Exception) -> bool:
    """True if a FinMind error looks like an account-tier/permission block (needs a higher tier)."""
    s = str(err).lower()
    return any(m in s for m in _TIER_BLOCK_MARKERS)


# --------------------------------------------------------------------------- #
# Pool enumeration
# --------------------------------------------------------------------------- #
def enumerate_common_stock_pool(token: str) -> pd.DataFrame:
    """All TWSE+TPEx COMMON stocks from ``TaiwanStockInfo`` → ``[stock_id, name, sector, type]``.

    Filters out ETFs (``00xx``), warrants/DR/preferred, and non-4-digit ids. This is the ranking
    pool; the cap-rank 51-250 cut happens later in the universe builder.
    """
    info = _finmind_get("TaiwanStockInfo", None, "2000-01-01", None, token)
    _assert_cols(info, {"stock_id", "industry_category", "type"}, "TaiwanStockInfo", "*")
    info = info.copy()
    info["stock_id"] = info["stock_id"].astype(str)
    name_col = "stock_name" if "stock_name" in info.columns else "industry_category"
    is_common = (
        info["stock_id"].str.fullmatch(r"\d{4}")
        & ~info["stock_id"].str.startswith("00")                       # ETFs
        & info["type"].astype(str).str.lower().isin(["twse", "tpex"])
        & ~info["industry_category"].astype(str).str.contains(
            "ETF|ETN|存託|受益|特別股|指數", case=False, na=False)
    )
    pool = (info[is_common][["stock_id", name_col, "industry_category", "type"]]
            .drop_duplicates("stock_id")
            .rename(columns={name_col: "name", "industry_category": "sector"})
            .sort_values("stock_id").reset_index(drop=True))
    return pool


# --------------------------------------------------------------------------- #
# Per-channel tidy fetch (each returns a long DataFrame with a causal avail_date)
# --------------------------------------------------------------------------- #
def fetch_month_revenue(sid: str, start: str, end: str | None, token: str) -> pd.DataFrame:
    """``TaiwanStockMonthRevenue`` → ``[stock_id, revenue_year, revenue_month, revenue, avail_date]``."""
    df = _finmind_get("TaiwanStockMonthRevenue", sid, start, end, token)
    if df.empty:
        return df
    _assert_cols(df, {"revenue_year", "revenue_month", "revenue"}, "TaiwanStockMonthRevenue", sid)
    out = pd.DataFrame({
        "stock_id": str(sid),
        "revenue_year": pd.to_numeric(df["revenue_year"], errors="coerce").astype("Int64"),
        "revenue_month": pd.to_numeric(df["revenue_month"], errors="coerce").astype("Int64"),
        "revenue": pd.to_numeric(df["revenue"], errors="coerce"),
    }).dropna(subset=["revenue_year", "revenue_month"])
    out["avail_date"] = [month_revenue_avail_date(y, m)
                         for y, m in zip(out["revenue_year"], out["revenue_month"])]
    return out.sort_values(["revenue_year", "revenue_month"]).reset_index(drop=True)


def fetch_margin_short(sid: str, start: str, end: str | None, token: str) -> pd.DataFrame:
    """``TaiwanStockMarginPurchaseShortSale`` → ``[date, stock_id, margin_balance, short_balance, avail_date]``."""
    df = _finmind_get("TaiwanStockMarginPurchaseShortSale", sid, start, end, token)
    if df.empty:
        return df
    _assert_cols(df, {"date", "MarginPurchaseTodayBalance", "ShortSaleTodayBalance"},
                 "TaiwanStockMarginPurchaseShortSale", sid)
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"]),
        "stock_id": str(sid),
        "margin_balance": pd.to_numeric(df["MarginPurchaseTodayBalance"], errors="coerce"),
        "short_balance": pd.to_numeric(df["ShortSaleTodayBalance"], errors="coerce"),
    })
    out["avail_date"] = _avail_plus_bdays(out["date"], 1)          # known at the next session (T+1)
    return out.sort_values("date").reset_index(drop=True)


def fetch_dividends(sid: str, start: str, end: str | None, token: str) -> pd.DataFrame:
    """``TaiwanStockDividendResult`` → ``[ex_date, stock_id, amount]`` (realized ex-dividend cash drop).

    ``amount = before_price - after_price`` is the exact reference-price adjustment on the ex-date;
    used by the eval to build a CAUSAL total-return label (forward add-back, never back-adjustment —
    the LEAK-2-clean shape of ``taiwan_panel_loader._apply_causal_total_return``). Optional channel.
    """
    df = _finmind_get("TaiwanStockDividendResult", sid, start, end, token)
    if df.empty:
        return df
    _assert_cols(df, {"date", "before_price", "after_price"}, "TaiwanStockDividendResult", sid)
    amt = pd.to_numeric(df["before_price"], errors="coerce") - pd.to_numeric(df["after_price"], errors="coerce")
    out = pd.DataFrame({"ex_date": pd.to_datetime(df["date"]), "stock_id": str(sid), "amount": amt}).dropna()
    return out[out["amount"] > 0].sort_values("ex_date").reset_index(drop=True)


def fetch_shareholding(sid: str, start: str, end: str | None, token: str,
                       *, threshold_lots: int = 400) -> pd.DataFrame:
    """``TaiwanStockHoldingSharesPer`` (集保, per-tier) → per-date ``big_holder_pct`` + ``total_shares``.

    Collapses the multi-tier weekly distribution to one row per (stock, date):
    ``[date, stock_id, big_holder_pct, total_shares, avail_date]``. Big-holder = tiers with lower
    bound ``> threshold_lots`` board lots (pre-registered 400). Loud-fails if NO date yields a
    parseable tier (contract drift), rather than silently emitting all-NaN.
    """
    df = _finmind_get("TaiwanStockHoldingSharesPer", sid, start, end, token)
    if df.empty:
        return df
    _assert_cols(df, {"date", "HoldingSharesLevel", "percent"}, "TaiwanStockHoldingSharesPer", sid)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    rows = []
    for d, g in df.groupby("date"):
        bhp = big_holder_percent(g, threshold_lots)
        tot = _holding_total_shares(g)
        rows.append({"date": d, "stock_id": str(sid), "big_holder_pct": bhp, "total_shares": tot})
    out = pd.DataFrame(rows)
    if out["big_holder_pct"].notna().sum() == 0:
        levels = sorted(df["HoldingSharesLevel"].astype(str).unique())[:20]
        raise RuntimeError(
            f"TaiwanStockHoldingSharesPer/{sid}: no tier parsed as big-holder — level labels seen: "
            f"{levels}. Fix parse_holding_lower_bound() for this scheme; do not ship all-NaN.")
    out["avail_date"] = _avail_plus_days(out["date"], 6)          # TDCC released the following week
    return out.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
_CHANNELS = {
    "prices": None,  # special-cased (OHLCV path)
    "month_revenue": ("month_revenue.parquet", fetch_month_revenue),
    "margin_short": ("margin_short.parquet", fetch_margin_short),
    "shareholding": ("shareholding.parquet", fetch_shareholding),
    "dividends": ("dividends.parquet", fetch_dividends),
}


def _fetch_channel(name: str, pool: list[str], start: str, end: str | None, token: str,
                   out_dir: Path, sleep: float, resume: bool) -> Path:
    fname, fn = _CHANNELS[name]
    dest = out_dir / fname
    done: set[str] = set()
    prior: list[pd.DataFrame] = []
    if resume and dest.exists():
        old = pd.read_parquet(dest)
        done = set(old["stock_id"].astype(str).unique())
        prior = [old]
        log.info("%s: resume — %d ids already present", name, len(done))
    frames = list(prior)
    todo = [s for s in pool if s not in done]
    for i, sid in enumerate(todo, 1):
        try:
            df = fn(sid, start, end, token)
        except RuntimeError as e:
            if "402" in str(e):
                raise
            if _is_tier_block(e):
                raise RuntimeError(
                    f"{name}: FinMind TIER/PERMISSION block ({str(e)[:200]}). This dataset needs a "
                    f"higher FinMind membership than your key has — upgrade the tier, or drop "
                    f"'{name}' from --datasets.") from e
            log.warning("%s/%s skipped: %s", name, sid, str(e)[:160])
            df = pd.DataFrame()
        if not df.empty:
            frames.append(df)
        if i % 50 == 0:
            log.info("%s: %d/%d fetched", name, i, len(todo))
        time.sleep(sleep)
    if not frames:
        log.warning("%s: no data collected", name)
        return dest
    pd.concat(frames, ignore_index=True).to_parquet(dest, index=False)
    log.info("%s: wrote %s (%d ids)", name, dest, pd.concat(frames, ignore_index=True)["stock_id"].nunique())
    return dest


def _fetch_prices(pool: list[str], start: str, end: str | None, token: str,
                  out_dir: Path, sleep: float, resume: bool) -> Path:
    dest = out_dir / "prices.parquet"
    done: set[str] = set()
    prior: list[pd.DataFrame] = []
    if resume and dest.exists():
        old = pd.read_parquet(dest)
        done = set(old["ticker"].astype(str).unique())
        prior = [old]
        log.info("prices: resume — %d ids already present", len(done))
    frames = list(prior)
    todo = [s for s in pool if s not in done]
    for i, sid in enumerate(todo, 1):
        try:
            raw = _normalize(_finmind_get("TaiwanStockPrice", sid, start, end, token), sid, _STOCK_MAP)
        except RuntimeError as e:
            if "402" in str(e):
                raise
            if _is_tier_block(e):
                raise RuntimeError(
                    f"prices: FinMind TIER/PERMISSION block ({str(e)[:200]}). TaiwanStockPrice needs a "
                    f"higher FinMind membership than your key has — upgrade the tier.") from e
            log.warning("prices/%s skipped: %s", sid, str(e)[:160])
            raw = pd.DataFrame()
        if not raw.empty:
            frames.append(raw)
        if i % 50 == 0:
            log.info("prices: %d/%d fetched", i, len(todo))
        time.sleep(sleep)
    if not frames:
        log.warning("prices: no data collected")
        return dest
    raw_long = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    clean_long, report = _clean_long(raw_long)
    clean_long.to_parquet(dest, index=False)
    (out_dir / "prices.clean_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("prices: wrote %s (%d ids, cleaned)", dest, clean_long["ticker"].nunique())
    return dest


def _selftest() -> int:
    """Offline sanity of the pure causal-lag + tier-parse helpers (no token, no network)."""
    assert month_revenue_avail_date(2020, 4) == pd.Timestamp("2020-05-10")
    assert month_revenue_avail_date(2020, 12) == pd.Timestamp("2021-01-10")
    assert _avail_plus_bdays(pd.Series([pd.Timestamp("2020-01-03")]), 1)[0] == pd.Timestamp("2020-01-06")
    assert parse_holding_lower_bound("400,001-600,000") == 400_001
    assert parse_holding_lower_bound("more than 1,000,001") == 1_000_001
    assert parse_holding_lower_bound("total") is None
    assert parse_holding_lower_bound(12) == 400_001            # integer TDCC code
    g = pd.DataFrame({
        "HoldingSharesLevel": ["1-999", "400,001-600,000", "more than 1,000,001", "total"],
        "percent": [10.0, 25.0, 15.0, 100.0], "unit": [0, 0, 0, 5_000_000]})
    assert abs(big_holder_percent(g) - 40.0) < 1e-9           # 25 + 15 (both > 400 lots)
    assert _holding_total_shares(g) == 5_000_000
    assert _is_tier_block(RuntimeError("FinMind HTTP 400 on X/2330: level is register"))
    assert _is_tier_block(RuntimeError("permission denied for this dataset"))
    assert not _is_tier_block(RuntimeError("FinMind HTTP 500 on X/2330: timeout"))
    assert not _is_tier_block(RuntimeError("data_id not exist"))
    print("selftest-ok")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Fetch Taiwan small/mid-cap alt-data via FinMind")
    ap.add_argument("--datasets", default="prices,month_revenue,margin_short,shareholding,dividends",
                    help="Comma subset of: prices,month_revenue,margin_short,shareholding,dividends")
    ap.add_argument("--pool", default="auto",
                    help="'auto' = enumerate TWSE+TPEx common stocks; or a comma id list; "
                         "or a path to a one-id-per-line / parquet(stock_id) file.")
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default="data/taiwan_smallcap")
    ap.add_argument("--token", default=os.environ.get("FINMIND_TOKEN", ""))
    ap.add_argument("--sleep", type=float, default=0.35, help="Seconds between FinMind calls")
    ap.add_argument("--threshold-lots", type=int, default=400, help="Big-holder tier boundary (lots)")
    ap.add_argument("--no-resume", action="store_true", help="Refetch from scratch (ignore cache)")
    ap.add_argument("--selftest", action="store_true", help="Offline helper sanity — no token/network")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if not args.token:
        ap.error("FINMIND_TOKEN not set and --token not given (the alt-data channels need Sponsor).")

    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    resume = not args.no_resume

    # Resolve the pool.
    if args.pool == "auto":
        pool_df = enumerate_common_stock_pool(args.token)
        pool_df.to_parquet(out_dir / "pool.parquet", index=False)
        pool = pool_df["stock_id"].tolist()
        log.info("pool: %d TWSE+TPEx common stocks → %s", len(pool), out_dir / "pool.parquet")
    elif Path(args.pool).exists():
        p = Path(args.pool)
        ids = (pd.read_parquet(p)["stock_id"].astype(str).tolist() if p.suffix == ".parquet"
               else [ln.strip() for ln in p.read_text().splitlines() if ln.strip()])
        pool = list(dict.fromkeys(ids))
        log.info("pool: %d ids from %s", len(pool), p)
    else:
        pool = [s.strip() for s in args.pool.split(",") if s.strip()]
        log.info("pool: %d ids from --pool literal", len(pool))

    channels = [c.strip() for c in args.datasets.split(",") if c.strip()]
    unknown = set(channels) - set(_CHANNELS)
    if unknown:
        ap.error(f"unknown --datasets {sorted(unknown)}; allowed {sorted(_CHANNELS)}")

    for ch in channels:
        if ch == "prices":
            _fetch_prices(pool, args.start, args.end, args.token, out_dir, args.sleep, resume)
        elif ch == "shareholding":
            # threshold flows via a closure so the orchestrator signature stays uniform
            def _fn(sid: str, s: str, e: str | None, tk: str, _t=args.threshold_lots):
                return fetch_shareholding(sid, s, e, tk, threshold_lots=_t)
            _CHANNELS["shareholding"] = ("shareholding.parquet", _fn)
            _fetch_channel(ch, pool, args.start, args.end, args.token, out_dir, args.sleep, resume)
        else:
            _fetch_channel(ch, pool, args.start, args.end, args.token, out_dir, args.sleep, resume)

    log.info("done → %s", out_dir)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    raise SystemExit(main())
