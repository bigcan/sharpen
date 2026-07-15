"""Taiwan small/mid-cap alt-data probes through the cross-sectional IC funnel (probe step 3).

Pre-registration: ``docs/research/taiwan_smallcap_altdata_probes_preregistration_2026-07-15.md``.
Reads the step-1 tidy parquets + the step-2 ``membership.parquet``, builds a
:class:`finrl_pro_ds.signals.Panel` over the cap-rank 51-250 small/mid band with the THREE
alt-data channels injected as causally as-of-aligned ``feature_slots`` (CR-9), and runs the three
pre-registered signals through the existing deflated 6-tier funnel (``evaluate_batch``) under the
parallel-pathway gates (``configs/taiwan_smallcap_altdata.gates.yaml``).

  P1 ``tw_smallcap_mom_rev``      sign +1  month-revenue YoY growth (post-announcement drift)
  P2 ``tw_smallcap_margin_crowd`` sign -1  Δ21d retail margin utilization (leverage crowding → reversal)
  P3 ``tw_smallcap_holder_conc``  sign +1  Δ4w big-holder (>400-lot) 集保 concentration (accumulation)

CAUSALITY (LEAK-2): every channel enters the panel through :func:`_asof_grid`, which stamps
``value[t,n]`` = the last event whose ``avail_date <= panel.dates[t]`` — the public-availability lag
computed once in ``fetch_taiwan_fundamentals_finmind``. ``Panel.truncated`` slices the slots, so the
Tier-0 truncation tripwire (``compute(truncated)[t] == compute(full)[t]``) holds by construction.
Size-neutralization is MANDATORY and lives in each ``SignalSpec.neutralization`` (read by
``evaluate_signal``) — it cannot be silently dropped.

Usage:
    python scripts/research/taiwan_smallcap_altdata_eval.py \
        --data data/taiwan_smallcap --gates configs/taiwan_smallcap_altdata.gates.yaml \
        --out results/taiwan_smallcap_altdata
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finrl_pro_ds.signals import Gates, Panel, evaluate_batch, to_markdown, write_scorecard  # noqa: E402
from finrl_pro_ds.signals.spec import SignalSpec  # noqa: E402

log = logging.getLogger("taiwan_smallcap_altdata")

# Pre-registered spec constants — MUST reproduce the frozen content-hashes (asserted below and in
# tests). Any edit here that changes a hash is a pre-registration violation.
_NEU = ("winsor", "zscore", "sector", "size")   # "size" MANDATORY (killed the June mirage)
_HZ = (1, 5, 10, 21, 63)
_UNI = "twse_smallcap_caprank_51_250"
_CP = "taiwan_standard"
PREREG_HASHES = {
    "tw_smallcap_mom_rev": "60680e61ff85",
    "tw_smallcap_margin_crowd": "e0a4c719bfe0",
    "tw_smallcap_holder_conc": "1be26f02ee6a",
}
MARGIN_LOOKBACK = 21   # Δ trading days for margin utilization (≈ 1 month)
HOLDER_LOOKBACK = 20   # Δ trading days for 集保 concentration (≈ 4 weeks)


# --------------------------------------------------------------------------- #
# Pre-registered signals — read their channel from panel.feature_slots (CR-9)
# --------------------------------------------------------------------------- #
class _SlotLevel:
    """P1: return the causally-aligned channel level directly (the 'momentum' IS the YoY level)."""

    def __init__(self, name: str, hypothesis: str, sign: int, slot: str) -> None:
        self.slot = slot
        self.spec = SignalSpec(name=name, hypothesis=hypothesis, family="altdata",
                               expected_sign=sign, horizons=_HZ, neutralization=_NEU,
                               universe=_UNI, cost_profile=_CP)

    def compute(self, panel: Panel) -> np.ndarray:
        v = np.asarray(panel.feature_slots[self.slot], dtype=np.float64)
        return v if v.ndim == 2 else np.tile(v[:, None], (1, panel.N))


class _SlotDelta:
    """P2/P3: causal Δ over ``lookback`` trading days of an aligned channel level (row t uses <= t)."""

    def __init__(self, name: str, hypothesis: str, sign: int, slot: str, lookback: int) -> None:
        self.slot, self.lookback = slot, lookback
        self.spec = SignalSpec(name=name, hypothesis=hypothesis, family="altdata",
                               expected_sign=sign, horizons=_HZ, neutralization=_NEU,
                               universe=_UNI, cost_profile=_CP)

    def compute(self, panel: Panel) -> np.ndarray:
        v = np.asarray(panel.feature_slots[self.slot], dtype=np.float64)
        out = np.full(v.shape, np.nan, dtype=np.float64)
        lb = self.lookback
        if lb < v.shape[0]:
            out[lb:] = v[lb:] - v[:-lb]          # causal: t vs t-lb, both <= t
        return out


def build_signals() -> dict[str, object]:
    p1 = _SlotLevel("tw_smallcap_mom_rev",
                    "Monthly-revenue YoY growth predicts cross-sectional continuation in TWSE/TPEx "
                    "small-mid caps (post-announcement drift under thin analyst coverage)",
                    1, "mrev_yoy")
    p2 = _SlotDelta("tw_smallcap_margin_crowd",
                    "Rising retail margin-financing balance (leverage crowding, normalized by shares) "
                    "predicts cross-sectional reversal in small-mid caps",
                    -1, "margin_util", MARGIN_LOOKBACK)
    p3 = _SlotDelta("tw_smallcap_holder_conc",
                    "Rising big-holder shareholding concentration (>400-lot tier share, 集保) predicts "
                    "cross-sectional continuation in small-mid caps (informed accumulation)",
                    1, "holder_conc", HOLDER_LOOKBACK)
    sigs = {s.spec.name: s for s in (p1, p2, p3)}
    for name, s in sigs.items():                 # anti-p-hacking seal
        got = s.spec.content_hash()
        if got != PREREG_HASHES[name]:
            raise SystemExit(f"SPEC DRIFT: {name} hash {got} != pre-registered {PREREG_HASHES[name]} "
                             "— a pre-registration violation; revert the spec edit.")
    return sigs


# --------------------------------------------------------------------------- #
# Causal as-of alignment of an event stream to the trading-day × ticker grid
# --------------------------------------------------------------------------- #
def _asof_grid(events: pd.DataFrame, dates: np.ndarray, tickers: tuple[str, ...],
               value_col: str, avail_col: str = "avail_date", id_col: str = "stock_id") -> np.ndarray:
    """``(T,N)`` where ``grid[t,n]`` = last ``value_col`` for ticker n with ``avail_date <= dates[t]``.

    Per ticker: ``merge_asof`` the trading dates onto the event stream sorted by availability
    (backward direction), so no value appears before it is public. NaN before the first event.
    """
    T, N = len(dates), len(tickers)
    grid = np.full((T, N), np.nan, dtype=np.float64)
    if events is None or events.empty or value_col not in events.columns:
        return grid
    d = pd.DataFrame({avail_col: pd.to_datetime(dates)}).sort_values(avail_col).reset_index(drop=True)
    ev = events.dropna(subset=[avail_col, value_col]).copy()
    ev[avail_col] = pd.to_datetime(ev[avail_col])
    ev[id_col] = ev[id_col].astype(str)
    col = {tk: j for j, tk in enumerate(tickers)}
    for tk, g in ev.groupby(id_col):
        j = col.get(str(tk))
        if j is None:
            continue
        g = g[[avail_col, value_col]].sort_values(avail_col)
        # collapse duplicate avail dates to the LAST print that day (keep it public-consistent)
        g = g.groupby(avail_col, as_index=False).last()
        merged = pd.merge_asof(d, g, on=avail_col, direction="backward")
        grid[:, j] = merged[value_col].to_numpy(dtype=np.float64)
    return grid


def _month_revenue_yoy(mrev: pd.DataFrame) -> pd.DataFrame:
    """``[stock_id, avail_date, yoy]`` — YoY revenue growth, availability from the 10th-of-next-month."""
    if mrev.empty:
        return pd.DataFrame(columns=["stock_id", "avail_date", "yoy"])
    m = mrev.copy()
    m["stock_id"] = m["stock_id"].astype(str)
    m = m.sort_values(["stock_id", "revenue_year", "revenue_month"])
    prev = m.groupby("stock_id")["revenue"].shift(12)               # same month, prior year
    yoy = np.where((prev > 0) & prev.notna(), m["revenue"] / prev - 1.0, np.nan)
    return pd.DataFrame({"stock_id": m["stock_id"], "avail_date": pd.to_datetime(m["avail_date"]),
                         "yoy": yoy}).dropna(subset=["yoy"])


def _margin_util(margin: pd.DataFrame, shareholding: pd.DataFrame) -> pd.DataFrame:
    """``[stock_id, avail_date, margin_util]`` = margin balance / causal total shares (utilization)."""
    if margin.empty:
        return pd.DataFrame(columns=["stock_id", "avail_date", "margin_util"])
    mg = margin.copy()
    mg["stock_id"] = mg["stock_id"].astype(str)
    mg["date"] = pd.to_datetime(mg["date"])
    if shareholding.empty or "total_shares" not in shareholding.columns:
        return pd.DataFrame(columns=["stock_id", "avail_date", "margin_util"])
    sh = (shareholding.dropna(subset=["total_shares"]).copy())
    sh["stock_id"] = sh["stock_id"].astype(str)
    sh["avail_date"] = pd.to_datetime(sh["avail_date"])
    frames = []
    for tk, g in mg.groupby("stock_id"):
        s = sh[sh["stock_id"] == tk][["avail_date", "total_shares"]].sort_values("avail_date")
        if s.empty:
            continue
        g = g.sort_values("date")
        # shares known as of the margin row's own trading date (causal): avail_date(shares) <= date
        merged = pd.merge_asof(g[["date", "margin_balance"]].rename(columns={"date": "avail_date"}),
                               s, on="avail_date", direction="backward")
        util = np.where(merged["total_shares"] > 0, merged["margin_balance"] / merged["total_shares"], np.nan)
        frames.append(pd.DataFrame({"stock_id": tk,
                                    "avail_date": pd.to_datetime(g["date"].to_numpy())
                                    + pd.tseries.offsets.BDay(1),      # margin known T+1
                                    "margin_util": util}))
    if not frames:
        return pd.DataFrame(columns=["stock_id", "avail_date", "margin_util"])
    return pd.concat(frames, ignore_index=True).dropna(subset=["margin_util"])


def _causal_total_return_factor(close_w: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
    """Cumulative forward div-add-back factor ``cf`` (T×N), applied to ALL FOUR OHLC fields.

    Per ticker ``m[ex] = 1 + amount/close_raw[ex]`` on each ex-date, ``cf = m.cumprod()`` — starts at
    1 and only ratchets UP at an ex-date, so bars BEFORE a dividend are byte-for-byte unchanged
    (forward, not back-adjustment → LEAK-2 clean, matching ``taiwan_panel_loader``). All four OHLC
    fields must share this one factor so intraday ratios (and OHLC sanity) are preserved.
    """
    idx = close_w.index
    factors = pd.DataFrame(1.0, index=idx, columns=close_w.columns)
    if dividends is None or dividends.empty:
        return factors
    dv = dividends.copy()
    dv["stock_id"] = dv["stock_id"].astype(str)
    dv["ex_date"] = pd.to_datetime(dv["ex_date"])
    for tk, g in dv.groupby("stock_id"):
        if tk not in close_w.columns:
            continue
        col = close_w[tk]
        j = close_w.columns.get_loc(tk)
        for ex_date, amount in zip(g["ex_date"], g["amount"]):
            pos = int(idx.searchsorted(pd.Timestamp(ex_date), side="left"))
            if pos >= len(idx):
                continue
            p = col.iloc[pos]
            if not np.isfinite(p) or p <= 0:
                continue
            factors.iloc[pos, j] *= (1.0 + float(amount) / float(p))
    return factors.cumprod(axis=0)


# --------------------------------------------------------------------------- #
# Panel construction
# --------------------------------------------------------------------------- #
def _daily_membership(members: pd.DataFrame, dates: np.ndarray,
                      tickers: tuple[str, ...]) -> np.ndarray:
    """Expand monthly constituents to a causal daily ``(T,N)`` mask (member since the last rebalance)."""
    T, N = len(dates), len(tickers)
    mask = np.zeros((T, N), dtype=bool)
    if members.empty:
        return mask
    m = members.copy()
    m["rebalance_date"] = pd.to_datetime(m["rebalance_date"])
    m["stock_id"] = m["stock_id"].astype(str)
    wide = (m.assign(v=True).pivot_table(index="rebalance_date", columns="stock_id", values="v",
                                         aggfunc="first", fill_value=False)
            .reindex(columns=list(tickers), fill_value=False).sort_index())
    rebal = wide.index.to_numpy(dtype="datetime64[ns]")
    M = wide.to_numpy(dtype=bool)
    idx = np.searchsorted(rebal, dates, side="right") - 1        # last rebalance <= date
    valid = idx >= 0
    mask[valid] = M[idx[valid]]
    return mask


def build_panel(data: Path, *, adv_window: int = 20) -> Panel:
    prices = pd.read_parquet(data / "prices.parquet")
    members = pd.read_parquet(data / "universe" / "membership.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    prices["ticker"] = prices["ticker"].astype(str)
    keep = set(members["stock_id"].astype(str)) if not members.empty else set(prices["ticker"])
    prices = prices[prices["ticker"].isin(keep)].drop_duplicates(["date", "ticker"], keep="first")

    tickers = tuple(sorted(prices["ticker"].unique()))
    dates = np.array(sorted(prices["date"].unique()), dtype="datetime64[ns]")

    def wide(col: str) -> pd.DataFrame:
        return prices.pivot(index="date", columns="ticker", values=col).reindex(
            index=pd.DatetimeIndex(dates), columns=list(tickers))

    open_w, high_w, low_w, close_w, vol_w = (wide(c) for c in ("open", "high", "low", "close", "volume"))
    div_path = data / "dividends.parquet"
    dividend_adjusted = div_path.exists()
    if dividend_adjusted:                                        # causal total-return (all OHLC × cf)
        cf = _causal_total_return_factor(close_w, pd.read_parquet(div_path))
        open_w, high_w, low_w, close_w = (f * cf for f in (open_w, high_w, low_w, close_w))

    o, h, lo, c, v = (df.to_numpy(dtype=np.float64) for df in (open_w, high_w, low_w, close_w, vol_w))
    tradeable = (np.isfinite(o) & np.isfinite(h) & np.isfinite(lo) & np.isfinite(c) & (c > 0)
                 & np.isfinite(v) & (v > 0))
    dollar = np.where(tradeable, c * v, np.nan)
    adv = pd.DataFrame(dollar).rolling(adv_window, min_periods=10).mean().to_numpy()
    membership = _daily_membership(members, dates, tickers)
    active = membership & tradeable & np.isfinite(adv)
    for arr in (o, h, lo, c, v):
        arr[~tradeable] = np.nan

    # sectors from the enumerated pool (FinMind industry_category); unknown → own bucket
    pool_path = data / "pool.parquet"
    if pool_path.exists():
        pool = pd.read_parquet(pool_path)
        cat = dict(zip(pool["stock_id"].astype(str), pool["sector"].astype(str)))
        codes = {s: i for i, s in enumerate(sorted({cat.get(t, "__unk__") for t in tickers}))}
        sector_id = np.array([codes[cat.get(t, "__unk__")] for t in tickers], dtype=np.int64)
    else:
        sector_id = np.zeros(len(tickers), dtype=np.int64)

    # --- the three causally-aligned alt-data channels → feature_slots ---
    def _pq(name: str) -> pd.DataFrame:
        p = data / name
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()

    mrev, margin, shareholding = _pq("month_revenue.parquet"), _pq("margin_short.parquet"), _pq("shareholding.parquet")
    slot_mrev = _asof_grid(_month_revenue_yoy(mrev), dates, tickers, "yoy")
    slot_margin = _asof_grid(_margin_util(margin, shareholding), dates, tickers, "margin_util")
    slot_holder = _asof_grid(shareholding.rename(columns={}), dates, tickers, "big_holder_pct") \
        if not shareholding.empty else np.full((len(dates), len(tickers)), np.nan)

    liq_days = int((active.sum(axis=1) >= 25).sum())
    meta = {
        "survivorship_free": False,
        "source": "finmind_http+taiwan_smallcap_altdata",
        "universe_def": _UNI,
        "adjusted": dividend_adjusted,
        "return_basis": "causal_total_return_forward" if dividend_adjusted else "raw_close_unadjusted",
        "n_names_pool": len(tickers),
        "liquid_days_ge25": liq_days,
        "note": ("cap-rank 51-250 monthly PIT membership; RAW unadjusted prices; current-listing pool "
                 "⇒ survivorship UPPER BOUND"),
    }
    return Panel(dates, tickers, o, h, lo, c, v, active, adv, sector_id, meta,
                 feature_slots={"mrev_yoy": slot_mrev, "margin_util": slot_margin,
                                "holder_conc": slot_holder})


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Taiwan small/mid-cap alt-data probes through the funnel")
    ap.add_argument("--data", default="data/taiwan_smallcap")
    ap.add_argument("--gates", default="configs/taiwan_smallcap_altdata.gates.yaml")
    ap.add_argument("--out", default="results/taiwan_smallcap_altdata")
    ap.add_argument("--adv-window", type=int, default=20)
    args = ap.parse_args()

    data = (ROOT / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    gates_path = (ROOT / args.gates) if not Path(args.gates).is_absolute() else Path(args.gates)
    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)

    panel = build_panel(data, adv_window=args.adv_window)
    log.info("Panel: pool N=%d, T=%d (%s..%s) | liquid days(>=25)=%d | %s | return=%s",
             panel.N, panel.T, str(panel.dates[0])[:10], str(panel.dates[-1])[:10],
             panel.meta["liquid_days_ge25"], panel.meta["universe_def"], panel.meta["return_basis"])
    for k, s in panel.feature_slots.items():
        log.info("  slot %-12s coverage: %.1f%% of active cells",
                 k, 100.0 * np.isfinite(s)[panel.active].mean() if panel.active.any() else 0.0)

    gates = Gates.from_yaml(gates_path)
    signals = build_signals()
    rs = evaluate_batch(signals, panel, gates, "taiwan_smallcap_altdata")
    jp, mp = write_scorecard(rs, out_dir)
    print("\n" + to_markdown(rs) + "\n")
    log.info("Scorecard: %s | %s", mp, jp)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    raise SystemExit(main())
