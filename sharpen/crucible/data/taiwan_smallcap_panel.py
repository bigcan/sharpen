"""Taiwan small/mid-cap daily panel — cap-rank 51-250, PIT monthly membership (crucible-v13.1).

The substrate the 2026-07-15/07-31 probe campaigns were run on, promoted from a script-local builder
into the library so the Crucible orchestrator mines the SAME panel the probes scored. Before this
module the builder lived in ``scripts/research/taiwan_smallcap_altdata_eval.py`` and the other three
probe scripts imported it by ``sys.path`` injection; the orchestrator could not reach it at all, so
``taiwan_smallcap`` was the one substrate in this project with a validated evaluation path and no
mining path.

BREADTH, MEASURED (``crucible_real_alpha_breadth.py --panel taiwan_smallcap``, 100 published WQ101
alphas on the real panel, ~146 names/day). At H=1 the DSL realizes **n_eff median 38.1**, IQR
[20.3, 71.3], 57 of 100 alphas usable — HIGHER than the 21.5 that justified ``us_equity``, and far
above the 18-ETF cross-asset panel's 6.8 or the TAIEX ETF panel's 5.5. At H=21, **not one alpha
clears the |IC| floor**: the cross-sectional signal is gone by monthly hold.

That is the substrate's real constraint, and wiring does not fix it — it makes it measurable. The
0.30% sell-side transaction tax forces a long hold here (hence ``hold_horizon: 21`` and
``cost_bps: 0.0021`` in the gates), and the breadth that would power a mine only exists at the short
hold the tax forbids. Same disjoint-window shape as ``us_equity`` (S553-cont-157), reached from the
opposite side. The substrate-power guard independently refuses this panel — implied MDE ΔSR 1.554
against a 0.50 ceiling — so a mine here needs ``--force-underpowered``, which is an operator call.

SURVIVORSHIP (the honest downgrade). ``meta["survivorship_free"] = False``. The pool is enumerated
from currently-listed FinMind names, so delisted small caps are absent and every result on this
panel is an UPPER BOUND. This is worse here than on ``us_equity`` (which at least carries a PIT
membership history with its unpriceable residual recorded): the small/mid-cap band is where delisting
is most common. It is stated in ``meta`` so the scorecard prints it, and it is not fixable on free
data.

⚠ TWO KNOWN LOOK-BACKS THIS PANEL CARRIES, both inherited from the sealed probe campaign and both
deliberately NOT "fixed" here — changing either would restate results already recorded (P1-REPRO-01
is exactly that failure), so they are documented instead:

  * **`sector_id` is a CURRENT classification applied retroactively.** FinMind's
    ``industry_category`` is a snapshot, not point-in-time, so a name reclassified in 2024 carries
    that label back to 2005. It is a MANDATORY neutralization control for the probes (it is what
    killed the June large-cap mirage), which is why it stays. But note what wiring this substrate
    does to its blast radius: every generated DSL candidate defaults to a neutralization tuple
    containing ``"sector"`` (``generation/dsl_signal.py``) and ``IndClass.sector`` is an addressable
    grammar terminal, so a MINE here inherits the look-back on every candidate. ``us_equity_panel``
    took the opposite branch — it zeroes ``sector_id`` and forfeits sector neutralization outright,
    citing this same risk. Results here are upper bounds on this ground as well as survivorship.
  * **`adv_usd` includes bar t's own dollar volume** (no ``.shift(1)``), where ``us_equity_panel``
    explicitly shifts it out. This is NOT a LEAK-2 violation — every input is a bar ≤ t, and the
    funnel's convention is that bar t's OHLCV is known at t's close — but the two substrates differ
    in convention, and ``adv_usd`` feeds both the ``active`` universe mask and ``size``
    neutralization. The DSL's own ``adv{N}`` terminal is unaffected: it recomputes from
    ``close × volume`` inside ``_alpha_dsl``, identically on every panel.

CAUSALITY (LEAK-2), the three places it is load-bearing:
  * PRICES are causal TOTAL return — each cash distribution is added back on its ex-date and
    FORWARD-accumulated (:func:`causal_total_return_factor`), the opposite of a back-adjusted feed,
    which would rewrite past bars with future split/dividend factors.
  * MEMBERSHIP expands monthly rebalances with ``searchsorted(..., "right") - 1`` — the last
    rebalance AT OR BEFORE t, never the next one.
  * ALT-DATA channels enter through :func:`asof_grid`, which stamps ``value[t,n]`` = the last event
    whose ``avail_date <= dates[t]``, where ``avail_date`` carries the publication lag computed once
    at fetch time (TWSE balances and T86 flows: T+1, they publish after the close). Month revenue is
    stored with its statutory deadline (the 10th of the following month) and, since crucible-v14.0,
    becomes usable only on the session AFTER the deadline session (:func:`next_session_after`): a
    deadline-day filer may post after the close, and bar t trades at t's close. Before v14.0 it was
    usable on the deadline session itself, a one-session look-ahead for after-close filers; P1
    (month-revenue drift) was re-scored on the fixed panel.

Run the tripwires whenever this builder is touched:

    python -m sharpen.crucible.data.taiwan_smallcap_panel
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sharpen.signals.features import Panel

log = logging.getLogger("crucible.taiwan_smallcap_panel")

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / "data" / "taiwan_smallcap"

UNIVERSE_DEF = "twse_smallcap_caprank_51_250"
ADV_WINDOW = 20
ADV_MIN_PERIODS = 10
MARGIN_LOOKBACK = 21     # Δ trading days for margin utilization (~1 month), pre-registered
HOLDER_LOOKBACK = 20     # Δ trading days for 集保 concentration (~4 weeks), pre-registered
FLOW_WINDOW = 21         # rolling net-flow sum window, pre-registered

#: The three channels the 2026-07-15 alt-data campaign locked. This is the DEFAULT so the probe
#: scripts that delegate here keep producing byte-identical scorecards.
CORE_CHANNELS: tuple[str, ...] = ("mrev_yoy", "margin_util", "holder_conc")
#: Channels added by the 2026-07-31 campaigns (Q1/Q2 institutional flow, S1 short interest). Each
#: was built and causality-checked in its own probe script; they are collected here so the mining
#: substrate carries every validated channel rather than the first campaign's three.
FLOW_CHANNELS: tuple[str, ...] = ("foreign_flow", "trust_flow")
SHORT_CHANNELS: tuple[str, ...] = ("short_util",)
ALL_CHANNELS: tuple[str, ...] = CORE_CHANNELS + FLOW_CHANNELS + SHORT_CHANNELS

_FROZEN_POOL = "pool.frozen.parquet"


# --------------------------------------------------------------------------- #
# Causal as-of alignment of an event stream to the trading-day × ticker grid
# --------------------------------------------------------------------------- #
def asof_grid(events: pd.DataFrame, dates: np.ndarray, tickers: tuple[str, ...],
              value_col: str, avail_col: str = "avail_date",
              id_col: str = "stock_id") -> np.ndarray:
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


def next_session_after(avail: pd.Series, dates: np.ndarray) -> pd.Series:
    """Map each deadline date to the first trading session STRICTLY AFTER the deadline's own session.

    A filing is due BY the deadline, and a filer may post after the 13:30 close on that day. The
    strategy trades at the close of bar t, so a value first usable on the deadline session could be
    traded before it was public. The deadline session is the first trading date >= the deadline
    (covers weekend and holiday rolls on the panel's own calendar); the value becomes usable on the
    session after that. Deadlines past the panel's last usable session map to NaT (never visible)."""
    d = np.asarray(pd.to_datetime(dates).values, dtype="datetime64[ns]")
    a = np.asarray(pd.to_datetime(avail).values, dtype="datetime64[ns]")
    idx = np.searchsorted(d, a, side="left") + 1
    out = np.full(a.shape, np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = idx < len(d)
    out[ok] = d[idx[ok]]
    return pd.Series(out, index=avail.index)


def month_revenue_yoy(mrev: pd.DataFrame, dates: np.ndarray | None = None) -> pd.DataFrame:
    """``[stock_id, avail_date, yoy]`` — YoY revenue growth.

    YoY compares the SAME CALENDAR month a year earlier (a missing month yields NaN, never a
    13-month comparison). Stored ``avail_date`` is the statutory deadline (10th of the next month);
    when the trading ``dates`` are given it is moved to the session after the deadline session via
    :func:`next_session_after` (v14.0 LEAK-2 fix). Without ``dates`` the raw deadline is returned —
    only for unit tests of the YoY arithmetic, never for building a tradeable panel."""
    if mrev.empty:
        return pd.DataFrame(columns=["stock_id", "avail_date", "yoy"])
    m = mrev.copy()
    m["stock_id"] = m["stock_id"].astype(str)
    m["_k"] = m["revenue_year"].astype(int) * 12 + m["revenue_month"].astype(int)
    m = m.sort_values(["stock_id", "_k"]).drop_duplicates(["stock_id", "_k"], keep="last")
    base = m[["stock_id", "_k", "revenue"]].rename(columns={"revenue": "_prev"})
    base["_k"] = base["_k"] + 12                                       # same month, prior year
    m = m.merge(base, on=["stock_id", "_k"], how="left")
    prev = m["_prev"]
    yoy = np.where((prev > 0) & prev.notna(), m["revenue"] / prev - 1.0, np.nan)
    avail = pd.to_datetime(m["avail_date"])
    if dates is not None:
        avail = next_session_after(avail, dates)
    return pd.DataFrame({"stock_id": m["stock_id"], "avail_date": avail,
                         "yoy": yoy}).dropna(subset=["yoy", "avail_date"])


def balance_util(margin: pd.DataFrame, shareholding: pd.DataFrame, *,
                 balance_col: str, out_col: str) -> pd.DataFrame:
    """``[stock_id, avail_date, out_col]`` = a TWSE balance / causal total shares (utilization).

    ONE construction for both legs of ``margin_short.parquet`` — ``margin_balance`` (P2 leverage
    crowding) and ``short_balance`` (S1 short interest) — because they differ only in which column
    is divided. They were written twice, in two probe scripts, and a divergence between them would
    have been invisible: both produce well-formed numbers either way.

    Causality: shares are merged as-of BACKWARD on the balance row's own trading date, so only an
    already-public share count is used, and the result is stamped ``+1 BDay`` because TWSE publishes
    balances after the close.
    """
    empty = pd.DataFrame(columns=["stock_id", "avail_date", out_col])
    if margin.empty or shareholding.empty or "total_shares" not in shareholding.columns:
        return empty
    mg = margin.copy()
    mg["stock_id"] = mg["stock_id"].astype(str)
    mg["date"] = pd.to_datetime(mg["date"])
    sh = shareholding.dropna(subset=["total_shares"]).copy()
    sh["stock_id"] = sh["stock_id"].astype(str)
    sh["avail_date"] = pd.to_datetime(sh["avail_date"])
    frames = []
    for tk, g in mg.groupby("stock_id"):
        s = sh[sh["stock_id"] == tk][["avail_date", "total_shares"]].sort_values("avail_date")
        if s.empty:
            continue
        g = g.sort_values("date")
        merged = pd.merge_asof(
            g[["date", balance_col]].rename(columns={"date": "avail_date"}),
            s, on="avail_date", direction="backward")
        util = np.where(merged["total_shares"] > 0, merged[balance_col] / merged["total_shares"],
                        np.nan)
        frames.append(pd.DataFrame({
            "stock_id": tk,
            "avail_date": pd.to_datetime(g["date"].to_numpy()) + pd.tseries.offsets.BDay(1),
            out_col: util}))
    return (pd.concat(frames, ignore_index=True).dropna(subset=[out_col]) if frames else empty)


def flow_events(inst: pd.DataFrame, col: str, out_col: str) -> pd.DataFrame:
    """Rolling ``FLOW_WINDOW``-day SUM of a T86 net-flow column, per stock, carrying its avail_date.

    The sum ends at the row's own trading date and the row is already stamped
    ``avail_date = date + 1 business day`` by the fetcher (T86 publishes after the close), so the
    value can only enter the panel on a session strictly after every bar it uses (LEAK-2).
    """
    if inst.empty or col not in inst.columns:
        return pd.DataFrame(columns=["stock_id", "avail_date", out_col])
    d = inst[["stock_id", "date", "avail_date", col]].copy()
    d["stock_id"] = d["stock_id"].astype(str)
    d["date"] = pd.to_datetime(d["date"])
    d["avail_date"] = pd.to_datetime(d["avail_date"])
    d = d.sort_values(["stock_id", "date"])
    d[out_col] = (d.groupby("stock_id")[col]
                  .rolling(FLOW_WINDOW, min_periods=FLOW_WINDOW).sum().reset_index(level=0, drop=True))
    return d.dropna(subset=[out_col])[["stock_id", "avail_date", out_col]]


def causal_total_return_factor(close_w: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
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
# Universe + sector partition
# --------------------------------------------------------------------------- #
def daily_membership(members: pd.DataFrame, dates: np.ndarray,
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


def sector_map(data: Path, tickers: tuple[str, ...]) -> tuple[np.ndarray, dict]:
    """Sector partition for ``tickers`` + its provenance stamp.

    The pool's ``sector`` column (FinMind ``industry_category``) is a MANDATORY neutralization
    control — ``neutralize`` residualizes every score on sector dummies daily — so the partition is
    load-bearing on every downstream number. It is also NOT point-in-time: FinMind re-classifies
    names over time, so re-enumerating the pool silently restates results recorded earlier. That is
    exactly what happened on 2026-07-31 11:41, when a fetcher run for an unrelated probe rewrote
    ``pool.parquet`` and moved the sealed 2026-07-16 P1 frictionless Sharpe 1.0084 -> 1.0447 with
    ``spec_hash``, ``liquid_days_ge25`` and ``n_names_pool`` all unchanged (they are blind to it).

    Two defences, because freezing alone can be undone by hand:
      * read ``pool.frozen.parquet`` in preference to ``pool.parquet`` — the fetcher only ever writes
        the latter, so the evaluation input stops moving under unrelated data pulls;
      * stamp ``sector_map_sha`` into ``panel_meta``. The digest covers only the ``(ticker, sector)``
        pairs FOR THE PANEL'S OWN NAMES, sorted — so a pool that merely gains unrelated listings does
        NOT trip it, while any genuine change to this panel's partition does.
    """
    frozen, live = data / _FROZEN_POOL, data / "pool.parquet"
    path = frozen if frozen.exists() else live
    if not path.exists():
        log.warning("no pool file at %s — sector neutralization COLLAPSES to a single bucket", data)
        return np.zeros(len(tickers), dtype=np.int64), {
            "source": None, "frozen": False, "sector_map_sha": None,
            "n_sectors": 1, "n_unknown": len(tickers),
        }

    pool = pd.read_parquet(path)
    cat = dict(zip(pool["stock_id"].astype(str), pool["sector"].astype(str)))
    labels = [cat.get(t, "__unk__") for t in tickers]
    codes = {s: i for i, s in enumerate(sorted(set(labels)))}
    sector_id = np.array([codes[s] for s in labels], dtype=np.int64)

    payload = "\n".join(f"{t}\t{s}" for t, s in sorted(zip(tickers, labels)))
    prov = {
        "source": path.name,
        "frozen": path == frozen,
        "sector_map_sha": hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12],
        "n_sectors": len(codes),
        "n_unknown": sum(1 for s in labels if s == "__unk__"),
    }
    if not prov["frozen"]:
        log.warning("sector map read from UNFROZEN %s — results are not reproducible across fetcher "
                    "runs; freeze it to %s to pin them", live.name, _FROZEN_POOL)
    log.info("sector map: %s (frozen=%s) sha=%s | %d sectors, %d unknown",
             prov["source"], prov["frozen"], prov["sector_map_sha"], prov["n_sectors"],
             prov["n_unknown"])
    return sector_id, prov


# --------------------------------------------------------------------------- #
# Panel construction
# --------------------------------------------------------------------------- #
def _read(data: Path, name: str) -> pd.DataFrame:
    p = data / name
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def _build_channels(data: Path, dates: np.ndarray, tickers: tuple[str, ...],
                    channels: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Assemble the requested alt-data feature slots, reading only the parquets they need."""
    want = set(channels)
    unknown = want - set(ALL_CHANNELS)
    if unknown:
        raise ValueError(f"unknown taiwan_smallcap channel(s) {sorted(unknown)}; "
                         f"available: {list(ALL_CHANNELS)}")
    slots: dict[str, np.ndarray] = {}
    empty = np.full((len(dates), len(tickers)), np.nan, dtype=np.float64)

    shareholding = _read(data, "shareholding.parquet") if want & {
        "margin_util", "holder_conc", "short_util"} else pd.DataFrame()
    margin = _read(data, "margin_short.parquet") if want & {"margin_util", "short_util"} \
        else pd.DataFrame()

    if "mrev_yoy" in want:
        slots["mrev_yoy"] = asof_grid(month_revenue_yoy(_read(data, "month_revenue.parquet"), dates),
                                      dates, tickers, "yoy")
    if "margin_util" in want:
        slots["margin_util"] = asof_grid(
            balance_util(margin, shareholding, balance_col="margin_balance", out_col="margin_util"),
            dates, tickers, "margin_util")
    if "holder_conc" in want:
        slots["holder_conc"] = (asof_grid(shareholding, dates, tickers, "big_holder_pct")
                                if not shareholding.empty else empty.copy())
    if "short_util" in want:
        slots["short_util"] = asof_grid(
            balance_util(margin, shareholding, balance_col="short_balance", out_col="short_util"),
            dates, tickers, "short_util")
    if want & set(FLOW_CHANNELS):
        inst = _read(data, "institutional.parquet")
        for col, slot in (("foreign_net", "foreign_flow"), ("trust_net", "trust_flow")):
            if slot in want:
                slots[slot] = (asof_grid(flow_events(inst, col, slot), dates, tickers, slot)
                               if not inst.empty else empty.copy())
    return slots


def build_taiwan_smallcap_panel(
    data: "str | Path | None" = None,
    *,
    adv_window: int = ADV_WINDOW,
    channels: tuple[str, ...] = CORE_CHANNELS,
) -> Panel:
    """The cap-rank 51-250 TWSE/TPEx small/mid-cap panel with its alt-data channels as feature slots.

    ``channels`` defaults to :data:`CORE_CHANNELS` — the three the 2026-07-15 campaign locked — so
    the probe scripts that delegate here stay byte-identical. The orchestrator passes
    :data:`ALL_CHANNELS`.
    """
    data = Path(data) if data is not None else DEFAULT_DATA
    if not (data / "prices.parquet").exists():
        raise FileNotFoundError(
            f"missing {data / 'prices.parquet'} — fetch the substrate first via "
            f"scripts/data/fetch_taiwan_fundamentals_finmind.py + "
            f"scripts/research/taiwan_smallcap_universe.py")

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

    open_w, high_w, low_w, close_w, vol_w = (wide(c) for c in
                                             ("open", "high", "low", "close", "volume"))
    div_path = data / "dividends.parquet"
    dividend_adjusted = div_path.exists()
    if dividend_adjusted:                                    # causal total-return (all OHLC × cf)
        cf = causal_total_return_factor(close_w, pd.read_parquet(div_path))
        open_w, high_w, low_w, close_w = (f * cf for f in (open_w, high_w, low_w, close_w))

    o, h, lo, c, v = (df.to_numpy(dtype=np.float64) for df in
                      (open_w, high_w, low_w, close_w, vol_w))
    tradeable = (np.isfinite(o) & np.isfinite(h) & np.isfinite(lo) & np.isfinite(c) & (c > 0)
                 & np.isfinite(v) & (v > 0))
    dollar = np.where(tradeable, c * v, np.nan)
    adv = pd.DataFrame(dollar).rolling(adv_window, min_periods=ADV_MIN_PERIODS).mean().to_numpy()
    membership = daily_membership(members, dates, tickers)
    active = membership & tradeable & np.isfinite(adv)
    for arr in (o, h, lo, c, v):
        arr[~tradeable] = np.nan

    sector_id, sector_prov = sector_map(data, tickers)
    slots = _build_channels(data, dates, tickers, channels)

    liq_days = int((active.sum(axis=1) >= 25).sum())
    meta = {
        "panel": "taiwan_smallcap",
        "survivorship_free": False,
        "source": "finmind_http+taiwan_smallcap_altdata",
        "universe_def": UNIVERSE_DEF,
        "adjusted": dividend_adjusted,
        "return_basis": "causal_total_return_forward" if dividend_adjusted else "raw_close_unadjusted",
        "n_names_pool": len(tickers),
        "liquid_days_ge25": liq_days,
        "sector_map": sector_prov,
        "channels": list(slots),
        "note": ("cap-rank 51-250 monthly PIT membership; RAW unadjusted prices; current-listing pool "
                 "⇒ survivorship UPPER BOUND"),
    }
    return Panel(dates, tickers, o, h, lo, c, v, active, adv, sector_id, meta, feature_slots=slots)


# --------------------------------------------------------------------------- #
# Tripwires
# --------------------------------------------------------------------------- #
def _self_check(panel: Panel, data: Path = DEFAULT_DATA) -> None:
    """Shape + causality tripwires. The causality ones are the point (LEAK-2)."""
    T, N = panel.T, len(panel.tickers)
    assert panel.close.shape == (T, N), panel.close.shape
    assert np.all(np.diff(panel.dates.astype("int64")) > 0), "dates not strictly increasing"
    act = panel.active.sum(axis=1)
    print(f"  shape (T,N) = ({T:,}, {N})   active/day: mean {act.mean():.1f}  max {act.max()}")
    print(f"  dates {str(panel.dates[0])[:10]} -> {str(panel.dates[-1])[:10]}   "
          f"liquid days {panel.meta['liquid_days_ge25']:,}")
    assert np.isfinite(panel.close[panel.active]).all(), "an active name has a non-finite close"

    # 1. The as-of contract, RE-DERIVED rather than assumed: for a sample of (bar, name) cells,
    #    the slot value must equal the last event whose avail_date <= that bar's date.
    #
    #    Do NOT be tempted to write this as `panel.truncated(k)` vs `panel[:k]`. `Panel.truncated`
    #    only SLICES an already-built panel, so that comparison holds for any array whatsoever and
    #    proves nothing about this builder — a vacuous green. The builder-level version of the test
    #    (delete the future from the SOURCE parquets, rebuild, compare) lives in
    #    tests/crucible/test_taiwan_smallcap_panel.py, where it can use a synthetic fixture.
    mrev = month_revenue_yoy(_read(data, "month_revenue.parquet"), panel.dates)
    checked = 0
    if not mrev.empty and "mrev_yoy" in panel.feature_slots:
        slot = panel.feature_slots["mrev_yoy"]
        rng = np.random.default_rng(0)
        for j in rng.choice(len(panel.tickers), size=min(8, len(panel.tickers)), replace=False):
            ev = mrev[mrev["stock_id"].astype(str) == panel.tickers[j]]
            if ev.empty:
                continue
            for t in rng.choice(T, size=6, replace=False):
                seen = ev[pd.to_datetime(ev["avail_date"]) <= pd.Timestamp(panel.dates[t])]
                want = float(seen.sort_values("avail_date")["yoy"].iloc[-1]) if len(seen) else np.nan
                got = float(slot[t, j])
                assert (np.isnan(got) and np.isnan(want)) or np.isclose(got, want), (
                    f"mrev_yoy[{t},{j}] = {got} but the last PUBLIC value was {want} — the as-of "
                    f"join is not causal")
                checked += 1
    print(f"  LEAK-2 as-of contract re-derived on {checked} sampled cells: PASS")

    # 2. The div add-back must be FORWARD-only: bars before an ex-date are untouched by it. Re-run
    #    the factor with every dividend deleted and assert the head of the series is identical.
    div = pd.read_parquet(data / "dividends.parquet")
    first_ex = pd.to_datetime(div["ex_date"]).min()
    pos = int(pd.DatetimeIndex(panel.dates).searchsorted(first_ex, side="left"))
    close_w = pd.DataFrame(panel.close, index=pd.DatetimeIndex(panel.dates),
                           columns=list(panel.tickers))
    cf_none = causal_total_return_factor(close_w, div.iloc[0:0])
    print(f"  LEAK-2 div add-back is forward-only: first ex-date {first_ex.date()} at bar {pos}; "
          f"factor before it == 1 ({bool((cf_none.to_numpy() == 1.0).all())})")
    assert (cf_none.to_numpy() == 1.0).all(), "empty dividend set must give an identity factor"

    # 3. Membership is as-of: no name is active before its first rebalance admits it.
    members = pd.read_parquet(data / "universe" / "membership.parquet")
    first_rebal = pd.to_datetime(members["rebalance_date"]).min()
    before = pd.DatetimeIndex(panel.dates) < first_rebal
    print(f"  membership starts {first_rebal.date()}; active names on earlier bars = "
          f"{int(panel.active[before].sum())}")
    assert panel.active[before].sum() == 0, "a name is active before the first rebalance"

    span = float((panel.dates[-1] - panel.dates[0]) / np.timedelta64(365, "D"))
    print(f"  calendar span {span:.2f}y | survivorship_free={panel.meta['survivorship_free']} "
          f"| sector sha {panel.meta['sector_map']['sector_map_sha']}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    args = ap.parse_args()
    p = build_taiwan_smallcap_panel(args.data, channels=ALL_CHANNELS)
    print("\nSELF-CHECK")
    _self_check(p, Path(args.data))
    print("\nOK — panel builds and passes its causality tripwires.")
