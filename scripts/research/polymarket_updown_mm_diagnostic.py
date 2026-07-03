"""Polymarket 5-min up/down maker diagnostic — winner/loser dissection.

Dissects WHY specific systematic two-sided maker wallets win or lose on
Polymarket's 5-minute "Up or Down" crypto binaries, using only free on-chain
trade prints (data/polymarket_updown/, see scripts/data/fetch_polymarket_updown.py)
plus authoritative CLOB-API resolutions. No order-book depth exists for this
period (endpoint dead since 2026-02-20), so everything here is print-based:
realized settlement P&L, fill-price calibration, timing, pairing, inventory.

Stages:
  1. Streaming pass over the trades parquet (row-group at a time):
     - global maker-fill calibration surface: (tau_bucket, price_bin, side) ->
       volume, realized win rate, realized P&L  [all makers, resolved markets]
     - 15s-bucket per-market tape VWAP in "Up-space" (price_up = p(token1)),
       used later for post-fill drift (immediate adverse-selection proxy)
     - full enriched fill set for the wallets under study (maker side), plus
       their (rare) taker-side rows
  2. In-memory diagnostics per wallet: headline P&L (validated against the
     prior pass's pnl_detail parquet), P&L/edge by time-to-resolution bucket,
     calibration by price bin, dual-side pairing per market (locked spread vs
     residual directional exposure), per-market distributions, sizing, per-asset
     and per-day breakdowns, post-fill drift.
  3. Writes parquet/JSON outputs under --out-dir and logs summary tables.

Usage:
    python scripts/research/polymarket_updown_mm_diagnostic.py \
        --data-dir data/polymarket_updown --out-dir results/polymarket_updown_mm
    # smoke test: --max-row-groups 3
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

log = logging.getLogger("pm_updown_diag")

WINDOW_S = 300  # 5-minute market window
TAPE_BUCKET_S = 15  # tape VWAP granularity for drift proxy

# time-to-resolution buckets (seconds), right-open; 'pre' = before window start
TAU_EDGES = [0, 5, 15, 30, 60, 120, 180, 240, 300]
TAU_LABELS = ["0-5s", "5-15s", "15-30s", "30-60s", "60-120s", "120-180s", "180-240s", "240-300s"]

TRADE_COLS = [
    "timestamp", "condition_id", "maker", "taker", "price",
    "usd_amount", "token_amount", "maker_direction", "nonusdc_side",
]


def tau_bucket(tau: np.ndarray) -> np.ndarray:
    """Map seconds-to-resolution to labels; tau>300 -> 'pre', tau<0 -> 'post'."""
    out = np.full(tau.shape, "post", dtype=object)
    idx = np.searchsorted(TAU_EDGES, tau, side="right")  # tau in [0,300] -> 1..8
    ok = (tau >= 0) & (tau <= WINDOW_S)
    out[ok] = np.array(TAU_LABELS, dtype=object)[np.clip(idx[ok] - 1, 0, 7)]
    out[tau > WINDOW_S] = "pre"
    return out


def load_market_meta(data_dir: Path) -> pd.DataFrame:
    m = pd.read_parquet(
        data_dir / "markets.parquet",
        columns=["condition_id", "slug", "token1", "token2", "end_date"],
    )
    m = m[m.slug.str.contains("updown-5m", na=False)].copy()
    m["asset"] = m.slug.str.extract(r"^([a-z]+)-updown")[0]
    m["end_ts"] = (m.end_date.astype("int64") // 1000).astype("int64")  # ms -> s
    m["start_ts"] = m.end_ts - WINDOW_S
    res = pd.read_parquet(data_dir / "resolutions_1month_top10makers.parquet")
    res = res[res.winner_token.notna()][["condition_id", "winner_token"]]
    m = m.merge(res, on="condition_id", how="left")
    m["winner_idx"] = np.select(
        [m.winner_token == m.token1, m.winner_token == m.token2], [1, 2], default=0
    ).astype("int8")  # 0 = unresolved/unknown
    log.info("5m market meta: %d markets, %d resolved", len(m), (m.winner_idx > 0).sum())
    return m.set_index("condition_id")[["asset", "start_ts", "end_ts", "winner_idx"]]


def stream_pass(
    trades_path: Path, meta: pd.DataFrame, wallets: list[str], max_rgs: int | None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Single pass over the tape. Returns (wallet_fills, wallet_taker_rows,
    calibration, tape_vwap, market_summary)."""
    pf = pq.ParquetFile(trades_path)
    n_rgs = pf.num_row_groups if max_rgs is None else min(max_rgs, pf.num_row_groups)
    wallet_set = set(wallets)

    end_ts = meta.end_ts
    winner_idx = meta.winner_idx
    asset = meta.asset

    fills_parts, taker_parts, calib_parts, tape_parts, mkt_parts = [], [], [], [], []
    n_unmapped = 0
    # negative-tau (post-window-settled) notional profile: how far past window end?
    neg_bins = np.array([-1e18, -300, -120, -60, -30, -15, -5, 0.0])
    neg_usd = np.zeros(len(neg_bins) - 1)

    for i in range(n_rgs):
        t = pf.read_row_group(i, columns=TRADE_COLS).to_pandas()
        t["end_ts"] = t.condition_id.map(end_ts)
        unmapped = t.end_ts.isna()
        if unmapped.any():
            n_unmapped += int(unmapped.sum())
            t = t[~unmapped]
        t["end_ts"] = t.end_ts.astype("int64")
        t["winner_idx"] = t.condition_id.map(winner_idx).astype("int8")
        t["token_idx"] = np.where(t.nonusdc_side == "token1", 1, 2).astype("int8")
        t["side"] = np.where(t.maker_direction == "BUY", 1, -1).astype("int8")
        t["tau"] = t.end_ts - t.timestamp.astype("int64")
        t["win"] = (t.token_idx == t.winner_idx).astype("int8")
        resolved = t.winner_idx > 0
        # maker fill P&L at settlement: BUY  -> size*win - usd ; SELL -> usd - size*win
        t["pnl"] = np.where(
            resolved, t.side * (t.token_amount * t.win - t.usd_amount), np.nan
        )

        # --- global calibration (resolved rows only; every row is a maker fill)
        r = t[resolved].copy()
        r["tb"] = tau_bucket(r.tau.to_numpy())
        r["pbin"] = np.clip((r.price * 20).astype(int), 0, 19)  # 0.05-wide bins
        g = (
            r.groupby(["tb", "pbin", "side"], observed=True)
            .agg(n=("price", "size"), size=("token_amount", "sum"),
                 usd=("usd_amount", "sum"), win_size=("token_amount", lambda s: 0.0),
                 pnl=("pnl", "sum"))
        )
        # win_size needs a weighted sum; do it directly (lambda above is placeholder-free)
        ws = r.assign(ws=r.token_amount * r.win).groupby(["tb", "pbin", "side"], observed=True).ws.sum()
        g["win_size"] = ws
        calib_parts.append(g.reset_index())

        # --- 15s tape VWAP in Up-space (in-window rows only, resolved or not)
        t["price_up"] = np.where(t.token_idx == 1, t.price, 1.0 - t.price)
        t["tbk"] = np.clip(t.tau // TAPE_BUCKET_S, 0, WINDOW_S // TAPE_BUCKET_S)  # 0..20
        inwin = t.tau >= 0
        tp = (
            t[inwin].assign(pw=lambda d: d.price_up * d.token_amount)
            .groupby(["condition_id", "tbk"], observed=True)
            .agg(pw=("pw", "sum"), sz=("token_amount", "sum"))
        )
        tape_parts.append(tp.reset_index())

        # --- per-market totals
        mkt_parts.append(
            t.groupby("condition_id", observed=True)
            .agg(n_fills=("price", "size"), usd=("usd_amount", "sum"))
            .reset_index()
        )

        # --- wallets under study
        wm = t[t.maker.isin(wallet_set)]
        fills_parts.append(
            wm[["condition_id", "maker", "timestamp", "tau", "price", "usd_amount",
                "token_amount", "side", "token_idx", "winner_idx", "win", "pnl",
                "end_ts", "price_up"]].copy()
        )
        wt = t[t.taker.isin(wallet_set)]
        if len(wt):
            taker_parts.append(
                wt[["condition_id", "taker", "timestamp", "tau", "price", "usd_amount",
                    "token_amount", "side", "token_idx", "win", "winner_idx"]].copy()
            )
        neg = t.tau < 0
        if neg.any():
            neg_usd += np.histogram(
                t.tau[neg], bins=neg_bins, weights=t.usd_amount[neg]
            )[0]
        if i % 10 == 0:
            log.info("row group %d/%d done", i, n_rgs)

    log.info("unmapped trade rows (not in 5m meta): %d", n_unmapped)
    log.info(
        "post-window-settled notional by seconds past end "
        "[<-300,-300..-120,-120..-60,-60..-30,-30..-15,-15..-5,-5..0]: %s",
        np.round(neg_usd, 0).tolist(),
    )
    fills = pd.concat(fills_parts, ignore_index=True)
    takers = pd.concat(taker_parts, ignore_index=True) if taker_parts else pd.DataFrame()
    calib = (
        pd.concat(calib_parts, ignore_index=True)
        .groupby(["tb", "pbin", "side"], observed=True)
        .sum()
        .reset_index()
    )
    tape = (
        pd.concat(tape_parts, ignore_index=True)
        .groupby(["condition_id", "tbk"], observed=True)
        .sum()
        .reset_index()
    )
    tape["vwap_up"] = tape.pw / tape.sz
    mkt = (
        pd.concat(mkt_parts, ignore_index=True)
        .groupby("condition_id", observed=True)
        .sum()
        .reset_index()
    )
    mkt["asset"] = mkt.condition_id.map(asset)
    mkt["end_ts"] = mkt.condition_id.map(end_ts)
    return fills, takers, calib, tape, mkt


def validate_against_prior(fills: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    prior = (
        pd.read_parquet(data_dir / "pnl_detail_top10makers_1month.parquet")
        .groupby("maker").pnl_row.sum()
    )
    mine = fills.groupby("maker").pnl.sum()
    cmpdf = pd.DataFrame({"prior": prior, "recomputed": mine})
    cmpdf["delta"] = (cmpdf.recomputed - cmpdf.prior).abs()
    log.info("P&L reconciliation vs prior pass:\n%s", cmpdf.round(2).to_string())
    if (cmpdf.delta > 100.0).any():
        log.warning("P&L reconciliation drift >$100 for some wallet — check conventions")
    return cmpdf


def wallet_diagnostics(
    fills: pd.DataFrame, meta: pd.DataFrame, tape: pd.DataFrame, out: Path
) -> dict:
    fills = fills.copy()
    fills["maker"] = fills.maker.str[:10]  # short labels; all distinct among top10
    fills["tb"] = pd.Categorical(
        tau_bucket(fills.tau.to_numpy()), categories=["post", *TAU_LABELS, "pre"], ordered=True
    )
    fills["asset"] = fills.condition_id.map(meta.asset)
    fills["day"] = pd.to_datetime(fills.end_ts, unit="s").dt.date

    # ---------- headline
    def _headline(g: pd.DataFrame) -> pd.Series:
        resolved = g.winner_idx > 0
        usd_res = g.usd_amount[resolved].sum()
        return pd.Series({
            "fills": len(g),
            "markets": g.condition_id.nunique(),
            "notional_usd": g.usd_amount.sum(),
            "resolved_notional": usd_res,
            "pnl": g.pnl.sum(),
            "bps": 1e4 * g.pnl.sum() / usd_res if usd_res else np.nan,
            "buy_share": (g.side == 1).mean(),
            "sell_pnl": g.pnl[g.side == -1].sum(),
            "median_fill_usd": g.usd_amount.median(),
            "p95_fill_usd": g.usd_amount.quantile(0.95),
        })

    headline = fills.groupby("maker").apply(_headline, include_groups=False)
    log.info("HEADLINE per wallet:\n%s", headline.round(2).to_string())

    # ---------- P&L / edge by time-to-resolution bucket
    tb = (
        fills.groupby(["maker", "tb"], observed=False)
        .agg(usd=("usd_amount", "sum"), pnl=("pnl", "sum"), fills=("price", "size"))
    )
    tb["bps"] = 1e4 * tb.pnl / tb.usd
    tb["usd_share"] = tb.usd / tb.groupby("maker").usd.transform("sum")
    log.info("EDGE by tau bucket (bps of notional):\n%s",
             tb.reset_index().pivot(index="tb", columns="maker", values="bps").round(1).to_string())
    log.info("NOTIONAL share by tau bucket:\n%s",
             tb.reset_index().pivot(index="tb", columns="maker", values="usd_share").round(3).to_string())

    # ---------- calibration: BUY fills, price bin -> realized win rate - price
    buys = fills[(fills.side == 1) & (fills.winner_idx > 0)].copy()
    buys["pbin"] = np.clip((buys.price * 10).astype(int), 0, 9)  # 0.1-wide bins
    cal = buys.groupby(["maker", "pbin"], observed=True).apply(
        lambda g: pd.Series({
            "vw_price": np.average(g.price, weights=g.token_amount),
            "win_rate": np.average(g.win, weights=g.token_amount),
            "size": g.token_amount.sum(),
        }),
        include_groups=False,
    )
    cal["edge_per_share"] = cal.win_rate - cal.vw_price
    log.info("CALIBRATION (BUY: win_rate - vw_price, per 0.1 price bin):\n%s",
             cal.reset_index().pivot(index="pbin", columns="maker",
                                     values="edge_per_share").round(4).to_string())

    # ---------- pairing / inventory per wallet x market (BUY legs)
    bm = buys.groupby(["maker", "condition_id", "token_idx"], observed=True).agg(
        qty=("token_amount", "sum"), usd=("usd_amount", "sum")
    ).reset_index()
    piv = bm.pivot_table(index=["maker", "condition_id"], columns="token_idx",
                         values=["qty", "usd"], fill_value=0.0)
    piv.columns = [f"{a}{b}" for a, b in piv.columns]
    for c in ("qty1", "qty2", "usd1", "usd2"):
        if c not in piv.columns:
            piv[c] = 0.0
    piv["vwap1"] = np.where(piv.qty1 > 0, piv.usd1 / piv.qty1, np.nan)
    piv["vwap2"] = np.where(piv.qty2 > 0, piv.usd2 / piv.qty2, np.nan)
    piv["paired"] = np.minimum(piv.qty1, piv.qty2)
    piv["pair_cost"] = piv.vwap1 + piv.vwap2
    piv["locked_pnl"] = piv.paired * (1.0 - piv.pair_cost)
    piv["resid_qty"] = piv.qty1 - piv.qty2  # >0: net long Up
    mkt_pnl = fills[fills.winner_idx > 0].groupby(["maker", "condition_id"], observed=True).pnl.sum()
    piv = piv.join(mkt_pnl.rename("pnl"))
    piv["resid_pnl"] = piv.pnl - piv.locked_pnl.fillna(0.0)
    piv["imbalance"] = np.abs(piv.resid_qty) / (piv.qty1 + piv.qty2).replace(0, np.nan)

    def _pairing(g: pd.DataFrame) -> pd.Series:
        both = (g.qty1 > 0) & (g.qty2 > 0)
        return pd.Series({
            "mkts": len(g),
            "both_sides_frac": both.mean(),
            "vw_pair_cost": np.average(g.pair_cost[both], weights=g.paired[both])
            if both.any() else np.nan,
            "locked_pnl": g.locked_pnl.sum(),
            "resid_pnl": g.resid_pnl.sum(),
            "med_imbalance": g.imbalance.median(),
            "mkt_win_rate": (g.pnl > 0).mean(),
            "mkt_pnl_p5": g.pnl.quantile(0.05),
            "mkt_pnl_p50": g.pnl.median(),
            "mkt_pnl_p95": g.pnl.quantile(0.95),
            "worst10_loss": g.pnl.nsmallest(10).sum(),
        })

    pairing = piv.reset_index().groupby("maker").apply(_pairing, include_groups=False)
    log.info("PAIRING / inventory per wallet:\n%s", pairing.round(3).to_string())

    # ---------- per-asset
    pa = fills.groupby(["maker", "asset"], observed=True).agg(
        usd=("usd_amount", "sum"), pnl=("pnl", "sum"))
    pa["bps"] = 1e4 * pa.pnl / pa.usd
    log.info("EDGE by asset (bps):\n%s",
             pa.reset_index().pivot(index="asset", columns="maker", values="bps").round(1).to_string())

    # ---------- per-day pnl consistency
    daily = fills.groupby(["maker", "day"], observed=True).pnl.sum().reset_index()
    dstat = daily.groupby("maker").pnl.agg(["mean", "std", "min"])
    dstat["pos_days"] = daily[daily.pnl > 0].groupby("maker").size() / daily.groupby("maker").size()
    dstat["daily_sharpe"] = dstat["mean"] / dstat["std"]
    log.info("DAILY P&L consistency:\n%s", dstat.round(2).to_string())

    # ---------- post-fill drift (immediate adverse-selection proxy), per wallet
    tape_idx = tape.set_index(["condition_id", "tbk"]).vwap_up
    f = fills[(fills.tau >= 0) & (fills.tau <= WINDOW_S)].copy()
    f["tbk"] = (f.tau // TAPE_BUCKET_S).astype(int)
    f["tbk_next"] = f.tbk - 1  # next 15s bucket in wall-clock = lower tau bucket
    f = f[f.tbk_next >= 0]
    f["vwap_next"] = tape_idx.reindex(
        pd.MultiIndex.from_frame(f[["condition_id", "tbk_next"]])
    ).to_numpy()
    f = f[f.vwap_next.notna()]
    # drift experienced on the token bought/sold: buy -> next - fill ; sell -> fill - next
    f["fill_up"] = f.price_up
    dir_up = np.where(f.token_idx == 1, 1, -1) * f.side  # +1 if net long Up from this fill
    f["drift"] = dir_up * (f.vwap_next - f.fill_up)
    drift = f.groupby("maker").apply(
        lambda g: pd.Series({
            "vw_drift_next15s": np.average(g.drift, weights=g.usd_amount),
            "n": len(g),
        }),
        include_groups=False,
    )
    log.info("POST-FILL 15s DRIFT (vw, + = fill immediately profitable):\n%s",
             drift.round(5).to_string())

    # ---------- persist
    out.mkdir(parents=True, exist_ok=True)
    headline.to_parquet(out / "wallet_headline.parquet")
    tb.reset_index().to_parquet(out / "wallet_tau_buckets.parquet")
    cal.reset_index().to_parquet(out / "wallet_calibration.parquet")
    piv.reset_index().to_parquet(out / "wallet_market_pairing.parquet")
    daily.to_parquet(out / "wallet_daily_pnl.parquet")
    return {
        "headline": headline.round(4).to_dict(orient="index"),
        "pairing": pairing.round(4).to_dict(orient="index"),
        "daily": dstat.round(4).to_dict(orient="index"),
        "drift": drift.round(6).to_dict(orient="index"),
    }


def global_calibration_report(calib: pd.DataFrame, out: Path) -> None:
    calib = calib.copy()
    calib["win_rate"] = calib.win_size / calib["size"]
    calib["bps"] = 1e4 * calib.pnl / calib.usd
    calib.to_parquet(out / "global_calibration.parquet")
    tot = calib.groupby("tb", observed=True).agg(usd=("usd", "sum"), pnl=("pnl", "sum"))
    tot["bps"] = 1e4 * tot.pnl / tot.usd
    order = ["pre", *reversed(TAU_LABELS), "post"]
    tot = tot.reindex([b for b in order if b in tot.index])
    log.info("GLOBAL maker edge by tau bucket (ALL makers, resolved mkts):\n%s",
             tot.round(2).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/polymarket_updown")
    ap.add_argument("--out-dir", default="results/polymarket_updown_mm")
    ap.add_argument("--trades", default="trades_1month_updown_2026-04-04_2026-05-04.parquet")
    ap.add_argument("--max-row-groups", type=int, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    data_dir, out = Path(args.data_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    wallets = (
        pd.read_parquet(data_dir / "pnl_detail_top10makers_1month.parquet")["maker"]
        .unique().tolist()
    )
    log.info("wallets under study: %d", len(wallets))

    meta = load_market_meta(data_dir)
    fills, takers, calib, tape, mkt = stream_pass(
        data_dir / args.trades, meta, wallets, args.max_row_groups
    )
    log.info("wallet maker fills: %d | wallet taker rows: %d", len(fills), len(takers))
    fills.to_parquet(out / "top10_fills_enriched.parquet")
    if len(takers):
        takers.to_parquet(out / "top10_taker_rows.parquet")
    mkt.to_parquet(out / "market_summary.parquet")
    tape.to_parquet(out / "tape_vwap_15s.parquet")

    if args.max_row_groups is None:
        validate_against_prior(fills, data_dir)

    report = wallet_diagnostics(fills, meta, tape, out)
    global_calibration_report(calib, out)

    # taker-side context for the studied wallets
    if len(takers):
        tk = takers[takers.winner_idx > 0].copy()
        # 'side' col was the MAKER's direction; the taker bought when the maker sold
        tk["taker_buy"] = tk.side == -1
        tk["pnl"] = np.where(tk.taker_buy, tk.token_amount * tk.win - tk.usd_amount,
                             tk.usd_amount - tk.token_amount * tk.win)
        ts = tk.groupby("taker").agg(usd=("usd_amount", "sum"), pnl=("pnl", "sum"),
                                     n=("price", "size"))
        log.info("TAKER-side activity of studied wallets:\n%s", ts.round(2).to_string())
        report["taker_side"] = ts.round(2).to_dict(orient="index")

    with open(out / "diagnostic_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info("done -> %s", out)


if __name__ == "__main__":
    main()
