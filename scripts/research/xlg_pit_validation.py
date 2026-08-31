"""Survivorship-free(ish) validation of the top-100 reversal overlay sleeve (sub-step A).

The cont-74 lead (alpha032+alpha024 L/S overlay, OOS Sharpe ~0.70) was measured on the CURRENT
top-100 by market cap — survivorship-LEANING (today's winners). This re-runs it on a POINT-IN-TIME
universe built from free fja05680 S&P 500 membership history: each day the universe = the top-100
by trailing-60d dollar-volume AMONG the names actually in the S&P 500 that day (PIT membership via
the harness's native `active` mask). This removes the dominant survivorship bias (universe
selection). Residual bias: yfinance cannot price FULLY-delisted names (acquisitions usually retain
history) — quantified as `unpriced_member_days`.

To isolate survivorship from neutralization, both PIT and the current-top100 baseline use
(winsor, zscore, size) neutralization (former-member GICS sectors are unknown).

Decisive: if the PIT OOS sleeve Sharpe holds vs the current-top100 baseline -> survivorship was
NOT the driver (strong GO for the lead). If it collapses -> survivorship inflated it (NO-GO).

Reuses the committed harness + cross_asset_loader (canonical DATA-CLEAN). Research probe.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sharpen.data import cross_asset_loader as cal  # noqa: E402
from sharpen.signals.eval_harness import _ls_weights, compute_scores  # noqa: E402
from sharpen.signals.features import Panel  # noqa: E402
from sharpen.signals.library.alphas101 import SIGNALS as ALPHAS  # noqa: E402

MEMBERS = Path(r"C:\tmp\sp500_pit_members.csv")
CACHE = ROOT / "data" / "raw" / "equity_panel" / "_pit_union.pkl"
GATE = ROOT / "results" / "signal_eval" / "xlg_megacap_gate"
OUT = GATE
START_FETCH = "2014-06-01"      # warmup for the 230d alpha window before 2015
START_EVAL = np.datetime64("2015-01-01")
OOS_SPLIT = np.datetime64("2021-01-01")
TOPN = 100
HOLD = 21
NEU = ("winsor", "zscore", "size")     # no sector (former-member sectors unknown)
NAMES = ["alpha032", "alpha024"]
BY = {s.spec.name: s for s in ALPHAS}


def load_membership() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(MEMBERS)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df[df["date"] >= "2014-01-01"].reset_index(drop=True)
    union: set[str] = set()
    snaps = []
    for _, row in df.iterrows():
        ts = {t.strip().replace(".", "-") for t in str(row["tickers"]).split(",") if t.strip()}
        snaps.append((row["date"], ts))
        union |= ts
    df.attrs["snaps"] = snaps
    return df, sorted(union)


def batched_fetch(tickers: list[str], start: str, batch: int = 120) -> dict:
    frames: dict[str, list] = {f: [] for f in ("open", "high", "low", "close", "volume")}
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        print(f"  fetch {i + 1}-{i + len(chunk)}/{len(tickers)} ...")
        try:
            wide = cal.fetch_ohlcv_wide(chunk, start, None)
        except Exception as exc:  # noqa: BLE001
            print(f"   [warn] chunk failed: {exc!r}"[:120])
            continue
        for f in frames:
            frames[f].append(wide[f])
    return {f: pd.concat(frames[f], axis=1).sort_index() for f in frames}


def build_pit_panel():
    if CACHE.exists():
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)
    memdf, union = load_membership()
    print(f"[members] {len(union)} unique tickers ever-in-index 2014+")
    wide = batched_fetch(union, START_FETCH)
    close = wide["close"]
    # drop all-NaN columns (fully-delisted / unpriceable on yfinance)
    keep = [c for c in close.columns if close[c].notna().any()]
    unpriceable = len(close.columns) - len(keep)
    wide = {f: wide[f][keep] for f in wide}
    close = wide["close"]
    dates = close.index.to_numpy(dtype="datetime64[ns]")
    cols = list(close.columns)
    T, N = len(dates), len(cols)
    col_ix = {c: j for j, c in enumerate(cols)}

    # PIT membership matrix M[t, i] = ticker i in the S&P 500 at date t (latest snapshot <= t)
    snaps = memdf.attrs["snaps"]
    snap_dates = np.array([d.to_datetime64() for d, _ in snaps])
    M = np.zeros((T, N), dtype=bool)
    for t in range(T):
        k = int(np.searchsorted(snap_dates, dates[t], side="right") - 1)
        if k < 0:
            continue
        for tk in snaps[k][1]:
            j = col_ix.get(tk)
            if j is not None:
                M[t, j] = True

    def arr(f):
        return np.asarray(wide[f].to_numpy(), dtype=np.float64)
    open_, high, low, closev, volume = (arr(f) for f in ("open", "high", "low", "close", "volume"))
    priced = np.isfinite(closev) & (closev > 0)
    dollar = closev * np.where(np.isfinite(volume), volume, np.nan)
    adv = pd.DataFrame(dollar).rolling(60, min_periods=20).mean().to_numpy()

    # active[t,i] = PIT member AND priced AND in the top-TOPN by trailing dollar-vol that day
    active = np.zeros((T, N), dtype=bool)
    member_days = 0
    for t in range(T):
        elig = np.where(M[t] & priced[t] & np.isfinite(adv[t]))[0]
        member_days += int((M[t] & priced[t]).sum())
        if elig.size == 0:
            continue
        order = elig[np.argsort(-adv[t][elig])]
        active[t, order[:TOPN]] = True

    sector_id = np.zeros(N, dtype=int)   # unused (no sector neutralization)
    meta = {"survivorship_free": True, "source": "yfinance+fja05680_PIT",
            "universe_def": f"PIT S&P500 -> top-{TOPN} by 60d dollar-vol",
            "n_universe_union": N, "unpriceable_dropped": unpriceable,
            "member_days": member_days}
    panel = Panel(dates, tuple(cols), open_, high, low, closev, volume, active, adv, sector_id, meta)
    with open(CACHE, "wb") as fh:
        pickle.dump(panel, fh)
    return panel


def daily_rets(close, active):
    r = np.full(close.shape, np.nan)
    denom = np.where(close[:-1] > 0, close[:-1], np.nan)
    both = active[1:] & active[:-1]
    r[1:] = np.where(both, close[1:] / denom - 1.0, np.nan)
    return r


def ens_scores(names, panel, neu):
    stack = []
    for n in names:
        s = BY[n]
        sc = compute_scores(s, panel, neu) * s.spec.expected_sign
        m, sd = np.nanmean(sc, axis=1, keepdims=True), np.nanstd(sc, axis=1, keepdims=True)
        stack.append((sc - m) / np.where(sd > 0, sd, np.nan))
    return np.nanmean(np.stack(stack), axis=0)


def ls_daily(eff, rets, active, h, bps):
    T, N = eff.shape
    pnl, w, prev = np.zeros(T), np.zeros(N), np.zeros(N)
    for t in range(T - 1):
        if t % h == 0:
            w = _ls_weights(eff[t], active[t])
            pnl[t + 1] -= np.abs(w - prev).sum() * bps
            prev = w
        pnl[t + 1] += float(np.nansum(w * np.where(np.isfinite(rets[t + 1]), rets[t + 1], 0.0)))
    return pnl


def sr(r):
    r = r[np.isfinite(r)]
    return float(r.mean() / r.std() * np.sqrt(252)) if r.size > 20 and r.std() > 0 else float("nan")


def main():
    panel = build_pit_panel()
    # restrict to eval window (>= 2015), keep warmup history for compute by slicing AFTER scores
    print(f"[PIT panel] union N={panel.N}  T={panel.T}  {panel.dates[0]}..{panel.dates[-1]}")
    print(f"  unpriceable dropped: {panel.meta['unpriceable_dropped']}  "
          f"avg active/day: {panel.active.sum(axis=1)[panel.dates >= START_EVAL].mean():.0f}")
    n_ever = int((panel.active.any(axis=0)).sum())
    print(f"  unique names ever in PIT top-{TOPN}: {n_ever}")

    rets = daily_rets(panel.close, panel.active)
    eff = ens_scores(NAMES, panel, NEU)
    ev = panel.dates >= START_EVAL
    oos = panel.dates >= OOS_SPLIT
    is_ = ev & (panel.dates < OOS_SPLIT)

    pnl10 = ls_daily(eff, rets, panel.active, HOLD, 0.0010)
    pnl25 = ls_daily(eff, rets, panel.active, HOLD, 0.0025)
    pnl0 = ls_daily(eff, rets, panel.active, HOLD, 0.0)

    print("\n=== PIT survivorship-free sleeve (alpha032+alpha024, h21, neu=winsor/zscore/size) ===")
    print(f"{'period':14} {'fric':>7} {'10bps':>7} {'25bps':>7}")
    for lbl, mk in (("FULL_2015_26", ev), ("IS_2015_2020", is_), ("OOS_2021_2026", oos)):
        print(f"{lbl:14} {sr(pnl0[mk]):>7.2f} {sr(pnl10[mk]):>7.2f} {sr(pnl25[mk]):>7.2f}")

    out = {"meta": dict(panel.meta), "n_names_ever_top100": n_ever,
           "pit_sleeve": {lbl: {"fric": sr(pnl0[mk]), "net10": sr(pnl10[mk]), "net25": sr(pnl25[mk])}
                          for lbl, mk in (("FULL", ev), ("IS", is_), ("OOS", oos))}}

    # apples-to-apples baseline: current-top100 (survivorship-LEANING) with the SAME neutralization
    base_cache = ROOT / "data" / "raw" / "equity_panel" / "_top100_panel.pkl"
    if base_cache.exists():
        with open(base_cache, "rb") as fh:
            bp = pickle.load(fh)
        br = daily_rets(bp.close, bp.active)
        beff = ens_scores(NAMES, bp, NEU)
        bo = bp.dates >= OOS_SPLIT
        b10 = ls_daily(beff, br, bp.active, HOLD, 0.0010)
        b25 = ls_daily(beff, br, bp.active, HOLD, 0.0025)
        print("\n=== BASELINE current-top100 (survivorship-LEANING), SAME neu=winsor/zscore/size ===")
        print(f"  OOS @10bps {sr(b10[bo]):+.2f}  @25bps {sr(b25[bo]):+.2f}")
        out["baseline_current_top100_same_neu"] = {"oos_net10": sr(b10[bo]), "oos_net25": sr(b25[bo])}
        print(f"\n  >>> SURVIVORSHIP DELTA (OOS @10bps): PIT {sr(pnl10[oos]):+.2f}  "
              f"vs current {sr(b10[bo]):+.2f}  = {sr(pnl10[oos]) - sr(b10[bo]):+.2f}")

    (OUT / "pit_validation.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\n[wrote] {OUT/'pit_validation.json'}")
    print("\nCompare to survivorship-LEANING current-top100 (cont-74, sector-neu): OOS @10bps +0.70.")
    print("(this PIT run drops sector-neu; run the baseline with the same neu for an exact delta.)")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    main()
