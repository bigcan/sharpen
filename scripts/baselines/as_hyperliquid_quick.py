"""One-shot Avellaneda-Stoikov baseline on the 9.87h Hyperliquid BTC recording.

Reuses the same math as scripts/baselines/avellaneda_stoikov_mm.py, but:
  - single split (full window) — recording is too short for train/val/test
  - sweeps fee in {0.0, 1.5} bps (HL retail maker = 1.5 bps per S453 gate)
  - builds bars inline from raw recorder parquets (no `spread` column needed)

Run:
  python scripts/baselines/as_hyperliquid_quick.py --bar_seconds 10
"""
from __future__ import annotations

import argparse
import glob
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sharpen.data.fill_model import PriceCrossFillModel
from sharpen.analytics.pyfolio_analyzer import PyfolioAnalyzer


GAMMAS = [0.01, 0.1, 1.0]
FEES_BPS = [0.0, 1.5]
ORDER_SIZE = 0.01
MAX_INVENTORY = 0.5
VOL_WINDOW = 60
ADVERSE_SLIP_BPS = 0.5
ADVERSE_VEL_THRESH = 0.001
INITIAL_BALANCE = 100_000.0
MAX_DD_STOP = 0.10


@dataclass
class Result:
    gamma: float
    fee_bps: float
    trade_count: int
    profit_factor: float
    sharpe: float
    sortino: float
    total_return: float
    max_drawdown: float
    halted: bool
    final_equity: float


def build_bars(lob_glob: str, trades_glob: str, bar_seconds: int) -> pd.DataFrame:
    lob_files = sorted(glob.glob(lob_glob))
    trade_files = sorted(glob.glob(trades_glob))
    lob = pd.concat([pd.read_parquet(f) for f in lob_files], ignore_index=True)
    trades = pd.concat([pd.read_parquet(f) for f in trade_files], ignore_index=True) if trade_files else pd.DataFrame()
    lob["spread"] = lob["ask_price_1"] - lob["bid_price_1"]
    lob["ts"] = pd.to_datetime(lob["timestamp_ms"], unit="ms", utc=True)
    lob = lob.set_index("ts").sort_index()
    rule = f"{bar_seconds}s"
    ohlc = lob["mid_price"].resample(rule).ohlc()
    bars = pd.concat({
        "open": ohlc["open"], "high": ohlc["high"], "low": ohlc["low"], "close": ohlc["close"],
        "bbo_bid_qty": lob["bid_size_1"].resample(rule).last(),
        "bbo_ask_qty": lob["ask_size_1"].resample(rule).last(),
        "spread_mean": lob["spread"].resample(rule).mean(),
    }, axis=1)
    if not trades.empty:
        trades["ts"] = pd.to_datetime(trades["timestamp_ms"], unit="ms", utc=True)
        trades = trades.set_index("ts")
        bars["volume"] = trades["sz"].resample(rule).sum().reindex(bars.index).fillna(0.0)
    else:
        bars["volume"] = 0.0
    bars = bars.dropna(subset=["open", "high", "low", "close"])
    bars["mid"] = (bars["open"] + bars["close"]) / 2.0
    return bars.reset_index()


def estimate_kappa(bars: pd.DataFrame) -> float:
    half = bars["spread_mean"].astype(float).values / 2.0
    half = half[half > 0]
    med = float(np.median(half))
    return 1.0 / max(med, 1e-6)


def backtest(bars: pd.DataFrame, gamma: float, kappa: float, fee_bps: float, bar_seconds: float) -> Result:
    n = len(bars)
    mid = bars["mid"].values
    open_ = bars["open"].values
    high = bars["high"].values
    low = bars["low"].values
    close = bars["close"].values
    volume = bars["volume"].values
    log_ret = np.diff(np.log(close), prepend=np.log(close[0]))
    sigma = pd.Series(log_ret).rolling(VOL_WINDOW, min_periods=5).std().bfill().fillna(1e-6).values
    sigma = np.maximum(sigma, 1e-6)
    fill_model = PriceCrossFillModel(
        adverse_slippage_bps=ADVERSE_SLIP_BPS,
        adverse_velocity_threshold=ADVERSE_VEL_THRESH,
    )
    cash = INITIAL_BALANCE
    inventory = 0.0
    equity_curve = np.zeros(n, dtype=np.float64)
    equity_curve[0] = cash
    peak = cash
    trade_count = 0
    halted = False
    fee_rate = fee_bps / 10000.0
    tau = 1.0
    ln_term = math.log(1.0 + gamma / kappa) / gamma
    for i in range(1, n):
        m = mid[i - 1]
        if not np.isfinite(m) or m <= 0:
            equity_curve[i] = equity_curve[i - 1]
            continue
        sig_price = sigma[i - 1] * m
        var = sig_price * sig_price
        reservation = m - inventory * gamma * var * tau
        half_spread = gamma * var * tau / 2.0 + ln_term
        bid_post = reservation - half_spread
        ask_post = reservation + half_spread
        qb = ORDER_SIZE if inventory < MAX_INVENTORY else 0.0
        qa = ORDER_SIZE if inventory > -MAX_INVENTORY else 0.0
        res = fill_model.check_fills(
            bid_price=bid_post, ask_price=ask_post,
            bid_qty=qb, ask_qty=qa,
            bar_open=open_[i], bar_high=high[i], bar_low=low[i],
            bar_close=close[i], bar_volume=volume[i], prev_close=close[i - 1],
        )
        if res.bid_filled:
            inventory += res.bid_fill_qty
            cash -= res.bid_fill_price * res.bid_fill_qty * (1.0 + fee_rate)
            trade_count += 1
        if res.ask_filled:
            inventory -= res.ask_fill_qty
            cash += res.ask_fill_price * res.ask_fill_qty * (1.0 - fee_rate)
            trade_count += 1
        equity = cash + inventory * close[i]
        equity_curve[i] = equity
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        if dd >= MAX_DD_STOP:
            halted = True
            equity_curve[i:] = equity
            break
    returns = np.diff(equity_curve) / equity_curve[:-1]
    returns = returns[np.isfinite(returns)]
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    if losses > 1e-12:
        pf = float(gains / losses)
    elif gains > 0:
        pf = 10.0
    else:
        pf = 0.0
    pf = min(pf, 10.0)
    try:
        a = PyfolioAnalyzer(returns=pd.Series(returns), bar_minutes=bar_seconds / 60.0)
        mtr = a.get_audit_metrics()
        sharpe = float(mtr.get("sharpe_ratio", 0.0))
        sortino = float(mtr.get("sortino_ratio", 0.0))
        mdd = float(mtr.get("max_drawdown", 0.0))
    except Exception:
        sharpe = sortino = mdd = 0.0
    tr = (equity_curve[-1] - INITIAL_BALANCE) / INITIAL_BALANCE
    return Result(gamma, fee_bps, trade_count, pf, sharpe, sortino, float(tr), float(mdd), halted, float(equity_curve[-1]))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lob-glob", default="data/hyperliquid/btc/lob_*.parquet")
    p.add_argument("--trades-glob", default="data/hyperliquid/btc/trades_*.parquet")
    p.add_argument("--bar_seconds", type=int, default=10)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    if args.out is None:
        args.out = ROOT / "data" / "baselines" / f"avellaneda_stoikov_hyperliquid_btc_{args.bar_seconds}s.csv"

    bars = build_bars(args.lob_glob, args.trades_glob, args.bar_seconds)
    dur_h = (bars["ts"].iloc[-1] - bars["ts"].iloc[0]).total_seconds() / 3600.0
    print(f"[data] bars={len(bars):,}  duration={dur_h:.2f}h  "
          f"span={bars['ts'].iloc[0]}..{bars['ts'].iloc[-1]}")
    kappa = estimate_kappa(bars)
    print(f"[kappa] kappa={kappa:.4f}  median half-spread={1.0/kappa:.4f} USD")

    results: List[Result] = []
    for fee in FEES_BPS:
        for gamma in GAMMAS:
            r = backtest(bars, gamma, kappa, fee, args.bar_seconds)
            results.append(r)
            print(f"[bt] fee={fee:>4.1f}bps gamma={gamma:<5} trades={r.trade_count:>6} "
                  f"PF={r.profit_factor:5.2f} Sharpe={r.sharpe:6.2f} Sortino={r.sortino:6.2f} "
                  f"TR={r.total_return*100:7.3f}% MDD={r.max_drawdown*100:5.2f}% "
                  f"{'HALTED' if r.halted else ''}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.__dict__ for r in results]).to_csv(args.out, index=False)
    print(f"\n[out] wrote {args.out}")

    best_pf_at_fee = {fee: max(r.profit_factor for r in results if r.fee_bps == fee) for fee in FEES_BPS}
    print("\n=== SUMMARY ===")
    for fee, pf in best_pf_at_fee.items():
        verdict = "PASS" if pf >= 1.05 else ("BREAK-EVEN" if pf >= 1.00 else "KILL")
        print(f"  fee={fee:>4.1f} bps  best PF={pf:.3f}  -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
