"""Avellaneda-Stoikov analytical MM baseline on BTC LOB 10s data.

Decisive test for the MM workstream: if a closed-form optimal market maker with
perfect knowledge of its own parameters cannot produce PF >= 1.0 on the same
data the SAC agent trained on, the failure is structural (resolution / fees /
no edge) rather than a reward-design problem.

Formulas (Avellaneda & Stoikov 2008, infinite-horizon with tau=1):
    reservation:   r   = mid - q * gamma * sigma^2 * tau
    half-spread:   d   = gamma * sigma^2 * tau / 2 + (1/gamma) * ln(1 + gamma/kappa)
    bid post:      p_b = r - d
    ask post:      p_a = r + d

gamma : inventory aversion (preference knob, grid = {0.01, 0.1, 1.0})
kappa : fill-intensity / market-structure constant (fit empirically, not tuned)
sigma : per-bar return stdev x mid (rolling window)

Kappa estimator:
    In the gamma -> 0 limit, half-spread -> 1/kappa. We anchor kappa so the
    zero-aversion posting width equals the observed median half-spread in the
    train split (in price units). This is the simplest empirical fit that
    respects the "measured, not tuned" constraint.

Splits and fees match configs/mm_sac_btc_lob_10s.yaml. Maker fee = 0 (zero-fee
curriculum start), matching v5 run vq9ba4en so results are comparable.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from finrl_pro_ds.data.fill_model import PriceCrossFillModel
from finrl_pro_ds.analytics.pyfolio_analyzer import PyfolioAnalyzer


# --------------------------------------------------------------------------
# Config (mirrors configs/mm_sac_btc_lob_10s.yaml)
# --------------------------------------------------------------------------
DATA_PATH = ROOT / "data" / "processed" / "btcusdt_lob_10s.parquet"
BAR_SECONDS = 10  # default; override with --bar_seconds
SPLITS = {
    "train": ("2025-10-24", "2025-11-03"),
    "val":   ("2026-02-07", "2026-02-24"),
    "test":  ("2026-03-15", "2026-03-21"),
}
GAMMAS = [0.01, 0.1, 1.0]
ORDER_SIZE = 0.01           # BTC per side (smaller than env max_order_size 0.5 -> ~$1K notional)
MAX_INVENTORY = 0.5         # BTC hard inventory cap (matches env)
MAKER_FEE = 0.0             # bps; v5 run used zero-fee curriculum start
VOL_WINDOW = 60             # bars for rolling sigma (=10 min at 10s bars)
ADVERSE_SLIP_BPS = 0.5      # matches env config
ADVERSE_VEL_THRESH = 0.001
INITIAL_BALANCE = 100_000.0
MAX_DD_STOP = 0.10          # halt if equity drops 10% from peak (matches env)


# --------------------------------------------------------------------------


@dataclass
class BacktestResult:
    gamma: float
    kappa: float
    split: str
    trade_count: int
    profit_factor: float
    sharpe: float
    sortino: float
    total_return: float
    max_drawdown: float
    halted: bool
    final_equity: float


def load_and_split(path: Path) -> Dict[str, pd.DataFrame]:
    df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    df["mid"] = (df["open"] + df["close"]) / 2.0  # fallback mid when bbo missing
    out = {}
    for name, (a, b) in SPLITS.items():
        start = pd.Timestamp(a, tz="UTC")
        end = pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1)
        sub = df[(df["ts"] >= start) & (df["ts"] < end)].reset_index(drop=True)
        out[name] = sub
    return out


def estimate_kappa(train_df: pd.DataFrame) -> float:
    """kappa = 1 / median(half-spread in price units) on train split.

    Zero-aversion limit of A-S optimal half-spread -> 1/kappa, so this anchors
    the baseline posting width to observed microstructure.
    """
    half_spread = train_df["spread_mean"].astype(float).values / 2.0
    # spread_mean is in price units already (USD for BTCUSDT)
    half_spread = half_spread[half_spread > 0]
    med = float(np.median(half_spread))
    kappa = 1.0 / max(med, 1e-6)
    return kappa


def rolling_sigma(returns: np.ndarray, window: int) -> np.ndarray:
    """Rolling stdev of per-bar log returns, forward-filled at head."""
    s = pd.Series(returns).rolling(window, min_periods=5).std()
    s = s.bfill().fillna(1e-6).values
    return np.maximum(s, 1e-6)


def backtest(
    df: pd.DataFrame,
    gamma: float,
    kappa: float,
    split_name: str,
    bar_seconds: float = BAR_SECONDS,
) -> BacktestResult:
    n = len(df)
    if n < VOL_WINDOW + 2:
        raise ValueError(f"{split_name} split too short: {n} bars")

    mid = df["mid"].values
    high = df["high"].values
    low = df["low"].values
    open_ = df["open"].values
    close = df["close"].values
    volume = df["volume"].values
    bid_qty_top = df["bbo_bid_qty"].values
    ask_qty_top = df["bbo_ask_qty"].values

    log_ret = np.diff(np.log(close), prepend=np.log(close[0]))
    sigma_per_bar = rolling_sigma(log_ret, VOL_WINDOW)  # fractional per-bar

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
    fee_rate = MAKER_FEE / 10000.0

    tau = 1.0  # infinite-horizon steady-state approximation
    ln_term = math.log(1.0 + gamma / kappa) / gamma

    for i in range(1, n):
        m = mid[i - 1]
        if not np.isfinite(m) or m <= 0:
            equity_curve[i] = equity_curve[i - 1]
            continue

        # sigma in price units per bar
        sig_price = sigma_per_bar[i - 1] * m
        var = sig_price * sig_price

        # A-S reservation and half-spread
        reservation = m - inventory * gamma * var * tau
        half_spread = gamma * var * tau / 2.0 + ln_term
        bid_post = reservation - half_spread
        ask_post = reservation + half_spread

        # Inventory caps: stop quoting the side that would grow exposure further
        quote_bid_qty = ORDER_SIZE if inventory < MAX_INVENTORY else 0.0
        quote_ask_qty = ORDER_SIZE if inventory > -MAX_INVENTORY else 0.0

        # Fill check against next bar's O/H/L/C
        res = fill_model.check_fills(
            bid_price=bid_post,
            ask_price=ask_post,
            bid_qty=quote_bid_qty,
            ask_qty=quote_ask_qty,
            bar_open=open_[i],
            bar_high=high[i],
            bar_low=low[i],
            bar_close=close[i],
            bar_volume=volume[i],
            prev_close=close[i - 1],
        )

        if res.bid_filled:
            inventory += res.bid_fill_qty
            cash -= res.bid_fill_price * res.bid_fill_qty
            cash -= res.bid_fill_price * res.bid_fill_qty * fee_rate
            trade_count += 1
        if res.ask_filled:
            inventory -= res.ask_fill_qty
            cash += res.ask_fill_price * res.ask_fill_qty
            cash -= res.ask_fill_price * res.ask_fill_qty * fee_rate
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

    # Metrics
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
        analyzer = PyfolioAnalyzer(
            returns=pd.Series(returns),
            bar_minutes=bar_seconds / 60.0,
        )
        metrics = analyzer.get_audit_metrics()
        sharpe = float(metrics.get("sharpe_ratio", 0.0))
        sortino = float(metrics.get("sortino_ratio", 0.0))
        mdd = float(metrics.get("max_drawdown", 0.0))
    except Exception:
        std_r = returns.std()
        sharpe = float(returns.mean() / std_r * math.sqrt(525600 / (bar_seconds / 60.0))) if std_r > 0 else 0.0
        neg = returns[returns < 0]
        dstd = neg.std() if len(neg) > 0 else 0.0
        sortino = float(returns.mean() / dstd * math.sqrt(525600 / (bar_seconds / 60.0))) if dstd > 0 else 0.0
        mdd = 0.0

    total_return = (equity_curve[-1] - INITIAL_BALANCE) / INITIAL_BALANCE

    return BacktestResult(
        gamma=gamma,
        kappa=kappa,
        split=split_name,
        trade_count=trade_count,
        profit_factor=pf,
        sharpe=sharpe,
        sortino=sortino,
        total_return=float(total_return),
        max_drawdown=float(mdd),
        halted=halted,
        final_equity=float(equity_curve[-1]),
    )


def apply_gate(val_pf: float, test_pf: float) -> str:
    m = min(val_pf, test_pf)
    if m >= 1.2:
        return f"GREENLIGHT v6 at 10s (min PF={m:.2f})"
    if m < 1.0:
        return f"KILL v6 at 10s — reopen 1s debate (min PF={m:.2f})"
    return f"MARGINAL (min PF={m:.2f}) — try A-S at 1s next"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--bar_seconds", type=float, default=BAR_SECONDS)
    parser.add_argument("--tag", default=None, help="label used in default --out filename")
    args = parser.parse_args()
    if args.out is None:
        tag = args.tag or args.data.stem
        args.out = ROOT / "data" / "baselines" / f"avellaneda_stoikov_{tag}.csv"

    splits = load_and_split(args.data)
    for name, sub in splits.items():
        print(f"[data] {name}: {len(sub)} bars from {sub['ts'].iloc[0]} to {sub['ts'].iloc[-1]}")

    kappa = estimate_kappa(splits["train"])
    med_half_spread = 1.0 / kappa
    print(f"[kappa] estimated kappa={kappa:.4f} (1/kappa={med_half_spread:.4f} USD median half-spread on train)")

    results: List[BacktestResult] = []
    for gamma in GAMMAS:
        for split_name in ["train", "val", "test"]:
            r = backtest(splits[split_name], gamma=gamma, kappa=kappa, split_name=split_name, bar_seconds=args.bar_seconds)
            results.append(r)
            print(
                f"[bt] gamma={gamma:<5}  split={r.split:<5}  "
                f"trades={r.trade_count:>6}  PF={r.profit_factor:5.2f}  "
                f"Sharpe={r.sharpe:6.2f}  Sortino={r.sortino:6.2f}  "
                f"TR={r.total_return*100:6.2f}%  MDD={r.max_drawdown*100:5.2f}%  "
                f"{'HALTED' if r.halted else ''}"
            )

    df_out = pd.DataFrame([r.__dict__ for r in results])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.out, index=False)
    print(f"\n[out] wrote {args.out}")

    # Gate decision per gamma
    print("\n=== GATE DECISION ===")
    for gamma in GAMMAS:
        val = next(r for r in results if r.gamma == gamma and r.split == "val")
        test = next(r for r in results if r.gamma == gamma and r.split == "test")
        print(f"gamma={gamma}: {apply_gate(val.profit_factor, test.profit_factor)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
