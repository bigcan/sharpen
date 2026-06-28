"""Taiwan cross-sectional momentum — one low-turnover signal through the deflated funnel.

Step two of the "less-liquid market = more extractable alpha" thesis test (artifact
``.agent/artifacts/taiwan_market_data_sourcing_research_s553.md``). Reads the cleaned +
manifested TWSE large-cap panel produced by ``scripts/data/fetch_taiwan_finmind.py``,
builds a :class:`finrl_pro_ds.signals.Panel`, and runs ONE low-turnover signal family —
cross-sectional 12-1 / 6-1 / 3-1 price momentum (Jegadeesh-Titman, skip-the-last-month) —
through the existing 6-tier deflated signal-eval harness under Taiwan-specific gates
(``configs/taiwan_signal_eval.gates.yaml``), whose cost model bakes in the 0.30%
securities-transaction sell tax.

WHY momentum, WHY 12-1, WHY 21-day hold: the US/crypto failures were largely high-turnover
(VWAP/ORB/MACD scalps cost-killed). The thesis only earns a fair test on a LOW-turnover
structural signal where the punitive 0.3% sell tax does not dominate. 12-1 momentum
rebalanced monthly (primary_horizon=21) is the canonical low-turnover cross-sectional
equity factor.

HONEST CAVEATS (surfaced in the scorecard, not hidden):
  - Universe is a CURRENT 0050-style large-cap list → SURVIVORSHIP-BIASED. Every IC here is
    an UPPER BOUND (``panel.meta.survivorship_free=False``; the harness prints it).
  - Three momentum lookbacks are run as the deflation batch (n_trials=3) so the DSR is not
    the degenerate single-trial value; the 12-1 variant is the headline.
  - Net-of-cost capturability uses the Taiwan cost trio (see the gates file): the 0.3% sell
    tax ≈ doubles friction vs US equity. ``us_equiv_ref`` (10 bps) is shown for contrast.

Usage:
    python scripts/data/fetch_taiwan_finmind.py --stocks <45 ids> --start 2017-01-01 \
        --out data/taiwan_universe                      # step 1 (clean + manifest)
    python scripts/research/taiwan_xsec_momentum_eval.py \
        --parquet data/taiwan_universe/taiwan_daily.parquet \
        --gates configs/taiwan_signal_eval.gates.yaml   # step 2 (this script)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from finrl_pro_ds.signals import Gates, Panel, evaluate_batch, to_markdown, write_scorecard
from finrl_pro_ds.signals.spec import SignalSpec

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("taiwan_xsec_momentum")

# Coarse 7-bucket sector map for the PoC universe (for cross-sectional neutralization /
# breadth). NOT GICS — a hand-assigned approximation; a PIT GICS map is a promotion gate.
SECTORS: dict[str, int] = {
    # 0 semiconductors
    "2330": 0, "2454": 0, "2303": 0, "2379": 0, "3034": 0, "3711": 0, "2408": 0, "6415": 0,
    # 1 electronics / hardware
    "2317": 1, "2308": 1, "2382": 1, "2357": 1, "2395": 1, "3008": 1, "3037": 1, "2474": 1,
    # 2 financials
    "2881": 2, "2882": 2, "2884": 2, "2885": 2, "2886": 2, "2887": 2, "2890": 2, "2891": 2,
    "2892": 2, "5871": 2,
    # 3 materials / plastics / steel / cement
    "1301": 3, "1303": 3, "1326": 3, "6505": 3, "2002": 3, "1101": 3, "1102": 3, "2105": 3,
    # 4 telecom / utilities
    "2412": 4, "3045": 4, "4904": 4,
    # 5 shipping / transport
    "2603": 5, "2609": 5, "2615": 5, "2618": 5,
    # 6 consumer / other
    "1216": 6, "2912": 6, "9910": 6, "2207": 6,
}


# --------------------------------------------------------------------------- #
# Signal: skip-a-month trailing momentum (causal — verified by the Tier-0 tripwire)
# --------------------------------------------------------------------------- #
class TrailingMomentum:
    """``(lookback-skip)``-day trailing log-return ending ``skip`` days ago.

    12-1 momentum = ``lookback=252, skip=21``: the return from t-252 to t-21, skipping the
    most recent month (the canonical short-term-reversal guard). Uses only close <= t-skip,
    so it is causal by construction.
    """

    def __init__(self, lookback: int, skip: int = 21,
                 neutralization: tuple[str, ...] = ("winsor", "zscore", "sector", "size"),
                 universe: str = "twse_largecap_current") -> None:
        self.lookback, self.skip = lookback, skip
        months = round(lookback / 21)
        self.spec = SignalSpec(
            name=f"mom_{months}_1",
            hypothesis=f"{months}m-1m cross-sectional price momentum predicts continuation on TWSE",
            family="technical", expected_sign=1,
            neutralization=neutralization, universe=universe)

    def compute(self, panel: Panel) -> np.ndarray:
        c = np.log(panel.close)
        out = np.full(c.shape, np.nan, dtype=np.float64)
        lb, sk, T = self.lookback, self.skip, panel.T
        if lb < T:
            out[lb:] = c[lb - sk:T - sk] - c[:T - lb]
        return out


# --------------------------------------------------------------------------- #
# Panel construction from the cleaned long parquet
# --------------------------------------------------------------------------- #
def _sector_map(tickers: tuple[str, ...], stock_info: Path | None) -> np.ndarray:
    """Per-name integer sector from FinMind ``industry_category``; dead/unknown → own bucket.

    Falls back to the hardcoded 45-name ``SECTORS`` map when no stock-info file is given.
    """
    if stock_info is None or not Path(stock_info).exists():
        return np.array([SECTORS.get(t, 7) for t in tickers], dtype=np.int64)
    info = pd.read_parquet(stock_info)
    cat = dict(zip(info["stock_id"].astype(str), info["industry_category"].astype(str)))
    cats = sorted({cat.get(t, "__unknown__") for t in tickers})
    code = {c: i for i, c in enumerate(cats)}                 # stable codes; unknown gets its own
    return np.array([code[cat.get(t, "__unknown__")] for t in tickers], dtype=np.int64)


def build_panel(parquets: list[Path], *, adv_window: int = 20, min_adv_twd: float = 0.0,
                stock_info: Path | None = None, survivorship_free: bool = False,
                universe_def: str = "twse_largecap_current_45") -> Panel:
    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    df = df.drop_duplicates(["date", "ticker"], keep="first")
    tickers = tuple(sorted(df["ticker"].unique()))
    dates = np.array(sorted(df["date"].unique()), dtype="datetime64[ns]")

    def wide(col: str) -> np.ndarray:
        w = df.pivot(index="date", columns="ticker", values=col).reindex(
            index=pd.DatetimeIndex(dates), columns=list(tickers))
        return w.to_numpy(dtype=np.float64)

    open_, high, low, close, volume = (wide(c) for c in ("open", "high", "low", "close", "volume"))
    tradeable = (np.isfinite(open_) & np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
                 & (close > 0) & np.isfinite(volume) & (volume > 0))
    # Trailing TWD dollar-volume (size proxy + the PIT liquidity gate). Causal: rolling mean
    # over <= t, min_periods=10 so a name needs real history before it can be "active".
    dollar = np.where(tradeable, close * volume, np.nan)
    adv = pd.DataFrame(dollar).rolling(adv_window, min_periods=10).mean().to_numpy()
    # PIT membership = tradeable AND liquid. A delisted name goes inactive when its data ends
    # (NaN pivot); a thin name when its trailing ADV falls below the floor — survivorship and
    # liquidity handled by the SAME causal mask, so the cross-section is point-in-time tradeable.
    active = tradeable & np.isfinite(adv) & (adv >= float(min_adv_twd))
    for arr in (open_, high, low, close, volume):
        arr[~tradeable] = np.nan
    sector_id = _sector_map(tickers, stock_info)
    meta = {"survivorship_free": survivorship_free, "source": "finmind_http_free",
            "universe_def": universe_def, "adjusted": False, "min_adv_twd": float(min_adv_twd),
            "n_names_pool": len(tickers),
            "note": ("RAW unadjusted prices; PIT ADV floor + delisted names included"
                     if survivorship_free else
                     "RAW unadjusted; current large-cap list ⇒ survivorship UPPER BOUND")}
    return Panel(dates, tickers, open_, high, low, close, volume, active, adv, sector_id, meta)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Taiwan cross-sectional momentum through the funnel")
    ap.add_argument("--parquet", default="data/taiwan_universe/taiwan_daily.parquet",
                    help="Comma-separated cleaned panel parquet(s); unioned into one pool.")
    ap.add_argument("--stock-info", default="data/taiwan_universe/stock_info.parquet",
                    help="FinMind TaiwanStockInfo parquet → real industry sectors.")
    ap.add_argument("--min-adv-twd", type=float, default=0.0,
                    help="PIT trailing-ADV floor (TWD/day). >0 enforces a tradeable cross-section.")
    ap.add_argument("--universe-def", default="twse_largecap_current_45")
    ap.add_argument("--gates", default="configs/taiwan_signal_eval.gates.yaml")
    ap.add_argument("--out", default="results/taiwan_xsec_momentum")
    args = ap.parse_args()

    parquets = [(ROOT / p) if not Path(p).is_absolute() else Path(p)
                for p in args.parquet.split(",")]
    gates_path = (ROOT / args.gates) if not Path(args.gates).is_absolute() else Path(args.gates)
    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    info = (ROOT / args.stock_info) if not Path(args.stock_info).is_absolute() else Path(args.stock_info)

    panel = build_panel(parquets, min_adv_twd=args.min_adv_twd, stock_info=info,
                        universe_def=args.universe_def)
    liq = int((panel.active.sum(axis=1) >= 25).sum())
    log.info("Panel: pool N=%d, T=%d (%s..%s) | liquid days(>=25 names)=%d | min_adv=%.1e TWD | %s",
             panel.N, panel.T, str(panel.dates[0])[:10], str(panel.dates[-1])[:10], liq,
             args.min_adv_twd, panel.meta["universe_def"])

    gates = Gates.from_yaml(gates_path)
    signals = {  # ONE family, three lookbacks → deflation n_trials=3 (12-1 is the headline)
        "mom_12_1": TrailingMomentum(252, 21, universe=args.universe_def),
        "mom_6_1": TrailingMomentum(126, 21, universe=args.universe_def),
        "mom_3_1": TrailingMomentum(63, 21, universe=args.universe_def),
    }
    rs = evaluate_batch(signals, panel, gates, f"taiwan_xsec_mom_{args.universe_def}")
    jp, mp = write_scorecard(rs, out_dir)
    print("\n" + to_markdown(rs) + "\n")
    log.info("Scorecard written: %s | %s", mp, jp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
