"""T2 — CRYPTO time-series momentum as a candidate diversifying sleeve.

Pre-registration: `docs/research/crypto_tsmom_preregistration_2026-07-31.md` (committed with an
empty Results section BEFORE this ran).

Identical treatment to T1 (`country_tsmom_sleeve.py`) so the two are directly comparable: house
machinery unchanged, no new statistic, no tuned parameter.
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

log = logging.getLogger("crypto_tsmom")
OUT = ROOT / "results" / "crypto_xsec"
DATA = ROOT / "data" / "crypto_cache" / "silver_ohlcv.parquet"
COSTS = {"frictionless": 0.0, "c5bps": 0.0005, "c10bps": 0.0010, "c20bps": 0.0020}

MIN_STANDALONE_SR = 0.30     # at 10bp (pre-reg §3)
MAX_CORR_TO_TSMOM = 0.60
MIN_COMBINED_SR = 0.66


def _sharpe(d: pd.Series) -> float:
    d = d.dropna()
    s = d.std()
    return float(d.mean() / s * np.sqrt(mom.ANN)) if s > 0 else 0.0


def _max_dd(d: pd.Series) -> float:
    eq = (1 + d.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = pd.read_parquet(DATA)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["day"] = df["timestamp"].dt.normalize()
    close = (df.groupby(["day", "ticker"])["close"].last()
             .unstack("ticker").sort_index())
    close.index = close.index.tz_localize(None)
    rets = close.pct_change()
    rebal = mom.last_trading_of_period(close.index, "monthly")
    warm = max(mom.LOOKBACKS) + mom.SKIP + mom.VOL_WIN
    if warm >= len(close):
        log.error("panel too short (%d days) for warmup %d", len(close), warm)
        return 1
    rebal = rebal[rebal >= close.index[warm]]
    log.info("crypto panel: %d coins, %d days (%s..%s), %d rebalances",
             close.shape[1], len(close), close.index[0].date(), close.index[-1].date(), len(rebal))

    w = mom.vol_scaled_weights(mom.tsmom_signal(close, rebal), rets, rebal)
    gross, _cd, turn = mom.backtest(w, rets)
    dw = w.fillna(0.0).diff().abs().sum(axis=1)
    dw.iloc[0] = w.iloc[0].abs().sum()

    rows, nets = {}, {}
    for name, bps in COSTS.items():
        net = (gross - dw.reindex(rets.index).fillna(0.0) * bps).dropna()
        nets[name] = net
        rows[name] = {"sharpe": round(_sharpe(net), 3),
                      "ann_ret_pct": round(float(net.mean() * mom.ANN * 100), 2),
                      "ann_vol_pct": round(float(net.std() * np.sqrt(mom.ANN) * 100), 2),
                      "max_dd_pct": round(_max_dd(net) * 100, 2)}
    t2 = nets["c10bps"]

    base = pf.build_momentum_net()
    idx = t2.index.intersection(base.index)
    corr = float(t2.reindex(idx).corr(base.reindex(idx)))
    combined, _ci, _al = pf.risk_parity([base.reindex(idx), t2.reindex(idx)])
    sr_base, sr_t2, sr_comb = (_sharpe(base.reindex(idx)), _sharpe(t2.reindex(idx)),
                               _sharpe(combined))

    c1 = rows["c10bps"]["sharpe"] >= MIN_STANDALONE_SR
    c2 = corr <= MAX_CORR_TO_TSMOM
    c3 = sr_comb > MIN_COMBINED_SR
    verdict = "PASS" if (c1 and c2 and c3) else "NO-GO"

    out = {"sleeve": "crypto_perp_tsmom", "n_coins": int(close.shape[1]),
           "days": int(len(close)), "turnover_ann": round(turn, 2), "by_cost": rows,
           "overlap_days": int(len(idx)),
           "standalone_sharpe_10bp": rows["c10bps"]["sharpe"],
           "corr_to_cross_asset_tsmom": round(corr, 3),
           "sharpe_cross_asset_tsmom_on_overlap": round(sr_base, 3),
           "sharpe_t2_on_overlap": round(sr_t2, 3),
           "combined_risk_parity_sharpe": round(sr_comb, 3),
           "combined_max_dd_pct": round(_max_dd(combined) * 100, 2),
           "bars": {"min_standalone_sr": MIN_STANDALONE_SR, "max_corr": MAX_CORR_TO_TSMOM,
                    "min_combined_sr": MIN_COMBINED_SR},
           "conditions": {"standalone_ge_0.30@10bp": bool(c1), "corr_le_0.60": bool(c2),
                          "combined_gt_0.66": bool(c3)},
           "verdict": verdict,
           "caveat": ("4.3y ~ one crypto cycle; N=10 with BTC/ETH dominant; not survivorship-free. "
                      "A pass is a CANDIDATE needing forward incubation, not a conclusion.")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "crypto_tsmom_sleeve.json").write_text(json.dumps(out, indent=2))

    print("=" * 76)
    print("T2 — CRYPTO time-series momentum as a diversifying sleeve")
    print("=" * 76)
    print(f"coins {close.shape[1]}   days {len(close)}   turnover/yr {turn:.2f}")
    print(f"{'cost':<14}{'Sharpe':>8}{'ret%':>9}{'vol%':>9}{'maxDD%':>9}")
    for k, r in rows.items():
        print(f"{k:<14}{r['sharpe']:>8.3f}{r['ann_ret_pct']:>9.2f}"
              f"{r['ann_vol_pct']:>9.2f}{r['max_dd_pct']:>9.2f}")
    print(f"\noverlap with cross-asset TSMOM: {len(idx)} days")
    print(f"  cross-asset TSMOM Sharpe (overlap): {sr_base:.3f}")
    print(f"  T2 Sharpe (overlap):                {sr_t2:.3f}")
    print(f"  correlation:                        {corr:+.3f}")
    print(f"  risk-parity COMBINED Sharpe:        {sr_comb:.3f}   maxDD {_max_dd(combined)*100:.1f}%")
    print("\npre-committed bars:")
    print(f"  standalone@10bp >= {MIN_STANDALONE_SR}: {rows['c10bps']['sharpe']:>7.3f}  {'PASS' if c1 else 'FAIL'}")
    print(f"  corr <= {MAX_CORR_TO_TSMOM}:             {corr:>+7.3f}  {'PASS' if c2 else 'FAIL'}")
    print(f"  combined > {MIN_COMBINED_SR}:          {sr_comb:>7.3f}  {'PASS' if c3 else 'FAIL'}")
    print(f"\nVERDICT: {verdict}")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
