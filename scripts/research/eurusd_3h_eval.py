"""F1/F2 — EURUSD 3-hour TSMOM + negative control.

Pre-registration: `docs/research/eurusd_3h_preregistration_2026-07-31.md`, sealed in c55bacef
*while the data was still downloading* (12,510 of ~141,000 bars), so the signs provably predate
the sample.

Signal construction is the house's, unchanged in form: trend score = mean of sign(trailing return)
over lookbacks 63/126/252 bars with skip=5, position = sign-scaled by trailing vol, leverage capped.
Everything is strictly causal — the score at bar t uses only bars <= t-skip, and the position at t
is applied to the return from t to t+1.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("eurusd_3h")
OUT = ROOT / "results" / "eurusd_3h"

LOOKBACKS = (63, 126, 252)      # pre-registered (house constants)
SKIP = 5                        # pre-registered
VOL_WIN = 63
LEV_CAP = 3.0
TARGET_VOL = 0.10
COSTS_BP = (0.256, 0.5, 1.0)    # measured median spread, then two stress levels
MIN_SHARPE = 0.30               # pre-registered economic bar
BARS_PER_YEAR = 2075            # 3h bars: ~6,225 hourly / 3


def _sharpe(r: pd.Series) -> float:
    r = r.dropna()
    s = r.std()
    return float(r.mean() / s * math.sqrt(BARS_PER_YEAR)) if s > 0 else 0.0


def _boot_ci(r: np.ndarray, block: int = 200, n: int = 2000, seed: int = 7) -> tuple[float, float]:
    """Moving-block bootstrap CI on the Sharpe — blocks preserve the autocorrelation that inflates
    a naive t (the S1 lesson: t = +5.00 with a CI straddling zero)."""
    rng = np.random.default_rng(seed)
    m = len(r)
    nb = m // block
    out = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m - block, size=nb)
        samp = np.concatenate([r[j:j + block] for j in idx])
        sd = samp.std()
        out[i] = samp.mean() / sd * math.sqrt(BARS_PER_YEAR) if sd > 0 else 0.0
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def tsmom_position(close: pd.Series) -> pd.Series:
    """House trend score -> vol-scaled position. Causal: score[t] uses bars <= t-SKIP."""
    scores = []
    for lb in LOOKBACKS:
        past = close.shift(SKIP)
        ref = close.shift(SKIP + lb)
        scores.append(np.sign(past / ref - 1.0))
    score = pd.concat(scores, axis=1).mean(axis=1)
    ret = close.pct_change()
    vol = ret.rolling(VOL_WIN).std().shift(1) * math.sqrt(BARS_PER_YEAR)
    lev = (TARGET_VOL / vol).clip(upper=LEV_CAP)
    return (score * lev).replace([np.inf, -np.inf], np.nan)


def null_position(close: pd.Series, seed: int = 20260731) -> pd.Series:
    """F2 control: no price information at all."""
    rng = np.random.default_rng(seed)
    v = pd.Series(rng.standard_normal(len(close)), index=close.index)
    return v.where(close.notna())


def evaluate(close: pd.Series, pos: pd.Series, cost_bp: float) -> dict:
    ret = close.pct_change()
    pnl = pos.shift(1) * ret                       # position known at t-1, earns t-1 -> t
    turn = pos.diff().abs()
    net = (pnl - turn * cost_bp / 1e4).dropna()
    return {"sharpe": round(_sharpe(net), 4),
            "ann_ret_pct": round(float(net.mean() * BARS_PER_YEAR * 100), 3),
            "turnover_per_yr": round(float(turn.dropna().sum() / (len(net) / BARS_PER_YEAR)), 1),
            "_net": net}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="EURUSD 3h TSMOM (F1) + negative control (F2)")
    ap.add_argument("--data", default="data/dukascopy/EURUSD_1h.parquet")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    d = pd.read_parquet(ROOT / args.data).sort_values("timestamp")
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    h1 = d.set_index("timestamp")["close"]
    c3 = h1.resample("3h", label="left", closed="left").last().dropna()
    log.info("EURUSD 3h bars: %d (%s..%s)", len(c3), c3.index[0], c3.index[-1])

    holdout = int(len(c3) * 0.25)
    log.info("holdout %d bars -> N_eff at H=1 rebalance = %d", holdout, holdout)

    res = {"bars_3h": int(len(c3)),
           "span": [str(c3.index[0]), str(c3.index[-1])],
           "min_sharpe_bar": MIN_SHARPE, "signals": {}}

    for name, pos in (("F1_tsmom", tsmom_position(c3)), ("F2_null_control", null_position(c3))):
        entry = {}
        for bp in COSTS_BP:
            e = evaluate(c3, pos, bp)
            net = e.pop("_net")
            entry[f"cost_{bp}bp"] = e
            if bp == COSTS_BP[0]:
                lo, hi = _boot_ci(net.to_numpy())
                entry["bootstrap_ci95"] = [round(lo, 4), round(hi, 4)]
                entry["ci_excludes_zero"] = bool(lo > 0 or hi < 0)
                # four equal subperiods
                q = np.array_split(net, 4)
                entry["subperiod_sharpe"] = [round(_sharpe(pd.Series(x)), 3) for x in q]
                entry["subperiods_positive"] = int(sum(1 for x in q if _sharpe(pd.Series(x)) > 0))
        res["signals"][name] = entry

    f1 = res["signals"]["F1_tsmom"]
    f2 = res["signals"]["F2_null_control"]
    base = f1[f"cost_{COSTS_BP[0]}bp"]["sharpe"]
    c1 = base >= MIN_SHARPE
    c2 = bool(f1["ci_excludes_zero"])
    c3_ = f1["subperiods_positive"] >= 3
    c4 = f1[f"cost_{COSTS_BP[1]}bp"]["sharpe"] > 0
    c5 = not f2["ci_excludes_zero"]
    res["conditions"] = {"sharpe_ge_0.30": c1, "ci_excludes_zero": c2,
                         "subperiods_ge_3": c3_, "positive_at_0.5bp": c4, "control_fails": c5}
    res["verdict"] = "PASS" if all((c1, c2, c3_, c4, c5)) else "NO-GO"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eurusd_3h.json").write_text(json.dumps(res, indent=2, default=str))

    print("=" * 74)
    print("F1/F2 — EURUSD 3h TSMOM + negative control")
    print("=" * 74)
    print(f"3h bars: {len(c3):,}   {c3.index[0].date()} .. {c3.index[-1].date()}\n")
    for nm in ("F1_tsmom", "F2_null_control"):
        s = res["signals"][nm]
        print(f"{nm}")
        for bp in COSTS_BP:
            e = s[f"cost_{bp}bp"]
            print(f"   @{bp:>5}bp  Sharpe {e['sharpe']:>7.4f}  ret {e['ann_ret_pct']:>7.3f}%/yr  "
                  f"turnover {e['turnover_per_yr']:>7.1f}/yr")
        print(f"   bootstrap CI95 {s['bootstrap_ci95']}  excludes zero: {s['ci_excludes_zero']}")
        print(f"   subperiods {s['subperiod_sharpe']}  positive {s['subperiods_positive']}/4\n")
    print("pre-committed bars:")
    for k, v in res["conditions"].items():
        print(f"   {k:<22} {'PASS' if v else 'FAIL'}")
    print(f"\nVERDICT: {res['verdict']}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
