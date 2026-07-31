"""G1/G2 — EURUSD 3-hour one-bar REVERSAL + negative control (scored GROSS).

Pre-registration: docs/research/eurusd_3h_reversal_preregistration_2026-08-01.md (sealed 1d37931f).
Reuses the F1 harness helpers so the two results are directly comparable.
"""
from __future__ import annotations

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
from scripts.research.eurusd_3h_eval import (  # noqa: E402
    BARS_PER_YEAR, LEV_CAP, TARGET_VOL, VOL_WIN, _boot_ci, _sharpe, null_position,
)

log = logging.getLogger("eurusd_rev")
Z_WIN, COSTS_BP, MIN_SHARPE = 63, (0.256, 0.5, 1.0), 0.30


def reversal_position(close: pd.Series) -> pd.Series:
    """G1: -tanh(z) on the LAST COMPLETED bar. z[t] uses r[t] and sigma[t], both known at close t;
    the position is applied to t -> t+1 by the shift(1) in evaluate()."""
    r = close.pct_change()
    z = r / r.rolling(Z_WIN).std()
    vol = r.rolling(VOL_WIN).std().shift(1) * math.sqrt(BARS_PER_YEAR)
    lev = (TARGET_VOL / vol).clip(upper=LEV_CAP)
    return (-np.tanh(z) * lev).replace([np.inf, -np.inf], np.nan)


def evaluate(close, pos, cost_bp):
    ret = close.pct_change()
    pnl = pos.shift(1) * ret
    turn = pos.diff().abs()
    net = (pnl - turn * cost_bp / 1e4).dropna()
    return net, float(turn.dropna().sum() / (len(net) / BARS_PER_YEAR))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    d = pd.read_parquet(ROOT / "data/dukascopy/EURUSD_1h.parquet").sort_values("timestamp")
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    c3 = d.set_index("timestamp")["close"].resample("3h", label="left", closed="left").last().dropna()
    log.info("EURUSD 3h bars: %d (%s..%s)", len(c3), c3.index[0].date(), c3.index[-1].date())

    res = {"bars": int(len(c3)), "signals": {}}
    for nm, pos in (("G1_reversal", reversal_position(c3)), ("G2_control", null_position(c3))):
        e = {}
        for bp in COSTS_BP:
            net, turn = evaluate(c3, pos, bp)
            e[f"cost_{bp}bp"] = {"sharpe": round(_sharpe(net), 4),
                                 "ann_ret_pct": round(float(net.mean() * BARS_PER_YEAR * 100), 3),
                                 "turnover_per_yr": round(turn, 1)}
        gross, _ = evaluate(c3, pos, 0.0)
        glo, ghi = _boot_ci(gross.to_numpy())
        e["gross_sharpe"] = round(_sharpe(gross), 4)
        e["gross_ci95"] = [round(glo, 4), round(ghi, 4)]
        e["gross_ci_excludes_zero"] = bool(glo > 0 or ghi < 0)
        net, _ = evaluate(c3, pos, COSTS_BP[0])
        lo, hi = _boot_ci(net.to_numpy())
        e["net_ci95"] = [round(lo, 4), round(hi, 4)]
        e["net_ci_excludes_zero"] = bool(lo > 0 or hi < 0)
        q = np.array_split(net, 4)
        e["subperiod_sharpe"] = [round(_sharpe(pd.Series(x)), 3) for x in q]
        e["subperiods_positive"] = int(sum(1 for x in q if _sharpe(pd.Series(x)) > 0))
        res["signals"][nm] = e

    g1, g2 = res["signals"]["G1_reversal"], res["signals"]["G2_control"]
    c = {"sharpe_ge_0.30": g1[f"cost_{COSTS_BP[0]}bp"]["sharpe"] >= MIN_SHARPE,
         "net_ci_excludes_zero": g1["net_ci_excludes_zero"],
         "subperiods_ge_3": g1["subperiods_positive"] >= 3,
         "positive_at_0.5bp": g1[f"cost_{COSTS_BP[1]}bp"]["sharpe"] > 0,
         "control_gross_null": not g2["gross_ci_excludes_zero"]}
    res["conditions"] = c
    res["verdict"] = "PASS" if all(c.values()) else "NO-GO"
    out = ROOT / "results" / "eurusd_3h"
    out.mkdir(parents=True, exist_ok=True)
    (out / "eurusd_3h_reversal.json").write_text(json.dumps(res, indent=2, default=str))

    print("=" * 74)
    print("G1/G2 — EURUSD 3h one-bar REVERSAL + control")
    print("=" * 74)
    print(f"3h bars: {len(c3):,}\n")
    for nm in ("G1_reversal", "G2_control"):
        s = res["signals"][nm]
        print(nm)
        for bp in COSTS_BP:
            x = s[f"cost_{bp}bp"]
            print(f"   @{bp:>5}bp  Sharpe {x['sharpe']:>8.4f}  ret {x['ann_ret_pct']:>8.3f}%/yr"
                  f"  turnover {x['turnover_per_yr']:>7.1f}/yr")
        print(f"   GROSS Sharpe {s['gross_sharpe']:>8.4f}  CI {s['gross_ci95']}  excl0={s['gross_ci_excludes_zero']}")
        print(f"   net CI {s['net_ci95']}  excl0={s['net_ci_excludes_zero']}")
        print(f"   subperiods {s['subperiod_sharpe']}  positive {s['subperiods_positive']}/4\n")
    for k, v in c.items():
        print(f"   {k:<24} {'PASS' if v else 'FAIL'}")
    print(f"\nVERDICT: {res['verdict']}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
