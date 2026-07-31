"""H1 — pooled FX-majors reversal on the FROZEN G1 signal.

Pre-registration: `docs/research/fx_majors_reversal_preregistration_2026-08-01.md`.

The signal is imported unchanged from `eurusd_3h_reversal_eval` — nothing about it is re-specified.
Only the instruments differ. EURUSD generated the hypothesis and is therefore reported separately
and EXCLUDED from the pooled out-of-sample statistic.

Pooling = equal-weight average of the per-instrument net return series on their common timestamps.
That is the standard way to buy power for a weak effect: the SE of the mean falls with the number of
imperfectly-correlated series, which is exactly what G1's sub-MDE point estimate needed.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.research.eurusd_3h_eval import BARS_PER_YEAR, _boot_ci, _sharpe, null_position  # noqa: E402
from scripts.research.eurusd_3h_reversal_eval import reversal_position  # noqa: E402

log = logging.getLogger("fx_majors")
OOS = ["USDJPY", "GBPUSD", "AUDUSD", "USDCHF"]     # pre-registered, locked before download
INSAMPLE = "EURUSD"                                 # generated the hypothesis — reported, not pooled
MIN_SHARPE = 0.30


def load_3h(instr: str) -> tuple[pd.Series, float]:
    f = ROOT / "data" / "dukascopy" / f"{instr}_1h.parquet"
    d = pd.read_parquet(f).sort_values("timestamp")
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    g = d.set_index("timestamp")
    c3 = g["close"].resample("3h", label="left", closed="left").last().dropna()
    sp3 = (g["mean_spread"] / g["close"]).resample("3h", label="left", closed="left").mean()
    return c3, float((sp3.reindex(c3.index).dropna() * 1e4).median())


def net_series(close: pd.Series, pos: pd.Series, cost_bp: float) -> pd.Series:
    ret = close.pct_change()
    return (pos.shift(1) * ret - pos.diff().abs() * cost_bp / 1e4).dropna()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    per: dict[str, dict] = {}
    net_by, gross_by, nullg_by = {}, {}, {}

    for ins in OOS + [INSAMPLE]:
        try:
            c3, spread_bp = load_3h(ins)
        except FileNotFoundError:
            log.warning("%s: no data — skipping", ins)
            continue
        pos = reversal_position(c3)
        npos = null_position(c3)
        g = net_series(c3, pos, 0.0)
        n1 = net_series(c3, pos, spread_bp)
        n2 = net_series(c3, pos, 2 * spread_bp)
        per[ins] = {"bars": int(len(c3)), "spread_bp": round(spread_bp, 4),
                    "gross_sharpe": round(_sharpe(g), 4),
                    "net_sharpe": round(_sharpe(n1), 4),
                    "net_sharpe_2x": round(_sharpe(n2), 4),
                    "turnover_per_yr": round(float(pos.diff().abs().dropna().sum()
                                                   / (len(n1) / BARS_PER_YEAR)), 1)}
        gross_by[ins], net_by[ins] = g, n1
        nullg_by[ins] = net_series(c3, npos, 0.0)
        log.info("%s: %d bars, spread %.3fbp, gross %+.4f, net %+.4f",
                 ins, len(c3), spread_bp, _sharpe(g), _sharpe(n1))

    have = [i for i in OOS if i in per]
    if len(have) < 2:
        log.error("need >=2 out-of-sample instruments, have %d", len(have))
        return 1

    def pool(d: dict, keys: list[str]) -> pd.Series:
        return pd.concat([d[k] for k in keys], axis=1).dropna().mean(axis=1)

    pg, pn, pnull = pool(gross_by, have), pool(net_by, have), pool(nullg_by, have)
    pn2 = pool({k: net_series(*load_3h(k)[:1], reversal_position(load_3h(k)[0]),
                              2 * per[k]["spread_bp"]) for k in have}, have) \
        if False else None  # (2x pooled computed below from per-instrument series)
    n2_by = {}
    for k in have:
        c3, sp = load_3h(k)
        n2_by[k] = net_series(c3, reversal_position(c3), 2 * sp)
    pn2 = pool(n2_by, have)

    glo, ghi = _boot_ci(pg.to_numpy())
    nlo, nhi = _boot_ci(pn.to_numpy())
    clo, chi = _boot_ci(pnull.to_numpy())
    pos_gross = sum(1 for k in have if per[k]["gross_sharpe"] > 0)

    c = {"pooled_gross_ci_excludes_zero": bool(glo > 0 or ghi < 0),
         "pooled_net_ge_0.30": bool(_sharpe(pn) >= MIN_SHARPE),
         "at_least_3of4_gross_positive": bool(pos_gross >= 3),
         "pooled_net_positive_at_2x": bool(_sharpe(pn2) > 0),
         "control_gross_ci_includes_zero": bool(not (clo > 0 or chi < 0))}
    res = {"out_of_sample": have, "in_sample_reported_only": INSAMPLE,
           "per_instrument": per, "pooled_bars": int(len(pg)),
           "pooled_gross_sharpe": round(_sharpe(pg), 4), "pooled_gross_ci95": [round(glo, 4), round(ghi, 4)],
           "pooled_net_sharpe": round(_sharpe(pn), 4), "pooled_net_ci95": [round(nlo, 4), round(nhi, 4)],
           "pooled_net_sharpe_2x": round(_sharpe(pn2), 4),
           "n_gross_positive": pos_gross,
           "control_pooled_gross_sharpe": round(_sharpe(pnull), 4),
           "control_pooled_gross_ci95": [round(clo, 4), round(chi, 4)],
           "conditions": c, "verdict": "PASS" if all(c.values()) else "NO-GO"}

    out = ROOT / "results" / "fx_majors"
    out.mkdir(parents=True, exist_ok=True)
    (out / "fx_majors_reversal.json").write_text(json.dumps(res, indent=2, default=str))

    print("=" * 76)
    print("H1 — POOLED FX-majors reversal (frozen G1 signal)")
    print("=" * 76)
    print(f"{'instrument':<10}{'bars':>8}{'spread':>9}{'gross':>9}{'net':>9}{'net@2x':>9}{'turn/yr':>9}")
    for k in have + ([INSAMPLE] if INSAMPLE in per else []):
        p = per[k]
        tag = "  (in-sample)" if k == INSAMPLE else ""
        print(f"{k:<10}{p['bars']:>8,}{p['spread_bp']:>9.3f}{p['gross_sharpe']:>9.4f}"
              f"{p['net_sharpe']:>9.4f}{p['net_sharpe_2x']:>9.4f}{p['turnover_per_yr']:>9.1f}{tag}")
    print(f"\nPOOLED out-of-sample ({', '.join(have)}), {len(pg):,} common bars")
    print(f"   gross {res['pooled_gross_sharpe']:+.4f}  CI {res['pooled_gross_ci95']}"
          f"  excludes zero: {c['pooled_gross_ci_excludes_zero']}")
    print(f"   net   {res['pooled_net_sharpe']:+.4f}  CI {res['pooled_net_ci95']}")
    print(f"   net@2x {res['pooled_net_sharpe_2x']:+.4f}   gross-positive {pos_gross}/{len(have)}")
    print(f"   CONTROL gross {res['control_pooled_gross_sharpe']:+.4f} CI {res['control_pooled_gross_ci95']}")
    print()
    for k, v in c.items():
        print(f"   {k:<34} {'PASS' if v else 'FAIL'}")
    print(f"\nVERDICT: {res['verdict']}")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
