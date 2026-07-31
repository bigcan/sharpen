"""V1 — volatility-managed OVERLAY on the existing cross-asset TSMOM book.

Pre-registration: `docs/research/vol_managed_overlay_preregistration_2026-07-31.md` (committed with
an empty Results section BEFORE this ran).

Not a sleeve: adds no instrument and no data, so no correlation/admission bar applies. It re-times
the exposure of the validated book, which `tailwind_v1.yaml` deliberately leaves un-vol-timed at the
portfolio level (`target_portfolio_vol: null`).
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

log = logging.getLogger("vol_managed")
OUT = ROOT / "results" / "vol_managed_overlay"

WINDOWS = (21, 63)            # pre-registered, the ENTIRE grid
LEV_CAP = 3.0                 # mirrors env.lev_cap
COSTS = {"2bps": 0.0002, "5bps": 0.0005}
MIN_GAIN = 0.10               # pre-registered pass bar at 5bp
SUBPERIODS = {"2006-09": ("2006-01-01", "2009-12-31"), "2010-15": ("2010-01-01", "2015-12-31"),
              "2016-20": ("2016-01-01", "2020-12-31"), "2021-26": ("2021-01-01", "2026-12-31")}


def _sharpe(d: pd.Series) -> float:
    d = pd.Series(d).dropna()
    s = d.std()
    return float(d.mean() / s * np.sqrt(mom.ANN)) if s > 0 else 0.0


def _overlay(base: pd.Series, window: int, cost_bps: float) -> tuple[pd.Series, float]:
    """w[t] = clip(target_var / realised_var[t-1], 0, cap); cost charged on |dw|."""
    var = base.rolling(window).var().shift(1)      # LAGGED: known strictly before the return scaled
    target = float(base.var())                     # a CONSTANT -> sets scale only, Sharpe-invariant
    w = (target / var).clip(upper=LEV_CAP)
    w = w.reindex(base.index)
    managed = (w * base).dropna()
    dw = w.diff().abs().reindex(managed.index).fillna(0.0)
    cost = dw * cost_bps
    return (managed - cost).dropna(), float(dw.sum() / (len(dw) / mom.ANN))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    base = pf.build_momentum_net().dropna()
    base_sr = _sharpe(base)
    log.info("base cross-asset TSMOM: %d days (%s..%s) Sharpe %.3f",
             len(base), base.index[0].date(), base.index[-1].date(), base_sr)

    results = {}
    for wnd in WINDOWS:
        for cname, bps in COSTS.items():
            managed, turn = _overlay(base, wnd, bps)
            idx = managed.index
            b = base.reindex(idx)
            sr_b, sr_m = _sharpe(b), _sharpe(managed)
            subs = {}
            for k, (a, z) in SUBPERIODS.items():
                bb, mm = b.loc[a:z], managed.loc[a:z]
                if len(mm) > 60:
                    subs[k] = round(_sharpe(mm) - _sharpe(bb), 3)
            results[f"w{wnd}_{cname}"] = {
                "window": wnd, "cost": cname,
                "base_sharpe_on_overlap": round(sr_b, 3),
                "managed_sharpe": round(sr_m, 3),
                "gain": round(sr_m - sr_b, 3),
                "exposure_turnover_ann": round(turn, 1),
                "subperiod_gain": subs,
                "subperiods_positive": sum(1 for v in subs.values() if v > 0),
                "n_subperiods": len(subs)}

    key5 = {w: results[f"w{w}_5bps"] for w in WINDOWS}
    c1 = all(r["gain"] >= MIN_GAIN for r in key5.values())
    c2 = all(r["subperiods_positive"] >= 3 for r in key5.values())
    c3 = all(r["gain"] > 0 for r in key5.values())
    verdict = "PASS" if (c1 and c2 and c3) else "NO-GO"

    out = {"overlay": "moreira_muir_vol_managed", "base_sharpe": round(base_sr, 3),
           "lev_cap": LEV_CAP, "windows": list(WINDOWS), "results": results,
           "bars": {"min_gain_at_5bp": MIN_GAIN, "min_subperiods_positive": 3,
                    "both_windows_positive": True},
           "conditions": {"gain_ge_0.10_both_windows": bool(c1),
                          "subperiods_ge_3_both_windows": bool(c2),
                          "both_windows_positive": bool(c3)},
           "verdict": verdict}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "vol_managed_overlay.json").write_text(json.dumps(out, indent=2))

    print("=" * 78)
    print("V1 — volatility-managed overlay on the existing cross-asset TSMOM book")
    print("=" * 78)
    print(f"base Sharpe: {base_sr:.3f}\n")
    print(f"{'variant':<14}{'base':>8}{'managed':>10}{'gain':>8}{'expTurn':>9}  subperiod gains")
    for k, r in results.items():
        print(f"{k:<14}{r['base_sharpe_on_overlap']:>8.3f}{r['managed_sharpe']:>10.3f}"
              f"{r['gain']:>+8.3f}{r['exposure_turnover_ann']:>9.1f}  {r['subperiod_gain']}")
    print("\npre-committed bars (evaluated at 5bp):")
    for w in WINDOWS:
        r = key5[w]
        print(f"  w={w:<3} gain {r['gain']:+.3f} (need >= +{MIN_GAIN})   "
              f"subperiods + {r['subperiods_positive']}/{r['n_subperiods']} (need >= 3)")
    print(f"\nVERDICT: {verdict}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
