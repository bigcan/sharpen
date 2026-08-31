"""Null calibration for the cross-asset ETF substrate at H=1 (pre-mine gate, S553-cont-151).

The same mandatory check applied to every substrate this session: score ZERO-ALPHA candidates through
the SHIPPED corrected contract against the REAL base book, and refuse to mine if the base book
manufactures uplift. On the intraday panel this caught a comparator bleeding 30.6%/yr in friction that
let null candidates clear the FULL gate 15.3% of the time (nominal ~1%) purely by diluting the bleed.

Here the base is the VALIDATED linear core ({tsmom, rates_carry}, net SR 0.601/0.467 at their monthly
cadence) held at `base_hold_horizon: 21` while the candidate runs at H=1, so the dilution channel
should be absent — but "should be" is what the intraday panel also looked like before it was measured.

PRE-REGISTERED READING (unchanged from the intraday version):
  * pass rate <= ~2%  -> baseline fair, the uplift gate measures prediction, MINE.
  * pass rate >> 2%   -> the base book manufactures uplift, DO NOT MINE against it.

Usage:
    python scripts/research/crucible_cross_asset_null_calibration.py [--n-null 150]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sharpen.crucible.corrected_contract import (  # noqa: E402
    CorrectedConfig,
    corrected_contract_fitness,
    fresh_lord_level,
)
from sharpen.data.cross_asset_panel_loader import load_cross_asset_panel  # noqa: E402
from sharpen.signals.eval_harness import _ann_sharpe, _ls_weights  # noqa: E402
from sharpen.signals.generation.base_sleeves import (  # noqa: E402
    _book_from_target_weights,
    production_base_sleeves,
)
from sharpen.signals.generation.config import (  # noqa: E402
    load_generation_config,
    load_generation_meta,
)


def _null_candidate(panel, rng, *, hold_horizon, cost_bps, ls_min_names):
    """Gross-1 dollar-neutral rank-L/S book on RANDOM random-walk scores, net of cost."""
    scores = np.cumsum(rng.standard_normal((panel.T, len(panel.tickers))), axis=0)
    w = np.zeros((panel.T, len(panel.tickers)))
    for t in range(panel.T):
        w[t] = _ls_weights(scores[t], panel.active[t], min_names=ls_min_names)
    return np.asarray(_book_from_target_weights(
        w, panel.forward_returns(1), hold_horizon=hold_horizon, cost_bps=cost_bps))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-null", type=int, default=150)
    ap.add_argument("--config", default="configs/cross_asset_h1_signal_eval.gates.yaml")
    ap.add_argument("--corrected-config", default="configs/crucible_corrected_contract.gates.yaml")
    ap.add_argument("--out",
                    default="results/crucible_power_units/cross_asset_h1_null_calibration.json")
    args = ap.parse_args()
    logging.disable(logging.INFO)

    cfg, ek = load_generation_config(ROOT / args.config)
    meta = load_generation_meta(ROOT / args.config)
    cc = CorrectedConfig.from_yaml(ROOT / args.corrected_config)
    lord = fresh_lord_level(cc)
    panel = load_cross_asset_panel(
        "2007-01-01", config_path=ROOT / "configs" / "cross_asset_momentum.yaml")
    ppy = float(cfg.periods_per_year)
    base_hold = meta.get("base_hold_horizon") or ek["hold_horizon"]

    base, _c = production_base_sleeves(
        panel, hold_horizon=int(base_hold), cost_bps=ek["cost_bps"], return_components=True)
    ts = panel.dates.astype("datetime64[s]").astype(np.int64).astype(np.float64)
    for k, v in base.items():
        f = v[np.isfinite(v)]
        print(f"base '{k}': calendar SR {_ann_sharpe(f, ppy):+.4f}  (hold {base_hold}, "
              f"{ek['cost_bps'] * 1e4:.1f} bp)")
    print(f"candidate hold {ek['hold_horizon']}, t_min {cc.t_min}, "
          f"uplift_min {cfg.min_combination_uplift}\n")

    rows = []
    rng = np.random.default_rng(20260804)
    for i in range(args.n_null):
        cand = _null_candidate(panel, rng, hold_horizon=ek["hold_horizon"],
                               cost_bps=ek["cost_bps"], ls_min_names=ek["ls_min_names"])
        res = corrected_contract_fitness(cand, base, ts, cfg, cc, lord_level=lord)
        fin = cand[np.isfinite(cand)]
        rows.append({"standalone_sr": float(_ann_sharpe(fin, ppy)),
                     "delta_sr": float(res.delta_sr), "z": float(res.corrected_t),
                     "passes": bool(res.passes_corrected), "t_pass": bool(res.t_pass),
                     "lord_pass": bool(res.lord_pass), "uplift_pass": bool(res.uplift_pass),
                     "fragility_pass": bool(res.fragility_pass),
                     "collinearity_pass": bool(res.collinearity_pass)})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{args.n_null}  pass rate "
                  f"{sum(r['passes'] for r in rows) / len(rows):.1%}")

    n = len(rows)
    pr = sum(r["passes"] for r in rows) / n
    legs = {k: float(np.mean([r[k] for r in rows])) for k in
            ("t_pass", "lord_pass", "uplift_pass", "fragility_pass", "collinearity_pass")}
    print(f"\n{'=' * 60}\nZERO-ALPHA NULLS, n={n}")
    print(f"  standalone SR median {np.median([r['standalone_sr'] for r in rows]):+.4f}")
    print(f"  marginal dSR  median {np.median([r['delta_sr'] for r in rows]):+.4f}")
    print(f"  FULL-GATE PASS RATE {pr:.1%}   (nominal ~1%)")
    for k, v in legs.items():
        print(f"      {k:<20}{v:.1%}")
    print(f"\n=> {'MINE PERMITTED' if pr <= 0.02 else 'DO NOT MINE — base manufactures uplift'}")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_null": n, "pass_rate": pr, "leg_pass_rates": legs,
                               "base_hold_horizon": int(base_hold),
                               "candidate_hold_horizon": int(ek["hold_horizon"]),
                               "rows": rows}, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
