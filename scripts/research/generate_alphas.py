"""Component 3 runner — cost-aware automated alpha generation (ADVISORY).

Evolves DSL alphas (warm-started from the 99-alpha library) against the C1 combined-book
fitness on the FERTILE cross-asset cell, writes an advisory scorecard. The harness tops out at
PROMISING; a deploy read of any survivor requires a Tier-2 deep lifecycle audit.

Modes:
  --mode synthetic   end-to-end CALIBRATION run on a synthetic panel (no network); the key
                     property is that a NOISE panel yields 0 PROMISING (the cont-73 lesson).
  --mode real        load a real panel + its PRODUCTION base book, dispatched on the gates'
                     ``generation.panel`` (GP8-02, validated in ``config._validate``):
                       * ``cross_asset`` → US ETF panel + linear-core TSMOM + rates-carry
                         (validated net SR 0.601 / 0.467), ``production_base_sleeves``.
                       * ``taiwan``      → TAIEX ETF panel (``taiwan_panel_loader``) + TX/TE/TF
                         TSMOM-only book (``taiwan_base_sleeves``, S553-cont steps 1-2).
                     The candidate must improve the REAL book on the panel clock.

Usage:
  python scripts/research/generate_alphas.py --mode synthetic --planted
  python scripts/research/generate_alphas.py --mode real --start 2008-01-01
  python scripts/research/generate_alphas.py --mode real --force \
      --config configs/taiwan_signal_eval.gates.yaml --start 2010-01-01
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finrl_pro_ds.signals.eval_harness import _ls_weights  # noqa: E402
from finrl_pro_ds.signals.features import Panel  # noqa: E402
from finrl_pro_ds.signals.generation.config import (  # noqa: E402
    load_generation_config,
    load_generation_meta,
)
from finrl_pro_ds.signals.generation.evolve import evolve  # noqa: E402
from finrl_pro_ds.signals.library._alpha_formulas import FORMULAS  # noqa: E402
from finrl_pro_ds.signals.library.alphas101 import SKIP  # noqa: E402

log = logging.getLogger("generate_alphas")
DEFAULT_GATES = ROOT / "configs" / "signal_eval.gates.yaml"
SEED_NUMS = (1, 3, 4, 6, 9, 12, 14, 19, 33, 53)   # a low-turnover-ish warm-start subset


def _seed_formulas() -> list[str]:
    return [FORMULAS[n] for n in SEED_NUMS if n not in SKIP]


def _panel_ts(panel: Panel) -> np.ndarray:
    """Per-bar epoch-second decision stamps from the panel's REAL calendar (GP6-01): the C1
    combiner's monthly-meta cadence must rotate on true month-ends, not a synthetic B-day grid
    that ignores holidays (a proxy↔serve calendar skew vs the paper executor)."""
    return panel.dates.astype("datetime64[s]").astype(np.int64).astype(np.float64)


def _synthetic_panel(t: int, n: int, *, planted: bool, seed: int) -> Panel:
    rng = np.random.default_rng(seed)
    base = np.cumsum(0.01 * rng.standard_normal((t, n)), axis=0)
    if planted:                               # weak cross-sectional momentum a DSL alpha can see
        base += 0.0015 * np.cumsum(np.sign(np.diff(base, axis=0, prepend=0.0)), axis=0)
    close = np.exp(base + rng.uniform(3.0, 5.0, size=n))
    open_ = close * (1 + 0.001 * rng.standard_normal((t, n)))
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    vol = rng.uniform(1e6, 1e8, (t, n))
    dates = (np.datetime64("2010-01-04") + np.arange(t) * np.timedelta64(1, "D")
             ).astype("datetime64[ns]")
    return Panel(dates, tuple(f"S{i:02d}" for i in range(n)), open_, high, low, close, vol,
                 np.ones((t, n), bool), close * vol, rng.integers(0, 4, size=n),
                 {"survivorship_free": True, "source": f"synthetic_{'planted' if planted else 'noise'}"})


def _proxy_base_sleeves(panel: Panel, hold: int = 21) -> dict[str, np.ndarray]:
    """Inline TSMOM (252-1) + short-momentum 'carry' PROXY rank-L/S books on the panel.

    NOT the production sleeves — a stand-in so the loop runs end-to-end on real data. Flagged.
    """
    c = panel.close
    fwd1 = panel.forward_returns(1)

    def book(score: np.ndarray) -> np.ndarray:
        out = np.full(panel.T, np.nan)
        w = np.zeros(panel.N)
        for t in range(panel.T - 1):
            if t % hold == 0:
                w = _ls_weights(score[t], panel.active[t], min_names=6)
            out[t] = float(np.nansum(w * fwd1[t]))
        return np.nan_to_num(out)

    mom = np.full_like(c, np.nan)
    mom[252:] = c[252:] / c[:-252] - 1.0      # 12-month momentum (causal)
    rev = np.full_like(c, np.nan)
    rev[21:] = -(c[21:] / c[:-21] - 1.0)      # 1-month reversal proxy ('carry'-ish, low corr)
    return {"tsmom": book(mom), "rates_carry": book(rev)}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="Component 3 — advisory alpha generation")
    ap.add_argument("--config", default=str(DEFAULT_GATES))
    ap.add_argument("--mode", choices=("synthetic", "real"), default="synthetic")
    ap.add_argument("--planted", action="store_true", help="synthetic: plant a weak signal")
    ap.add_argument("--start", default="2008-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--t", type=int, default=900)
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--out", default=str(ROOT / "results" / "signal_eval" / "generation"))
    ap.add_argument("--force", action="store_true",
                    help="run even when generation.enabled is false in the gates (explicit opt-in)")
    args = ap.parse_args()

    cfg, ek = load_generation_config(args.config)
    meta = load_generation_meta(args.config)           # GP8-02: panel/base_sleeves validated here
    if not meta["enabled"] and not args.force:         # GP8-01: the documented opt-in gate, now wired
        log.warning("generation.enabled is false in %s — no-op. Pass --force to run anyway.",
                    args.config)
        return 0
    log.info("generation config loaded (advisory; harness caps at PROMISING). evolve kwargs=%s", ek)

    if args.mode == "synthetic":
        panel = _synthetic_panel(args.t, args.n, planted=args.planted, seed=0)
        base = _proxy_base_sleeves(panel, hold=ek["hold_horizon"])
    elif meta["panel"] == "taiwan":
        from finrl_pro_ds.data.taiwan_panel_loader import load_taiwan_panel
        from finrl_pro_ds.signals.generation.base_sleeves import taiwan_base_sleeves
        panel = load_taiwan_panel(  # 10-ETF cross-asset panel from taiwan_cross_asset.yaml
            args.start, args.end)
        log.info("TAIWAN base book (TX/TE/TF futures TSMOM, single-leg) on the panel clock.")
        base = taiwan_base_sleeves(
            panel, hold_horizon=ek["hold_horizon"], cost_bps=ek["cost_bps"],
            start=args.start, end=args.end)
    else:                                # cross_asset (the default, validated substrate)
        from finrl_pro_ds.data.cross_asset_panel_loader import load_cross_asset_panel
        from finrl_pro_ds.signals.generation.base_sleeves import production_base_sleeves
        panel = load_cross_asset_panel(  # universe comes from the cross-asset config, not gates
            args.start, args.end, config_path=ROOT / "configs" / "cross_asset_momentum.yaml")
        log.info("PRODUCTION base sleeves (linear-core TSMOM + rates-carry) on the panel clock.")
        base = production_base_sleeves(
            panel, hold_horizon=ek["hold_horizon"], cost_bps=ek["cost_bps"],
            start=args.start, end=args.end)
    ts = _panel_ts(panel)                              # GP6-01: real-calendar decision stamps

    rep = evolve(_seed_formulas(), panel, base, ts, cfg, **ek)

    out = Path(args.out) / meta["panel"]               # per-substrate subdir: no cross_asset/taiwan collision
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "advisory": "harness caps at PROMISING; deploy read requires a Tier-2 deep audit",
        "mode": args.mode, "panel": meta["panel"], "panel_source": panel.meta.get("source"),
        "gen_n_total": rep.gen_n_total, "gen_n_eff": rep.gen_n_eff,
        "n_promising": len(rep.promising),
        "hall_of_fame": [
            {"formula": c.formula, "fitness": c.fitness,
             "delta_sr_oos": None if c.result is None else c.result.delta_sr_oos,
             "dsr_aug": None if c.result is None else c.result.dsr_aug,
             "passes_gate": None if c.result is None else c.result.passes_gate}
            for c in rep.hall_of_fame],
        "holdout_validation": rep.holdout_validation,
        "pbo": rep.pbo,        # advisory CSCV Probability of Backtest Overfitting (GP7-03)
    }
    (out / "generation_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("mode=%s  gen_n_total=%d  PROMISING=%d  -> %s",
             args.mode, rep.gen_n_total, len(rep.promising), out / "generation_report.json")
    if not rep.promising:
        log.info("0 PROMISING (expected on noise / efficient cells — the filter is the point).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
