"""WHICH corrected-contract leg actually binds? Re-score mined candidates and show every leg.

WHY. Eight mining rounds returned 0 PROMISING, and the headline count was all that was ever read.
The ledger tells a more specific story that deserves following: every scored candidate carries a
POSITIVE ``delta_sr_oos`` (+0.19..+0.42 across runs) — comfortably above the 0.10 uplift floor and far
above the measured null distribution (median -0.03 intraday / -0.21 cross-asset) — while its
``marginal_hlz_t`` is NEGATIVE. That second number is the F1-sealed leg which the corrected contract
DROPS (loop.py:250-253 states both ``dsr`` and ``marginal_hlz_t`` are diagnostics only under
contract="corrected"), so it explains nothing about the verdict.

The decision-bearing fields are ``corrected_t`` and ``p_value``, and the ledger does NOT persist them.
So the actual question — which of the five corrected legs refused these candidates, and by how much —
has never been answered. It matters in both directions:

  * if the binding leg is the JKM z at, say, 0.5, these are honest nulls and the substrate is closed;
  * if it is at 2.0-2.3 against a 2.33 bar, they are near-misses whose right home is the incubation
    lockbox, not the bin — and reporting "0 PROMISING" without saying so understates the search;
  * if the binding leg is LORD++ (a multiplicity budget, not evidence about the candidate) then the
    refusal is an accounting artifact of batch size and should be labelled as such.

This script re-scores each mined candidate through the SHIPPED
``corrected_contract.corrected_contract_fitness`` on its own substrate and prints all five legs. It
changes NO threshold and promotes nothing — it is a read of what already happened.

Usage:
    python scripts/research/crucible_leg_forensics.py --run results/crucible_orchestrator_xa_h1
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
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
from sharpen.signals.generation.config import (  # noqa: E402
    load_generation_config,
    load_generation_meta,
)


def build_substrate(cfg, ek, meta):
    """Rebuild the panel + base book exactly as the runner did for this substrate."""
    panel_kind = meta["panel"]
    base_hold = meta.get("base_hold_horizon") or ek["hold_horizon"]
    if panel_kind == "cross_asset":
        from sharpen.data.cross_asset_panel_loader import load_cross_asset_panel
        from sharpen.signals.generation.base_sleeves import production_base_sleeves
        panel = load_cross_asset_panel(
            "2007-01-01", config_path=ROOT / "configs" / "cross_asset_momentum.yaml")
        base, _ = production_base_sleeves(
            panel, hold_horizon=int(base_hold), cost_bps=ek["cost_bps"], return_components=True)
    else:
        from sharpen.crucible.data.intraday_panel import (
            build_fx_majors_panel,
            build_intraday_panel,
        )
        from sharpen.signals.generation.base_sleeves import intraday_base_sleeves
        panel = (build_fx_majors_panel() if panel_kind == "intraday_fx"
                 else build_intraday_panel())
        base, _ = intraday_base_sleeves(
            panel, hold_horizon=int(base_hold), cost_bps=ek["cost_bps"],
            periods_per_year=float(cfg.periods_per_year), return_components=True)
    return panel, base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/crucible_orchestrator_xa_h1")
    ap.add_argument("--config", default="configs/cross_asset_h1_signal_eval.gates.yaml")
    ap.add_argument("--corrected-config", default="configs/crucible_corrected_contract.gates.yaml")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    logging.disable(logging.INFO)

    cfg, ek = load_generation_config(ROOT / args.config)
    meta = load_generation_meta(ROOT / args.config)
    cc = CorrectedConfig.from_yaml(ROOT / args.corrected_config)
    lord = fresh_lord_level(cc)
    panel, base = build_substrate(cfg, ek, meta)
    ts = panel.dates.astype("datetime64[s]").astype(np.int64).astype(np.float64)

    db = ROOT / args.run / "real" / "trial_ledger.db"
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT * FROM trial_ledger WHERE delta_sr_oos IS NOT NULL "
        "ORDER BY delta_sr_oos DESC")][: args.top]
    print(f"{db}: re-scoring {len(rows)} candidates through the SHIPPED corrected contract")
    print(f"thresholds: t_min {cc.t_min}, lord {lord:.5f}, uplift_min {cfg.min_combination_uplift}, "
          f"max_base_corr {cfg.max_base_corr}\n")

    from sharpen.signals.generation.evolve import _candidate_returns

    print(f"{'type':<16}{'z':>8}{'p':>9}{'dSR':>8}{'rho':>7}  "
          f"{'t':>2}{'L':>2}{'U':>2}{'F':>2}{'C':>2}  formula")
    out = []
    for r in rows:
        try:
            got = _candidate_returns(r["formula"], panel, hold_horizon=ek["hold_horizon"],
                                     cost_bps=ek["cost_bps"], min_names=ek["ls_min_names"])
        except Exception as e:                                    # noqa: BLE001
            print(f"  (skip: {type(e).__name__} {e})")
            continue
        if got is None:
            print(f"  (skip: degenerate genome) {(r['formula'] or '')[:50]}")
            continue
        cand = got[0]
        res = corrected_contract_fitness(cand, base, ts, cfg, cc, lord_level=lord)
        flags = "".join("Y" if b else "." for b in
                        (res.t_pass, res.lord_pass, res.uplift_pass,
                         res.fragility_pass, res.collinearity_pass))
        print(f"{r['candidate_type']:<16}{res.corrected_t:>8.3f}{res.p_value:>9.4f}"
              f"{res.delta_sr:>8.3f}{res.rho:>7.3f}  "
              f"{flags[0]:>2}{flags[1]:>2}{flags[2]:>2}{flags[3]:>2}{flags[4]:>2}  "
              f"{(r['formula'] or '')[:44]}")
        out.append({"formula": r["formula"], "candidate_type": r["candidate_type"],
                    "corrected_t": res.corrected_t, "p_value": res.p_value,
                    "delta_sr": res.delta_sr, "rho": res.rho,
                    "t_pass": res.t_pass, "lord_pass": res.lord_pass,
                    "uplift_pass": res.uplift_pass, "fragility_pass": res.fragility_pass,
                    "collinearity_pass": res.collinearity_pass,
                    "passes": res.passes_corrected})
    if out:
        zs = np.array([o["corrected_t"] for o in out], dtype=float)
        zs = zs[np.isfinite(zs)]
        legs = {k: float(np.mean([o[k] for o in out])) for k in
                ("t_pass", "lord_pass", "uplift_pass", "fragility_pass", "collinearity_pass")}
        print(f"\nmax z = {zs.max():.3f}  (bar {cc.t_min})   median z = {np.median(zs):.3f}")
        print("leg pass rates over these candidates:", {k: f"{v:.0%}" for k, v in legs.items()})
        binding = min(legs, key=legs.get)
        print(f"BINDING leg = {binding} ({legs[binding]:.0%} pass)")
    p = ROOT / args.run / "leg_forensics.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
