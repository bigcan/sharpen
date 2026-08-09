"""Did the pre-registered hypotheses ever reach the binding holdout test? (S553-cont-153)

WHY. The us_equity tick reported ``mined=True fdr_tests=8 promising=0`` and was recorded as "the first
clean negative on a substrate that could actually have detected something". But the tick's own log line
reads ``prereg_only: 104 of 104 train-survivors are evolved offspring``. Under
``evolve``'s corrected path that arithmetic is decisive:

    train_passers = [c for c in ranked if _passes_cheap_prefilter(c.result, corrected_cfg)]   # -> 104
    train_passers = [c for c in train_passers if c.formula in prereg]                         # -> 0

i.e. ZERO of the eight pre-registered seeds cleared the TRAIN cheap pre-filter, the holdout loop
iterated over an empty list, and ``corrected_contract_fitness`` — the one test the substrate's whole
power argument is about — never executed on a single pre-registered hypothesis. `promising=0` is then
true by vacuity, not by evidence, and the LORD++ account was still charged.

That is the same SHAPE as the defect the corrected contract was built to remove (audit 2026-07-29: the
holdout gate never executed in production because train re-applied the final gate). It reappears
through a different door: the cheap guards are ECONOMIC-SIZE guards (uplift >= 0.10 dSR against a base
book that now WINS at SR +0.519), applied on train, to the very specs that are the charging unit.

WHAT THIS SCRIPT DECIDES. Two pre-registered reads, both on the eight seeds already in the ledger:

  R1 (confirm the mechanism) — score each seed on the TRAIN split exactly as ``evolve.score`` does and
     print all five cheap-prefilter legs. Expected under the hypothesis: 0/8 pass, and the binding leg
     is ``uplift``.
  R2 (does it COST anything) — run each seed through the SHIPPED ``corrected_contract_fitness`` on the
     embargoed holdout regardless of its train verdict, at the tick's own LORD++ level, and print all
     five corrected legs. This is the test the run never ran.
       * if 0/8 would have passed the holdout, the pre-filter cost nothing here and the negative
         stands on its merits (though not for the reason recorded);
       * if any would have passed, the pre-filter is discarding hypotheses the substrate was powered
         to adjudicate, and the run's headline is wrong.

R2 is DIAGNOSTIC ONLY. It promotes nothing, writes no ledger row, charges no wealth, and moves no
gate: the LORD++ level is read, not advanced. Reporting it as a discovery would be exactly the
file-drawer laundering the funnel exists to prevent — it is a measurement of what the gate WOULD have
said, run after the fact, and any candidate it flags has to be re-mined through the real path.

Usage:
    python scripts/research/crucible_prereg_prefilter_forensics.py \
        --run results/crucible_orchestrator/real --substrate us_equity
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

from finrl_pro_ds.crucible.corrected_contract import (  # noqa: E402
    CorrectedConfig,
    corrected_contract_fitness,
    fresh_lord_level,
)
from finrl_pro_ds.signals.generation.config import (  # noqa: E402
    load_generation_config,
    load_generation_meta,
)
from finrl_pro_ds.signals.generation.grammar import node_count, parse  # noqa: E402
from finrl_pro_ds.signals.generation.evolve import (  # noqa: E402
    _candidate_returns,
    _panel_market_returns,
    _passes_cheap_prefilter,
    _split,
)
from finrl_pro_ds.signals.generation.fitness import combination_fitness  # noqa: E402

log = logging.getLogger("prereg_forensics")


def build_us_equity(cfg, ek, meta):
    """Rebuild the panel + base book exactly as `crucible_orchestrator.prepare()` did."""
    from finrl_pro_ds.crucible.data.us_equity_panel import build_us_equity_panel
    from finrl_pro_ds.signals.generation.base_sleeves import us_equity_base_sleeves
    panel = build_us_equity_panel()
    base_hold = meta.get("base_hold_horizon") or ek["hold_horizon"]
    base, _ = us_equity_base_sleeves(panel, hold_horizon=int(base_hold),
                                     cost_bps=ek["cost_bps"], return_components=True)
    return panel, base


def _panel_ts(panel) -> np.ndarray:
    """Identical to `crucible_orchestrator._panel_ts` — the scorer wants epoch seconds as float."""
    return panel.dates.astype("datetime64[s]").astype(np.int64).astype(np.float64)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/crucible_orchestrator/real")
    ap.add_argument("--substrate", default="us_equity")
    ap.add_argument("--config", default="configs/us_equity_signal_eval.gates.yaml")
    ap.add_argument("--corrected-config", default="configs/us_equity_corrected_contract.gates.yaml")
    ap.add_argument("--out", default="results/crucible_prereg_forensics")
    args = ap.parse_args()

    run = ROOT / args.run
    cfg, ek = load_generation_config(ROOT / args.config)
    meta = load_generation_meta(ROOT / args.config)
    cc = CorrectedConfig.from_yaml(ROOT / args.corrected_config)

    # --- the pre-registered seeds, read from the ledger (the run's own record) ---------------------
    con = sqlite3.connect(run / "trial_ledger.db")
    con.row_factory = sqlite3.Row
    seeds = [dict(r) for r in con.execute(
        "SELECT candidate_hash, formula, candidate_type, verdict, family FROM trial_ledger "
        "WHERE verdict = 'SCORED_NOT_SELECTED' AND formula IS NOT NULL ORDER BY candidate_hash")]
    log.info("read %d pre-registered seeds from %s", len(seeds), run / "trial_ledger.db")
    if not seeds:
        log.error("no SCORED_NOT_SELECTED rows — nothing to adjudicate")
        return 1

    panel, base = build_us_equity(cfg, ek, meta)
    ts = _panel_ts(panel)
    train, hold = _split(panel, ek["holdout_frac"], ek["holdout_embargo"])
    n_train, n_hold = train.T, hold.T
    log.info("panel T=%d  train=%d  holdout=%d bars (%.2f calendar yr)",
             panel.T, n_train, n_hold, n_hold / float(cfg.periods_per_year))

    base_tr = {k: np.asarray(v)[:n_train] for k, v in base.items()}
    ts_tr = ts[:n_train]
    base_ho = {k: np.asarray(v)[panel.T - n_hold:] for k, v in base.items()}
    ts_ho = ts[panel.T - n_hold:]
    mkt_full = _panel_market_returns(panel)
    mkt_ho = None if mkt_full is None else mkt_full[panel.T - n_hold:]
    lord_level = fresh_lord_level(cc)
    log.info("LORD++ level (fresh account, read-only) = %.6g ; market_returns wired = %s",
             lord_level, mkt_full is not None)

    rows: list[dict] = []
    # gen_n_eff for the train fitness: the run's own file-drawer count. The cheap guards this script
    # reads (uplift / median / frac_positive / base_corr / not_degenerate) are all independent of the
    # deflation term, so this only affects the reported `dsr_aug` diagnostic, not any leg below.
    gen_n_eff = float(len(seeds))
    for s in seeds:
        f = s["formula"]
        rec: dict = {"candidate_hash": s["candidate_hash"], "formula": f}
        # --- R1: TRAIN cheap pre-filter ------------------------------------------------------------
        try:
            cr = _candidate_returns(f, train, hold_horizon=ek["hold_horizon"],
                                    cost_bps=ek["cost_bps"], min_names=ek["ls_min_names"])
        except Exception as exc:                                       # noqa: BLE001
            rec["train"] = {"error": f"eval raised: {exc!r}"}
            cr = None
        if cr is None:
            rec.setdefault("train", {"error": "degenerate (all-NaN)"})
        else:
            cand_tr, turn_tr = cr
            res = combination_fitness(cand_tr, base_tr, ts_tr, cfg, gen_n_eff=gen_n_eff,
                                      turnover_ann=turn_tr, n_nodes=node_count(parse(f)))
            rec["train"] = {
                "uplift_delta_sr": float(res.delta_sr_oos),
                "uplift_pass": bool(np.isfinite(res.delta_sr_oos)
                                    and res.delta_sr_oos >= cc.uplift_min),
                "delta_median": float(res.delta_sr_median),
                "median_pass": bool(np.isfinite(res.delta_sr_median)
                                    and res.delta_sr_median >= cc.delta_median_min),
                "frac_paths_positive": float(res.frac_paths_positive),
                "frac_pass": bool(np.isfinite(res.frac_paths_positive)
                                  and res.frac_paths_positive >= cc.frac_positive_min),
                "max_base_corr": float(res.max_base_corr_obs),
                "corr_pass": bool(np.isfinite(res.max_base_corr_obs)
                                  and res.max_base_corr_obs <= cc.max_base_corr),
                "not_degenerate": bool(res.not_degenerate),
                "turnover_ann": float(turn_tr),
                "prefilter_pass": bool(_passes_cheap_prefilter(res, cc)),
            }
        # --- R2: the HOLDOUT test the run never ran ------------------------------------------------
        # Full-panel scoring, holdout rows sliced off — identical to evolve's holdout path, so
        # trailing-window operators warm up from causal train history (F2).
        try:
            full = _candidate_returns(f, panel, hold_horizon=ek["hold_horizon"],
                                      cost_bps=ek["cost_bps"], min_names=ek["ls_min_names"])
        except Exception as exc:                                       # noqa: BLE001
            rec["holdout"] = {"error": f"eval raised: {exc!r}"}
            full = None
        if full is None:
            rec.setdefault("holdout", {"error": "degenerate (all-NaN)"})
        else:
            cand_ho = full[0][panel.T - n_hold:]
            try:
                cres = corrected_contract_fitness(cand_ho, base_ho, ts_ho, cfg, cc,
                                                  lord_level=lord_level, market_returns=mkt_ho)
            except Exception as exc:                                   # noqa: BLE001
                rec["holdout"] = {"error": f"fitness raised: {exc!r}"}
            else:
                rec["holdout"] = {
                    "delta_sr": float(cres.delta_sr),
                    "corrected_t": float(cres.corrected_t),
                    "p_value": float(cres.p_value),
                    "n_eff": float(cres.n_eff),
                    "n_bars": int(cres.n_bars),
                    "legs": {"t": bool(cres.t_pass), "lord": bool(cres.lord_pass),
                             "uplift": bool(cres.uplift_pass),
                             "fragility": bool(cres.fragility_pass),
                             "collinearity": bool(cres.collinearity_pass)},
                    "passes_corrected": bool(cres.passes_corrected),
                }
        rows.append(rec)

    # --- report ------------------------------------------------------------------------------------
    n_pref = sum(1 for r in rows if r.get("train", {}).get("prefilter_pass"))
    n_hold_pass = sum(1 for r in rows if r.get("holdout", {}).get("passes_corrected"))
    print("\n=== R1: TRAIN cheap pre-filter (the gate that actually decided this run) ===")
    print(f"{'hash':<14}{'uplift':>9}{'med':>8}{'frac':>7}{'corr':>7}{'ndeg':>6}  {'PASS':>5}")
    for r in rows:
        t = r.get("train", {})
        if "error" in t:
            print(f"{r['candidate_hash']:<14}  {t['error']}")
            continue
        print(f"{r['candidate_hash']:<14}{t['uplift_delta_sr']:>9.3f}{t['delta_median']:>8.3f}"
              f"{t['frac_paths_positive']:>7.2f}{t['max_base_corr']:>7.2f}"
              f"{str(t['not_degenerate']):>6}  {str(t['prefilter_pass']):>5}")
    print(f"-> {n_pref}/{len(rows)} pre-registered seeds cleared the train pre-filter "
          f"(uplift floor {cc.uplift_min}, median {cc.delta_median_min}, "
          f"frac {cc.frac_positive_min}, corr {cc.max_base_corr})")

    print("\n=== R2: the CORRECTED holdout test, run on all 8 regardless (DIAGNOSTIC ONLY) ===")
    print(f"{'hash':<14}{'dSR':>8}{'t':>8}{'p':>10}   legs(t/lord/upl/frag/coll)  PASS")
    for r in rows:
        h = r.get("holdout", {})
        if "error" in h:
            print(f"{r['candidate_hash']:<14}  {h['error']}")
            continue
        lg = h["legs"]
        flags = "/".join("Y" if lg[k] else "n"
                         for k in ("t", "lord", "uplift", "fragility", "collinearity"))
        print(f"{r['candidate_hash']:<14}{h['delta_sr']:>8.3f}{h['corrected_t']:>8.2f}"
              f"{h['p_value']:>10.4f}   {flags:<24} {h['passes_corrected']}")
    print(f"-> {n_hold_pass}/{len(rows)} would have passed the corrected holdout gate "
          f"(t_min {cc.t_min}, lord level {lord_level:.6g})")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "substrate": args.substrate, "run": str(run),
        "panel_T": int(panel.T), "train_bars": int(n_train), "holdout_bars": int(n_hold),
        "holdout_calendar_years": n_hold / float(cfg.periods_per_year),
        "gates": {"uplift_min": cc.uplift_min, "delta_median_min": cc.delta_median_min,
                  "frac_positive_min": cc.frac_positive_min, "max_base_corr": cc.max_base_corr,
                  "t_min": cc.t_min, "max_market_beta": cc.max_market_beta,
                  "lord_level_read": lord_level},
        "n_prefilter_pass": n_pref, "n_would_pass_holdout": n_hold_pass,
        "seeds": rows,
    }
    (out / f"{args.substrate}_prereg_forensics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out / f'{args.substrate}_prereg_forensics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
