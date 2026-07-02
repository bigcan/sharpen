"""Run the prop-firm challenge simulator on the TAILWIND books — Fable review item 1.

Answers the challenge objective (P(hit target before breaching DD/daily limit), NOT Sharpe):
  * optimal sizing (vol multiplier, bank-and-derisk) per firm;
  * does dropping BAB (momentum-only) beat the combined book for P(pass)? (review predicts yes);
  * which limit binds — the daily loss limit or the max total DD? (review predicts daily);
  * sensitivity to the intraday-MAE proxy (a daily book held overnight) + a crash-block overlay.

Books (same `mom`/`pf` basis as the R1 audit, cached prices, no network):
  momentum-only = the challenge book the review recommends (drop the BAB carry-bleed);
  momentum+BAB  = the own-capital book, for the head-to-head.

Emits results/tailwind_v1/challenge_sim.json.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tailwind_v1"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_tailwind_book as at             # noqa: E402  (build_defensive_net)
import portfolio_frontier as pf              # noqa: E402

from finrl_pro_ds.prop.challenge_simulator import (  # noqa: E402
    FirmRules, SizingPolicy, evaluate, sweep,
)


def _firm(cfg: dict) -> FirmRules:
    return FirmRules(
        name=cfg["name"], profit_target=float(cfg["profit_target"]),
        max_total_dd=float(cfg["max_total_dd"]), daily_loss_limit=float(cfg["daily_loss_limit"]),
        max_days=(None if cfg.get("max_days") is None else int(cfg["max_days"])),
        min_trading_days=int(cfg.get("min_trading_days", 0)),
        dd_mode=str(cfg.get("dd_mode", "static")))


def _crash_returns(series):
    """Daily returns in the two worst modern crash windows (GFC + COVID) — the stress overlay."""
    import pandas as pd
    gfc = series.loc["2008-06-01":"2009-06-30"]
    covid = series.loc["2020-02-01":"2020-05-31"]
    return pd.concat([gfc, covid]).dropna().to_numpy()


def main():
    rules = yaml.safe_load((ROOT / "configs" / "prop_firm_rules.yaml").read_text(encoding="utf-8"))
    sim = rules["simulation"]
    grid = rules["policy_grid"]
    base_vol = float(sim["base_vol_ann"])
    n_paths = int(sim["n_paths"])
    block = int(sim["block_days"])
    cap = int(sim["no_deadline_cap"])
    seed = int(sim["seed"])
    vms = [float(x) for x in grid["vol_multipliers"]]
    banks = [None if x is None else float(x) for x in grid["bank_thresholds"]]
    derisk = float(grid["derisk_multiplier"])
    mae_sens = [float(x) for x in grid["intraday_mae_sensitivity"]]

    # ---- books (dated Series so crash windows can be sliced) ----
    mom_net = pf.build_momentum_net()
    def_net = at.build_defensive_net()
    combined, _, _ = pf.risk_parity([mom_net, def_net])
    books = {"momentum_only": mom_net, "momentum_plus_bab": combined}

    ev_kw = dict(n_paths=n_paths, block=block, base_vol_ann=base_vol, seed=seed,
                 no_deadline_cap=cap)

    out = {"note": "P(pass) Monte-Carlo — Fable review item 1. Model outputs, not guarantees; "
                   "prop rules in configs/prop_firm_rules.yaml MUST be verified before a real attempt.",
           "base_vol_ann": base_vol, "n_paths": n_paths, "block_days": block,
           "firms": {}}

    for fkey, fcfg in rules["firms"].items():
        firm = _firm(fcfg)
        fout = {"rules": {k: fcfg[k] for k in ("profit_target", "max_total_dd",
                          "daily_loss_limit", "max_days", "dd_mode")}, "books": {}}
        for bkey, series in books.items():
            r = series.to_numpy()
            sw = sweep(r, firm, vol_multipliers=vms, bank_thresholds=banks,
                       derisk_multiplier=derisk, **ev_kw)
            best = sw.best("p_pass")
            fout["books"][bkey] = {
                "best": {k: best[k] for k in ("p_pass", "vol_multiplier", "bank_threshold",
                         "effective_vol_ann", "e_days_to_pass", "median_days_to_pass",
                         "p_dd_breach", "p_daily_breach", "p_timeout", "binding_constraint")},
                "grid": [{k: row[k] for k in ("vol_multiplier", "bank_threshold", "p_pass",
                          "p_daily_breach", "p_dd_breach", "p_timeout", "binding_constraint")}
                         for row in sw.rows],
            }
        # ---- sensitivity (on momentum-only best policy): intraday MAE proxy + crash overlay ----
        mo_best = fout["books"]["momentum_only"]["best"]
        best_pol = dict(vol_multiplier=mo_best["vol_multiplier"],
                        bank_threshold=mo_best["bank_threshold"], derisk_multiplier=derisk)
        crash = _crash_returns(mom_net)
        sens = {}
        for mae in mae_sens:
            e = evaluate(mom_net.to_numpy(), firm,
                         SizingPolicy(intraday_mae_mult=mae, **best_pol), **ev_kw)
            sens[f"intraday_mae_{mae}"] = {"p_pass": e["p_pass"], "p_daily_breach": e["p_daily_breach"],
                                           "binding_constraint": e["binding_constraint"]}
        e_crash = evaluate(mom_net.to_numpy(), firm, SizingPolicy(**best_pol),
                           crash_returns=crash, crash_block_prob=0.15, **ev_kw)
        sens["crash_overlay_p0.15"] = {"p_pass": e_crash["p_pass"],
                                       "p_daily_breach": e_crash["p_daily_breach"],
                                       "p_dd_breach": e_crash["p_dd_breach"]}
        fout["sensitivity_momentum_only_best"] = sens
        out["firms"][fkey] = fout

    (OUT / "challenge_sim.json").write_text(json.dumps(out, indent=2, default=str))

    # ---- console ----
    print("=" * 82)
    print("PROP-FIRM CHALLENGE SIMULATOR — TAILWIND books (Fable review item 1)")
    print(f"base vol {base_vol:.0%}  ·  {n_paths} paths  ·  block {block}d  ·  no-deadline cap {cap}d")
    print("=" * 82)
    for fkey, fout in out["firms"].items():
        r = fout["rules"]
        print(f"\n### {fkey}  (target {r['profit_target']:.0%}, maxDD {r['max_total_dd']:.0%} "
              f"{r['dd_mode']}, daily {r['daily_loss_limit']:.0%}, "
              f"{'no deadline' if r['max_days'] is None else str(r['max_days'])+'d'})")
        for bkey, b in fout["books"].items():
            be = b["best"]
            print(f"  {bkey:18s} best P(pass)={be['p_pass']:.3f} @ vol×{be['vol_multiplier']} "
                  f"({be['effective_vol_ann']:.0%}) bank={be['bank_threshold']}  "
                  f"E[days]={be['e_days_to_pass']} binds={be['binding_constraint']}  "
                  f"(daily {be['p_daily_breach']:.2f}/DD {be['p_dd_breach']:.2f}/TO {be['p_timeout']:.2f})")
        s = fout["sensitivity_momentum_only_best"]
        print("  sensitivity(mom-only best): "
              + "  ".join(f"{k}=P{v['p_pass']:.2f}" for k, v in s.items()))
    print("=" * 82)
    print(f"wrote {OUT / 'challenge_sim.json'}")
    return out


if __name__ == "__main__":
    main()
