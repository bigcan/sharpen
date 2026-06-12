"""Holdout cadence grade for the options-VRP linear core (V2-03 / V7-02 NOW item).

Closes the in-sample cadence multiplicity flagged by the Tier-2 tail audit: the
turnover/cadence conclusion in ``results/options_vrp/verdict.json`` ("14-30d GO,
21d primary") was both *picked* AND *graded* on the full 2021-2026 sample. Here we

  1. PICK the roll cadence on a 2021-2023 TRAIN slice (best train portfolio net
     Sharpe among configs that clear the phase1_linear gate), and
  2. GRADE that chosen cadence on an untouched 2024-2026 HOLDOUT slice.

If the train-picked cadence still clears the gate on the holdout, the cadence choice
is validated out-of-sample (not a multiplicity artifact). The pre-registered 21d is
graded alongside for reference.

Eval-only / CPU. Reuses the (fee-corrected) ``simulate_asset`` + metric helpers +
yaml-loaded gates from the gate script — no re-implementation, so the numbers are on
the identical convention as the headline verdict. Reads the cached Deribit parquets.

Run:
    python scripts/research/options_vrp_holdout_cadence.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finrl_pro_ds.crypto.data import deribit_options_loader as dol  # noqa: E402
from finrl_pro_ds.crypto.data import options_array_builder as oab  # noqa: E402

# Load the gate script by path (scripts/ is not a package) and reuse its engine.
_spec = importlib.util.spec_from_file_location(
    "ovf_gate", ROOT / "scripts" / "research" / "options_vrp_falsification.py")
ovf = importlib.util.module_from_spec(_spec)
sys.modules["ovf_gate"] = ovf
_spec.loader.exec_module(ovf)

RESULTS_DIR = Path("results/options_vrp")
TRAIN_MAX_YEAR = 2023   # TRAIN = 2021..2023 (pick the cadence here)
HOLDOUT_MIN_YEAR = 2024  # HOLDOUT = 2024..2026 (grade the chosen cadence here)
PREMIUM_FRAC = 0.05      # shipped sizing; the cadence axis is what we validate OOS
ROLL_GRID = (7, 14, 21, 30)
SHIPPED_ROLL = 21


def _metrics(spot_ary, iv_ary, fund_ary, assets, mask, cfg) -> dict:
    """Per-asset + equal-weight portfolio net metrics over the masked (contiguous)
    date slice, IDENTICAL convention to the gate's run() (avg of per-asset daily PnL).
    The portfolio is the pre-registered gate object; the per-asset BTC number is the
    headline sleeve (the portfolio is dragged by ETH, the audit's weak asset)."""
    pnls, per_asset = [], {}
    for i, a in enumerate(assets):
        net = ovf.simulate_asset(spot_ary[mask, i], iv_ary[mask, i], fund_ary[mask, i], cfg)
        pnls.append(net["daily_pnl"])
        ret = ovf._returns_from_pnl(net["daily_pnl"], net["eq_curve"])
        per_asset[a] = {"net_sharpe": ovf._sharpe(ret),
                        "net_pf": ovf._profit_factor(net["daily_pnl"]),
                        "net_max_dd": ovf._max_drawdown(net["eq_curve"])}
    port_pnl = np.nanmean(np.column_stack(pnls), axis=1)
    port_eq = np.cumsum(port_pnl) + cfg.initial_capital
    port_ret = ovf._returns_from_pnl(port_pnl, port_eq)
    return {
        "net_sharpe": ovf._sharpe(port_ret),
        "net_pf": ovf._profit_factor(port_pnl),
        "net_max_dd": ovf._max_drawdown(port_eq),
        "net_total_return": float(port_eq[-1] / cfg.initial_capital - 1.0),
        "n_bars": int(mask.sum()),
        "per_asset": per_asset,
    }


def _is_go(m: dict, gates: dict) -> bool:
    return m["net_sharpe"] >= gates["min_net_sharpe"] and m["net_pf"] >= gates["min_net_pf"]


def main() -> None:
    gates = ovf.load_gates()
    raw = dol.load({"universe": {"assets": ["BTC", "ETH"]}})
    panels = oab.build_panels(raw)
    years = panels.dates.year.to_numpy()
    spot, iv, fund = panels.spot_ary, panels.iv_ary, panels.funding_ary
    assets = panels.assets

    train_mask = years <= TRAIN_MAX_YEAR
    hold_mask = years >= HOLDOUT_MIN_YEAR

    # Full cadence sweep on BOTH slices (transparency: is the ranking stable?).
    grid = []
    for roll in ROLL_GRID:
        cfg = ovf.SimConfig(premium_frac=PREMIUM_FRAC, roll_days=roll)
        tr = _metrics(spot, iv, fund, assets, train_mask, cfg)
        ho = _metrics(spot, iv, fund, assets, hold_mask, cfg)
        grid.append({
            "roll_days": roll,
            "train": tr, "train_go": _is_go(tr, gates),
            "holdout": ho, "holdout_go": _is_go(ho, gates),
        })

    # PICK on train: best train net Sharpe among configs that clear the gate on TRAIN.
    train_go = [g for g in grid if g["train_go"]]
    picked = max(train_go, key=lambda g: g["train"]["net_sharpe"]) if train_go else \
        max(grid, key=lambda g: g["train"]["net_sharpe"])
    shipped = next(g for g in grid if g["roll_days"] == SHIPPED_ROLL)

    out = {
        "purpose": "V2-03/V7-02 holdout cadence grade — pick on 2021-23, grade on 2024-26",
        "premium_frac": PREMIUM_FRAC,
        "gates": {"min_net_sharpe": gates["min_net_sharpe"], "min_net_pf": gates["min_net_pf"]},
        "train_period": [str(panels.dates[train_mask][0].date()),
                         str(panels.dates[train_mask][-1].date())],
        "holdout_period": [str(panels.dates[hold_mask][0].date()),
                           str(panels.dates[hold_mask][-1].date())],
        "grid": grid,
        "picked_on_train": {
            "roll_days": picked["roll_days"],
            "selection_rule": "max train net Sharpe among train-GO configs",
            "train_net_sharpe": picked["train"]["net_sharpe"],
            "holdout_net_sharpe": picked["holdout"]["net_sharpe"],
            "holdout_net_pf": picked["holdout"]["net_pf"],
            "holdout_clears_gate": picked["holdout_go"],
            "holdout_btc_net_sharpe": picked["holdout"]["per_asset"]["BTC"]["net_sharpe"],
            "holdout_btc_net_pf": picked["holdout"]["per_asset"]["BTC"]["net_pf"],
            "holdout_btc_clears_gate": _is_go(picked["holdout"]["per_asset"]["BTC"], gates),
        },
        "shipped_21d": {
            "train_net_sharpe": shipped["train"]["net_sharpe"],
            "holdout_net_sharpe": shipped["holdout"]["net_sharpe"],
            "holdout_net_pf": shipped["holdout"]["net_pf"],
            "holdout_clears_gate": shipped["holdout_go"],
            "holdout_btc_net_sharpe": shipped["holdout"]["per_asset"]["BTC"]["net_sharpe"],
            "holdout_btc_net_pf": shipped["holdout"]["per_asset"]["BTC"]["net_pf"],
            "holdout_btc_clears_gate": _is_go(shipped["holdout"]["per_asset"]["BTC"], gates),
        },
        # The headline OOS verdict: did the train-picked cadence survive the holdout?
        "oos_cadence_validated": bool(picked["holdout_go"]),
        "oos_cadence_validated_btc": _is_go(picked["holdout"]["per_asset"]["BTC"], gates),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "holdout_cadence.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")

    # Console summary
    print(f"TRAIN {out['train_period'][0]}..{out['train_period'][1]} "
          f"({grid[0]['train']['n_bars']} bars)  |  "
          f"HOLDOUT {out['holdout_period'][0]}..{out['holdout_period'][1]} "
          f"({grid[0]['holdout']['n_bars']} bars)  |  premium_frac={PREMIUM_FRAC}")
    print(f"gate: net Sharpe >= {gates['min_net_sharpe']}, net PF >= {gates['min_net_pf']}\n")
    print(f"{'roll':>5} | {'PORTFOLIO: trSh':>15} {'hoSh':>6} {'GO':>3} | "
          f"{'BTC: trSh':>10} {'hoSh':>6} | {'ETH hoSh':>8}")
    for g in grid:
        btc_tr = g["train"]["per_asset"]["BTC"]["net_sharpe"]
        btc_ho = g["holdout"]["per_asset"]["BTC"]["net_sharpe"]
        eth_ho = g["holdout"]["per_asset"]["ETH"]["net_sharpe"]
        print(f"{g['roll_days']:>4}d | {g['train']['net_sharpe']:>15.2f} "
              f"{g['holdout']['net_sharpe']:>6.2f} {'Y' if g['holdout_go'] else 'n':>3} | "
              f"{btc_tr:>10.2f} {btc_ho:>6.2f} | {eth_ho:>8.2f}")
    p = out["picked_on_train"]
    print(f"\nPICKED on train: {p['roll_days']}d (train Sharpe {p['train_net_sharpe']:.2f}) "
          f"-> holdout PORTFOLIO Sharpe {p['holdout_net_sharpe']:.2f} / PF {p['holdout_net_pf']:.2f} "
          f"-> {'CLEARS' if p['holdout_clears_gate'] else 'FAILS'} the gate OOS")
    print(f"   ...same cadence, BTC-only holdout Sharpe {p['holdout_btc_net_sharpe']:.2f} "
          f"/ PF {p['holdout_btc_net_pf']:.2f} -> "
          f"{'CLEARS' if p['holdout_btc_clears_gate'] else 'FAILS'}")
    s = out["shipped_21d"]
    print(f"shipped 21d: portfolio train {s['train_net_sharpe']:.2f} -> holdout "
          f"{s['holdout_net_sharpe']:.2f}/PF {s['holdout_net_pf']:.2f} "
          f"({'CLEARS' if s['holdout_clears_gate'] else 'FAILS'}); "
          f"BTC-only holdout {s['holdout_btc_net_sharpe']:.2f}/PF {s['holdout_btc_net_pf']:.2f} "
          f"({'CLEARS' if s['holdout_btc_clears_gate'] else 'FAILS'})")
    print(f"\nOOS cadence validated (portfolio): {out['oos_cadence_validated']}  |  "
          f"BTC-only: {out['oos_cadence_validated_btc']}")
    print(f"Wrote {RESULTS_DIR / 'holdout_cadence.json'}")


if __name__ == "__main__":
    main()
