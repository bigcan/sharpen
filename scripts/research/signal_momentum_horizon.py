"""Horizon-resolved momentum capturability — the cont-77 momentum cut, done at the RIGHT hold.

The first pass (signal_momentum_confirmed.py) tested momentum at h21 and found ~0 net, but the
gross IC shows momentum predicts the 5-DAY forward return (mom_12_1 5d IC-IR +0.066 t=3.4) and
DECAYS to ~0 by 21d. A 12-1 signal is slow-moving (stable rankings) => it can be captured at h5
with LOW turnover. This re-tests at h5/h10/h21 and re-runs the survivorship injection at the hold
where the edge actually lives. Reuses both sibling engines. Research probe.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import signal_lead_distress_filter as R  # noqa: E402
import signal_momentum_confirmed as M  # noqa: E402
from sharpen.signals import Gates  # noqa: E402
from sharpen.signals.eval_harness import tier2_capturability  # noqa: E402

OOS, EVAL = np.datetime64("2021-01-01"), np.datetime64("2015-01-01")
OUT = ROOT / "results" / "signal_eval" / "momentum_confirmed"
HS = (5, 10, 21)


def run_book_h(eff, rets, active, bps, hold):
    return R.run_book(eff, rets, active, np.zeros_like(eff, bool), bps,
                      filter_on=False, daily_stop=False, hold=hold)[0]


def inj_sweep_h(panel, sig, neu, hold, ks=(0, 5, 10, 20)):
    out = {}
    for k in ks:
        cohort = R.synth_cohort(panel.dates, k, seed=20260624 + k) if k else []
        aug, _ = R.augment(panel, cohort, [f"CR{k}_{i}" for i in range(len(cohort))])
        eff = M.eff_of(sig, aug, neu)
        rets = R.daily_rets(aug.close, aug.active)
        out[k] = R.sr(run_book_h(eff, rets, aug.active, 0.0010, hold), np.ones(aug.T, bool))
    return out


def evaluate(panel, neu, gates, label):
    ins = (panel.dates >= EVAL) & (panel.dates < OOS)
    oos = panel.dates >= OOS
    print(f"\n########## {label} (neu={neu}) ##########")
    sigs = {"mom_12_1": M.MomSignal(231, 21, "mom_12_1", neu),
            "mom_6_1":  M.MomSignal(105, 21, "mom_6_1", neu)}
    rep = {"universe": panel.meta.get("universe_def")}
    for nm, sig in sigs.items():
        print(f"\n  [{nm}] horizon sweep — fric / net@10 / net@25 (turnover) | IS@10 OOS@10 OOS@25")
        rep[nm] = {}
        for h in HS:
            cap = tier2_capturability(sig, panel, gates, neutralization=neu, expected_sign=1,
                                      hold_horizon=h)
            std, hsh = cap.by_cost["standard"], cap.by_cost["harsh"]
            eff = M.eff_of(sig, panel, neu)
            rets = R.daily_rets(panel.close, panel.active)
            p10 = run_book_h(eff, rets, panel.active, 0.0010, h)
            p25 = run_book_h(eff, rets, panel.active, 0.0025, h)
            rep[nm][f"h{h}"] = {"fric": cap.frictionless_sharpe, "net10": std.net_sharpe,
                                "net25": hsh.net_sharpe, "turnover": std.turnover_ann,
                                "is10": R.sr(p10, ins), "oos10": R.sr(p10, oos), "oos25": R.sr(p25, oos)}
            r = rep[nm][f"h{h}"]
            print(f"    h{h:<2}: {r['fric']:>5.2f} / {r['net10']:>5.2f} / {r['net25']:>5.2f} "
                  f"(to {r['turnover']:>4.0f}) | {r['is10']:>5.2f} {r['oos10']:>6.2f} {r['oos25']:>6.2f}")
        # survivorship injection at h5 (where the edge lives)
        sw = inj_sweep_h(panel, sig, neu, 5)
        rep[nm]["injection_h5"] = sw
        print("    injection @h5 (net@10bps, k=0/5/10/20): "
              + " ".join(f"{sw[k]:>5.2f}" for k in (0, 5, 10, 20)))
    return rep


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gates = Gates.from_yaml(ROOT / "configs" / "signal_eval.gates.yaml")
    rep = {"A_current300": evaluate(R.load_panel(), ("winsor", "zscore", "sector"), gates, "A) CURRENT-300"),
           "B_pit300": evaluate(M.build_pit300(), ("winsor", "zscore", "size"), gates, "B) PIT-300")}
    with open(OUT / "horizon_results.json", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, default=float)
    print(f"\n[written] {OUT / 'horizon_results.json'}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    main()
