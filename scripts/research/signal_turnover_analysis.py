"""Turnover / ensemble capturability analysis for the 101-alpha hunt (S553-cont-72, C part 2).

The full hunt found real gross IC (40/100 FDR-significant) but nearly all cost-blocked; the
cost-wall column says capturable alpha lives in the LOW-turnover constructions. This probe
quantifies that:
  1) horizon sweep — does net Sharpe improve when the lead candidates are held longer?
  2) ensemble — does combining the top low-correlation significant alphas (netting turnover)
     produce a capturable, higher-IR signal?

Research probe (no production code touched). Reuses the harness tiers verbatim.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN slice means on thin days

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from finrl_pro_ds.data.equity_panel_loader import load_sp500_panel  # noqa: E402
from finrl_pro_ds.signals import Gates, SignalSpec  # noqa: E402
from finrl_pro_ds.signals.eval_harness import (  # noqa: E402
    compute_scores,
    tier1_gross_power,
    tier2_capturability,
)
from finrl_pro_ds.signals.library.alphas101 import SIGNALS as ALPHAS  # noqa: E402

SCORECARD = ROOT / "results" / "signal_eval" / "sp500_alpha101_full" / "scorecard.json"
HORIZONS = (5, 10, 21, 42, 63)


def _rowz(x: np.ndarray) -> np.ndarray:
    m = np.nanmean(x, axis=1, keepdims=True)
    s = np.nanstd(x, axis=1, keepdims=True)
    return (x - m) / np.where(s > 0, s, np.nan)


class Ensemble:
    """Equal-weight mean of per-day z-scored, direction-adjusted component signals."""

    def __init__(self, components: list, name: str) -> None:
        self.components = components
        self.spec = SignalSpec(name=name, hypothesis="equal-weight ensemble of significant alphas",
                               family="101alpha", expected_sign=1, neutralization=())

    def compute(self, panel) -> np.ndarray:
        stack = []
        for s in self.components:
            sc = compute_scores(s, panel, s.spec.neutralization) * s.spec.expected_sign
            stack.append(_rowz(sc))
        return np.nanmean(np.stack(stack), axis=0)


def _net(sig, panel, gates, h):
    cap = tier2_capturability(sig, panel, gates, neutralization=sig.spec.neutralization,
                              expected_sign=sig.spec.expected_sign, hold_horizon=h)
    std = cap.by_cost["standard"]
    return std.net_sharpe, std.turnover_ann, cap.by_cost["harsh"].net_sharpe


def main() -> None:
    print("[loading panel: 300 S&P500 names 2015->latest]")
    panel = load_sp500_panel("2015-01-01", max_names=300)
    gates = Gates.from_yaml(ROOT / "configs" / "signal_eval.gates.yaml")
    sc = json.load(open(SCORECARD, encoding="utf-8"))

    def _q(c):
        v = (c.get("deflation") or {}).get("fdr_q")
        return v if isinstance(v, (int, float)) else 1.0

    sig_names = [c["name"] for c in sc["cards"] if _q(c) < 0.05]
    by_name = {s.spec.name: s for s in ALPHAS}
    print(f"[{len(sig_names)} FDR-significant alphas]\n")

    print("=== 1) HORIZON SWEEP — net Sharpe @ standard cost (turnover_ann) ===")
    leads = ["alpha032", "alpha024", "alpha069", "alpha004", "alpha005", "alpha047"]
    for name in leads:
        sig = by_name[name]
        cells = []
        for h in HORIZONS:
            ns, to, _ = _net(sig, panel, gates, h)
            cells.append(f"h{h}: {ns:+.2f} (to {to:.0f})")
        print(f"  {name}: " + "  ".join(cells))

    print("\n=== 2) ENSEMBLE of top significant alphas (by IC-IR) ===")
    ranked = [c["name"] for c in sc["cards"]
              if c["name"] in sig_names and (c.get("gross") or {}).get("by_horizon", {}).get("5")]
    for k in (5, 10, 20):
        comps = [by_name[n] for n in ranked[:k]]
        ens = Ensemble(comps, f"ens_top{k}")
        gp = tier1_gross_power(ens, panel, gates.horizons, primary_horizon=gates.primary_horizon,
                               neutralization=ens.spec.neutralization,
                               expected_sign=ens.spec.expected_sign, min_names=gates.min_names_per_day)
        hp = gp.by_horizon[gates.primary_horizon]
        n5, to5, h5 = _net(ens, panel, gates, 5)
        n21, _, _ = _net(ens, panel, gates, 21)
        print(f"  top{k}: IC-IR={hp.ic_ir:.3f} t={hp.ic_tstat:.1f} breadth={gp.breadth:.2f} "
              f"| net@std h5={n5:+.2f} h21={n21:+.2f} harsh_h5={h5:+.2f} (turnover {to5:.0f})")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    main()
