"""FALSIFICATION GATE — does the 101-alpha cross-sectional IC survive on MEGA-CAPS?

XLG = Invesco S&P 500 Top 50 (cap-weighted mega-caps). The signal-eval program found real
gross IC (40/100 FDR-significant) on a BROAD 300-name S&P universe, with a capturable lead at
10-21d hold (alpha024/alpha032 + a top-5 ensemble). But that edge was measured on a broad
universe with wide cross-sectional dispersion. Mega-caps are FAR more efficient (less
dispersion) — the prior price-momentum XLG probe (S553-cont-70) already found a momentum tilt
of the top-50 cannot beat XLG (beta, not alpha).

GATE QUESTION: does the cross-sectional IC + capturability survive when the universe is
restricted to the most efficient mega-cap tier (top-50, top-100 by market cap)?
  - If it COLLAPSES on the top-50 -> a cross-sectional alpha tilt cannot beat XLG -> STOP.
  - If it SURVIVES -> proceed to strategy design (Researcher -> Architect).

Reuses the committed harness (evaluate_batch / eval_harness tiers) + equity_panel_loader
VERBATIM. Universe = current S&P 500 ranked by market cap (xlg_megacap_universe.py cache);
survivorship-LEANING => UPPER BOUND (a NO-GO here is decisive; a GO needs clean re-validation).

Research probe (no production code touched).
"""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sharpen.data.equity_panel_loader import load_sp500_panel  # noqa: E402
from sharpen.signals import (  # noqa: E402
    Gates,
    Multiplicity,
    SignalSpec,
    evaluate_batch,
    load_multiplicity_gates,
    write_scorecard,
)
from sharpen.signals.eval_harness import (  # noqa: E402
    compute_scores,
    tier1_gross_power,
    tier2_capturability,
    tier4_deflation,
)
from sharpen.signals.features import Panel  # noqa: E402
from sharpen.signals.library.alphas101 import SIGNALS as ALPHAS  # noqa: E402

MCAP = ROOT / "data" / "raw" / "equity_panel" / "sp500_mktcap_rank.csv"
OUT = ROOT / "results" / "signal_eval" / "xlg_megacap_gate"
START = "2015-01-01"
HORIZONS = (5, 10, 21, 42)
# Frozen lead set discovered on the broad-300 run (top-5 by DSR/IC-IR) — the TRANSFER test.
FROZEN5 = ["alpha069", "alpha032", "alpha004", "alpha043", "alpha038"]
# The low-turnover capturable leads from the turnover analysis.
LEADS = ["alpha032", "alpha024", "alpha069", "alpha004", "alpha047"]
BY_NAME = {s.spec.name: s for s in ALPHAS}


def _rowz(x: np.ndarray) -> np.ndarray:
    m = np.nanmean(x, axis=1, keepdims=True)
    s = np.nanstd(x, axis=1, keepdims=True)
    return (x - m) / np.where(s > 0, s, np.nan)


class Ensemble:
    """Equal-weight mean of per-day z-scored, direction-adjusted component signals."""

    def __init__(self, components: list, name: str) -> None:
        self.components = components
        self.spec = SignalSpec(name=name, hypothesis="equal-weight ensemble",
                               family="101alpha", expected_sign=1, neutralization=())

    def compute(self, panel) -> np.ndarray:
        stack = [_rowz(compute_scores(s, panel, s.spec.neutralization) * s.spec.expected_sign)
                 for s in self.components]
        return np.nanmean(np.stack(stack), axis=0)


def subset_cols(p: Panel, k: int) -> Panel:
    """Top-k columns (== top-k by market cap, since tickers are passed mcap-sorted)."""
    return Panel(p.dates, p.tickers[:k], p.open[:, :k], p.high[:, :k], p.low[:, :k],
                 p.close[:, :k], p.volume[:, :k], p.active[:, :k], p.adv_usd[:, :k],
                 p.sector_id[:k], {**p.meta, "n_universe": k,
                                   "universe_def": f"top-{k} S&P500 by market cap"})


def tier1_sweep(panel: Panel, gates: Gates, multiplicity: Multiplicity | None = None) -> dict:
    """Tier-1 gross IC + Tier-4 deflation for ALL alphas on this universe (bootstrap off
    for speed; FDR falls back to the normal-approx one-sided p).

    ``multiplicity`` (U5) carries the count for the WHOLE sweep. This function is called once
    per nested universe with the identical alpha list, so deflating each call against
    ``len(ALPHAS)`` counted the same search four times over as four independent searches.
    """
    gp = {}
    for s in ALPHAS:
        sc = compute_scores(s, panel, s.spec.neutralization)
        gp[s.spec.name] = tier1_gross_power(
            s, panel, gates.horizons, primary_horizon=gates.primary_horizon,
            neutralization=s.spec.neutralization, expected_sign=s.spec.expected_sign,
            scores=sc, min_names=gates.min_names_per_day, bootstrap=False)
    defl = tier4_deflation(gp, None, multiplicity)
    ph = gates.primary_horizon
    rows = []
    for n, g in gp.items():
        hp = g.by_horizon[ph]
        rows.append((n, hp.ic_ir, hp.ic_tstat, defl[n].fdr_q, defl[n].dsr))
    rows.sort(key=lambda r: (r[1] if np.isfinite(r[1]) else -1), reverse=True)
    irs = [r[1] for r in rows if np.isfinite(r[1])]
    n_fdr = sum(1 for r in rows if np.isfinite(r[3]) and r[3] < 0.05)
    return {"rows": rows, "n_fdr_sig": n_fdr, "median_ic_ir": float(np.median(irs)),
            "max_ic_ir": float(np.max(irs)), "ranked_names": [r[0] for r in rows]}


def horizon_sweep(panel: Panel, gates: Gates, named_signals: list[tuple[str, object]]) -> dict:
    """net Sharpe @ standard & harsh cost across hold horizons for each named signal."""
    out = {}
    for label, sig in named_signals:
        cells = {}
        for h in HORIZONS:
            cap = tier2_capturability(sig, panel, gates, neutralization=sig.spec.neutralization,
                                      expected_sign=sig.spec.expected_sign, hold_horizon=h)
            std, harsh = cap.by_cost["standard"], cap.by_cost["harsh"]
            cells[h] = {"net_std": std.net_sharpe, "net_harsh": harsh.net_sharpe,
                        "turn": std.turnover_ann, "fric": cap.by_cost["frictionless"].net_sharpe}
        out[label] = cells
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rank = pd.read_csv(MCAP)
    sectors = dict(zip(rank["ticker"], rank["sector"].fillna("")))
    top = rank["ticker"].tolist()
    big_n = min(250, len(top))
    universe = (top[:big_n], sectors)
    print(f"[panel] fetching top-{big_n} S&P500 by market cap, {START}->latest ...")
    big = load_sp500_panel(START, universe=universe)
    # the loader may reorder/drop columns (unpriced names) — re-assert mcap order
    print(f"[panel] built: T={big.T} N={big.N}  span {big.dates[0]}..{big.dates[-1]}")
    print(f"[panel] top-10 by mcap present: {big.tickers[:10]}")

    universes = {"top50": (subset_cols(big, 50), 40),
                 "top100": (subset_cols(big, 100), 50),
                 "top150": (subset_cols(big, 150), 50),
                 "top250": (subset_cols(big, big_n), 50)}

    summary = {"meta": {"big_n": big_n, "T": big.T, "span": [str(big.dates[0]), str(big.dates[-1])],
                        "survivorship_free": False, "start": START}}

    # U5 multiplicity. Every ALPHA is tested on EVERY nested universe, so one invocation of
    # this script runs len(ALPHAS) x len(universes) hypotheses, not len(ALPHAS). Deflating each
    # universe against len(ALPHAS) alone was the batch-shape leak in its clearest form: a
    # for-loop silently dividing the multiple-comparison correction by four.
    n_hyp = len(ALPHAS) * len(universes)
    mult = Multiplicity.preregistered(n_hyp, substrate="xlg_megacap",
                                      provenance=f"{len(ALPHAS)} alphas x "
                                                 f"{len(universes)} nested universes "
                                                 f"({', '.join(universes)}), one script run",
                                      gates=load_multiplicity_gates())
    summary["meta"]["multiplicity"] = {"n_hypotheses": n_hyp, "source": mult.source,
                                       "provenance": mult.provenance}
    print(f"[multiplicity] deflating against {n_hyp} hypotheses "
          f"({len(ALPHAS)} alphas x {len(universes)} universes)")

    # ---- 1) nested gross-IC + capturability curve -------------------------------------
    print("\n=== 1) NESTED GROSS-IC CURVE (does IC collapse toward the mega-cap tier?) ===")
    print(f"{'universe':9} {'N':>4} {'minNm':>5} {'#FDRsig':>7} {'medIC-IR':>9} {'maxIC-IR':>9}")
    curve = {}
    for name, (p, mn) in universes.items():
        g = Gates.from_yaml(ROOT / "configs" / "signal_eval.gates.yaml")
        g = replace(g, min_names_per_day=mn)
        sw = tier1_sweep(p, g, mult)
        curve[name] = {k: sw[k] for k in ("n_fdr_sig", "median_ic_ir", "max_ic_ir", "ranked_names")}
        print(f"{name:9} {p.N:>4} {mn:>5} {sw['n_fdr_sig']:>7} "
              f"{sw['median_ic_ir']:>9.4f} {sw['max_ic_ir']:>9.4f}")
    summary["nested_curve"] = {k: {kk: vv for kk, vv in v.items() if kk != "ranked_names"}
                               for k, v in curve.items()}
    summary["nested_curve"]["_ref_U300_alphabetical"] = {"n_fdr_sig": 40, "median_ic_ir": 0.0475,
                                                         "max_ic_ir": 0.1079}

    # ---- 2) full scorecards for top50 / top100 ----------------------------------------
    print("\n=== 2) FULL SCORECARD — top50 & top100 ===")
    for name in ("top50", "top100"):
        p, mn = universes[name]
        g = replace(Gates.from_yaml(ROOT / "configs" / "signal_eval.gates.yaml"),
                    min_names_per_day=mn)
        rs = evaluate_batch({s.spec.name: s for s in ALPHAS}, p, g, batch_name=f"xlg_{name}",
                            multiplicity=mult)
        write_scorecard(rs, OUT / name)
        n_prom = sum(1 for c in rs.cards if c.verdict == "PROMISING")
        n_sig = sum(1 for c in rs.cards if c.deflation and np.isfinite(c.deflation.fdr_q)
                    and c.deflation.fdr_q < 0.05)
        top = rs.cards[0]
        hp = top.gross.by_horizon[top.gross.primary_horizon]
        print(f"  {name}: #FDR-sig={n_sig}  #PROMISING={n_prom}  "
              f"top={top.name}(IC-IR {hp.ic_ir:.3f}, DSR {top.deflation.dsr:.3f})")
        summary.setdefault("full", {})[name] = {"n_fdr_sig": n_sig, "n_promising": n_prom,
                                                "top": top.name, "top_ic_ir": float(hp.ic_ir),
                                                "top_dsr": float(top.deflation.dsr)}

    # ---- 3) horizon/turnover capturability of the LEADS + ensembles -------------------
    print("\n=== 3) HORIZON SWEEP — net Sharpe @ standard cost (the capturable lead) ===")
    summary["horizon_sweep"] = {}
    for name in ("top50", "top100", "top250"):
        p, mn = universes[name]
        g = replace(Gates.from_yaml(ROOT / "configs" / "signal_eval.gates.yaml"),
                    min_names_per_day=mn)
        refit5 = [BY_NAME[n] for n in curve[name]["ranked_names"][:5]]
        named = [(n, BY_NAME[n]) for n in LEADS]
        named.append(("ens_frozen5", Ensemble([BY_NAME[n] for n in FROZEN5], "ens_frozen5")))
        named.append(("ens_refit5", Ensemble(refit5, "ens_refit5")))
        sw = horizon_sweep(p, g, named)
        summary["horizon_sweep"][name] = sw
        print(f"\n  --- {name} (N={p.N}) ---   refit top-5: {curve[name]['ranked_names'][:5]}")
        for label, cells in sw.items():
            s = "  ".join(f"h{h}:{cells[h]['net_std']:+.2f}(t{cells[h]['turn']:.0f})"
                          for h in HORIZONS)
            print(f"    {label:13}: {s}")

    # ---- 4) XLG buy-and-hold benchmark (for the strategy stage) ------------------------
    print("\n=== 4) XLG buy-and-hold benchmark ===")
    import yfinance as yf
    xlg = yf.download("XLG", start=START, progress=False, auto_adjust=True)
    close = xlg["Close"].squeeze()
    ret = close.pct_change().dropna()
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252))
    cagr = float((close.iloc[-1] / close.iloc[0]) ** (252 / len(close)) - 1)
    dd = float((close / close.cummax() - 1).min())
    xlg_df = pd.DataFrame({"close": close})
    xlg_df.to_csv(OUT / "xlg_bh.csv")
    print(f"  XLG B&H: Sharpe={sharpe:.3f}  CAGR={cagr:.1%}  MaxDD={dd:.1%}  "
          f"(n={len(close)}, {close.index[0].date()}..{close.index[-1].date()})")
    summary["xlg_bh"] = {"sharpe": sharpe, "cagr": cagr, "max_dd": dd, "n": len(close)}

    (OUT / "gate_summary.json").write_text(json.dumps(summary, indent=2, default=float),
                                           encoding="utf-8")
    print(f"\n[wrote] {OUT/'gate_summary.json'}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    main()
