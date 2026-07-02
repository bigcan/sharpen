"""Breadth expansion — does widening the FROZEN TSMOM universe raise the honest DSR?

Fable review item 2 (`.agent/artifacts/tailwind_v1_fable_review.md`): TSMOM Sharpe scales
~sqrt(effective breadth); breadth is the one institutional edge available for free, and it is
the ONLY honest lever that can raise the Deflated-Sharpe (adding hedges cannot lift a
return-DSR; adding SEARCHED sleeves worsens multiplicity). R1 (`tailwind_v1_R1_dsr_pbo`) left
mom+BAB at honest DSR 0.896 < 0.95, the miss driven entirely by the momentum sleeve's honest
Sharpe.

BUT the prior evidence cuts both ways: Fable's own "untouched" 32-ETF test gave momentum
Sharpe 0.389 vs the curated 18-ETF ~0.60 (`portfolio_frontier.py`) — i.e. breadth may REVEAL
the 18-ETF was favorably selected rather than raise Sharpe. This script runs the honest test:
apply the FROZEN signal (63/126/252, skip 5, vol_win 63, tgt 0.10, cap 2.0 — UNCHANGED) to a
sensible ~32-instrument liquid cross-asset universe, rebuild the momentum grid + BAB, and
re-run the R1 DSR/PBO. Verdict is whatever it shows.

Method: monkeypatch the momentum-falsification universe globals (so the SAME validated frozen
machinery runs on the wide panel — no reimplementation, signal guaranteed identical), fetch to
a separate cache, then recompute DSR + CSCV-PBO. Emits results/tailwind_v1/breadth_expansion.json.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd
import yaml as _yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tailwind_v1"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import xsec_momentum_falsification as mom    # noqa: E402

# ---- Widened universe: liquid, mostly-long-history free-data ETFs, 4 classes (~32 names) ----
# Every class keeps >=5 names so within-class BAB + XSMOM have a cross-section. Names with a
# later inception (HYG/EMB/UNG/DBB 2007) contribute 0 until they have data (per-name warmup) —
# the honest breadth effect: instruments enter the book as their history begins.
WIDE_UNIVERSE = {
    "equity":    ["SPY", "QQQ", "IWM", "EFA", "EEM", "EWJ", "FXI", "EWZ", "VGK", "VNQ"],
    "rates":     ["TLT", "IEF", "LQD", "SHY", "TIP", "HYG", "EMB", "AGG"],
    "commodity": ["GLD", "SLV", "DBC", "USO", "DBA", "UNG", "DBB"],
    "fx":        ["UUP", "FXE", "FXY", "FXB", "FXA", "FXC", "FXF"],
}
WIDE_TICKERS = [t for v in WIDE_UNIVERSE.values() for t in v]
WIDE_CLASS_OF = {t: c for c, v in WIDE_UNIVERSE.items() for t in v}
ANN = mom.ANN


def _patch_universe(universe, tickers, class_of, cache_name):
    """Point the frozen momentum machinery at a different universe + cache (signal unchanged)."""
    mom.UNIVERSE = universe
    mom.ALL_TICKERS = tickers
    mom.CLASS_OF = class_of
    mom.CACHE = OUT / cache_name


def _combined_dsr_pbo(n_trials, min_dsr, block_days, bracket_N, label):
    """Recompute the R1 numbers (DSR on honest mom+BAB, CSCV-PBO on the momentum grid) on the
    universe the momentum globals currently point at. Imports the R1 helpers late so they read
    the patched `mom` globals."""
    import portfolio_frontier as pf
    import audit_tailwind_book as at
    from finrl_pro_ds.crypto.eval.statistics import (
        block_bootstrap_sharpe_ci, deflated_sharpe_ratio, excess_kurtosis,
        probability_of_backtest_overfitting, skewness,
    )

    mom_net = pf.build_momentum_net()
    def_net = at.build_defensive_net()
    combined, common, (m_al, d_al) = pf.risk_parity([mom_net, def_net])

    m_sh = mom.sharpe(m_al)
    d_sh = mom.sharpe(d_al)
    corr = float(m_al.corr(d_al))
    # Honest basis: use the wide pooled momentum Sharpe DIRECTLY (untouched breadth IS the
    # honesty mechanism — no artificial 0.389 haircut). Report curated == honest here.
    cd = combined.dropna().to_numpy()
    obs_sr_daily = float(mom.sharpe(combined) / (ANN ** 0.5))
    grid = mom.graded_book_sharpes()
    trial_sharpes_daily = [s / (ANN ** 0.5) for s in grid.values()]
    g1, g2 = skewness(cd.tolist()), excess_kurtosis(cd.tolist())

    def _dsr(N):
        return deflated_sharpe_ratio(obs_sr_daily, trial_sharpes_daily, n_obs=len(cd),
                                     skew=g1, excess_kurt=g2, n_trials=N, periods_per_year=ANN)

    dsr_pre = _dsr(n_trials) or {}
    dsr_val = dsr_pre.get("dsr")
    boot = block_bootstrap_sharpe_ci(cd, block=block_days, periods_per_year=ANN)

    # PBO on the (wide) momentum grid selection surface
    books = mom.build_books(*(lambda c: (c, c.pct_change(), c["SPY"].pct_change()))(
        mom.get_prices()[[t for t in mom.ALL_TICKERS if t in mom.get_prices().columns]]))
    grid_df = pd.DataFrame({b: bk["_net_standard"] for b, bk in books.items()}).dropna(how="any")
    pbo = probability_of_backtest_overfitting(grid_df.to_numpy(dtype=float), n_splits=16)

    return {
        "label": label,
        "n_instruments": len([t for t in mom.ALL_TICKERS if t in mom.get_prices().columns]),
        "n_momentum_books_graded": len(trial_sharpes_daily),
        "common_window": [str(common.min().date()), str(common.max().date())],
        "momentum_pooled_net_sharpe": round(m_sh, 3),
        "bab_net_sharpe": round(d_sh, 3),
        "corr_mom_bab": round(corr, 3),
        "combined_sharpe": round(mom.sharpe(combined), 3),
        "combined_oos_2018": round(mom.sharpe(combined.loc["2018-01-01":]), 3),
        "n_trials": n_trials, "min_dsr": min_dsr,
        "dsr_at_n_trials": (None if dsr_val is None else round(dsr_val, 4)),
        "dsr_bracket": {str(N): (None if (d := _dsr(N)) is None else round(d["dsr"], 4))
                        for N in bracket_N},
        "dsr_pass": bool(dsr_val is not None and dsr_val >= min_dsr),
        "block_bootstrap_ci95": (None if boot is None else
                                 {k: (round(v, 4) if isinstance(v, float) else v)
                                  for k, v in boot.items()}),
        "pbo": (None if pbo is None else round(float(pbo["pbo"]), 4)),
        "pbo_pass": bool(pbo is not None and pbo["pbo"] <= 0.50),
    }


def main():
    of = _yaml.safe_load(
        (ROOT / "configs" / "tailwind_v1.gates.yaml").read_text(encoding="utf-8"))["overfitting"]
    min_dsr = float(of["min_dsr"])
    block_days = int(of.get("block_bootstrap_block_days", 21))

    # ---- WIDE universe: fetch + recompute (n_trials scales with the wide grid) ----
    _patch_universe(WIDE_UNIVERSE, WIDE_TICKERS, WIDE_CLASS_OF, "prices_wide_daily.parquet")
    close = mom.get_prices()                                   # fetches + caches the wide panel
    present = [t for t in WIDE_TICKERS if t in close.columns]
    missing = [t for t in WIDE_TICKERS if t not in close.columns]
    n_grid = len(mom.graded_book_sharpes())                    # honest grid size on the wide panel
    n_trials_wide = n_grid + 3                                 # grid + 3rd-sleeve/combine selection
    bracket = [n_grid, n_trials_wide, n_trials_wide + 7]
    wide = _combined_dsr_pbo(n_trials_wide, min_dsr, block_days, bracket,
                             f"wide_{len(present)}etf")
    wide["missing_tickers"] = missing

    payload = {
        "test": "breadth expansion — frozen TSMOM signal on a widened universe",
        "frozen_signal": {"lookbacks": mom.LOOKBACKS, "skip": mom.SKIP, "vol_win": mom.VOL_WIN,
                          "target_vol_asset": mom.TARGET_VOL_ASSET, "lev_cap": mom.LEV_CAP},
        "baseline_18etf_R1": {"momentum_pooled_net_sharpe": 0.601, "honest_dsr_N24": 0.896,
                              "curated_dsr_N24": 0.9744, "pbo": 0.0009,
                              "ref": "docs/research/tailwind_v1_R1_dsr_pbo_2026-07-01.md"},
        "wide": wide,
        "verdict": {},
    }
    # Verdict: did breadth RAISE the honest momentum Sharpe, and does the combined clear DSR?
    raised = wide["momentum_pooled_net_sharpe"] >= 0.601
    payload["verdict"] = {
        "breadth_raised_momentum_sharpe": bool(raised),
        "wide_momentum_sharpe": wide["momentum_pooled_net_sharpe"],
        "vs_18etf_0.601": round(wide["momentum_pooled_net_sharpe"] - 0.601, 3),
        "wide_combined_dsr": wide["dsr_at_n_trials"],
        "wide_dsr_clears_0.95": wide["dsr_pass"],
        "wide_pbo_clears_0.50": wide["pbo_pass"],
        "R1_clears_on_wide": bool(wide["dsr_pass"] and wide["pbo_pass"]),
        "note": "If wide momentum Sharpe < 0.601, breadth REVEALS the 18-ETF was favorably "
                "selected (Fable's 32-ETF=0.389 precedent) rather than raising the edge — a "
                "NO-GO for the breadth path. If it rises AND DSR clears, breadth is the honest "
                "route to capital.",
    }
    (OUT / "breadth_expansion.json").write_text(json.dumps(payload, indent=2, default=str))

    print("=" * 78)
    print("BREADTH EXPANSION — frozen TSMOM on a widened universe")
    print("=" * 78)
    print(f"wide universe: {len(present)}/{len(WIDE_TICKERS)} instruments present "
          f"(missing: {missing or 'none'})")
    print(f"common window: {wide['common_window']}   momentum grid: {n_grid} books  "
          f"(n_trials={n_trials_wide})")
    print("-" * 78)
    print(f"  momentum pooled net Sharpe: WIDE {wide['momentum_pooled_net_sharpe']}  "
          f"vs 18-ETF 0.601  ({payload['verdict']['vs_18etf_0.601']:+})")
    print(f"  BAB net Sharpe: {wide['bab_net_sharpe']}   corr(mom,BAB): {wide['corr_mom_bab']}")
    print(f"  combined Sharpe: {wide['combined_sharpe']}   OOS-2018: {wide['combined_oos_2018']}")
    print(f"  DSR(N={n_trials_wide}) = {wide['dsr_at_n_trials']}  (min {min_dsr})  "
          f"bracket {wide['dsr_bracket']}")
    print(f"  PBO = {wide['pbo']}  (max 0.50)")
    print("-" * 78)
    v = payload["verdict"]
    print(f"breadth raised momentum Sharpe? {v['breadth_raised_momentum_sharpe']}")
    print(f"R1 clears on wide universe (DSR>=0.95 AND PBO<=0.5)? {v['R1_clears_on_wide']}")
    print("=" * 78)
    return payload


if __name__ == "__main__":
    main()
