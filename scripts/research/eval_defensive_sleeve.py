#!/usr/bin/env python
"""Marginal-contribution eval for the DEFENSIVE / BAB candidate sleeve (S553-cont-95, ADVISORY).

Answers the only question that matters for an added sleeve: does folding ``defensive`` into the
live {tsmom, rates_carry} book improve the inverse-vol COMBINED book, net of cost, at low
correlation? This is the portfolio bar (marginal uplift), NOT the sleeve's standalone Sharpe.

Method (parameter-free combiner ⇒ no in-sample selection to overfit):
  * Build all three sleeve return streams on the cross-asset panel clock via
    ``production_base_sleeves(include_defensive=True)`` — one basis, LEAK-2 tripwired.
  * Combine {tsmom, rates_carry} vs {tsmom, rates_carry, defensive} with a CAUSAL trailing
    inverse-vol combiner (the static ``inverse_vol`` combiner ADR-C1-4), on the SAME bars.
  * Report: corr(defensive, each base sleeve), full-sample + held-out-OOS-tail net Sharpe of each
    book, the ΔSharpe, and a moving-block-bootstrap p-value that ΔSharpe(full) > 0.
  * Read the accept bars (``max_base_corr``, ``min_combination_uplift``) from the gates file — never
    hardcoded. Verdict is ADVISORY: a real add-decision needs C2 CPCV + a Tier-2 deep audit.

Usage:
  python scripts/research/eval_defensive_sleeve.py --start 2008-01-01
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("eval_defensive_sleeve")

ANN = 252


def _net_sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ANN)) if sd > 1e-12 else 0.0


def _inverse_vol_combined(streams: dict[str, np.ndarray], *, vol_window: int = 63,
                          min_periods: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """Causal trailing inverse-vol combined book return + a validity mask.

    Per bar, weight_s ∝ 1/trailing_std_s (``.shift(1)`` ⇒ vol reads returns <= t-1), convex
    (weights sum to 1). ``valid`` marks bars where every input is finite AND the weights/return
    are defined (past the vol warmup)."""
    names = list(streams)
    df = pd.DataFrame({n: streams[n] for n in names})
    vol = df.rolling(vol_window, min_periods=min_periods).std().shift(1)
    inv = (1.0 / vol.replace(0.0, np.nan))
    w = inv.div(inv.sum(axis=1), axis=0)
    comb = (w * df).sum(axis=1, min_count=len(names))
    valid = np.isfinite(df.to_numpy()).all(axis=1) & np.isfinite(w.to_numpy()).all(axis=1) \
        & np.isfinite(comb.to_numpy())
    return comb.to_numpy(dtype=np.float64), valid


def _block_bootstrap_p(r2: np.ndarray, r3: np.ndarray, *, block: int = 21,
                       n_boot: int = 2000, seed: int = 7) -> float:
    """Moving-block bootstrap p-value that Sharpe(book3) - Sharpe(book2) > 0 on the paired series
    (resample the SAME bar-blocks for both books so the pairing/dependence is preserved)."""
    rng = np.random.default_rng(seed)
    n = len(r2)
    if n < block * 2:
        return float("nan")
    n_blocks = int(np.ceil(n / block))
    starts_pool = np.arange(0, n - block + 1)
    ge = 0
    for _ in range(n_boot):
        starts = rng.choice(starts_pool, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        d = _net_sharpe(r3[idx]) - _net_sharpe(r2[idx])
        if d > 0:
            ge += 1
    return 1.0 - ge / n_boot                          # p(ΔSharpe <= 0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Defensive/BAB sleeve marginal-contribution eval")
    ap.add_argument("--start", default="2008-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--gen_gates", default=str(ROOT / "configs" / "signal_eval.gates.yaml"),
                    help="gates file whose generation block holds max_base_corr / min_combination_uplift")
    ap.add_argument("--split_frac", type=float, default=0.7, help="held-out OOS tail fraction")
    ap.add_argument("--boot_alpha", type=float, default=0.10,
                    help="bootstrap significance level for the full-sample uplift (stat level, not a "
                         "strategy gate); ADD_CANDIDATE requires the uplift be significant at this")
    ap.add_argument("--out", default=str(ROOT / "results" / "signal_eval" / "defensive_sleeve_eval.json"))
    args = ap.parse_args()

    from finrl_pro_ds.data.cross_asset_panel_loader import load_cross_asset_panel
    from finrl_pro_ds.signals.generation.base_sleeves import production_base_sleeves

    gen = (yaml.safe_load(Path(args.gen_gates).read_text(encoding="utf-8")) or {}).get("generation", {})
    max_base_corr = float(gen.get("max_base_corr", 0.70))
    min_uplift = float(gen.get("min_combination_uplift", 0.10))
    hold = int(gen.get("hold_horizon", 21))
    cost_bps = float(gen.get("cost_bps", 0.0010))

    panel = load_cross_asset_panel(
        args.start, args.end, config_path=ROOT / "configs" / "cross_asset_momentum.yaml")
    log.info("panel: %d bars × %d names (%s)", panel.T, panel.N, panel.meta.get("source"))
    base = production_base_sleeves(panel, hold_horizon=hold, cost_bps=cost_bps,
                                   start=args.start, end=args.end, include_defensive=True)

    tsmom, rates, defensive = base["tsmom"], base["rates_carry"], base["defensive"]

    # correlations on the common finite bars (the diversification check, GP4-01 analog)
    m = np.isfinite(tsmom) & np.isfinite(rates) & np.isfinite(defensive)
    corr_tsmom = float(np.corrcoef(defensive[m], tsmom[m])[0, 1])
    corr_rates = float(np.corrcoef(defensive[m], rates[m])[0, 1])
    max_abs_corr = max(abs(corr_tsmom), abs(corr_rates))

    comb2, v2 = _inverse_vol_combined({"tsmom": tsmom, "rates_carry": rates})
    comb3, v3 = _inverse_vol_combined({"tsmom": tsmom, "rates_carry": rates, "defensive": defensive})
    valid = v2 & v3                                   # SAME bars for a fair comparison
    r2, r3 = comb2[valid], comb3[valid]

    n = len(r2)
    split = int(n * args.split_frac)
    sr2_full, sr3_full = _net_sharpe(r2), _net_sharpe(r3)
    sr2_oos, sr3_oos = _net_sharpe(r2[split:]), _net_sharpe(r3[split:])
    uplift_full, uplift_oos = sr3_full - sr2_full, sr3_oos - sr2_oos
    p_boot = _block_bootstrap_p(r2, r3)

    # Honest tri-state verdict. A large single-split OOS uplift is NOT proof — the full-sample
    # bootstrap is the trustworthy stat (one 30% tail is high-variance / regime-concentrated). So:
    #   ADD_CANDIDATE               — uncorrelated, OOS uplift clears the gate, AND the full-sample
    #                                 uplift is bootstrap-significant.
    #   UNCORRELATED_UPLIFT_UNPROVEN — a clean diversifier whose uplift is positive but not yet
    #                                 significant (the recent-regime-concentrated case) → needs CPCV.
    #   NO_ADD                       — fails correlation or shows no OOS uplift.
    corr_ok = max_abs_corr <= max_base_corr
    uplift_ok = uplift_oos >= min_uplift
    boot_sig = np.isfinite(p_boot) and p_boot < args.boot_alpha
    if corr_ok and uplift_ok and boot_sig:
        verdict = "ADD_CANDIDATE"
    elif corr_ok and (uplift_ok or uplift_full > 0):
        verdict = "UNCORRELATED_UPLIFT_UNPROVEN"
    else:
        verdict = "NO_ADD"

    payload = {
        "advisory": "portfolio marginal-contribution SCREEN; a real add needs C2 CPCV + Tier-2",
        "sleeve": "defensive_bab", "panel_source": panel.meta.get("source"),
        "n_bars_combined": int(n), "cost_bps": cost_bps, "hold_horizon": hold,
        "corr_to_tsmom": round(corr_tsmom, 4), "corr_to_rates": round(corr_rates, 4),
        "max_abs_base_corr": round(max_abs_corr, 4), "max_base_corr_gate": max_base_corr,
        "corr_gate_pass": bool(corr_ok),
        "book2_net_sharpe_full": round(sr2_full, 4), "book3_net_sharpe_full": round(sr3_full, 4),
        "book2_net_sharpe_oos": round(sr2_oos, 4), "book3_net_sharpe_oos": round(sr3_oos, 4),
        "uplift_full": round(uplift_full, 4), "uplift_oos": round(uplift_oos, 4),
        "min_combination_uplift_gate": min_uplift, "uplift_gate_pass": bool(uplift_ok),
        "bootstrap_p_uplift_le_0": round(p_boot, 4), "boot_alpha": args.boot_alpha,
        "bootstrap_significant": bool(boot_sig),
        "verdict": verdict,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.warning("ADVISORY verdict=%s | corr(tsmom)=%.3f corr(rates)=%.3f (gate<=%.2f) | "
                "book2->book3 OOS SR %.3f->%.3f (uplift %.3f, gate>=%.2f) | boot p=%.3f",
                verdict, corr_tsmom, corr_rates, max_base_corr, sr2_oos, sr3_oos,
                uplift_oos, min_uplift, p_boot)
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
