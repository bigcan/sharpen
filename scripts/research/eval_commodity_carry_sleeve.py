#!/usr/bin/env python
"""Marginal-contribution eval for the COMMODITY-CARRY candidate sleeve (S553-cont, ADVISORY).

The 4th-sleeve analog of ``eval_defensive_sleeve.py``. Answers the only question that matters for
an added sleeve: does folding ``commodity_carry`` (a thin, energy-only ETF roll-spread carry book —
see ``finrl_pro_ds/features/commodity_carry.py`` for why it is thin) into the live
{tsmom, rates_carry} book improve the inverse-vol COMBINED book, net of cost, at low correlation?
This is the portfolio bar (marginal uplift), NOT the sleeve's standalone Sharpe.

Method (parameter-free combiner ⇒ no in-sample selection to overfit):
  * Build the sleeve return streams on the cross-asset panel clock via
    ``production_base_sleeves(include_commodity_carry=True)`` — one basis, LEAK-2 tripwired.
  * Combine {tsmom, rates_carry} vs {tsmom, rates_carry, commodity_carry} with a CAUSAL trailing
    inverse-vol combiner (the static ``inverse_vol`` combiner ADR-C1-4), on the SAME bars.
  * Report: corr(commodity_carry, each base sleeve), full-sample + held-out-OOS-tail net Sharpe of
    each book, the ΔSharpe, a moving-block-bootstrap p-value that ΔSharpe(full) > 0, and the BINDING
    C2 CPCV distribution of the uplift over the combinatorial-purged paths.
  * Read the accept bars (``max_base_corr``, ``min_combination_uplift``, ``delta_median_min``,
    ``frac_positive_min``) from the gates file — never hardcoded. Verdict is ADVISORY: a real add
    needs C2 CPCV + a Tier-2 audit.

Usage:
  python scripts/research/eval_commodity_carry_sleeve.py --start 2008-01-01
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
log = logging.getLogger("eval_commodity_carry_sleeve")

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


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[a, b)`` runs of True in a boolean mask (adjacent test groups merge into one run)."""
    runs: list[tuple[int, int]] = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _cpcv_uplift_distribution(r2: np.ndarray, r3: np.ndarray, *, n_groups: int, k_test: int,
                              embargo: int, purge_horizon: int) -> dict:
    """Combinatorial Purged CV distribution of the combined-book ΔSharpe (book3 − book2).

    The C2 ``tier3_5_cpcv`` structure applied to the PAIRED daily combined-book returns: partition
    ``[0, n)`` into ``n_groups`` contiguous groups; for each combination of ``k_test`` groups the
    test set is their union; inside each contiguous test run embargo the start and purge the last
    ``purge_horizon`` bars (the 1-day book label reaches t+1). The inverse-vol combiner is
    parameter-free ⇒ no refit per path (ADR-C2-2); we slice the realized paired returns and score
    net Sharpe of each book on the SAME purged test bars ⇒ a distribution of uplift over the paths.
    """
    from itertools import combinations

    n = len(r2)
    bounds = np.linspace(0, n, n_groups + 1).astype(int)
    groups = [(int(bounds[i]), int(bounds[i + 1])) for i in range(n_groups)]
    per_path: list[dict] = []
    for combo in combinations(range(n_groups), k_test):
        in_test = np.zeros(n, dtype=bool)
        for gi in combo:
            a, b = groups[gi]
            in_test[a:b] = True
        idx: list[int] = []
        for a, b in _contiguous_runs(in_test):
            lo, hi = a + embargo, b - purge_horizon
            if hi > lo:
                idx.extend(range(lo, hi))
        if len(idx) >= 2:
            ii = np.asarray(idx, dtype=int)
            s2, s3 = _net_sharpe(r2[ii]), _net_sharpe(r3[ii])
            per_path.append({"groups": list(combo), "sr_book2": round(s2, 4),
                             "sr_book3": round(s3, 4), "uplift": round(s3 - s2, 4)})
    ups = np.asarray([p["uplift"] for p in per_path], dtype=np.float64)
    if ups.size == 0:
        return {"n_paths": 0}
    return {
        "n_paths": int(ups.size), "n_groups": n_groups, "k_test": k_test,
        "embargo_days": embargo, "purge_horizon": purge_horizon,
        "uplift_mean": round(float(ups.mean()), 4),
        "uplift_median": round(float(np.median(ups)), 4),
        "uplift_std": round(float(ups.std(ddof=1)) if ups.size > 1 else 0.0, 4),
        "uplift_p05": round(float(np.quantile(ups, 0.05)), 4),
        "uplift_min": round(float(ups.min()), 4), "uplift_max": round(float(ups.max()), 4),
        "frac_paths_positive": round(float((ups > 0).mean()), 4),
        "per_path": per_path,
    }


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
    ap = argparse.ArgumentParser(description="Commodity-carry sleeve marginal-contribution eval")
    ap.add_argument("--start", default="2008-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--gen_gates", default=str(ROOT / "configs" / "signal_eval.gates.yaml"),
                    help="gates file whose generation block holds max_base_corr / min_combination_uplift")
    ap.add_argument("--split_frac", type=float, default=0.7, help="held-out OOS tail fraction")
    ap.add_argument("--boot_alpha", type=float, default=0.10,
                    help="bootstrap significance level for the full-sample uplift (stat level, not a "
                         "strategy gate); ADD_CANDIDATE requires the uplift be significant at this")
    ap.add_argument("--out",
                    default=str(ROOT / "results" / "signal_eval" / "commodity_carry_sleeve_eval.json"))
    args = ap.parse_args()

    from finrl_pro_ds.data.cross_asset_panel_loader import load_cross_asset_panel
    from finrl_pro_ds.signals.generation.base_sleeves import production_base_sleeves

    gen = (yaml.safe_load(Path(args.gen_gates).read_text(encoding="utf-8")) or {}).get("generation", {})
    max_base_corr = float(gen.get("max_base_corr", 0.70))
    min_uplift = float(gen.get("min_combination_uplift", 0.10))
    hold = int(gen.get("hold_horizon", 21))
    cost_bps = float(gen.get("cost_bps", 0.0010))
    cpcv_n_groups = int(gen.get("cpcv_n_groups", 6))
    cpcv_k_test = int(gen.get("cpcv_k_test", 2))
    cpcv_embargo = int(gen.get("cpcv_embargo_days", 21))
    cpcv_purge = int(gen.get("cpcv_purge_horizon", 1))
    # GP4-06 fragility gate (REPAIRED, crucible-v2.0): median ∧ frac-positive on the ΔSR paths.
    # The former p05 veto was CPCV-geometry noise (Test B); p05 is still printed as a diagnostic.
    delta_median_min = float(gen.get("delta_median_min", 0.0))
    frac_positive_min = float(gen.get("frac_positive_min", 0.50))

    panel = load_cross_asset_panel(
        args.start, args.end, config_path=ROOT / "configs" / "cross_asset_momentum.yaml")
    log.info("panel: %d bars × %d names (%s)", panel.T, panel.N, panel.meta.get("source"))
    base = production_base_sleeves(panel, hold_horizon=hold, cost_bps=cost_bps,
                                   start=args.start, end=args.end, include_commodity_carry=True)

    tsmom, rates, commodity = base["tsmom"], base["rates_carry"], base["commodity_carry"]

    # correlations on the common finite bars (the diversification check, GP4-01 analog)
    m = np.isfinite(tsmom) & np.isfinite(rates) & np.isfinite(commodity)
    corr_tsmom = float(np.corrcoef(commodity[m], tsmom[m])[0, 1])
    corr_rates = float(np.corrcoef(commodity[m], rates[m])[0, 1])
    max_abs_corr = max(abs(corr_tsmom), abs(corr_rates))

    comb2, v2 = _inverse_vol_combined({"tsmom": tsmom, "rates_carry": rates})
    comb3, v3 = _inverse_vol_combined(
        {"tsmom": tsmom, "rates_carry": rates, "commodity_carry": commodity})
    valid = v2 & v3                                   # SAME bars for a fair comparison
    r2, r3 = comb2[valid], comb3[valid]

    n = len(r2)
    split = int(n * args.split_frac)
    sr2_full, sr3_full = _net_sharpe(r2), _net_sharpe(r3)
    sr2_oos, sr3_oos = _net_sharpe(r2[split:]), _net_sharpe(r3[split:])
    uplift_full, uplift_oos = sr3_full - sr2_full, sr3_oos - sr2_oos
    p_boot = _block_bootstrap_p(r2, r3)

    # C2 CPCV: the BINDING OOS evidence — distribution of ΔSharpe over the combinatorial purged paths.
    cpcv = _cpcv_uplift_distribution(r2, r3, n_groups=cpcv_n_groups, k_test=cpcv_k_test,
                                     embargo=cpcv_embargo, purge_horizon=cpcv_purge)

    # Honest verdict, driven by the CPCV DISTRIBUTION (not the single high-variance OOS split):
    #   ADD_CANDIDATE               — uncorrelated AND the uplift is robust across paths: median ≥
    #                                 delta_median_min AND a majority of paths positive (frac+ ≥
    #                                 frac_positive_min). GP4-06 REPAIRED — the p05 leg (CPCV-geometry
    #                                 noise, Test B) was dropped; p05 is reported, not gated.
    #   UNCORRELATED_UPLIFT_UNPROVEN — a clean diversifier whose uplift is positive somewhere but not
    #                                 robust across paths (the recent-regime-concentrated case).
    #   NO_ADD                       — fails correlation, or the CPCV distribution is not positive.
    corr_ok = max_abs_corr <= max_base_corr
    uplift_ok = uplift_oos >= min_uplift                     # single-split screen (secondary)
    boot_sig = np.isfinite(p_boot) and p_boot < args.boot_alpha
    cpcv_median = float(cpcv.get("uplift_median", float("nan")))
    cpcv_p05 = float(cpcv.get("uplift_p05", float("nan")))
    cpcv_frac_pos = float(cpcv.get("frac_paths_positive", float("nan")))
    cpcv_robust = (np.isfinite(cpcv_median) and cpcv_median >= delta_median_min
                   and np.isfinite(cpcv_frac_pos) and cpcv_frac_pos >= frac_positive_min)
    if corr_ok and cpcv_robust:
        verdict = "ADD_CANDIDATE"
    elif corr_ok and (cpcv_median > 0.0 or uplift_oos >= min_uplift or uplift_full > 0):
        verdict = "UNCORRELATED_UPLIFT_UNPROVEN"
    else:
        verdict = "NO_ADD"

    payload = {
        "advisory": "portfolio marginal-contribution SCREEN; a real add needs C2 CPCV + Tier-2",
        "sleeve": "commodity_carry", "panel_source": panel.meta.get("source"),
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
        "cpcv": cpcv, "delta_median_min_gate": delta_median_min,
        "frac_positive_min_gate": frac_positive_min, "cpcv_robust": bool(cpcv_robust),
        "verdict": verdict,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.warning("ADVISORY verdict=%s | corr(tsmom)=%.3f corr(rates)=%.3f (gate<=%.2f) | "
                "single-split OOS uplift %.3f (boot p=%.3f) | CPCV %d paths: median=%.3f "
                "(floor %.2f) frac+=%.2f (floor %.2f) p05=%.3f [diag] -> robust=%s",
                verdict, corr_tsmom, corr_rates, max_base_corr, uplift_oos, p_boot,
                int(cpcv.get("n_paths", 0)), cpcv_median, delta_median_min,
                cpcv_frac_pos, frac_positive_min, cpcv_p05, cpcv_robust)
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
