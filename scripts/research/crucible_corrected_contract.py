"""Crucible corrected-contract CALIBRATION — E1 (null FPR) + power curve vs the shipped gate.

Tier-3 of the anatomy paper (figure F2 second half + §5). Scores the SAME planted/null candidates
through BOTH the shipped funnel gate (``combination_fitness.passes_gate``) and the §5 corrected contract
(``corrected_contract_fitness.passes_corrected``), on the shared planted machinery from
``crucible_calibration`` (``_planted_panel`` / ``_planted_base_sleeves`` / ``_overlay_returns``).

  * E1 (null, β=0): per-candidate FPR of the corrected contract on pure-noise substrates. Reports the
    significance-stat-alone FPR (the ADR-2 check on the full-panel Sharpe-diff SE — target ~1%) AND the
    full-contract FPR with guards+binding-LORD++ (target << 1%). Not-a-rubber-stamp check.
  * Power: sweep the plant strength β → realized marginal ΔSR; report the DETECTION power of the shipped
    gate vs the corrected contract at each. Target: shipped ≈0 at realistic ΔSR while corrected ≈0.37 @
    ΔSR 0.5, ≈0.81 @ 0.8 (T=4044) — the F2 flagship contrast.

Zero funnel gate bytes touched (CRU-1): reads the shipped scorers as-is; the corrected contract has its
own gates file (``configs/crucible_corrected_contract.gates.yaml``).

Usage (from repo root):
    python scripts/research/crucible_corrected_contract.py --exp e1
    python scripts/research/crucible_corrected_contract.py --exp power
    python scripts/research/crucible_corrected_contract.py --exp both --quick
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.research.crucible_calibration as cal  # noqa: E402  (shared planted machinery)
from sharpen.crucible.corrected_contract import (  # noqa: E402
    CorrectedConfig,
    corrected_contract_fitness,
    fresh_lord_level,
)
from sharpen.signals.generation.config import load_generation_config  # noqa: E402
from sharpen.signals.generation.evolve import _overlay_returns  # noqa: E402
from sharpen.signals.generation.fitness import _combined_book, combination_fitness  # noqa: E402

log = logging.getLogger("crucible_corrected_contract")

DEFAULT_FUNNEL_GATES = ROOT / "configs" / "signal_eval.gates.yaml"
DEFAULT_CORR_GATES = ROOT / "configs" / "crucible_corrected_contract.gates.yaml"

_GEN_N_EFF = 50.0        # file-drawer N for the SHIPPED gate's DSR (matches E2 gate_probe_gen_n_eff)
_COST_BPS = 0.0010


def _load_fit_cfg(path: Path):
    """The shared MECHANICS config (combiner + CPCV + periods) from the funnel gates — NOT its verdict."""
    fit_cfg, _ = load_generation_config(path)
    return fit_cfg


def _candidate(t: int, n: int, beta: float, seed: int, fit_cfg, cost_bps: float):
    """One planted overlay candidate on a β-plant base book (β=0 → null). Mirrors the E2 direct-gate path."""
    panel, s = cal._planted_panel(t, n, seed=seed)
    base = cal._planted_base_sleeves(s, beta=beta, seed=seed)
    ts = cal._panel_ts(panel)
    base_book = _combined_book(base, ts, fit_cfg)
    out = _overlay_returns("macro:plant", panel, base_book, cost_bps=cost_bps)
    if out is None:
        return None
    cand, turnover = out
    return cand, base, ts, turnover


# ------------------------------------------------------------------ E1: corrected-contract null FPR
def run_e1_corrected(fit_cfg, cc: CorrectedConfig, *, t: int, n: int, n_seeds: int,
                     cost_bps: float, alpha: float = 0.05) -> dict:
    lvl = fresh_lord_level(cc)
    t_pass = passes = denom = 0
    for k in range(n_seeds):
        m = _candidate(t, n, 0.0, 6000 + k, fit_cfg, cost_bps)
        if m is None:
            continue
        cand, base, ts, _turnover = m
        r = corrected_contract_fitness(cand, base, ts, fit_cfg, cc, lord_level=lvl)
        denom += 1
        t_pass += int(r.t_pass)
        passes += int(r.passes_corrected)
    fpr_t = t_pass / max(1, denom)
    fpr_full = passes / max(1, denom)
    cp_t = cal.clopper_pearson_upper(t_pass, denom, alpha)
    cp_full = cal.clopper_pearson_upper(passes, denom, alpha)
    # ADR-2: the significance-stat-alone FPR should be ~t_min's nominal one-sided level (2.33 -> ~0.01).
    # If materially above 0.01 the full-panel Sharpe-diff SE is too optimistic -> switch n_eff_mode
    # (ar1|raw) / tune t_min. (The first design's CPCV-path t-stat gave ~14% FPR — voided; see ADR-1/2.)
    nominal = float(norm.sf(cc.t_min))
    return {
        "t": t, "n": n, "n_candidates": denom, "lord_level": lvl,
        "t_min": cc.t_min, "n_eff_mode": cc.n_eff_mode, "nominal_one_sided_level": nominal,
        "t_stat_fpr": fpr_t, "t_stat_fpr_cp_upper95": cp_t,
        "full_contract_fpr": fpr_full, "full_contract_fpr_cp_upper95": cp_full,
        "t_stat_calibrated": bool(cp_t <= max(0.02, 2.0 * nominal)),   # ~1% target, CP-upper within 2x
        "full_contract_green": bool(cp_full <= 0.01),
    }


# ------------------------------------------------------------------ power: shipped vs corrected
def run_power_corrected(fit_cfg, cc: CorrectedConfig, *, t: int, n: int, betas: list[float],
                        n_seeds: int, cost_bps: float, gen_n_eff: float) -> dict:
    lvl = fresh_lord_level(cc)
    rows: list[dict[str, Any]] = []
    for beta in betas:
        shipped = corrected = denom = 0
        deltas: list[float] = []
        for k in range(n_seeds):
            m = _candidate(t, n, beta, 5000 + k, fit_cfg, cost_bps)
            if m is None:
                continue
            cand, base, ts, turnover = m
            fr = combination_fitness(cand, base, ts, fit_cfg, gen_n_eff=gen_n_eff,
                                     turnover_ann=turnover, n_nodes=1)
            cr = corrected_contract_fitness(cand, base, ts, fit_cfg, cc, lord_level=lvl)
            denom += 1
            shipped += int(fr.passes_gate)
            corrected += int(cr.passes_corrected)
            if np.isfinite(cr.delta_sr):
                deltas.append(float(cr.delta_sr))
        rows.append({
            "beta": beta, "n": denom,
            "realized_delta_sr": float(np.mean(deltas)) if deltas else float("nan"),
            "shipped_power": shipped / max(1, denom),
            "corrected_power": corrected / max(1, denom),
        })
        log.info("  beta=%.4f: realized_dSR=%.3f shipped=%.2f corrected=%.2f",
                 rows[-1]["beta"], rows[-1]["realized_delta_sr"],
                 rows[-1]["shipped_power"], rows[-1]["corrected_power"])
    return {"t": t, "n": n, "n_seeds": n_seeds, "gen_n_eff": gen_n_eff, "rows": rows,
            "corrected_at_dsr_0.5": _interp_power(rows, 0.5),
            "corrected_at_dsr_0.8": _interp_power(rows, 0.8),
            "shipped_at_dsr_0.5": _interp_power(rows, 0.5, key="shipped_power"),
            "note": "power vs realized marginal ΔSR; target corrected ~0.37@0.5, ~0.81@0.8 (T=4044) vs shipped ~0"}


def _interp_power(rows: list[dict], target_dsr: float, key: str = "corrected_power") -> float | None:
    """Linear-interpolate detection power at a target realized ΔSR from the (sorted) curve."""
    pts = sorted(((r["realized_delta_sr"], r[key]) for r in rows if np.isfinite(r["realized_delta_sr"])),
                 key=lambda p: p[0])
    if len(pts) < 2 or target_dsr < pts[0][0] or target_dsr > pts[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= target_dsr <= x1:
            w = 0.0 if x1 == x0 else (target_dsr - x0) / (x1 - x0)
            return float(y0 + w * (y1 - y0))
    return None


# ------------------------------------------------------------------ driver
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="Corrected-contract calibration: E1 null-FPR + power vs shipped")
    ap.add_argument("--exp", choices=("e1", "power", "both"), default="both")
    ap.add_argument("--config", default=str(DEFAULT_FUNNEL_GATES))
    ap.add_argument("--corr-config", default=str(DEFAULT_CORR_GATES))
    ap.add_argument("--e1-t", type=int, default=2048)
    ap.add_argument("--e1-seeds", type=int, default=500)
    ap.add_argument("--power-t", type=int, default=4044)
    ap.add_argument("--power-seeds", type=int, default=40)
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--betas", type=float, nargs="*",
                    default=[0.0, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.012])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "crucible_corrected_contract"))
    args = ap.parse_args()

    fit_cfg = _load_fit_cfg(Path(args.config))
    cc = CorrectedConfig.from_yaml(args.corr_config)
    e1_t, e1_seeds = (760, 60) if args.quick else (args.e1_t, args.e1_seeds)
    pw_t, pw_seeds = (760, 8) if args.quick else (args.power_t, args.power_seeds)
    betas = [0.0, 0.004, 0.012] if args.quick else [float(b) for b in args.betas]

    report: dict = {"ts": datetime.now(timezone.utc).isoformat(), "quick": args.quick,
                    "corr_gates": args.corr_config, "funnel_gates_for_mechanics": args.config,
                    "corrected_thresholds": {"t_min": cc.t_min, "n_eff_mode": cc.n_eff_mode,
                                             "p_value_model": cc.p_value_model}}
    if args.exp in ("e1", "both"):
        log.info("E1 corrected-contract null-FPR: %d null candidates (T=%d)", e1_seeds, e1_t)
        report["e1"] = run_e1_corrected(fit_cfg, cc, t=e1_t, n=args.n, n_seeds=e1_seeds, cost_bps=_COST_BPS)
    if args.exp in ("power", "both"):
        log.info("Power (shipped vs corrected): %d betas x %d seeds (T=%d)", len(betas), pw_seeds, pw_t)
        report["power"] = run_power_corrected(fit_cfg, cc, t=pw_t, n=args.n, betas=betas,
                                              n_seeds=pw_seeds, cost_bps=_COST_BPS, gen_n_eff=_GEN_N_EFF)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "corrected_calibration_quick" if args.quick else "corrected_calibration"
    out_path = out_dir / f"{stem}_{args.exp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n============ CRUCIBLE CORRECTED-CONTRACT CALIBRATION ============")
    if "e1" in report:
        e = report["e1"]
        print(f"E1 null-FPR (T={e['t']}, {e['n_candidates']} null candidates, t_min={e['t_min']}, "
              f"n_eff={e['n_eff_mode']}):")
        print(f"   t-stat alone : FPR={e['t_stat_fpr']:.4f} (CP<={e['t_stat_fpr_cp_upper95']:.4f}) "
              f"vs nominal {e['nominal_one_sided_level']:.4f} -> calibrated={e['t_stat_calibrated']}")
        print(f"   full contract: FPR={e['full_contract_fpr']:.4f} (CP<={e['full_contract_fpr_cp_upper95']:.4f}) "
              f"-> green={e['full_contract_green']}")
    if "power" in report:
        pw = report["power"]
        print(f"POWER shipped vs corrected (T={pw['t']}):")
        print(f"   {'beta':>7} {'realized_dSR':>13} {'shipped':>8} {'corrected':>10}")
        for r in pw["rows"]:
            print(f"   {r['beta']:>7.4f} {r['realized_delta_sr']:>13.3f} {r['shipped_power']:>8.2f} "
                  f"{r['corrected_power']:>10.2f}")
        print(f"   @ realized dSR 0.5: corrected={pw['corrected_at_dsr_0.5']} shipped={pw['shipped_at_dsr_0.5']}")
        print(f"   @ realized dSR 0.8: corrected={pw['corrected_at_dsr_0.8']}")
    print(f"report -> {out_path}")
    print("================================================================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
