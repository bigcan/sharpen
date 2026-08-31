"""Fixed-lot stress pass — Protocol v2 Stage 3 stress sub-report (audit X3 / P7-03 / P7-07).

Canonical (pipeline) implementation. Used by:
  - ``scripts/sg1_btc_velotrade_ensemble_eval.py`` — WF-eval ``stress`` verdict block (X3)

Supersedes the frozen local copy of ``reconstruct_fold`` / ``_max_drawdown_*`` in the
standalone P7-03 analysis ``scripts/sg1_btc_fixed_notional_backtest.py`` (S553-cont-8);
that script retains its own copy so its already-committed outputs stay reproducible.
This module adds the early-term guard exception, ``peak_abs_position``, and the
multi-fold ``compute_stress_subreport`` aggregator the WF eval needs.

**Why this exists (protocol_v2.md Stage 3 + the S466 SG-1-XAUUSD incident).** The engine
runs a buffered early-termination (8% trailing) so the recorded per-fold *trailing DD can
be truncated*, which makes the Velotrade G4 buffer near non-binding (audit P7-07: a 1.09%
apparent DD was a 3-sim-day artifact; the fixed-lot replay revealed 4.59%). The stress pass
replays the full window with a **fixed lot size and NO prop-firm early-term truncation** and
gates the honest worst DD + peak leverage.

**Faithfulness (S553-cont-8).** When no fold early-terminated (``prop_firm_termination``
all-null — the sg1-btc case), the recorded path *is* the un-truncated path, so the fixed-lot
equity reconstructs exactly from the recorded per-bar returns — no re-rollout needed.
:func:`reconstruct_fold` HARD-FAILS (``EarlyTerminatedFold``) if a fold did early-terminate,
because a reconstruction would understate DD; that fold must be re-rolled with termination
disabled. :func:`compute_stress_subreport` catches that and marks the block
``INCOMPLETE_NEEDS_REROLLOUT`` rather than reporting a truncated path as honest.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class EarlyTerminatedFold(RuntimeError):
    """A fold's recorded trajectory was early-terminated → reconstruction would
    understate DD. Subclasses RuntimeError so existing ``except RuntimeError``
    callers (the standalone P7-03 script) keep their abort behavior."""


def _max_drawdown_of_curve(curve: np.ndarray) -> float:
    """Max trailing peak-to-trough of an equity curve, as a fraction (<=0)."""
    if len(curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(curve)
    return float(np.min(curve / np.maximum(peak, 1e-12)) - 1.0)


def _max_drawdown_of_cumret(cumret: np.ndarray) -> float:
    """Max peak-to-trough decline of a cumulative-return curve, in units of the
    fixed notional (i.e. fraction OF INITIAL capital, not of running equity).
    Returns a non-positive fraction."""
    if len(cumret) == 0:
        return 0.0
    running_peak = np.maximum.accumulate(cumret)
    return float(np.min(cumret - running_peak))


def reconstruct_fold(parquet_path: Path, initial_balance: float,
                     recorded_metrics: Optional[dict]) -> dict:
    """Reconstruct one fold's fixed-notional (un-compounded) path from its
    recorded trajectory parquet.

    Raises :class:`EarlyTerminatedFold` if the fold's recorded path was
    truncated by a prop-firm early-termination (reconstruction would understate
    DD). Otherwise returns compounded + fixed-notional return/DD/PF metrics, with
    PF/return cross-checks against ``recorded_metrics`` when supplied.
    """
    df = pd.read_parquet(parquet_path)
    pv = df["portfolio_value"].to_numpy(dtype=np.float64)
    n_bars = len(pv)
    if n_bars < 2:
        raise ValueError(f"{parquet_path}: <2 bars, cannot reconstruct")

    # Per-bar fractional return on the compounded curve == equity-independent r_t.
    r = np.diff(pv) / pv[:-1]

    # --- Integrity guard: no early termination (P7-03) --------------------
    term = None
    if "prop_firm_termination" in df.columns:
        nz = df["prop_firm_termination"].dropna()
        term = (nz.iloc[-1] if len(nz) else None)
    if term is not None:
        raise EarlyTerminatedFold(
            f"{parquet_path}: prop_firm_termination={term!r} -> trajectory was "
            f"EARLY-TERMINATED; a faithful no-early-term fixed-lot stress pass "
            f"requires re-running the env with termination disabled. Reconstruction "
            f"would understate DD. Aborting (do not report a truncated path as honest)."
        )

    # --- Compounded (recorded) -------------------------------------------
    comp_return_pct = float((pv[-1] - pv[0]) / pv[0] * 100.0)
    comp_mdd_pct = _max_drawdown_of_curve(pv) * 100.0

    # --- Fixed-notional (reconstructed) ----------------------------------
    cumret = np.cumsum(r)                       # cum PnL / initial
    E_fixed = initial_balance * (1.0 + np.concatenate([[0.0], cumret]))
    fixed_return_pct = float(np.sum(r) * 100.0)
    # DD as % of running peak EQUITY (same definition as compute_gate_metrics)
    fixed_mdd_peak_pct = _max_drawdown_of_curve(E_fixed) * 100.0
    # DD as % of INITIAL notional (cleaner fixed-lot risk unit; <=0)
    fixed_mdd_initial_pct = _max_drawdown_of_cumret(cumret) * 100.0

    # PF from r_t (compounding-invariant) — cross-check vs recorded.
    wins = r[r > 0]
    losses = r[r < 0]
    gp = float(wins.sum()) if len(wins) else 0.0
    gl = float(-losses.sum()) if len(losses) else 0.0
    pf = gp / gl if gl > 1e-12 else (10.0 if gp > 1e-12 else 0.0)

    pf_recorded = (recorded_metrics or {}).get("pf_bar")
    comp_ret_recorded = (recorded_metrics or {}).get("total_return_pct")
    pf_xcheck_ok = (pf_recorded is None) or (abs(pf - pf_recorded) < 1e-6)
    ret_xcheck_ok = (comp_ret_recorded is None) or (abs(comp_return_pct - comp_ret_recorded) < 1e-3)

    # Peak leverage = max |position| over the fold (max_leverage units; X3 gate).
    peak_abs_position: Optional[float] = None
    if "position" in df.columns:
        pos = df["position"].to_numpy(dtype=np.float64)
        if pos.size:
            peak_abs_position = float(np.max(np.abs(pos)))

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce") if "timestamp" in df.columns else None
    ts_start = str(ts.dropna().iloc[0]) if (ts is not None and ts.notna().any()) else None
    ts_end = str(ts.dropna().iloc[-1]) if (ts is not None and ts.notna().any()) else None

    return {
        "n_bars": n_bars,
        "ts_start": ts_start,
        "ts_end": ts_end,
        "pf_bar": pf,
        "comp_return_pct": comp_return_pct,
        "fixed_return_pct": fixed_return_pct,
        "compounding_inflation_x": (comp_return_pct / fixed_return_pct
                                    if abs(fixed_return_pct) > 1e-9 else None),
        "comp_trailing_mdd_pct": comp_mdd_pct,
        "fixed_trailing_mdd_peak_pct": fixed_mdd_peak_pct,
        "fixed_mdd_of_initial_pct": fixed_mdd_initial_pct,
        "fixed_return_over_mdd": (fixed_return_pct / abs(fixed_mdd_initial_pct)
                                  if abs(fixed_mdd_initial_pct) > 1e-9 else None),
        "peak_abs_position": peak_abs_position,
        "pf_xcheck_ok": pf_xcheck_ok,
        "ret_xcheck_ok": ret_xcheck_ok,
    }


def compute_stress_subreport(
    fold_parquets: List[Tuple[str, Path]],
    *,
    gates: Dict[str, Any],
    initial_balance: float,
    trailing_cap_pct: float,
    graded_rule: str,
) -> Dict[str, Any]:
    """Aggregate the fixed-lot stress pass into a WF-verdict ``stress`` block.

    ``fold_parquets`` is ``[(fold_id, trajectory_parquet_path), ...]`` for the
    GRADED rule (the Stage-2.5 chosen rule / solo). Gates (protocol §Stage 3):
      - ``gates.stress_dd_buffer_pp`` (default 0.5pp): headroom of the honest
        worst fixed-lot trailing DD from ``trailing_cap_pct`` must be >= this.
      - ``gates.stress_leverage_max`` (default 1.0): peak ``|position|`` must be <=.

    A real breach → ``status="FAIL"`` (caller flips the WF verdict). A fold that
    early-terminated → that fold is recorded under ``early_terminated_folds`` and
    the block is ``status="INCOMPLETE_NEEDS_REROLLOUT"`` (NOT a pass, NOT a breach
    — a faithfulness gap requiring a termination-disabled re-rollout).
    """
    dd_buffer_min = float(gates.get("stress_dd_buffer_pp", 0.5))
    lev_max = float(gates.get("stress_leverage_max", 1.0))

    per_fold: List[Dict[str, Any]] = []
    incomplete: List[Dict[str, str]] = []
    worst_dd = 0.0          # most-negative fixed trailing DD across folds (<=0)
    peak_lev = 0.0
    lev_known = False

    for fold_id, p in fold_parquets:
        try:
            rec = reconstruct_fold(Path(p), initial_balance, None)
        except EarlyTerminatedFold as e:
            incomplete.append({"fold": fold_id, "reason": str(e)})
            continue
        dd = float(rec["fixed_trailing_mdd_peak_pct"])      # <=0
        lev = rec.get("peak_abs_position")
        per_fold.append({
            "fold": fold_id,
            "n_bars": rec["n_bars"],
            "fixed_trailing_mdd_pct": dd,
            "fixed_mdd_of_initial_pct": rec["fixed_mdd_of_initial_pct"],
            "peak_abs_position": lev,
        })
        worst_dd = min(worst_dd, dd)
        if lev is not None:
            peak_lev = max(peak_lev, float(lev))
            lev_known = True

    n_eval = len(per_fold)
    dd_buffer_pp = trailing_cap_pct - abs(worst_dd)         # headroom from cap
    dd_pass = (n_eval > 0) and (dd_buffer_pp >= dd_buffer_min)
    # Leverage gate fails ONLY on a KNOWN breach. An absent ``position`` column is
    # a data-completeness gap (surfaced via ``leverage_known``), NOT a risk breach —
    # it must not flip the verdict to FAIL.
    lev_breach = lev_known and (peak_lev > lev_max)

    if incomplete:
        status = "INCOMPLETE_NEEDS_REROLLOUT"
    elif n_eval == 0:
        status = "SKIPPED"
    elif (not dd_pass) or lev_breach:
        status = "FAIL"
    else:
        status = "PASS"

    return {
        "graded_rule": graded_rule,
        "trailing_cap_pct": trailing_cap_pct,
        "n_folds_evaluated": n_eval,
        "worst_fixed_trailing_mdd_pct": worst_dd,
        "stress_dd_buffer_pp": dd_buffer_pp,
        "stress_dd_buffer_pp_min": dd_buffer_min,
        "stress_dd_pass": dd_pass,
        "peak_leverage": peak_lev if lev_known else None,
        "leverage_known": lev_known,
        "stress_leverage_max": lev_max,
        "stress_leverage_pass": not lev_breach,
        "early_terminated_folds": incomplete,
        "per_fold": per_fold,
        "stress_pass": status == "PASS",
        "status": status,
        "note": (
            "recorded path = un-truncated (no early-term fired) → fixed-lot "
            "reconstruction is exact"
            if not incomplete
            else "one or more folds early-terminated — re-rollout with prop-firm "
            "termination disabled required for an honest DD on those folds"
        ),
    }
