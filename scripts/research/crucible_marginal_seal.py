"""Crucible F3 figure — the ``marginal_t`` SUBSTITUTION-RESIDUAL seal (independent-audit finding F1).

WHY THIS EXISTS (S553-cont-139, anatomy-paper build plan §4 figure F3)
----------------------------------------------------------------------
The shipped funnel's significance leg is ``marginal_t`` — a t-stat on the marginal contribution stream
``marg = b_aug − b_base`` (``fitness.py:combination_fitness``). The Crucible independent audit
(``docs/research/crucible_independent_audit_report_2026-07-14.md`` §2.1 / finding F1) proved this leg is
STRUCTURALLY MIS-SPECIFIED: under the convex sum-to-1 inverse-vol combiner (``dynamic_sleeve_alphas``),

    marg == w_c · (r_c − b_base)          (an IDENTITY on the shipped path; audit max dev 1.7e-18)

so ``E[marg] = w̄_c · (μ_cand − μ_base_book)`` — a **substitution residual**. The leg therefore demands the
candidate *out-mean the book it joins*, not that it *contribute Sharpe*. Two consequences, both DEMONSTRATED
here on synthetic data (no ledger, fully reproducible):

  1. SIGN INVERSION — a genuine variance-reducing diversifier that RAISES the book Sharpe
     (``delta_sr_oos`` > 0) but whose own mean sits below the base book earns a NEGATIVE ``marginal_t``.
     It is the exact pathology the audit found on all 30 recorded cross_asset+overlay candidates
     (ΔSR > 0, t < 0), and it is pinned as a KNOWN seal in
     ``tests/signals/test_tier_c_seals_registered.py::test_f1_marginal_t_is_sign_inverted...``.
  2. SCALE DEPENDENCE — the SAME candidate signal at different leverage produces a different ``marginal_t``,
     crossing zero (audit: t = −1.67 at 0.25×, +1.81 at 8×). A coherent significance test cannot flip sign
     with a pure position-size rescale.

This is NOT a pass/fail gate and touches ZERO funnel/gate bytes (CRU-1 safe): it reads the shipped
``combination_fitness`` / ``_combined_book`` / ``dynamic_sleeve_alphas`` and reports what they do.

Usage (from repo root):
    python scripts/research/crucible_marginal_seal.py
    python scripts/research/crucible_marginal_seal.py --out results/crucible_marginal_seal
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sharpen.envs.allocator_factory import dynamic_sleeve_alphas  # noqa: E402
from sharpen.signals.generation.fitness import (  # noqa: E402
    _CAND,
    _combined_book,
    FitnessConfig,
    combination_fitness,
)

log = logging.getLogger("crucible_marginal_seal")

# Construction (K=900 business days, ~3.5y). The base book is positive-mean (two vol-scaled sleeves,
# ann Sharpe ~0.6, mean ~0.128/yr); the diversifier is INDEPENDENT of it, with mean BELOW the base
# book (0.04/yr) but a solid OWN Sharpe (1.8) — a real variance-reducer that nonetheless "loses" the
# mean-substitution test. The SAME construction as the pinned seal (test_tier_c_seals_registered.py
# ::test_f1_...: independent, low-mean-vs-book diversifier), but with the own-Sharpe/mean CONTROLLED
# (noise demeaned) so the leverage axis spans a legible −2.3 → +1.9 that reproduces the audit's
# scale-dependence datapoint (t = −1.67 @ 0.25×, +1.81 @ 8×); here t = −1.68 @ 1×, +1.86 @ 8×.
_K_DEFAULT = 900
_BASE_SEED = 0
_DIV_SEED = 7
_CAND_MEAN_ANN = 0.04           # candidate mean (annualized) — BELOW the ~0.128/yr base book -> E[marg]<0
_CAND_OWN_SHARPE_ANN = 1.8      # candidate's own annualized Sharpe (it is a GOOD standalone stream)
_LEVERAGES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def _ts(k: int) -> np.ndarray:
    """Business-day timestamps (seconds) — matches the combiner's monthly-meta cadence expectations."""
    return (np.datetime64("2014-01-02")
            + np.arange(k) * np.timedelta64(1, "D")).astype("datetime64[s]").astype(np.float64)


def _base_book_sleeves(k: int, *, seed: int = _BASE_SEED) -> dict[str, np.ndarray]:
    """Positive-mean base book: two independent vol-scaled sleeves (the base a candidate must beat)."""
    rng = np.random.default_rng(seed)
    return {"tsmom": 0.0008 + 0.010 * rng.standard_normal(k),
            "rates_carry": 0.0006 + 0.008 * rng.standard_normal(k)}


def _diversifier(k: int, *, seed: int = _DIV_SEED, mean_ann: float = _CAND_MEAN_ANN,
                 own_sharpe_ann: float = _CAND_OWN_SHARPE_ANN) -> np.ndarray:
    """A genuine diversifier: independent of the base, mean BELOW the base book (so it reduces
    combined variance / raises book Sharpe while failing the mean-substitution test). The noise is
    DEMEANED so the realized mean/own-Sharpe hit their targets (a controlled, reproducible instance —
    the pathology is construction-independent; this just makes the leverage axis legible)."""
    mu = mean_ann / 252.0
    sigma = mu * np.sqrt(252.0) / own_sharpe_ann          # own_sharpe_ann = mu/sigma * sqrt(252)
    z = np.random.default_rng(seed).standard_normal(k)
    z = z - z.mean()                                       # pin realized mean to mu (kill sample drift)
    return mu + sigma * z


def _ann_sharpe(x: np.ndarray, ppy: int = 252) -> float:
    x = x[np.isfinite(x)]
    if x.size < 2 or x.std() == 0.0:
        return float("nan")
    return float(x.mean() / x.std() * np.sqrt(ppy))


def _score(div: np.ndarray, base: dict[str, np.ndarray], ts: np.ndarray,
           cfg: FitnessConfig) -> Any:
    return combination_fitness(div, base, ts, cfg, gen_n_eff=50, turnover_ann=1.0, n_nodes=3)


# ---------------------------------------------------------------- (1) sign inversion (the headline)
def demo_sign_inversion(cfg: FitnessConfig, *, k: int) -> dict:
    ts = _ts(k)
    base = _base_book_sleeves(k)
    div = _diversifier(k)
    res = _score(div, base, ts, cfg)
    b_base = _combined_book(base, ts, cfg)
    base_book_mean_ann = float(np.nanmean(b_base) * 252)
    cand_mean_ann = float(np.nanmean(div) * 252)
    sign_inverted = bool(res.delta_sr_oos > 0.0 and res.marginal_t < 0.0)
    return {
        "delta_sr_oos": float(res.delta_sr_oos),               # book Sharpe uplift (the diversifier WORKS)
        "marginal_t": float(res.marginal_t),                   # ...yet the significance leg is NEGATIVE
        "cand_hlz_pass": bool(res.cand_hlz_pass),              # never passes t >= hlz_t_min
        "hlz_t_min": float(cfg.hlz_t_min),
        "base_book_mean_ann": base_book_mean_ann,              # candidate mean < base book mean -> E[marg]<0
        "cand_mean_ann": cand_mean_ann,
        "cand_own_sharpe_ann": _ann_sharpe(div),               # the candidate has POSITIVE own Sharpe
        "sign_inverted": sign_inverted,
    }


# ---------------------------------------------------------------- (2) the identity (max dev ~1e-18)
def demo_identity(cfg: FitnessConfig, *, k: int) -> dict:
    """Reproduce the audit's identity: on the shipped combiner path, ``b_aug − b_base`` equals
    ``w_c·(r_c − b_base)`` to machine precision, where ``w_c`` is the candidate's inverse-vol combiner
    weight. ``w_c`` is read from the SAME ``dynamic_sleeve_alphas`` call ``_combined_book`` makes."""
    ts = _ts(k)
    base = _base_book_sleeves(k)
    div = _diversifier(k)
    aug = {**base, _CAND: div}
    # The candidate's per-bar combiner weight w_c — identical args to _combined_book(aug, ...).
    alphas = dynamic_sleeve_alphas(
        aug, ts, window=cfg.combiner_window, min_periods=cfg.combiner_min_periods,
        monthly_meta=cfg.combiner_monthly_meta, target_portfolio_vol=None,
        tilt_strength=cfg.tilt_strength, perf_window=cfg.perf_window,
        perf_min_periods=cfg.perf_min_periods, tilt_clip=cfg.tilt_clip,
        redundancy_strength=cfg.combiner_redundancy_strength)
    w_c = np.asarray(alphas[_CAND], dtype=np.float64)
    b_base = _combined_book(base, ts, cfg)
    b_aug = _combined_book(aug, ts, cfg)
    marg = b_aug - b_base
    rhs = w_c * (div - b_base)
    mask = np.isfinite(marg) & np.isfinite(rhs)
    max_abs_dev = float(np.max(np.abs(marg[mask] - rhs[mask]))) if mask.any() else float("nan")
    return {
        "max_abs_dev": max_abs_dev,
        "identity_holds": bool(max_abs_dev < 1e-10),
        "n_finite": int(mask.sum()),
        "w_c_mean": float(np.nanmean(w_c)),                    # the candidate's mean combiner weight
        "note": "marg == w_c*(r_c - b_base) on the convex sum-to-1 inverse-vol path (tilt/redundancy off)",
    }


# ---------------------------------------------------------------- (3) scale dependence (t sign flip)
def demo_scale_sweep(cfg: FitnessConfig, *, k: int, leverages=_LEVERAGES) -> dict:
    """The SAME diversifier signal rescaled by leverage lambda. marginal_t crosses zero as lambda grows
    (low lev: residual is base-book-dominated -> t<0; high lev: candidate-dominated -> t>0). delta_sr_oos
    is reported alongside to expose the incoherence (a position-size rescale is not a signal change)."""
    ts = _ts(k)
    base = _base_book_sleeves(k)
    div = _diversifier(k)
    rows: list[dict[str, Any]] = []
    for lam in leverages:
        res = _score(div * float(lam), base, ts, cfg)
        rows.append({
            "leverage": float(lam),
            "marginal_t": float(res.marginal_t),
            "delta_sr_oos": float(res.delta_sr_oos),
            "cand_hlz_pass": bool(res.cand_hlz_pass),
        })
    ts_vals = [r["marginal_t"] for r in rows if np.isfinite(r["marginal_t"])]
    t_min_lev = rows[0]["marginal_t"]
    t_max_lev = rows[-1]["marginal_t"]
    return {
        "leverages": [float(x) for x in leverages],
        "rows": rows,
        "t_at_min_leverage": float(t_min_lev),
        "t_at_max_leverage": float(t_max_lev),
        "t_sign_flips": bool(len(ts_vals) >= 2 and min(ts_vals) < 0.0 < max(ts_vals)),
        "note": "same signal, different leverage: marginal_t is not scale-invariant and changes sign",
    }


# ------------------------------------------------------------------ driver
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Crucible F3: the marginal_t substitution-residual seal (audit F1)")
    ap.add_argument("--k", type=int, default=_K_DEFAULT, help="panel length in business days")
    ap.add_argument("--embargo", type=int, default=10, help="CPCV embargo (matches the pinned seal test)")
    ap.add_argument("--out", default=str(ROOT / "results" / "crucible_marginal_seal"))
    args = ap.parse_args()

    cfg = FitnessConfig(embargo=args.embargo)          # all combiner params at shipped defaults
    ts = datetime.now(timezone.utc).isoformat()
    report: dict = {
        "artifact": "crucible_marginal_seal_F3",
        "ts": ts,
        "audit_ref": "independent audit F1 (2026-07-14 §2.1); anatomy-paper figure F3",
        "cfg": {"k": args.k, "embargo": args.embargo, "hlz_t_min": cfg.hlz_t_min,
                "tilt_strength": cfg.tilt_strength,
                "combiner_redundancy_strength": cfg.combiner_redundancy_strength},
        "sign_inversion": demo_sign_inversion(cfg, k=args.k),
        "identity": demo_identity(cfg, k=args.k),
        "scale_sweep": demo_scale_sweep(cfg, k=args.k),
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "marginal_seal.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    si = report["sign_inversion"]
    idn = report["identity"]
    sw = report["scale_sweep"]
    print("\n============= CRUCIBLE F3: marginal_t SUBSTITUTION-RESIDUAL SEAL =============")
    print("(1) SIGN INVERSION -- a genuine diversifier raises book Sharpe but gets a NEGATIVE t:")
    print(f"    delta_sr_oos (book uplift)  = {si['delta_sr_oos']:+.4f}  (>0: the diversifier WORKS)")
    print(f"    marginal_t                  = {si['marginal_t']:+.4f}  (<0: the leg REJECTS it; "
          f"pass t>={si['hlz_t_min']}? {si['cand_hlz_pass']})")
    print(f"    cand mean {si['cand_mean_ann']:+.3f} < base-book mean {si['base_book_mean_ann']:+.3f} "
          f"(ann)  ->  E[marg]<0.  cand own Sharpe = {si['cand_own_sharpe_ann']:+.2f} (ann)")
    print(f"    SIGN-INVERTED (dSR>0 & t<0) = {si['sign_inverted']}")
    print(f"(2) IDENTITY  marg == w_c*(r_c - b_base): max|dev| = {idn['max_abs_dev']:.2e} "
          f"over {idn['n_finite']} bars -> holds={idn['identity_holds']}")
    print("(3) SCALE DEPENDENCE -- same signal, marginal_t flips sign with leverage:")
    print(f"    {'leverage':>9} {'marginal_t':>11} {'delta_sr_oos':>13} {'pass t>=hlz':>12}")
    for r in sw["rows"]:
        print(f"    {r['leverage']:>9.2f} {r['marginal_t']:>+11.3f} {r['delta_sr_oos']:>+13.4f} "
              f"{str(r['cand_hlz_pass']):>12}")
    print(f"    t sign flips across leverage = {sw['t_sign_flips']} "
          f"({sw['t_at_min_leverage']:+.2f} @ {sw['leverages'][0]}x -> "
          f"{sw['t_at_max_leverage']:+.2f} @ {sw['leverages'][-1]}x)")
    print(f"report -> {out_path}")
    print("=============================================================================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
