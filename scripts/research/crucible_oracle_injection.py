"""Crucible F2 (oracle half) — the shipped gate rejects PERFECT FORESIGHT (independent-audit §2 S-1/S-2).

WHY THIS EXISTS (S553-cont-139, anatomy-paper build plan §4 figure F2)
----------------------------------------------------------------------
The flagship figure's most damning half: the shipped Crucible gate cannot detect a signal handed to it
on a silver platter. The independent audit (``docs/research/crucible_independent_audit_report_2026-07-14``
§2 "(A)" / findings F1+F2) planted a PERFECT-FORESIGHT timing oracle and found the gate rejects it —
"a machine that rejects perfect weekly foresight cannot return a plausible edge." The auditor's
``scratchpad/phase2/planted_sweep.py`` is gone; this rebuilds it as a reproducible experiment.

THE DEMONSTRATION — the gate's promotion bar is an IMPLAUSIBLE realized Sharpe
A timing oracle knows the SIGN of a synthetic market's next-block return with skill ``p`` (p=0.5 = no
skill, p=1.0 = perfect foresight), holds that position for ``hold`` bars, and earns ``pos·market``. It
is scored through the SHIPPED ``combination_fitness`` against a ~0-Sharpe base book (the un-sign-sealed
regime, matching the real Taiwan train book SR ≈ −0.007). Two sweeps:

  (1) HOLD sweep at PERFECT skill (p=1.0): the oracle's realized Sharpe falls as the hold lengthens
      (daily ~18 → semiannual ~1.1). The gate promotes an oracle ONLY above a realized-Sharpe WALL of
      ≈4 — so every perfect oracle at monthly-or-slower frequency (realized SR ≤ ~2.7) is REJECTED,
      first on the book-level ``dsr_aug`` seal (F2), then also on the ``marginal_t`` substitution-
      residual seal (F1). A promotion bar at realized SR ≈4 is one no plausible strategy can meet.
  (2) SKILL sweep at a REALISTIC-edge frequency (quarterly hold; perfect-foresight realized SR ~1.5,
      bracketing the audit's real-substrate weekly oracle SR 1.80): detection stays 0/N at EVERY skill
      level from p=0.52 to p=1.00 — PERFECT foresight at a realistic realized Sharpe is never promoted.

Reproduces the audit's structural claim (a perfect-foresight oracle at a realistic realized Sharpe is
rejected; only near-daily foresight passes) and the daily-oracle Sharpe (~18 vs audit 12.6). The
audit's WEEKLY oracle SR (1.80) is substrate-specific — a realistic market caps weekly timing far lower
than this idealized iid market (SR 5.7). The INVARIANT the figure isolates is the gate's realized-Sharpe
WALL (≈4): the achievable-oracle-Sharpe is substrate-dependent, the wall is not, and it sits far above
both any plausible edge (SR 0.3–0.8) and the audit's real perfect-weekly oracle (SR 1.80).

Zero funnel/gate bytes touched (CRU-1 safe): reads ``combination_fitness`` as shipped.

Usage (from repo root):
    python scripts/research/crucible_oracle_injection.py
    python scripts/research/crucible_oracle_injection.py --quick
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

from sharpen.signals.generation.fitness import FitnessConfig, combination_fitness  # noqa: E402

log = logging.getLogger("crucible_oracle_injection")

# Hold periods (business days): daily / weekly / monthly / quarterly / semiannual — the oracle's
# realized Sharpe falls as the hold lengthens (fewer, coarser bets).
_HOLDS = ((1, "daily"), (5, "weekly"), (21, "monthly"), (63, "quarterly"), (126, "semiannual"))
_SKILLS = (0.52, 0.60, 0.70, 0.85, 1.00)
_T = 1011                # ~Taiwan holdout length (the audit's marginal_t arithmetic reference)
_N_SEEDS = 20
_COST_BPS = 0.0010
_GEN_N_EFF = 50.0        # file-drawer N for a single planted hypothesis (matches E2 gate_probe_gen_n_eff)
# Skill sweep runs at a REALISTIC-edge frequency: quarterly perfect-foresight realizes SR ~1.5, which
# brackets the audit's real-substrate weekly oracle (SR 1.80) — a Sharpe below the gate's promotion wall.
_SKILL_SWEEP_HOLD = 63

# Audit cross-check markers (independent audit §2 "(A)"); NOT reproduced here (real substrate absent).
AUDIT_WEEKLY_SR = 1.80
AUDIT_WEEKLY_MARGINAL_T = 2.92
AUDIT_DAILY_SR = 12.6


def _ts(t: int) -> np.ndarray:
    return (np.datetime64("2014-01-02")
            + np.arange(t) * np.timedelta64(1, "D")).astype("datetime64[s]").astype(np.float64)


def _market(t: int, *, seed: int) -> np.ndarray:
    """A synthetic tradable market: zero-drift iid daily returns (the thing the oracle times)."""
    return 0.01 * np.random.default_rng(seed).standard_normal(t)


def _zero_sharpe_base(t: int, *, seed: int) -> dict[str, np.ndarray]:
    """~0-Sharpe base book (the un-sign-sealed regime; matches the real Taiwan train book SR ≈ −0.007)."""
    rng = np.random.default_rng(seed)
    return {"tsmom": 0.010 * rng.standard_normal(t), "rates_carry": 0.008 * rng.standard_normal(t)}


def _oracle_return(market: np.ndarray, *, hold: int, skill_p: float, seed: int,
                   cost_bps: float) -> tuple[np.ndarray, float]:
    """A ``hold``-bar timing oracle on ``market``. ``skill_p`` is the DIRECTIONAL ACCURACY: every block
    the position equals the sign of the block's OWN (future) return with probability ``skill_p``, else
    the OPPOSITE sign — so p=1.0 = perfect foresight, p=0.5 = no skill (SR ~0), matching the audit's
    'p=0.52 to 1.00' sweep. Earns ``pos·market`` net of a turnover charge at each rebalance."""
    rng = np.random.default_rng(seed)
    t = market.size
    r = np.zeros(t, dtype=np.float64)
    prev_pos = 0.0
    turnover = 0.0
    for s in range(0, t, hold):
        e = min(s + hold, t)
        blk = market[s:e]
        true_sign = np.sign(blk.sum())
        true_sign = true_sign if true_sign != 0 else 1.0
        pos = true_sign if rng.random() < skill_p else -true_sign     # accuracy == skill_p
        r[s:e] = pos * blk
        r[s] -= cost_bps * abs(pos - prev_pos)      # rebalance cost on the switch bar
        turnover += abs(pos - prev_pos)
        prev_pos = pos
    turnover_ann = turnover / (t / 252.0)
    return r, turnover_ann


def _ann_sharpe(x: np.ndarray, ppy: int = 252) -> float:
    x = x[np.isfinite(x)]
    if x.size < 2 or x.std() == 0.0:
        return float("nan")
    return float(x.mean() / x.std() * np.sqrt(ppy))


def _legs(r: Any, cfg: FitnessConfig) -> dict[str, bool]:
    """The 5 AND-legs of the shipped gate (mirrors combination_fitness.passes_gate / LegTally)."""
    return {
        "uplift": bool(np.isfinite(r.delta_sr_oos) and r.delta_sr_oos >= cfg.min_combination_uplift),
        "dsr": bool(np.isfinite(r.dsr_aug) and r.dsr_aug >= cfg.promising_dsr),
        "marginal_t": bool(r.cand_hlz_pass),
        "not_redundant": bool(np.isfinite(r.max_base_corr_obs) and r.max_base_corr_obs <= cfg.max_base_corr),
        "not_fragile": bool(np.isfinite(r.delta_sr_median) and r.delta_sr_median >= cfg.delta_median_min
                            and np.isfinite(r.frac_paths_positive)
                            and r.frac_paths_positive >= cfg.frac_positive_min),
    }


def _score(market: np.ndarray, base: dict[str, np.ndarray], ts: np.ndarray, cfg: FitnessConfig,
           *, hold: int, skill_p: float, seed: int, cost_bps: float, gen_n_eff: float) -> dict:
    r, turnover_ann = _oracle_return(market, hold=hold, skill_p=skill_p, seed=seed, cost_bps=cost_bps)
    res = combination_fitness(r, base, ts, cfg, gen_n_eff=gen_n_eff, turnover_ann=turnover_ann, n_nodes=1)
    legs = _legs(res, cfg)
    return {
        "own_sharpe": _ann_sharpe(r),
        "delta_sr_oos": float(res.delta_sr_oos),
        "marginal_t": float(res.marginal_t),
        "dsr_aug": float(res.dsr_aug),
        "passes_gate": bool(res.passes_gate),
        "legs": legs,
        "failing_legs": [k for k, ok in legs.items() if not ok],
    }


def _aggregate(cfg: FitnessConfig, *, t: int, hold: int, skill_p: float, n_seeds: int,
               cost_bps: float, gen_n_eff: float) -> dict:
    ts = _ts(t)
    rows = []
    for k in range(n_seeds):
        market = _market(t, seed=3000 + k)
        base = _zero_sharpe_base(t, seed=7000 + k)
        rows.append(_score(market, base, ts, cfg, hold=hold, skill_p=skill_p, seed=100 + k,
                           cost_bps=cost_bps, gen_n_eff=gen_n_eff))
    det = float(np.mean([row["passes_gate"] for row in rows]))
    # which legs fail most often among the seeds (the binding wall)
    fail_counts: dict[str, int] = {}
    for row in rows:
        for leg in row["failing_legs"]:
            fail_counts[leg] = fail_counts.get(leg, 0) + 1
    binding = sorted(fail_counts, key=lambda leg: fail_counts[leg], reverse=True)
    return {
        "hold": hold, "skill_p": skill_p, "n_seeds": n_seeds,
        "mean_own_sharpe": float(np.nanmean([row["own_sharpe"] for row in rows])),
        "mean_marginal_t": float(np.nanmean([row["marginal_t"] for row in rows])),
        "mean_dsr_aug": float(np.nanmean([row["dsr_aug"] for row in rows])),
        "detection_rate": det,
        "detections": int(round(det * n_seeds)),
        "binding_fail_legs": binding,       # legs that gate the (failing) oracle, most-frequent first
    }


def _promotion_wall(rows: list[dict]) -> dict:
    """The realized-Sharpe promotion wall: the boundary between the lowest-Sharpe perfect oracle the
    gate PROMOTES (detection > 0.5) and the highest-Sharpe one it REJECTS (detection <= 0.5)."""
    promoted = [r["mean_own_sharpe"] for r in rows if r["detection_rate"] > 0.5]
    rejected = [r["mean_own_sharpe"] for r in rows if r["detection_rate"] <= 0.5]
    lowest_promoted = min(promoted) if promoted else float("nan")
    highest_rejected = max(rejected) if rejected else float("nan")
    return {
        "lowest_promoted_sharpe": lowest_promoted,
        "highest_rejected_sharpe": highest_rejected,
        "wall_sharpe_approx": float(np.nanmean([lowest_promoted, highest_rejected])),
    }


def run_hold_sweep(cfg: FitnessConfig, *, t: int, n_seeds: int, cost_bps: float,
                   gen_n_eff: float) -> dict:
    """Perfect foresight (p=1.0) across hold periods: the gate promotes an oracle only above a
    realized-Sharpe WALL; every slower/lower-Sharpe perfect oracle fails, first on dsr_aug then also on
    marginal_t as its realized Sharpe drops."""
    rows = []
    for hold, label in _HOLDS:
        agg = _aggregate(cfg, t=t, hold=hold, skill_p=1.0, n_seeds=n_seeds, cost_bps=cost_bps,
                         gen_n_eff=gen_n_eff)
        agg["label"] = label
        rows.append(agg)
        log.info("  hold=%-11s SR=%5.2f marg_t=%5.2f dsr=%.2f detect=%.2f binds=%s",
                 label, agg["mean_own_sharpe"], agg["mean_marginal_t"], agg["mean_dsr_aug"],
                 agg["detection_rate"], agg["binding_fail_legs"])
    wall = _promotion_wall(rows)
    return {"skill_p": 1.0, "rows": rows,
            "holds_that_pass": [r["label"] for r in rows if r["detection_rate"] > 0.5],
            "promotion_wall": wall}


def run_skill_sweep(cfg: FitnessConfig, *, t: int, hold: int, n_seeds: int, cost_bps: float,
                    gen_n_eff: float) -> dict:
    """At a fixed REALISTIC-edge frequency (quarterly; perfect-foresight realized SR ~1.5, bracketing
    the audit's real weekly oracle SR 1.80), sweep skill p from 0.52 to perfect: detection stays 0/N
    throughout — a perfect-foresight oracle at a realistic realized Sharpe is never promoted."""
    rows = []
    for p in _SKILLS:
        agg = _aggregate(cfg, t=t, hold=hold, skill_p=p, n_seeds=n_seeds, cost_bps=cost_bps,
                         gen_n_eff=gen_n_eff)
        rows.append(agg)
        log.info("  p=%.2f (weekly): SR=%5.2f marg_t=%5.2f dsr=%.2f detect=%.2f",
                 p, agg["mean_own_sharpe"], agg["mean_marginal_t"], agg["mean_dsr_aug"],
                 agg["detection_rate"])
    return {"hold": hold, "rows": rows,
            "never_detected": all(r["detection_rate"] == 0.0 for r in rows),
            "max_detection_rate": max((r["detection_rate"] for r in rows), default=0.0)}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="Crucible F2 oracle injection: the gate rejects perfect foresight")
    ap.add_argument("--t", type=int, default=_T)
    ap.add_argument("--n-seeds", type=int, default=_N_SEEDS)
    ap.add_argument("--embargo", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="fewer seeds/shorter panel — smoke test.")
    ap.add_argument("--out", default=str(ROOT / "results" / "crucible_oracle_injection"))
    args = ap.parse_args()

    t = 520 if args.quick else args.t
    n_seeds = 5 if args.quick else args.n_seeds
    cfg = FitnessConfig(embargo=args.embargo)

    log.info("F2 oracle: hold sweep @ perfect skill (T=%d, %d seeds)", t, n_seeds)
    hold_sweep = run_hold_sweep(cfg, t=t, n_seeds=n_seeds, cost_bps=_COST_BPS, gen_n_eff=_GEN_N_EFF)
    log.info("F2 oracle: skill sweep @ realistic-edge (quarterly) hold")
    skill_sweep = run_skill_sweep(cfg, t=t, hold=_SKILL_SWEEP_HOLD, n_seeds=n_seeds, cost_bps=_COST_BPS,
                                  gen_n_eff=_GEN_N_EFF)

    report = {
        "artifact": "crucible_oracle_injection_F2",
        "ts": datetime.now(timezone.utc).isoformat(),
        "audit_ref": "independent audit §2 '(A)' / F1+F2 (2026-07-14); anatomy-paper figure F2 (oracle half)",
        "cfg": {"t": t, "n_seeds": n_seeds, "cost_bps": _COST_BPS, "gen_n_eff": _GEN_N_EFF,
                "hlz_t_min": cfg.hlz_t_min, "promising_dsr": cfg.promising_dsr,
                "base_book": "~0 train Sharpe (un-sign-sealed)"},
        "hold_sweep": hold_sweep,
        "skill_sweep": skill_sweep,
        "audit_markers": {"weekly_oracle_sr": AUDIT_WEEKLY_SR,
                          "weekly_oracle_marginal_t": AUDIT_WEEKLY_MARGINAL_T,
                          "daily_oracle_sr": AUDIT_DAILY_SR,
                          "note": "cross-check markers from the audit's real substrate; not reproduced here"},
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "oracle_injection_quick" if args.quick else "oracle_injection"
    out_path = out_dir / f"{stem}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    wall = hold_sweep["promotion_wall"]
    print("\n============ CRUCIBLE F2: THE GATE REJECTS PERFECT FORESIGHT ============")
    print("(1) HOLD sweep @ PERFECT skill (p=1.0) -- the gate's promotion bar is a realized-Sharpe WALL:")
    print(f"    {'hold':<12} {'own_SR':>7} {'marg_t':>7} {'dsr_aug':>8} {'detect':>7} {'binds_on':>22}")
    for r in hold_sweep["rows"]:
        print(f"    {r['label']:<12} {r['mean_own_sharpe']:>7.2f} {r['mean_marginal_t']:>7.2f} "
              f"{r['mean_dsr_aug']:>8.2f} {r['detection_rate']:>7.2f} "
              f"{(','.join(r['binding_fail_legs']) or '-- (PASS)'):>22}")
    print(f"    PROMOTION WALL ~ realized SR {wall['wall_sharpe_approx']:.1f} "
          f"(highest REJECTED perfect oracle SR {wall['highest_rejected_sharpe']:.2f}, "
          f"lowest PROMOTED SR {wall['lowest_promoted_sharpe']:.2f})")
    print("(2) SKILL sweep @ REALISTIC-edge (quarterly) hold -- perfect foresight never promoted:")
    print(f"    {'skill_p':>8} {'own_SR':>7} {'marg_t':>7} {'dsr_aug':>8} {'detect':>7}")
    for r in skill_sweep["rows"]:
        print(f"    {r['skill_p']:>8.2f} {r['mean_own_sharpe']:>7.2f} {r['mean_marginal_t']:>7.2f} "
              f"{r['mean_dsr_aug']:>8.2f} {r['detection_rate']:>7.2f}")
    print(f"    NEVER detected across skill 0.52->1.0 (realized SR ~1.5): {skill_sweep['never_detected']}")
    print(f"AUDIT markers (real substrate, not reproduced): weekly oracle SR {AUDIT_WEEKLY_SR} (< wall) "
          f"rejected @ t {AUDIT_WEEKLY_MARGINAL_T}<3 | daily SR {AUDIT_DAILY_SR} (> wall) passed")
    print(f"report -> {out_path}")
    print("========================================================================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
