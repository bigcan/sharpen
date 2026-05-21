"""AlphaSeek v3 — Theoretical-PnL upper-bound probe (S544-cont-4 / M5.5).

Kill-or-cure probe before committing M6 HPO GPU spend. Replays the Stage 0.5
shadow corpus (1.35M Binance perp BTCUSDT post-only orders, 23 wallclock days)
through the M2 fill model with the fitted α/β/γ, plus a β-shrinkage sweep
(1.0×, 0.7×, 0.5×) since the head-of-queue assumption over-predicts fills.

For each candidate quote in the corpus, a clairvoyant policy picks only the
quotes whose per-fill PnL would have been positive after fees + per-trade
penalty. The PF achieved on that subset is the upper bound any trained agent
operating under the M2 contract could approach.

Decision rule (per .agent/artifacts/alphaseek_v3_m2_fit/finding.md §4):
    PF ≥ 1.30          → GO       (build LOB manifest, launch M6, β shrunk 0.7×)
    PF ∈ [1.05, 1.30)  → SCOPED   (10-trial / 80K-step mini-HPO probe)
    PF <  1.05         → KILL     (redirect GPU to SG-1-EURUSD / GMGP1-MGC)

PnL accounting — two modes, switchable via --pnl-mode:

    spread_capture_free_exit  (default; OPTIMISTIC CEILING)
        per_fill_pnl_bps = side * (mid_at_submit - limit_px) / mid_at_submit * 1e4
                         - maker_fee_bps - per_trade_penalty_bps
        Assumes the strategy can instantly mark out at the same mid after fill
        with zero exit cost. Best-case framing — a clairvoyant agent achieves
        AT MOST this PF. If this returns KILL, the M2 contract is structurally
        infeasible regardless of agent capability.

    spread_capture_taker_exit (CONSERVATIVE; realistic round-trip)
        Subtracts a taker exit leg: -taker_fee_bps - half_spread_exit_bps
        Half-spread estimated from corpus median spread_bps (≈0.013 → 0.0065 bps).
        Mirrors the actual round-trip cost a paper-deployed v3 would face.

Data limitation acknowledged: the shadow corpus does not log mid_at_outcome
(only ts_outcome), and the LOB parquet (data/lob_parquet/btcusdt_lob_1s.parquet)
covers 2025-08-22 → ~2025-09-26, eight months before the corpus window
(2026-04-28 → 2026-05-18) — no synchronized forward mid lookup available.
A directional-drift-aware probe would require relaunching the shadow logger v2
with mid_at_outcome capture, which is post-verdict scope.

Usage:
    # Default run (spread_capture_free_exit, full β sweep, all shards):
    python scripts/alphaseek_v3_theoretical_pnl_probe.py

    # Conservative round-trip, more bootstrap iters, custom corpus glob:
    python scripts/alphaseek_v3_theoretical_pnl_probe.py \
        --pnl-mode spread_capture_taker_exit \
        --bootstrap-iters 10000 \
        --corpus-glob 'data/alphaseek_shadow_corpus/*.parquet'

    # Smoke test on smallest shard:
    python scripts/alphaseek_v3_theoretical_pnl_probe.py \
        --corpus-glob 'data/alphaseek_shadow_corpus/btcusdt_passive_20260518T160833Z.parquet' \
        --bootstrap-iters 500

Outputs:
    .agent/artifacts/alphaseek_v3_m2_fit/probe_results.json   (machine-readable)
    Stdout: verdict + recommendation block
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CORPUS_GLOB = str(PROJECT_ROOT / "data" / "alphaseek_shadow_corpus" / "*.parquet")
DEFAULT_FIT_REPORT = PROJECT_ROOT / ".agent" / "artifacts" / "alphaseek_v3_m2_fit" / "fit_report.json"
DEFAULT_HPO_CONFIG = PROJECT_ROOT / "configs" / "alphaseek_v3_hpo.yaml"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / ".agent" / "artifacts" / "alphaseek_v3_m2_fit" / "probe_results.json"

SPREAD_CLAMP_BPS = 0.5           # matches MakerBiasExecutor.spread_clamp_bps
DEFAULT_MAX_AGE = 60             # matches limit_max_age in alphaseek_v3_*.yaml

VERDICT_GO_PF = 1.30
VERDICT_SCOPED_PF = 1.05

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("alphaseek_probe")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FillModel:
    alpha_ofi: float
    beta_spread: float
    gamma_age: float
    base_rate: float          # informational only
    n_obs_fit: int
    n_pos_fit: int

    @classmethod
    def from_fit_report(cls, path: Path) -> "FillModel":
        d = json.loads(path.read_text())
        return cls(
            alpha_ofi=d["alpha_ofi"],
            beta_spread=d["beta_spread"],
            gamma_age=d["gamma_age"],
            base_rate=d["base_rate"],
            n_obs_fit=d["n_obs"],
            n_pos_fit=d["n_pos"],
        )


@dataclass
class FeeContract:
    maker_fee_bps: float           # paid on every maker fill
    taker_fee_bps: float           # paid on taker exit leg (mode=spread_capture_taker_exit)
    per_trade_penalty_bps: float   # subtractive per-fill drag
    half_spread_exit_bps: float    # paid on taker exit leg
    spread_clamp_bps: float
    limit_max_age: int

    @classmethod
    def from_hpo_config(cls, path: Path, half_spread_exit_bps: float | None = None) -> "FeeContract":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        env = cfg["env"]
        return cls(
            maker_fee_bps=float(env["maker_fee"]) * 1e4,
            taker_fee_bps=float(env["taker_fee"]) * 1e4,
            per_trade_penalty_bps=float(env["per_trade_penalty_bps"]),
            half_spread_exit_bps=half_spread_exit_bps if half_spread_exit_bps is not None else 0.5 * 0.013,
            spread_clamp_bps=SPREAD_CLAMP_BPS,
            limit_max_age=int(env["limit_max_age"]),
        )


# ---------------------------------------------------------------------------
# Corpus loading + PnL accounting
# ---------------------------------------------------------------------------

def load_corpus(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no shards matched: {pattern}")
    logger.info("loading %d shard(s):", len(files))
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        logger.info("  %s  rows=%-9d  outcomes=%s", Path(f).name, len(df), df["outcome"].value_counts().to_dict())
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    # Treat cancelled_on_shutdown == cancelled.
    out["outcome"] = out["outcome"].replace({"cancelled_on_shutdown": "cancelled"})
    out = out[out["outcome"].isin(["filled", "cancelled"])].reset_index(drop=True)
    logger.info("corpus TOTAL: rows=%d  filled=%d  cancelled=%d",
                len(out), int((out.outcome == "filled").sum()), int((out.outcome == "cancelled").sum()))
    return out


def compute_per_fill_pnl_bps(
    df: pd.DataFrame,
    fees: FeeContract,
    pnl_mode: str,
) -> np.ndarray:
    """PnL per fill in bps of notional. Conditional on fill happening."""
    spread_capture_bps = (
        df["side"].to_numpy() * (df["mid_at_submit"].to_numpy() - df["limit_px"].to_numpy())
        / df["mid_at_submit"].to_numpy() * 1e4
    )
    if pnl_mode == "spread_capture_free_exit":
        return spread_capture_bps - fees.maker_fee_bps - fees.per_trade_penalty_bps
    if pnl_mode == "spread_capture_taker_exit":
        return (
            spread_capture_bps
            - fees.maker_fee_bps
            - fees.taker_fee_bps
            - fees.half_spread_exit_bps
            - 2.0 * fees.per_trade_penalty_bps
        )
    raise ValueError(f"unknown pnl_mode: {pnl_mode!r}")


# ---------------------------------------------------------------------------
# Fill simulator (fitted α/β/γ with β-shrinkage)
# ---------------------------------------------------------------------------

def simulate_fills(
    df: pd.DataFrame,
    model: FillModel,
    beta_shrinkage: float,
    max_age: int,
    seed: int,
) -> np.ndarray:
    """Per-bar Bernoulli sampling using sigmoid(α·OFI + β'·sp_inv + γ·age).

    Returns int array length n_orders: bars-to-fill (1..max_age) or 0 for unfilled.
    Uses static per-order features (ofi_signed, spread_bps from submit snapshot)
    — same approximation as fit_alpha_beta_gamma.py.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    side = df["side"].to_numpy()
    ofi_signed = df["ofi_signed"].to_numpy()
    spread_bps = np.clip(df["spread_bps"].to_numpy(), SPREAD_CLAMP_BPS, None)

    ofi_dir = ofi_signed * side
    spread_term = 1.0 / spread_bps
    beta_eff = beta_shrinkage * model.beta_spread

    base_logit = model.alpha_ofi * ofi_dir + beta_eff * spread_term  # static per order
    fill_age = np.zeros(n, dtype=np.int32)  # 0 == not filled
    alive = np.ones(n, dtype=bool)

    for age in range(1, max_age + 1):
        if not alive.any():
            break
        logit = base_logit[alive] + model.gamma_age * age
        p_fill = 1.0 / (1.0 + np.exp(-logit))
        draws = rng.random(alive.sum())
        filled_now = draws < p_fill
        alive_idx = np.flatnonzero(alive)
        newly_filled = alive_idx[filled_now]
        fill_age[newly_filled] = age
        alive[newly_filled] = False
    return fill_age


# ---------------------------------------------------------------------------
# Clairvoyant policy + metrics
# ---------------------------------------------------------------------------

def clairvoyant_filter(
    pnl_bps: np.ndarray,
    filled_mask: np.ndarray,
) -> np.ndarray:
    """Boolean mask: orders the clairvoyant agent would have submitted.

    Picks only orders that (a) actually filled (per `filled_mask`) AND
    (b) have positive per-fill PnL after fees + penalty. Skips the rest.
    This is the upper bound any policy under the M2 contract could approach.
    """
    return filled_mask & (pnl_bps > 0.0)


def profit_factor(pnl: np.ndarray) -> float:
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def daily_sharpe(pnl_bps: np.ndarray, ts_ns: np.ndarray) -> float:
    """Sharpe from per-day PnL aggregates. Annualization factor sqrt(365)."""
    if len(pnl_bps) == 0:
        return 0.0
    days = (ts_ns // (86_400 * 10**9)).astype(np.int64)
    daily = pd.Series(pnl_bps).groupby(days).sum().to_numpy()
    if daily.std() <= 0:
        return 0.0
    return float(daily.mean() / daily.std() * np.sqrt(365.0))


def bootstrap_pf_ci(
    pnl: np.ndarray,
    n_iters: int,
    seed: int,
    ci: float = 0.95,
) -> dict:
    """Bootstrap CI on profit factor."""
    if len(pnl) == 0:
        return {"pf_median": 0.0, "pf_low": 0.0, "pf_high": 0.0, "n_iters": 0}
    rng = np.random.default_rng(seed)
    pfs = np.empty(n_iters, dtype=np.float64)
    n = len(pnl)
    for i in range(n_iters):
        idx = rng.integers(0, n, size=n)
        pfs[i] = profit_factor(pnl[idx])
    alpha = (1.0 - ci) / 2.0
    return {
        "pf_median": float(np.median(pfs)),
        "pf_low": float(np.quantile(pfs, alpha)),
        "pf_high": float(np.quantile(pfs, 1.0 - alpha)),
        "n_iters": n_iters,
    }


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def run_sweep(
    df: pd.DataFrame,
    model: FillModel,
    fees: FeeContract,
    pnl_mode: str,
    beta_shrinkages: list[float],
    bootstrap_iters: int,
    seed: int,
    max_age: int,
) -> list[dict]:
    pnl_per_fill_bps = compute_per_fill_pnl_bps(df, fees, pnl_mode)
    empirical_filled = (df["outcome"].to_numpy() == "filled")
    ts_ns = df["ts_submit"].to_numpy()

    n_total = len(df)
    n_filled_emp = int(empirical_filled.sum())
    logger.info("per-fill PnL bps  mean=%+.4f  median=%+.4f  p5=%+.4f  p95=%+.4f",
                pnl_per_fill_bps.mean(),
                np.median(pnl_per_fill_bps),
                np.quantile(pnl_per_fill_bps, 0.05),
                np.quantile(pnl_per_fill_bps, 0.95))

    # Clairvoyant on empirical fills (β-shrinkage independent).
    clair_mask = clairvoyant_filter(pnl_per_fill_bps, empirical_filled)
    clair_pnl = pnl_per_fill_bps[clair_mask]
    clair_pf = profit_factor(clair_pnl)
    clair_boot = bootstrap_pf_ci(clair_pnl, bootstrap_iters, seed=seed)
    clair_sharpe = daily_sharpe(clair_pnl, ts_ns[clair_mask])
    logger.info("clairvoyant on empirical fills:  kept=%d/%d (%.2f%%)  PF=%.3f  Sharpe_d=%.2f  CI95=[%.3f, %.3f]",
                int(clair_mask.sum()), n_filled_emp, 100.0 * clair_mask.sum() / max(n_filled_emp, 1),
                clair_pf, clair_sharpe, clair_boot["pf_low"], clair_boot["pf_high"])

    results = [{
        "scope": "clairvoyant_empirical_fills",
        "beta_shrinkage": None,
        "n_total_orders": n_total,
        "n_filled": n_filled_emp,
        "n_clairvoyant_kept": int(clair_mask.sum()),
        "keep_rate_of_filled": float(clair_mask.sum() / max(n_filled_emp, 1)),
        "pf_point": clair_pf,
        "pf_bootstrap": clair_boot,
        "sharpe_daily": clair_sharpe,
        "avg_pnl_bps_per_fill": float(clair_pnl.mean()) if len(clair_pnl) else 0.0,
    }]

    # β-shrinkage sweep — simulator + clairvoyant.
    for shrink in beta_shrinkages:
        fill_age = simulate_fills(df, model, beta_shrinkage=shrink, max_age=max_age, seed=seed)
        sim_filled = fill_age > 0
        n_filled_sim = int(sim_filled.sum())
        clair_sim_mask = clairvoyant_filter(pnl_per_fill_bps, sim_filled)
        clair_sim_pnl = pnl_per_fill_bps[clair_sim_mask]
        pf_point = profit_factor(clair_sim_pnl)
        boot = bootstrap_pf_ci(clair_sim_pnl, bootstrap_iters, seed=seed + int(shrink * 1000))
        sharpe = daily_sharpe(clair_sim_pnl, ts_ns[clair_sim_mask])
        avg_btf = float(fill_age[sim_filled].mean()) if n_filled_sim else 0.0
        logger.info(
            "beta*%.2f  fill_rate_sim=%.2f%%  avg_btf=%.2f  clair_kept=%-7d  PF=%.3f  Sharpe_d=%.2f  CI95=[%.3f, %.3f]",
            shrink, 100.0 * n_filled_sim / n_total, avg_btf,
            int(clair_sim_mask.sum()), pf_point, sharpe, boot["pf_low"], boot["pf_high"],
        )
        results.append({
            "scope": "fitted_fill_sim",
            "beta_shrinkage": shrink,
            "beta_effective": shrink * model.beta_spread,
            "n_total_orders": n_total,
            "n_filled_sim": n_filled_sim,
            "fill_rate_sim": float(n_filled_sim / n_total),
            "avg_bars_to_fill_sim": avg_btf,
            "n_clairvoyant_kept": int(clair_sim_mask.sum()),
            "pf_point": pf_point,
            "pf_bootstrap": boot,
            "sharpe_daily": sharpe,
            "avg_pnl_bps_per_fill": float(clair_sim_pnl.mean()) if len(clair_sim_pnl) else 0.0,
        })
    return results


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def decide_verdict(results: list[dict]) -> dict:
    """GO/SCOPED/KILL based on best PF across the β-shrinkage sweep.

    Best PF = max of the bootstrap median PFs across all sim runs. Use the
    median (not point) for robustness against single-bootstrap noise.
    """
    sim_results = [r for r in results if r["scope"] == "fitted_fill_sim"]
    if not sim_results:
        return {"verdict": "ERROR", "reason": "no sim runs in results"}
    best = max(sim_results, key=lambda r: r["pf_bootstrap"]["pf_median"])
    best_pf = best["pf_bootstrap"]["pf_median"]
    if best_pf >= VERDICT_GO_PF:
        verdict = "GO"
        action = "Build LOB manifest (scripts/build_data_manifest.py) and launch M6 HPO with β shrunk 0.7×."
    elif best_pf >= VERDICT_SCOPED_PF:
        verdict = "SCOPED"
        action = "Run 10-trial / 80K-step mini-HPO probe before committing to full M6 chain."
    else:
        verdict = "KILL"
        action = ("Redirect GPU/attention to queued workstreams (SG-1-EURUSD deferred S520 or "
                  "GMGP1-MGC retrain queued S477). Document v3 falsification in randd_log.md.")
    return {
        "verdict": verdict,
        "best_pf_bootstrap_median": best_pf,
        "best_beta_shrinkage": best["beta_shrinkage"],
        "best_pf_ci95": [best["pf_bootstrap"]["pf_low"], best["pf_bootstrap"]["pf_high"]],
        "thresholds": {"GO": VERDICT_GO_PF, "SCOPED": VERDICT_SCOPED_PF},
        "action": action,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-glob", default=DEFAULT_CORPUS_GLOB,
                    help=f"shadow corpus parquet glob (default: {DEFAULT_CORPUS_GLOB})")
    ap.add_argument("--fit-report", default=str(DEFAULT_FIT_REPORT),
                    help=f"fitted α/β/γ JSON (default: {DEFAULT_FIT_REPORT.name})")
    ap.add_argument("--hpo-config", default=str(DEFAULT_HPO_CONFIG),
                    help="HPO YAML for fee fields (default: configs/alphaseek_v3_hpo.yaml)")
    ap.add_argument("--pnl-mode", default="spread_capture_free_exit",
                    choices=["spread_capture_free_exit", "spread_capture_taker_exit"],
                    help="PnL accounting mode (default: spread_capture_free_exit = optimistic ceiling)")
    ap.add_argument("--beta-shrinkages", default="1.0,0.7,0.5",
                    help="CSV of β shrinkage factors to sweep (default: 1.0,0.7,0.5)")
    ap.add_argument("--bootstrap-iters", type=int, default=5000,
                    help="bootstrap resamples for PF CI (default: 5000)")
    ap.add_argument("--max-age", type=int, default=DEFAULT_MAX_AGE,
                    help=f"limit_max_age in bars (default: {DEFAULT_MAX_AGE})")
    ap.add_argument("--seed", type=int, default=20260521,
                    help="RNG seed (default: 20260521, the S544-cont-4 date)")
    ap.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON),
                    help=f"output JSON path (default: {DEFAULT_OUTPUT_JSON.name})")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()

    model = FillModel.from_fit_report(Path(args.fit_report))
    logger.info("fit model  alpha=%+.4f  beta=%+.4f  gamma=%+.4f  base_rate=%.4f%%  (n_fit=%d)",
                model.alpha_ofi, model.beta_spread, model.gamma_age,
                model.base_rate * 100, model.n_obs_fit)

    fees = FeeContract.from_hpo_config(Path(args.hpo_config))
    logger.info("fee contract  maker=%.2f bps  taker=%.2f bps  penalty=%.2f bps  half_spread_exit=%.4f bps  max_age=%d",
                fees.maker_fee_bps, fees.taker_fee_bps, fees.per_trade_penalty_bps,
                fees.half_spread_exit_bps, fees.limit_max_age)
    logger.info("pnl_mode=%s   beta shrinkages=%s   bootstrap_iters=%d",
                args.pnl_mode, args.beta_shrinkages, args.bootstrap_iters)

    df = load_corpus(args.corpus_glob)
    beta_shrinkages = [float(x) for x in args.beta_shrinkages.split(",")]
    results = run_sweep(
        df, model, fees,
        pnl_mode=args.pnl_mode,
        beta_shrinkages=beta_shrinkages,
        bootstrap_iters=args.bootstrap_iters,
        seed=args.seed,
        max_age=args.max_age,
    )
    verdict = decide_verdict(results)

    output = {
        "schema_version": 1,
        "run_ts": pd.Timestamp.utcnow().isoformat(),
        "corpus_glob": args.corpus_glob,
        "n_orders": int(len(df)),
        "pnl_mode": args.pnl_mode,
        "fit_model": asdict(model),
        "fees": asdict(fees),
        "beta_shrinkages": beta_shrinkages,
        "bootstrap_iters": args.bootstrap_iters,
        "seed": args.seed,
        "results": results,
        "verdict": verdict,
        "limitations": (
            "Corpus does not log mid_at_outcome; LOB parquet covers 2025-08 (no overlap with 2026-05 corpus). "
            "Spread-capture-only PnL is an OPTIMISTIC CEILING under the free-exit assumption -- "
            "a true upper bound that also incorporates post-fill directional drift would require "
            "synchronized LOB tick data not available in this corpus. KILL under this conservative "
            "framing is dispositive (no realistic drift addition rescues a PF<<1 ceiling); "
            "GO would warrant a follow-up probe with shadow logger v2 (mid_at_outcome captured)."
        ),
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("wrote results -> %s", out_path)

    print()
    print("=" * 70)
    print(f"VERDICT: {verdict['verdict']}")
    print("=" * 70)
    print(f"  best PF (bootstrap median) = {verdict['best_pf_bootstrap_median']:.4f}")
    print(f"  best beta shrinkage        = {verdict['best_beta_shrinkage']}")
    print(f"  PF CI95                    = [{verdict['best_pf_ci95'][0]:.4f}, {verdict['best_pf_ci95'][1]:.4f}]")
    print(f"  thresholds                 = GO >= {VERDICT_GO_PF}  SCOPED >= {VERDICT_SCOPED_PF}  else KILL")
    print(f"  action: {verdict['action']}")
    print(f"  pnl_mode: {args.pnl_mode}")
    print(f"  elapsed: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
