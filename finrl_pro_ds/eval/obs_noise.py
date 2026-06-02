"""Stage 3.5 Observation-Noise Robustness — Protocol v2.7-B (S553-cont-25).

Price-path randomization for a frozen PROMOTE policy: re-roll the ensemble
through *price-path-randomized* OHLC (multiplicative log-noise on the raw 1-min
bars -> real feature recompute via the unmodified handler -> ensemble rollout),
aggregate PF/MDD across ``noise_seeds x WF folds x sigma-levels``, and gate on
edge survival under perturbation.

Architecture: ``.agent/artifacts/protocol_v27_b_obs_noise_stage_3_5_architecture.md``
Math gate:    same doc, "§ Math Audit Verdict" (S553-cont-25, PASS WITH NOTES)
Researcher:   ``.agent/artifacts/mc_robustness_methods_research.md`` (Method #4 GO)

This module is the **B1 pure core**: the noise function + parquet writer + the
fold aggregator + the gate resolver + dataclasses. It is torch-free and
env-free — every function here is unit-tested directly. The impure orchestrator
``run_obs_noise_stage`` (B2) reuses ``scripts.sg1_xauusd_ensemble_eval.run_rule``
and lands separately so this core stays dormant (no caller) and independently
revertible.

Promoted Math invariants baked in here (see the Math Audit Verdict):
  - MATH-OBS-1 — multiplicative-lognormal noise; positivity is structural
    (``exp(eps) > 0``); ``apply_ohlc_noise`` still ASSERTS OHLC>0 & no-NaN on
    output (DATA-CLEAN).
  - MATH-OBS-2 — sigma is a *log-space* std; gate-key bps labels are
    ``bps = sigma * 1e4`` (10bps<->0.001, 50bps<->0.005). Kept explicit via
    ``sigma_to_bps`` / ``bps_to_sigma`` so the unit is never re-derived in code.
  - MATH-OBS-3 — the min/max repair injects a conservative upward range bias on
    low-range bars; the gate is therefore a *lower bound* on path-robustness.
    Documented; not a correctness defect.
  - MATH-OBS-4 (load-bearing) — MDD-degradation = ``q95(|mdd_seeds|) -
    |mdd_nominal|``, absolute value applied **before** the quantile. Computing
    ``|quantile(signed_mdd, 0.95)|`` would report the *least*-negative (rosy)
    tail and silently defeat the risk gate — a CRITICAL gate-defeat.
  - MATH-OBS-5 — per-``(fold, sigma, k)`` seed derivation MUST be a stable
    integer via ``np.random.SeedSequence([base, fold, level_idx, k])`` (or
    explicit arithmetic), never Python ``hash(tuple)`` (PYTHONHASHSEED-salted
    => non-reproducible, breaks OBSNOISE-4 determinism).

Structural invariants (v2.7-B):
  - OBSNOISE-1 — post-noise OHLC ordering holds every row: ``H>=max(O,L,C)``,
    ``L<=min(O,H,C)`` (min/max repair).
  - OBSNOISE-2 — bars with ``timestamp < noise_start`` are byte-identical to the
    source (train+val EMA-Z warmup feeds on real data; LEAK-1).
  - OBSNOISE-3 — ``sigma == 0`` => noised frame is byte-identical => the nominal
    rollout reproduces Stage 3.
  - OBSNOISE-4 — noise is deterministic & causal: same ``(seed)`` -> same array;
    bar ``t``'s noise depends only on ``(seed, row_index)``, never on bars ``>t``.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Single source of the PF clip convention (torch-free import; sensitivity_audit
# lazy-imports run_rule so this does NOT pull torch/env into the pure core).
from finrl_pro_ds.eval.sensitivity_audit import PF_CAP

log = logging.getLogger("obs-noise")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OHLC_COLS: Tuple[str, ...] = ("open", "high", "low", "close")
"""Channels that receive multiplicative log-noise. Volume is passthrough (ADR-8)."""

PARQUET_COLS: Tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")
"""Columns the real handler requires (multiscale_handler.py:277). ``write_noised_parquet``
keeps only these (intersection with the source) so the noised artifact is minimal."""

DEFAULT_NOISE_BASE_SEED = 20260602
"""Base entropy for the (fold, sigma, k) seed derivation (MATH-OBS-5). The
orchestrator may override via ``gates.obs_noise_base_seed`` for a fresh re-roll."""

SCHEMA_VERSION = "1.0"
"""obs_noise_report.json / verdict schema version (IC-2)."""


class InvariantViolation(RuntimeError):
    """Raised when a CLAUDE.md / OBSNOISE invariant would be violated."""


# ---------------------------------------------------------------------------
# Spec / result dataclasses (IC-1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoiseSpec:
    """One sigma-level of the obs-noise sweep."""

    sigma: float          # log-noise std (0.001 ~= 10bps, 0.005 ~= 50bps)
    label: str            # "10bps" | "50bps" — gate-key suffix; explicit, no magic
    n_seeds: int          # noise seeds per (fold, sigma)

    @property
    def bps(self) -> float:
        """Per-bar bps scale (MATH-OBS-2: bps = sigma * 1e4)."""
        return sigma_to_bps(self.sigma)


@dataclass(frozen=True)
class FoldNoiseResult:
    """Aggregated obs-noise metrics for one (fold, sigma) cell."""

    fold_idx: int
    sigma_label: str
    pf_nominal: float            # sigma=0 in-band reference (ADR-3)
    mdd_nominal: float           # signed, in [-1, 0]
    pf_seeds: Tuple[float, ...]  # len == n_seeds
    mdd_seeds: Tuple[float, ...]
    pf_q05: float
    pf_q50: float
    pf_q95: float
    mdd_q05: float               # signed MDD quantiles (in [-1,0]), REPORTING only
    mdd_q50: float
    mdd_q95: float
    absmdd_q95: float            # q95 of |mdd_seeds| — worst-case DD magnitude (gate input; MATH-OBS-4)
    pf_ratio: float              # pf_q50 / pf_nominal (ADR-6)
    mdd_degradation_pp: float    # (q95(|mdd_seeds|) - |mdd_nominal|) * 100 — abs FIRST, then q95 (MATH-OBS-4)
    anomaly: Optional[str]       # "pf_nominal<1.0" | "pf_nominal==CAP" -> excluded from gate (ADR-6)


@dataclass(frozen=True)
class ObsNoiseVerdict:
    """Outcome of the Stage 3.5 edge-survival gate across folds x sigma-levels."""

    decision: str                # "PASS" | "FAIL" | "UNKNOWN_INSUFFICIENT_FOLDS"
    per_sigma: Dict[str, dict]   # label -> {pf_floor, worst_pf_ratio, worst_fold,
                                 #          mdd_buffer_pp, worst_mdd_degr_pp,
                                 #          worst_mdd_fold, pf_pass, mdd_pass}
    n_folds: int
    n_folds_graded: int


# ---------------------------------------------------------------------------
# sigma <-> bps (MATH-OBS-2)
# ---------------------------------------------------------------------------


def sigma_to_bps(sigma: float) -> float:
    """Log-space std -> per-bar bps (MATH-OBS-2: ``bps = sigma * 1e4``).

    Exact mapping ``std(x'/x - 1) = sqrt(exp(sigma^2)(exp(sigma^2)-1))`` equals
    ``sigma * 1e4`` bps to <0.01% for sigma <= 50bps (Math verdict §1).
    """
    return float(sigma) * 1e4


def bps_to_sigma(bps: float) -> float:
    """Per-bar bps -> log-space std (inverse of :func:`sigma_to_bps`)."""
    return float(bps) * 1e-4


# ---------------------------------------------------------------------------
# Seed derivation (MATH-OBS-5) — stable int, NEVER hash(tuple)
# ---------------------------------------------------------------------------


def derive_noise_seed(
    fold_idx: int,
    level_idx: int,
    k: int,
    base_seed: int = DEFAULT_NOISE_BASE_SEED,
) -> int:
    """Stable integer seed for the ``(fold, sigma-level, noise-seed)`` cell.

    MATH-OBS-5 (load-bearing): derive via ``np.random.SeedSequence`` so the seed
    is reproducible across processes/sessions. Python's ``hash(tuple)`` is
    PYTHONHASHSEED-salted per process and would silently break OBSNOISE-4
    determinism — it must NEVER be used here.
    """
    ss = np.random.SeedSequence([int(base_seed), int(fold_idx), int(level_idx), int(k)])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


# ---------------------------------------------------------------------------
# Core noise function (OBSNOISE-1..4, MATH-OBS-1)
# ---------------------------------------------------------------------------


def apply_ohlc_noise(
    df: pd.DataFrame,
    sigma: float,
    seed: int,
    noise_start: Any,
) -> pd.DataFrame:
    """Return a COPY of ``df`` with multiplicative log-noise on O/H/L/C for rows
    with ``timestamp >= noise_start``; ordering-repaired; volume untouched (ADR-8).

    Model (Researcher spec, Math-blessed): for each masked bar ``t`` and channel
    ``c in {O,H,L,C}``: ``eps_{t,c} ~ N(0, sigma^2)`` i.i.d.;
    ``x'_{t,c} = x_{t,c} * exp(eps_{t,c})``; then repair
    ``H'_t = max(O',H',L',C')``, ``L'_t = min(O',H',L',C')`` (OBSNOISE-1).

    Determinism & causality (OBSNOISE-4 / MATH-OBS-5): the noise block is drawn
    positionally — ``rng.normal(size=(n_masked, 4))`` over the timestamp-sorted
    masked rows — so bar ``t``'s noise depends only on ``(seed, row_index)``,
    never on bars ``> t``.

    ``sigma == 0.0`` -> byte-identical copy (OBSNOISE-3).

    Preconditions: ``df`` sorted by timestamp; has ``timestamp`` + OHLC columns;
    masked OHLC strictly positive, no NaN (DATA-CLEAN).
    Postconditions: ``H>=max(O,L,C)``, ``L<=min(O,H,C)`` per masked row
    (OBSNOISE-1); OHLC>0, no NaN; rows with ``timestamp < noise_start``
    byte-identical to input (OBSNOISE-2); volume column unchanged.
    """
    if float(sigma) < 0.0:
        raise ValueError(f"sigma must be >= 0; got {sigma}")
    missing = [c for c in ("timestamp", *OHLC_COLS) if c not in df.columns]
    if missing:
        raise ValueError(f"apply_ohlc_noise: missing required columns {missing}")

    out = df.copy(deep=True)

    # sigma == 0 -> exact identity (OBSNOISE-3): no RNG draw, no float churn.
    if float(sigma) == 0.0:
        return out

    ts = pd.to_datetime(out["timestamp"], utc=True)
    if not ts.is_monotonic_increasing:
        raise InvariantViolation(
            "apply_ohlc_noise: df must be sorted by timestamp (OBSNOISE-4 "
            "positional draw assumes time order)."
        )
    ns = pd.Timestamp(noise_start)
    ns = ns.tz_localize("UTC") if ns.tzinfo is None else ns.tz_convert("UTC")

    m = np.flatnonzero((ts >= ns).to_numpy())
    if m.size == 0:
        log.warning(
            "apply_ohlc_noise: noise_start %s is past every bar; returning "
            "un-noised copy.", ns,
        )
        return out

    # Gather original masked OHLC as float64 (math in float64; written back at
    # the column's native dtype so unmasked rows stay byte-identical).
    o = out["open"].to_numpy(dtype=np.float64)[m]
    h = out["high"].to_numpy(dtype=np.float64)[m]
    lo = out["low"].to_numpy(dtype=np.float64)[m]
    c = out["close"].to_numpy(dtype=np.float64)[m]
    if not np.all(np.isfinite([o, h, lo, c])) or np.any(np.array([o, h, lo, c]) <= 0.0):
        raise InvariantViolation(
            "apply_ohlc_noise: masked source OHLC must be finite and strictly "
            "positive (DATA-CLEAN precondition violated)."
        )

    rng = np.random.default_rng(int(seed))
    factor = np.exp(rng.normal(loc=0.0, scale=float(sigma), size=(m.size, 4)))
    o_n = o * factor[:, 0]
    h_n = h * factor[:, 1]
    l_n = lo * factor[:, 2]
    c_n = c * factor[:, 3]
    # OBSNOISE-1 min/max repair over the noised channels.
    hh = np.maximum.reduce([o_n, h_n, l_n, c_n])
    ll = np.minimum.reduce([o_n, h_n, l_n, c_n])

    # Postconditions checked on the float64 noised arrays (MATH-OBS-1 /
    # DATA-CLEAN / OBSNOISE-1) — before the native-dtype cast, so the check is
    # exact and immune to float32 round-trip rounding.
    final = np.array([o_n, hh, ll, c_n])
    if not np.all(np.isfinite(final)) or np.any(final <= 0.0):
        raise InvariantViolation(
            "apply_ohlc_noise: post-noise OHLC must be finite and positive "
            "(MATH-OBS-1)."
        )
    if np.any(hh < ll):  # repaired high below repaired low — impossible unless a coding bug
        raise InvariantViolation(
            "apply_ohlc_noise: OHLC ordering repair failed (OBSNOISE-1)."
        )

    for col, arr in (("open", o_n), ("high", hh), ("low", ll), ("close", c_n)):
        new_col = out[col].to_numpy().copy()  # native dtype, full length
        new_col[m] = arr.astype(new_col.dtype)
        out[col] = new_col
    return out


def _file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def write_noised_parquet(
    source_path: Any,
    sigma: float,
    seed: int,
    noise_start: Any,
    out_path: Any,
) -> Tuple[Path, str]:
    """Load the source 1-min parquet (handler columns only), apply_ohlc_noise,
    write ``out_path``; return ``(out_path, sha256)``.

    ``sigma == 0`` still writes (so the nominal rollout walks the SAME code path;
    ADR-3). The unmodified handler then loads this as ordinary data and
    recomputes every feature / the X2 causal map / EMA-Z / ATR from the noised
    bars (single source of truth). Caller deletes unless ``--keep-noised``.
    """
    source_path = Path(source_path)
    out_path = Path(out_path)
    df = pd.read_parquet(source_path)
    cols = [c for c in PARQUET_COLS if c in df.columns]
    df = df[cols]
    noised = apply_ohlc_noise(df, sigma, seed, noise_start)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    noised.to_parquet(out_path)
    return out_path, _file_sha256(out_path)


# ---------------------------------------------------------------------------
# Fold aggregator (ADR-6 / MATH-OBS-4)
# ---------------------------------------------------------------------------


def aggregate_fold(
    fold_idx: int,
    spec: NoiseSpec,
    pf_nominal: float,
    mdd_nominal: float,
    pf_seeds: Sequence[float],
    mdd_seeds: Sequence[float],
) -> FoldNoiseResult:
    """Quantiles + pf_ratio + mdd_degradation_pp + anomaly flag for one cell.

    ADR-6 / MATH-OBS-4: ``pf_ratio = median_k(pf) / pf_nominal``;
    ``mdd_degradation_pp = (q95(|mdd_seeds|) - |mdd_nominal|) * 100`` — the
    absolute value is taken **before** the 95th percentile so the gate sees the
    worst-case drawdown magnitude, not the rosy least-negative tail.

    Anomaly (ADR-6): ``pf_nominal < 1.0`` (un-profitable nominal — Stage 3 should
    preclude) excludes the fold from grading; ``pf_nominal == PF_CAP`` is flagged
    (clipped denominator -> ratio understates robustness) but reported, not failed.
    """
    pf = np.asarray(pf_seeds, dtype=np.float64)
    mdd = np.asarray(mdd_seeds, dtype=np.float64)
    if pf.size == 0 or mdd.size == 0:
        raise ValueError(
            f"aggregate_fold(fold={fold_idx}, {spec.label}): empty seed arrays"
        )

    pf_q05, pf_q50, pf_q95 = (float(x) for x in np.quantile(pf, [0.05, 0.50, 0.95]))
    mdd_q05, mdd_q50, mdd_q95 = (
        float(x) for x in np.quantile(mdd, [0.05, 0.50, 0.95])
    )
    # MATH-OBS-4: abs FIRST, then q95. NOT abs(quantile(signed, 0.95)).
    absmdd_q95 = float(np.quantile(np.abs(mdd), 0.95))

    anomaly: Optional[str] = None
    if pf_nominal < 1.0:
        anomaly = "pf_nominal<1.0"
    elif pf_nominal >= PF_CAP - 1e-9:
        anomaly = "pf_nominal==CAP"

    pf_ratio = float(pf_q50 / pf_nominal) if pf_nominal > 0.0 else float("nan")
    mdd_degradation_pp = float((absmdd_q95 - abs(float(mdd_nominal))) * 100.0)

    return FoldNoiseResult(
        fold_idx=int(fold_idx),
        sigma_label=spec.label,
        pf_nominal=float(pf_nominal),
        mdd_nominal=float(mdd_nominal),
        pf_seeds=tuple(float(x) for x in pf),
        mdd_seeds=tuple(float(x) for x in mdd),
        pf_q05=pf_q05,
        pf_q50=pf_q50,
        pf_q95=pf_q95,
        mdd_q05=mdd_q05,
        mdd_q50=mdd_q50,
        mdd_q95=mdd_q95,
        absmdd_q95=absmdd_q95,
        pf_ratio=pf_ratio,
        mdd_degradation_pp=mdd_degradation_pp,
        anomaly=anomaly,
    )


# ---------------------------------------------------------------------------
# Gate resolver (ADR-6)
# ---------------------------------------------------------------------------


def resolve_obs_noise_gate(
    results: Sequence[FoldNoiseResult],
    gates: Dict[str, Any],
) -> ObsNoiseVerdict:
    """Per sigma-level: ``worst_pf_ratio = min`` over graded folds, gated vs
    ``gates.obs_noise_pf_floor_<label>``; ``worst_mdd_degr_pp = max`` over graded
    folds, gated vs ``gates.obs_noise_mdd_buffer_pp_<label>``.

    The worst fold governs each axis (Stage-3 "all windows" ethos). Anomaly folds
    (``pf_nominal < 1.0``) are excluded from grading. If the number of graded
    folds is below ``gates.obs_noise_required_min_folds`` (default = number of WF
    windows present in ``results``) the verdict is
    ``UNKNOWN_INSUFFICIENT_FOLDS``. Otherwise ``PASS`` iff every sigma passes BOTH
    its PF floor and its MDD buffer, else ``FAIL``.
    """
    by_label: Dict[str, List[FoldNoiseResult]] = defaultdict(list)
    for r in results:
        by_label[r.sigma_label].append(r)

    all_folds = sorted({r.fold_idx for r in results})
    graded_folds = sorted({r.fold_idx for r in results if r.anomaly is None})
    n_folds = len(all_folds)
    n_folds_graded = len(graded_folds)
    required_min = int(gates.get("obs_noise_required_min_folds", n_folds))

    per_sigma: Dict[str, dict] = {}
    all_pass = True
    for label in sorted(by_label):
        rs = by_label[label]
        graded = [r for r in rs if r.anomaly is None]
        floor_key = f"obs_noise_pf_floor_{label}"
        buffer_key = f"obs_noise_mdd_buffer_pp_{label}"
        if floor_key not in gates or buffer_key not in gates:
            raise KeyError(
                f"resolve_obs_noise_gate: gates missing {floor_key!r} / "
                f"{buffer_key!r} for sigma-level {label!r}"
            )
        pf_floor = float(gates[floor_key])
        mdd_buffer_pp = float(gates[buffer_key])

        if not graded:
            per_sigma[label] = {
                "pf_floor": pf_floor,
                "worst_pf_ratio": float("nan"),
                "worst_fold": None,
                "mdd_buffer_pp": mdd_buffer_pp,
                "worst_mdd_degr_pp": float("nan"),
                "worst_mdd_fold": None,
                "pf_pass": False,
                "mdd_pass": False,
                "n_graded": 0,
            }
            all_pass = False
            continue

        worst_pf = min(graded, key=lambda r: r.pf_ratio)
        worst_mdd = max(graded, key=lambda r: r.mdd_degradation_pp)
        pf_pass = bool(worst_pf.pf_ratio >= pf_floor)
        mdd_pass = bool(worst_mdd.mdd_degradation_pp <= mdd_buffer_pp)
        per_sigma[label] = {
            "pf_floor": pf_floor,
            "worst_pf_ratio": float(worst_pf.pf_ratio),
            "worst_fold": int(worst_pf.fold_idx),
            "mdd_buffer_pp": mdd_buffer_pp,
            "worst_mdd_degr_pp": float(worst_mdd.mdd_degradation_pp),
            "worst_mdd_fold": int(worst_mdd.fold_idx),
            "pf_pass": pf_pass,
            "mdd_pass": mdd_pass,
            "n_graded": len(graded),
        }
        if not (pf_pass and mdd_pass):
            all_pass = False

    if n_folds_graded < required_min:
        decision = "UNKNOWN_INSUFFICIENT_FOLDS"
    elif all_pass:
        decision = "PASS"
    else:
        decision = "FAIL"

    return ObsNoiseVerdict(
        decision=decision,
        per_sigma=per_sigma,
        n_folds=n_folds,
        n_folds_graded=n_folds_graded,
    )
