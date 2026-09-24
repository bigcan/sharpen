"""ATL x Jev screening legs P1–P3 and the K3 decision (Phase 2 step 6; prereg §5, architecture §6, ADR-11).

* **P1:** the five registered signals in one batch through the frozen funnel, deflated at the pre-registered count
  (``phase1.n_hypotheses``).
* **P2, shuffled-firm placebo:** within each calendar week of release rows, filing scores are shuffled across
  filers ``p2_placebo.permutations`` times. Each placebo's IC at the funnel's primary horizon comes from the
  funnel's own ``compute_scores`` and ``cross_sectional_ic``. p = (1 + #{placebo ≥ real}) / (1 + permutations).
* **P3, beats the text baselines:** a daily cross-sectional OLS of forward-return ranks on the ranks of the Jev
  signal and the two baselines (all scored by ``compute_scores``). The Jev coefficient's t uses the Newey–West
  effective count, ``t = mean(β) / sqrt(γ0 / hac_effective_n(β, nw_lags))``.
* **K3 (ADR-11):** a signal passes screening when it is ``PROMISING`` and passes P2 and P3. No passer is NO-GO and
  the clean window stays closed. P4 takes the top-ranked passer and is built only once a passer exists.

Every number is read from ``configs/atl_jev.gates.yaml`` (``phase1``) or the frozen funnel gates; nothing here
is a threshold.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from sharpen.jev.baselines import baseline_signal
from sharpen.jev.release import load_release_rule, release_dates
from sharpen.jev.scoring import to_filing_scores
from sharpen.signals._ic import cross_sectional_ic, hac_effective_n
from sharpen.signals.eval_harness import compute_scores
from sharpen.signals.features import Panel
from sharpen.signals.gates import Gates
from sharpen.signals.library.jev_filings import Construction, JevFilingSignal, build_registered_signals
from sharpen.signals.multiplicity import Multiplicity
from sharpen.signals.scorecard import RankedScorecard, evaluate_batch

FROZEN_FUNNEL_GATES_HASH = "519158fa1450"          # prereg §4.4 and CRU-1: the funnel the verdict is read on
PREREG = "docs/research/atl_jev_prereg.md"
BLOCK_COLUMNS = ("surprise", "tone", "quality")


def registered(phase1: Mapping, scores: pd.DataFrame, calendar: np.ndarray,
               names: Sequence[str] | None = None) -> dict[str, JevFilingSignal]:
    """The registered screening signals (all, or only ``names``), built by the step-2 builder itself."""
    p1 = dict(phase1)
    if names is not None:
        unknown = set(names) - {s["name"] for s in phase1["signals"]}
        if unknown:
            raise ValueError(f"not registered signals: {sorted(unknown)}")
        p1["signals"] = [s for s in phase1["signals"] if s["name"] in set(names)]
    sigs = build_registered_signals(p1, to_filing_scores(scores), calendar, mode="screening")
    return {s.spec.name: s for s in sigs}


# --------------------------------------------------------------------------- P1
def run_p1(signals: Mapping[str, JevFilingSignal], panel: Panel, gates: Gates, phase1: Mapping) -> RankedScorecard:
    mult = Multiplicity.preregistered(int(phase1["n_hypotheses"]), substrate=str(phase1["universe"]),
                                      provenance=f"{PREREG} (questionnaire {phase1['questionnaire']['hash']})")
    return evaluate_batch(dict(signals), panel, gates, "atl-jev-p1", multiplicity=mult)


def promising(rs: RankedScorecard) -> list[str]:
    """PROMISING signal names in the funnel's rank order."""
    return [c.name for c in rs.cards if c.verdict == "PROMISING"]


# --------------------------------------------------------------------------- P2
def funnel_ic(sig: JevFilingSignal, panel: Panel, gates: Gates) -> float:
    """Mean cross-sectional IC at the primary horizon, exactly as the funnel's Tier-1 computes it."""
    h = gates.primary_horizon
    sc = compute_scores(sig, panel, sig.spec.neutralization)
    r = cross_sectional_ic(sc * sig.spec.expected_sign, panel.forward_returns(h), active=panel.active,
                           min_names=gates.min_names_per_day, overlap=h)
    return float(r.ic_mean)


def release_weeks(scores: pd.DataFrame, calendar: np.ndarray, phase1: Mapping) -> np.ndarray:
    """ISO (year, week) of each filing's release row as one int; -1 where the release is outside the calendar."""
    rel = release_dates(pd.to_datetime(scores["accepted_utc"]).to_numpy(dtype="datetime64[ns]"),
                        np.asarray(calendar, dtype="datetime64[D]"), load_release_rule(phase1))
    out = np.full(len(scores), -1, dtype=np.int64)
    ok = ~np.isnat(rel)
    iso = pd.DatetimeIndex(rel[ok]).isocalendar()
    out[ok] = iso["year"].to_numpy(dtype=np.int64) * 100 + iso["week"].to_numpy(dtype=np.int64)
    return out


def permute_within_weeks(scores: pd.DataFrame, weeks: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle each filing's block scores (as one unit) among the filings released in the same week."""
    out = scores.copy()
    vals = scores[list(BLOCK_COLUMNS)].to_numpy(copy=True)
    new = vals.copy()
    for w in np.unique(weeks[weeks >= 0]):
        idx = np.flatnonzero(weeks == w)
        if idx.size > 1:
            new[idx] = vals[rng.permutation(idx)]
    out[list(BLOCK_COLUMNS)] = new
    return out


def _seed(phase1: Mapping, leg: str, name: str) -> int:
    """Fixed by the pre-registered questionnaire hash and the signal name: never chosen after a result."""
    key = f"{phase1['questionnaire']['hash']}:{leg}:{name}".encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def run_p2(name: str, phase1: Mapping, scores: pd.DataFrame, calendar: np.ndarray, panel: Panel,
           gates: Gates) -> dict:
    cfg = phase1["p2_placebo"]
    perms, max_p = int(cfg["permutations"]), float(cfg["max_p"])
    real = funnel_ic(registered(phase1, scores, calendar, [name])[name], panel, gates)
    weeks = release_weeks(scores, calendar, phase1)
    rng = np.random.default_rng(_seed(phase1, "p2", name))
    null = np.array([funnel_ic(registered(phase1, permute_within_weeks(scores, weeks, rng), calendar, [name])[name],
                               panel, gates) for _ in range(perms)])
    # a NaN placebo counts against the signal (fail closed), as does a NaN real IC
    beats = int(np.sum(~(null < real))) if math.isfinite(real) else perms
    p = (1 + beats) / (1 + perms)
    fin = null[np.isfinite(null)]
    return {"signal": name, "real_ic": real, "permutations": perms, "placebo_ge_real": beats,
            "null_mean": float(fin.mean()) if fin.size else math.nan,
            "null_p95": float(np.percentile(fin, 95)) if fin.size else math.nan, "p": p,
            "max_p": max_p, "pass": bool(math.isfinite(real) and p <= max_p)}


# --------------------------------------------------------------------------- P3
def fama_macbeth_betas(y: np.ndarray, xs: Sequence[np.ndarray], active: np.ndarray, min_names: int) -> np.ndarray:
    """Per day: OLS of the ranks of ``y`` on an intercept and the ranks of each ``x`` (all scaled to (0, 1)), over
    the active names where every series is finite. Returns the first regressor's coefficient per kept day."""
    out = []
    k = len(xs)
    for t in range(y.shape[0]):
        m = active[t] & np.isfinite(y[t])
        for x in xs:
            m &= np.isfinite(x[t])
        n = int(m.sum())
        if n < max(int(min_names), k + 2):
            continue
        ry = (rankdata(y[t, m]) - 0.5) / n
        design = np.column_stack([np.ones(n)] + [(rankdata(x[t, m]) - 0.5) / n for x in xs])
        beta, *_ = np.linalg.lstsq(design, ry, rcond=None)
        out.append(beta[1])
    return np.asarray(out, dtype=np.float64)


def hac_t(betas: np.ndarray, lags: int) -> tuple[float, float]:
    """(t, n_eff): mean / sqrt(γ0 / n_eff), γ0 the population variance, n_eff from ``hac_effective_n``."""
    b = betas[np.isfinite(betas)]
    if b.size < 3:
        return math.nan, float(b.size)
    g0 = float(np.var(b))
    n_eff = hac_effective_n(b, lags)
    if g0 <= 0:
        return math.nan, n_eff
    return float(b.mean() / math.sqrt(g0 / n_eff)), n_eff


def run_p3(name: str, phase1: Mapping, scores: pd.DataFrame, baselines: pd.DataFrame, calendar: np.ndarray,
           panel: Panel, gates: Gates) -> dict:
    cfg = phase1["p3_baseline"]
    lags, min_t = int(cfg["nw_lags"]), float(cfg["min_marginal_t"])
    sig = registered(phase1, scores, calendar, [name])[name]
    cons, rule = Construction.from_phase1(phase1), load_release_rule(phase1)
    cal = np.asarray(calendar, dtype="datetime64[D]")
    base = [baseline_signal(f"{name}~{col}", baselines, col, sig.hold, cons, cal, rule)
            for col in ("lm_tone", "similarity")]
    ns = sig.spec.neutralization
    xs = [compute_scores(s, panel, ns) * s.spec.expected_sign for s in (sig, *base)]
    h = gates.primary_horizon
    betas = fama_macbeth_betas(panel.forward_returns(h), xs, panel.active, gates.min_names_per_day)
    t, n_eff = hac_t(betas, lags)
    return {"signal": name, "horizon": h, "n_days": int(betas.size), "beta_mean": float(np.mean(betas)) if betas.size
            else math.nan, "n_eff": n_eff, "t": t, "min_t": min_t, "nw_lags": lags,
            "pass": bool(math.isfinite(t) and t >= min_t)}


# --------------------------------------------------------------------------- K3
def k3_decision(ranked_promising: Sequence[str], p2: Mapping[str, Mapping], p3: Mapping[str, Mapping]) -> dict:
    """Screening passers in the funnel's rank order; K3 fires when there are none."""
    passers = [n for n in ranked_promising if p2.get(n, {}).get("pass") is True and p3.get(n, {}).get("pass") is True]
    return {"promising": list(ranked_promising), "passers": passers, "k3_fired": not passers,
            "p4_candidate": passers[0] if passers else None,
            "verdict": "NO-GO (K3): the clean window stays closed" if not passers else "PROCEED to P4"}
