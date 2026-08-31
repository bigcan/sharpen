"""Prop-firm challenge simulator — Monte-Carlo P(hit target before breaching DD/daily limit).

Fable review item 1 (`.agent/artifacts/tailwind_v1_fable_review.md`), the highest-ROI prop
item: the challenge objective is NOT long-run Sharpe. It is a **path-dependent, finite-horizon,
asymmetric** problem — maximize P(reach the profit target before breaching the max total
drawdown OR the daily loss limit) inside the challenge window, with downside capped at the fee.
This module block-bootstraps a strategy's daily return stream (preserving fat tails +
autocorrelation), applies a sizing policy (leverage, bank-and-derisk), enforces a firm's exact
rule set on each path, and reports P(pass), E[days], and the breach-cause split.

Rule conventions (match the project's live/eval logic):
  * profit target  — cumulative return >= `profit_target` (step1 0.10 / step2 0.05,
    `live/challenge_state_machine.ChallengePhase`), reachable only after `min_trading_days`.
  * max total DD   — `static`: equity < initial·(1 − max_total_dd) (FTMO "max loss" floor from
    initial); `trailing`: equity < running-peak·(1 − max_total_dd)
    (`eval/fixed_lot_stress._max_drawdown_of_curve`).
  * daily loss     — a day's loss relative to the day-start equity >= `daily_loss_limit`.

**KEY MODELING LIMITATION (be honest about it).** The book is DAILY-rebalanced and held
overnight, so the daily-loss check on CLOSE-TO-CLOSE returns UNDERSTATES real breach risk: a
prop firm measures the daily loss on INTRADAY equity, and an overnight-held book can breach
intraday then recover by the close. `intraday_mae_mult` (>=1) inflates the daily loss for the
limit check to approximate the intraday max-adverse-excursion; default 1.0 (close-to-close) with
a note to sensitivity-test 1.3-1.6 on a daily trend book. All P(pass) figures are model outputs
to be validated, not guarantees.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

TRADING_DAYS = 252


@dataclass(frozen=True)
class FirmRules:
    """One challenge phase's rule set. Verify against the firm's CURRENT published rules."""
    name: str
    profit_target: float                 # e.g. 0.10 (FTMO step1), 0.05 (step2)
    max_total_dd: float                  # e.g. 0.10 (fraction; breach threshold)
    daily_loss_limit: float              # e.g. 0.05 (fraction of day-start equity)
    max_days: int | None = 30            # challenge window in trading days; None = no deadline (cap applies)
    min_trading_days: int = 0            # firm minimum before a pass counts
    dd_mode: str = "static"              # "static" (floor from initial) | "trailing" (from peak)

    def __post_init__(self) -> None:
        if self.dd_mode not in ("static", "trailing"):
            raise ValueError(f"dd_mode must be 'static' or 'trailing', got {self.dd_mode!r}")
        for k in ("profit_target", "max_total_dd", "daily_loss_limit"):
            if getattr(self, k) <= 0:
                raise ValueError(f"{k} must be > 0, got {getattr(self, k)}")


@dataclass(frozen=True)
class SizingPolicy:
    """How the book is sized during the challenge (the levers the review says matter)."""
    vol_multiplier: float = 1.0          # leverage on the base-vol-normalized book (2.0 = 2x base vol)
    bank_threshold: float | None = None  # cumulative return at which to de-risk (None = never bank)
    derisk_multiplier: float = 0.5       # vol multiplier is scaled by this once banked
    intraday_mae_mult: float = 1.0       # inflate the daily loss for the limit check (intraday MAE proxy)

    def __post_init__(self) -> None:
        if self.vol_multiplier <= 0:
            raise ValueError("vol_multiplier must be > 0")
        if self.intraday_mae_mult < 1.0:
            raise ValueError("intraday_mae_mult must be >= 1.0")


# Outcome codes
PASS, DD_BREACH, DAILY_BREACH, TIMEOUT = "pass", "dd_breach", "daily_breach", "timeout"


def normalize_to_vol(returns: Sequence[float], target_vol_ann: float = 0.10,
                     trading_days: int = TRADING_DAYS) -> np.ndarray:
    """Scale a daily return series to a target ANNUALIZED vol (Sharpe-invariant constant scalar),
    so a policy's ``vol_multiplier`` maps to ``target_vol_ann · vol_multiplier`` annualized vol."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1)
    if sd <= 0:
        return r
    return r * (target_vol_ann / (sd * np.sqrt(trading_days)))


def moving_block_bootstrap(returns: Sequence[float], n_paths: int, path_len: int, *,
                           block: int = 10, rng: np.random.Generator,
                           crash_returns: Sequence[float] | None = None,
                           crash_block_prob: float = 0.0) -> np.ndarray:
    """``(n_paths, path_len)`` moving-block bootstrap (preserves autocorrelation + fat tails).

    Each path is assembled from length-``block`` contiguous slices sampled with replacement from
    ``returns``. With probability ``crash_block_prob`` a block is instead drawn from
    ``crash_returns`` (a crash-period subset) — a stress overlay that OVER-weights tail regimes
    beyond their historical frequency. ``crash_block_prob=0`` = pure in-sample bootstrap (crashes
    still appear at their natural frequency)."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < block:
        raise ValueError(f"series length {n} < block {block}")
    n_blocks = int(np.ceil(path_len / block))
    offs = np.arange(block)
    # Vectorized moving-block gather: draw n_blocks start indices per path, expand to per-day
    # indices, index the return series in one shot (no per-path Python loop).
    starts = rng.integers(0, n - block + 1, size=(n_paths, n_blocks))
    idx = starts[:, :, None] + offs[None, None, :]                 # (P, B, block)
    out = r[idx].reshape(n_paths, n_blocks * block)[:, :path_len]
    cr = None if crash_returns is None else np.asarray(crash_returns, dtype=np.float64)
    if cr is not None:
        cr = cr[np.isfinite(cr)]
        if len(cr) >= block and crash_block_prob > 0.0:
            cmask = rng.random((n_paths, n_blocks)) < crash_block_prob
            cstarts = rng.integers(0, len(cr) - block + 1, size=(n_paths, n_blocks))
            cidx = cstarts[:, :, None] + offs[None, None, :]
            cblocks = cr[cidx].reshape(n_paths, n_blocks * block)[:, :path_len]
            cmask_day = np.repeat(cmask, block, axis=1)[:, :path_len]   # per-day crash flag
            out = np.where(cmask_day, cblocks, out)
    return out


def simulate_path(path: np.ndarray, firm: FirmRules, policy: SizingPolicy) -> tuple[str, int, float]:
    """Run one bootstrapped path under the firm rules + sizing policy.

    Returns ``(outcome, day_index_1based, final_cumulative_return)``. Order of checks each day:
    (1) bank-and-derisk if the threshold is reached; (2) daily-loss breach (with the intraday MAE
    proxy) — checked BEFORE booking, since a daily breach ends the day; (3) book the return;
    (4) total-DD breach; (5) profit target (only after min_trading_days). Target beats DD on the
    same day only if the day's close is both above target AND above the DD floor (a day that
    closes below the floor is a breach regardless of an intraday target touch — the conservative
    reading for an overnight-held book)."""
    equity = 1.0
    peak = 1.0
    vm = policy.vol_multiplier
    banked = False
    for day in range(len(path)):
        cum = equity - 1.0
        if policy.bank_threshold is not None and not banked and cum >= policy.bank_threshold:
            vm *= policy.derisk_multiplier
            banked = True
        r = vm * float(path[day])
        # (2) daily-loss breach on the day-start equity (intraday MAE proxy inflates a loss).
        eff_loss = r if r >= 0 else r * policy.intraday_mae_mult
        if -eff_loss >= firm.daily_loss_limit:
            return DAILY_BREACH, day + 1, equity - 1.0
        # (3) book
        equity *= (1.0 + r)
        peak = max(peak, equity)
        # (4) total-DD breach
        floor = (1.0 - firm.max_total_dd) if firm.dd_mode == "static" \
            else peak * (1.0 - firm.max_total_dd)
        if equity <= floor:
            return DD_BREACH, day + 1, equity - 1.0
        # (5) profit target
        if equity - 1.0 >= firm.profit_target and (day + 1) >= firm.min_trading_days:
            return PASS, day + 1, equity - 1.0
    return TIMEOUT, len(path), equity - 1.0


# Integer outcome codes for the vectorized core (mapped back to strings in `evaluate`).
_ACTIVE, _PASS, _DD, _DAILY, _TIMEOUT = 0, 1, 2, 3, 4


def _simulate_vectorized(paths: np.ndarray, firm: FirmRules,
                         policy: SizingPolicy) -> tuple[np.ndarray, np.ndarray]:
    """Run ALL paths in lockstep (day-by-day numpy) — same logic as :func:`simulate_path`,
    ~1000× faster. Returns ``(outcome_codes, day_1based)`` per path."""
    n = paths.shape[0]
    equity = np.ones(n)
    peak = np.ones(n)
    vm = np.full(n, policy.vol_multiplier)
    banked = np.zeros(n, dtype=bool)
    active = np.ones(n, dtype=bool)
    outcome = np.zeros(n, dtype=np.int8)          # _ACTIVE
    day_out = np.zeros(n, dtype=np.int64)
    static = firm.dd_mode == "static"
    for day in range(paths.shape[1]):
        if policy.bank_threshold is not None:
            to_bank = active & (~banked) & ((equity - 1.0) >= policy.bank_threshold)
            vm = np.where(to_bank, vm * policy.derisk_multiplier, vm)
            banked |= to_bank
        r = vm * paths[:, day]
        eff_loss = np.where(r >= 0, r, r * policy.intraday_mae_mult)
        db = active & (-eff_loss >= firm.daily_loss_limit)
        outcome[db], day_out[db] = _DAILY, day + 1
        active &= ~db
        equity = np.where(active, equity * (1.0 + r), equity)
        peak = np.maximum(peak, equity)
        floor = (1.0 - firm.max_total_dd) if static else peak * (1.0 - firm.max_total_dd)
        ddb = active & (equity <= floor)
        outcome[ddb], day_out[ddb] = _DD, day + 1
        active &= ~ddb
        tgt = active & ((equity - 1.0) >= firm.profit_target) & ((day + 1) >= firm.min_trading_days)
        outcome[tgt], day_out[tgt] = _PASS, day + 1
        active &= ~tgt
        if not active.any():
            break
    outcome[active] = _TIMEOUT
    day_out[active] = paths.shape[1]
    return outcome, day_out


def evaluate(returns: Sequence[float], firm: FirmRules, policy: SizingPolicy, *,
             n_paths: int = 10000, block: int = 10, base_vol_ann: float = 0.10,
             seed: int = 7, crash_returns: Sequence[float] | None = None,
             crash_block_prob: float = 0.0, no_deadline_cap: int = 504) -> dict:
    """Bootstrap ``n_paths`` and simulate each (vectorized); return the aggregate distribution.

    ``returns`` is normalized to ``base_vol_ann`` first, so ``policy.vol_multiplier`` is leverage
    on that base vol. ``path_len`` = ``firm.max_days`` (fixed window) or ``no_deadline_cap`` (a
    no-deadline firm runs until pass/breach up to the cap)."""
    r_norm = normalize_to_vol(returns, base_vol_ann)
    path_len = firm.max_days if firm.max_days is not None else no_deadline_cap
    rng = np.random.default_rng(seed)
    paths = moving_block_bootstrap(r_norm, n_paths, path_len, block=block, rng=rng,
                                   crash_returns=(None if crash_returns is None
                                                  else normalize_to_vol(crash_returns, base_vol_ann)),
                                   crash_block_prob=crash_block_prob)
    codes, days = _simulate_vectorized(paths, firm, policy)
    is_pass = codes == _PASS
    pass_days = days[is_pass]
    return {
        "firm": firm.name, "vol_multiplier": policy.vol_multiplier,
        "bank_threshold": policy.bank_threshold, "derisk_multiplier": policy.derisk_multiplier,
        "intraday_mae_mult": policy.intraday_mae_mult,
        "base_vol_ann": base_vol_ann, "effective_vol_ann": round(base_vol_ann * policy.vol_multiplier, 4),
        "n_paths": n_paths, "path_len_days": path_len, "block": block,
        "crash_block_prob": crash_block_prob,
        "p_pass": round(float(is_pass.mean()), 4),
        "p_dd_breach": round(float((codes == _DD).mean()), 4),
        "p_daily_breach": round(float((codes == _DAILY).mean()), 4),
        "p_timeout": round(float((codes == _TIMEOUT).mean()), 4),
        "e_days_to_pass": (round(float(pass_days.mean()), 1) if pass_days.size else None),
        "median_days_to_pass": (int(np.median(pass_days)) if pass_days.size else None),
        "binding_constraint": ("daily_limit" if (codes == _DAILY).sum()
                               > (codes == _DD).sum() else "max_dd"),
    }


@dataclass
class SweepResult:
    rows: list[dict] = field(default_factory=list)

    def best(self, key: str = "p_pass") -> dict:
        return max(self.rows, key=lambda d: d[key]) if self.rows else {}


def sweep(returns: Sequence[float], firm: FirmRules, *, vol_multipliers: Sequence[float],
          bank_thresholds: Sequence[float | None], derisk_multiplier: float = 0.5,
          intraday_mae_mult: float = 1.0, **eval_kw) -> SweepResult:
    """Grid-sweep {vol_multiplier × bank_threshold} for one firm; return every cell's outcome."""
    res = SweepResult()
    for vm in vol_multipliers:
        for bt in bank_thresholds:
            pol = SizingPolicy(vol_multiplier=vm, bank_threshold=bt,
                               derisk_multiplier=derisk_multiplier,
                               intraday_mae_mult=intraday_mae_mult)
            res.rows.append(evaluate(returns, firm, pol, **eval_kw))
    return res
