"""Protocol v2.2 §2 ``challenge_target_hit_rate_per_window`` helper.

Per the prop-firm decoupling (S495, Open Question #5 resolved 2026-04-24),
L1 and WF manifests carry a per-seed (and per-ensemble) summary of how
reliably the policy hits the canonical prop-firm profit targets across
the eval window. Used as a tiebreaker in the S495 val-selection rule
when ensemble-rule PFs are within Δ < 0.1%.

A "window" is the slice of the trajectory over which a phase target
could plausibly be hit. With ``window_bars=None`` the full trajectory is
one window — adequate at L1 (single eval window) and per-fold at WF.
With ``window_bars=N`` the trajectory is sliced into non-overlapping
N-bar windows so a single long L1 eval can yield multiple windows for a
more stable hit-rate estimate.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np


# Canonical prop-firm phases. Funded has math.inf target → never hits;
# kept as an explicit row for symmetry with seed_report consumers that
# iterate the dict.
DEFAULT_PHASE_SPECS: tuple[dict, ...] = (
    {"label": "step1", "target_pct": 0.10, "window_bars": None},
    {"label": "step2", "target_pct": 0.05, "window_bars": None},
)


def _cumulative_return(pv: np.ndarray) -> np.ndarray:
    """Convert portfolio-value series to cumulative return relative to pv[0]."""
    if pv.size == 0:
        return pv
    base = float(pv[0])
    if base <= 0:
        # Defensive — caller should pass a positive baseline. Avoid divide-by-0.
        return np.zeros_like(pv, dtype=np.float64)
    return (pv.astype(np.float64) - base) / base


def compute_challenge_target_hit_rate(
    pv: np.ndarray,
    *,
    target_pct: float,
    window_bars: Optional[int] = None,
    stride_bars: Optional[int] = None,
) -> dict:
    """Hit-rate of profit-target across non-overlapping windows.

    Parameters
    ----------
    pv : array-like
        Per-bar portfolio value. ``pv[0]`` is the per-window baseline
        when ``window_bars=None``; otherwise each window's first bar is
        its own baseline.
    target_pct : float
        Profit target as fraction of baseline (e.g. 0.10 = +10%). Use
        ``math.inf`` to mark a funded/no-target phase — the function
        returns ``hit_rate=0`` for that case without erroring.
    window_bars : int or None
        Window length in bars. ``None`` ⇒ full trajectory = 1 window.
    stride_bars : int or None
        Stride between window starts. ``None`` defaults to
        ``window_bars`` (non-overlapping). Ignored when
        ``window_bars=None``.
    """
    pv_arr = np.asarray(pv, dtype=np.float64).ravel()
    n = pv_arr.size

    if not math.isfinite(target_pct):
        return {
            "target_pct": target_pct,
            "window_bars": window_bars,
            "n_windows": 0,
            "n_hits": 0,
            "hit_rate": 0.0,
            "bars_to_target_per_window": [],
            "bars_to_target_median": None,
            "max_cum_return": 0.0,
        }

    if n == 0:
        return {
            "target_pct": target_pct,
            "window_bars": window_bars,
            "n_windows": 0,
            "n_hits": 0,
            "hit_rate": 0.0,
            "bars_to_target_per_window": [],
            "bars_to_target_median": None,
            "max_cum_return": 0.0,
        }

    if window_bars is None or window_bars >= n:
        windows: list[tuple[int, int]] = [(0, n)]
    else:
        if window_bars <= 0:
            raise ValueError(f"window_bars must be > 0, got {window_bars}")
        stride = stride_bars if stride_bars is not None else window_bars
        if stride <= 0:
            raise ValueError(f"stride_bars must be > 0, got {stride}")
        windows = []
        start = 0
        while start + window_bars <= n:
            windows.append((start, start + window_bars))
            start += stride

    bars_to_target: list[Optional[int]] = []
    n_hits = 0
    max_cum_return = -math.inf
    for lo, hi in windows:
        slice_pv = pv_arr[lo:hi]
        cum = _cumulative_return(slice_pv)
        slice_max = float(cum.max()) if cum.size else 0.0
        if slice_max > max_cum_return:
            max_cum_return = slice_max
        hit_idx = np.argmax(cum >= target_pct) if cum.size else 0
        if cum.size and cum[hit_idx] >= target_pct:
            n_hits += 1
            bars_to_target.append(int(hit_idx))
        else:
            bars_to_target.append(None)

    n_windows = len(windows)
    hits_only = [b for b in bars_to_target if b is not None]
    median_btt = float(np.median(hits_only)) if hits_only else None
    if max_cum_return == -math.inf:
        max_cum_return = 0.0

    return {
        "target_pct": float(target_pct),
        "window_bars": window_bars,
        "n_windows": n_windows,
        "n_hits": n_hits,
        "hit_rate": (n_hits / n_windows) if n_windows else 0.0,
        "bars_to_target_per_window": bars_to_target,
        "bars_to_target_median": median_btt,
        "max_cum_return": float(max_cum_return),
    }


def compute_challenge_target_hit_rates(
    pv: np.ndarray,
    *,
    phase_specs: Sequence[dict] = DEFAULT_PHASE_SPECS,
) -> dict:
    """Multi-phase wrapper. Returns ``{label: per-phase hit-rate dict}``.

    ``phase_specs`` items must have keys ``label``, ``target_pct`` and
    optionally ``window_bars`` / ``stride_bars``.
    """
    out: dict[str, dict] = {}
    for spec in phase_specs:
        label = spec["label"]
        out[label] = compute_challenge_target_hit_rate(
            pv,
            target_pct=spec["target_pct"],
            window_bars=spec.get("window_bars"),
            stride_bars=spec.get("stride_bars"),
        )
    return out


def parse_phase_spec_arg(arg: str) -> list[dict]:
    """CLI helper — parse ``"step1:0.10:1170,step2:0.05"`` into phase_specs.

    Format per phase: ``label:target_pct[:window_bars[:stride_bars]]``.
    """
    specs: list[dict] = []
    for chunk in arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) < 2 or len(parts) > 4:
            raise ValueError(
                f"phase spec must be 'label:target_pct[:window_bars[:stride_bars]]', got {chunk!r}",
            )
        spec: dict = {"label": parts[0], "target_pct": float(parts[1])}
        if len(parts) >= 3 and parts[2]:
            spec["window_bars"] = int(parts[2])
        if len(parts) == 4 and parts[3]:
            spec["stride_bars"] = int(parts[3])
        specs.append(spec)
    return specs
