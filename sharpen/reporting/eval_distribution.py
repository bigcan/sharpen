"""Protocol v2.2 §2 `eval_distribution` helper.

Produces the action-distribution block logged in `seed_report.json` (Stage 2)
and `ensemble_report.json` (Stage 2.5). The bucketed `by_vol_quartile` form
is the live-monitoring baseline for §8.2 action drift.

Scalar action spaces (V7 `Box(-1,1,(1,))`) → scalar histogram + moments.
Multi-dim action spaces (CryptoPerp, Funding-Arb) → per-asset marginal
histograms under `by_asset.<asset_key>`.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# v2.2 reference bins. 8 uniform-width bins over the [-1, 1] action range.
# Edges are inclusive-left, exclusive-right except the last bin which is
# inclusive-right (numpy.histogram default).
DEFAULT_HIST_EDGES: tuple[float, ...] = (
    -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0,
)

# Per v2.2 §8.2 defaults. Deadband/saturation thresholds define how the
# baseline is binarized for §8.2 deadband_frac_delta / saturation_frac_delta
# drift signals. Env-specific overrides flow through `deadband`/`saturation`.
DEFAULT_DEADBAND_ABS: float = 0.25
DEFAULT_SATURATION_ABS: float = 0.95


def _shannon_entropy(counts: np.ndarray) -> float:
    """Shannon entropy (nats) of a histogram-counts vector."""
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


def summarize_scalar_actions(
    actions: np.ndarray,
    *,
    deadband: float = DEFAULT_DEADBAND_ABS,
    saturation: float = DEFAULT_SATURATION_ABS,
    hist_edges: Sequence[float] = DEFAULT_HIST_EDGES,
) -> dict:
    """Summarize a 1-D action array into the v2.2 eval_distribution sub-block.

    Keys: histogram_bins, counts, mean, std, entropy, deadband_frac,
    saturation_frac, no_consensus_frac, n. ``n`` is the count of bars with
    *non-NaN* actions (i.e., consensus bars), so downstream comparison
    across baselines and live windows is on apples-to-apples consensus
    populations.

    Fix 2 (S538-cont, 2026-05-08): the ensemble aggregator emits NaN on
    no >=2 directional consensus. NaN bars are excluded from
    deadband_frac / saturation_frac numerators *and* denominators, and
    the share is reported separately as ``no_consensus_frac``. Pre-Fix-2
    NaN was impossible (aggregator returned 0); now it must be filtered
    so the baseline written here matches the live tracker's NaN-aware
    measurement.
    """
    a_all = np.asarray(actions, dtype=np.float64).ravel()
    n_all = int(a_all.size)
    nan_mask = np.isnan(a_all)
    n_no_consensus = int(nan_mask.sum())
    a = a_all[~nan_mask]
    no_consensus_frac = float(n_no_consensus / n_all) if n_all else 0.0

    if a.size == 0:
        return {
            "histogram_bins": list(hist_edges),
            "counts": [0] * (len(hist_edges) - 1),
            "mean": 0.0,
            "std": 0.0,
            "entropy": 0.0,
            "deadband_frac": 0.0,
            "saturation_frac": 0.0,
            "no_consensus_frac": no_consensus_frac,
            "n": 0,
            "n_total_bars": n_all,
        }

    counts, _ = np.histogram(a, bins=list(hist_edges))
    abs_a = np.abs(a)
    return {
        "histogram_bins": list(hist_edges),
        "counts": counts.astype(int).tolist(),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=0)),
        "entropy": _shannon_entropy(counts.astype(np.float64)),
        "deadband_frac": float((abs_a < deadband).mean()),
        "saturation_frac": float((abs_a > saturation).mean()),
        "no_consensus_frac": no_consensus_frac,
        "n": int(a.size),
        "n_total_bars": n_all,
    }


def _bucket_indices(
    vol_series: np.ndarray,
    regime_quartiles: Optional[dict],
) -> Optional[tuple[dict[str, np.ndarray], list[float]]]:
    """Map a per-bar vol series to q1..q4 bucket index arrays + cutpoints.

    `regime_quartiles` holds per-quartile-share-of-bars from the data manifest
    (e.g. {"vol_q1": 0.27, ...}). The absolute cutpoints come from per-sample
    quantile on the live series: q1 = [0, 25%), q2 = [25%, 50%), etc. We use
    the manifest's key naming (vol_q1..vol_q4) only to validate that the four
    quartiles exist; the actual cutpoints are computed from `vol_series`
    because the manifest stores *shares*, not cutpoint values.

    Returns `(masks, cutpoints)` where `cutpoints` is the [q25,q50,q75] list
    used to classify bars. Live monitoring must consume these cutpoints to
    bucket incoming bars consistently with the baseline.
    """
    if regime_quartiles is None:
        return None
    required = {"vol_q1", "vol_q2", "vol_q3", "vol_q4"}
    if not required.issubset(regime_quartiles.keys()):
        return None
    v = np.asarray(vol_series, dtype=np.float64).ravel()
    finite = v[np.isfinite(v)]
    if finite.size < 4:
        return None
    cuts = np.quantile(finite, [0.25, 0.5, 0.75])
    idx = np.digitize(v, cuts)  # 0..3 corresponding to q1..q4
    masks = {
        "q1": idx == 0,
        "q2": idx == 1,
        "q3": idx == 2,
        "q4": idx == 3,
    }
    return masks, [float(c) for c in cuts]


def compute_eval_distribution(
    actions: np.ndarray,
    *,
    bar_vol: Optional[np.ndarray] = None,
    regime_quartiles: Optional[dict] = None,
    deadband: float = DEFAULT_DEADBAND_ABS,
    saturation: float = DEFAULT_SATURATION_ABS,
    asset_keys: Optional[Sequence[str]] = None,
    composition_rule: Optional[str] = None,
) -> dict:
    """Compute the Protocol v2.2 §2 `eval_distribution` block.

    Scalar action case (`actions.ndim == 1` or shape (N,1)): returns the
    scalar schema with optional `by_vol_quartile`. Multi-dim (shape (N, K))
    emits `by_asset.<asset_key>` instead — each asset summarized
    independently. `composition_rule` is attached unchanged (used by the
    Stage 2.5 writer for ensemble distributions).

    When `bar_vol` or `regime_quartiles` is missing, `by_vol_quartile` is
    omitted and a `regime_bucketing` diagnostic records the reason so
    downstream §8.2 consumers can fall back to log-only mode.
    """
    a = np.asarray(actions, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2:
        raise ValueError(f"actions must be 1-D or 2-D, got shape {a.shape}")

    n_samples, n_dim = a.shape
    is_scalar = n_dim == 1

    if is_scalar:
        block = summarize_scalar_actions(
            a[:, 0], deadband=deadband, saturation=saturation,
        )
    else:
        if asset_keys is not None and len(asset_keys) != n_dim:
            raise ValueError(
                f"asset_keys length {len(asset_keys)} != action dim {n_dim}"
            )
        keys = list(asset_keys) if asset_keys is not None else [
            f"asset_{i}" for i in range(n_dim)
        ]
        per_asset = {
            keys[i]: summarize_scalar_actions(
                a[:, i], deadband=deadband, saturation=saturation,
            )
            for i in range(n_dim)
        }
        # Aggregate moments for multi-dim are per-asset; the top-level block
        # records `n` plus the by_asset sub-structure. Live drift compares
        # per-asset marginals (v2.2 §8.2: joint KL intractable at 20-asset).
        block = {
            "histogram_bins": list(DEFAULT_HIST_EDGES),
            "by_asset": per_asset,
            "n": int(n_samples),
        }

    if composition_rule is not None:
        block["composition_rule"] = composition_rule

    # ----- regime bucketing -----
    if bar_vol is None:
        block["regime_bucketing"] = "unavailable: bar_vol not provided"
        return block
    bar_vol_arr = np.asarray(bar_vol, dtype=np.float64).ravel()
    if bar_vol_arr.size != n_samples:
        block["regime_bucketing"] = (
            f"unavailable: bar_vol length {bar_vol_arr.size} != actions {n_samples}"
        )
        return block

    bucket_result = _bucket_indices(bar_vol_arr, regime_quartiles)
    if bucket_result is None:
        block["regime_bucketing"] = "unavailable: regime_quartiles missing/invalid"
        return block
    masks, cutpoints = bucket_result

    by_vq: dict[str, dict] = {}
    for q_key in ("q1", "q2", "q3", "q4"):
        mask = masks[q_key]
        if is_scalar:
            by_vq[q_key] = summarize_scalar_actions(
                a[mask, 0], deadband=deadband, saturation=saturation,
            )
        else:
            keys = list(asset_keys) if asset_keys is not None else [
                f"asset_{i}" for i in range(n_dim)
            ]
            by_vq[q_key] = {
                "by_asset": {
                    keys[i]: summarize_scalar_actions(
                        a[mask, i], deadband=deadband, saturation=saturation,
                    )
                    for i in range(n_dim)
                },
                "n": int(mask.sum()),
            }
    block["by_vol_quartile"] = by_vq
    # v2.2 §8.2 EVAL-DIST-CUTPOINTS-01 fix: record the cutpoints used to
    # classify bars into q1..q4. Live ActionDriftTracker reads these to
    # bucket incoming bars consistently with the baseline.
    block["regime_cutpoints"] = cutpoints
    return block
