"""Regression tripwire for the crypto multiscale coarse-bar X2 leak (P2-01).

Mirrors ``tests/data/test_multiscale_causality.py`` (the V7 single-asset guard)
for the multi-asset crypto path used by CMGP1 / CryptoPerpSwingEnv.

The 2026-06-01 deep lifecycle audit (P2-01, S1) reproduced the leak:
``MultiScaleCryptoHandler`` resamples coarse bars with ``label='left'`` (stamped
at INTERVAL START) but built its base->coarse index map with the leaky
``np.searchsorted(coarse_ts, base_ts, 'right') - 1`` (no ``-scale_ns``), so every
base bar inside an in-progress coarse interval was mapped to that not-yet-closed
coarse bar — leaking up to ``scale - base_scale`` HOURS of its own future into the
window mean/std AND the ``last`` summary value. Crypto scales are HOURS, so the
fix subtracts ``scale * 3600 * 1e9`` ns (vs the V7 handler's ``* 60`` for minutes).

Causal invariant under test: the coarse bar assigned to a base bar at time ``t``
must have already CLOSED at or before ``t`` — i.e. ``coarse_start + scale <= t``.

This test fails on the pre-fix leaky map (the leaky counterfactual is checked
explicitly in ``test_leaky_map_would_be_caught``) and passes on the fixed map.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sharpen.crypto.data.multiscale_crypto_handler import MultiScaleCryptoHandler


def _build_ohlcv(assets: list[str], n_hours: int, start: str = "2025-01-01") -> pd.DataFrame:
    """Merged OHLCV on a gentle deterministic ramp, full grid for every asset."""
    ts = pd.date_range(start, periods=n_hours, freq="1h")
    rows = []
    for ai, asset in enumerate(assets):
        mid = 100.0 + 1000.0 * ai + np.cumsum(np.full(n_hours, 0.01))
        for i, t in enumerate(ts):
            rows.append({
                "timestamp": t,
                "ticker": asset,
                "open": mid[i],
                "high": mid[i] + 0.5,
                "low": mid[i] - 0.5,
                "close": mid[i],
                "volume": 1000.0 + i,
            })
    return pd.DataFrame(rows)


def _feature_cfg() -> dict:
    return {
        "scales": [1, 4, 24],
        "window_size": 10,
        "obs_mode": "summary_stats",
        "summary_feature_indices": [0, 1, 2, 6, 7],
        "norm_span": 120,
        "feature_set_version": "v1",
    }


def _make_handler() -> MultiScaleCryptoHandler:
    assets = ["BTC", "ETH", "SOL"]
    # 15 days → the 24h coarse scale has ~15 closed bars to map against.
    ohlcv = _build_ohlcv(assets, n_hours=15 * 24)
    return MultiScaleCryptoHandler(
        ohlcv_df=ohlcv,
        funding_df=None,
        assets=assets,
        feature_config=_feature_cfg(),
    )


def test_coarse_scale_alignment_is_causal():
    """The fixed base->coarse map assigns only CLOSED coarse bars (no look-ahead)."""
    handler = _make_handler()

    base_ns = np.asarray(handler._base_timestamps).astype("datetime64[ns]").astype("int64")
    base_scale = min(handler.scales)

    for scale in handler.scales:
        if scale == base_scale:
            continue
        coarse_ns = (
            np.asarray(handler._scale_timestamps[scale]).astype("datetime64[ns]").astype("int64")
        )
        idx_map = np.asarray(handler._scale_index_map[scale])
        scale_ns = scale * 3600 * 1_000_000_000  # HOURS, not minutes
        coarse_close_ns = coarse_ns[idx_map] + scale_ns  # interval END of the assigned bar

        # Only assert over base bars old enough that a fully-CLOSED coarse bar exists;
        # the leading bars before the first coarse close legitimately have no causal
        # coarse data (env window_size warmup covers them).
        has_closed_bar = base_ns >= (coarse_ns[0] + scale_ns)
        leaked = (coarse_close_ns > base_ns) & has_closed_bar
        assert not leaked.any(), (
            f"scale={scale}h: {int(leaked.sum())}/{int(has_closed_bar.sum())} eligible base "
            f"bars map to a coarse bar that has not yet closed (forward look-ahead of up to "
            f"{scale - base_scale}h)"
        )


def test_leaky_map_would_be_caught():
    """The pre-fix leaky map (no -scale_ns) violates the causal invariant.

    Proves this is a genuine tripwire: the historical leaky index map maps EVERY
    eligible base bar to a not-yet-closed coarse bar at both coarse scales.
    """
    handler = _make_handler()
    base_ns = np.asarray(handler._base_timestamps).astype("datetime64[ns]").astype("int64")
    base_scale = min(handler.scales)

    for scale in handler.scales:
        if scale == base_scale:
            continue
        coarse_ns = (
            np.asarray(handler._scale_timestamps[scale]).astype("datetime64[ns]").astype("int64")
        )
        # Reconstruct the OLD leaky map: searchsorted on base_ts directly.
        leaky_idx = np.searchsorted(coarse_ns, base_ns, side="right") - 1
        leaky_idx = np.clip(leaky_idx, 0, len(coarse_ns) - 1)

        scale_ns = scale * 3600 * 1_000_000_000
        coarse_close_ns = coarse_ns[leaky_idx] + scale_ns
        has_closed_bar = base_ns >= (coarse_ns[0] + scale_ns)
        leaked = (coarse_close_ns > base_ns) & has_closed_bar
        assert leaked.any(), (
            f"scale={scale}h: leaky map should leak but did not — tripwire is not "
            f"actually exercising the look-ahead it guards against"
        )
