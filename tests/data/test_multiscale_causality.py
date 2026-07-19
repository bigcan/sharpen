"""Regression test for V7 multiscale coarse-bar causality (no forward look-ahead).

Guards the S1 leak found in the 2026-05-29 sg1-btc strategy audit
(`docs/research/sg1_btc_strategy_audit_2026-05-29.md`, finding P2-P4-01):

``multiscale_handler._resample_ohlcv`` stamps each coarse bar at its INTERVAL
START (pandas ``resample`` default ``label='left'``), and the base->coarse index
map (``multiscale_handler.py`` ``np.searchsorted(coarse_ts, base_ts, side='right') - 1``)
maps every base bar inside a coarse interval to that same start-stamped — and
still in-progress — coarse bar. A base bar early in a 15m/60m interval therefore
sees OHLCV aggregated over the WHOLE interval, i.e. up to ``scale - base`` minutes
of its own future. This leaks into every HPO/L1/WF sim-PF figure and diverges
from live (which only ever sees the partial in-progress bar).

Causal invariant under test: the coarse bar assigned to a base bar at time ``t``
must have already CLOSED at or before ``t`` — i.e. ``coarse_start + scale <= t``.

The X2 causal-alignment fix (``multiscale_handler._scale_index_map`` searchsorts on
``base_ts - scale_ns``) maps each base bar to the most recent COMPLETED coarse bar,
so this now asserts that no forward look-ahead remains. Previously
``xfail(strict=True)``; the marker was removed when X2 landed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from finrl_pro_ds.data.multiscale_handler import MultiScaleOHLCVHandler


def _write_synth_parquet(path) -> None:
    """3 days of tz-naive 1-min OHLCV on a gentle deterministic ramp."""
    n = 3 * 24 * 60
    ts = pd.date_range("2024-01-01", periods=n, freq="1min")
    mid = 100.0 + np.cumsum(np.full(n, 0.01))  # monotone, keeps OHLC invariants trivially valid
    pd.DataFrame(
        {
            "timestamp": ts,
            "open": mid,
            "high": mid + 0.5,
            "low": mid - 0.5,
            "close": mid,
            "volume": np.ones(n),
        }
    ).to_parquet(path)


def test_coarse_scale_alignment_is_causal(tmp_path):
    parquet = tmp_path / "synth_1min.parquet"
    _write_synth_parquet(parquet)

    handler = MultiScaleOHLCVHandler(
        file_path=str(parquet),
        ticker="SYNTH",
        feature_config={"scales": [3, 15, 60], "window_size": 30},
    )

    # Mirror the source dtype handling (multiscale_handler builds the map in int64 ns).
    base_ns = np.asarray(handler._base_timestamps).astype("datetime64[ns]").astype("int64")
    base_scale = min(handler.scales)

    for scale in handler.scales:
        if scale == base_scale:
            continue
        coarse_ns = (
            np.asarray(handler._scale_timestamps[scale]).astype("datetime64[ns]").astype("int64")
        )
        idx_map = np.asarray(handler._scale_index_map[scale])
        scale_ns = scale * 60 * 1_000_000_000
        coarse_close_ns = coarse_ns[idx_map] + scale_ns  # interval END of the assigned coarse bar

        # Only assert over base bars old enough that a fully-CLOSED coarse bar exists;
        # the leading bars before the first coarse close legitimately have no causal
        # coarse data (the env's window_size warmup covers them).
        has_closed_bar = base_ns >= (coarse_ns[0] + scale_ns)
        leaked = (coarse_close_ns > base_ns) & has_closed_bar
        assert not leaked.any(), (
            f"scale={scale}m: {int(leaked.sum())}/{int(has_closed_bar.sum())} eligible base bars "
            f"are mapped to a coarse bar that has not yet closed (forward look-ahead of up to "
            f"{scale - base_scale}m)"
        )
