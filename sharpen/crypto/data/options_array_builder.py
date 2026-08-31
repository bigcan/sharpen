"""Options panel builder — assembles per-bar arrays consumed *env-free* by the
Phase-1 VRP falsification (and, later, by ``OptionsVolHarvestEnv``).

For the Phase-1 gate the tradeable vol instrument is a synthetic constant-maturity
ATM straddle repriced from DVOL inside the simulator, so this builder only needs
to deliver the clean, date-aligned spot / IV / RV / funding panels plus the
constant-maturity **roll schedule**. The full per-instrument greeks panel
(``instr_premium``/``greeks``/``instr_cost`` for the real expired-chain path) is a
documented Phase-3 extension (see
``.agent/artifacts/options_vol_harvest_architecture.md``) and is intentionally
left as ``None`` here rather than stubbed with fabricated values.

Mirrors ``crypto/data/crypto_array_builder.build_env_arrays`` (asset-agnostic,
returns aligned numpy arrays; within-window renorm / LEAK-1 is the consumer's job).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sharpen.crypto.features import options_vol_features as ovf

logger = logging.getLogger(__name__)


@dataclass
class OptionsPanels:
    """Date-aligned per-asset panels (T = #dates, A = #assets).

    Phase-1 fields are populated; Phase-3 (real-chain) fields stay ``None``.
    """
    assets: list[str]
    timestamps: np.ndarray              # (T,) int64 epoch-ms
    spot_ary: np.ndarray               # (T, A) perp close (hedge/PnL leg)
    iv_ary: np.ndarray                 # (T, A) ATM 30d implied vol (fraction)
    iv_rv_spread_ary: np.ndarray       # (T, A) atm_iv - rv_30 (the VRP signal)
    rv_ary: dict                       # window -> (T, A) trailing realized vol
    funding_ary: np.ndarray            # (T, A) daily perp funding (fraction)
    # --- Phase-3 (real expired-chain instruments) — not built for the gate ---
    instr_premium: np.ndarray | None = None
    greeks: dict | None = None
    instr_cost: np.ndarray | None = None
    baseline_book: np.ndarray | None = None

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.to_datetime(self.timestamps, unit="ms", utc=True)


def build_panels(raw, config: dict | None = None) -> OptionsPanels:
    """Align per-asset feature frames onto a common date index and stack to arrays.

    Uses an **inner** join across assets so every row has all assets present
    (no NaN-padding that would silently distort portfolio aggregation).
    """
    config = config or {}
    feat_cfg = config.get("features", {})
    rv_windows = tuple(feat_cfg.get("rv_windows", (10, 30, 90)))
    skip_bars = int(feat_cfg.get("skip_bars", 0))
    ref_w = int(feat_cfg.get("iv_rv_ref_window", 30))

    feats = ovf.compute(raw, rv_windows=rv_windows, skip_bars=skip_bars,
                        iv_rv_ref_window=ref_w)
    assets = list(feats.keys())

    # Common date index (inner join) across all assets.
    common = None
    for ccy in assets:
        idx = feats[ccy].index
        common = idx if common is None else common.intersection(idx)
    common = common.sort_values()

    def _stack(col: str) -> np.ndarray:
        return np.column_stack([feats[ccy].loc[common, col].to_numpy(float) for ccy in assets])

    panels = OptionsPanels(
        assets=assets,
        timestamps=(common.asi8 // 1_000_000),
        spot_ary=_stack("spot"),
        iv_ary=_stack("atm_iv"),
        iv_rv_spread_ary=_stack("iv_rv_spread"),
        rv_ary={w: _stack(f"rv_{w}") for w in rv_windows},
        funding_ary=_stack("funding_daily"),
    )
    logger.info("Built panels: assets=%s, T=%d (%s -> %s)",
                assets, len(common), common[0].date(), common[-1].date())
    return panels
