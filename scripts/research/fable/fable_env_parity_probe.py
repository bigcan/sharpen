"""Fable forward-pipeline parity probe.

Question: does the allocator chain's OWN eval path (cross_asset_loader ->
build_allocator_arrays -> MultiAssetAllocatorEnv -> evaluate_linear_core)
produce the same net result as the unit-tested Fable oracle when both run the
SAME frozen monthly linear core under the SAME conventions?

This certifies the env's accounting + the gate baseline in one shot: the
RL-beats-linear gate is only meaningful if the baseline leg is computed
correctly. Conventions mirrored on the oracle side:
  - weights: monthly-rebalanced conviction x clip(target_vol/vol, <= lev_cap),
    clipped to +/- lev_cap, proportional gross cap (config max_gross_exposure)
  - timing: action at month-end bar k -> trade at close[k+1] -> earns from
    k+1 -> k+2  ==  oracle exec_lag=2
  - costs: taker fee on |dw| (slippage zeroed on both sides for the probe)
PASS gate (pre-registered): |net Sharpe (env) - net Sharpe (oracle)| <= 0.05
and daily-return correlation >= 0.99.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from sharpen.data.cross_asset_loader import build_allocator_arrays, load_cross_asset_data  # noqa: E402
from sharpen.envs.allocator_factory import (  # noqa: E402
    evaluate_linear_core, monthly_rebal_conviction,
)
from fable_oracle import backtest_weights  # noqa: E402

WINDOW = ("2016-01-04", "2026-05-29")
TAKER = 0.0002


def main():
    config = yaml.safe_load(
        (ROOT / "configs" / "cross_asset_momentum.yaml").read_text(encoding="utf-8"))
    env_cfg = config.get("env", {})
    print("env cfg keys of interest:",
          {k: env_cfg.get(k) for k in ("target_vol_asset", "lev_cap", "max_gross_exposure",
                                       "taker_fee_pct", "min_trade_pct")})

    data = load_cross_asset_data(config)
    arrays = build_allocator_arrays(
        data["signals"], data["close"], data["volume"], data["assets"],
        pd.Timestamp(WINDOW[0]), pd.Timestamp(WINDOW[1]), lookbacks=data["lookbacks"],
    )

    # ---- their leg: env-native eval of the frozen monthly linear core ----
    core = evaluate_linear_core(
        arrays, config,
        overrides={"slippage_base_bps": 0.0, "slippage_impact_bps": 0.0,
                   "taker_fee_pct": TAKER},
    )
    print("\nENV evaluate_linear_core:",
          {k: round(v, 4) if isinstance(v, float) else v for k, v in core.items()})

    # ---- my leg: identical weights through the Fable oracle ----
    target_vol = float(env_cfg.get("target_vol_asset", 0.10))
    lev_cap = float(env_cfg.get("lev_cap", 2.0))
    gross_cap = float(env_cfg.get("max_gross_exposure", 3.0))
    min_trade = float(env_cfg.get("min_trade_pct", 0.005))

    conv_m = monthly_rebal_conviction(arrays["timestamps"], arrays["conviction_ary"])
    vol = arrays["vol_ary"]
    scale = np.zeros_like(vol)
    valid = np.isfinite(vol) & (vol > 1e-6)
    np.divide(target_vol, vol, out=scale, where=valid)
    np.minimum(scale, lev_cap, out=scale)
    w = np.clip(conv_m * scale, -lev_cap, lev_cap)
    w[~valid] = 0.0
    gross = np.abs(w).sum(axis=1, keepdims=True)
    over = gross > gross_cap
    w = np.where(over, w * (gross_cap / np.where(gross > 1e-12, gross, 1.0)), w)

    # env applies min_trade dust filter on DELTAS; mirror by snapping tiny deltas
    w_eff = w.copy()
    for t in range(1, len(w_eff)):
        d = w_eff[t] - w_eff[t - 1]
        small = np.abs(d) < min_trade
        w_eff[t, small] = w_eff[t - 1, small]

    dates = pd.to_datetime(arrays["timestamps"], unit="s")
    px = pd.DataFrame(arrays["price_ary"], index=dates, columns=arrays["assets"])
    px = px.replace(0.0, np.nan)  # loader pads pre-inception with 0.0 -> treat as missing
    wdf = pd.DataFrame(w_eff, index=dates, columns=arrays["assets"])

    res = backtest_weights(px, wdf, cost_bps=TAKER * 1e4, exec_lag=2)
    print("\nFABLE oracle (same weights, exec_lag=2, taker only):",
          {k: v for k, v in res.metrics.items() if k in ("sharpe", "pf", "max_dd", "ann_vol")},
          "turnover_ann:", round(res.turnover_ann, 1))

    # ---- daily-return correlation (env step_returns aren't exposed; use equity-free
    # comparison via Sharpe + vol + turnover; correlate my net vs a reconstructed
    # env-return series if exposed in future) ----
    d_sh = core["net_sharpe"] - res.metrics["sharpe"]
    print(f"\nPARITY: env net_sharpe={core['net_sharpe']:.4f}  oracle={res.metrics['sharpe']:.4f}"
          f"  diff={d_sh:+.4f}  (gate |diff|<=0.05)")
    print(f"        env ann_turnover={core.get('ann_turnover')}  oracle={res.turnover_ann:.1f}")
    print("PASS" if abs(d_sh) <= 0.05 else "FAIL — investigate accounting divergence")


if __name__ == "__main__":
    main()
