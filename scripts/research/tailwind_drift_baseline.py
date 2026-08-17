"""TAILWIND-v1 action-drift baseline — the artifact `drift.baseline_path` points at.

Protocol v2.2 §8.2 requires a prop-firm config to declare `drift.enabled: true` and resolve a
baseline. TAILWIND has no Stage-2/2.5 `seed_report.json` (it is a LINEAR book — there is no RL
seed sweep to emit one), so the baseline is built here directly from the deployable book.

BASIS. Same three modules as `audit_tailwind_book.py` (DSR/PBO) and
`tailwind_forward_path_render.py` (forward path), so all three describe ONE book — the
"one-basis" rule the 2026-07-01 deep audit's dominant finding (P7-01: gates grade the WRONG
book) exists to enforce. Any divergence here would reintroduce exactly that defect.

WHAT IS MEASURED. The per-sleeve CONVICTION vector sampled at each monthly rebalance — the
`Box(-1, 1, (n_assets,))` quantity `MultiAssetAllocatorEnv` consumes as its action, and what
`drive_with_conviction` receives per sleeve on the executor path. NOT the post-vol-scaling
weights: those are unbounded above by `lev_cap` and would make the schema's deadband /
saturation fractions meaningless against the [-1, 1] reference bins.

SHAPE. Emits one `eval_distribution` block PER SLEEVE rather than a single top-level block.
The executor drives each sleeve's env independently (`two_sleeve.py`: "full env-render
combine, not additive conviction"), so a consumer instantiates one `ActionDriftTracker` per
sleeve. There is no meaningful single combined action vector, and inventing one would be a
fabricated baseline.

⚠️ NO CONSUMER YET. `ActionDriftTracker` is instantiated only at
`finrl_pro_ds/crypto/live/live_engine.py:140` (the RL live engine). The TAILWIND executor has
no tracker call site, so this artifact satisfies the protocol contract and is ready for the
consumer, but nothing enforces drift at runtime on this path today. Wiring that consumer is a
pre-attempt item for the Tier-2 audit.

Usage:
    python scripts/research/tailwind_drift_baseline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tailwind_v1"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))                              # finrl_pro_ds (bare-script run)
sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling research modules

import audit_tailwind_book as atb              # noqa: E402
import portfolio_frontier as pf                # noqa: E402
import xsec_momentum_falsification as mom      # noqa: E402

from finrl_pro_ds.features import defensive_signals as dfs  # noqa: E402
from finrl_pro_ds.reporting.eval_distribution import (      # noqa: E402
    DEFAULT_DEADBAND_ABS,
    DEFAULT_SATURATION_ABS,
    compute_eval_distribution,
)

BASELINE_PATH = OUT / "drift_baseline.json"
SCHEMA = "tailwind_drift_baseline/v1"

# The four keys `_bucket_indices` requires to be present. It computes the ABSOLUTE cutpoints
# from the supplied vol series itself (the manifest stores shares, not cutpoints), so equal
# shares here simply declare "bucket by sample quartile".
EQUAL_QUARTILE_SHARES = {"vol_q1": 0.25, "vol_q2": 0.25, "vol_q3": 0.25, "vol_q4": 0.25}


def sleeve_convictions():
    """(rebal, tickers, {sleeve: conviction DataFrame}) on the shared audit basis.

    Both convictions are documented `[-1, 1]` and strictly causal (value at ``t`` is a pure
    function of close ``<= t-1``); `defensive_conviction` is guarded by `assert_causal` at
    source. Sampled at the monthly rebalance dates the book actually trades on.
    """
    close, _rets, _bench, rebal = atb._mom_frame()
    tickers = list(close.columns)
    momentum = mom.tsmom_signal(close, rebal).reindex(columns=tickers)
    defensive = dfs.defensive_conviction(close, mom.CLASS_OF).reindex(rebal).reindex(
        columns=tickers,
    )
    return rebal, tickers, {"momentum": momentum, "defensive": defensive}


def book_trailing_vol(rebal):
    """Causal trailing realized vol of the COMBINED book, sampled at each rebalance.

    Regime bucketing must key off the book's own volatility state, not a single asset's.
    `.shift(1)` before sampling keeps the value at a rebalance date a function of returns
    strictly before it.
    """
    mom_net = pf.build_momentum_net()
    def_net = atb.build_defensive_net()
    combined, _idx, _sleeves = pf.risk_parity([mom_net, def_net])
    trailing = combined.rolling(mom.VOL_WIN).std().shift(1) * np.sqrt(mom.ANN)
    return trailing.reindex(rebal)


def build_baseline() -> dict:
    rebal, tickers, convictions = sleeve_convictions()
    bar_vol = book_trailing_vol(rebal)

    by_sleeve = {}
    for name, conv in convictions.items():
        arr = conv.to_numpy(dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size and (finite.min() < -1.0 - 1e-9 or finite.max() > 1.0 + 1e-9):
            raise ValueError(
                f"{name} conviction outside [-1, 1] "
                f"(min={finite.min():.6f}, max={finite.max():.6f}) — the [-1, 1] reference "
                f"bins and deadband/saturation fractions would be meaningless. Did a "
                f"vol-scaled WEIGHT series get passed instead of conviction?"
            )
        by_sleeve[name] = compute_eval_distribution(
            arr,
            bar_vol=bar_vol.to_numpy(dtype=np.float64),
            regime_quartiles=EQUAL_QUARTILE_SHARES,
            deadband=DEFAULT_DEADBAND_ABS,
            saturation=DEFAULT_SATURATION_ABS,
            asset_keys=tickers,
        )

    return {
        "schema": SCHEMA,
        "strategy": "tailwind-v1-challenge",
        "eval_distribution_by_sleeve": by_sleeve,
        "provenance": {
            "basis": "audit_tailwind_book._mom_frame (shared with the DSR/PBO audit and "
                     "the forward-path render)",
            "measured": "per-sleeve conviction in [-1,1] at monthly rebalance dates",
            "n_rebalances": int(len(rebal)),
            "first_rebalance": str(rebal[0].date()),
            "last_rebalance": str(rebal[-1].date()),
            "tickers": tickers,
            "deadband_abs": DEFAULT_DEADBAND_ABS,
            "saturation_abs": DEFAULT_SATURATION_ABS,
            "consumer": "NONE on the TAILWIND path as of 2026-08-17 — ActionDriftTracker is "
                        "instantiated only in crypto/live/live_engine.py. One tracker per "
                        "sleeve is the intended wiring.",
            "momentum_is_discrete": (
                "tsmom_signal is the mean of sign() over 3 lookbacks, so momentum conviction "
                "takes only the 4 values {-1, -1/3, +1/3, +1}. Two consequences for whoever "
                "wires the consumer: (a) deadband_frac is STRUCTURALLY 0.000 (min |a| = 1/3 > "
                "the 0.25 deadband), so a deadband alarm on this sleeve can only ever move "
                "upward off zero and is not a useful two-sided signal; (b) only 4 of the 8 "
                "reference bins are populated, so a KL against this baseline is computed "
                "against zero-count bins — check ActionDriftTracker's smoothing before "
                "trusting action_kl_warn/crit on the momentum sleeve. The defensive sleeve is "
                "continuous (rank-based) and has neither problem."
            ),
        },
    }


def main() -> dict:
    baseline = build_baseline()
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    print(f"TAILWIND drift baseline -> {BASELINE_PATH.relative_to(ROOT)}")
    prov = baseline["provenance"]
    print(f"  rebalances: {prov['n_rebalances']}  "
          f"({prov['first_rebalance']} .. {prov['last_rebalance']})")
    for name, block in baseline["eval_distribution_by_sleeve"].items():
        bucketing = block.get("regime_bucketing", "by_vol_quartile present")
        n_assets = len(block.get("by_asset", {}))
        print(f"  {name:<10} n={block['n']}  assets={n_assets}  regime: {bucketing}")
        for tkr in list(block.get("by_asset", {}))[:3]:
            sub = block["by_asset"][tkr]
            print(f"      {tkr:<5} deadband={sub['deadband_frac']:.3f} "
                  f"saturation={sub['saturation_frac']:.3f} entropy={sub['entropy']:.3f}")
    return baseline


if __name__ == "__main__":
    main()
