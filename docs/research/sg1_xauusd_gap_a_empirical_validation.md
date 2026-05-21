# sg1-xauusd GAP-A Empirical Validation Report

> **Generated:** 2026-05-21T07:22:59.178275+00:00
> **WandB run:** `bigcan-chiwin-technology/FinRL-Pro-DS/live-sg1-xauusd-ctrader-paper-vs-v2-phase2-20260520`
> **Audit reference:** docs/research/sg1_xauusd_sim_to_live_gap_audit.md §4.1

## Verdict

**INSUFFICIENT_N** — Only n=5 executed trades observed; need >=30 for statistical significance.  Per-trade drag = 0.5073 bps; projected cumulative drag at n=30: 0.0015% -> PROJECTED_PASS.  Re-run after additional soak.

## Live Sample Coverage

- Bars observed: 152 (bar 1 … 152)
- Deadband-cleared bars: 5
- Executed trades: 5 (min for significance: 30)
- Broker-induced omissions (deadband cleared, no fill): 0
- Median lot step in fractional units (`L_min*P*S_lot/V`): 0.04507

## Spec Snapshot

- Deadband threshold (D): 0.35
- Lot size (S_lot): 100.0 oz
- Min lot (L_min): 0.01 lots
- Significance gate: n_trades ≥ 30

## Δ Distributions (executed-trade subset)

| Metric | Sim (target − prev_pos) | Live (position − prev_pos) | Quant error (live − target) |
|---|---:|---:|---:|
| n         | 5 | 5 | 5 |
| mean      | -0.031144 | -0.027062 | 0.004082 |
| abs_mean  | 0.525725 | 0.53334 | 0.013381 |
| std       | 0.537619 | 0.547216 | 0.013578 |
| p50 (median)  | -0.374971 | -0.360555 | 0.007574 |
| p90       | 0.62556 | 0.64159 | 0.018766 |
| max       | 0.654895 | 0.676561 | 0.021667 |

## PnL Impact Projection

- Empirical 3-min XAU log-return σ (from fill prices): 0.003791
- Per-trade drag (|quant_error| × σ): 0.5073 bps
- Cumulative drag over observed trades: 0.0254%
- Acceptance threshold (audit §4.1): cumulative drag < 1% AND zero broker-induced omissions

## Per-Trade Quantization Events

| _step | bar | target | actual | quant_error | sim_delta | live_delta |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 11 | -0.4850 | -0.4963 | -0.0113 | -0.6203 | -0.6316 |
| 11 | 12 | 0.1586 | 0.1803 | +0.0217 | +0.6549 | +0.6766 |
| 17 | 18 | -0.1947 | -0.1803 | +0.0144 | -0.3750 | -0.3606 |
| 32 | 33 | -0.5771 | -0.5891 | -0.0120 | -0.3969 | -0.4089 |
| 59 | 60 | -0.0076 | 0.0000 | +0.0076 | +0.5816 | +0.5891 |

---
_Reproduce: `python scripts/audit_gap_a_quantization.py --run bigcan-chiwin-technology/FinRL-Pro-DS/live-sg1-xauusd-ctrader-paper-vs-v2-phase2-20260520`_
