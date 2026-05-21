# Daily-Loss Gate Alignment Audit Report

> **Generated:** 2026-05-21T08:18:50.320279+00:00
> **Source:** `scripts/audit_daily_loss_gate_alignment.py` (persistence_window = 5 bars)

## Verdict

**GO_DOCUMENT** — Across **4 active live runs**, sim's instant-trip predicate would not have fired on any bar.  The sim-to-live asymmetry is **dormant in practice** — the architectural decoupling is intentional and harmless on the observed fleet. Close S470 item 6 as documented design choice.

## Fleet Summary

| Run | max_daily | n_bars | n_trades | min_daily_loss% | sim_trips | absorbed | halt-boundary |
|---|---:|---:|---:|---:|---:|---:|---:|
| `live-gold-ib-paper-ensemble-v1-20260509` | 5.00% | 390 | 85 | -1.372 | 0 | 0 | 0 |
| `live-sg1-xauusd-ctrader-paper-vs-v2-phase2-20260520` | 4.00% | 171 | 5 | -0.130 | 0 | 0 | 0 |
| `live-gmgp1-xauusd-ctrader-paper-20260424` | 4.00% | 1398 | 249 | -2.698 | 0 | 0 | 0 |
| `live-btc-bybit-testnet-paper-20260502` | 5.00% | 1095 | 134 | -3.620 | 0 | 0 | 0 |

## Sample Sim-Trip Bars (first 10 per run)

---
_Reproduce: see usage at the top of `scripts/audit_daily_loss_gate_alignment.py`._
