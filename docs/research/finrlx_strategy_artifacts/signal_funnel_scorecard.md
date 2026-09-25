# Signal Scorecard — finrlx_signal_component

- primary horizon: **21d**  ·  batch pool: **3**  ·  deflation multiplicity: **3** (preregistered; docs/research/finrlx_strategy_prereg_2026-09-25.md)
- survivorship-free data: **True**  (False ⇒ all results are UPPER BOUNDS)

| # | signal | family | verdict | IC-IR | DSR | Neff | FDR-q | BHY-q | HLZ | cpcvOOS | fricSh | netSh@std | costWall | breadth |
|---|--------|--------|---------|-------|-----|------|-------|-------|-----|---------|--------|-----------|----------|---------|
| 1 | finrlx_null_control | technical | LOGGED | 0.019 | 0.345 | 2.0 | 0.256 | 0.469 | ✗ | 0.01 | 0.22 | 0.05 | 0.17 | 1.00 |
| 2 | finrlx_tsmom_conviction | technical | LOGGED | -0.026 | 0.148 | 2.0 | 0.764 | 1.000 | ✗ | -0.06 | 0.07 | 0.01 | 0.06 | 1.00 |
| 3 | finrlx_tsmom_position | technical | LOGGED | -0.036 | 0.108 | 2.0 | 0.764 | 1.000 | ✗ | -0.33 | -0.10 | -0.16 | 0.07 | 0.00 |

**Caveats (top signal):** regime-fragile: min subperiod IC-IR -0.009 < 0 (inverts in a sampled subperiod); CPCV fragile: p05 OOS Sharpe -0.53<0 (53% of 15 paths +)