# Ep05 Figures

> **Source:** a study from the **KISI** YouTube channel (Keep It Simple Investing), by Keng. Channel: [youtube.com/channel/UCPyTohwSp1URsZL9EX-NJ2A](https://www.youtube.com/channel/UCPyTohwSp1URsZL9EX-NJ2A) · Code: [github.com/bigcan/sharpen](https://github.com/bigcan/sharpen) · Support: [ko-fi.com/bigcan](https://ko-fi.com/bigcan).
> Video: link added when Ep05 is published.

> Figures for KISI Ep05; study overview in [README.md](README.md). "Check" = asserted by `verify.py` against the saved results (✅), or cited from the final report only (—).

| # | Figure | Value | Source file | Check |
|---|---|---|---|---|
| F1 | Signals tested; promoted | 5; 1 (`jev-surprise-63`) | phase3/p1.json | ✅ |
| F2 | Trials charged (multiplicity) | 8 | phase3/p1.json | ✅ |
| F3 | 5-day IC; funnel t | 0.0090; 3.21 | phase3/p1.json | ✅ |
| F4 | Deflated Sharpe; FDR q | 0.946; 0.025 | phase3/p1.json | ✅ |
| F5 | Sharpe before costs | 0.02 | phase3/p1.json (capturability) | ✅ |
| F6 | Sharpe after standard / harsh costs | −0.28 / −0.73 | phase3/p1.json | ✅ |
| F7 | Turnovers a year | about 8.1 | phase3/p1.json | ✅ |
| F8 | Sharpe needed to break even at standard costs | about 0.30 | phase3/p1.json (cost_wall) | ✅ |
| F9 | Break-even cost per trade | about 0.7 bp one way | final report §5 | — |
| F10 | Resampled train/test paths positive | 47% (7 of 15) | phase3/p1.json (cpcv) | ✅ |
| F11 | P2 placebo | 0 of 200 shuffles reach the real IC; p = 0.005 | phase3/p2.json | ✅ |
| F12 | P3 Newey–West t (bar 2.0) | 3.18 | phase3/p3.json | ✅ |
| F13 | Decision | proceed to P4; stop rule not fired | phase3/screening.json | ✅ |
| F14 | Filings scored; answers; model version | 24,676 (24,654 with text); 140,614; `jev-1.13.0` only | phase3/scores_screening_manifest.json | ✅ |
| F15 | Jev scoring cost | $6.57 (about $7 with Phase 0) | phase3/scores_screening_manifest.json | ✅ |
| F16 | 8-Ks listed; companies | 98,119; 584 | phase2/corpus_screening_manifest.json | ✅ |
| F17 | Masked filings: filer picked from 10 / year picked | 93.5% (chance 10%) / 49.5% (chance 14%) | phase0/phase0_summary.json | ✅ |
| F18 | Regime memory, monthly correlation | 2020 +0.69; 2023 +0.58; 2026 −0.72 | phase0/phase0_summary.json | ✅ |
| F19 | Clean window | from 2025-02-01 | phase0/phase0_summary.json | ✅ |
| F20 | Stricter t estimators | Hansen–Hodrick 2.66; Newey–West auto 2.90; non-overlapping 2.60 | deep audit | — |
| F21 | Effect by era | 2018–2024 IC 0.0045, t 1.1 | final report §6 | — |
| F22 | Rank Sharpe vs dollar Sharpe | 0.76 vs 0.02 | final report §5 | — |
| F23 | P4 power | 20–25% (411 clean days) | final report §5 | — |
| F24 | Deep audit | 15 AI agents; 111 findings confirmed, 3 refuted; verdict not ready | final report §5 | — |
| F25 | Duration; cost | 3 calendar days (2026-09-23 to 09-25); about $7 | final report | — |

**Filings in scope.** The final report gives 24,676 filings in scope (23,214 earnings releases and 1,462 event filings). The saved corpus manifest gives 24,607 (23,156 earnings, 1,451 events). The scored set is 24,676 filings (scores manifest). This page quotes the scored count and "about 25,000", not the earnings/event split.
