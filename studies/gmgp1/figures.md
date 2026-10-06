# Ep04 Figures

> **Source:** a study from the **KISI** YouTube channel (Keep It Simple Investing), by Keng. Channel: [youtube.com/channel/UCPyTohwSp1URsZL9EX-NJ2A](https://www.youtube.com/channel/UCPyTohwSp1URsZL9EX-NJ2A) · Code: [github.com/bigcan/sharpen](https://github.com/bigcan/sharpen) · Support: [ko-fi.com/bigcan](https://ko-fi.com/bigcan). Historical simulation only; not investment advice.
> Video: link added when Ep04 is published.

> Figures for KISI Ep04; study overview in [README.md](README.md). Source: the saved GMGP1-BTC clean-canary runs in `saved_runs/`, re-derived by `rerun.py` (CPU, 2026-10-06). Column "Rerun" = reproduced by `rerun.py` (✅) or cited from a research note only (—). **Draft until Keng locks it.**

| # | Figure | Value | Source | Rerun |
|---|---|---|---|---|
| F1 | Runs | 4 folds x 5 seeds = 20; costs taker 5.5 bps + slippage 5 bps one way | reverify | ✅ |
| F2 | PF range, 20 runs | 0.8343 to 0.9973; median 0.906 | reverify | ✅ |
| F3 | Best seed per fold | 0.9946 / 0.9602 / 0.9973 / 0.9335; median 0.9774 | reverify | ✅ |
| F4 | Pooled best-solo | PF 0.9722; 95% CI 0.9237 to 1.025; P(PF >= 1.1) = 0.000 | reverify | ✅ |
| F5 | Worst fixed trailing drawdown | -33.09% | reverify | ✅ |
| F6 | Realised IC | negative in all 20 runs | reverify | ✅ |
| F7 | Gross add-back PF (approximate) | below 1.0 in 13 of 20; above 1.0 in 7; best seeds about 1.07 to 1.09 in folds 0 to 2 | reverify | ✅ |
| F8 | Buy-and-hold 1x per fold | -4.65% / -7.41% / -19.04% / +5.74% | reverify | ✅ |
| F9 | Overlays | 53 overlay arms: 0 reach median PF >= 1.0; best 0.9841; baseline 0.9057; 59 of 1,060 cells reach PF >= 1 | overlay lab | ✅ |
| F10 | PPO-GAE screen | 4 configs, walk-forward median PF 0.963 to 0.993 (val 1.024 to 1.070, test 0.997 to 1.014); policy near-static hold; gate (median >= 1.10) not met | research note | — (GPU) |
| F11 | Data | Bybit linear BTCUSDT 1-minute; fetched copy differs from saved by 1 of 1,225,440 bars; results identical | checksums.txt | ✅ |

**Wording rules.** Say "no edge after costs, and costs were not the main reason". Say "median PF" or "arm", never "no run reached 1.0" for overlays. The 0.977 is the best run in each of four months, not the median of 20 (0.906). Do not say "even before fees".
