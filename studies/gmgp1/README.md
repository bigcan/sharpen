# GMGP1-BTC: a reinforcement-learning agent trades Bitcoin

> **Source:** a study from the **KISI** YouTube channel (Keep It Simple Investing), by Keng. Channel: [youtube.com/channel/UCPyTohwSp1URsZL9EX-NJ2A](https://www.youtube.com/channel/UCPyTohwSp1URsZL9EX-NJ2A) · Code: [github.com/bigcan/sharpen](https://github.com/bigcan/sharpen) · Support: [ko-fi.com/bigcan](https://ko-fi.com/bigcan). Historical simulation only; not investment advice.
> Video: link added when Ep04 is published.

Can a reinforcement-learning agent (SAC) find a tradable edge on Bitcoin 15-minute bars? We trained it with 5 seeds on each of 4 walk-forward test months (Dec 2025 to Mar 2026), charged realistic costs (taker fee 5.5 bps plus 5 bps slippage, one way), and checked the 20 resulting runs.

**Verdict:** no edge after costs, and costs were not the main reason. All 20 runs have a profit factor (PF) below 1.0, between 0.83 and 1.00 (median of the 20: 0.906). Taking the best run in each of the four test months, the median PF is 0.977, still below 1. Pooling the best run from each month gives PF 0.972, and the 95% interval (0.924 to 1.025) excludes the PF of 1.1 we set as the bar. The realised information coefficient (IC) is negative in all 20 runs, and the worst fixed-size trailing drawdown is -33%. Adding the costs back (approximately) lifts only 7 of the 20 runs above PF 1.0, so costs were not the main reason. Neither 53 risk overlays nor a different algorithm (PPO) changed the picture. This is one asset, one agent design and four months. It is evidence about this setup, not a claim that Bitcoin cannot be traded.

## Rerun it (CPU, no private data, no keys)

```bash
git clone https://github.com/bigcan/sharpen && cd sharpen
pip install -e ".[dev]" pyarrow
python studies/gmgp1/fetch_btc_1min.py     # public Bybit 1-minute candles -> data/ (a few minutes)
python studies/gmgp1/rerun.py              # re-derives the numbers below from the saved runs
```

The saved runs in [`saved_runs/`](saved_runs/) are the 20 trained agents' recorded trajectories (not the agents). `rerun.py` step 1 recomputes PF, return and drawdown for every run from those trajectories and stops if any differs from the stored metrics (expected: 0 mismatches). Step 2 replays the same trajectories under 53 risk overlays. Expected output is in [`expected/`](expected/).

**Data note.** The fetched data matches the file used for the original runs except for one 1-minute bar (2026-04-06 00:16 UTC, 1 of 1,225,440); the re-derived numbers are identical either way. Checksums for both are in [`checksums.txt`](checksums.txt). Bybit's redistribution terms were not confirmed, so the candles are fetched, not committed.

## Results

| Test month (fold) | Best seed | Its PF | Buy-and-hold 1x return |
|---|---|---|---|
| 0 | 789 | 0.9946 | -4.65% |
| 1 | 789 | 0.9602 | -7.41% |
| 2 | 1024 | 0.9973 | -19.04% |
| 3 | 123 | 0.9335 | +5.74% |

- 20 runs: PF 0.8343 to 0.9973; median 0.906.
- Best run per fold: median PF 0.9774.
- Gross add-back (approximate) PF below 1.0 in 13 of 20 runs and above 1.0 in 7; the best seeds reach about 1.07 to 1.09 in folds 0 to 2.
- Risk overlays (stop-loss, take-profit, time stop, volatility target, drawdown throttle, de-leveraging): **0 of 53** overlay arms reach a median PF of 1.0 (best 0.9841; no-overlay baseline 0.9057). Some individual runs do (59 of 1,060 run-and-arm cells), but no arm does on its median.
- PPO with GAE, same environment and data, 4 configurations: walk-forward median PF 0.96 to 0.99. Cited from [`docs/research/ppo_ge_gmgp1_btc_screen_2026-06-19.md`](../../docs/research/ppo_ge_gmgp1_btc_screen_2026-06-19.md); it needs GPU training and is **not rerunnable here**.

Full figure table: [figures.md](figures.md).

## Sources in this repo

- [`NEGATIVE_RESULTS.md`](../../NEGATIVE_RESULTS.md) (GMGP1-BTC clean canary, PPO-GAE vs SAC)
- [`docs/research/gmgp1_btc_canary_reinvestigation_2026-06-09.md`](../../docs/research/gmgp1_btc_canary_reinvestigation_2026-06-09.md)
- [`docs/research/gmgp1-btc_deep_lifecycle_audit_2026-06-03.md`](../../docs/research/gmgp1-btc_deep_lifecycle_audit_2026-06-03.md)
- [`scripts/research/reverify_gmgp1_btc_canary.py`](../../scripts/research/reverify_gmgp1_btc_canary.py) and [`risk_overlay_lab.py`](../../scripts/research/risk_overlay_lab.py) (`--substrate gmgp1`)

Not rerunnable from a fresh clone: training the agents (needs GPU time and the private training pipeline) and the PPO-GAE screen.

Video: KISI Ep04, working title "I Trained An AI To Trade Bitcoin. Here's What Happened."
