# Crucible power study

Why Crucible's automated mining promoted zero alphas, and how to check it yourself.

**Verdict:** on every data set Crucible mined, the smallest edge an honest test could detect (MDE) was 1.31–1.96 annualised Sharpe, 2.6–3.9× a realistic edge of about 0.50. Zero promotions is the expected outcome at that power. It is evidence about the test's power and the idea bank, not a claim that the market has no alpha.

## Rerun it (about 15 s, laptop CPU, no data or keys)

```bash
git clone https://github.com/bigcan/sharpen && cd sharpen
pip install -e ".[dev]"
python scripts/research/crucible_calibration.py --exp mde_sweep --quick
```

[`scripts/research/crucible_calibration.py`](../../scripts/research/crucible_calibration.py) builds synthetic markets with an edge of known size planted inside, then measures the smallest edge Crucible can detect. Expected output:

| Holdout length | Bars | Smallest detectable edge (ΔSharpe) |
|---|---|---|
| ~3 years daily | 756 | 3.92 |
| ~11 years daily | 2782 | 2.00 |

The MDE falls roughly as 1/√T. These are synthetic panels (illustrative), not market results.

## Results from the mining log (historical simulation)

| Data set | MDE | × realistic 0.50 |
|---|---|---|
| FX intraday | 1.96 | 3.9 |
| Intraday | 1.91 | 3.8 |
| Cross-asset | 1.68 | 3.4 |
| Taiwan small-cap | 1.55 | 3.1 |
| Taiwan | 1.40 | 2.8 |
| US equity | 1.31 | 2.6 |

- Orchestrator lifetime: 0 PROMISING. The gate needed an information ratio of 1.2–1.8 for a coin-flip promotion; the one validated edge (cross-asset momentum, net Sharpe 0.60) passes 5–7% of the time.
- Campaign C12, US equities (Aug 11, 2026): 145 pre-registered, 98 holdout-tested, 0 promising. At a ≤2-day hold the test is powered but 97 of 100 WQ101 alphas are untradeable; at a 21-day hold the power ceiling is 0.626 vs MDE 1.312, needing 30.1 holdout-years vs a 19.56-year panel.
- Taiwan C03–C05: 225 pre-registered, 160 scored, 0 promising; stopped by the power guard at MDE 1.40 vs 0.50.

Full figure table with sources: [figures.md](figures.md).

## Sources in this repo

- [`docs/research/crucible_zero_alpha_root_cause_2026-08-09.md`](../../docs/research/crucible_zero_alpha_root_cause_2026-08-09.md) (verdict)
- [`docs/research/crucible_mining_log.md`](../../docs/research/crucible_mining_log.md) (substrate board, campaigns, lessons L1–L6)
- [`NEGATIVE_RESULTS.md` §8](../../NEGATIVE_RESULTS.md#8-automated-mining-substrates) (data sets closed on power)
- [`community/ledger/`](../../community/ledger/) (share a search that found nothing)

Not rerunnable from a fresh clone: [`scripts/research/crucible_crossmarket_power.py`](../../scripts/research/crucible_crossmarket_power.py) needs `results/crucible_calibration/calibration_mde_sweep.json` (gitignored), which the full calibration sweep writes first.

Video: KISI Ep02, "I Built an AI to Mine for Alpha. Here's Why It Promoted Zero".
