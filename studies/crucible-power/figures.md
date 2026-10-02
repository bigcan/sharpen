# Ep02 Figures (locked)

> Figures for KISI Ep02; study overview in [README.md](README.md). Checked against public Sharpen `main` @ **9f5befb** (2026-09-30; `CRUCIBLE_VERSION = crucible-v14.0`). **Locked by Keng 2026-09-30. Re-locked 2026-10-01** after a recheck against public `main` @ **50855de** (`crucible-v17.0`; 9f5befb is no longer in the public history): F9, F10 and F16 changed (see the recheck section below); every other figure was found unchanged.

## G1 · the one-line rerun ✅ (2026-09-30)
```
git clone https://github.com/bigcan/sharpen.git && cd sharpen
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/research/crucible_calibration.py --exp mde_sweep --quick
```
Fresh clone, fresh venv, Python 3.11, CPU only, no data, keys, WandB or checkpoints. **14.2 s wall.** Output:

| T (bars) | holdout | MDE ΔSharpe | @ power |
|---|---|---|---|
| 756 | 189 | **3.92** | 0.88 |
| 2782 | 696 | **2.00** | 1.00 |

"MDE falls with T (expected ~1/√T): True". Matches the draft exactly. Label on screen: **Illustrative (synthetic panels)**.

## Rerun figures
| # | Figure | Value | Source |
|---|---|---|---|
| R1 | Smallest detectable edge, ~3 years daily (756 bars) | ΔSharpe **3.92** | G1 command |
| R2 | Same, ~11 years (2782 bars) | ΔSharpe **2.00** | G1 command |
| R3 | A realistic edge | ≈ **0.50** annualised Sharpe | NEGATIVE_RESULTS.md §8 |

Sieve line: 2.00 ÷ 0.50 = 4× → "about four times a realistic one".

## Record figures (project's logged history; not rerunnable from a clone)

| # | Figure | Value | Source (verified 9f5befb) |
|---|---|---|---|
| F1 | Orchestrator lifetime result | **0 PROMISING** | root cause 2026-08-09, §1 table ("PROMISING, lifetime: 0") |
| F2 | Trials in the ledger | **559** rows, **445** distinct formulas | root cause §1 |
| F3 | Substrates | **6**: 5 closed (4 underpowered; `us_equity` closed on both axes), 1 wired but refused by the power guard (`taiwan_smallcap`) | mining log, substrate board |
| F4 | Edge the gate needed for a coin-flip promotion | information ratio **1.2–1.8** | root cause, Verdict |
| F5 | Pass rate of the one validated edge (cross-asset momentum, net Sharpe **0.60**) | **5–7%** | root cause, Verdict; mining log L6 |
| F6 | MDE per substrate (production) | cross-asset **1.68** · Taiwan **1.40** · US equity **1.31** · intraday **1.91** · FX intraday **1.96** · Taiwan small-cap **1.55** | mining log, substrate board |
| F7 | Logged mining campaigns | **C01 2026-07-02 → C13 2026-08-11** | mining log |
| F8 | Free-data connectors | **7**: FRED, CFTC COT, SEC EDGAR, GDELT, Stooq, TWSE, TAIFEX | README, Crucible section (line 154) |
| F10 | Community-ledger export command | **Landed** (256820ab): `python scripts/crucible_export_closed_search.py --ledger results/crucible_orchestrator/real/trial_ledger.db --out community/ledger/ --universe … --data-source … --data-start … --data-end …`, then `python scripts/crucible_community_ledger.py validate`. On screen (12.14): `python scripts/crucible_export_closed_search.py --ledger <your trial_ledger.db> --out community/ledger/`. Verdicts `CLOSED_DECISIVE` / `OPEN_UNDERPOWERED` (`PROMISING_FOUND` only with `--include-promising`). Seeded ledger: 12 entries, all `OPEN_UNDERPOWERED` | CONTRIBUTING.md "Share a search that found nothing"; community/ledger/README.md; sharpen/crucible/community.py |

## Added 2026-09-30 (10-min expansion, Keng's request) — verified on 9f5befb
| # | Figure | Value | Source |
|---|---|---|---|
| F11 | Campaign C12, US equities at a 21-day hold (Aug 11, 2026; WQ101 alpha bank) | **145** pre-registered · **98** holdout-tested · **0** promising | mining log, C12 |
| F12 | Short hold (H ≤ 2): powered, but untradeable | **97 of 100** WQ101 alphas untradeable (**68–135** trades/yr vs a **24**/yr cap) | mining log, C12 |
| F13 | ~1-month hold (H = 21): tradeable, but underpowered | power ceiling **0.626** vs MDE **1.312**; needs **30.1** holdout-years vs a **19.56**-year panel | mining log, C12 |
| F14 | C12 reopens if | ~**2.5×** more years, or a higher-IC low-turnover bank than WQ101 | mining log, C12 |
| F15 | Expected discoveries at the observed hit rate | ≈ **0.25** per campaign | mining log, L6 |
| F16 | LLM proposer test (C05, Taiwan) | did not raise the hit rate **in our test**; on screen with the footnote "Test ran before a sign fix: 10 of 41 AI ideas were scored reversed (v15 audit, 2026-09-30)". "The proposer is not the bottleneck" rests on the power argument (F18: MDE 1.40 vs 0.50), not on this test | mining log, C03–C05; deep audit 2026-09-30, finding 2 |
| F17 | Mining ticks that holdout-tested nothing | **17 of 23** | mining log, L2 |
| F18 | Taiwan campaigns C03–C05 | **225** pre-registered, **160** scored, **0** promising; stopped by the power guard at MDE **1.40** vs **0.50** | mining log, C03–C05 |

Voice roundings: "about a hundred and fifty … close to a hundred" (F11); "about thirty years … less than twenty" (F13); "about two and a half times" (F14); "one discovery every four campaigns" (F15); "most of our mining runs" (F17); "a couple of hundred ideas", "almost three times" = 1.40 ÷ 0.50 = 2.8 (F18).

## Illustrative (not a result)
| # | Figure | Value | Use |
|---|---|---|---|
| I1 | Alpha vs beta worked example | market +10% · beta 1.5 → expected +15% · actual +18% → alpha +3% | alpha-vs-beta scene; labelled **Illustrative** on screen; arithmetic only, no data source |

## Changes vs draft
- F3: split the 5 closed into 4 underpowered + 1 on both axes, per the board.
- Nothing the ledger PR would touch is cited yet; recheck F1–F9 sources once it lands (the PR is docs/scripts only and must not change gate bytes).

## Recheck 2026-10-01 (public main @ 50855de, crucible-v17.0)
- G1 reran on Keng's laptop: 3.92 @ 756, 2.00 @ 2782 (unchanged). It took 4.1 s there, against 14.2 s on the locked fresh Windows clone. The voice keeps "about fifteen seconds" (true for a first run), and the 11.3 clip has no clock chip.
- F10: landed command (see the row). F16: qualified, with an on-screen footnote (see the row).
- F1–F8, F11–F15, F17, F18 unchanged. F11 is also confirmed by the ledger entry `us_equity/2026-08-11-5f1c75cd5b86.json` (145 / 98 / 0).
