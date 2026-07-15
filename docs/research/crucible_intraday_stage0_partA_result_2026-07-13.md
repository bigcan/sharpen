# RESULT — Crucible Intraday Substrate, Stage-0 Part A (power)

**Date:** 2026-07-13 · **Session:** S553-cont-129 · **Verdict: A1 = FAIL (at tradeable holding) → Stage-0 NO-GO**
**Pre-registration:** `crucible_intraday_stage0_preregistration_2026-07-13.md` · **Scope:** `crucible_intraday_substrate_scoping_2026-07-13.md`

---

## Bottom line

The intraday substrate does **NOT** give the deflated funnel enough power to detect a realistic marginal
edge (MDE ≤ 0.50) at any comfortably-tradeable holding period. Per the pre-registration's execution order
(**A-FAIL → stop before the FinMind pull**), Part B was **not run**. Intraday is a **NO-GO** — a rigorous,
measured falsification, not an unresolved gap.

## What was measured (30 seeds, fine betas, real `combination_fitness` gate)

The funnel's MDE-vs-N_eff, measured at H=1 (autocorrelation enters purely via N_eff = holdout/(2H−1);
the N_eff proxy passed TA-3 at N_eff=500 (ratio 1.06) but **failed** at N_eff=1000 — see caveat 4).
Full-panel operating points via N_eff = 0.25·4.86M/(2H−1):

| operating point (full panel) | N_eff | measured MDE | clears ≤ 0.50 |
|---|---|---|---|
| **anchor** (= daily calibration) | 1,011 | **1.40** ✓ reproduces daily 1.40 | — (validation) |
| 25-min hold (H=300) | 2,028 | **1.42** | ❌ FAIL |
| **5-min hold (H=60)** | 10,210 | **0.86** | ❌ FAIL |
| 1-min hold (H=12) | 52,826 | *unmeasured* (~0.5–0.6 extrap.) | borderline |
| 5-sec hold (H=1, untradeable) | 1,215,000 | ~0.3 extrap. | (only sub-minute clears) |

## The finding: the funnel's MDE flattens — `1/√N` does NOT hold

- The daily calibration only ever measured down to N_eff≈1011 (no daily substrate is longer); the regime
  **above** that was never probed. This is the first measurement of it.
- MDE scaling exponent **decays**: ~`N^-0.57` over N_eff 189→1011 (daily calibration's own sweep) →
  ~`N^-0.21` over 1011→10210 (this run). Far slower than the naive `N^-0.5`.
- Consequence: **intraday's data abundance does not buy proportional power.** Even the full 4.86M-bar
  2019+ panel gives MDE 0.86 at 5-min holding. Only *sub-minute* (untradeable) frequencies approach 0.50.
- This **effectively closes the "higher-frequency data" lever** — one of only two escapes the E1/E2
  calibration named from the power-bound trap (the other, "extend history," was already a dead end). For
  *tradeable* strategies, more/faster data does not rescue the deflated funnel's power.

## Honesty caveats (from the Tier-1 audit, PASS WITH NOTES)

1. The **flattening mechanism is measured, not derived** — which gate leg (DSR_aug / HLZ-t / CPCV
   fragility) drives it is not pinned down. But it is a continuation of the daily calibration's own
   exponent decay on an **anchor-validated** harness (reproduces the known daily MDE 1.40 exactly), so it
   is as trustworthy as the E1/E2 calibration the whole project relies on.
2. The **1-min point is unmeasured** (T=200k×30 seeds ≈ 1 h; operator chose to close out without it). The
   verdict rests on the clear 5-min+ failure **and** 1-min holding being barely tradeable anyway (a
   1-min-turnover strategy faces brutal cost drag even on low-cost futures).
3. The verdict's large margins (0.86, 1.42 vs 0.50) make it robust to the N_eff-proxy slack.
4. **The N_eff-proxy tripwire (TA-3) FAILED on one of its two validation cells** — surfaced by the
   S553-cont-131 independent audit; the original write-up cited only the passing cell. In
   `results/crucible_intraday_power/intraday_power.json` the wide/shaping run recorded
   `TA3_autocorr_via_neff = false` and `A1_verdict = INCONCLUSIVE`: cell [46000, H=12] N_eff=500 passed
   (measured MDE 2.10 vs H1-curve 1.98, ratio 1.06, on-curve), but cell [92000, H=12] N_eff=1000
   **failed** (measured MDE 1.22 vs H1-curve 2.06, ratio **0.59**, off-curve). The decisive 30-seed
   confirm run then measured only H=1 cells (no H>1 re-validation) and returned `A1_verdict = FAIL`.
   **Direction of the failure:** the measured intraday MDE came in *below* what the N_eff = holdout/(2H−1)
   proxy predicts, i.e. the proxy may **overstate** intraday difficulty by up to ~1.7× at N_eff≈1000; if
   that held at the 5-min operating point the true MDE could be ~0.86/1.7 ≈ 0.51 — *at* the 0.50 ceiling
   rather than clearly above it. The Stage-0 NO-GO still stands, but on two *independent* grounds rather
   than the clean margin claimed: (a) even the ~1.7×-corrected 5-min MDE only reaches the ceiling, not
   below it, and (b) the cont-131 audit pinned a data-independent `marginal_t` floor (~0.4) that blocks
   the 0.50 target regardless of frequency. The advertised "clear, clean falsification" overstated the
   rigor; the *conclusion* is unchanged.

## Reproduce

```
python scripts/research/crucible_intraday_power.py            # wide MDE-vs-N_eff curve (8 seeds, shaping)
python scripts/research/crucible_intraday_power_confirm.py    # decisive 30-seed frontier + live anchor
```
Config: `configs/crucible_calibration_intraday.gates.yaml`. Results (gitignored):
`results/crucible_intraday_power/`. Funnel `gates_hash 519158fa1450` untouched.

## Consequence for the roadmap

Crucible-Taiwan discovery is now **power-exhausted on the current deflated funnel** across all three
attempted levers: extend history (dead end, ~130 yr), broaden data (0 PROMISING), and **higher frequency
(this — closed for tradeable holdings)**. The only remaining lever the calibration named is a
**deliberately less-deflated gate** (trade E1 false-positive protection for power) — documented as an
option but **not recommended** (it trades away the funnel's one proven strength and would not change any
of the ~20 NO-GOs). Recommended direction: return to the deployable **TSMOM → paper** path.

**Related:** `project_crucible_calibration_e1e2_s553`, `project_crucible_llm_proposer_via_cli_s553` (cont-129).
