# TAILWIND-v1 — R1: correct-book DSR + PBO on momentum + BAB

**Date:** 2026-07-01 · **Answers:** Tier-2 audit R1/N1 (`tailwind_v1_deep_lifecycle_audit_2026-07-01.md`,
P7-01/P7-06/P11-01/P11-02 — "the pre-capital gates grade momentum+**rates**, not the momentum+**BAB**
book tailwind trades"). · **Script:** `scripts/research/audit_tailwind_book.py` (fork of
`audit_two_sleeve_book.py` onto sleeve-2 = defensive/BAB). · **Artifact:**
`results/tailwind_v1/audit_tailwind.json`. · **Basis:** the `mom`/`pf` momentum cache
(coherent with `graded_book_sharpes`), cached prices, no network. 5133 daily obs, 2006→2026.

## The binding question

> Does the momentum + BAB book clear **DSR ≥ 0.95 AND PBO ≤ 0.5** at the honest pre-registered
> multiplicity (`tailwind_v1.gates.yaml` `dsr_n_trials = 24`)?

## Answer: **No — it BLOCKS on multiplicity (DSR), but for an encouraging reason, and passes PBO decisively.**

| Gate | Value | Bar | Verdict |
|------|-------|-----|---------|
| **Deflated-Sharpe (honest)** | **0.896** (N=24; bracket 0.903/0.896/0.882 at N=21/24/31) | ≥ 0.95 | **FAIL** |
| Deflated-Sharpe (curated momentum) | **0.974** (N=24; bracket 0.977/0.974/0.970) | ≥ 0.95 | PASS |
| **PBO (CSCV, 18-book grid)** | **0.0009** (logit-mean 2.84, 12 870 combos) | ≤ 0.50 | **PASS** |

**Decision: `BLOCK_multiplicity` — 0 S1, 1 S2 (DSR below floor). Capital stays correctly BLOCKED.**

## What the fork found (7 attacks on the correct book)

- **A — look-ahead:** PASS. BAB causal Sharpe 0.412 vs same-day 0.440, gap 0.028 (<0.10). No leak
  (the `.shift(1)` beta + T+1 execution hold).
- **B — BAB standalone subperiods (descriptive, not gated):** full 0.412, and **positive in all four
  subperiods** (0.386 / 0.533 / 0.408 / 0.288). *Note:* the 18-ETF **within-class** BAB (low-beta vs
  high-beta inside equity, rates, commodity, fx) behaves as a **modest, broadly-positive diversifier**
  here — NOT the pure equity crash-hedge the cont-96 convexity test (β_SPY −0.76, "loses 16 bps/day on
  90% of days") analyzed. The cross-asset construction is a different object; treat "BAB = insurance
  only" as specific to the equity-β version, not this sleeve.
- **C — corr stability:** PASS. corr(mom, BAB) = **−0.043** full; worst-subperiod 0.166 (<0.40). A
  genuinely clean, regime-stable diversifier.
- **D — combined robustness:** PASS. Combined **Sharpe 0.732** (curated), **OOS-2018 0.807**, positive
  in all four subperiods (0.622 / 0.911 / 0.482 / 0.886). Stronger and more robust than the momentum+
  rates book on these attacks.
- **E — cost:** PASS. Combined survives harsh 10 bps at Sharpe 0.647.
- **F — DSR:** **FAIL at 0.896** on the honest book (see below).
- **G — PBO:** PASS at **0.0009** — the momentum-grid selection is *not* overfit; the in-sample-best
  book is essentially always the OOS-best. This is the first time PBO has been computed on the
  deployable book (P11-01), and it is decisively clean.

## Why DSR fails — and why it's close

The combined Sharpe math checks out exactly: two ~uncorrelated equal-risk sleeves give
`(SR_mom + SR_bab) / sqrt(2·(1+ρ))`. With ρ = −0.04:
- **curated** momentum 0.601 + BAB 0.412 → **0.732** → DSR **0.974** (clears).
- **Fable-honest** momentum 0.389 + BAB 0.412 → **0.579** → DSR **0.896** (fails).

**The swing factor is entirely the momentum sleeve's honest Sharpe**, not BAB. The project's binding
convention (mirroring `audit_two_sleeve_book.py`) is the conservative Fable-honest haircut (0.389, from
the 32-ETF untouched-universe test), so the binding DSR is **0.896 < 0.95**. BAB is nearly free on the
downside here (uncorrelated, cost-survivable, positive standalone) — it *raises* the combined Sharpe,
but not enough to lift the honest DSR over 0.95 by itself.

## Comparison to the momentum + rates-carry book

| | mom + rates (prior audit) | mom + BAB (this fork) |
|---|---|---|
| honest combined Sharpe | 0.601 | 0.579 |
| **honest DSR (N)** | **0.918 (N=21)** | **0.896 (N=24)** |
| PBO on deployable book | not computed | **0.0009** |
| combined OOS-2018 | ~0.52 (D-check) | **0.807** |
| sleeve-2 corr to momentum | 0.014 | −0.043 |

mom+BAB's honest DSR is *slightly lower* (BAB adds less standalone return than rates-carry's 0.467, and
N is higher at 24), but it is **more robust** (higher OOS, positive every subperiod) and now carries a
**decisive PBO pass** the rates book never had.

## Implications

1. **R1 is answered: tailwind does not clear DSR≥0.95 at the honest momentum haircut (0.896), but
   passes PBO decisively (0.0009) and clears DSR (0.974) on the curated momentum Sharpe.** The gate that
   binds is entirely the momentum sleeve's honest Sharpe, not the BAB hedge or selection-overfit.
2. **The honest path to clear DSR is the momentum sleeve, not more sleeves** — consistent with Fable
   review item 2 (universe breadth). Widening the frozen momentum signal to ~35–40 free-data
   instruments would raise the honest momentum Sharpe toward the curated value and could push DSR over
   0.95 *without* new selection bias (PBO already ≈ 0). Adding hedges cannot fix a return-DSR.
3. **The tailwind gates should now cite THIS number (0.896 on mom+BAB, N=24), not the 0.918 mom+rates
   figure** (audit N2 / P7-06). The correct-book DSR is computed and artifacted; a promotion check
   should read it and BLOCK on sub-0.95 (audit N2 / P11-02).
4. **Re-examine the "BAB = insurance only" framing for the 18-ETF within-class sleeve** — here it is a
   positive, uncorrelated, regime-stable diversifier, not the equity-β crash-hedge the convexity test
   described. (Does not change R1; changes how BAB should be *described* and possibly sized.)

**Next (not done here):** N2 (make the correct-book DSR structurally blocking + cite it in
`tailwind_v1.gates.yaml`); the breadth-expansion research (Fable item 2) as the honest route to lift the
momentum DSR; re-run this fork after any universe change.
