# Crucible U7 — `uplift_min` calibrated against its own null (2026-07-30)

Closes audit §5 **U7** / finding **RC-9** (`docs/research/crucible_design_implementation_audit_2026-07-29.md`).
Harness: `scripts/research/crucible_uplift_null.py`. Reports:
`results/crucible_uplift_null/uplift_null_{cross_asset,taiwan,cross_asset_keep-tsmom}.json` (raw draws included).

## The ask, and what the measurement actually says

RC-9's charge: `guards.uplift_min = 0.10` "passes 170/170 — never calibrated against its own null;
becomes the sole economic leg after U1", and U7 was made a precondition on the corrected contract going
live. It went live anyway at `crucible-v8.0`, so this measurement was overdue.

**Verdict: the floor is calibrated, and the shipped value survives — but RC-9's premise was wrong in
direction, and the measurement surfaced a new defect (RC-11) that matters more than the number.**

### 1. The "170/170 pass" was a train-split selection artifact, not a loose gate

The 170 rows behind that statistic are ledger rows read off `delta_sr_oos`, which the ledger schema
itself documents as a **train-split** CPCV value ("Despite the `_oos` name (kept for schema
back-compat), this is NOT out-of-sample", `ledger.py`). The search *maximizes* fitness, of which that
delta is a component — so a 100% pass rate on it is what selection looks like, not what an inert
threshold looks like. On the **holdout**, against a real null, the same leg rejects ~97% of null draws
(`uplift_pass` = 0.0265 on cross_asset).

### 2. Measured null distribution of the decision statistic

The guarded quantity is `CorrectedResult.delta_sr` — full-panel annualized ΔSR between the augmented
and base combined books on the embargoed holdout. (The YAML comment calling it "mean ΔSR over CPCV
paths" is stale; the code uses the full-panel value.) Two independent nulls, both scored through the
production `corrected_contract_fitness` on the **real** base books:

* **`noise_dsl`** — real DSL genomes mined on a `_realistic_noise_panel` (GARCH-t fat tails, vol
  clustering, common factor; no planted edge), augmenting the real base book. Realistic turnover/cost/vol,
  zero edge by construction. **The primary null.**
* **`rotate_demean`** — the real candidate stream, holdout mean removed, circularly rotated. Preserves
  vol/serial structure exactly, destroys alignment. Cross-check.
* **`rotate_signflip`** — mean kept, sign randomized. Reported but **excluded** from the recommendation:
  its magnitude is the real seed's own realized |mean|, so on seeds carrying genuine edge the upper tail
  is inflated by that edge rather than by chance.

| substrate | path | n | null ΔSR mean | **q95 (calibrated floor)** | bootstrap CI95 | max |
|---|---|---|---|---|---|---|
| cross_asset (2 sleeves, base SR +0.045) | cross_sectional | 1200 | −0.244 | **+0.080** | [+0.052, +0.118] | +0.473 |
| cross_asset | overlay | 800 | −0.477 | −0.004 | [−0.007, −0.000] | +0.034 |
| cross_asset | rotate_demean/x-sec | 600 | −0.049 | +0.019 | [+0.012, +0.025] | +0.075 |
| taiwan (1 sleeve, base SR +1.364) | cross_sectional | 900 | −0.587 | −0.074 | [−0.115, −0.038] | +0.778 |
| **taiwan** | **overlay** | 600 | **+0.448** | **+0.831** | [+0.806, +0.842] | +0.992 |

The two independent nulls agree on cross_asset (q95 +0.080 / +0.019 — same order, same sign), which is
what makes them usable. They disagree on Taiwan, for the reason RC-11 records below.

### 3. Decision: keep `uplift_min = 0.10`

`0.10` lies **inside** the bootstrap CI of the best-measured calibration cell (cross_asset
cross_sectional: [+0.052, +0.118]) and is strictly conservative in two more. A calibrated floor that
cannot be distinguished from the shipped one is a confirmation, not a change — and editing the value
would re-hash `crucible_corrected_contract.gates.yaml` and void verdict comparability for nothing. The
YAML now carries the measurement provenance so the number is defensible instead of inherited.

**The uplift leg is not the FPR control, and was never going to be.** Under every null measured,
`t_pass` = 0.000 and `lord_pass` ≤ 0.018, and the JOINT corrected-contract null pass rate is
**0/2600** (cross_asset, CP-upper95 0.0012) and **0/2100** (taiwan, CP-upper95 0.0014) — at the shipped
floor *and* at every calibrated alternative. Significance is carried by the JKM Sharpe-difference z and
the binding LORD++ p-gate; `uplift_min`'s job is economic size. RC-9's fear ("swap a zero-power machine
for a poorly-controlled one") is not realized: the FPR is controlled, measurably, without this leg.

### 4. Scale reference — what a real edge looks like here

Un-nulled library seeds on the same holdout: cross_asset ΔSR ∈ [−0.51, +0.21] (2/6 positive);
taiwan ∈ [−1.06, −0.29] (0/6 positive). So on cross_asset the floor of 0.10 sits just under the best
real seed (+0.21) — tight, but not vetoing everything. On Taiwan every real seed *dilutes* a base book
running SR +1.36; nothing there is close to promotable, which is consistent with the 0-PROMISING record.

---

## RC-11 (NEW, HIGH) — the uplift null is substrate-dependent by ~200×, and the Taiwan overlay path is not gate-ready

| substrate | base sleeves | base-book holdout SR | overlay null q95 | null draws clearing `uplift_min=0.10` |
|---|---|---|---|---|
| cross_asset | tsmom + rates_carry | +0.045 | −0.004 | 2.7% |
| cross_asset (isolation) | tsmom only | +0.496 | +0.061 | 0.4% |
| **taiwan** | tsmom only | **+1.364** | **+0.831** | **37.6%** |

On Taiwan, **37.6% of pure-noise overlay candidates clear the uplift leg**, and the null is *centred at
+0.45* — a random tilt on that base book systematically "adds" ΔSR. A single global floor therefore
cannot carry a stated null rate on both substrates: 0.10 is a ~q97 threshold on cross_asset and a ~q60
threshold on Taiwan, and the value that would make Taiwan's overlay leg a 5% test (+0.83) exceeds every
real edge ever measured anywhere in this project, so it is not a usable gate either.

**Mechanism: partially resolved.**
* **FALSIFIED — base-sleeve COUNT.** The obvious explanation (a single-sleeve base book gives an overlay
  copy an outsized share of the recombined book) was tested directly by restricting cross_asset to
  `tsmom` alone: overlay null q95 moved only +0.061, ~13× smaller than Taiwan's. Sleeve count is not the
  driver.
* **NOT the cross-section width.** The overlay seeds read broadcast `(T,)` slots, and
  `_overlay_returns` collapses the cross-section with `nanmean` — for a constant-across-N row that is
  the slot value itself, independent of N. Taiwan's N=10 vs cross_asset's N=18 cannot act here.
* **Consistent with, but not proven by, base-book Sharpe.** The three measured cells order monotonically
  in base SR (+0.045 → +0.496 → +1.364) and in overlay null q95 (−0.004 → +0.061 → +0.831). Two
  candidate routes remain untested: the combiner's performance tilt re-levering a high-Sharpe direction
  when handed a correlated near-copy of it, and the F14 cost decomposition (`m·b_gross − |m|·c_base`)
  interacting with Taiwan's very different gross-exposure profile. **Unresolved — stated as a
  correlation across three points, not a mechanism.**

**Consequence for re-discovery:** the Taiwan **overlay** path must be treated as un-calibrated. Its
uplift leg is not a 5% economic test there, so a PROMISING survivor mined through it would carry a
guard that admits >1-in-3 of pure noise. The cross_sectional path on both substrates, and the overlay
path on cross_asset, are calibrated and conservative.

**Remediation options (not taken here — they are design decisions, not measurements):**
1. Per-substrate calibrated floors in the corrected-contract gates (mechanically easy; only cross_asset
   has a defensible measured value today).
2. Resolve the mechanism first, since if the combiner is re-levering a correlated copy, the *statistic*
   is wrong on high-Sharpe base books and no threshold fixes it — the same shape as the F2 `dsr_aug`
   seal this contract was built to escape.
3. Require the overlay path's null to be measured per substrate before that path may promote — a
   power-guard-style precondition, matching how substrate power is already gated.

Option 2 is the one worth doing first: RC-11 may be a **seal**, not a threshold error.

---

## Reproduce

```bash
python scripts/research/crucible_uplift_null.py --noise-panels 200 --rotations 100
python scripts/research/crucible_uplift_null.py --panel taiwan --noise-panels 150 --rotations 100
python scripts/research/crucible_uplift_null.py --keep-sleeves tsmom --noise-panels 120 --rotations 40
```

Zero funnel/gate bytes are touched by the harness (CRU-1 safe): it calls the shipped
`corrected_contract_fitness` / `_candidate_returns` / `_overlay_returns` / `_split` as they are.
