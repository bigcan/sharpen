# Stock-Index-Futures Basis Arbitrage — Stage-0 Falsification Pre-Registration

> **Status:** SPEC (pre-registered 2026-07-12, Session S553-cont-128) — **gates locked BEFORE any result.**
> **Design philosophy:** falsify-before-optimize (R1 / FFD / ETF-pairs pattern). Stage-0 CPU probe (~$0 GPU) before any Crucible grammar change.
> **Operator stance (this book):** trading fees are **deprioritized** (operator decision 2026-07-12). Therefore the signal is judged on **GROSS structure + statistical validity**, not a net-of-cost floor. Net is still computed and reported *for the record* (a caveat, not a gate) — same discipline used to reclassify the ETF spread as a valid-but-execution-gated signal ([[project_etf_pairs_arb_nogo_s553]]).
> **Mechanism:** the TAIEX index-futures **basis** (`F − S`) is anchored by (a) cost-of-carry `(r − q)·τ` and (b) **convergence to zero at expiry**. The tradeable question: does the basis's *deviation from its own recent level* mean-revert enough to predict the future-vs-cash return spread? Distinct from every prior NO-GO (all directional/momentum/carry/vol/pairs).
> **Prior:** unlike the same-index ETF spread (a pure MM edge), the basis has genuine economic anchors, so real gross reversion structure is *plausible*. But two caveats gate any GO to "valid → Stage-1 confirm," not "deploy": (1) the cash leg `S` is not directly tradeable (needs a 0050/basket proxy — deferred), (2) daily close timing (cash 13:30 vs futures session 13:45) can manufacture spurious reversion — the exact artifact the TXO-VRP audit caught ([[project_taiwan_txo_vrp_deltahedge_go_s553]]).

---

## 0. Instruments, data, scope

| Leg | Series | Source | Coverage |
|-----|--------|--------|----------|
| Future `F` | **front-month TX** (TAIEX index future), roll **>3 calendar days** before expiry | `data/taiwan_options_ext/TX_daily.parquet` (per-contract) | 2001-11 → 2026-01 |
| Cash `S` | TAIEX spot index close | stitched: `_ext` (≤2019-02-27) + `taiwan_options` (>2019-02-27) | 2001-11 → 2025-12 |

- **Expiry:** 3rd Wednesday of `contract_month`. **Held front contract** at `t` = smallest-expiry contract with `days_to_expiry > 3` (roll buffer). Verified: merged n≈5,956 daily, ttx median 20d, convergence holds (mean\|log-basis\| 0.0049 @20-35d → 0.0029 @4-10d).
- **NOT in scope:** the tradeable cash proxy (0050/basket tracking — a Stage-1 concern if this passes); intraday 13:30-aligned prices (only `TX_15min` 2019+ exists, no intraday cash); any Crucible grammar change (built only on a clean pass).

---

## 1. Construction (locked)

- **Basis:** `b_t = ln(F_t) − ln(S_t)`; **annualized** `a_t = b_t / τ_t`, `τ_t = days_to_expiry / 365` (normalizes out the deterministic convergence so the z-score measures *deviation*, not the term-structure ramp).
- **Signal:** causal rolling z-score `z_t = (a_t − mean_W(a)) / std_W(a)`, **W ∈ {20, 40, 60}** (frozen set, no OOS tuning), bars ≤ t only.
- **Position on the (F−S) spread:** `pos_t = −clip(z_t, ±2)/2` (short the spread when the basis is rich).
- **Roll-safe return:** `r^F_{t+1} = ln(F^{c}_{t+1}) − ln(F^{c}_t)` using the **same held contract `c`** at both t and t+1 (the >3d buffer guarantees `c` still trades at t+1 — no roll jump). `r^S_{t+1} = ln(S_{t+1}) − ln(S_t)`. Spread return `= r^F_{t+1} − r^S_{t+1}`.
- **PnL:** `pos_t · (r^F_{t+1} − r^S_{t+1})`. Gross. Net (reported only) charges a nominal **1 bp** futures one-way cost on turnover.

---

## 2. Metrics & Gates (FROZEN — gross/validity, per operator stance)

**Primary family** = front-TX/TAIEX × `W ∈ {20,40,60}`. Reported primary statistic = **median-over-W GROSS Sharpe** (kills W cherry-pick).

**Metrics/cell:** gross & net Sharpe (×√252), net PF, turnover, MDD.

**GO requires ALL:**
1. **G1 — structure:** median-W **gross** Sharpe ≥ **0.50**.
2. **G2 — significance:** circular block-bootstrap (block 5d, 10,000 draws) **5th-pct gross Sharpe > 0** (median-W cell).
3. **G3 — OOS:** last **30%** holdout gross Sharpe > 0.
4. **G4 — timing-artifact guard:** signal survives **both** `F=close` **and** `F=settlement_price` (median-W gross Sharpe > 0 under each). If it appears under only one price mark, it is the 13:30/13:45 timing artifact ⇒ treated as artifact, NOT a signal.

Pass all ⇒ **VALID SIGNAL → Stage-1** (intraday 13:30-aligned confirmation + 0050-proxy tradeability). Fail ⇒ **NO-GO**, book closed, no grammar change.

**Tripwires (must PASS or INVALID — reported, never overridden):**
- **TW-1 causality-gap:** `gross_causal − gross_samebar > 0.30` (confirms PnL earns the *next*-bar spread, not the signal-construction bar).
- **TW-2 shuffled-null:** permute the spread-return series (fixed seed) → gross-Sharpe bootstrap p05 brackets 0.
- **TW-3 roll-jump guard:** max \|r^F\| within a held contract < 0.15 (no roll contamination leaked into returns).
- **TW-4 data/convergence:** F,S>0, τ>0, and mean\|b\| in the near-expiry third < mean\|b\| in the far third (the basis genuinely converges).

---

## 3. Predictions

- **H1:** the annualized basis deviation mean-reverts; G1–G4 clear under both price marks; TAIEX basis is a **valid gross signal** → Stage-1 (proxy tradeability + intraday alignment).
- **H0:** no cost-free daily reversion beyond the convergence ramp already normalized out; fails G1/G2, or only survives one price mark (timing artifact). Book closed for a few CPU-minutes.

---

## 7. VERDICT (run 2026-07-12, S553-cont-128) — VALID SIGNAL → Stage-1

`scripts/research/futures_basis_arb_probe.py`, n=5,956 daily (2001-11 → 2025-12), seed 20260712.

**All gates + tripwires PASS on the artifact-robust (lag+1) primary, F=close:**

| Gate | Result |
|------|--------|
| G1 median-W gross Sharpe ≥ 0.50 | **+1.373** ✅ |
| G2 block-bootstrap p05 > 0 | **+1.139** ✅ |
| G3 OOS (last 30%) gross > 0 | **+1.422** ✅ |
| G4 survives close AND settlement marks | +1.373 / +1.383 ✅ |
| net Sharpe @1bp (record only) | +0.99 to +1.21 |
| TW-1 oracle ≫ causal | +14.10 ✅ |
| TW-2 permutation null p | **0.0000** ✅ |
| TW-3 roll-jump max\|rF\| < 0.15 | 0.106 ✅ |
| TW-4 convergence | True ✅ |

**Two bugs the tripwires caught before any verdict was read** (falsification-first working):
1. **Timing artifact confirmed** — the naive same-day (lag0) gross Sharpe is **+4.10**, but delaying execution one day collapses it to **+1.37**; that ~2.73 gap is the cash-13:30-vs-futures-13:45 mismatch inflating the same-day version. Primary was switched to the lag+1 (artifact-robust) form; lag0 retained only as a labelled upper bound.
2. **TW-2 shuffled-null was mathematically void** (permuted the final PnL, but Sharpe is order-invariant) — replaced with a proper permutation test that breaks the position↔return *pairing* (perm_p=0.0000, decisive). *(The ETF probe shared this latent bug; its NO-GO was unaffected.)*

**Interpretation:** real, statistically-decisive, OOS-stable 24-year basis-reversion structure with a genuine economic anchor — a VALID gross signal, NOT a proven deployable edge. **Stage-1 gates harvestability:** (a) intraday 13:30-aligned prices to pin the true magnitude, (b) `0050`/basket proxy tradeability (the cash leg is not directly tradeable; residual risk that part of the edge is cash-index staleness). No Crucible grammar change yet; a `basis`/`pairs` archetype is justified only after Stage-1 confirms harvestability.

---

## 8. STAGE-1 pre-registration (gates frozen 2026-07-12, S553-cont-128)

Stage-0 returned VALID gross signal with two open caveats. Stage-1 falsifies them. **Sub-test 1a (this run): intraday 13:30 alignment.** Sub-test 1b (0050/basket proxy tradeability) is a separate follow-on.

**1a — data:** futures at 13:30 = close of the `TX_15min` 13:15→13:30 bar (continuous ≈ most-active/front); cash TAIEX 13:30 close (`taiwan_options/TAIEX_spot`, 2018-2025). Overlap ≈ 1,584 daily (2019-2025). Traded return stays the Stage-0 roll-safe front-month `spread_ret` (TX_daily per-contract); ONLY the signal's basis timing changes. τ reused from the Stage-0 held contract.

**1a — three signals, each z-reversion of `(lnF − lnS)/τ`, W∈{20,40,60}, on the same roll-safe return:**
- `EOD`  = Stage-0 basis (TX_daily EOD close) — reproduces the Stage-0 subsample.
- `1345` = intraday EOD basis (F@13:45) — should ≈ EOD.
- `1330` = **aligned** basis (F@13:30, synchronized with cash) — artifact removed.

**1a — GATES (frozen):**
- **ART-1 (artifact real):** `EOD lag0 gross − 1330 lag0 gross > 1.0` (same-day inflation is large & driven by the 13:30/13:45 mismatch). Prior evidence: corr(15-min futures move, next-day cash ret)=+0.097.
- **ALIGN-1 (real aligned edge):** `1330` median-W lag0 gross ≥ **0.50** AND block-bootstrap p05 > 0 AND permutation p < 0.05 — a properly-synchronized same-day basis still has real, significant reversion.
- **CONSIST-1 (cross-validation):** `|1330 lag0 − EOD lag+1| ≤ 0.5` Sharpe — the aligned same-day estimate agrees with the Stage-0 artifact-free (lag+1) number, confirming +1.37 was the honest magnitude.

**Verdict:** ART-1 ∧ ALIGN-1 ∧ CONSIST-1 ⇒ **1a CONFIRMED** (timing caveat resolved; true edge ≈ aligned number) → proceed to 1b (proxy tradeability). Else ⇒ the daily edge was timing/staleness ⇒ **downgrade**.

---

## 9. STAGE-1a VERDICT (run 2026-07-12, S553-cont-128) — NOT confirmed → signal DOWNGRADED

`scripts/research/futures_basis_stage1_intraday_probe.py`, aligned overlap n=1,584 daily (2019-2025).

| basis mark | median-W lag0 | lag+1 |
|-----------|---------------|-------|
| EOD (TX_daily) | +4.55 | +1.38 |
| intraday 13:45 | +4.45 | +1.43 |
| **intraday 13:30 (aligned)** | **+3.50** | +1.13 |

- **ART-1 PASS (barely):** aligning the futures to 13:30 removed only ~1.06 Sharpe of same-day edge (4.55→3.50). The 13:30/13:45 mark mismatch is real but is NOT the main inflator.
- **CONSIST-1 FAIL:** the aligned same-day edge (+3.50) does not collapse to the lag+1 (+1.38) — a large same-day reversion persists even with synchronized prices ⇒ prime suspect is **cash-index staleness** (TAIEX index built from constituent last-trades, stale at 13:30), which aligning the futures timestamp cannot fix. (Corroborated: corr(15-min futures move, next-day cash return)=+0.097.)
- **Leg decomposition (decisive):** the aligned PnL `pos·(rF − rS)` splits — **futures leg `pos·rF` (the only liquid, tradeable leg) = +0.26 same-day, −0.07 delayed**; **cash leg `pos·(−rS)` (needs an untradeable index/proxy) = +0.55 / +0.29.** The edge is almost entirely the **stale cash index catching up to the future**, not the future converging.

**Conclusion:** the TAIEX futures-basis reversion is **statistically real but NOT harvestable** — a "real ≠ tradeable" outcome (same class as the gamma-intraday and ETF-spread findings). It holds **independent of fees**: the tradeable futures leg has ~no standalone edge; capturing the reversion requires being long the (untradeable) cash index, and a liquid proxy would not carry the staleness. **Signal downgraded from "VALID → Stage-1" to real-but-unharvestable.** Stage-1b (0050 proxy) is not expected to rescue it (a liquid ETF lacks the staleness) and is not pursued unless a genuinely tradeable, non-stale spot proxy is sourced. No Crucible grammar change. gates_hash untouched.
