# TAILWIND-v1 X1 re-certification — result (2026-09-30)

**Verdict: BLOCK** (exit 1). The pass fails both registered gates by wide margins. The record stays `BLOCK_multiplicity`, and nothing is promoted.

**Provenance.**
- One pass, run as registered. Pre-registration `da21c684`: `configs/tailwind_v1_x1.prereg.yaml` and `docs/research/tailwind_x1_preregistration_2026-09-29.md`.
- The run was at HEAD `da21c684`, with no executor-path code changed or uncommitted.
- **SHA mapping (rebase before push, 2026-09-30).** Both X1 commits were replayed onto origin's tip, which added the other session's T4-10 crypto-env commits `d74fd6d7` and `d965e1e9`.
  - Pre-registration `da21c684` became **`894fa9d2`**; result `e583e1b9` became **`a55cd2f0`**. The X1 files are byte-identical.
  - The artifact's lineage still says `da21c684`, the code it actually ran on.
  - The only executor-path file those commits touch is `sharpen/envs/multi_asset_allocator_env.py`, and only in a docstring, so the result holds on the pushed tree.
- All four data pins and the manifest hash matched, and the offline guard held.
- Artifact: `results/tailwind_v1/x1_recertification_2026-09-30.json`.

## 1. Gates

| Gate | Registered rule | Result | |
|---|---|---|---|
| DSR, combined executor | ≥ 0.95 at every N in {64, 77, 96} | **0.887 / 0.881 / 0.874** | FAIL |
| P(pass), declared kills | Wilson lower ≥ 0.65 at MAE 1.0 and 1.4 | **0.425** (16/26) / **0.348** (16/31) | FAIL |

The DSR would fail even at the legacy N = 24, where it is 0.917 (diagnostic).

## 2. Every registered number

**Series.** Executor lead 1, excess of the 3m T-bill with 25 bp borrow, 2007-04-30 .. 2026-05-29, 4,802 days. Sharpe 0.520, vol 10.27%, return 5.33%/yr, max drawdown −21.9%, worst day −5.66%.

**DSR.** Sharpe values are annualized. SR* is the annualized deflation benchmark at N = 77.

| Book | Sharpe | Pool SR sd | DSR at N 24 · 64 · 77 · 96 | SR* at 77 |
|---|---|---|---|---|
| **Executor, combined (gate)** | 0.520 | 0.103 | 0.917 · **0.887 · 0.881 · 0.874** | 0.250 |
| Momentum engine alone | 0.502 | 0.138 | 0.840 · 0.777 · 0.765 · 0.750 | 0.336 |
| Research basis (diagnostic) | 0.607 | 0.103 | 0.961 · 0.944 · 0.941 · 0.936 | 0.250 |
| Executor, 0 bp borrow | 0.540 | — | 0.932 · 0.906 · 0.901 · 0.894 | — |
| Executor, 50 bp borrow | 0.499 | — | 0.900 · 0.865 · 0.859 · 0.850 | — |

- **n* = 14.9** on the 18-book pool. The estimator σ_cs·e(N) over-states the true null maximum by ×1.12 at N = 24 and ×1.38 at N = 77, relative to that pool's own 18 trials. The ledger N therefore fails closed against the measured pool, which is the intended direction: N also stands in for the 59 ledger trials that are not in the pool.

**P(pass).** Disjoint walk over the window, 756-session horizon, TIMEOUTs censored.

| MAE | Declared kills (gate) | Wilson 95% | Kills, horizon-cap TIMEOUT = fail (M2 diagnostic) | Firm only | Needless share | Daily-breach rate (kills) |
|---|---|---|---|---|---|---|
| 1.0 | 0.615 = 16/26 (2 horizon-cap, 1 data-end censored) | [**0.425**, 0.776] | 0.571 (lower 0.391) | 0.652 = 15/23 (lower 0.449) | 0.000 = 0/15 (upper 0.204) | 0.115 |
| 1.4 | 0.516 = 16/31 (1 + 1 censored) | [**0.348**, 0.680] | 0.500 (lower 0.336) | 0.556 = 15/27 (lower 0.373) | 0.133 = 2/15 (upper 0.379) | 0.323 |

Borrow sensitivity under the kills gives a Wilson lower of 0.425 / 0.348 at 0 bp and 0.424 / 0.348 at 50 bp. Borrow does not move P(pass).

**Executor-basis render: RENDER_FLAGS.**
- The daily-breach rate is over its 0.10 budget at both MAEs (0.115, 0.323).
- The needless share is over its 0.05 cap at MAE 1.4 (0.133).
- Gross exposure: mean 2.22, p95 2.63, max 2.80, which is inside the render's 4.7 ceiling and the env's 3.0.
- 7 distinct drawdowns cross the 8% kill and 4 cross the firm's 10%. Firm-daily breaches occur on 2 days close to close.

**Executor-basis Stage-4: HOLD, underpowered.**
- OOS 2026-06-01 .. 2026-07-31 (43 days, 2 rebalances): PF 0.936, Sharpe −0.41, −0.44%.
- The pre-cutoff subperiod median PF is 1.098, so the HOLD bar is 0.878.
- `power_sufficient: false` (2 rebalances against the 6 required). The bootstrap Sharpe CI is [−3.77, 3.44], and P(SR < 0) = 0.60.
- Compliance is indeterminate for the window, though the day-share leg passes.
- The curve is carried forward past 2026-06-12, and the verdict is not deploy-gating.

## 3. Reading

- **The executor fails on both legs, so BLOCK stands on the audit's own specification.**
  - DSR is 0.063–0.076 short of the gate across the bracket.
  - The P(pass) lower bound is ~0.23–0.30 short. Even the point estimates (0.615, 0.516) are below 0.65.
  - The audit's forecast, DSR ≈ 0.91–0.93 and a P(pass) bound well under 0.65, held; the DSR came in lower. The whole difference is the ledger N: 24 → 77 costs 0.036.
- **The momentum engine fails far harder than the combined book** (0.765 at N = 77). BAB is what lifts the combined DSR, which confirms T3-03.
- **The research basis now fails too, once the N comes from the ledger** (0.941 at N = 77). This is only a diagnostic. It removes the last "passes somewhere" claim: at 25 bp it clears only below roughly N = 45.
- **For the challenge, intraday risk dominates.** MAE 1.4 raises daily breaches from 3 to 10 of the resolved windows. Borrow is immaterial there.
- **What would change the verdict is the book, not the method.** It would take roughly SR ≈ 0.6+ on excess returns for the DSR, and fewer tail days for P(pass). The Tier-2 remedies on that path are X4 (venue), X5 (order semantics) and constant-gross sizing. None is a re-run of X1.

## 4. Deviations and defects

- **Deviations: none.** One pass, registered settings, no re-run. The single integrity warning is the treasury-curve manifest's `WARN` (7 negative 3m prints repaired; the curve ends 2026-06-12). It was pre-declared, and it can only turn a PASS into REVIEW.
- **Lineage defect, found after the pass and fixed.**
  - What happened: `git_dirty_paths` lost the first character of its first entry (`ocs/sharpops.md`), because the status output was stripped before its positional columns were parsed.
  - Why the verdict is unaffected: the dependency checks were correctly empty, since git status showed no dirty path under `sharpen/` or `scripts/research/`.
  - The fix: `dirty_paths()`, with a test. The artifact is kept as written.
- **Out of scope, unchanged:** the gate record in `configs/tailwind_v1.gates.yaml` (operator), N3's `promotion_check`, N7's challenge-soak performance block, and the move of the ledger into the gates file (N12, operator).
