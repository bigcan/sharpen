# TAILWIND-v1 — Prop-firm challenge simulator (Fable review item 1)

**Date:** 2026-07-02 · **Engine:** `finrl_pro_ds/prop/challenge_simulator.py` (+ tests
`tests/prop/test_challenge_simulator.py`, 11 green) · **Runner:**
`scripts/research/run_challenge_simulator.py` · **Rules:** `configs/prop_firm_rules.yaml`
(⚠️ verify vs each firm's current published rules) · **Artifact:**
`results/tailwind_v1/challenge_sim.json` · **Method:** 20 000 moving-block-bootstrap paths
(block 10d, preserves autocorrelation + fat tails), books normalized to 10% base vol, firm rules
enforced per path. Books built on the R1 `mom`/`pf` basis (cached prices, no network).

## The question (NOT Sharpe)

Maximize **P(hit profit target before breaching max-DD OR daily-loss limit)** within the
challenge window — a path-dependent, finite-horizon, asymmetric problem. Sweep {vol multiplier ×
bank-and-derisk} per firm for momentum-only vs momentum+BAB.

## Headline: the simulator OVERTURNS three of the review's priors — because FTMO/Velotrade have NO DEADLINE

| Review prior (item 1) | Simulator result | Why |
|---|---|---|
| Run **hot** (2–2.5× vol) | **Run LOW (1.0×, 10% vol)** maximizes P(pass) | No deadline ⇒ no time pressure; survival dominates. P(pass) falls monotonically with vol. |
| **Drop BAB** in the challenge | **Keep BAB** — it *raises* P(pass) at every vol | max-DD is the binding constraint; BAB cuts DD, so it helps despite the carry. |
| Bank-and-derisk | **Don't bank** (best cell = no banking) | Banking only helps when running hot; at low vol it just adds timeouts. |
| Daily-loss limit sets the vol ceiling | **CONFIRMED** — daily-limit binds above ~3× and craters P(pass) | But at the *optimum* (low vol) the residual binding constraint is max-DD. |

The review **explicitly caveated** this: *"verify each firm's current time rules first — FTMO
dropped its hard 30-day window; 'no deadline' shifts the optimum toward lower vol + patience."* The
simulator confirms that caveat decisively: with no deadline, the "run hot + drop BAB" advice
inverts. It would be correct again under a fixed short window.

## Best policy per firm (momentum+BAB, no banking, 10% vol)

| Firm (rules) | book | **P(pass)** | E[days] | binds | daily / DD / timeout |
|---|---|---|---|---|---|
| FTMO step1 (10% tgt, 10% static DD, 5% daily, no deadline) | momentum_only | 0.690 | 200 | max_dd | 0.03 / 0.17 / 0.10 |
| | **momentum+BAB** | **0.746** | 196 | max_dd | 0.03 / 0.12 / 0.10 |
| FTMO step2 (5% tgt) | momentum+BAB | **0.870** | 110 | max_dd | 0.01 / 0.10 / 0.02 |
| Velotrade (9% tgt, 10% **trailing** DD, 5% daily) | momentum+BAB | **0.720** | 167 | max_dd | 0.03 / 0.23 / 0.02 |

**Two-phase FTMO funded probability ≈ 0.746 × 0.870 ≈ 0.65 per attempt** (mom+BAB, low vol) — a
strong, serviceable number for the prop goal. Velotrade's *trailing* DD is harsher (DD-breach 0.23
vs FTMO's static 0.12), so its P(pass) is lower on the same book.

## The vol frontier (FTMO step1, momentum+BAB, no banking) — speed vs safety

| vol× | eff. vol | P(pass) | daily-breach | DD-breach | timeout | binds |
|---|---|---|---|---|---|---|
| **1.0** | 10% | **0.746** | 0.034 | 0.120 | 0.099 | max_dd |
| 1.5 | 15% | 0.711 | 0.090 | 0.191 | 0.008 | max_dd |
| 2.0 | 20% | 0.633 | 0.165 | 0.201 | 0.000 | max_dd |
| 2.5 | 25% | 0.593 | 0.198 | 0.209 | 0.000 | max_dd |
| 3.0 | 30% | 0.547 | 0.267 | 0.186 | 0.000 | **daily_limit** |
| 4.0 | 40% | 0.504 | 0.349 | 0.147 | 0.000 | **daily_limit** |

- **P(pass) is monotone-decreasing in vol** — the P(pass)-maximizer is the vol *floor*, not the ceiling.
- **The daily-loss limit is confirmed as the vol ceiling**: above ~3× it becomes the binding
  constraint and daily breaches explode (0.03 → 0.35). Running hot is actively bad here.
- **Speed-vs-safety tradeoff:** 1.0× gives the highest P(pass) but E[days] ≈ 196 (~10 months) and a
  10% timeout tail (slow drift to target). 1.5–2.0× resolves fast (timeout → 0) at 0.63–0.71 P(pass).
  If you value faster funded capital (or hedge against a hidden deadline / inactivity rule), **1.5×
  (15% vol) is the sensible compromise** — near-max P(pass), no slow tail.

## Sensitivity (momentum-only best policy)

- **Intraday-MAE proxy** (a daily book held overnight can breach the daily limit intraday):
  1.0 → 1.4 barely moves P(pass) (0.69 → 0.67) *at low vol* because daily breaches are already rare.
  This is a low-vol artifact — at high vol the intraday risk is real and makes hot policies even worse.
- **Crash overlay** (15% of blocks resampled from GFC + COVID): P(pass) 0.69 → 0.64. The book
  survives crash-heavy regimes reasonably (trend + BAB both help in crashes) — a modest, not
  catastrophic, hit.

## Caveats (this is a model, not a guarantee)

1. **Rules must be verified.** Results are extremely sensitive to the no-deadline assumption. If any
   firm imposes a real deadline, the optimum shifts toward higher vol and the review's "run hot"
   advice returns. `configs/prop_firm_rules.yaml` carries best-known rules; Velotrade's DD basis and
   caps are placeholders — re-scrape before any attempt.
2. **The 504-day cap understates low-vol P(pass).** With truly no deadline, most of the 10% timeout
   tail at 1.0× would convert to passes (the residual failure is the 12% DD-breach), so asymptotic
   FTMO-step1 P(pass) at low vol is likely ~0.83, not 0.75.
3. **Daily-loss modeling is close-to-close** with an intraday-MAE knob; a real intraday equity-based
   daily limit on an overnight-held book is harsher at high vol (another reason not to run hot).
4. **In-sample bootstrap.** Paths are resampled from 2006–2026 history; a regime with no analog in
   that window is not represented (the crash overlay partially stresses this).

## Decision

Under the current **no-deadline** FTMO/Velotrade rules, the P(pass)-optimal TAILWIND challenge
configuration is the **opposite** of the review's prior: **run low vol (10–15%), keep BAB, don't
bank-and-derisk.** Expected ~0.75 / 0.87 P(pass) on FTMO step1/step2 (≈0.65 funded per attempt) and
~0.72 on Velotrade. This is a strong, serviceable result — the ~0.73-Sharpe uncorrelated book that
does NOT clear the own-capital DSR bar (R1 + breadth) is nonetheless a **fit-for-purpose
prop-challenge vehicle** at low leverage. The remaining work is execution + risk control (survival
at low vol), not more edge — exactly the small-operator reframe. **Verify each firm's current rules,
then the challenge-mode config = the own-capital `tailwind_v1.yaml` book at 1.0–1.5× vol with BAB
retained** (NOT the review's momentum-only hot mode).
