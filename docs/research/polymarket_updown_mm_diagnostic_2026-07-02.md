# Polymarket 5-min Up/Down Market Making — Maker Diagnostic & Pre-Registered Validation Plan

**Session:** 2026-07-02 (S553-cont-104 thread) · **Status:** DIAGNOSTIC COMPLETE — **VERDICT: CONDITIONAL-GO**
(mechanism validated on-chain; no honest historical backtest is possible without book depth; validation = pre-registered forward paper test)
**Data:** `data/polymarket_updown/` (on-chain trade prints 2026-04-04→2026-05-04, ~79.3M rows; provenance `scripts/data/fetch_polymarket_updown.py`)
**Diagnostic engine:** `python scripts/research/polymarket_updown_mm_diagnostic.py` → `results/polymarket_updown_mm/`
**Pre-registered gates:** `configs/polymarket_updown_mm_paper.gates.yaml` (frozen before any live/paper data exists)
**Forward collector (Stage-B prerequisite):** `scripts/data/collect_polymarket_updown_l2.py`
**Context:** options-as-alpha ledger 0-for-everything (`docs/research/txo_vrp_backward_extension_2026-07-02.md`); Polymarket maker-side was the last unevaluated corner (`.agent/artifacts/polymarket_vilkov_blocker_audit_2026-07-02.md`).

## Plain-English summary

Using one month of true on-chain maker fills plus authoritative CLOB-API resolutions, we dissected
*why* systematic makers win or lose on Polymarket's 5-minute crypto binaries. The population-level
maker edge is thin (+12.0 bps on $532.6M), but a replicated cluster of four wallets earns +135.7 bps
with 87–100% positive days via one identifiable mechanism: **track a live fair value off the
resolution-relevant price stream, keep re-pricing both bids as it moves, tolerate residual inventory
only on the side fair value favors, and stay out of the last seconds**. Their residual inventory sits
on the winning side in 73–87% of markets. The one consistent loser does the naive textbook thing —
symmetric spread capture around a lagging reference: his paired legs lock +$75k of spread, but the
residual inventory the informed flow leaves him loses −$90k, and 57% of his net loss accrues on
fills settled in the final minute. The market's fee curve (taker-only, `0.07·p(1−p)`, makers zero +
20% rebate share) structurally subsidizes exactly the maker behavior the winners exhibit. This is a
real, mechanistically-understood opportunity — but a fill-probability backtest cannot be built from
prints (no historical book depth exists anywhere), so the honest validation is a forward shadow-
quoting test against live L2, with gates frozen now, before any such data exists. Capital remains
blocked regardless of outcome until the operator resolves venue/jurisdiction (Taiwan geoblock;
Polymarket US question) and a Tier-2 audit passes.

---

## 1. Method and data honesty

- **P&L convention:** maker BUY of token *i* at price *p*, size *s* shares → settlement P&L
  `s·1{i won} − usd_amount`; maker SELL mirrored. Recomputed per fill from raw prints; reconciles
  with the first pass (`pnl_detail_top10makers_1month.parquet`) to **$0.00 on all 10 wallets**.
- **Resolutions** from `GET clob.polymarket.com/markets/{condition_id}` → `tokens[].winner`
  (the bulk dataset's `outcome_prices` is untrustworthy — placeholder ~0.5 values on closed markets).
- **Timestamp caveat (affects all timing claims):** on-chain `timestamp` = Polygon settlement time
  of the operator's batched match, which lags the off-chain match by seconds. $114.3M of maker
  notional (21% of total) is stamped *after* window end, almost all within 60s past end. Time-bucket
  claims below are therefore smeared toward lateness; relative ordering across buckets survives this.
- **Maker-side only.** The studied wallets are near-pure makers (7.89M maker fills vs 42.8k taker
  rows). Rebates are NOT included in any figure (they would only add).
- **Fee regime in-sample:** taker fees on short-horizon crypto markets went live 2026-01-07 (quartic
  curve) and were simplified to `feeRate·p(1−p)` around April 2026; the 2026-07-01 schedule update
  postdates the sample. Sample economics ≈ current regime in kind, not necessarily in level — gate
  anchors therefore come from the forward test, not from this sample.
- **Selection bias:** the 10 wallets were selected as the most-active dual-sided makers
  (survivorship-tinted). Population aggregates below are the honest base rate.

## 2. Population structure (ALL makers, 60,040 resolved markets)

| fact | value |
|---|---|
| total maker notional / P&L | $532.6M / **+$637.8k = +12.0 bps** |
| maker BUY side | +17.9 bps on $498.4M |
| maker SELL side | **−73.6 bps** on $34.2M (resting offers get lifted by informed flow) |
| edge by time-to-resolution | +35.0 bps (240–300s) → +13.4 → **−25.3 → −12.6** (60–180s) → +49.3 → +39.4 (15–60s) → **−36.9 (5–15s) → −75.7 (0–5s)**; post-settled +26.7 |
| calibration by price | **U-shaped**: bids at 0.00–0.20 earn +52…+420 bps (longshots underpriced); belly 0.25–0.60 **loses** −20…−211 bps; 0.85–0.999 earns +36…+117 bps on the largest notional ($183M in the 0.95–1.00 bin) |
| concentration | top-10 wallets = 5.7% of maker volume but **42% of all maker P&L** |
| capacity | BTC = **87%** of maker notional (median window $47.2k, p95 $99.1k); ETH median $3.7k; SOL/XRP/DOGE/BNB/HYPE ≤ $1.2k (dust). 2,016 windows/day, ~$17.9M/day maker notional |

Reading: presence alone earns ≈ nothing. Money is made at the price extremes and early/mid-window,
and lost in the belly and the final 15 seconds (stale-quote pick-off as gamma explodes). The venue
is a textbook Budish-style sniping environment — fair value is a deterministic function of a freely
observable external price — with the `p(1−p)` taker fee acting as the anti-sniping tax (the 500ms
taker delay was removed 2026-02-20; latency is the remaining moat).

## 3. The winner/loser split (the brief's step-2 question, answered)

Six behavioral archetypes fell out of the decomposition (per wallet×market: `locked_pnl` = paired
qty × (1 − VWAP₁ − VWAP₂), `resid_pnl` = rest; `alignment` = fraction of markets whose residual
inventory sat on the eventual winner; `drift` = volume-weighted tape move over the 15s after each
fill, + = in the maker's favor):

| archetype | wallets | bps | pos-days | drift | align | locked / resid |
|---|---|---|---|---|---|---|
| **FV-tracker** (the winning design) | `0x424eb20F` +172.9, `0xba679aB7` +192.3, `0x48AC40Fc` +101.4, `0xd9013df8` +87.4 | **+135.7 agg** ($181.4k / $13.37M) | 87–100% | +0.0036…+0.0090 | **0.73–0.87** | −$887k / **+$1,069k** |
| static spreader (breakeven) | `0x2E9B93FA` +14.2, `0x38e59896` +16.1 (busiest: 20.3k mkts) | ≈ +15 | 52–55% | −0.0014…+0.0014 | 0.47–0.52 | +$50k / −$38k |
| **the loser** | `0x76D4D470` | **−26.3** (−$15.0k / $5.69M) | 33% | −0.0007 | **0.31** | **+$75.1k / −$90.0k** |
| longshot picker | `0xb0F85BAa` | +464.9 (small: $0.66M) | 81% | +0.0097 | 0.32 | pair-cost 0.830, mkt win-rate 0.398 |
| death-zone farmer | `0x5C4600b3` | +238.3 | 90% | +0.0031 | 0.55 | 24% of notional at τ<30s/post at **positive** edge (0–5s +609 bps) |
| hybrid sniper | `0x2653e7D5` | +170.3 maker ($258k) **+605 taker** ($75.8k) | 81% | +0.0025 | 0.47 | tiny quotes across 36.9k mkts |

**Why `0xd9013df8` (and its three siblings) win:**
1. **They re-price with the underlying, paying up as fair value moves** — their VWAP pair cost is
   ~1.10 (>1!), so *paired* legs lock a loss; all profit is residual-directional (+$324k resid vs
   −$288k locked for `0xd9013df8`). That is the signature of chasing a moving fair value rather
   than capturing a static spread.
2. **Their residual lands on the winning side 73–87% of markets** and their fills show *positive*
   15s post-fill drift — whoever hits their quotes is, on average, immediately wrong. Both facts
   require a fair-value estimate ahead of the resting book (the resolution-relevant Chainlink
   stream and Binance mid are both freely streamed by Polymarket's own RTDS websocket).
3. **They avoid the death zone:** ≤2.2% of notional inside the last 15s, ≈0% in the last 5s —
   where the global maker edge is −76 bps.
4. Caveat: the four wallets' pair costs (1.102–1.109) and signatures are so uniform they may be
   1–2 operators, not 4 independent confirmations. Treated as ≥2 independent replications.

**Why `0x76D4D470` loses, touching similar markets:**
1. Symmetric ~0.49/0.49 bids around a lagging reference: pair cost 0.986 locks +1.4¢/pair (+$75k)
   but residual inventory is **anti-aligned** (winning side only 31% of markets, −$90k residual).
   His fills' post-drift is negative: he is systematically the counterparty to informed flow.
2. **Stale late quotes:** 19% of his notional is stamped inside the last 60s/post, carrying −$8.5k
   — 57% of his entire net loss. Negative calibration in every price bin below 0.7.
3. Even his taker activity loses (−$1.7k on $40k, −417 bps). This is not bad luck; it is a lagging
   fair value. Two independent windows (8-day, 1-month) agree.

## 4. External corroboration (web literature pass, 2026-07-02)

NotebookLM KB `65fbc6b5-…` was unreachable (400 on bare `notebook_list` — the pre-warned tool/session
failure; documented recovery inapplicable), so per the brief the pass ran on primary web sources:

- **Theory blueprint exists:** arXiv **2510.15205** "Toward Black-Scholes for Prediction Markets"
  — Avellaneda-Stoikov adapted to logit space: reservation `r = x − q·γ·σ_b²(T−t)`, spread
  `≈ γσ_b²(T−t) + (2/k)ln(1+γ/k)`, inventory caps ∝ 1/(p(1−p)), toxicity filters pull quotes.
  Matches the observed winning behavior point-for-point (spread compression near 0/1, quote
  withdrawal as belief-vol explodes ATM near expiry).
- **Mechanics:** resolution = Chainlink **Data Streams** (e.g. `btc/usd`), tie → **Up** (`≥`);
  windows tradeable from creation (~24h ahead) through window end; tick 0.01 (0.001 past
  0.96/0.04), min order 5 shares; complementary matching mints/merges $1 pairs.
  **Polymarket streams the resolution-relevant Chainlink price + Binance spot free & unauthenticated**
  (`wss://ws-live-data.polymarket.com`) — the quoting bot can watch the exact settlement feed.
- **Fees (docs, fetched 2026-07-02):** taker-only `C·feeRate·p(1−p)`, crypto feeRate 0.07
  (≈$1.75/100 shares max; third parties say 0.072/$1.80 — minor discrepancy flagged); makers zero
  fee + **20%** of category taker fees rebated daily pro-rata (Gamma: `rebateRate: 0.2`).
  Polymarket US (CFTC DCM): taker `0.05·p(1−p)`, maker **direct rebate** `0.0125·p(1−p)`.
- **Ecosystem:** Dune/press: 5-min markets ≈ $385M/wk peak, bots 55–62% of volume, BTC ~77%;
  taker latency-arb (98–99% win-rate bots, $270–515k/mo) was killed by the Jan-2026 fee curve —
  "be a maker, not a taker" is the surviving public advice; the 500ms taker delay was removed
  2026-02-20 (Protos), making >200ms cancel/replace = adverse selection. No public repo of a
  profitable rebate-optimized 5-min maker exists (edges unpublished).
- **Execution stack:** `nautilus_trader` Polymarket adapter confirmed (L2 WS deltas, BinaryOption
  instruments, live CLOB execution incl. post-only, up/down instrument discovery). Local venv has
  1.221.0; **v1.222 fixed a maker-fill order-side inversion** — upgrade before any Stage-B use.
  Python `py-clob-client` signing ≈1s/order — too slow for live quoting; fine for shadow/paper.

## 5. What print data can and cannot validate (honesty boundary)

**Validated here:** the opportunity exists at population level; the winning mechanism and the losing
mechanism; where in (price × time-to-resolution) space maker fills earn; capacity concentration.

**NOT validatable from prints:** whether *our* quotes would be filled — requires book depth, queue
position, and counterfactual cannibalization; no historical L2 exists anywhere (official endpoint
dead since 2026-02-20; paid vendor covers 1-min snapshots only). Any backtest claiming otherwise
would be the look-ahead-free-in-name-only construction this project's doctrine forbids. Therefore
the design below is validated **forward**, by shadow quoting against live L2 with a deliberately
conservative fill model — gates frozen in `configs/polymarket_updown_mm_paper.gates.yaml` *before*
any of that data exists.

## 6. Strategy design v0 (structure frozen; exact parameters in the gates file)

**Name:** `pm-updown-mm-v0` — fair-value-tracking two-sided bid maker with death-zone withdrawal.

1. **Universe:** BTC + ETH 5-min windows (87% + 7.7% of maker notional). Others are dust.
2. **Fair value:** `p̂ = Φ(ln(S_t/S_open)/(σ_t·√τ))`, S = RTDS Chainlink stream (the resolution
   feed), σ_t = EWMA realized vol of 1s log-returns (halflife 600s, floored). Tie rule (≥ → Up)
   adds a half-tick Up bias near pins.
3. **Quotes:** bids on BOTH tokens only (the data: maker SELL side = −74 bps; winners are ~100%
   BUY-side) — bid Up at `p̂ − δ`, bid Down at `(1−p̂) − δ`, on the 0.01 grid, A-S-lite
   half-spread `δ = max(1 tick, κ·σ_b·√τ_rem + λ·inventory_skew)` in probability space.
4. **Death zone:** τ < 30s → cancel all interior quotes; only re-quote if `p̂ ∉ [0.05, 0.95]`, at
   ≥3 ticks behind p̂. τ < 5s → no quotes, period. (Global maker edge −76 bps at 0–5s; the
   winners' last-5s exposure ≈ 0.)
5. **Inventory:** per-market net-exposure cap; shed anti-FV inventory via one-tick skew; NEVER
   cross the spread to flatten (taker fee + adverse timing); hold to settlement.
6. **Re-price trigger:** |Δp̂| ≥ 1 tick → cancel/replace; internal decision latency budget 250ms
   (logged; the venue's surviving moat is latency).
7. **Explicitly out of v0 scope:** pre-window quoting, the 0.95–0.999 "death-zone farmer" module,
   SELL-side resting offers, cross-market (15m/1h) hedging — each is a separate pre-registered
   iteration if v0 passes.

## 7. Pre-registered forward validation (Stage B) — gates frozen 2026-07-02

Full machine-readable contract: `configs/polymarket_updown_mm_paper.gates.yaml`. Skeleton:

- **Data prerequisite:** ≥ 28 days of forward L2 + RTDS collection
  (`scripts/data/collect_polymarket_updown_l2.py`, started ASAP — forward-only data, every day
  not collecting is unrecoverable).
- **Protocol:** 7-day burn-in (parameter calibration allowed) → **21-day evaluation, parameters
  frozen**, conservative fill simulation (fills only strictly-through our price, queue-ahead
  decremented by trades only, 1000ms quote-update latency), fees per live schedule, **rebates
  excluded from the primary gate** (report-only).
- **Primary gates (all must pass):** net ≥ +30 bps of simulated maker notional; block-bootstrap
  p(mean daily ≤ 0) < 0.05; positive days ≥ 60%; death-zone control (τ<30s share ≤ 10% of
  notional AND its bps ≥ −30); residual alignment ≥ 0.55; BOTH fill models (conservative + upper
  bound) net-positive with conservative/upper P&L ratio ≥ 0.25.
- **Tripwire (HALT = engine bug, not verdict):** our stream-derived outcome must match the CLOB
  resolution in ≥ 99% of markets.
- **Kill (early NO-GO):** cumulative net < −$400 at any point, or bps < −25 after ≥ 10 eval days.
- **Multiplicity:** `max_design_iterations: 2` — one bounded revision permitted only if the
  mechanism gates (alignment, death-zone) pass while economics fail; otherwise the book closes.
- **Capital prerequisites (unchanged by any pass):** operator resolves venue/jurisdiction
  (Taiwan geoblock — International execution is blocked; Polymarket-US question open), Tier-2
  deep lifecycle audit, and the project's standing capital bar.

## 8. Verdict

**CONDITIONAL-GO.** This is the first corner of the options/derivatives-adjacent search in this
project's history where (a) the population-level counterparty subsidy is directly measurable, (b)
the winning mechanism is identified, replicated across independent wallets, theory-grounded, and
*designable-toward*, and (c) the losing mechanism is equally clear and avoidable by construction.
It is also unbacktestable by any honest means — so the GO is conditional on the pre-registered
forward paper test above, with the expectation (per project doctrine) that it may well fail: the
sample's +135.7 bps winner-cluster edge is a *pre-rebate, latency-privileged, possibly-1-operator*
anchor, and our simulated fills will be strictly more pessimistic. A clean failure closes the book;
a pass earns a Tier-2 audit, not capital.

## References

1. Diagnostic outputs: `results/polymarket_updown_mm/` (`wallet_headline`, `wallet_tau_buckets`,
   `wallet_calibration`, `wallet_market_pairing`, `global_calibration`, `market_summary`,
   `tape_vwap_15s`, `top10_fills_enriched`, `diagnostic_report.json`; log `diagnostic_run.log`)
2. arXiv 2510.15205 — A-S in logit space for prediction markets · Budish-Cramton-Shim QJE 2015 ·
   Aquilina-Budish-O'Neill BIS WP 955 · Milionis et al. arXiv 2208.06046 (LVR)
3. docs.polymarket.com: `/trading/fees`, `/polymarket-learn/trading/maker-rebates-program`,
   `/developers/RTDS/RTDS-crypto-prices`, `/developers/CLOB/websocket/market-channel` ·
   Protos 2026-02-20 (taker-delay removal) · Dune "Anatomy of Polymarket's Fastest Markets"
   (via blockchain.news / crowdfundinsider) · nautilustrader.io Polymarket integration docs
4. Prior internal: `.agent/artifacts/polymarket_vilkov_blocker_audit_2026-07-02.md` ·
   `docs/research/txo_vrp_backward_extension_2026-07-02.md` (discipline template)
