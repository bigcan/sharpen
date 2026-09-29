# ATL × Jev Phase 2: architecture design

> **Created:** 2026-09-23 | **Status:** APPROVED. Build order below; steps 1–6 are built (P4 deferred, ADR-11).
> **Scope:** everything between the frozen pre-registration (`9c7f6df1`) and running legs P1–P4: timestamps, corpus,
> scoring, signals, baselines and evaluation. The B8 lockbox seam is designed here but deferred (ADR-5).
> **Inputs:** `docs/research/atl_jev_prereg.md` (the contract this code must implement), `configs/atl_jev.gates.yaml`
> (`phase1`), `docs/research/atl_jev_phase1_research.md`. Design lives in `docs/`, per the repository boundary.

## Executive summary

Phase 2 builds the pipeline that turns EDGAR 8-Ks into the five pre-registered signals and scores them only
through the frozen funnel. Three parts carry correctness risk, and each gets a negative test:

- **The release row** (LEAK-2): when a filing may first move a position.
- **The construction rules:** latest versus minimum in the hold window, NaN rather than 0.
- **The clean-window firewall:** nothing in screening mode can read a post-2024 price or score.

Everything else is data plumbing.

## Architecture overview

```
EDGAR submissions + full-text ──► corpus builder ─────────────► corpus (parquet, by year)
 (edgar_filings, throttled/retry)   scope: questions_for(items)     accession, cik, ticker, items, accepted_utc,
                                    mask: anonymize (doc-type aware) doc_type, text (zstd), text_sha256
                                            │
                                            ▼
                                  scorer (refuses unless the YAML hash == questionnaire_hash();
                                          stops on a served-version change)
                                            │   JevClient + SQLite answer cache
                                            ▼
                                  filing scores (parquet)
                                    accession, cik, ticker, accepted_utc, surprise, tone, quality,
                                    served_model, questionnaire_hash
                                            │
calendar = the price panel's own rows ──► release dates (sharpen.jev.release; phase1.release_row)
                                            │
                                            ▼
          JevFilingSignal.compute(panel) ──► (T, N) per phase1.construction ──► frozen funnel (P1)
                                                                            ├─► placebo (P2)
          baselines (LM tone, prior-release similarity) on the same rows ───┴─► Fama–MacBeth (P3)
          P4 mode only: clean panel + clean corpus ──► tier1/tier2 on 411 rows ──► p4 manifest (one look)
```

**Type boundaries:**
- `accepted_utc` is `datetime64[ns]`, UTC, naive.
- A release date is `datetime64[D]`, and must be a row of the panel's calendar or NaT.
- Scores are float64 on [−1, 1], NaN when absent.
- Signal outputs are (T, N) float64 aligned to `panel.dates` × `panel.tickers`.

## Interface contracts

### 1. `sharpen.jev.release` (build step 1: the A10 blocker)

**Loader:**
```
def load_release_rule(phase1: Mapping) -> ReleaseRule
```
- Reads `phase1.release_row.{timezone, cutoff}`.
- Raises on a missing key, an unknown zone or a malformed "HH:MM". It never falls back to a default.

**Mapping:**
```
def release_dates(accepted_utc: np.ndarray, calendar: np.ndarray, rule: ReleaseRule) -> np.ndarray
```
- **Pre:** `accepted_utc` is (K,) `datetime64` UTC. `calendar` is sorted, unique trading days, `datetime64[D]`.
- **Post:** each release date is the first calendar day `d` satisfying either
  `d == local_date(τ) and local_time(τ) < cutoff`, or `d > local_date(τ)`.
  NaT if no such day exists in the calendar (fail closed).
- **Invariant:** LEAK-2. The filing is usable at the close of its release row and never earlier. It feeds the funnel
  pairing `close[t+h]/close[t] − 1`.
- **Config keys:** `phase1.release_row.timezone`, `phase1.release_row.cutoff`.
- **Breaking changes:** none (new module).

### 2. Corpus — `sharpen.jev.corpus` + `scripts/research/jev_build_corpus.py` (step 3, built)

**Universe to CIKs:**
```
def universe_tickers(dates, members, start, end) -> set[str]         # members on any day of the window
def build_cik_map(tickers, *, constituents, sec_tickers) -> CikMap
def membership_spells(dates, members, tickers, start, end) -> list[(first day, day after the last)]
```
- Exact-key sources only: `sp500_constituents.csv` and SEC `company_tickers.json` (cached under
  `data/raw/fundamentals/`). `CikMap` holds `mapped` (ticker → CIKs), `unmapped` and `conflicts`; where the sources
  disagree, every CIK is kept. `coverage_by_year` reports the share of members and of member-days mapped.
- The membership file is read fail-closed: a window before its first row or past its last row (2026-06-02) raises.

**Builder:**
```
def corpus_window(phase1, *, mode, p4_authorization=None, narrow=None) -> CorpusWindow
def build_corpus(filers, window, edgar, out_dir, *, max_chars, workers=4) -> CorpusStats
def read_corpus(out_dir, *, expect_stamp=None) -> DataFrame
```
- **Scope:** form `8-K` with a non-empty `questions_for(items)`. A filing is fetched only if accepted during one of
  the filer's membership spells, each opened early by the warm-up (`acceptance_spells`). A non-member row is NaN
  in every signal, so this changes no signal value; it saves about 27% of the pull.
- **Text** per prereg §2 and §9: the press release is `EX-99.1`, else `EX-99.01`, else `EX-99` (ADR-10). Masked with
  `anonymize` (the cover page is stripped from the 8-K body only) and cut to `max_chars`. Each row records
  `text_sha256` (exactly the text Jev will see), `doc_source`, `raw_chars` and `truncated`.
- **Storage and resume (ADR-6):** one shard per CIK, then a sidecar holding the stamp (window, form, document types,
  `max_chars`, questionnaire hash, text-pipeline hash) and the filer's spells. A current shard is skipped, a failed
  filer keeps no shard, and a changed stamp or spell rebuilds. `read_corpus(expect_stamp=...)` refuses a stale or
  mixed corpus.
- **Firewall (ADR-3):** a screening build ends at `screening_window[1]`. A filing outside the window, or of another
  form, raises `FirewallError`, which aborts the whole build rather than skipping. Clean mode requires the P4
  sentinel.
- **Config keys:** `phase1.universe`, `phase1.screening_window`, `phase1.clean_window`,
  `phase1.p4_clean_window.min_filing_accepted`, `phase1.filings.max_chars`, and `phase1.signals[].hold_days` (for
  the warm-up).
- **EDGAR rate limit:** the first full pull, at 4 requests/s over 4 connections, drew HTTP 429 after about 200
  requests, although SEC states a 10/s ceiling. The pull now runs 1 connection at 2/s, and on a 429 the client pauses
  every request for 11 minutes: SEC lifts a block only after 10 minutes below its limit, and a quick retry prolongs it.

### 3. Scorer — `sharpen.jev.scoring` + `scripts/research/jev_score_filings.py` (step 4, built)

```
def check_questionnaire(phase1) -> None                    # SystemExit unless version and hash are this code's
def check_corpus_stamp(stamp, *, mode) -> None             # same mode, questionnaire and text pipeline, or SystemExit
def score_corpus(corpus, client, *, concurrency, chunk=256) -> (DataFrame, ScoreStats)
def to_filing_scores(scores) -> FilingScores               # one row per share class, for the step-2 signals
```
- **Pre:** `phase1.questionnaire` must be `VERSION` with hash `questionnaire_hash()`, and the corpus stamp must
  match the current questionnaire and text pipeline. A full run also needs the corpus's full window and no failed
  filer. Otherwise `SystemExit` (fail closed).
- **Information boundary:** `score_corpus` accepts exactly the corpus columns, so a frame carrying a price, return
  or verdict is refused before any call. Jev's request is the masked text plus the filing's questions; a test
  checks that no accession, ticker, date or CIK reaches it.
- **Post:** per-filing block scores from `filing_score(..., blocks=[b])` for each `b` in `BLOCKS`. An unasked block
  is NaN, and empty text is never sent. Every mapped answer is kept (`ans_E1` … `ans_M1`) for audits.
- Every row carries `served_model`. A second served version appearing mid-run **stops the run** (plan C6).
- **Budget:** `--max-usd` is required. The run refuses to start if a pre-run estimate exceeds it, and stops between
  chunks if the actual bill does.
- **Smoke (2019, 8 names):**
  - 39 filings and 219 questions for $0.0093, all served by `jev-1.13.0`, the Phase 1 model.
  - Answers match known events: Apple's January 2019 guidance cut scores E2 = 0.99; Northrop's and GE's raises
    score E1 = 0.95–0.99; Kraft Heinz's impairment, restatement and other adverse filings score M1 = 0.94–0.99.
  - About 5.7k billed tokens per filing, so the screening corpus should cost about $6.50.
  - 10 tests; 9/9 planted bugs caught.

### 4. Signals — `sharpen/signals/library/jev_filings.py` (step 2; pure logic, testable before any data)

```
class JevFilingSignal:
    def __init__(self, name: str, blocks: tuple[str, ...], hold_days: int, filings: FilingScores,
                 construction: Construction, calendar: np.ndarray, rule: ReleaseRule,
                 min_filing_accepted: np.datetime64 | None = None): ...
    spec: SignalSpec    # family "altdata", expected_sign +1, universe "sp500_pit"
    def compute(self, panel: Panel) -> np.ndarray   # (panel.T, panel.N)
```

`compute` in detail:
- **Which filings count:** the name's filings with release date ≤ `panel.dates[t]` and within the last `hold_days`
  trading days of `calendar`.
- **Blocks:** surprise and tone take the latest, quality the minimum. Same-row filings are averaged first.
- **Composite:** the mean of the available blocks. NaN if none.
- **Alignment:** date-aligned, so a truncated or suffix panel returns exactly the matching rows. That is what the
  Tier-0 truncation tripwire checks, and P4 needs it to slice.

```
def build_registered_signals(phase1, filings, calendar, *, mode) -> list[JevFilingSignal]
```
- Reads `phase1.signals` and `phase1.construction`.
- **Rejects any construction value it does not implement** (`ValueError`). No silent default.

### 5. Baselines — `sharpen.jev.baselines` (step 5, built)

```
def load_lm_lexicon(path, *, sha256) -> Lexicon      # refuses unpinned, missing, or a different file
def lm_net_tone(text, lexicon) -> float              # (positive − negative) / words; NaN with no words
def prior_similarity(corpus) -> Series               # cosine of word counts vs the same CIK's previous release
def baseline_scores(corpus, lexicon) -> DataFrame    # per filing; NaN on event filings
def baseline_signal(name, scores, column, hold_days, construction, calendar, rule) -> JevFilingSignal
```
- Computed on the corpus text, exactly what Jev reads. "Previous release" is the same company's previous
  earnings release (prereg §9).
- The daily signal is built by the Jev signal's own machinery (latest release in the hold window, on its LEAK-2
  release row). A test checks it occupies exactly the Jev signal's rows.
- **The LM lexicon is used on a research-use reading (ADR-9).** Its free license covers academic research only;
  a commercial license is requested if a result is pursued. Therefore:
  - the operator supplies the file in `data/lexicons/` (gitignored); the code never downloads it;
  - the pin lives in `configs/atl_jev_baselines.yaml` (`lm_lexicon.sha256`, `null` until the file arrives). The
    loader refuses if the pin is unset, the file is missing, or its sha256 differs (fail closed). P3 cannot pass
    without its baseline.
- 7 tests; 7/7 planted bugs caught.

### 6. Evaluation — `sharpen.jev.evaluation` + `scripts/research/atl_jev_evaluate.py --leg {p1,p2,p3,screening}` (step 6, built; P4 deferred by ADR-11)

- **Screening panel** (`sharpen.jev.panel`, `scripts/research/atl_jev_build_panel.py`), per ADR-8:
  - cut at `screening_window[1]` before anything runs, then the project cleaner and a recorded OHLC bracketing;
  - `active` = S&P 500 member (the audited as-of join) and priced; ADV excludes the current bar; current GICS
    sectors, with an Unknown bucket;
  - saved with the full trading calendar from 2007. Signals must get that calendar: with the 2012+ rows alone, a
    late-2011 filing would land on the first 2012 row as if it were new.
- **Legs:** each writes `results/atl_jev/phase3/<leg>.json` stamped with the panel hash, the scores manifest,
  both gates hashes and the questionnaire.
  - P2, P3 and `screening` refuse unless `p1.json` is complete on the same stamps.
  - Every leg refuses unless the funnel gates hash to the frozen `519158fa1450`.
  - P3 writes BLOCKED while the lexicon is unpinned.
- 14 evaluation tests and 8 panel tests. P2 and P3 are each shown to pass a planted signal and fail noise.
  7/7 planted bugs are caught in the evaluation code.
- **Tier-1 audit follow-ups (2026-09-24):**
  - The panel's look-ahead guarantees rest on two reused helpers, `us_equity_panel._trailing_adv` and
    `_membership_matrix`, which had no tripwire anywhere in the repo. Both now have one, mutation-verified.
  - P1 records how many filing rows map onto a panel ticker, so a spelling mismatch cannot silently drop filings.
  - P2 must reproduce P1's IC for the same signal to 1e-9, or it stops.

| Leg | Computation | Reads | Writes |
|---|---|---|---|
| **p1** | `evaluate_batch(5 signals, screening panel, frozen gates, Multiplicity.preregistered(n_hypotheses, provenance=prereg))` | `n_hypotheses`, frozen funnel gates | `results/atl_jev/phase3/p1.json` (`status`, cards) |
| **p2** | Permute filing scores across filers whose release dates share an ISO week, `permutations` times. Each placebo is re-scored with `compute_scores` + `cross_sectional_ic` (the funnel's own functions). p = (1 + #{placebo ≥ real}) / (1 + perms). | `phase1.p2_placebo.*` | `p2.json` |
| **p3** | Daily cross-sectional OLS of 5-day forward-return ranks on the ranks of [Jev, LM tone, similarity]. t = mean β_jev / √(var β_jev / `hac_effective_n(β, nw_lags)`). | `phase1.p3_baseline.*` | `p3.json` |
| **p4** | Requires `p1`–`p3` `status == PASS` (K3 not fired). Writes a one-time sentinel, builds the clean panel, corpus and scores (filings from `min_filing_accepted`), then runs `tier1_gross_power` and `tier2_capturability` on the 411 clean rows for the single top-ranked `PROMISING` signal. | `phase1.p4_clean_window.*` | `p4.json` |

## Architecture decision records

#### ADR-1: Signals are `Signal` classes over a filing-score table (not DSL formulas over connector slots)
- **Status:** Accepted.
- **Context:** The funnel scores `Signal.compute(panel)` objects. The DSL-plus-feature-slot path is what the lockbox
  and Crucible mining consume, but the bridges produce broadcast (T,) slots only. The construction rules (expiry,
  in-window minimum) are not DSL-expressible without new operators.
- **Options:**
  1. Signal classes. Small, and follows the "new signal = `Signal.compute`" rule.
  2. `DataConnector` + a new per-name (T, N) bridge + DSL operators. Larger, and touches the orchestrator.
- **Decision:** 1. Project rule deviation, stated plainly: the *filing-text* client is not a `DataConnector`, because
  the protocol models numeric point-in-time series, not documents. The *derived* score series gets a connector at
  B8, only if P4 passes.
- **Consequences:** a parity test keeps the latest-in-window leg honest against the audited `asof_join` (no expiry
  case).

#### ADR-2: Release date = a trading-day stamp from acceptance time; the calendar is the panel's own rows
- **Status:** Accepted.
- **Decision:** Stamp the release as a trading day (00:00), which composes with `asof_join`'s `release ≤ bar`
  convention. The zone and cutoff come from the gates file.
- **Calendar:** the price panel's row dates, with no external exchange-calendar dependency. A date past the calendar
  returns NaT (fail closed).
- **Consequences:** DST is handled by `zoneinfo`. A fixed −5 h offset would put a July 19:45Z filing (15:45 EDT) on
  the same day, which is exactly what the negative test catches.

#### ADR-3: Clean-window firewall
- **Status:** Accepted.
- **Hard limits in screening mode:**
  - the price panel is truncated at `screening_window[1]`, so the last five rows' forward returns are NaN rather than
    peeking;
  - the corpus and scorer refuse filings accepted after it.
- **P4 mode:**
  - requires the p1–p3 manifests to show PASS;
  - writes `results/atl_jev/phase4/OPENED.json` (commit, UTC time);
  - **refuses if that sentinel exists** — one look.
- **Consequences:** clean-window Jev answers do not exist before P4, so even the scores cannot be peeked at.

#### ADR-4: CIK mapping is exact-key only, and coverage is reported, never imputed
- **Status:** Accepted.
- **Context:** Renamed or delisted tickers have no free historical ticker-to-CIK source.
- **Options:**
  1. Exact keys only.
  2. Fuzzy name matching, which risks attaching another company's filings. That is a silent error, worse than a gap.
- **Decision:** 1. Unmapped names get NaN signals and drop out of the cross-section. Coverage by year is reported
  beside the survivorship caveat the panel already carries.
- **Measured (2026-09-23):**
  - 98.9% of the panel's priced S&P 500 member-days in 2012–2024 map to a CIK (97.8% in 2012, 100% from 2022).
    The price panel, not the map, is the binding limit: only 13 priced tickers are unmapped (e.g. ESRX, AET, TWX).
  - One conflict: XOM. ExxonMobil moved to a new holding-company CIK in July 2026, so both CIKs are kept.

#### ADR-5: B8 lockbox seam — deferred to Phase 5
- **Status:** Proposed, deferred.
- **Context:** `forward_evidence` only accepts DSL formulas.
- **Options:**
  1. Add an additive `forward_evidence_from_returns(candidate_net_returns, base_returns, timestamps, proposal_ts,
     cfg)`, built by the funnel's own long-short builder. CRU-1 safe, and the lockbox entry records the spec hash.
  2. A per-name terminal connector plus an orchestrator panel refresh.
- **Decision:** 1, built only if P4 passes. Building it now is effort a likely K3/K4 NO-GO would waste.

#### ADR-6: Storage
- **Status:** Accepted; revised at step 3.
- **Decision:** one zstd parquet shard per CIK under `data/atl_jev/corpus/<mode>/`, instead of year partitions. The
  pull is resumable at company granularity, and a shard is only trusted with its stamp sidecar. The EDGAR document
  cache (`json.gz`) and the SQLite answer cache are reused. Measured: about 17 KB per filing, so about 0.5 GB for the
  screening corpus.

#### ADR-7: Gates stamps hash LF-normalized bytes
- **Status:** Accepted.
- **Context:** This checkout's `core.autocrlf` rewrites working copies to CRLF, so a raw-byte sha256 moves with
  checkouts.
- **Decision:** New code stamps `gates_sha(path)` = sha256 of CRLF→LF-normalized bytes, reproducible from the git
  blob.

#### ADR-8: Price panels
- **Status:** Accepted.
- **Screening panel:** rebuild the PIT-union panel through the project's DATA-CLEAN path
  (`cross_asset_loader._clean_wide`) with a manifest. No in-memory repair.
- **Clean panel (P4 only):** prices through 2026-09-22 from Alpaca SIP daily bars, plus a refresh of PIT membership
  past 2026-06-02 (the file's last date).
- **Built 2026-09-24** (`data/atl_jev/panels/screening.pkl` + manifest). 3,270 rows (2012-01-03 → 2024-12-31),
  about 420 members per day. What the rebuild found in the cached fetch (`_pit_union_2007.pkl`):
  - **It was never cleaned.** The script that fetched it skipped `_clean_wide`. The cleaner, built for intraday
    futures, clamps daily high/low wicks beyond 5%: 7,656 highs and 7,604 lows in the screening rows (27,094
    over 2007–2024). It changed 0 of 2,006,533 closes, so every close-to-close return is untouched. Afterwards no
    OHLC violation remained and the bracketing had nothing to do.
  - **Its metadata said `survivorship_free: True`.** The fetch dropped 270 delisted members it could not price, so
    the flag was wrong; the funnel adds its UPPER BOUND caveat only when it is False. The screening panel sets it
    False.
  - **Its ADV included the current bar.** The screening panel uses the audited trailing ADV, which excludes it.
  - 12 tickers are flagged for stale prints; 5 are members in the window (AMCR, COL, EP, HOT, SW). COL and HOT
    have no CIK, so they never carry a signal.

#### ADR-9: The Loughran–McDonald baseline and its license
- **Status:** Accepted — the operator's decision, 2026-09-23.
- **Context:** the pre-registered P3 baseline includes LM net tone. The license, verified from the raw pages on
  2026-09-23, reads:
  - dictionary page ([sraf.nd.edu](https://sraf.nd.edu/loughranmcdonald-master-dictionary/)): "free for use in
    academic research … For commercial licenses, please contact us at loughranmcdonald@gmail.com";
  - repository-wide ([SRAF](https://sraf.nd.edu/)): all software and data are supplied as-is, without warranty,
    for non-commercial purposes only.

  This workstream's research is aimed at trading for profit, so the free academic grant does not clearly cover it.
- **Options:**
  1. Ask the authors for a commercial license now. Keeps the pre-registration exactly as written.
  2. Use LM now on a research-use reading, and request the license only if a result is worth pursuing. Also keeps
     the pre-registration exactly as written.
  3. Amend P3, with a written amendment in the prereg, to a license-clean tone baseline **at least as demanding**
     as LM. There is no verified candidate:
     - a general-purpose lexicon such as VADER would make P3 *easier* to pass, which is disqualifying;
     - the two most-used FinBERT checkpoints (`ProsusAI/finbert`, `yiyanghkust/finbert-tone`) declare no license
       on Hugging Face (checked 2026-09-23), so they are not clean either.
  4. Keep only the self-implemented Lazy-Prices similarity baseline. Also an amendment, and a weaker P3.
- **Decision:** option 2. It is the operator's call on the license risk; the pre-registration is unchanged.
- **Consequences:**
  - the operator downloads the March 2026 release (`Loughran-McDonald_MasterDictionary_1993-2025.csv`) from the
    official page into `data/lexicons/`. That path is gitignored (`/data/`), so the file is never committed and
    never reaches the public Sharpen repo. The code never downloads it.
  - the version is fixed by name before any Jev score exists: `configs/atl_jev_baselines.yaml` names the March 2026
    release and its file. Choosing a version after Jev scores exist would be a forking path. The sha256 pin, set when
    the operator's file arrives, only verifies that the bytes are that release.
  - **License trigger:** if P1–P4 pass and the operator decides to pursue the result, the commercial license is
    requested before Stage 5 (forward lockbox) begins. Holding it is a precondition for the Tier-2 audit and for
    any capital.

#### ADR-10: The press release is exhibit 99.1 under any of its observed labels
- **Status:** Accepted, step 3. Recorded in the prereg as §9.
- **Context:** an exhibit census of 60 Item 2.02 filings (2012–2024) found 9 releases filed as `EX-99` or
  `EX-99.01`; in the 2019 smoke build it was 22 of 36 (e.g. TXN, NOC, CMI, SO). A literal `EX-99.1` match would have
  sent Jev the one-paragraph 8-K body instead.
- **Decision:** the press release is the first of `EX-99.1`, `EX-99.01`, `EX-99`; the 8-K body is used only when none
  exists. `doc_source` records which label was read, so remaining fallbacks can be audited on the full corpus.

#### ADR-11: The screening decision rule, and P4 built only when needed
- **Status:** Accepted, step 6. The rule is recorded in the prereg as §9, before any score existed.
- **Context:** the prereg names P1–P3 and K3 ("no signal is `PROMISING` … or none clears P3") but not which signals
  P2 and P3 run on, nor how P4 picks among several passers.
- **Decision:**
  - P2 and P3 run on every `PROMISING` signal. A signal passes screening only if it passes P1, P2 and P3; P2 is part
    of the pass bar, so it cannot be skipped.
  - K3 fires when no signal passes. P4 takes the top-ranked passer in the funnel's order.
  - **P4 is built only after a signal passes screening.** It needs a clean-window panel (Alpaca SIP prices), a
    membership refresh past 2026-06-02, and the clean corpus and scores. A likely K3 NO-GO would waste that work,
    the same reasoning as ADR-5. Its contract (sentinel, one look, the funnel's own T1/T2) is unchanged.
- **Consequences:** `atl_jev_evaluate.py` has legs p1, p2, p3 and `screening`. The `screening` leg writes the K3
  verdict and names the P4 candidate.

## Dependency map

| Module | Impact | Changes |
|---|---|---|
| `sharpen/jev/release.py` | new | step 1 |
| `sharpen/jev/corpus.py` (+ `stamps.py`, ADR-7), `scoring.py`, `baselines.py`, `panel.py`, `evaluation.py` | new | steps 3–6 |
| `sharpen/crucible/data/edgar_filings.py` | changed | a shared 11-minute pause on HTTP 429 (step 3) |
| `sharpen/signals/library/jev_filings.py` | new | step 2 |
| `sharpen/signals/*` (funnel), `configs/signal_eval.gates.yaml` | **read only** | none (CRU-1) |
| `sharpen/crucible/lockbox/*` | none now | ADR-5 (Phase 5) |
| `configs/atl_jev.gates.yaml` | **read only** (frozen) | none. Phase 2 adds no threshold. |

## Invariant checklist

| ID | Preserved? | How |
|---|---|---|
| LEAK-2 | Yes | Release row from acceptance time (DST-aware). Negative tests for 15:31, after-close, weekend, holiday and DST. `compute` truncation equality. A filing released at t+1 never touches row t. |
| LEAK-1 | N/A | No normalization is fit. The funnel's neutralization is per-day cross-sectional. |
| DATA-CLEAN | Yes | Panels rebuilt through the project cleaner, with a manifest (ADR-8). |
| CRU-1 | Yes | Frozen funnel gates are only read; `519158fa1450` is unchanged. |
| CRU-2 | N/A | Not `crucible/agentic`. Jev sees filing text only, never returns, scores or verdicts. |
| BUG-03, PF-XCHECK, SHORT-ACCT, MARGIN-CFG | N/A | No RL, no PF, no simulated account. |

## Protocol and gate conformance

| Check | Verdict | Where |
|---|---|---|
| Threshold provenance | PASS | Every threshold is in `configs/atl_jev.gates.yaml` `phase1` (p2/p3/p4 bars, release cutoff) or the frozen funnel file. Phase 2 adds none. |
| Gate consumer | PASS (by test) | Each `phase1` key has a named consumer (§ Interface contracts) and a test that changes the key and asserts the behavior changes. |
| SharpOps staging | N/A | No training. Evaluation is staged as legs p1 → p4, one manifest each. |
| Manifest contract | PASS | p4 consumes p1–p3 only when `status == "PASS"`. Every manifest records the git commit, `gates_sha` and questionnaire hash. |
| Pre-flight validation | N/A | No run config. Loaders refuse unknown or missing keys (fail closed). |
| HPO search space | N/A | Nothing is tuned. The design is pre-registered. |
| Fail direction | PASS | Beyond-calendar → NaT; hash or version mismatch → refuse; a second P4 → refuse; missing or unpinned lexicon → P3 refuses; unmapped CIK → NaN. |
| Information boundary | PASS | Jev sees the masked document only. The scorer's request body is built from `text` + `questionnaire` alone, with a tripwire test against price, return or verdict fields reaching it. |
| Causality argument | PASS | ADR-2 plus the §1/§4 negative tests. |
| Frozen-artifact respect | PASS | Funnel gates and `phase1` are unchanged. The questionnaire hash is bound by an existing test. |

## Migration plan

Everything is additive. Each step is independently committable and tested:
1. `release.py` + negative tests (A10).
2. `jev_filings.py` signal construction + `build_registered_signals` + causality, parity and wiring tests, all on
   synthetic filings.
3. CIK map + corpus builder, with a small live smoke test (a few names, 2019). **Built:** 24 tests, 19/19 planted
   bugs caught; smoke of 8 names in 2019 gave 39 in-scope filings, no failures, no surviving name, ticker or year.
4. Scorer with hash, version and firewall gates. **Built:** 10 tests, 9/9 planted bugs caught; live smoke above.
5. Baselines, with the operator-supplied LM file pinned by sha256 (ADR-9). **Built;** the pin waits for the file.
6. Evaluation legs, plus DATA-CLEAN screening-panel rebuild (ADR-8). **Built:** p1, p2, p3 and the K3 decision, and
   the screening panel; P4 is deferred (ADR-11).
7. **Phase 3 run:** build and score the screening corpus, then p1 → p3. Phase 2 ends here.

**Rollback:** revert any step's commit. Nothing downstream consumes a Phase 2 artifact until step 7.

## Test plan

| Test | File | Verifies |
|---|---|---|
| 15:29 same day, 15:31 next day, after close, Friday → Monday, holiday → next, DST both sides, beyond calendar → NaT | `tests/jev/test_release.py` | LEAK-2 release row (negative) |
| Loader refuses missing zone or cutoff, unknown zone, malformed cutoff; a changed cutoff changes the result | `tests/jev/test_release.py` | fail closed; gate consumed |
| Latest vs minimum per block; same-row mean; NaN not 0; expiry after `hold_days`; composite mean | `tests/jev/test_jev_filings_signal.py` | construction |
| `compute(panel.truncated(t)) == compute(panel)[:t+1]`; suffix slice equals rows; a filing at t+1 leaves row t unchanged | same | causality (negative) |
| Latest leg without expiry == `asof_join` | same | parity with the audited join |
| An unknown construction value raises; changing `phase1.signals` hold changes coverage | same | gate consumers |
| Hash or version mismatch refuses; a served-version change stops; firewall refuses post-2024 filings in screening mode | `tests/jev/test_scoring.py` | fail closed, C6, ADR-3 |
| A second p4 refuses; p4 refuses without p1–p3 PASS | `tests/jev/test_evaluation.py` | one look, manifest contract |
| Placebo p-value on a planted vs null panel; Fama–MacBeth t on a synthetic regression | same | P2 and P3 math (Math skill) |

## Config changes

None. Phase 2 reads the frozen `phase1` section and adds no key and no threshold. Operational parameters are script
arguments, not gates: EDGAR rate, output paths, concurrency (read from the existing `jev` section).

## Estimated effort

| Step | LOC (code + tests) | Risk |
|---|---|---|
| 1 release | 90 + 120 | low (pure) — **causality-critical** |
| 2 signals | 180 + 200 | medium — **correctness-critical** |
| 3 corpus | 220 + 80 | medium (EDGAR scale, CIK coverage) |
| 4 scorer | 120 + 100 | low |
| 5 baselines | 100 + 60 | low (license check) |
| 6 evaluation + panel | 300 + 150 | medium (P2/P3 math → Math skill) |
