# ATL × Jev Phase 2: architecture design

> **Created:** 2026-09-23 | **Status:** APPROVED. Build order below; step 1 is being implemented now.
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

### 2. Corpus — `sharpen.jev.corpus` + `scripts/research/jev_build_corpus.py` (step 3)

**Universe to CIKs:**
```
def build_cik_map(tickers: Iterable[str]) -> CikMap
```
- Exact-key sources only: `sp500_constituents.csv` and SEC `company_tickers.json` (cached under `data/`).
- Returns `mapped` and `unmapped`, plus a per-year coverage report.

**Builder:**
```
def build_corpus(names, window, edgar, out_dir, *, max_chars, mode) -> CorpusStats
```
- Resumable: existing accessions are skipped.
- A filing is in scope when `questions_for(items)` is non-empty.
- Masking: `anonymize` with `drop_cover_page = (doc_type == "8-K")`.
- Records `text_sha256`, the hash of exactly the text Jev will see.
- **Firewall:** `mode="screening"` refuses any `accepted_utc` after `screening_window[1]`. `mode="clean"` needs a P4
  authorization (ADR-3).
- **Config keys:** `phase1.filings.max_chars`, `phase1.screening_window`, `phase1.clean_window`, `phase1.universe`.

### 3. Scorer — `sharpen.jev.scoring` + `scripts/research/jev_score_filings.py` (step 4)

```
def score_corpus(corpus, client, phase1, out) -> ScoreStats
```
- **Pre:** `phase1.questionnaire.hash == questionnaire_hash()` and `version == VERSION`. Otherwise `SystemExit`
  (fail closed).
- **Post:** per-filing block scores computed with `filing_score(..., blocks=[b])` for each `b` in `BLOCKS`.
- Every row carries `served_model`. A second served version appearing mid-run **stops the run** (plan C6).

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

### 5. Baselines — `sharpen.jev.baselines` (step 5)

- `lm_net_tone(text, lexicon) -> float`: (positive − negative) / tokens.
- `prior_similarity(text, prev_text) -> float`: cosine of term-frequency vectors against the name's previous
  earnings release.
- Both become daily signals through the same construction machinery (latest in window, earnings releases only).
- **The LM lexicon is downloaded by the operator into `data/lexicons/`.** It is never committed; its terms of use are
  checked at step 5. The loader refuses if the lexicon is absent: P3 cannot pass without its baseline.

### 6. Evaluation — `sharpen.jev.evaluation` + `scripts/research/atl_jev_evaluate.py --leg {p1,p2,p3,p4}` (step 6)

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

#### ADR-5: B8 lockbox seam — deferred to Phase 5
- **Status:** Proposed, deferred.
- **Context:** `forward_evidence` only accepts DSL formulas.
- **Options:**
  1. Add an additive `forward_evidence_from_returns(candidate_net_returns, base_returns, timestamps, proposal_ts,
     cfg)`, built by the funnel's own long-short builder. CRU-1 safe, and the lockbox entry records the spec hash.
  2. A per-name terminal connector plus an orchestrator panel refresh.
- **Decision:** 1, built only if P4 passes. Building it now is effort a likely K3/K4 NO-GO would waste.

#### ADR-6: Storage
- **Status:** Accepted.
- **Decision:** Parquet partitioned by acceptance year, with masked text zstd-compressed. The EDGAR document cache
  (`json.gz`) and the SQLite answer cache are reused. Estimated ≤ 2 GB against ~13 GB free.

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

## Dependency map

| Module | Impact | Changes |
|---|---|---|
| `sharpen/jev/release.py` | new | step 1 |
| `sharpen/jev/corpus.py`, `scoring.py`, `baselines.py`, `evaluation.py` | new | steps 3–6 |
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
| Protocol v2 staging | N/A | No training. Evaluation is staged as legs p1 → p4, one manifest each. |
| Manifest contract | PASS | p4 consumes p1–p3 only when `status == "PASS"`. Every manifest records the git commit, `gates_sha` and questionnaire hash. |
| Pre-flight validation | N/A | No run config. Loaders refuse unknown or missing keys (fail closed). |
| HPO search space | N/A | Nothing is tuned. The design is pre-registered. |
| Fail direction | PASS | Beyond-calendar → NaT; hash or version mismatch → refuse; a second P4 → refuse; missing lexicon → P3 refuses; unmapped CIK → NaN. |
| Information boundary | PASS | Jev sees the masked document only. The scorer's request body is built from `text` + `questionnaire` alone, with a tripwire test against price, return or verdict fields reaching it. |
| Causality argument | PASS | ADR-2 plus the §1/§4 negative tests. |
| Frozen-artifact respect | PASS | Funnel gates and `phase1` are unchanged. The questionnaire hash is bound by an existing test. |

## Migration plan

Everything is additive. Each step is independently committable and tested:
1. `release.py` + negative tests (A10).
2. `jev_filings.py` signal construction + `build_registered_signals` + causality, parity and wiring tests, all on
   synthetic filings.
3. CIK map + corpus builder, with a small live smoke test (a few names, 2019).
4. Scorer with hash, version and firewall gates.
5. Baselines, after the LM lexicon's terms are checked.
6. Evaluation legs, plus DATA-CLEAN screening-panel rebuild (ADR-8).
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
