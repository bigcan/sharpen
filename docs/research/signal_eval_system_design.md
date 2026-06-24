# Signal Evaluation System: Architecture Design

> **Created:** 2026-06-23 | **Session:** 553-cont-71
> **Status:** PROPOSED (awaiting operator sign-off — see §13 Open Questions)
> **Scope:** New module `finrl_pro_ds/signals/` — a standardized harness for discovering
> equity single-name **cross-sectional predictive signals** (alpha factors), ranked by
> **deflated gross Information Coefficient**. Consolidates the per-probe falsification
> skeleton into one reusable signal-plug interface + ranked Signal Scorecard.
> **Chain:** Researcher (signal candidates) → **Architect (this doc)** → implement → Audit.

---

## 1. Executive Summary

Every NO-GO verdict in this project (gmgp1-btc, carry, value, PEAD, illiquidity, 0DTE) was
produced by the *same* hand-rewritten skeleton: causal features → rank-IC → cost model →
walk-forward → deflated verdict JSON. This design freezes that skeleton into **one harness**
with a clean `Signal` plug interface and a ranked **Signal Scorecard**, then points it at the
operator-chosen profile: **liquid US single-name equity, cross-sectional, ranked by deflated
gross IC-IR** ("find signals with high predictive power, not a strategy").

The system's differentiator over a generic Alphalens is this project's *falsification
discipline*, ported intact: a per-signal **causality truncation-tripwire** (Tier 0), a
**cost-wall** that separates gross predictive power from net capturability (Tier 2),
**multiple-testing deflation** by batch size (Tier 4, DSR/PSR/FDR — the antidote to ranking
hundreds of TradingView indicators), and an equity-specific **survivorship-bias gate**
(LEAK-EQ-1). It is **purely additive** — a new package importing existing primitives; zero
breaking changes.

---

## 2. Context

- **Pivot:** From "build strategies" to "find predictive signals, eval, rank." Signal types
  in scope: technical indicators (v1), alternative data + LOB (extension points). Candidate
  soil: TradingView custom Pine indicators (reimplemented as causal Python) + Kakushadze 2015
  "101 Formulaic Alphas" as a codeable seed/benchmark batch.
- **Operator-locked this session:** universe = liquid US single-name equity X-section;
  primary mode = cross-sectional rank; top-line rank key = **deflated gross IC-IR**.
- **Reuse thesis (recon complete):** ~90% of the evaluator already exists, scattered across
  `scripts/research/*_falsification.py` and `finrl_pro_ds/crypto/eval/statistics.py`. This
  doc cites the exact reusable functions; the build is mostly *consolidation*, not new math.
- **Honesty precedents to honor:** the cont-58 audit flagged "0.601 Sharpe with no DSR on a
  multiple-comparison winner" → DSR is mandatory here. The VRP sleeve "needs paid-data
  re-validation" before capital → survivorship-free re-validation is the GO-gate here.

---

## 3. Architecture Overview

```
                         ┌─────────────────────────────────────────────────────┐
   data sources          │                  SIGNAL EVAL HARNESS                 │
   (yfinance v1,         │                                                      │
    Sharadar GO-gate) ──▶│  PanelLoader ──▶ Panel(T,N) ──┐                       │
                         │                               │                       │
   Signal registry      │   SignalSpec(hash) ──▶ Signal.compute(panel)─▶ scores │
   (101-alphas,         ─│─────────────────────────────────────────────(T,N)──┐ │
    TV technicals)       │                                                    │ │
                         │   ┌──────────────── 6-TIER FUNNEL ────────────────┐ │ │
                         │   │ T0 Hygiene  (causality tripwire, OHLC, cover) │◀┘ │
                         │   │ T1 Gross    (multi-h rank-IC, IC-IR, decay)   │   │
                         │   │ T2 Capture  (net Sharpe/PF, cost-wall)        │   │
                         │   │ T3 Robust   (walk-forward OOS, subperiod)     │   │
                         │   │ T4 Deflate  (DSR/PSR/MinTRL/FDR, n_trials=B)  │◀──┤ batch-level
                         │   │ T5 Orthog   (corr + marginal-IC vs FF book)   │   │
                         │   └────────────────────┬──────────────────────────┘   │
                         │                        ▼                              │
                         │      SignalScorecard ──▶ rank (lexicographic) ──▶ md/json
                         └─────────────────────────────────────────────────────┘
```

**The fundamental object is the daily cross-sectional IC series** `ic[t] = Spearman(signal[t,:],
fwd_ret[t,:])` across names at day `t`. Everything in T1/T4 derives from it:
`ic_mean`, `ic_std`, `ic_ir = ic_mean/ic_std`, `ic_tstat = ic_ir·√n`, block-bootstrap CI, and
— crucially — **DSR treats the IC series as returns**, so `deflated_sharpe_ratio` deflates
`ic_ir` by `n_trials = batch size` with no new code (ADR-1).

---

## 4. Interface Contracts

### 4.1 `finrl_pro_ds/signals/spec.py` — pre-registration

```python
@dataclass(frozen=True, slots=True)
class SignalSpec:
    name: str                       # unique registry key, kebab-case
    hypothesis: str                 # falsifiable claim, one line
    family: str                     # "technical" | "101alpha" | "altdata" | "lob"
    expected_sign: int              # +1 long-high, -1 long-low, 0 two-sided (abs-IC)
    horizons: tuple[int, ...] = (1, 5, 10, 21, 63)          # forward trading days
    neutralization: tuple[str, ...] = ("winsor", "zscore", "sector")
    universe: str = "sp500_current"
    cost_profile: str = "equity_standard"
    sample_window: tuple[str | None, str | None] = (None, None)

    def content_hash(self) -> str:  # sha256 of all fields, 12-hex; written BEFORE run
```
- **Postcondition:** `content_hash()` is deterministic and recorded in the scorecard +
  committed before results exist (anti-p-hacking; mirrors spec commits like `6efc6a17`).
- **Invariant:** `expected_sign == 0` ⇒ ranking uses **|IC|** and a two-sided p-value.

### 4.2 `finrl_pro_ds/signals/protocol.py` — the plug

```python
class Signal(Protocol):
    spec: SignalSpec
    def compute(self, panel: "Panel") -> np.ndarray:
        """Return (T, N) float64 scores. CAUSAL CONTRACT: row t may use only
        panel data with date <= panel.dates[t]. NaN where score undefined/inactive."""
```
- **Design decision (ADR-2):** signals are **whole-panel vectorized** functions returning
  `(T, N)` — NOT per-bar `compute(panel, t)` — to honor the coding standard "no per-sample
  Python loops in hot paths." Causality is *not* assumed from the signature; it is **verified**
  by the Tier-0 truncation tripwire. A leaky signal compiles and runs but is **rejected at the
  gate**.
- **Precondition:** `panel` validated (OHLC sanity, coverage) before `compute`.
- **Postcondition:** output shape `== (panel.T, panel.N)`; dtype float64; `±inf` forbidden.

### 4.3 `finrl_pro_ds/signals/registry.py`

```python
_REGISTRY: dict[str, type[Signal]] = {}
def register(spec: SignalSpec):                 # decorator
    def _wrap(cls): _REGISTRY[spec.name] = cls; cls.spec = spec; return cls
    return _wrap
def get_registry(family: str | None = None) -> dict[str, type[Signal]]: ...
```

### 4.4 `Panel` (in `signals/features.py`) — aligned X-section data

```python
@dataclass(frozen=True, slots=True)
class Panel:
    dates: np.ndarray          # (T,) datetime64[ns], sorted unique
    tickers: tuple[str, ...]   # (N,)
    open: np.ndarray; high: np.ndarray; low: np.ndarray
    close: np.ndarray; volume: np.ndarray        # each (T, N) float64, NaN=inactive
    active: np.ndarray         # (T, N) bool — point-in-time membership/tradeable
    adv_usd: np.ndarray        # (T, N) trailing dollar ADV (cost & liquidity screen)
    sector_id: np.ndarray      # (N,) int  (v1 current GICS; PIT at GO-gate)
    meta: dict                 # {survivorship_free: bool, source, universe_def, ...}
    @property
    def T(self) -> int: ...
    @property
    def N(self) -> int: ...
    def truncated(self, t: int) -> "Panel":      # rows [0..t] inclusive — for tripwire
    def forward_returns(self, h: int) -> np.ndarray:   # (T,N), close[t+h]/close[t]-1, gated on active
```
- Generalizes the crypto `(T,N)` `bclose`/`active` pattern that `_xs_ic` already consumes
  (`cmgp1_x2_leak_ic_probe_v2.py:104-118`).

### 4.5 `finrl_pro_ds/signals/eval_harness.py` — the tiers

```python
def tier0_hygiene(sig: Signal, panel: Panel, gates: Gates) -> HygieneResult
    # causality truncation-tripwire + OHLC sanity + coverage. FAIL ⇒ excluded.

def tier1_gross_power(sig: Signal, panel: Panel, horizons) -> GrossPower
    # per-horizon daily IC series; ic_mean/ic_std/ic_ir/ic_tstat; block-bootstrap CI;
    # IC-decay half-life; decile-spread monotonicity; breadth (sign-stability across sectors)
    # REUSES: _xs_ic logic, block_bootstrap_ic, neutralization

def tier2_capturability(sig: Signal, panel: Panel, cost_models) -> Capturability
    # signal -> dollar-neutral vol-scaled L/S weights -> net Sharpe/PF/turnover/maxDD
    # per cost model; cost_wall = frictionless_sharpe - net_sharpe
    # REUSES: vol_scaled_weights pattern, COST_MODELS, fable_oracle.backtest_weights, max_dd

def tier3_robustness(sig: Signal, panel: Panel, gates) -> Robustness
    # walk-forward OOS IC (purged), recent-OOS IC, subperiod-min IC, param sensitivity

def tier4_deflation(batch: list[GrossPower], n_trials: int) -> list[Deflation]
    # BATCH-LEVEL. DSR on each signal's daily-IC series with n_trials=len(batch);
    # PSR, MinTRL; BH-FDR across batch one-sided IC p-values
    # REUSES: deflated_sharpe_ratio, probabilistic_sharpe_ratio, min_track_record_length,
    #         one_sided_p, bh_fdr

def tier5_orthogonality(sig_returns: np.ndarray, book: FactorBook) -> Orthogonality
    # corr(signal L/S returns, each book factor) + marginal IC-IR after regressing out book

def evaluate_signal(sig, panel, gates) -> SignalScorecard          # T0..T3,T5 (single)
def evaluate_batch(sigs, panel, gates, batch_name) -> RankedScorecard   # adds T4, ranks, emits
```

**Tier-1 IC core (exact reuse of `_xs_ic`, generalized):**
```python
# per day t: rank-correlate signal vs forward return across active names
ic[t] = spearman(signal[t, active], fwd_ret[t, active])    # need >=4 names, >=3 unique
ic_mean = mean(ic); ic_std = std(ic)
ic_ir   = ic_mean / ic_std                 # per-period information ratio (Grinold)
ic_tstat = ic_ir * sqrt(n_days)            # == the "ic_ir" of _xs_ic (significance)
```

**Tier-4 deflation wiring (exact reuse of `deflated_sharpe_ratio`):**
```python
# the daily IC series IS a return series; its per-period Sharpe == ic_ir
dsr = deflated_sharpe_ratio(
    observed_sr = ic_ir,                          # per-period, non-annualized
    trial_sharpes = [c.ic_ir for c in batch],     # every candidate's IC-IR (the search)
    n_obs = n_days, skew = skew(ic), excess_kurt = exkurt(ic),
    n_trials = len(batch),                         # honest multiple-testing count
    periods_per_year = 252,
)["dsr"]                                            # = P(true IC-IR > 0 after deflation)
```

### 4.6 `finrl_pro_ds/signals/scorecard.py`

```python
@dataclass
class SignalScorecard:
    name: str; family: str; spec_hash: str
    hygiene: HygieneResult                  # pass/fail + reasons
    gross: dict[int, GrossPower]            # by horizon: ic_mean, ic_ir, ic_tstat, ci, decay
    gross_neutralized: dict[int, GrossPower]   # raw IC vs sector/size-neutralized IC
    capturability: dict[str, Capturability]    # by cost model
    robustness: Robustness
    deflation: Deflation                    # dsr, psr, mintrl_years, fdr_q, n_trials
    orthogonality: Orthogonality            # corr_book, marginal_ic_ir
    rank_key: float                         # PRIMARY = deflation.dsr (on best horizon IC)
    verdict: str                            # GATE_FAIL | LOGGED | PROMISING
    caveats: list[str]                      # e.g. "survivorship-biased (yfinance) — UPPER BOUND"

def rank(cards: list[SignalScorecard]) -> list[SignalScorecard]:
    # 1. drop GATE_FAIL (hygiene)
    # 2. sort by (deflation.dsr DESC, primary IC-IR DESC)         [primary = gross power]
    # 3. tie-break: orthogonality.marginal_ic_ir DESC, robustness.oos_ic_ir DESC
def to_markdown(cards) -> str; def to_json(cards) -> dict
```
- **Invariant:** the scorecard ALWAYS reports *both* gross IC and net capturability, so a
  cost-blocked-but-predictive signal is **LOGGED, never silently dropped** (the
  "structure-without-capture" lesson from r1-illiquidity / PEAD).
- **Verdict vocabulary** stops at **PROMISING** — never "GO". Promotion to any capital
  requires survivorship-free re-validation **and** a Tier-2 deep lifecycle audit (CLAUDE.md
  anti-pattern). The scorecard emits the escalation, not the green light.

---

## 5. Data Flow (shapes verified at every boundary)

```
yfinance/Sharadar ─prices─▶ PanelLoader ─(T,N) OHLCV+active+adv+sector─▶ Panel
Panel ─compute()─▶ scores (T,N) ─neutralize(winsor→z→sector)─▶ scores_n (T,N)
Panel.forward_returns(h) ─▶ fwd (T,N)
(scores_n, fwd) ─per-day Spearman over active names─▶ ic (n_days,)        [T1]
ic ─mean/std/skew/kurt─▶ {ic_mean, ic_ir, ic_tstat}                       [T1]
{ic_ir per candidate} ─deflated_sharpe_ratio(n_trials=B)─▶ dsr (scalar)   [T4]
scores_n ─rank→dollar-neutral vol-scaled weights (T,N)─▶ w; w⊗fwd(1)─net P&L  [T2]
net P&L ─sharpe/pf/max_dd─▶ capturability; frictionless−net ─▶ cost_wall  [T2]
L/S returns (n_days,) ⟂ FactorBook (n_days, k) ─OLS resid IC─▶ marginal   [T5]
```
- **Boundary check #1:** `scores.shape == fwd.shape == (T, N)` (asserted).
- **Boundary check #2:** `periods_per_year` is passed **explicitly = 252** to every
  `statistics.py` call (default is 8760 for crypto-hourly — silent annualization bug if
  omitted). This is a first-class gotcha (cf. "always pass metric_keys explicitly").

---

## 6. Architecture Decision Records

#### ADR-1: Rank by DSR on the daily-IC series (reuse `deflated_sharpe_ratio` verbatim)
- **Status:** Proposed.
- **Context:** Operator wants "deflated gross IC-IR" as the rank key; we screen hundreds of
  candidates, so multiple-testing deflation is mandatory.
- **Options:** (1) Invent a bespoke "deflated IC" estimator. (2) Recognize the daily
  cross-sectional IC is a return series whose per-period Sharpe **is** IC-IR, and feed it to
  the existing `deflated_sharpe_ratio(observed_sr=ic_ir, trial_sharpes=[batch ic_irs],
  n_trials=B)`.
- **Decision:** Option 2. Zero new statistics; the audited DSR/PSR/MinTRL suite applies
  directly. Primary `rank_key = dsr`; raw `ic_ir` and `ic_tstat` reported alongside.
- **Consequences:** Deflation is **batch-coupled** — `evaluate_batch` must hold all candidates
  to compute `trial_sharpes` and `n_trials`. Single-signal `evaluate_signal` returns
  un-deflated T1; deflation is filled in at batch time.

#### ADR-2: Whole-panel vectorized signals + verified (not assumed) causality
- **Status:** Proposed.
- **Context:** Coding standard forbids per-sample Python loops in hot paths; but causality
  must be guaranteed.
- **Decision:** `Signal.compute(panel) -> (T,N)` vectorized; causality enforced by the Tier-0
  **truncation tripwire** (generalizes `tests/prism_research/test_pathA_walk_forward.py`):
  for random `t`, `compute(panel.truncated(t))[t] == compute(panel)[t]`. A leaky signal fails
  the gate and is excluded — it never reaches ranking.
- **Consequences:** Signal authors write fast vectorized code; the harness, not the author's
  discipline, is the leak backstop. Negative-control test required (a deliberately leaky
  signal MUST fail).

#### ADR-3: Survivorship-bias is a gate, not a footnote (LEAK-EQ-1)
- **Status:** Proposed.
- **Context:** yfinance has only currently-listed names; a "current S&P 500" universe over
  history excludes dead losers ⇒ **inflated** IC. This is the #1 equity-X-section leak.
- **Decision:** v1 runs on free survivorship-biased data with every scorecard stamped
  `caveat: survivorship-biased — UPPER BOUND`. Any **PROMISING** verdict is gated behind
  re-validation on a survivorship-free, point-in-time dataset (Sharadar SF1/SEP, Norgate, or
  CRSP) before promotion. Mirrors the VRP "needs paid-data re-validation" discipline.
- **Consequences:** New invariant LEAK-EQ-1; `Panel.meta["survivorship_free"]` drives the
  caveat and the GO-gate. v1 results are explicitly upper bounds, used to *triage* which
  signals justify paid-data spend.

#### ADR-4: Report IC raw AND neutralized (sector/size)
- **Status:** Proposed.
- **Context:** Raw single-name IC can be a hidden sector/beta/size bet, not name-selection
  skill.
- **Decision:** Pipeline `winsorize(1/99 pct, per day) → cross-sectional z-score → residualize
  on sector dummies (+ log-ADV size proxy)`. Report IC on both raw and neutralized signal.
  Rank uses **neutralized** by default (configurable) so "predictive power" isn't a sector
  tilt. Rank-IC is invariant to monotone transforms of the forward return, so demeaning the
  *return* is irrelevant to IC; neutralizing the *signal* is what matters.
- **Consequences:** `neutralization` is a `SignalSpec` field; a signal whose IC vanishes after
  sector-neutralization is flagged "sector-driven."

#### ADR-5: Orthogonality basis = Fama-French/Carhart equity factors, not the ETF book
- **Status:** Proposed.
- **Context:** Tier 5 asks "does this add to what we own?" For *single-name equity*, the right
  control is the standard equity factor zoo (Mkt, SMB, HML, RMW, CMA, UMD — Ken French daily,
  free), not the cross-asset ETF momentum+rates-carry book (different universe).
- **Decision:** `FactorBook` defaults to FF5+UMD daily. Marginal IC-IR = IC of the residual
  after regressing signal L/S returns on the book. (The ETF book remains a selectable basis
  for cross-asset signals later.)
- **Consequences:** Adds a tiny Ken-French loader; orthogonality is measured in the *equity*
  factor space the signals actually live in.

#### ADR-6: Gates pre-registered in YAML, never hardcoded
- **Status:** Proposed (project invariant).
- **Decision:** All thresholds (coverage floors, IC-IR/DSR floors for PROMISING, FDR-q max,
  cost-wall caution, cost bps per profile, `survivorship_free_required`) live in
  `configs/signal_eval.gates.yaml`. Harness reads them; nothing numeric is hardcoded.

---

## 7. Dependency Map

| Module | Impact | Changes |
|---|---|---|
| `finrl_pro_ds/crypto/eval/statistics.py` | **Import only** | none — call DSR/PSR/MinTRL/bootstrap with `periods_per_year=252` |
| `scripts/research/cmgp1_x2_leak_ic_probe_v2.py` | Port `_xs_ic`/`_pooled_ic` | extract → `signals/_ic.py` (shared); leave probe importing the shared fn |
| `scripts/research/r1_illiquidity_probe.py` | Port `bh_fdr`/`one_sided_p`/`block_bootstrap_ic` | extract → `signals/_ic.py` |
| `scripts/research/xsec_momentum_falsification.py` | Reuse `COST_MODELS`, `max_dd`, vol-scaled weight pattern | new `signals/costs.py` promotes the template |
| `scripts/research/fable/fable_oracle.py` | Reuse `backtest_weights`, `profit_factor` | import for T2 |
| `finrl_pro_ds/data/multiscale_handler.py` | Pattern reference (causal resample, `(T,N)`) | none |
| `tests/prism_research/test_pathA_walk_forward.py` | Generalize tripwire | new `tests/signals/test_causality_tripwire.py` |
| **NEW** `finrl_pro_ds/signals/*` | new package | 7 files |
| **NEW** `configs/signal_eval.gates.yaml` | new config | pre-registered gates |
| **NEW** `scripts/research/eval_signals.py` | new CLI | batch driver |
| **NEW** `tests/signals/*` | new tests | 6 files |

**Import directions traced:** `signals/` imports `crypto.eval.statistics` + shared `_ic`;
nothing in `crypto/`, `agents/`, `envs/` imports `signals/` (leaf package — no cycles, no risk
to training/live paths). Respects the CLAUDE.md boundary (only touches `finrl_pro_ds/`,
`scripts/`, `configs/`, `tests/`).

---

## 8. Invariant Checklist

| ID | Preserved? | How |
|---|---|---|
| **LEAK-1** (norm reset at split) | ✅ | Neutralization z-score is **per-day cross-sectional** (never across time) ⇒ no train/test bleed by construction. Any time-series normalization resets at WF boundaries. |
| **LEAK-2** (temporal causality) | ✅ **enforced** | Tier-0 truncation tripwire on every signal; forward returns are LABEL-only; T+1 execution lag in T2 weights (`xsec_momentum` pattern). |
| **LEAK-EQ-1** (survivorship/PIT) | ✅ **new** | ADR-3: v1 caveated UPPER BOUND; PROMISING gated on survivorship-free re-validation. |
| **PF-XCHECK** (mid vs close >30% halt) | ✅ | T2 decile-spread PF cross-checked close-marked vs open-to-open-marked; >30% divergence ⇒ `hygiene` halt. |
| **DATA-CLEAN** (OHLCV validated, .bak) | ✅ | PanelLoader runs OHLC sanity (high≥max(o,c), low≤min(o,c), no stale prints); `.bak` on any cleaned source. |
| **BUG-01/03/04, SHORT-ACCT, MARGIN-CFG, NONBLOCK** | N/A | RL/env-specific; v1 is a linear statistical screen with no RL, no env, no `.to(device)`. Documented N/A. |
| Coding: no per-sample loops in hot path | ✅ | Signals vectorized `(T,N)`; T1 loops O(T) Spearman (≈5000 days) — acceptable, not a tensor hot path. |
| Coding: `logging` not `print` | ✅ | Package uses `MLOpsLogger`/`logging`; CLI may `print` the final table. |

---

## 9. Migration / Build Sequencing (purely additive — no breaking changes)

Each step independently testable + committable:

1. **`signals/_ic.py` + `costs.py`** — extract `_xs_ic`, `_pooled_ic`, `bh_fdr`, `one_sided_p`,
   `block_bootstrap_ic`, `COST_MODELS`, `max_dd` into shared module; re-point the two probe
   scripts to import them (refactor-in-place, behavior identical). *Tests: golden IC on
   synthetic panel; probes still pass.*
2. **`spec.py` + `protocol.py` + `registry.py`** — core abstractions. *Tests: spec hash
   determinism; registry round-trip.*
3. **`features.py` (Panel + PanelLoader)** — yfinance loader, OHLC sanity, neutralization.
   *Tests: shape/active invariants; neutralization removes a pure-sector signal's IC.*
4. **`eval_harness.py` T0+T1** — hygiene tripwire + gross power. *Tests: causality tripwire +
   negative control (leaky signal MUST fail); IC≈1 on signal==fwd, IC≈0 on shuffle.*
5. **`eval_harness.py` T2+T3+T5** — capturability, robustness, orthogonality.
6. **`eval_harness.py` T4 + `scorecard.py`** — batch deflation + ranking + md/json emit.
   *Tests: DSR wiring vs reference; lexicographic rank order; GATE_FAIL exclusion.*
7. **`configs/signal_eval.gates.yaml` + `scripts/research/eval_signals.py`** — CLI driver.
8. **Seed registry: 101 Formulaic Alphas** (codeable now) + a curated TradingView technical
   set → first batch → `results/signal_eval/batch01/scorecard.{json,md}`.

**Extension points (documented, not built in v1):** time-series (per-name) signal mode;
alt-data family (point-in-time fundamentals/sentiment loaders behind the same `Panel`);
LOB/microstructure family (intraday `Panel` with order-book fields). The `Signal` protocol and
`Panel` schema are designed so these are *additive*, not refactors.

---

## 10. Test Plan

| Layer | Test | File | Verifies |
|---|---|---|---|
| Unit | causality tripwire | `tests/signals/test_causality_tripwire.py` | causal signal passes; **leaky signal (uses close[t+1]) FAILS** (negative control) |
| Unit | IC golden | `tests/signals/test_ic_core.py` | signal==fwd ⇒ IC≈1; shuffled ⇒ IC≈0; `_xs_ic` parity with extracted fn |
| Unit | deflation wiring | `tests/signals/test_deflation.py` | DSR on IC series matches reference; dsr ↓ as n_trials ↑; PSR/MinTRL inverse identity |
| Unit | neutralization | `tests/signals/test_neutralization.py` | sector-demean kills a pure-sector signal's IC; raw vs neutralized both reported |
| Unit | ranking | `tests/signals/test_scorecard_rank.py` | lexicographic order (dsr→ic_ir→orthog); GATE_FAIL excluded; cost-blocked = LOGGED not dropped |
| Unit | PF-XCHECK | `tests/signals/test_pf_xcheck.py` | >30% close-vs-open PF divergence halts |
| Integration | smoke batch | — | `eval_signals.py` over a 3-signal toy registry emits valid scorecard.{json,md} |

---

## 11. Config Example — `configs/signal_eval.gates.yaml`

```yaml
# Pre-registered gates for the signal-eval harness. NEVER hardcode these in code.
universe:
  id: sp500_current
  min_names_per_day: 50          # else day excluded from IC
  min_adv_usd: 5.0e6             # liquidity floor per name-day
coverage:
  min_days: 1260                 # ~5y of daily bars
  max_nan_frac: 0.40
gross_power:
  primary_horizon: 5             # rank key uses 5d IC by default
  promising_ic_ir: 0.05          # per-period IC-IR floor for PROMISING
  promising_ic_tstat: 3.0
deflation:
  promising_dsr: 0.90            # P(true IC-IR>0 after deflation) floor
  fdr_q_max: 0.10                # Benjamini-Hochberg q ceiling
robustness:
  min_subperiod_ic_ir: 0.0       # no negative subperiod
  recent_oos_years: 2
capturability:                   # SECONDARY — reported, not gating (rank=gross power)
  cost_models: {frictionless: 0.0, standard: 0.0010, harsh: 0.0025}  # one-way, equity bps
  cost_wall_caution: 0.30        # frictionless−net sharpe gap flag
neutralization:
  winsor_pct: [0.01, 0.99]
  controls: [sector, size]       # size = log ADV proxy in v1
promotion:
  survivorship_free_required: true   # LEAK-EQ-1 GO-gate
  tier2_audit_required: true         # CLAUDE.md anti-pattern: no capital without Tier-2
```

---

## 12. Estimated Effort

| Phase | Files | LOC | Risk |
|---|---|---|---|
| 1 — extract shared `_ic`/`costs` | 2 + 2 refactors | ~180 | Low (behavior-preserving) |
| 2 — spec/protocol/registry | 3 | ~150 | Low |
| 3 — Panel + loader + neutralization | 1 | ~320 | **Med** (survivorship/PIT correctness, OHLC sanity) |
| 4 — T0+T1 | (harness) | ~280 | **Med** (causality gate is the backstop — must be airtight) |
| 5 — T2+T3+T5 | (harness) | ~300 | Med (cost model, FF loader) |
| 6 — T4 + scorecard | 1 | ~220 | Low (reuses DSR) |
| 7 — gates yaml + CLI | 2 | ~140 | Low |
| 8 — seed 101-alphas + TV set | 1 registry | ~400 | Med (faithful causal reimplementation) |
| Tests | 6 | ~400 | — |
| **Total** | ~20 files | **~2.4k LOC** | Med overall |

---

## 13. Open Questions (operator sign-off before implementation)

1. **Data source for v1.** Recommend **yfinance (free, survivorship-biased, caveated as UPPER
   BOUND)** for the screen, with survivorship-free (Sharadar ~$, Norgate, CRSP) as the
   PROMISING→GO re-validation gate. Confirm, or start paid immediately?
2. **Universe & size.** Recommend **current S&P 500** (~500 names; good liquidity + sector
   coverage) with the survivorship caveat. Alternatives: ~100-200 curated mega-caps (faster,
   less distortion) or Russell 1000 (broader, noisier). Confirm size.
3. **Orthogonality basis (ADR-5).** Recommend **Fama-French 5 + UMD (Ken French daily)** as
   the equity-factor control rather than the cross-asset ETF book. Confirm.
4. **Candidate seed batch.** Recommend batch-01 = **Kakushadze 101 Formulaic Alphas**
   (fully codeable now, a known benchmark) **+ a curated ~15 classic TradingView technicals**
   (RSI/Stoch/Supertrend/VWAP-bands/Squeeze-Momentum/etc., reimplemented causally). Note:
   there is **no API to pull arbitrary custom Pine scripts** — exotic TradingView customs are
   a manual reimplement-as-you-find path. Confirm this seed, or prioritize differently?
5. **Primary horizon for the rank key.** Default **5 trading days** (`gates.yaml`
   `primary_horizon`). Confirm, or rank on a different/blended horizon?
6. **NotebookLM dependency.** KB auth is expired; the literature pass on signal candidates
   (Phase 2) needs `notebooklm-mcp-auth` (interactive Chrome login). Will you re-auth, or
   should Phase-2 sourcing lean on web search + the 101-alphas paper directly?

> On sign-off I implement Phase 1→7 (the harness + tests, Audit-gated), then Phase 8 seeds the
> first batch and produces `results/signal_eval/batch01/scorecard.md`.
