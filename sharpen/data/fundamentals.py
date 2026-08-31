"""BALLAST P1 — point-in-time fundamentals bound to an equity ``Panel``.

Feeds the quality (S2) and value (S3) sleeves. Three things make this PIT-correct rather than
merely "fundamental data joined by date":

1. **Publication-time join, not reference-period join.** A fiscal quarter ending 2020-03-31 is not
   knowable on 2020-03-31; it becomes public when the 10-Q is filed, typically 30-45 days later.
   Every observation carries ``release_timestamp = filed + 1 day`` (the after-close convention,
   C1-02) and the bind uses :func:`quality_gate.asof_join`, which is reused verbatim — bar ``t``
   sees only what was public by ``t``, and revisions supersede in release order.
2. **Restatement-safe TTM.** Trailing-twelve-month aggregates are built at the *observation* level:
   the TTM stamped at quarter ``k``'s release sums the four most recent fiscal quarters **using the
   values known at that release**. A later restatement produces a new TTM observation with its own
   later release, so it never back-dates.
3. **Availability, not imputation.** Free XBRL coverage begins ~2009 (the mandate phased in
   2009Q2-2011). Pre-coverage bars are NaN and carry an explicit availability mask; they are never
   forward-filled from the first available filing, which would be a decade-long look-ahead.

**Split-adjustment trap (read before adding any price-based ratio).** EDGAR reports share counts and
per-share figures **as reported** — not split-adjusted. The panel's close comes from yfinance with
``auto_adjust=True`` — split *and* dividend adjusted. Multiplying the two gives a market cap that
jumps by the split factor on every split. So :func:`market_cap` demands an **unadjusted** close
matrix and refuses to guess; ratios that need no share count (gross profitability, ROE, accruals,
leverage) are computed from accounting quantities alone and are immune.

Transport is injectable, so every test here runs offline.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sharpen.crucible.data.connector import Provenance, SeriesData, SeriesRef
from sharpen.crucible.data.quality_gate import asof_join

log = logging.getLogger("fundamentals")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = ROOT / "data" / "raw" / "fundamentals"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_LICENSE = "SEC EDGAR — U.S. Government public data (no redistribution limit)"

# XBRL mandate phase-in. Before this, coverage is absent-to-partial by construction, not by bug.
XBRL_COVERAGE_START = np.datetime64("2009-04-01")

#: concept -> (us-gaap tags in preference order, kind). ``flow`` = duration (income/cash-flow
#: statement, needs the standalone-quarter span filter + TTM); ``stock`` = instant (balance sheet).
CONCEPTS: dict[str, tuple[tuple[str, ...], str]] = {
    "revenue": (("RevenueFromContractWithCustomerExcludingAssessedTax",
                 "Revenues", "SalesRevenueNet"), "flow"),
    "net_income": (("NetIncomeLoss",), "flow"),
    "gross_profit": (("GrossProfit",), "flow"),
    "operating_income": (("OperatingIncomeLoss",), "flow"),
    "cfo": (("NetCashProvidedByUsedInOperatingActivities",
             "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"), "flow"),
    "capex": (("PaymentsToAcquirePropertyPlantAndEquipment",), "flow"),
    "assets": (("Assets",), "stock"),
    "liabilities": (("Liabilities",), "stock"),
    "equity": (("StockholdersEquity",
                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"), "stock"),
    "shares": (("CommonStockSharesOutstanding",
                "WeightedAverageNumberOfDilutedSharesOutstanding"), "stock"),
}
_QUARTER_DAYS = (80, 100)
_PREFERRED_UNITS = ("USD", "shares", "USD/shares", "pure")


# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class FundamentalPanel:
    """PIT fundamentals aligned to a price panel's date × ticker grid.

    ``values[concept]`` is ``(T, N)`` float64, NaN where the figure was not yet public.
    ``available`` is ``(T, N)`` bool — True where *any* concept is known, i.e. where the
    fundamental sleeves may contribute at all.
    """

    dates: np.ndarray                     # (T,) datetime64[ns]
    tickers: tuple[str, ...]              # (N,)
    values: dict[str, np.ndarray]         # concept -> (T, N)
    available: np.ndarray                 # (T, N) bool
    meta: dict

    def get(self, concept: str) -> np.ndarray:
        if concept not in self.values:
            raise KeyError(f"unknown concept {concept!r}; have {sorted(self.values)}")
        return self.values[concept]

    @property
    def coverage_by_date(self) -> np.ndarray:
        """(T,) fraction of names with any fundamental known — the honest coverage curve."""
        return self.available.mean(axis=1)


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #
def _live_transport(user_agent: str | None = None, *, rate_per_sec: float = 8.0) -> Callable[[str], dict]:
    """SEC requires a descriptive User-Agent and rate-limits at 10 req/s; we stay under."""
    ua = user_agent if user_agent is not None else os.environ.get("SEC_EDGAR_UA", "")
    if not ua:
        raise RuntimeError(
            "SEC EDGAR requires a descriptive User-Agent. Set SEC_EDGAR_UA "
            "(e.g. 'FinRL-Pro-DS research you@example.com') or pass user_agent=/transport=.")
    min_gap = 1.0 / rate_per_sec
    state = {"last": 0.0}

    def _transport(url: str) -> dict:
        gap = time.monotonic() - state["last"]
        if gap < min_gap:
            time.sleep(min_gap - gap)
        req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip"})
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed SEC host
            raw = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip":
                import gzip
                raw = gzip.decompress(raw)
        state["last"] = time.monotonic()
        return json.loads(raw)

    return _transport


def load_ticker_cik_map(
    *, cache_dir: Path = DEFAULT_CACHE_DIR, transport: Callable[[str], dict] | None = None,
) -> dict[str, str]:
    """``{TICKER: 10-digit CIK}`` from SEC's official mapping (cached)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "company_tickers.json"
    if path.exists():
        payload = json.loads(path.read_text())
    else:
        payload = (transport or _live_transport())(_TICKERS_URL)
        path.write_text(json.dumps(payload))
    out: dict[str, str] = {}
    for row in payload.values() if isinstance(payload, dict) else payload:
        tk = str(row.get("ticker", "")).strip().upper().replace(".", "-")
        cik = str(row.get("cik_str", row.get("cik", ""))).strip()
        if tk and cik.isdigit():
            out[tk] = cik.zfill(10)
    return out


def _prune_facts(payload: dict, keep_tags: frozenset[str]) -> dict:
    """Keep only the us-gaap tags we model.

    A full ``companyfacts`` payload carries every tag the filer ever reported (~500 for a large
    company, ~4 MB); across the BALLAST universe that is >3 GB of cache for the ~10 tags the sleeves
    read. The kept set is recorded in the cache so a later CONCEPTS change re-fetches rather than
    silently serving a payload that is missing a newly-requested tag.
    """
    gaap = payload.get("facts", {}).get("us-gaap", {})
    return {"cik": payload.get("cik"), "entityName": payload.get("entityName"),
            "_kept_tags": sorted(keep_tags),
            "facts": {"us-gaap": {t: v for t, v in gaap.items() if t in keep_tags}}}


def all_tags(concepts: dict[str, tuple[tuple[str, ...], str]] | None = None) -> frozenset[str]:
    """Every us-gaap tag referenced by ``concepts`` (across all preference-order fallbacks)."""
    return frozenset(t for tags, _ in (concepts or CONCEPTS).values() for t in tags)


def fetch_company_facts(
    cik: str, *, cache_dir: Path = DEFAULT_CACHE_DIR,
    transport: Callable[[str], dict] | None = None,
    keep_tags: frozenset[str] | None = None,
) -> dict | None:
    """One ``companyfacts`` call returns EVERY concept for one filer.

    Deliberately not the per-concept ``companyconcept`` endpoint used by
    ``crucible/data/edgar.py``: that costs one request per (company, concept), which is ~8k requests
    for the BALLAST universe versus ~0.8k here. The PIT semantics are identical; only the payload
    shape differs. The response is pruned to ``keep_tags`` before caching (see :func:`_prune_facts`).
    """
    keep = keep_tags if keep_tags is not None else all_tags()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"facts_{cik}.json"
    if path.exists():
        try:
            cached = json.loads(path.read_text())
        except json.JSONDecodeError:
            log.warning("corrupt cache %s - refetching", path.name)
        else:
            # ``null`` is the negative cache for a filer with no XBRL; honour it.
            if cached is None or keep.issubset(set(cached.get("_kept_tags", ()))):
                return cached
            log.debug("cache %s lacks newly-requested tags - refetching", path.name)
    try:
        payload = (transport or _live_transport())(_FACTS_URL.format(cik=cik))
    except Exception as exc:  # noqa: BLE001 - a filer with no XBRL is normal, not fatal
        log.debug("companyfacts %s failed: %r", cik, exc)
        path.write_text("null")          # negative-cache so a rebuild does not re-hit SEC
        return None
    pruned = _prune_facts(payload, keep)
    path.write_text(json.dumps(pruned))
    return pruned


# --------------------------------------------------------------------------- #
# Parsing — same PIT conventions as crucible/data/edgar.py
# --------------------------------------------------------------------------- #
def _pick_unit(units: dict) -> list[dict]:
    for u in _PREFERRED_UNITS:
        if u in units:
            return list(units[u])
    return list(next(iter(units.values()), []))


def _parse_concept(facts: dict, tags: Sequence[str], kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(reference_period, value, release_timestamp)`` for the first tag that yields data.

    ``release_timestamp = filed + 1 day`` — an after-close filing is only actionable next session
    (C1-02; the same convention every connector in this repo uses). For ``flow`` concepts only the
    standalone-quarter span (80-100d) is kept, so year-to-date and annual frames of the same period
    do not build a sawtooth series.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        rows = _pick_unit(gaap.get(tag, {}).get("units", {}))
        refs: list[np.datetime64] = []
        vals: list[float] = []
        rels: list[np.datetime64] = []
        for row in rows:
            filed, end_p, val = row.get("filed"), row.get("end"), row.get("val")
            if filed is None or end_p is None or val is None:
                continue
            try:
                reference = np.datetime64(str(end_p)[:10], "ns")
                release = np.datetime64(str(filed)[:10], "ns") + np.timedelta64(1, "D")
                value = float(val)
            except (TypeError, ValueError):
                continue
            if kind == "flow":
                start_p = row.get("start")
                if start_p is None:
                    continue                    # a flow with no duration is unusable
                try:
                    span = int((reference - np.datetime64(str(start_p)[:10], "ns"))
                               / np.timedelta64(1, "D"))
                except (TypeError, ValueError):
                    continue
                if not _QUARTER_DAYS[0] <= span <= _QUARTER_DAYS[1]:
                    continue
            refs.append(reference)
            vals.append(value)
            rels.append(release)
        if refs:
            return (np.array(refs, dtype="datetime64[ns]"),
                    np.array(vals, dtype=np.float64),
                    np.array(rels, dtype="datetime64[ns]"))
    empty_d = np.array([], dtype="datetime64[ns]")
    return empty_d, np.array([], dtype=np.float64), empty_d


def ttm_observations(
    ref: np.ndarray, val: np.ndarray, rel: np.ndarray, *, min_quarters: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Restatement-safe trailing-twelve-month aggregation of a quarterly flow.

    Emitted at the *observation* level: for each release ``r`` (in release order), the reading in
    effect for every fiscal period is the newest value released at or before ``r``; the TTM stamped
    at ``r`` sums the four most recent such periods, and is emitted only when four are known. A
    restatement therefore produces a NEW TTM observation carrying its own later release — it never
    back-dates an earlier bar.
    """
    if ref.shape[0] == 0:
        return ref, val, rel
    order = np.lexsort((ref, rel))
    ref_s, val_s, rel_s = ref[order], val[order], rel[order]
    in_effect: dict[np.datetime64, float] = {}
    out_ref: list[np.datetime64] = []
    out_val: list[float] = []
    out_rel: list[np.datetime64] = []
    i, n = 0, ref_s.shape[0]
    while i < n:
        j = i
        while j < n and rel_s[j] == rel_s[i]:       # apply all observations of one release together
            in_effect[ref_s[j]] = float(val_s[j])
            j += 1
        periods = sorted(in_effect)[-min_quarters:]
        if len(periods) == min_quarters:
            out_ref.append(periods[-1])
            out_val.append(float(sum(in_effect[p] for p in periods)))
            out_rel.append(rel_s[i])
        i = j
    return (np.array(out_ref, dtype="datetime64[ns]"),
            np.array(out_val, dtype=np.float64),
            np.array(out_rel, dtype="datetime64[ns]"))


def _series(concept: str, ticker: str, ref, val, rel) -> SeriesData:
    sref = SeriesRef("edgar_facts", f"{ticker}:{concept}", "equity", title=concept, frequency="Q")
    return SeriesData(
        ref=sref, reference_period=ref, value=val, release_timestamp=rel,
        provenance=Provenance(source_id="edgar_facts",
                              url=_FACTS_URL.format(cik="<cik>"), license=_LICENSE,
                              as_of_policy="release-lag", release_lag_days=1,
                              revision_policy="revised"),
        meta={"n_obs": int(ref.shape[0])})


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_fundamental_panel(
    dates: np.ndarray,
    tickers: Sequence[str],
    *,
    concepts: dict[str, tuple[tuple[str, ...], str]] | None = None,
    ttm_concepts: Sequence[str] = ("revenue", "net_income", "gross_profit",
                                   "operating_income", "cfo", "capex"),
    cache_dir: Path = DEFAULT_CACHE_DIR,
    transport: Callable[[str], dict] | None = None,
    cik_map: dict[str, str] | None = None,
) -> FundamentalPanel:
    """Bind PIT fundamentals to ``dates`` × ``tickers``.

    Flow concepts are emitted **twice**: ``<name>`` (latest standalone quarter) and ``<name>_ttm``
    (trailing four quarters). Sleeve formulas should use the TTM form — a single quarter is
    seasonal, and seasonality would enter the cross-section as noise.
    """
    concepts = concepts or CONCEPTS
    dates = np.asarray(dates, dtype="datetime64[ns]")
    tickers = tuple(str(t).upper() for t in tickers)
    cikmap = cik_map if cik_map is not None else load_ticker_cik_map(
        cache_dir=cache_dir, transport=transport)

    names = list(concepts) + [f"{c}_ttm" for c in ttm_concepts]
    values = {c: np.full((len(dates), len(tickers)), np.nan, dtype=np.float64) for c in names}

    n_mapped = n_facts = 0
    for j, tk in enumerate(tickers):
        cik = cikmap.get(tk)
        if cik is None:
            continue
        n_mapped += 1
        facts = fetch_company_facts(cik, cache_dir=cache_dir, transport=transport,
                                    keep_tags=all_tags(concepts))
        if not facts:
            continue
        n_facts += 1
        for concept, (tags, kind) in concepts.items():
            ref, val, rel = _parse_concept(facts, tags, kind)
            if ref.shape[0] == 0:
                continue
            values[concept][:, j] = asof_join(_series(concept, tk, ref, val, rel), dates)
            if concept in ttm_concepts:
                t_ref, t_val, t_rel = ttm_observations(ref, val, rel)
                if t_ref.shape[0]:
                    values[f"{concept}_ttm"][:, j] = asof_join(
                        _series(f"{concept}_ttm", tk, t_ref, t_val, t_rel), dates)

    available = np.zeros((len(dates), len(tickers)), dtype=bool)
    for arr in values.values():
        available |= np.isfinite(arr)

    meta = {
        "source": "SEC EDGAR XBRL companyfacts",
        "pit": "release_timestamp = filed + 1 day; asof_join (publication-time)",
        "coverage_start": str(XBRL_COVERAGE_START)[:10],
        "coverage_note": "XBRL mandate phased 2009Q2-2011; pre-2009 is absent by construction",
        "n_tickers": len(tickers), "n_cik_mapped": n_mapped, "n_with_facts": n_facts,
        "concepts": names,
    }
    return FundamentalPanel(dates, tickers, values, available, meta)


# --------------------------------------------------------------------------- #
# Derived quantities
# --------------------------------------------------------------------------- #
def _safe_div(a: np.ndarray, b: np.ndarray, *, min_denom: float = 1e-6) -> np.ndarray:
    """Elementwise a/b, NaN where the denominator is ~0 or non-positive-and-meaningless."""
    b = np.where(np.abs(b) < min_denom, np.nan, b)
    with np.errstate(invalid="ignore", divide="ignore"):
        return a / b


def market_cap(fp: FundamentalPanel, close_unadjusted: np.ndarray) -> np.ndarray:
    """Shares × price — and it must be the **UNADJUSTED** close.

    EDGAR share counts are as-reported. The panel's ``close`` is split- and dividend-adjusted, so
    ``shares × adjusted_close`` jumps by the split factor at every split and silently corrupts every
    price-based ratio built on it. This function will not accept the panel close by default; the
    caller must fetch unadjusted prices (``fetch_ohlcv_wide(..., auto_adjust=False)``) and pass them
    here explicitly, so the requirement cannot be met by accident.
    """
    shares = fp.get("shares")
    if close_unadjusted.shape != shares.shape:
        raise ValueError(
            f"close_unadjusted {close_unadjusted.shape} must match the fundamental grid "
            f"{shares.shape}")
    return shares * close_unadjusted


def gross_profitability(fp: FundamentalPanel) -> np.ndarray:
    """Novy-Marx GP/A on TTM gross profit. Scale-free; needs no share count or price."""
    return _safe_div(fp.get("gross_profit_ttm"), fp.get("assets"))


def return_on_equity(fp: FundamentalPanel) -> np.ndarray:
    return _safe_div(fp.get("net_income_ttm"), fp.get("equity"))


def accruals(fp: FundamentalPanel) -> np.ndarray:
    """(TTM earnings − TTM operating cash flow) / assets. High = low earnings quality."""
    return _safe_div(fp.get("net_income_ttm") - fp.get("cfo_ttm"), fp.get("assets"))


def leverage(fp: FundamentalPanel) -> np.ndarray:
    return _safe_div(fp.get("liabilities"), fp.get("assets"))


def earnings_yield(fp: FundamentalPanel, mcap: np.ndarray) -> np.ndarray:
    return _safe_div(fp.get("net_income_ttm"), mcap)


def fcf_yield(fp: FundamentalPanel, mcap: np.ndarray) -> np.ndarray:
    """Free cash flow = TTM CFO − TTM capex (capex is reported positive as an outflow)."""
    return _safe_div(fp.get("cfo_ttm") - fp.get("capex_ttm"), mcap)


def book_to_price(fp: FundamentalPanel, mcap: np.ndarray) -> np.ndarray:
    return _safe_div(fp.get("equity"), mcap)
