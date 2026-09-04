# Data

**A fresh clone contains no market data.** `/data/` is gitignored (`.gitignore:93`, which
tracks only `*.dvc` pointers), and so is `results/`. Every command in the repo that names a
parquet under `data/` will fail until you fetch it yourself. This guide is the missing step
between installing the package and running anything on real prices.

If you only want to verify your install, skip this page — the synthetic paths in
[Getting started](getting-started.md) need no data at all.

---

## 1. Which source do you actually need?

Pick by what you intend to research, not by what is easiest to download.

| Research target | Source | Cost | Script |
|---|---|---|---|
| Cross-asset / ETF daily (TSMOM, the one surviving edge) | Stooq, yfinance | free, no key | `sharpen/crucible/data/stooq.py`, `sharpen/data/cross_asset_loader.py` |
| FX + metals intraday (XAUUSD, EURUSD) | Dukascopy ticks | free, no key | `scripts/data/fetch_dukascopy.py` |
| Equity intraday (SPY, SPX500) | Dukascopy | free, no key | `scripts/data/fetch_dukascopy_equity.py`, `scripts/data/prepare_spy_dataset.py` |
| Crypto spot/perp OHLCV | Binance | free, no key | `scripts/data/fetch_binance_ohlcv.py` |
| FX/CFD broker-native bars | OANDA v20 | free demo key | `scripts/fetch_xauusd_oanda.py`, `scripts/fetch_eurusd_oanda.py` |
| CME futures (GC/MGC) bars + MBP | Databento | **paid** | `scripts/fetch_gc_front_month.py`, `scripts/fetch_gc_mbp10.py` |
| Crypto L2 order book | CoinAPI / live recorders | paid / free-to-record | `scripts/data/fetch_coinapi_lob.py`, `scripts/record_bitfinex_lob.py` |
| Taiwan equities + options | FinMind | free tier | `scripts/data/fetch_taiwan_finmind.py` |
| Macro, positioning, filings, news | FRED, CFTC COT, SEC EDGAR, GDELT | free, key optional | `sharpen/crucible/data/` connectors |

### Credentials

Only these need one. Set them as environment variables (a `.env` at the repo root is read
via `python-dotenv`):

| Variable | Used by | Notes |
|---|---|---|
| `OANDA_TOKEN` | `fetch_xauusd_oanda.py`, `fetch_eurusd_oanda.py` | A free practice account issues one |
| `DATABENTO_API_KEY` | `fetch_gc_front_month.py` and the MBP fetchers | Paid; the only hard-paywalled source in the default paths |
| `COINAPI_KEY` | `fetch_coinapi_lob.py` | Or pass `--api-key` |
| `FINMIND_TOKEN` | Taiwan fetchers | **Optional.** The free tier works tokenless; the token only raises the quota. Free tier is ~600 requests/hour — fetch `universe/membership.parquet` (612 rows), not `pool.parquet` (2131), or you will exhaust it |
| `FRED_API_KEY` | `sharpen/crucible/data/fred.py` | Free from the St. Louis Fed |

Everything else — Stooq, Dukascopy, Binance public OHLCV, CFTC COT, SEC EDGAR, GDELT — is
genuinely keyless.

---

## 2. The mandatory hygiene pipeline

Raw data is never used directly. Three steps, in order, and the first is a hard project
invariant (`DATA-CLEAN`).

### Step 1 — clean

```bash
python scripts/clean_ohlcv.py --input data/raw/xauusd_1min.parquet
```

Fixes decimal-shift errors, re-derives inconsistent OHLC relationships, and flags stale
bars. **It writes a `.bak` alongside the input by default — do not pass `--no_backup`
unless you have your own copy.** Preview first with `--dry_run`.

This step exists because gold data was silently corrupted once (session S106) and the
corruption was invisible until it had contaminated downstream experiments. Treat a skipped
clean as a leak.

| Flag | Effect |
|---|---|
| `--dry_run` | Report only, write nothing |
| `--threshold 0.05` | Relative jump that counts as a decimal error |
| `--no_rederive` | Leave inconsistent OHLC alone instead of repairing |
| `--stale_strict` | Fail rather than warn on repeated identical bars |
| `--no_backup` | Skip the `.bak` (not recommended) |

### Step 2 — resample to the trading timeframe

```bash
python scripts/resample_ohlcv.py --input data/clean/xauusd_1min.parquet --output data/processed/xauusd_15min.parquet --resolution 15min
```

`scripts/resample_3min.py` is the fixed-3-minute variant used by the SG-1 workstream.

> **`LEAK-2` applies here.** Resampling conventions decide whether a bar can see its own
> future. The `label` and `closed` conventions must agree, so that a 15-minute bar stamped
> 10:00 covers 10:00–10:15 and is only readable at 10:15. Multi-scale features must map to
> the last **closed** coarse bar, never the in-progress one — that exact mistake (the "X2
> coarse-bar leak") lived undetected in `sharpen/data/multiscale_handler.py` for hundreds
> of sessions and erased an apparent edge when it was finally found.

### Step 3 — write a data manifest

```bash
python scripts/build_data_manifest.py data/processed/xauusd_15min.parquet --write
```

Note the **positional** argument — there is no `--data` flag. Without `--write` it is a dry
run. The manifest is Stage 0 of [the training protocol](rl-pipeline.md) and records the
content hash the later stages check against, so an accidental data swap mid-experiment is
caught instead of silently changing your results.

---

## 3. Worked example: free FX data end to end

```bash
python scripts/data/fetch_dukascopy.py --instrument XAUUSD --start 2015-01-01 --end 2026-01-01 --bar 1min --out data/dukascopy
```

```bash
python scripts/clean_ohlcv.py --input data/dukascopy/XAUUSD_1min.parquet
```

```bash
python scripts/resample_ohlcv.py --input data/dukascopy/XAUUSD_1min.parquet --output data/processed/xauusd_15min.parquet --resolution 15min
```

```bash
python scripts/build_data_manifest.py data/processed/xauusd_15min.parquet --write
```

`fetch_dukascopy.py` parallelizes with `--workers` (default 12) and resumes from partial
downloads, which matters — a decade of tick data is a long fetch.

> **Known source quirk:** Dukascopy's SPY series carries 18–75 hour gaps. Use OANDA's
> `SPX500` instead for equity-index work; this was measured, not assumed.

---

## 4. Point-in-time correctness

Every connector under `sharpen/crucible/data/` implements the `DataConnector` protocol and
stamps a `release_timestamp` separate from the `reference_period`.
`quality_gate.asof_join` binds each observation to the first bar **on or after its public
release**, never the period it describes. This is not decoration; it is the difference
between a real signal and a look-ahead artifact:

- **CFTC COT** describes Tuesday's positions but publishes the following Friday. Joining it
  to Tuesday is a three-day look-ahead that will manufacture an edge from nothing.
- **SEC EDGAR** is the cleanest true-PIT source in the stack — the XBRL company-concept API
  returns an actual `filed` date per observation rather than a modelled lag.
- **FRED** is the weakest by default: the standard fetch path is a *release-lag model*, not
  a true vintage read. It is PIT-safe only for series that are never meaningfully revised.
  For a revised series you need genuine ALFRED vintages (`output_type=2`).
- **Stooq** history is split- and dividend-*adjusted*, so a backward-looking adjustment is
  baked into old bars. Fine for momentum ranking; wrong for anything reading absolute price
  levels as they were quoted.

If you add a connector, implement the release model honestly and add a negative test that
fails when look-ahead is reintroduced. Every `LEAK-2` guard in this repo has one.

---

## 5. Survivorship bias

The signal scorecard prints `survivorship-free data: False` and labels its own results
**UPPER BOUNDS** when the panel cannot prove point-in-time membership. Take that literally.

Free equity data has a specific pathology this repo has measured: survivorship bias is a
**time gradient**, not a constant offset — your training window and your out-of-sample
window contain different universes, so the bias itself changes across the split. The
documented fix is to build the cohort from index **exit** events rather than current
membership. `sharpen/data/sp500_pit_panel.py` and `scripts/research/xlg_pit_validation.py`
implement and check this.

---

## 6. Storage layout

The repo expects, and gitignores, this shape:

```
data/
├── raw/          # exactly as fetched, never edited
├── clean/        # post clean_ohlcv.py, with .bak siblings
├── processed/    # resampled to trading timeframe + .manifest.json
├── equities/     # Dukascopy / OANDA equity series
└── universe/     # membership panels, sector maps
results/          # every run artifact — also gitignored
```

Both trees are excluded from git deliberately: run artifacts and price history do not
belong in history. Track large datasets with DVC pointers (`data/**/*.dvc` is the one
allowlisted pattern) or an external store.
