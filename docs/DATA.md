# Data sources and licensing

**This repository distributes no market data.** No price, order-book, positioning, filing or
news dataset is committed to it, and `data/` and `results/` are gitignored. Every dataset the
code reads must be fetched by you, from the provider, under that provider's terms.

For *how* to fetch, clean and store the data, see [guides/data.md](guides/data.md). This page
covers *where it comes from* and *why it is not included*.

## Why nothing is redistributed

1. **Provider terms.** Most of these sources permit personal or research use of data you fetch
   yourself, but not republication of the dataset. Several are commercial products. The code is
   Apache-2.0; the data is not ours to relicense.
2. **Reproducibility is better served by fetchers than by snapshots.** Each source below has a
   script or connector in the repo, and the hygiene pipeline (`scripts/clean_ohlcv.py`, resample,
   manifest) records what was fetched and how it was transformed.
3. **Size.** Tick and order-book history runs to gigabytes and does not belong in git history.

## Sources

Terms change. **Check each provider's current terms of use before fetching, and again before
sharing anything derived from the data.** "Free" below means free to access, not free to
redistribute.

| Source | What the repo uses it for | Access | Fetcher |
|---|---|---|---|
| [Stooq](https://stooq.com) | Daily OHLCV for ETFs, indices, FX, commodities (Crucible market connector) | free, no key | `sharpen/crucible/data/stooq.py` |
| Yahoo Finance via `yfinance` | Daily ETF history. **The TSMOM survivor, the TAILWIND audit and the value test all fetch from here** (`scripts/research/xsec_momentum_falsification.py`) | free, no key; unofficial API | `sharpen/data/cross_asset_loader.py`, the scripts above |
| [Dukascopy](https://www.dukascopy.com) | FX, metals and equity-index tick/intraday history | free, no key | `scripts/data/fetch_dukascopy.py`, `scripts/data/fetch_dukascopy_equity.py` |
| [OANDA](https://www.oanda.com) v20 API | Broker-native FX/CFD bars | free practice-account token | `scripts/fetch_xauusd_oanda.py`, `scripts/fetch_eurusd_oanda.py` |
| [Databento](https://databento.com) | CME futures bars and MBP-10 order book | **paid** | `scripts/fetch_gc_front_month.py`, `scripts/fetch_gc_mbp10.py` |
| [Binance](https://www.binance.com) public API | Crypto spot/perp OHLCV | free, no key | `scripts/data/fetch_binance_ohlcv.py` |
| [Bitfinex](https://www.bitfinex.com), [Bybit](https://www.bybit.com) | Crypto order-book recording, exchange connectivity | free public endpoints | `scripts/record_bitfinex_lob.py`, `sharpen/crypto/` |
| [CoinAPI](https://www.coinapi.io) | Historical crypto L2 order book | **paid** | `scripts/data/fetch_coinapi_lob.py` |
| [FinMind](https://finmindtrade.com) | Taiwan equities, futures and options | free tier (~600 req/h), token optional | `scripts/data/fetch_taiwan_finmind.py` |
| [TWSE](https://www.twse.com.tw), [TAIFEX](https://www.taifex.com.tw) | Taiwan institutional flows and large-trader open interest | free, public | `sharpen/crucible/data/twse_institutional.py`, `taifex_positioning.py` |
| [FRED / ALFRED](https://fred.stlouisfed.org) (St. Louis Fed) | Macro series | free API key | `sharpen/crucible/data/fred.py` |
| [CFTC Commitments of Traders](https://www.cftc.gov/MarketReports/CommitmentsofTraders) | Futures positioning | free, public | `sharpen/crucible/data/cftc_cot.py` |
| [SEC EDGAR](https://www.sec.gov/edgar) XBRL API | Reported fundamentals with filing dates | free, public (SEC fair-access rules apply: declare a User-Agent, respect rate limits) | `sharpen/crucible/data/edgar.py` |
| [GDELT 2.0](https://www.gdeltproject.org) | News tone / sentiment timelines | free, public | `sharpen/crucible/data/gdelt.py` |

Individual FRED series can carry their own copyright restrictions set by the originating
source, separate from FRED's own terms. Check the series notes.

## What does ship

- **Synthetic data generators** used by the test suite and the `--mode synthetic` Crucible paths.
  They need no external data and carry no third-party terms.
- **Configuration files** that *name* datasets and file paths. A path in a config is not the
  data.
- **Research reports** under `docs/research/` quote summary statistics (Sharpe ratios, profit
  factors, IC values) computed from these sources. They contain no raw series.

## Known data caveats

Two data properties invalidated results in this project's own history and are worth knowing
before you trust anything built on free data:

- **Survivorship bias is a time gradient**, not a constant offset: free equity universes
  contain today's survivors, so the bias differs between the training and holdout windows.
  Build cohorts from index-exit events. See [guides/data.md §5](guides/data.md).
- **Release time is not reference time.** CFTC COT describes Tuesday and publishes Friday;
  FRED's default fetch is a release-lag model, not a vintage read. Every connector stamps a
  separate `release_timestamp` for this reason. See [guides/data.md §4](guides/data.md) and
  [LEAKS_FOUND.md](LEAKS_FOUND.md).
