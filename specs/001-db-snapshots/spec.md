# Feature Specification: Database‑Backed Data Snapshots

**Feature Branch**: `001-db-snapshots`  
**Created**: 2025-11-06  
**Status**: Draft  
**Input**: User description: "Implement a database-backed data snapshot system for FinRL Pro: store OHLCV bars in TimescaleDB (market_bars hypertable), add snapshots + snapshot_assets tables, CLI finrl_pro.data.snapshot (fetch via yfinance/Alpaca, upsert to DB, record metadata: snapshot_id, params_json, lib_versions, code_hash), CLI finrl_pro.data.export_snapshot (DB→Parquet/CSV), resolve dataset_hash 'snapshot://<snapshot_id>' in finrl_pro.data.loader, integrate with Trainer + README, and add tests"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Create Snapshot via CLI (Priority: P1)

As a researcher, I can run a CLI to fetch market data (e.g., Yahoo/Alpaca), upsert rows into the database, and receive a `snapshot_id` I can use in experiments.

**Why this priority**: Establishes the system of record and reproducibility anchor for all downstream pipelines.

**Independent Test**: Run `python -m finrl_pro.data.snapshot --provider yahoo --tickers SPY,AAPL --start 2020-01-01 --end 2024-01-01 --interval 1d` and verify: (a) rows are present in `market_bars`; (b) a new entry in `snapshots` with correct params and `snapshot_id` is created; (c) CLI prints the `snapshot_id` JSON.

**Acceptance Scenarios**:

1. Given valid tickers and window, When I invoke the snapshot CLI, Then bars are upserted and a `snapshot_id` is printed.
2. Given a repeated request with the same params, When I invoke the snapshot CLI, Then idempotent upsert occurs and a new `snapshots` record is created with linkage to existing bars.

---

### User Story 2 - Export Snapshot to File (Priority: P2)

As a researcher, I can export a snapshot to Parquet/CSV for offline training or sharing.

**Why this priority**: Enables file-based workflows while keeping DB as the source of truth.

**Independent Test**: Run `python -m finrl_pro.data.export_snapshot --id <snapshot_id> --out data/snapshots --format parquet` and verify the file exists, row count matches DB, and checksum is recorded.

**Acceptance Scenarios**:

1. Given an existing `snapshot_id`, When I export to Parquet, Then a file is written with matching row counts and a checksum recorded.

---

### User Story 3 - Use Snapshot in Training (Priority: P2)

As a practitioner, I can reference `dataset_hash: snapshot://<snapshot_id>` in experiment YAML, and the loader resolves it from DB to a DataFrame for training.

**Why this priority**: Integrates snapshots into the FinRL Pro training loop for end‑to‑end reproducibility.

**Independent Test**: Update experiment config to `dataset_hash: snapshot://<id>`, run the trainer, and verify it loads data via the DB loader and completes with a fingerprint including the `snapshot_id`.

**Acceptance Scenarios**:

1. Given a valid `snapshot_id`, When I run the trainer, Then data loads via DB and the fingerprint persists `dataset_hash: snapshot://<id>`.

---

### Edge Cases

- Empty results window (no bars) → CLI returns error with guidance.
- Rate limits or provider outages → retry with backoff, partial failure reported.
- Duplicate bars (same timestamp,ticker) → upsert strategy ensures idempotency.
- Late corporate actions/adjustments → version bars via `vendor_rev`; snapshots retain params used.
- Timezone normalization and missing days/holidays → consistent PIT indexing enforced.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001 (DB Schema)**: Create TimescaleDB/PostgreSQL schema:
  - `market_bars` hypertable keyed by `(timestamp, ticker)` with columns: `open, high, low, close, volume, source, vendor_rev, ingest_ts`.
  - `snapshots` table with: `snapshot_id (UUID PK), provider, params_json, created_ts, code_hash, lib_versions_json, row_count`.
  - `snapshot_assets` table with: `snapshot_id (FK), ticker`.
- **FR-002 (Snapshot CLI)**: Implement `finrl_pro.data.snapshot` CLI to fetch (yfinance/Alpaca), upsert bars, insert a `snapshots` row, and print JSON `{snapshot_id, row_count}`.
- **FR-003 (Export CLI)**: Implement `finrl_pro.data.export_snapshot` CLI to materialize a snapshot to Parquet/CSV with checksum and output path.
- **FR-004 (Loader Resolution)**: Enhance `finrl_pro.data.loader` to resolve `dataset_hash` values of the form `snapshot://<snapshot_id>` by querying the database and returning a standardized DataFrame.
- **FR-005 (Integration)**: Trainer accepts `snapshot://<id>` unchanged and fingerprints include this value; README documents the snapshot workflow.
- **FR-006 (Provenance)**: Persist `code_hash` (git commit), `lib_versions` (e.g., `yfinance`, `pandas`), and `params_json` to ensure PIT reproducibility.
- **FR-007 (Secrets/Config)**: Read provider credentials from `conf/finrl_pro.env`; do not commit secrets; fail with clear messages if missing.
- **FR-008 (Resilience)**: Implement retries with backoff for transient fetch errors; log structured events for anomalies.
- **FR-009 (Performance)**: Batch inserts and use COPY where feasible; ensure reasonable throughput (e.g., ≥50k rows/min as a target on dev hardware).
- **FR-010 (Tests)**: Add unit tests for schema helpers and loader resolution; add integration tests that mock providers and validate snapshot→export→load flow.

### Key Entities *(include if feature involves data)*

- **market_bars**: PIT‑safe OHLCV time series per `(timestamp, ticker)` with vendor/source and ingest metadata.
- **snapshots**: Immutable records referencing a logical data slice, with full provenance and counts.
- **snapshot_assets**: Mapping of `snapshot_id` to constituent tickers.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Snapshot CLI completes with a valid `snapshot_id` and ≥ N rows inserted for a typical 2‑asset, 4‑year window.
- **SC-002**: Exported Parquet row count equals DB row count for the same `snapshot_id` (±0 tolerance).
- **SC-003**: Trainer runs using `snapshot://<id>` without code changes and records the same `snapshot_id` in fingerprints.
- **SC-004**: All tests pass locally (unit + integration), and the extension boundary guard remains green.

## Constitution Compliance Checklist *(mandatory)*

- Extension boundary respected? Yes — all new code under `finrl_pro/**`; no changes to `FinRLPodracer/**`.
- Reproducibility documented? Yes — `snapshot_id`, `code_hash`, `lib_versions`, and `params_json` captured; fingerprints reference `snapshot://<id>`.
- Risk controls defined? Yes — no change to risk policy; data anomalies logged; CLI exits non‑zero on empty results.
- Evaluation plan aligned? Yes — snapshots feed existing evaluation/reporting; exported files support offline workflows.
- Observability coverage? Yes — structured logging in CLIs and loader; MLflow tags may include `snapshot_id`.

