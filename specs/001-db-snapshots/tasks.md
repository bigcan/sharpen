# Tasks: Database-Backed Data Snapshots

Input: specs/001-db-snapshots/spec.md
Prerequisites: TimescaleDB/PostgreSQL available; provider creds in `conf/finrl_pro.env` (if Alpaca)

## Constitution Guards
- Extension boundary: FinRL Pro-only changes under `finrl_pro/**`, `specs/**`, `docs/**`, `tests/**`
- Reproducibility: Experiments reference `dataset_hash: snapshot://<snapshot_id>`

## Tasks (by requirement)

- [ ] T101 Schema helpers for TimescaleDB in `finrl_pro/data/db.py` (create hypertable `market_bars`, tables `snapshots`, `snapshot_assets`)
- [ ] T102 Upsert/batch insert utilities for OHLCV bars in `finrl_pro/data/db.py`
- [ ] T103 Snapshot CLI in `finrl_pro/data/snapshot.py` (yfinance/Alpaca fetch → DB upsert → insert into `snapshots`/`snapshot_assets` → print JSON)
- [ ] T104 Export CLI in `finrl_pro/data/export_snapshot.py` (DB → DataFrame → Parquet/CSV with checksum)
- [ ] T105 Loader resolution in `finrl_pro/data/loader.py` for `snapshot://<snapshot_id>`
- [ ] T106 Trainer integration (no change to interface; ensure fingerprints preserve snapshot URI)
- [ ] T107 README docs: snapshot workflow and commands
- [ ] T108 Unit tests: schema creation, resolver plumbing (mocks)
- [ ] T109 Integration test: snapshot → export → load flow (provider mocked)
- [ ] T110 Error handling: empty window, rate limits, retries, structured logs

## Execution Order
- Schema (T101) → Upsert (T102) → Snapshot CLI (T103)
- Export CLI (T104) can start after schema
- Loader (T105) and Trainer check (T106) after snapshot CLI
- Docs (T107) after CLI outline lands
- Tests (T108–T110) conclude the feature

