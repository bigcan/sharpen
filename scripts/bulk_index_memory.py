"""Bulk-index randd_log.md and stage3_research_plan.md into LanceDB Pro Memory.

Parses markdown into semantic chunks, batch-embeds via Gemini embedding-001,
and bulk-inserts into the shared GCS LanceDB table (gs://openclaw-memory-lance/v1).

Usage:
    python scripts/bulk_index_memory.py --dry-run
    python scripts/bulk_index_memory.py --randd randd_log.md --plan .agent/artifacts/stage3_research_plan.md
    python scripts/bulk_index_memory.py --force   # delete existing bulk rows, re-index
"""

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from socket import gethostname
from uuid import uuid4

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# --- Constants ---
LANCEDB_URI = "gs://openclaw-memory-lance/v1"
TABLE_NAME = "memories"
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 3072
GEMINI_BATCH_SIZE = 100
API_BASE = "https://generativelanguage.googleapis.com/v1beta"
SOURCE = "claude-code-bulk"
DEVICE = gethostname().lower()
SA_PATH = os.environ.get(
    "GOOGLE_SERVICE_ACCOUNT",
    str(Path.home() / ".openclaw" / "gcs-service-account.json"),
)

# --- Category / importance rules ---
DECISION_TAGS = {"decision", "architecture", "mdp", "redesign", "pivot"}
FACT_TAGS = {"experiment-result", "fail", "pass", "falsified", "result"}
BUGFIX_TAGS = {"bugfix", "fix", "critical", "bug-04", "bug-03", "bug-01"}
INFRA_TAGS = {"deploy", "infra", "monitoring", "cloud-sync"}

PLAN_DECISION_KEYWORDS = {"executive summary", "decision gate", "phase k", "phase l", "phase m", "phase r"}
PLAN_APPENDIX_KEYWORDS = {"appendix", "reference", "glossary"}


def get_gemini_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY env var required")
    return key


# --- Embedding ---
def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed texts via Gemini batchEmbedContents API (100 per call)."""
    key = get_gemini_key()
    url = f"{API_BASE}/models/{EMBED_MODEL}:batchEmbedContents?key={key}"
    all_vectors: list[list[float]] = []

    for i in range(0, len(texts), GEMINI_BATCH_SIZE):
        batch = texts[i : i + GEMINI_BATCH_SIZE]
        requests_body = [
            {"model": f"models/{EMBED_MODEL}", "content": {"parts": [{"text": t}]}}
            for t in batch
        ]
        resp = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={"requests": requests_body},
            timeout=60,
        )
        if not resp.ok:
            raise RuntimeError(f"Gemini batch embed failed ({resp.status_code}): {resp.text}")

        data = resp.json()
        for emb in data["embeddings"]:
            all_vectors.append(emb["values"])

        log.info("  Embedded batch %d-%d / %d", i + 1, i + len(batch), len(texts))
        if i + GEMINI_BATCH_SIZE < len(texts):
            time.sleep(0.2)

    return all_vectors


# --- Parsing ---
def parse_randd_log(path: str) -> list[dict]:
    """Split randd_log on date headers. Each session entry = 1 chunk."""
    text = Path(path).read_text(encoding="utf-8")
    # Split on ## YYYY-MM-DD headers
    pattern = re.compile(r"^(## \d{4}-\d{2}-\d{2} .+)$", re.MULTILINE)
    parts = pattern.split(text)

    chunks = []
    # parts[0] is the header/preamble, then alternating: header, body
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        chunk_text = f"{header}\n{body}".strip()

        # Extract tags
        tag_match = re.search(r">\s*tags?:\s*(.+)", chunk_text)
        tags = []
        if tag_match:
            tags = [t.strip().lower() for t in tag_match.group(1).split(",")]

        # Extract session number
        sess_match = re.search(r"Session\s+(\d+)", header)
        session = int(sess_match.group(1)) if sess_match else None

        # Extract date
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", header)
        entry_date = date_match.group(1) if date_match else None

        chunks.append({
            "text": chunk_text,
            "tags": tags,
            "session": session,
            "date": entry_date,
            "doc": "randd_log",
            "section_path": header,
        })

    return chunks


def parse_research_plan(path: str) -> list[dict]:
    """Hierarchical split: h2 → h3 → h4 if sections > 6K chars."""
    text = Path(path).read_text(encoding="utf-8")

    # Split at h2
    h2_pattern = re.compile(r"^(## .+)$", re.MULTILINE)
    h2_parts = h2_pattern.split(text)

    chunks = []
    # h2_parts[0] is preamble (title + metadata)
    preamble = h2_parts[0].strip()
    if preamble:
        chunks.append({
            "text": preamble,
            "tags": [],
            "session": None,
            "date": None,
            "doc": "stage3_research_plan",
            "section_path": "Preamble",
        })

    for i in range(1, len(h2_parts), 2):
        h2_header = h2_parts[i].strip()
        h2_body = h2_parts[i + 1] if i + 1 < len(h2_parts) else ""
        full_h2 = f"{h2_header}\n{h2_body}".strip()

        if len(full_h2) <= 6000:
            chunks.append({
                "text": full_h2,
                "tags": [],
                "session": None,
                "date": None,
                "doc": "stage3_research_plan",
                "section_path": h2_header,
            })
        else:
            # Sub-chunk at h3
            h3_pattern = re.compile(r"^(### .+)$", re.MULTILINE)
            h3_parts = h3_pattern.split(h2_body)

            # h3 preamble (text before first h3)
            h3_pre = h3_parts[0].strip()
            if h3_pre:
                chunks.append({
                    "text": f"{h2_header}\n{h3_pre}",
                    "tags": [],
                    "session": None,
                    "date": None,
                    "doc": "stage3_research_plan",
                    "section_path": f"{h2_header} > intro",
                })

            for j in range(1, len(h3_parts), 2):
                h3_header = h3_parts[j].strip()
                h3_body = h3_parts[j + 1] if j + 1 < len(h3_parts) else ""
                full_h3 = f"{h3_header}\n{h3_body}".strip()
                breadcrumb = f"{h2_header} > {h3_header}"

                if len(full_h3) <= 6000:
                    chunks.append({
                        "text": f"{h2_header}\n{full_h3}",
                        "tags": [],
                        "session": None,
                        "date": None,
                        "doc": "stage3_research_plan",
                        "section_path": breadcrumb,
                    })
                else:
                    # Sub-chunk at h4
                    h4_pattern = re.compile(r"^(#### .+)$", re.MULTILINE)
                    h4_parts = h4_pattern.split(h3_body)

                    h4_pre = h4_parts[0].strip()
                    if h4_pre:
                        chunks.append({
                            "text": f"{h2_header}\n{h3_header}\n{h4_pre}",
                            "tags": [],
                            "session": None,
                            "date": None,
                            "doc": "stage3_research_plan",
                            "section_path": f"{breadcrumb} > intro",
                        })

                    for k in range(1, len(h4_parts), 2):
                        h4_header = h4_parts[k].strip()
                        h4_body = h4_parts[k + 1] if k + 1 < len(h4_parts) else ""
                        full_h4 = f"{h4_header}\n{h4_body}".strip()
                        chunks.append({
                            "text": f"{h2_header}\n{h3_header}\n{full_h4}",
                            "tags": [],
                            "session": None,
                            "date": None,
                            "doc": "stage3_research_plan",
                            "section_path": f"{breadcrumb} > {h4_header}",
                        })

    return chunks


# --- Category / importance ---
def assign_category(entry: dict) -> str:
    tags = set(entry.get("tags", []))
    doc = entry.get("doc", "")
    section = entry.get("section_path", "").lower()

    if doc == "stage3_research_plan":
        if any(kw in section for kw in PLAN_DECISION_KEYWORDS):
            return "decision"
        return "other"

    if tags & DECISION_TAGS:
        return "decision"
    if tags & (FACT_TAGS | BUGFIX_TAGS):
        return "fact"
    return "other"


def assign_importance(entry: dict) -> float:
    tags = set(entry.get("tags", []))
    doc = entry.get("doc", "")
    section = entry.get("section_path", "").lower()

    if doc == "stage3_research_plan":
        if any(kw in section for kw in PLAN_DECISION_KEYWORDS):
            return 0.9
        if any(kw in section for kw in PLAN_APPENDIX_KEYWORDS):
            return 0.6
        return 0.7

    if tags & BUGFIX_TAGS:
        return 0.9
    if tags & DECISION_TAGS:
        return 0.9
    if tags & FACT_TAGS:
        return 0.8
    if tags & INFRA_TAGS:
        return 0.7
    return 0.6


def entry_to_epoch_ms(entry: dict) -> int:
    """Convert entry date to epoch ms, or use current time."""
    if entry.get("date"):
        dt = datetime.strptime(entry["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    return int(time.time() * 1000)


# --- LanceDB operations ---
def connect_db():
    import lancedb

    storage_options = {}
    if os.path.isfile(SA_PATH):
        storage_options["service_account"] = SA_PATH
        log.info("Using service account: %s", SA_PATH)
    else:
        log.warning("No service account file at %s — using default credentials", SA_PATH)

    db = lancedb.connect(LANCEDB_URI, storage_options=storage_options)
    return db


def get_table(db):
    try:
        return db.open_table(TABLE_NAME)
    except Exception:
        return None


def _dedup_key(chunk: dict) -> str:
    """Build dedup key from (date, session) or fall back to header prefix.

    Using date+session is robust to entry edits (corrections don't create
    duplicates).  For non-R&D chunks (research plan) we fall back to
    section_path which is stable across re-indexing.
    """
    date = chunk.get("date") or ""
    session = chunk.get("session")
    if date and session is not None:
        return f"{date}|S{session}"
    # Fallback for plan chunks or entries without session numbers
    section = chunk.get("section_path", "")
    return f"{chunk.get('doc', '')}|{section[:120]}"


def get_existing_keys(table, source: str) -> set[str]:
    """Get dedup keys for all existing bulk rows."""
    if table is None:
        return set()
    try:
        rows = table.search().where(f"source = '{source}'").select(["metadata", "text"]).limit(5000).to_list()
        keys: set[str] = set()
        for r in rows:
            meta = json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"]
            date = meta.get("date") or ""
            session = meta.get("session")
            section = meta.get("section_path", "")
            doc = meta.get("doc", "")
            # Migration: old rows lack 'date' in metadata — extract from text
            if not date and r.get("text"):
                m = re.search(r"^## (\d{4}-\d{2}-\d{2})", r["text"])
                if m:
                    date = m.group(1)
                if session is None:
                    sm = re.search(r"Session (\d+)", r["text"])
                    if sm:
                        session = int(sm.group(1))
            if date and session is not None:
                keys.add(f"{date}|S{session}")
            else:
                keys.add(f"{doc}|{section[:120]}")
        return keys
    except Exception as e:
        log.warning("Could not query existing rows: %s", e)
        return set()


def delete_bulk_rows(table, source: str) -> int:
    """Delete all rows with given source. Returns approximate count deleted."""
    if table is None:
        return 0
    try:
        before = table.count_rows()
        table.delete(f"source = '{source}'")
        after = table.count_rows()
        return before - after
    except Exception as e:
        log.warning("Could not delete bulk rows: %s", e)
        return 0


def bulk_insert(table, db, rows: list[dict], batch_size: int = 50):
    """Insert rows in batches."""
    if table is None:
        # Create table with first batch
        table = db.create_table(TABLE_NAME, rows[:batch_size])
        log.info("  Created table with %d rows", min(batch_size, len(rows)))
        remaining = rows[batch_size:]
    else:
        remaining = rows

    for i in range(0, len(remaining), batch_size):
        batch = remaining[i : i + batch_size]
        table.add(batch)
        log.info("  Inserted batch %d-%d / %d", i + 1, i + len(batch), len(rows))

    return table


# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Bulk-index markdown into LanceDB memory")
    parser.add_argument("--randd", type=str, default=None, help="Path to randd_log.md")
    parser.add_argument("--plan", type=str, default=None, help="Path to stage3_research_plan.md")
    parser.add_argument("--archive-dir", type=str, default=None, help="Path to randd_archive/ directory")
    parser.add_argument("--dry-run", action="store_true", help="Parse only, no embed/insert")
    parser.add_argument("--force", action="store_true", help="Delete existing bulk rows first")
    args = parser.parse_args()

    # Default paths
    project_root = Path(__file__).resolve().parent.parent
    if args.randd is None:
        args.randd = str(project_root / "randd_log.md")
    if args.plan is None:
        args.plan = str(project_root / ".agent" / "artifacts" / "stage3_research_plan.md")
    archive_dir = Path(args.archive_dir) if args.archive_dir else project_root / "randd_archive"

    # Parse
    all_chunks: list[dict] = []

    if Path(args.randd).exists():
        randd_chunks = parse_randd_log(args.randd)
        log.info("Parsed randd_log: %d entries", len(randd_chunks))
        all_chunks.extend(randd_chunks)
    else:
        log.warning("randd_log not found: %s", args.randd)

    # Auto-discover archive files
    if archive_dir.exists():
        for archive_file in sorted(archive_dir.glob("*.md")):
            archive_chunks = parse_randd_log(str(archive_file))
            for chunk in archive_chunks:
                chunk["doc"] = f"randd_archive/{archive_file.name}"
            log.info("Parsed archive %s: %d entries", archive_file.name, len(archive_chunks))
            all_chunks.extend(archive_chunks)
    else:
        log.info("No archive directory found at %s (skipping)", archive_dir)

    if Path(args.plan).exists():
        plan_chunks = parse_research_plan(args.plan)
        log.info("Parsed research plan: %d chunks", len(plan_chunks))
        all_chunks.extend(plan_chunks)
    else:
        log.warning("Research plan not found: %s", args.plan)

    if not all_chunks:
        log.error("No chunks to index")
        return

    log.info("Total chunks: %d", len(all_chunks))

    # Show size stats
    sizes = [len(c["text"]) for c in all_chunks]
    log.info("Chunk sizes: min=%d, max=%d, avg=%d chars", min(sizes), max(sizes), int(sum(sizes) / len(sizes)))

    if args.dry_run:
        log.info("=== DRY RUN — no embedding or insertion ===")
        for i, c in enumerate(all_chunks):
            cat = assign_category(c)
            imp = assign_importance(c)
            log.info("  [%3d] %s | cat=%s imp=%.1f | %d chars | %s",
                     i, c["doc"], cat, imp, len(c["text"]), c["section_path"][:80])
        return

    # Connect to LanceDB
    log.info("Connecting to LanceDB: %s", LANCEDB_URI)
    db = connect_db()
    table = get_table(db)

    if table is not None:
        log.info("Table '%s' exists: %d rows", TABLE_NAME, table.count_rows())
    else:
        log.info("Table '%s' does not exist — will create", TABLE_NAME)

    # Force delete
    if args.force and table is not None:
        deleted = delete_bulk_rows(table, SOURCE)
        log.info("Deleted %d existing bulk rows (--force)", deleted)

    # Dedup check
    existing = get_existing_keys(table, SOURCE)
    new_chunks = [c for c in all_chunks if _dedup_key(c) not in existing]
    log.info("New chunks to index: %d (skipping %d existing)", len(new_chunks), len(all_chunks) - len(new_chunks))

    if not new_chunks:
        log.info("Nothing to do — all chunks already indexed")
        return

    # Embed
    log.info("Embedding %d chunks via Gemini %s...", len(new_chunks), EMBED_MODEL)
    texts = [c["text"] for c in new_chunks]
    vectors = embed_batch(texts)
    assert len(vectors) == len(new_chunks), f"Vector count mismatch: {len(vectors)} vs {len(new_chunks)}"

    # Build rows
    rows = []
    for chunk, vector in zip(new_chunks, vectors):
        rows.append({
            "id": str(uuid4()),
            "text": chunk["text"],
            "vector": vector,
            "importance": assign_importance(chunk),
            "category": assign_category(chunk),
            "source": SOURCE,
            "device": DEVICE,
            "createdAt": entry_to_epoch_ms(chunk),
            "timestamp": None,
            "scope": None,
            "metadata": json.dumps({
                "doc": chunk["doc"],
                "date": chunk.get("date"),
                "session": chunk.get("session"),
                "tags": chunk.get("tags", []),
                "section_path": chunk.get("section_path", ""),
            }),
        })

    # Insert
    log.info("Inserting %d rows into LanceDB...", len(rows))
    table = bulk_insert(table, db, rows)

    final_count = table.count_rows()
    log.info("Done! Table '%s' now has %d rows", TABLE_NAME, final_count)


if __name__ == "__main__":
    main()
