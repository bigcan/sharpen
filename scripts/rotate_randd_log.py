"""Rotate old entries from randd_log.md into monthly archive files.

Keeps the last N months (default 1 = current month) in randd_log.md and
moves older entries to randd_archive/YYYY-MM.md.  Grep still works across
all files; the write path (/sync appends to randd_log.md) is unchanged.

Usage:
    python scripts/rotate_randd_log.py --dry-run
    python scripts/rotate_randd_log.py --keep-months 1
    python scripts/rotate_randd_log.py --keep-months 2 --randd randd_log.md
"""

import argparse
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Same regex used by bulk_index_memory.py
ENTRY_HEADER_RE = re.compile(r"^(## \d{4}-\d{2}-\d{2} .+)$", re.MULTILINE)
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

RANDD_HEADER = """\
# FinRL-Pro_DS Research & Development Log

**Purpose:** Rolling write buffer for R&D history.  Older entries are archived
to `randd_archive/YYYY-MM.md` via `scripts/rotate_randd_log.py`.
**Sorting Protocol:** Reverse chronological order.  New entries go at the TOP,
immediately after this header.  Run `rotate_randd_log.py` when the file exceeds ~300 KB.

---
"""

ARCHIVE_HEADER_TEMPLATE = """\
# R&D Log Archive — {month}
> Archived from randd_log.md. Entries in reverse chronological order.
---
"""


def parse_entries(text: str) -> tuple[str, list[dict]]:
    """Split text on ## YYYY-MM-DD headers, returning (preamble, entries).

    Each entry dict: {header, body, date_str, month}.
    """
    parts = ENTRY_HEADER_RE.split(text)
    preamble = parts[0]

    entries: list[dict] = []
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""

        date_match = DATE_RE.search(header)
        if date_match:
            date_str = date_match.group(1)
            month = date_str[:7]  # YYYY-MM
        else:
            date_str = "9999-99-99"
            month = "9999-99"

        entries.append({
            "header": header,
            "body": body,
            "date_str": date_str,
            "month": month,
        })

    return preamble, entries


def sort_entries_desc(entries: list[dict]) -> list[dict]:
    """Sort entries by date descending (newest first), stable within same date."""
    return sorted(entries, key=lambda e: e["date_str"], reverse=True)


def entry_to_text(entry: dict) -> str:
    """Reconstruct markdown text for a single entry."""
    body = entry["body"]
    # Ensure body ends with separator
    body_stripped = body.rstrip()
    if not body_stripped.endswith("---"):
        body_stripped += "\n\n---"
    return f"{entry['header']}\n{body_stripped}"


def months_to_keep(keep_months: int) -> set[str]:
    """Return set of YYYY-MM strings for the last N months including current."""
    now = datetime.now(timezone.utc)
    result = set()
    year, month = now.year, now.month
    for _ in range(keep_months):
        result.add(f"{year:04d}-{month:02d}")
        month -= 1
        if month < 1:
            month = 12
            year -= 1
    return result


def load_existing_archive(archive_path: Path) -> list[dict]:
    """Load entries from an existing archive file, for merge-dedup."""
    if not archive_path.exists():
        return []
    text = archive_path.read_text(encoding="utf-8")
    _, entries = parse_entries(text)
    return entries


def dedup_entries(entries: list[dict]) -> list[dict]:
    """Remove duplicate entries (same header string)."""
    seen: set[str] = set()
    unique: list[dict] = []
    for e in entries:
        if e["header"] not in seen:
            seen.add(e["header"])
            unique.append(e)
    return unique


def main():
    parser = argparse.ArgumentParser(
        description="Rotate old randd_log.md entries into monthly archive files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files",
    )
    parser.add_argument(
        "--keep-months",
        type=int,
        default=1,
        help="Number of recent months to keep in randd_log.md (default: 1 = current month only)",
    )
    parser.add_argument(
        "--randd",
        type=str,
        default=None,
        help="Path to randd_log.md (default: project root)",
    )
    parser.add_argument(
        "--archive-dir",
        type=str,
        default=None,
        help="Path to archive directory (default: randd_archive/ next to randd_log.md)",
    )
    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).resolve().parent.parent
    randd_path = Path(args.randd) if args.randd else project_root / "randd_log.md"
    archive_dir = Path(args.archive_dir) if args.archive_dir else randd_path.parent / "randd_archive"

    if not randd_path.exists():
        log.error("randd_log.md not found: %s", randd_path)
        return 1

    # Parse
    text = randd_path.read_text(encoding="utf-8")
    original_size = len(text.encode("utf-8"))
    _, all_entries = parse_entries(text)
    log.info("Parsed %d entries from %s (%.1f KB)", len(all_entries), randd_path.name, original_size / 1024)

    # Strip orphan lines that aren't part of any entry
    # (e.g., stray timestamp lines at the very end)
    # These are already excluded by parse_entries since they don't match ## headers

    # Determine which months to keep vs archive
    keep = months_to_keep(args.keep_months)
    log.info("Keeping months: %s", sorted(keep, reverse=True))

    # Group entries
    keep_entries: list[dict] = []
    archive_groups: dict[str, list[dict]] = defaultdict(list)

    for entry in all_entries:
        if entry["month"] in keep:
            keep_entries.append(entry)
        else:
            archive_groups[entry["month"]].append(entry)

    if not archive_groups:
        log.info("Nothing to archive — all %d entries are within the keep window", len(keep_entries))
        return 0

    # Sort kept entries by date desc
    keep_entries = sort_entries_desc(keep_entries)

    # Report
    total_archived = sum(len(v) for v in archive_groups.values())
    log.info("Entries to keep in randd_log.md: %d", len(keep_entries))
    log.info("Entries to archive: %d across %d months", total_archived, len(archive_groups))
    for month in sorted(archive_groups.keys()):
        entries = archive_groups[month]
        log.info("  %s: %d entries", month, len(entries))

    if args.dry_run:
        log.info("=== DRY RUN — no files written ===")
        log.info("Would create/update archive files in: %s", archive_dir)
        log.info("Would rewrite randd_log.md with %d entries", len(keep_entries))
        new_size_est = len(RANDD_HEADER.encode("utf-8"))
        for e in keep_entries:
            new_size_est += len(entry_to_text(e).encode("utf-8")) + 2
        log.info("Estimated new randd_log.md size: %.1f KB (was %.1f KB, ~%.0f%% reduction)",
                 new_size_est / 1024, original_size / 1024,
                 (1 - new_size_est / original_size) * 100)
        return 0

    # Create archive directory
    archive_dir.mkdir(parents=True, exist_ok=True)
    log.info("Archive directory: %s", archive_dir)

    # Write archive files (merge-dedup with existing)
    for month in sorted(archive_groups.keys()):
        archive_path = archive_dir / f"{month}.md"
        new_entries = archive_groups[month]

        # Merge with existing archive
        existing = load_existing_archive(archive_path)
        merged = dedup_entries(existing + new_entries)
        merged = sort_entries_desc(merged)

        # Build archive content
        content = ARCHIVE_HEADER_TEMPLATE.format(month=month) + "\n"
        for entry in merged:
            content += entry_to_text(entry) + "\n\n"

        archive_path.write_text(content.rstrip() + "\n", encoding="utf-8")
        archive_size = len(content.encode("utf-8"))
        log.info("Wrote %s: %d entries (%.1f KB)", archive_path.name, len(merged), archive_size / 1024)

    # Rewrite randd_log.md with only kept entries
    new_content = RANDD_HEADER + "\n"
    for entry in keep_entries:
        new_content += entry_to_text(entry) + "\n\n"

    randd_path.write_text(new_content.rstrip() + "\n", encoding="utf-8")
    new_size = len(new_content.encode("utf-8"))
    log.info("Rewrote %s: %d entries (%.1f KB, was %.1f KB, %.0f%% reduction)",
             randd_path.name, len(keep_entries), new_size / 1024,
             original_size / 1024, (1 - new_size / original_size) * 100)

    # Verify entry count
    total_after = len(keep_entries) + total_archived
    log.info("Entry count check: %d kept + %d archived = %d total (original: %d)",
             len(keep_entries), total_archived, total_after, len(all_entries))
    if total_after != len(all_entries):
        log.warning("ENTRY COUNT MISMATCH — expected %d, got %d", len(all_entries), total_after)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
