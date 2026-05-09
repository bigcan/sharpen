#!/usr/bin/env python3
# Mechanical pre-filter for /sync Step 3.5 reconciliation.
# Replaces inline grep-and-read prose with deterministic Python.
# Authoritative contract: ~/.claude/commands/sync.md.
# Fixture: .agent/artifacts/sync_reconciliation_false_positive_fixture_20260502.md.

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

DEFAULT_MEMORY_DIR = Path.home() / ".claude" / "projects" / "C--FinRL-FinRL-Pro-DS" / "memory"

# Captured 2026-05-02 (S517) at commit eb23c593. Hard regression gate.
FIXTURE_FILES = frozenset({
    "decision_prop_firm_decoupling_s495.md",
    "project_binance_demo_data_quality_s502.md",
    "project_alphaseek_v3_plan_handoff.md",
    "project_step5_handoff_s497_to_s498.md",
})

STALE_PHRASE = re.compile(
    r"\b(pending|open|blocked on|TBD|gated on|next session|not yet|untouched|"
    r"still on (?:old |prior )|pending operator|awaiting operator|"
    r"before live swap|before live capital)\b|\(NOT done",
    re.IGNORECASE,
)

RESOLVED_MARKER = re.compile(
    r"(✅|\bSHIPPED\b|\bRESOLVED\b|\bDONE\b|\bLIVE\b|"
    r"patched\s+\d{4}-\d{2}-\d{2}|\bMERGED\b|\bDEPLOYED\b|"
    r"\bCOMPLETED\b|\bclosed\b|\bsuperseded\b)",
    re.IGNORECASE,
)

HANDOFF_NAME = re.compile(
    r"handoff\s*\(\s*S\d+[\w\-]*\s*[→\-]\s*S\d+",
    re.IGNORECASE,
)

# Conjugations + commit-prefix verbs to exclude from token-overlap matching.
COMMON_TOKENS = frozenset({
    "fix", "fixed", "fixes", "ship", "shipped", "ships", "docs", "doc",
    "feat", "test", "tests", "chore", "refactor", "audit", "memory",
    "session", "sync", "with", "from", "into", "this", "that", "which",
    "have", "been", "after", "before", "added", "update", "updated",
    "merge", "merged", "fixme", "todo",
})

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{3,}")
FRONTMATTER_END = re.compile(r"^---\s*$", re.MULTILINE)


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in COMMON_TOKENS}


def _body_start_line(text: str) -> int:
    # Returns the first line index (1-based) that is part of the body, after a
    # closing frontmatter `---`. If no frontmatter, returns 1.
    if not text.startswith("---"):
        return 1
    matches = list(FRONTMATTER_END.finditer(text))
    if len(matches) < 2:
        return 1
    closing = matches[1]
    return text.count("\n", 0, closing.end()) + 2


def _frontmatter_field(text: str, field: str) -> str:
    m = re.search(rf"^{field}:\s*(.+)$", text[:2000], re.MULTILINE)
    return m.group(1).strip() if m else ""


def _is_handoff(text: str, filename: str) -> bool:
    name = _frontmatter_field(text, "name")
    return bool(HANDOFF_NAME.search(name)) or "handoff" in filename.lower()


def _resolution_in_body(text: str, body_start: int) -> bool:
    lines = text.splitlines()
    for i in range(body_start - 1, len(lines)):
        if RESOLVED_MARKER.search(lines[i]):
            return True
    return False


def _nearby_resolution(text: str, line_no: int, window: int = 15) -> tuple[int, str] | None:
    lines = text.splitlines()
    n = len(lines)
    # Cap window to the file's actual span; prevents wasted iterations when
    # callers pass a sentinel like 10**6 to mean "scan the whole file."
    max_offset = min(window, n)
    for offset in range(max_offset + 1):
        for direction in (-1, 1):
            i = (line_no - 1) + direction * offset
            if 0 <= i < n and RESOLVED_MARKER.search(lines[i]):
                return (i + 1, lines[i].rstrip())
    return None


def _scan_file(path: Path, text: str) -> tuple[list[dict], str]:
    # Returns (candidate-claims, skip-reason). skip-reason non-empty => whole file skipped.
    if _is_handoff(text, path.name):
        return [], "handoff (historical-by-construction)"

    body_start = _body_start_line(text)
    has_body_resolution = _resolution_in_body(text, body_start)
    lines = text.splitlines()

    out: list[dict] = []
    for i, line in enumerate(lines, start=1):
        if not STALE_PHRASE.search(line):
            continue
        # Skip if same line declares its own resolution.
        if RESOLVED_MARKER.search(line):
            continue
        # Frontmatter framing is treated as historical when body has any resolution marker.
        in_frontmatter = i < body_start
        if in_frontmatter and has_body_resolution:
            continue
        # ±15 line window resolution marker => assume nearby resolution.
        if _nearby_resolution(text, i, window=15):
            continue
        # Survivor: attach the closest resolution marker (or any in body) for evidence.
        evidence = _nearby_resolution(text, i, window=10**6)  # whole-file scan
        out.append({
            "claim_line": i,
            "claim": line.rstrip()[:200],
            "evidence_line": evidence[0] if evidence else None,
            "evidence_text": evidence[1][:200] if evidence else None,
        })
    return out, ""


def _select_in_scope(memory_dir: Path, commit_msg: str, age_days: float) -> list[Path]:
    cutoff = time.time() - age_days * 86400
    commit_tokens = _tokenize(commit_msg) if commit_msg else set()
    files: list[Path] = []
    for f in memory_dir.glob("*.md"):
        if f.name == "MEMORY.md":
            continue
        try:
            mtime = f.stat().st_mtime
        except OSError:
            continue
        if mtime >= cutoff:
            files.append(f)
            continue
        if not commit_tokens:
            continue
        slug_tokens = _tokenize(f.stem.replace("_", " ").replace("-", " "))
        if slug_tokens & commit_tokens:
            files.append(f)
            continue
        head = _read(f)[:1500]
        name_tokens = _tokenize(_frontmatter_field(head, "name"))
        if name_tokens & commit_tokens:
            files.append(f)
    return sorted(files, key=lambda p: p.name)


def _check_index(memory_dir: Path) -> list[dict]:
    idx = memory_dir / "MEMORY.md"
    if not idx.exists():
        return []
    flags: list[dict] = []
    line_re = re.compile(r"\s*-\s*\[([^\]]+)\]\(([^)]+\.md)\)\s*[—\-]\s*(.+)$")
    for ln, line in enumerate(_read(idx).splitlines(), start=1):
        m = line_re.match(line)
        if not m:
            continue
        title, link, hook = m.group(1), m.group(2), m.group(3)
        target = memory_dir / link
        if not target.exists():
            flags.append({"line": ln, "title": title, "issue": "linked file missing"})
            continue
        desc = _frontmatter_field(_read(target), "description").lower()
        if not desc:
            continue
        hook_words = {
            w.lower() for w in TOKEN_RE.findall(hook)
            if w.lower() not in COMMON_TOKENS and len(w) >= 5
        }
        if not hook_words:
            continue
        missing = [w for w in hook_words if w not in desc]
        # Strict gate: only flag when nearly every salient hook word is missing AND
        # there are enough hook words to be meaningful. Hooks naturally carry
        # command snippets / tickers / numbers that legitimately don't appear in
        # the canonical description, so a relaxed threshold drowns the report.
        if len(hook_words) >= 4 and len(missing) >= max(4, int(len(hook_words) * 0.85)):
            flags.append({
                "line": ln,
                "title": title,
                "issue": f"description missing {len(missing)}/{len(hook_words)} hook words: "
                         + ", ".join(sorted(missing)[:4]),
            })
    return flags


def _force_utf8_stdout() -> None:
    # Defensive: Python on Windows defaults stdout to the console codepage
    # (often cp950 in this project's environment), which crashes on `—`, `→`,
    # `✅`, etc. Reconfigure with errors='replace' so we never crash mid-report.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def main() -> int:
    _force_utf8_stdout()
    ap = argparse.ArgumentParser(description="/sync Step 3.5 mechanical reconciliation pre-filter.")
    ap.add_argument("--memory-dir", type=Path, default=DEFAULT_MEMORY_DIR)
    ap.add_argument("--commit-msg", default="",
                    help="Draft commit subject+body (used for token-overlap filter).")
    ap.add_argument("--age-days", type=float, default=3.0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--check-index", action="store_true",
                    help="Cross-reference MEMORY.md hook words vs. linked file descriptions "
                         "(off by default; noisy on hooks with command snippets/tickers).")
    ap.add_argument("--no-fixture-gate", action="store_true",
                    help="Disable the 4-file fixture regression gate (debug only).")
    args = ap.parse_args()

    if not args.memory_dir.is_dir():
        print(f"[scan] memory dir not found: {args.memory_dir}", file=sys.stderr)
        return 1

    in_scope = _select_in_scope(args.memory_dir, args.commit_msg, args.age_days)

    candidates: list[dict] = []
    skipped_handoff = 0
    for path in in_scope:
        text = _read(path)
        claims, skip_reason = _scan_file(path, text)
        if skip_reason:
            if "handoff" in skip_reason:
                skipped_handoff += 1
            continue
        for c in claims:
            candidates.append({"file": path.name, **c})

    flagged_fixture = sorted({c["file"] for c in candidates if c["file"] in FIXTURE_FILES})
    fixture_status = "PASS" if not flagged_fixture else "FAIL"
    if flagged_fixture and not args.no_fixture_gate:
        msg = (f"FIXTURE REGRESSION — {flagged_fixture} should never be flagged. "
               f"See .agent/artifacts/sync_reconciliation_false_positive_fixture_20260502.md")
        if args.json:
            print(json.dumps({"error": msg, "flagged_fixture": flagged_fixture}, indent=2))
        else:
            print(f"[scan] {msg}", file=sys.stderr)
        return 2

    index_flags = _check_index(args.memory_dir) if args.check_index else []

    if args.json:
        print(json.dumps({
            "scope_files": len(in_scope),
            "skipped_handoff_files": skipped_handoff,
            "candidates": candidates,
            "stale_index_entries": index_flags,
            "fixture_check": fixture_status,
        }, indent=2))
        return 0

    out: list[str] = []
    out.append("[/sync reconciliation scan]")
    out.append(f"scope: {len(in_scope)} files (mtime <= {args.age_days}d or token-overlap)"
               f" | handoffs skipped: {skipped_handoff} | fixture gate: {fixture_status}")
    out.append(f"candidates: {len(candidates)}"
               f" | index drift entries: {len(index_flags)}")
    out.append("")
    if not candidates and not index_flags:
        out.append("No flags.")
    for i, c in enumerate(candidates, start=1):
        out.append(f"{i}. {c['file']}:{c['claim_line']}")
        out.append(f"   claim: {c['claim'].strip()!r}")
        if c["evidence_line"]:
            out.append(f"   nearest in-body resolution: line {c['evidence_line']} -> "
                       f"{c['evidence_text'].strip()!r}")
            out.append("   -> verify before editing; likely real stale claim.")
        else:
            out.append("   no in-body resolution found anywhere.")
            out.append("   -> VERIFY by reading file before deciding.")
    if index_flags:
        out.append("")
        out.append("MEMORY.md index drift:")
        for f in index_flags[:20]:
            out.append(f"  line {f['line']} - {f['title']}: {f['issue']}")
        if len(index_flags) > 20:
            out.append(f"  ... ({len(index_flags) - 20} more)")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
