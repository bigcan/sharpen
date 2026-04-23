"""Protocol v2.2 §8.3 kill_file JSON format + repeat-CRIT lockout.

Legacy kill_file convention is "file exists = stop". v2.2 promotes the file
to structured JSON so the watchdog / engine-startup gate can distinguish
operator-issued halts (manual) from drift-CRIT halts (auto-engine) and
count repeat-CRITs within a 24-hour window (spec: ≥ 2 within 24h → lockout
until `kill_file.override` is written).

The file is written once per CRIT event; subsequent CRITs within the
window increment `count` and prepend to `history`, preserving the first
timestamp for lockout math. An empty file or a file that fails to parse
is treated as a legacy halt (reason=LEGACY) so old operator scripts still
stop the engine.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Well-known reasons — string-typed for forward compat.
REASON_DRIFT_CRIT = "drift_crit"
REASON_OPERATOR = "operator"
REASON_LEGACY = "legacy"

CRIT_LOCKOUT_WINDOW_HOURS = 24
CRIT_LOCKOUT_COUNT = 2


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_kill_file(path: Path) -> Optional[dict]:
    """Return parsed kill_file payload or None if file missing.

    Unparseable / empty files return a legacy marker so the engine still
    treats the file as a stop signal without losing restart semantics.
    """
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning(f"kill_file read failed ({path}): {e}")
        return {"reason": REASON_LEGACY, "note": f"unreadable: {e}"}
    raw = raw.strip()
    if not raw:
        return {"reason": REASON_LEGACY, "note": "empty file"}
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return {"reason": REASON_LEGACY, "note": "non-dict payload"}
        return payload
    except json.JSONDecodeError:
        return {"reason": REASON_LEGACY, "note": "non-JSON body"}


def write_kill_file(
    path: Path,
    *,
    reason: str,
    detail: str = "",
    extra: Optional[dict] = None,
) -> dict:
    """Write or update a kill_file JSON payload.

    If a CRIT file already exists with the same `reason`, increments `count`
    and prepends the new event to `history`. Otherwise writes a fresh file.
    Returns the final payload.

    The directory is created if missing. Permission errors bubble up: this
    is a safety-critical write and silent failure is worse than a crash.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    now = _now_utc_iso()
    existing = read_kill_file(path) if path.exists() else None
    if existing and existing.get("reason") == reason:
        payload = dict(existing)
        payload["count"] = int(payload.get("count", 1)) + 1
        payload["last_ts"] = now
        history = list(payload.get("history", []))
        history.insert(0, {"ts": now, "detail": detail})
        payload["history"] = history[:20]  # bounded
    else:
        payload = {
            "reason": reason,
            "first_ts": now,
            "last_ts": now,
            "count": 1,
            "detail": detail,
            "history": [{"ts": now, "detail": detail}],
        }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def should_lockout(
    payload: dict,
    override_path: Optional[Path],
    *,
    window_hours: int = CRIT_LOCKOUT_WINDOW_HOURS,
    count_threshold: int = CRIT_LOCKOUT_COUNT,
) -> tuple[bool, str]:
    """Decide whether a startup/engine-restart should be refused.

    Lockout semantics (Protocol v2.2 §8.3):
      * Any CRIT reason → refuse startup. Manual re-enable = delete kill_file.
      * ≥ `count_threshold` CRITs in `window_hours` → refuse startup *even if*
        operator is trying to re-enable. Require `kill_file.override` file to
        lift. Prevents restart thrash on genuine regime breaks.

    Returns `(locked, reason)`. An operator-issued (`REASON_OPERATOR`) or
    legacy halt falls into the single-CRIT bucket — operator must clear the
    file. A present `override_path` clears the repeat-CRIT branch but NOT
    the general presence branch (i.e. override is a stronger re-enable, but
    the kill_file itself still has to go).
    """
    reason = payload.get("reason", REASON_LEGACY)
    count = int(payload.get("count", 1))
    first_ts_raw = payload.get("first_ts")

    if override_path is not None and override_path.exists():
        return False, f"override file present ({override_path})"

    if reason == REASON_DRIFT_CRIT and count >= count_threshold:
        try:
            first_ts = datetime.fromisoformat(first_ts_raw) if first_ts_raw else None
        except ValueError:
            first_ts = None
        now = datetime.now(timezone.utc)
        if first_ts is not None:
            if first_ts.tzinfo is None:
                first_ts = first_ts.replace(tzinfo=timezone.utc)
            window = now - first_ts
            if window <= timedelta(hours=window_hours):
                return True, (
                    f"REPEAT-CRIT LOCKOUT: drift_crit count={count} "
                    f"within {window.total_seconds()/3600:.1f}h "
                    f"(threshold {count_threshold} in {window_hours}h). "
                    f"Clear {override_path} absent — engine refuses restart "
                    f"until human override."
                )

    # Any existing kill_file halts startup unless cleared. This mirrors
    # legacy "file exists = stop" semantics.
    return True, f"kill_file present (reason={reason}, count={count})"


def clear_kill_file(path: Path) -> None:
    """Operator re-enable path — delete the kill_file if it exists."""
    try:
        path.unlink(missing_ok=True)
    except OSError as e:
        logger.warning(f"kill_file clear failed ({path}): {e}")
