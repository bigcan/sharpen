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
      * Any kill_file present → refuse startup. Manual re-enable requires
        deleting the kill_file (operator is acknowledging the halt).
      * Repeat-CRIT (drift_crit reason, count ≥ threshold within window) →
        refuse startup even after operator deletes the kill_file, UNTIL
        `override_path` is ALSO written. The override is a narrow-purpose
        human-approval signal scoped to repeat-CRIT; it does NOT lift a
        single-CRIT or operator halt on its own (those just need the
        kill_file cleared).

    Returns `(locked, reason)`.
    """
    reason = payload.get("reason", REASON_LEGACY)
    count = int(payload.get("count", 1))
    first_ts_raw = payload.get("first_ts")

    override_present = override_path is not None and override_path.exists()

    is_repeat_crit = False
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
                is_repeat_crit = True
                if not override_present:
                    return True, (
                        f"REPEAT-CRIT LOCKOUT: drift_crit count={count} "
                        f"within {window.total_seconds()/3600:.1f}h "
                        f"(threshold {count_threshold} in {window_hours}h). "
                        f"override {override_path} absent — engine refuses "
                        f"restart until human override."
                    )

    # Single-CRIT / operator / legacy halt: kill_file presence alone locks
    # out. Override does NOT substitute for clearing the kill_file here —
    # that distinction is what preserves the "operator acknowledges halt"
    # semantic. Only the repeat-CRIT branch above is allowed to be lifted
    # by an override.
    if is_repeat_crit and override_present:
        # Repeat-CRIT window + operator has written override → still halt
        # until kill_file itself is cleared, but with a clearer diagnostic.
        return True, (
            f"kill_file present (reason={reason}, count={count}); "
            f"override seen — operator must also delete kill_file to restart"
        )
    return True, f"kill_file present (reason={reason}, count={count})"


def clear_kill_file(path: Path) -> None:
    """Operator re-enable path — delete the kill_file if it exists."""
    try:
        path.unlink(missing_ok=True)
    except OSError as e:
        logger.warning(f"kill_file clear failed ({path}): {e}")
