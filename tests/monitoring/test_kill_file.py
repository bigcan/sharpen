"""Tests for Protocol v2.2 §8.3 kill_file JSON + lockout state machine."""

from __future__ import annotations

from pathlib import Path

from finrl_pro_ds.monitoring.kill_file import (
    REASON_DRIFT_CRIT,
    REASON_LEGACY,
    REASON_OPERATOR,
    clear_kill_file,
    read_kill_file,
    should_lockout,
    write_kill_file,
)


# ---------- read_kill_file ---------------------------------------------------


def test_read_missing_file_returns_none(tmp_path: Path):
    assert read_kill_file(tmp_path / "nope") is None


def test_read_empty_file_is_legacy(tmp_path: Path):
    p = tmp_path / "kf"
    p.write_text("")
    payload = read_kill_file(p)
    assert payload["reason"] == REASON_LEGACY


def test_read_non_json_is_legacy(tmp_path: Path):
    p = tmp_path / "kf"
    p.write_text("plain text halt signal")
    payload = read_kill_file(p)
    assert payload["reason"] == REASON_LEGACY


def test_read_non_dict_json_is_legacy(tmp_path: Path):
    p = tmp_path / "kf"
    p.write_text('"just a string"')
    payload = read_kill_file(p)
    assert payload["reason"] == REASON_LEGACY


# ---------- write_kill_file --------------------------------------------------


def test_write_creates_fresh_payload(tmp_path: Path):
    p = tmp_path / "kf"
    payload = write_kill_file(p, reason=REASON_DRIFT_CRIT, detail="kl=1.2")
    assert payload["reason"] == REASON_DRIFT_CRIT
    assert payload["count"] == 1
    assert payload["detail"] == "kl=1.2"
    assert len(payload["history"]) == 1
    assert p.exists()


def test_write_increments_count_on_same_reason(tmp_path: Path):
    p = tmp_path / "kf"
    write_kill_file(p, reason=REASON_DRIFT_CRIT, detail="first")
    payload = write_kill_file(p, reason=REASON_DRIFT_CRIT, detail="second")
    assert payload["count"] == 2
    assert len(payload["history"]) == 2
    # Most recent first
    assert payload["history"][0]["detail"] == "second"


def test_write_different_reason_resets(tmp_path: Path):
    p = tmp_path / "kf"
    write_kill_file(p, reason=REASON_OPERATOR, detail="manual halt")
    payload = write_kill_file(p, reason=REASON_DRIFT_CRIT, detail="auto halt")
    # Different reason → fresh payload
    assert payload["count"] == 1
    assert payload["reason"] == REASON_DRIFT_CRIT


def test_write_bounds_history_at_20(tmp_path: Path):
    p = tmp_path / "kf"
    for i in range(25):
        write_kill_file(p, reason=REASON_DRIFT_CRIT, detail=f"event {i}")
    payload = read_kill_file(p)
    assert len(payload["history"]) == 20
    assert payload["count"] == 25


def test_write_extra_merges_into_payload(tmp_path: Path):
    p = tmp_path / "kf"
    payload = write_kill_file(
        p, reason=REASON_DRIFT_CRIT, detail="kl=1.5",
        extra={"kl": 1.5, "bucket": "q3"},
    )
    assert payload["kl"] == 1.5
    assert payload["bucket"] == "q3"


def test_write_creates_parent_dir(tmp_path: Path):
    p = tmp_path / "nested" / "dir" / "kf"
    write_kill_file(p, reason=REASON_DRIFT_CRIT)
    assert p.exists()


# ---------- should_lockout ---------------------------------------------------


def test_no_payload_means_no_lockout():
    # Caller is expected to short-circuit when read_kill_file returns None,
    # but passing an empty dict should default to locked (unknown reason).
    locked, _ = should_lockout({}, None)
    assert locked is True


def test_legacy_kill_file_locks_out(tmp_path: Path):
    p = tmp_path / "kf"
    p.write_text("")
    payload = read_kill_file(p)
    locked, reason = should_lockout(payload, None)
    assert locked is True
    assert "legacy" in reason


def test_single_crit_locks_out_even_with_override(tmp_path: Path):
    """KILL-FILE-OVERRIDE-01 fix: override ONLY lifts repeat-CRIT branch.

    A single CRIT (count=1) still requires clearing the kill_file manually;
    writing the override alone does NOT re-enable. This preserves the
    "operator acknowledges halt" semantic.
    """
    kf = tmp_path / "kf"
    ov = tmp_path / "kf.override"
    write_kill_file(kf, reason=REASON_DRIFT_CRIT, detail="kl=0.6")
    ov.touch()
    payload = read_kill_file(kf)
    locked, _ = should_lockout(payload, ov)
    assert locked is True


def test_repeat_crit_locks_out_without_override(tmp_path: Path):
    kf = tmp_path / "kf"
    ov = tmp_path / "kf.override"
    write_kill_file(kf, reason=REASON_DRIFT_CRIT, detail="1st")
    write_kill_file(kf, reason=REASON_DRIFT_CRIT, detail="2nd")
    payload = read_kill_file(kf)
    locked, reason = should_lockout(payload, ov)
    assert locked is True
    assert "REPEAT-CRIT LOCKOUT" in reason


def test_repeat_crit_still_locked_with_override_but_different_reason(tmp_path: Path):
    """Override present + repeat-CRIT: lockout persists because kill_file itself
    hasn't been cleared. Reason string clarifies the state for the operator."""
    kf = tmp_path / "kf"
    ov = tmp_path / "kf.override"
    write_kill_file(kf, reason=REASON_DRIFT_CRIT, detail="1st")
    write_kill_file(kf, reason=REASON_DRIFT_CRIT, detail="2nd")
    ov.touch()
    payload = read_kill_file(kf)
    locked, reason = should_lockout(payload, ov)
    assert locked is True
    assert "override seen" in reason


def test_operator_reason_locks_out_without_repeat_logic(tmp_path: Path):
    kf = tmp_path / "kf"
    write_kill_file(kf, reason=REASON_OPERATOR, detail="manual")
    payload = read_kill_file(kf)
    locked, reason = should_lockout(payload, None)
    assert locked is True
    assert "operator" in reason
    assert "REPEAT-CRIT" not in reason


def test_old_crit_past_window_not_repeat():
    """A drift_crit with a first_ts older than window_hours no longer counts
    as repeat-CRIT — falls back to standard kill_file-present lockout."""
    from datetime import datetime, timedelta, timezone
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    payload = {
        "reason": REASON_DRIFT_CRIT,
        "first_ts": old_ts,
        "last_ts": datetime.now(timezone.utc).isoformat(),
        "count": 5,
        "detail": "old crit",
    }
    locked, reason = should_lockout(payload, None)
    assert locked is True
    assert "REPEAT-CRIT" not in reason


# ---------- clear_kill_file --------------------------------------------------


def test_clear_kill_file_idempotent(tmp_path: Path):
    p = tmp_path / "kf"
    write_kill_file(p, reason=REASON_OPERATOR)
    clear_kill_file(p)
    assert not p.exists()
    # second clear doesn't raise
    clear_kill_file(p)
