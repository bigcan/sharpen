"""Protocol v2.3 §4.5 step 6 ensemble-bundle swap-approval handshake.

The atomic ensemble swap-bundle reader (`ensemble_bundle.py`) verifies
SHA256 integrity but does NOT gate on operator approval. For prop-firm
/ live-capital workstreams the protocol requires an explicit cutover
handshake — operator must place a `<kill_file>.swap_approved` sentinel
file before the engine will load a bundle that differs from the bundle
last loaded by this strategy. Same-bundle restarts (config unchanged,
container bounce) bypass the handshake.

The handshake is a deliberate human-in-the-loop step. Without it, an
operator pushing a freshly-promoted bundle into config could trigger a
silent cutover at the next container restart with no acknowledgement
that the new ensemble was the intended one. The handshake forces the
operator to (a) inspect the new bundle's manifest, (b) write the
sentinel, and (c) restart the container, in that order.

State files written by this module:
- `last_bundle_state_path` (default ``/app/state/last_loaded_bundle.txt``):
  records the absolute path AND SHA256 of the bundle this engine last
  loaded. Used to compare against the bundle declared in the live config
  to detect a "same vs new" bundle. SHA256 is recorded so that an
  in-place rebuild of the same path (operator overwrote the bundle file)
  is also caught as a swap.
- `swap_approved_path` (default ``<kill_file>.swap_approved``): the
  operator-written handshake sentinel. Consumed (deleted) on use so
  every swap requires a fresh approval. The file may be empty or contain
  JSON metadata for audit (operator signature, ticket ID); the contents
  are not parsed by the engine.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SwapHandshakeResult:
    approved: bool
    reason: str
    is_swap: bool
    previous_bundle_path: Optional[str]
    previous_bundle_sha256: Optional[str]
    new_bundle_sha256: str

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "reason": self.reason,
            "is_swap": self.is_swap,
            "previous_bundle_path": self.previous_bundle_path,
            "previous_bundle_sha256": self.previous_bundle_sha256,
            "new_bundle_sha256": self.new_bundle_sha256,
        }


def _sha256_file(path: Path, _chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(_chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _read_last_bundle_state(state_path: Path) -> tuple[Optional[str], Optional[str]]:
    """Return (path, sha256) of the previously loaded bundle, or (None, None)."""
    if not state_path.exists():
        return None, None
    try:
        raw = state_path.read_text(encoding="utf-8").strip()
    except OSError as e:
        logger.warning(f"last_bundle state unreadable ({state_path}): {e}")
        return None, None
    if not raw:
        return None, None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw, None  # legacy format: bare path
    if not isinstance(payload, dict):
        return None, None
    return payload.get("path"), payload.get("sha256")


def _write_last_bundle_state(
    state_path: Path,
    bundle_path: Path,
    sha256: str,
) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": str(bundle_path),
        "sha256": sha256,
        "loaded_at": datetime.now(timezone.utc).isoformat(),
    }
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def check_swap_approved(
    new_bundle_path: Path | str,
    *,
    last_bundle_state_path: Path | str,
    swap_approved_path: Path | str,
    is_prop_firm: bool,
    consume_handshake: bool = True,
) -> SwapHandshakeResult:
    """Decide whether the engine may load ``new_bundle_path``.

    Args:
        new_bundle_path: bundle the engine is about to load (the
            ``ensemble_v{N}.tar.gz`` file).
        last_bundle_state_path: file where the previous successful load
            was recorded (path + sha256). Created if absent.
        swap_approved_path: operator-written sentinel
            (``<kill_file>.swap_approved``). Required for prop-firm
            swaps. Consumed on success unless ``consume_handshake=False``
            (test-only).
        is_prop_firm: whether this workstream carries a prop-firm /
            live-capital tag. Non-prop-firm strategies skip the
            handshake entirely (but still record state for diagnostics).
        consume_handshake: when True (default), delete the approval
            sentinel after successful consumption so every swap requires
            a fresh approval.

    Returns:
        :class:`SwapHandshakeResult` — caller checks ``approved``.

    Notes:
        Caller is responsible for invoking
        :func:`record_successful_load` once the engine has finished
        loading; this is a separate step so a load that fails AFTER
        approval (e.g. broker connect rejected) doesn't poison the
        recorded state.
    """
    new_bundle_path = Path(new_bundle_path)
    last_bundle_state_path = Path(last_bundle_state_path)
    swap_approved_path = Path(swap_approved_path)

    if not new_bundle_path.exists():
        return SwapHandshakeResult(
            approved=False,
            reason=f"bundle file not found: {new_bundle_path}",
            is_swap=False,
            previous_bundle_path=None,
            previous_bundle_sha256=None,
            new_bundle_sha256="",
        )

    new_sha = _sha256_file(new_bundle_path)
    prev_path, prev_sha = _read_last_bundle_state(last_bundle_state_path)

    same_path = prev_path is not None and Path(prev_path) == new_bundle_path
    same_sha = prev_sha is not None and prev_sha == new_sha
    is_swap = not (same_path and same_sha)

    if not is_swap:
        return SwapHandshakeResult(
            approved=True,
            reason="same bundle as previous run (path + sha256 unchanged)",
            is_swap=False,
            previous_bundle_path=prev_path,
            previous_bundle_sha256=prev_sha,
            new_bundle_sha256=new_sha,
        )

    if not is_prop_firm:
        return SwapHandshakeResult(
            approved=True,
            reason="swap detected; non-prop-firm strategy — handshake skipped",
            is_swap=True,
            previous_bundle_path=prev_path,
            previous_bundle_sha256=prev_sha,
            new_bundle_sha256=new_sha,
        )

    if not swap_approved_path.exists():
        return SwapHandshakeResult(
            approved=False,
            reason=(
                f"BUNDLE SWAP REQUIRES HANDSHAKE: previous=({prev_path}, "
                f"{(prev_sha or '<none>')[:12]}…) new=({new_bundle_path}, "
                f"{new_sha[:12]}…) — operator must write sentinel "
                f"{swap_approved_path} to approve cutover. Bundle differs "
                f"by {'path' if not same_path else ''}{' + ' if not same_path and not same_sha else ''}"
                f"{'sha256 (in-place rebuild)' if not same_sha else ''}."
            ),
            is_swap=True,
            previous_bundle_path=prev_path,
            previous_bundle_sha256=prev_sha,
            new_bundle_sha256=new_sha,
        )

    if consume_handshake:
        try:
            swap_approved_path.unlink()
        except OSError as e:
            logger.warning(
                f"swap_approved consume failed ({swap_approved_path}): {e}",
            )

    return SwapHandshakeResult(
        approved=True,
        reason=(
            f"swap approved by operator handshake at {swap_approved_path}"
            + (" (consumed)" if consume_handshake else "")
        ),
        is_swap=True,
        previous_bundle_path=prev_path,
        previous_bundle_sha256=prev_sha,
        new_bundle_sha256=new_sha,
    )


def record_successful_load(
    bundle_path: Path | str,
    *,
    last_bundle_state_path: Path | str,
    bundle_sha256: Optional[str] = None,
) -> None:
    """Record a successful bundle load to ``last_bundle_state_path``.

    Call this AFTER the engine has loaded and accepted the bundle (the
    swap-approval handshake passing is necessary but not sufficient — a
    load that fails downstream should not poison the recorded state).
    Re-hashes the bundle when ``bundle_sha256`` is not supplied.
    """
    bundle_path = Path(bundle_path)
    last_bundle_state_path = Path(last_bundle_state_path)
    sha = bundle_sha256 if bundle_sha256 is not None else _sha256_file(bundle_path)
    _write_last_bundle_state(last_bundle_state_path, bundle_path, sha)
    logger.info(
        f"recorded successful bundle load: path={bundle_path} sha256={sha[:12]}… "
        f"→ {last_bundle_state_path}",
    )
