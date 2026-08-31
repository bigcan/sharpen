"""Asset-agnostic live-trading components.

Modules here are not crypto-specific (the CFD/futures engines reuse
them). Crypto-specific orchestration still lives under
``sharpen.crypto.live``.
"""

from sharpen.live.challenge_state_machine import (
    PHASE_ORDER,
    REASON_PHASE_COMPLETE,
    ChallengePhase,
    ChallengeStateMachine,
    ChallengeStatus,
)
from sharpen.live.ensemble_bundle import (
    BundleContents,
    BundleIntegrityError,
    extract_and_verify_bundle,
)
from sharpen.live.swap_handshake import (
    SwapHandshakeResult,
    check_swap_approved,
    record_successful_load,
)
from sharpen.live.agent_loader import (
    build_agent,
    resolve_agent_paths,
)

__all__ = [
    "PHASE_ORDER",
    "REASON_PHASE_COMPLETE",
    "ChallengePhase",
    "ChallengeStateMachine",
    "ChallengeStatus",
    "BundleContents",
    "BundleIntegrityError",
    "extract_and_verify_bundle",
    "SwapHandshakeResult",
    "check_swap_approved",
    "record_successful_load",
    "build_agent",
    "resolve_agent_paths",
]
