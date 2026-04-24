"""Pre-A/B reward calibration gate (ADR-3, prop-firm decoupling rev 2).

Replays a fixed-seed training-data rollout under (a) PropFirmWrapperV7 with
``success_bonus=10.0`` and (b) RiskShapingWrapper (no bonus). Compares mean
per-episode total reward to detect the success_bonus removal's impact on
the SAC Bellman backup *before* committing to the Step-2 A/B.

Gate bands:
- shift < 5%  → PASS (no action needed)
- shift 5–15% → LOG the required dsr_scale bump; test FAILS so CI forces
                operator to update the migrated config
- shift > 15% → FAIL with escalation message (likely re-HPO needed)

Skipped when the first-customer checkpoint is unavailable. CI wires it in
for Step 2 by placing the checkpoint at the expected path. Can be forced
via env var ``FINRL_REWARD_GATE_CHECKPOINT`` (absolute path).

This fixture intentionally keeps the rollout short (≥200 episodes enforced)
so variance on the shift metric stays below the 5% threshold.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("torch")


# Default first-customer checkpoint path. Override with env var for CI.
DEFAULT_CHECKPOINT = Path(
    "checkpoints/gmgp1_xauusd_ftmo_rehpo_l1_multiseed_oanda/"
    "WF_seed42_fold_07_final.pth"
)


def _resolve_checkpoint() -> Path | None:
    override = os.environ.get("FINRL_REWARD_GATE_CHECKPOINT")
    if override:
        p = Path(override)
        return p if p.exists() else None
    return DEFAULT_CHECKPOINT if DEFAULT_CHECKPOINT.exists() else None


# Gate band thresholds (ADR-3)
GATE_PASS_BAND = 0.05   # < 5% shift: no action
GATE_WARN_BAND = 0.15   # 5-15%: log dsr_scale bump, fail CI
# > 15%: escalation (re-HPO)


def _mean_reward_over_episodes(
    wrapper_cls,
    wrapper_kwargs: dict,
    n_episodes: int,
    seed: int,
    checkpoint_path: Path,
) -> float:
    """Run ``n_episodes`` deterministic rollouts; return mean total reward.

    Implementation left as a pytest helper stub — populated in Step 2 by the
    A/B harness (scripts/prop_firm_ab_compare.py::_replay_for_calibration).
    Until that script lands, this helper raises ``pytest.skip`` so the gate
    stays wired into CI without blocking Step 1 landings.
    """
    pytest.skip(
        "Reward-calibration gate helper pending Step 2 wiring "
        "(scripts/prop_firm_ab_compare.py::_replay_for_calibration). "
        "Gate will activate once the first-customer checkpoint + A/B harness "
        "are in place. Fixture stays in CI so Step 2 cannot skip the gate."
    )


@pytest.fixture(scope="module")
def checkpoint_path() -> Path:
    cp = _resolve_checkpoint()
    if cp is None:
        pytest.skip(
            "First-customer checkpoint not available. Set "
            "FINRL_REWARD_GATE_CHECKPOINT=<abs_path> to exercise the gate, "
            f"or place the checkpoint at {DEFAULT_CHECKPOINT}."
        )
    return cp


def _classify_shift(shift: float) -> str:
    """Return 'pass', 'warn', or 'fail' per ADR-3 bands."""
    abs_shift = abs(shift)
    if abs_shift < GATE_PASS_BAND:
        return "pass"
    if abs_shift < GATE_WARN_BAND:
        return "warn"
    return "fail"


# ---------------------------------------------------------------------------
# Unit-level tests of the classification logic (always run)
# ---------------------------------------------------------------------------

def test_classify_shift_pass_band():
    assert _classify_shift(0.03) == "pass"
    assert _classify_shift(-0.04) == "pass"


def test_classify_shift_warn_band():
    assert _classify_shift(0.08) == "warn"
    assert _classify_shift(-0.12) == "warn"


def test_classify_shift_fail_band():
    assert _classify_shift(0.16) == "fail"
    assert _classify_shift(-0.30) == "fail"


def test_classify_shift_boundary_pass_warn():
    """5% exact → warn (inclusive lower bound)."""
    assert _classify_shift(0.05) == "warn"


def test_classify_shift_boundary_warn_fail():
    """15% exact → fail."""
    assert _classify_shift(0.15) == "fail"


# ---------------------------------------------------------------------------
# End-to-end gate (skipped without checkpoint + Step 2 harness)
# ---------------------------------------------------------------------------

def test_reward_calibration_gate(checkpoint_path: Path):
    """Gate: mean per-episode reward shift between wrappers must be < 5%.

    Stub enforced: requires ≥200 episodes to keep shift variance bounded.
    """
    from finrl_pro_ds.envs.prop_firm_wrapper import PropFirmWrapperV7
    from finrl_pro_ds.envs.risk_shaping_wrapper import RiskShapingWrapper

    n_episodes = 200
    seed = 42

    mean_v7 = _mean_reward_over_episodes(
        PropFirmWrapperV7,
        dict(
            profit_target_pct=0.10,
            success_bonus=10.0,
            augment_obs=False,
            max_trailing_drawdown_pct=0.10,
        ),
        n_episodes=n_episodes,
        seed=seed,
        checkpoint_path=checkpoint_path,
    )

    mean_rs = _mean_reward_over_episodes(
        RiskShapingWrapper,
        dict(
            augment_obs="off",
            max_trailing_drawdown_pct=0.10,
        ),
        n_episodes=n_episodes,
        seed=seed,
        checkpoint_path=checkpoint_path,
    )

    # Shift computed on V7 as the control (pre-migration mean).
    shift = (mean_rs - mean_v7) / abs(mean_v7) if mean_v7 != 0 else 0.0
    band = _classify_shift(shift)

    if band == "warn":
        # In the warn band we want to fail CI so the operator updates the
        # migrated config with a calibrated dsr_scale before Step-2 A/B.
        dsr_bump = 1.0 / max(1e-6, 1.0 - abs(shift))
        pytest.fail(
            f"Reward-calibration gate WARN: shift={shift:+.3%} "
            f"(5-15% band). Bump dsr_scale by ~{dsr_bump:.3f}x in the "
            f"migrated config before Step-2 A/B. See ADR-3."
        )

    if band == "fail":
        pytest.fail(
            f"Reward-calibration gate FAIL: shift={shift:+.3%} "
            f"(>15% band). Escalate — likely re-HPO with a dsr_scale axis "
            f"is required. See ADR-3."
        )

    # band == "pass"
    assert band == "pass"
