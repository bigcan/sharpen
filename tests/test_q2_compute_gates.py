"""Q2 train_parity verdict tiers — S498 protocol revision.

Verifies the 3-tier classifier in ``scripts/q2_compute_gates.py`` matches
the bands declared in ``Q2_THRESHOLDS`` (mirrored in
``scripts/prop_firm_ab_compare.py``) and aligns with the S496 noise-floor
data (project_q2_train_parity_fail.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from q2_compute_gates import (  # noqa: E402
    CRITIC_LOSS_AMBIGUOUS,
    CRITIC_LOSS_BAND,
    _classify_critic_loss_ratio,
    compute_q2_decision,
)


# ---------------------------------------------------------------------------
# Band parity with prop_firm_ab_compare.py
# ---------------------------------------------------------------------------

def test_bands_mirror_q2_thresholds() -> None:
    """If the two files drift, the validator's verdict won't match the harness's
    config — surface the divergence loudly via this single import-time check."""
    from prop_firm_ab_compare import Q2_THRESHOLDS  # type: ignore

    assert tuple(Q2_THRESHOLDS["critic_loss_ratio_band"]) == CRITIC_LOSS_BAND
    assert tuple(Q2_THRESHOLDS["critic_loss_ratio_ambiguous"]) == CRITIC_LOSS_AMBIGUOUS
    # Dropped gates must stay dropped
    for dropped in (
        "terminal_q_ratio_band",
        "actor_loss_ratio_band",
        "return_kl_divergence_max",
    ):
        assert dropped not in Q2_THRESHOLDS, (
            f"{dropped} reintroduced; S496 showed it noise-bound at 100K SAC steps"
        )


# ---------------------------------------------------------------------------
# Verdict tiers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a,b,expected", [
    # Inside ambiguous band [0.95, 1.05]
    (1.0, 1.00, "AMBIGUOUS"),
    (1.0, 0.95, "AMBIGUOUS"),
    (1.0, 1.05, "AMBIGUOUS"),
    (1.0, 1.034, "AMBIGUOUS"),  # S496 V7-vs-V7 noise floor
    # Inside main band but outside ambiguous → PASS
    (1.0, 0.94, "PASS"),
    (1.0, 0.80, "PASS"),
    (1.0, 1.06, "PASS"),
    (1.0, 1.25, "PASS"),
    (1.0, 1.167, "PASS"),  # S496 V7-vs-RS critic_loss
    # Outside main band → FAIL
    (1.0, 0.79, "FAIL"),
    (1.0, 1.26, "FAIL"),
    (1.0, 0.50, "FAIL"),
])
def test_classify_critic_loss_ratio_tiers(a: float, b: float, expected: str) -> None:
    out = _classify_critic_loss_ratio(a, b)
    assert out["verdict"] == expected, (
        f"A={a} B={b} ratio={out.get('ratio')} → got {out['verdict']}, expected {expected}"
    )


def test_classify_handles_zero_v7() -> None:
    """Pathological V7 critic_loss collapse — only PASS if RS also collapsed."""
    pass_case = _classify_critic_loss_ratio(0.0, 0.0)
    assert pass_case["verdict"] == "PASS"
    assert pass_case["ratio"] is None
    assert "magnitudes" in pass_case["note"]

    fail_case = _classify_critic_loss_ratio(0.0, 0.5)
    assert fail_case["verdict"] == "FAIL"
    assert fail_case["ratio"] is None


# ---------------------------------------------------------------------------
# End-to-end compute_q2_decision
# ---------------------------------------------------------------------------

def test_compute_q2_decision_passes_above_noise() -> None:
    """20-point critic_loss tail with a clean 1.10 ratio (above noise band)."""
    v7 = {"train/critic_loss": [0.5] * 30}
    rs = {"train/critic_loss": [0.55] * 30}
    decision = compute_q2_decision(v7, rs)
    assert decision["verdict"] == "PASS"
    assert decision["critic_loss_ratio"]["ratio"] == pytest.approx(1.10, rel=1e-3)
    assert decision["samples"]["v7_critic_loss_n"] == 30


def test_compute_q2_decision_ambiguous_inside_noise_floor() -> None:
    """20-point critic_loss tail with ratio 1.02 → inside noise floor."""
    v7 = {"train/critic_loss": [0.5] * 30}
    rs = {"train/critic_loss": [0.51] * 30}
    decision = compute_q2_decision(v7, rs)
    assert decision["verdict"] == "AMBIGUOUS"
    assert decision["critic_loss_ratio"]["ratio"] == pytest.approx(1.02, rel=1e-3)


def test_compute_q2_decision_fails_outside_band() -> None:
    v7 = {"train/critic_loss": [0.5] * 30}
    rs = {"train/critic_loss": [1.0] * 30}
    decision = compute_q2_decision(v7, rs)
    assert decision["verdict"] == "FAIL"
    assert decision["critic_loss_ratio"]["ratio"] == pytest.approx(2.0, rel=1e-3)
