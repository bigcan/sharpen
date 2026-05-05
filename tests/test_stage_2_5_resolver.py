"""Decision-matrix tests for `_resolve_bootstrap_decision` (Protocol v2.5).

Pins the resolver's mapping from (P(ens_PF > solo_PF), P(ens_MDD > solo_MDD))
onto the v2.5 PROMOTE / PROMOTE_DD_ONLY / AMBIGUOUS_BOOT / SOLO_BEST_FALLBACK
decision space (S526). The matrix:

                P_MDD<0.90                           P_MDD>=0.90
  P_PF>=0.90    PROMOTE (PF dominance)               PROMOTE
  [0.75, 0.90)  AMBIGUOUS_BOOT                       PROMOTE_DD_ONLY
  P_PF<0.75     SOLO_BEST_FALLBACK                   SOLO_BEST_FALLBACK

Coverage: 4x4 grid of P_PF tier x P_MDD tier (16 cases) + edge cases for
LEGACY_GATE_DEFER, NO_DATA, custom thresholds, boundary semantics, and the
v2.5 dict return shape.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sg1_xauusd_ensemble_eval import _resolve_bootstrap_decision  # noqa: E402


def _bs(p_pf: float | None, p_mdd: float | None, *, n_bars: int = 5000) -> dict:
    """Construct a `block_bootstrap_pf_mdd`-shaped dict."""
    return {
        "n_bars": n_bars,
        "n_resamples": 10000,
        "block_len_mean": 70.0,
        "p_pf_ens_better": p_pf,
        "p_mdd_ens_better": p_mdd,
    }


def _gates(
    *,
    p_pf_promote: float = 0.90,
    p_mdd_promote: float = 0.90,
    p_pf_ambiguous: float | None = 0.75,
) -> dict:
    g = {
        "ensemble_bootstrap_p_pf_promote": p_pf_promote,
        "ensemble_bootstrap_p_mdd_promote": p_mdd_promote,
    }
    if p_pf_ambiguous is not None:
        g["ensemble_bootstrap_p_pf_ambiguous"] = p_pf_ambiguous
    return g


# ---------------------------------------------------------------------------
# 4x4 decision-matrix grid: P_PF tier x P_MDD tier (Protocol v2.5 target).
# Tiers chosen to land cleanly inside / outside the 0.75 ambiguous floor and
# the 0.90 promote threshold.
# ---------------------------------------------------------------------------
#
#                     P_MDD=0.30   P_MDD=0.50   P_MDD=0.85   P_MDD=0.95
# P_PF=0.30 (vlow)    SOLO         SOLO         SOLO         SOLO
# P_PF=0.60 (low)     SOLO         SOLO         SOLO         SOLO
# P_PF=0.80 (ambig)   AMBIG_BOOT   AMBIG_BOOT   AMBIG_BOOT   PROMOTE_DD_ONLY
# P_PF=0.95 (high)    PROMOTE      PROMOTE      PROMOTE      PROMOTE
GRID_CASES = [
    # (P_PF,  P_MDD,  expected_decision)
    (0.30,  0.30,  "SOLO_BEST_FALLBACK"),
    (0.30,  0.50,  "SOLO_BEST_FALLBACK"),
    (0.30,  0.85,  "SOLO_BEST_FALLBACK"),
    (0.30,  0.95,  "SOLO_BEST_FALLBACK"),
    (0.60,  0.30,  "SOLO_BEST_FALLBACK"),
    (0.60,  0.50,  "SOLO_BEST_FALLBACK"),
    (0.60,  0.85,  "SOLO_BEST_FALLBACK"),
    (0.60,  0.95,  "SOLO_BEST_FALLBACK"),
    (0.80,  0.30,  "AMBIGUOUS_BOOT"),
    (0.80,  0.50,  "AMBIGUOUS_BOOT"),
    (0.80,  0.85,  "AMBIGUOUS_BOOT"),
    (0.80,  0.95,  "PROMOTE_DD_ONLY"),
    (0.95,  0.30,  "PROMOTE"),
    (0.95,  0.50,  "PROMOTE"),
    (0.95,  0.85,  "PROMOTE"),
    (0.95,  0.95,  "PROMOTE"),
]


@pytest.mark.parametrize("p_pf, p_mdd, expected", GRID_CASES)
def test_decision_matrix_4x4(p_pf, p_mdd, expected):
    out = _resolve_bootstrap_decision(_bs(p_pf, p_mdd), _gates())
    assert out["decision"] == expected, (
        f"P_PF={p_pf} P_MDD={p_mdd}: expected {expected}, got {out['decision']} "
        f"(reason: {out['reason']})"
    )
    # Reason string must echo at least one probability so audit logs are diff-able.
    assert f"{p_pf:.3f}" in out["reason"] or f"{p_mdd:.3f}" in out["reason"]


# ---------------------------------------------------------------------------
# v2.5 dict return contract (C-1 in the architecture plan).
# ---------------------------------------------------------------------------

def test_dict_return_shape_for_promote():
    out = _resolve_bootstrap_decision(_bs(0.95, 0.95), _gates())
    assert set(out) >= {
        "decision", "primary_basis", "p_pf_ens_better", "p_mdd_ens_better",
        "thresholds_used", "reason",
    }
    assert out["primary_basis"] == "bootstrap"
    assert out["p_pf_ens_better"] == 0.95
    assert out["p_mdd_ens_better"] == 0.95
    assert out["thresholds_used"] == {
        "p_pf_promote": 0.90,
        "p_mdd_promote": 0.90,
        "p_pf_ambiguous": 0.75,
    }


def test_dict_return_shape_for_legacy_defer():
    out = _resolve_bootstrap_decision(
        _bs(0.95, 0.95), {"ensemble_uplift_min": 1.10}
    )
    assert out["decision"] == "LEGACY_GATE_DEFER"
    assert out["primary_basis"] == "legacy_uplift"
    # Even on defer, the resolver must still echo the probabilities (for audit logs).
    assert out["p_pf_ens_better"] == 0.95


# ---------------------------------------------------------------------------
# Boundary semantics: thresholds use >= (inclusive) on both channels.
# ---------------------------------------------------------------------------

def test_pf_at_promote_threshold_is_pass():
    out = _resolve_bootstrap_decision(_bs(0.90, 0.90), _gates())
    assert out["decision"] == "PROMOTE"


def test_pf_one_thousandth_below_promote_threshold_is_not_pass():
    out = _resolve_bootstrap_decision(_bs(0.899, 0.90), _gates())
    # PF fails, MDD passes → PROMOTE_DD_ONLY (P_PF still in [0.75, 0.90)).
    assert out["decision"] == "PROMOTE_DD_ONLY"


def test_pf_at_ambiguous_floor_is_ambiguous_boot():
    out = _resolve_bootstrap_decision(_bs(0.75, 0.50), _gates())
    assert out["decision"] == "AMBIGUOUS_BOOT"


def test_pf_just_below_ambiguous_floor_is_solo():
    out = _resolve_bootstrap_decision(_bs(0.749, 0.50), _gates())
    assert out["decision"] == "SOLO_BEST_FALLBACK"


# ---------------------------------------------------------------------------
# v2.5 PF-dominance row: P(PF) >= promote → PROMOTE regardless of MDD.
# Funding-Arb DSAC anchor: P_PF=1.0, P_MDD=0.36 was misclassified under v2.3
# (SOLO_BEST_FALLBACK). v2.5 promotes it on the PF channel.
# ---------------------------------------------------------------------------

def test_pf_dominance_promotes_regardless_of_mdd():
    out = _resolve_bootstrap_decision(_bs(1.0, 0.36), _gates())
    assert out["decision"] == "PROMOTE"
    assert "PF dominance" in out["reason"] or "wash" in out["reason"]


# ---------------------------------------------------------------------------
# Default p_pf_ambiguous when key absent: 0.75 (back-compat).
# ---------------------------------------------------------------------------

def test_p_pf_ambiguous_defaults_to_0_75_when_key_absent():
    gates_no_ambig = _gates(p_pf_ambiguous=None)  # only promote keys present
    # Above 0.75 → AMBIGUOUS_BOOT
    out_high = _resolve_bootstrap_decision(_bs(0.80, 0.50), gates_no_ambig)
    assert out_high["decision"] == "AMBIGUOUS_BOOT"
    # Below 0.75 → SOLO
    out_low = _resolve_bootstrap_decision(_bs(0.70, 0.50), gates_no_ambig)
    assert out_low["decision"] == "SOLO_BEST_FALLBACK"


def test_custom_ambiguous_floor_overrides_default():
    # Set ambiguous floor to 0.85 (BTC asset-class hypothetical override).
    gates_high = _gates(p_pf_ambiguous=0.85)
    # P_PF=0.80 now below the custom floor → SOLO (not AMBIGUOUS_BOOT).
    out = _resolve_bootstrap_decision(_bs(0.80, 0.50), gates_high)
    assert out["decision"] == "SOLO_BEST_FALLBACK"


# ---------------------------------------------------------------------------
# Custom promote thresholds (ADR-3 deferred per-asset-class override).
# ---------------------------------------------------------------------------

def test_custom_promote_threshold_relaxed_for_btc_hypothetical():
    gates_btc = _gates(p_pf_promote=0.85, p_mdd_promote=0.85)
    out = _resolve_bootstrap_decision(_bs(0.87, 0.86), gates_btc)
    assert out["decision"] == "PROMOTE"


# ---------------------------------------------------------------------------
# Fall-through paths.
# ---------------------------------------------------------------------------

def test_legacy_gate_defer_when_bootstrap_keys_absent():
    out = _resolve_bootstrap_decision(_bs(0.95, 0.95), {"ensemble_uplift_min": 1.10})
    assert out["decision"] == "LEGACY_GATE_DEFER"
    assert "no bootstrap gates" in out["reason"]


def test_either_bootstrap_key_alone_triggers_gate_path():
    # Either bootstrap key triggers the gate path; both absent → defer.
    gates_pf_only = {"ensemble_bootstrap_p_pf_promote": 0.90}
    out = _resolve_bootstrap_decision(_bs(0.95, 0.95), gates_pf_only)
    assert out["decision"] == "PROMOTE"


def test_no_data_when_p_pf_is_none():
    out = _resolve_bootstrap_decision(_bs(None, None, n_bars=10), _gates())
    assert out["decision"] == "NO_DATA"
    assert "insufficient" in out["reason"] or "unknown" in out["reason"]


def test_no_data_when_only_one_probability_missing():
    out = _resolve_bootstrap_decision(_bs(0.95, None), _gates())
    assert out["decision"] == "NO_DATA"


# ---------------------------------------------------------------------------
# Reason strings are audit-grade (probabilities echoed).
# ---------------------------------------------------------------------------

def test_promote_reason_echoes_both_probabilities():
    out = _resolve_bootstrap_decision(_bs(0.94, 0.92), _gates())
    assert "0.940" in out["reason"] and "0.920" in out["reason"]


def test_ambiguous_boot_reason_calls_out_band():
    out = _resolve_bootstrap_decision(_bs(0.80, 0.40), _gates())
    assert "ambiguous" in out["reason"]
    assert "0.75" in out["reason"] and "0.90" in out["reason"]
