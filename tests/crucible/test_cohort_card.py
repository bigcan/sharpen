"""Phase 4 Step 3 — CohortCard: verbatim verdict copy (CR-1), caps at PROMISING, JSON round-trip."""
from __future__ import annotations

from sharpen.crucible.agentic.cohort_card import CohortCard, card_from_verdict
from sharpen.signals.generation.cohort_eval import CohortVerdict


def _verdict(verdict: str = "PROMISING") -> CohortVerdict:
    return CohortVerdict(
        members=("h1", "h2", "h3"), n_members=3, n_candidates_seen=12, n_culled=1,
        mean_pairwise_corr=0.08, sr_star_cohort=0.42, dsr_cohort_book=0.93,
        passes_analytic_floor=True, mc_p_value=0.012, mc_t_obs=0.31, mc_n_reps=1000,
        mc_n_valid_reps=987, mc_block_length=21, passes_mc=True,
        holdout_delta_sr=0.14, holdout_passes=True, pool_content_hash="abc123def456",
        verdict=verdict)


def test_card_from_verdict_copies_fields_verbatim() -> None:
    v = _verdict()
    card = card_from_verdict(v, crucible_version="crucible-v2.7", funnel_gates_hash="519158fa1450",
                             cohort_gates_hash="dd563b4b4f7c", proposal_ts="2026-07-03T00:00:00",
                             data_snapshot_hash="snap42")
    assert card.cohort_hash == v.pool_content_hash            # cohort_hash == pool_content_hash (FDR key)
    assert card.members == ("h1", "h2", "h3") and card.n_members == 3
    assert card.mc_p_value == 0.012 and card.holdout_delta_sr == 0.14   # verbatim scorer numbers
    assert card.passes_mc is True and card.passes_analytic_floor is True
    assert card.verdict == "PROMISING"
    assert card.funnel_gates_hash == "519158fa1450" and card.cohort_gates_hash == "dd563b4b4f7c"


def test_card_caps_at_promising_not_human_eligible() -> None:
    """ADR-7 / caps-at-PROMISING: a fresh cohort card is PENDING_P4 and never yet human-eligible."""
    card = card_from_verdict(_verdict(), crucible_version="crucible-v2.7",
                             funnel_gates_hash="f", cohort_gates_hash="c")
    assert card.incubation_status == "PENDING_P4"
    assert card.eligible_for_human_gate is False


def test_card_json_round_trip(tmp_path) -> None:
    card = card_from_verdict(_verdict("LOGGED"), crucible_version="crucible-v2.7",
                             funnel_gates_hash="f", cohort_gates_hash="c")
    path = card.write(tmp_path / "cards")
    assert path.name == f"cohort_{card.cohort_hash}.json"
    import json
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw["members"], list)                  # tuple serialized as JSON list
    back = CohortCard.from_json(raw)
    assert back == card                                      # exact round-trip (members re-tupled)
