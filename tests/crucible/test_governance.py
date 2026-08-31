"""Crucible P5 — governance handoff (survivor → notification → Tier-2), spec §3 human gate / §7.1.

The moat's final wall: only a lockbox-CLEARED survivor (CR-8) may be handed off, the handoff carries
the EXACT human-run audit command (never executed here — CLAUDE.md), and the scan is idempotent (a
survivor is handed off exactly once). A non-CLEARED entry must be REFUSED.
"""
from __future__ import annotations

from sharpen.crucible import (
    CRUCIBLE_VERSION,
    FileNotifier,
    GovernanceStore,
    STATUS_CLEARED,
    STATUS_INCUBATING,
    Tier2Handoff,
    card_from_dir,
    deep_audit_invocation,
    handoff_for,
    scan_and_handoff,
)
from sharpen.crucible.agentic.card import DiscoveryCard
from sharpen.crucible.governance.notify import LogNotifier
from sharpen.crucible.lockbox.lockbox import Lockbox, LockboxEntry

_GATES = "configs/signal_eval.gates.yaml"


def _entry(status: str, chash: str = "surv01") -> LockboxEntry:
    return LockboxEntry(
        candidate_hash=chash, substrate_id="synthetic", formula="delta(fred:DGS10,20)",
        candidate_type="overlay", crucible_version=CRUCIBLE_VERSION, gates_hash="abc123",
        proposal_ts="2020-01-01T00:00:00", enrolled_tick_ts="2020-01-01T00:00:00",
        data_snapshot_hash="snap12", min_forward_bars=63, min_forward_sharpe=0.30,
        status=status, forward_sharpe=(0.55 if status == STATUS_CLEARED else None),
        n_forward_bars=(80 if status == STATUS_CLEARED else 10))


def _card(chash: str = "surv01") -> DiscoveryCard:
    return DiscoveryCard(
        candidate_hash=chash, formula="delta(fred:DGS10,20)", candidate_type="overlay",
        crucible_version=CRUCIBLE_VERSION, gates_hash="abc123", proposal_ts="2020-01-01T00:00:00",
        data_snapshot_hash="snap12", holdout_delta_sr=0.12, dsr_aug=0.5, marginal_t=3.1,
        economic_rationale="rates level conditions the book")


def test_deep_audit_invocation_is_human_command() -> None:
    cmd = deep_audit_invocation("crucible", "overlay:surv01")
    assert "deep_strategy_audit" in cmd
    assert '"workstream": "crucible"' in cmd and '"scope": "overlay:surv01"' in cmd


def test_handoff_for_cleared_carries_evidence_and_verdict() -> None:
    h = handoff_for(_entry(STATUS_CLEARED), workstream="crucible", scope="overlay:surv01",
                    card=_card())
    assert isinstance(h, Tier2Handoff)
    assert h.incubation_status == STATUS_CLEARED
    assert h.forward_sharpe == 0.55 and h.n_forward_bars == 80
    assert h.holdout_delta_sr == 0.12 and h.marginal_t == 3.1        # verbatim from the card
    assert "deep_strategy_audit" in h.audit_command
    assert "NOT PROMOTED" in h.to_json()["_advisory"]


def test_handoff_refuses_non_cleared() -> None:
    import pytest
    with pytest.raises(ValueError, match="not\\s+CLEARED|CLEARED"):
        handoff_for(_entry(STATUS_INCUBATING), workstream="w", scope="s")


def test_handoff_without_card_still_valid() -> None:
    h = handoff_for(_entry(STATUS_CLEARED), workstream="crucible", scope="s")
    assert h.forward_sharpe == 0.55                                  # entry evidence present
    assert h.holdout_delta_sr is None                               # no card -> no in-sample verdict


def test_scan_and_handoff_is_idempotent(tmp_path) -> None:
    lb = Lockbox(tmp_path / "lockbox.db")
    lb._write(_entry(STATUS_CLEARED))                               # a CLEARED survivor
    lb._write(_entry(STATUS_INCUBATING, chash="surv02"))           # not yet eligible
    store = GovernanceStore(tmp_path / "gov.db")
    (tmp_path / "cards").mkdir()
    _card().write(tmp_path / "cards")
    notifier = FileNotifier(tmp_path / "notify")

    created = scan_and_handoff(
        lb, store, notifier, handoff_ts="2020-04-01T00:00:00", workstream="crucible",
        scope="overlay", card_lookup=card_from_dir(tmp_path / "cards"), out_dir=tmp_path / "handoffs")
    assert len(created) == 1                                        # only the CLEARED one
    assert created[0].candidate_hash == "surv01"
    assert created[0].holdout_delta_sr == 0.12                      # card verdict merged in
    assert (tmp_path / "notify" / "NOTIFY_surv01.md").exists()
    assert (tmp_path / "handoffs" / "handoff_surv01.json").exists()

    # a second scan does NOT re-notify (idempotent) even though the entry is still CLEARED.
    again = scan_and_handoff(lb, store, notifier, handoff_ts="2020-04-02T00:00:00",
                             workstream="crucible", scope="overlay")
    assert again == []
    assert len(store.handed_off()) == 1


def test_log_notifier_does_not_raise() -> None:
    LogNotifier().notify(handoff_for(_entry(STATUS_CLEARED), workstream="w", scope="s"))
