"""DiscoveryCard — one record per PROMISING survivor (spec §6.2).

A card bundles everything a human needs to decide whether to spend a Tier-2 deep audit: the
pre-registered spec (CR-2), the **verbatim** scorer verdict (copied field-for-field — the Triage
Analyst may add narrative but never edits a number, CR-1), the CPCV marginal-ΔSR distribution, the
combiner marginal contribution, and the lockbox incubation status. A freshly-minted card is stamped
``incubation_status="PENDING_P4"`` and ``eligible_for_human_gate=False``: a survivor is PROMISING,
never "discovered". Once the P4 lockbox (``crucible/lockbox/``) enrolls it, ``crucible.lockbox
.updated_card`` transitions the status to INCUBATING → CLEARED/REJECTED and sets
``eligible_for_human_gate`` ONLY on CLEARED — forward-data incubation is the arbiter (CR-8).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

INCUBATION_PENDING = "PENDING_P4"     # the lockbox (CR-8) is built in P4; no card can bypass it


@dataclass(frozen=True, slots=True)
class DiscoveryCard:
    """A PROMISING survivor's discovery card (spec §6.2). All score fields are copied verbatim from
    the scorer; ``agent_narrative`` is the only free-text field the agent may author."""

    candidate_hash: str
    formula: str
    candidate_type: str
    crucible_version: str
    gates_hash: str
    proposal_ts: str | None = None
    data_snapshot_hash: str | None = None

    # pre-registration (CR-2) — the hypothesis, locked before OOS
    spec: dict = field(default_factory=dict)
    economic_rationale: str = ""

    # verbatim scorer verdict (CR-1) — the binding held-out numbers + the train-split richness
    verdict: str = "PROMISING"
    holdout_passes: bool | None = None
    holdout_delta_sr: float | None = None      # BINDING: ΔSR on the embargoed tail at full N
    train_delta_sr_oos: float | None = None     # train-split CPCV mean ΔSR
    delta_sr_median: float | None = None        # CPCV median path ΔSR (gate leg, crucible-v2.0)
    frac_paths_positive: float | None = None
    dsr_aug: float | None = None                # deflated augmented-book Sharpe (gen-N)
    marginal_t: float | None = None             # marginal-contribution t-stat (HLZ)
    n_paths: int | None = None                  # CPCV path count (C(n_groups,k_test))

    # combine (C1) — the marginal contribution to the combined book == the binding ΔSR
    combiner_marginal_delta_sr: float | None = None

    # lockbox (CR-8) — forward incubation; not built until P4, so a card is never yet human-eligible
    incubation_status: str = INCUBATION_PENDING
    incubation_forward_sharpe: float | None = None
    eligible_for_human_gate: bool = False

    # narrative — the ONLY field the agent authors; verdict fields above are untouchable
    agent_narrative: str = ""

    def to_json(self) -> dict:
        return asdict(self)

    def write(self, out_dir: str | Path) -> Path:
        """Write ``card_<candidate_hash>.json`` into ``out_dir``; return its path."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"card_{self.candidate_hash}.json"
        path.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, data: dict) -> "DiscoveryCard":
        fields = set(cls.__dataclass_fields__)                # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in fields})
