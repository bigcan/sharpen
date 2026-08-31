"""GPUHub burst router — where a tick's mining runs (spec §10.2, decision 2).

Default substrate mining runs LOCAL on the RTX 5090; a heavy tick may *burst* to GPUHub. The fleet
convention (``feedback_hpo_gpuhub1_nonhpo_gpuhub2``) is strict: **HPO → gpuhub-1, non-HPO → gpuhub-2**.
Crucible mining (C3 ``evolve`` genetic search) is a non-HPO workload, so a burst mine routes to
gpuhub-2; only an HPO-shaped tick (e.g. a future Optuna sweep over generation hyper-params) routes
to gpuhub-1.

This is a **pure advisory router**: it decides the target and records it in the tick provenance. It
does NOT SSH, deploy, or launch anything — remote provisioning is the Deploy skill's job and stays
a human/operator step (the orchestrator never auto-rents compute). Keeping it a pure function makes
the routing decision testable and the tick record honest about where work was *intended* to run.
"""
from __future__ import annotations

from dataclasses import dataclass

LOCAL = "local"
GPUHUB_1 = "gpuhub-1"          # HPO only (feedback_hpo_gpuhub1_nonhpo_gpuhub2)
GPUHUB_2 = "gpuhub-2"          # non-HPO (mining) burst target


@dataclass(frozen=True, slots=True)
class BurstDecision:
    """The routing verdict for one substrate's mining work (advisory)."""

    target: str                    # LOCAL | GPUHUB_1 | GPUHUB_2
    reason: str
    est_candidates: int            # the cost estimate the decision was made on


def route_burst(*, est_candidates: int, is_hpo: bool, local_capacity: int = 128) -> BurstDecision:
    """Decide where a mining workload of ``est_candidates`` should run.

    * An **HPO** workload always routes to ``gpuhub-1`` (fleet rule — never local, never gpuhub-2).
    * A **non-HPO** mine within ``local_capacity`` runs LOCAL (the RTX 5090 default, §10.2).
    * A non-HPO mine that exceeds ``local_capacity`` bursts to ``gpuhub-2``.

    ``local_capacity`` is the per-tick candidate count the local box handles comfortably; above it,
    bursting keeps the nightly loop from starving interactive local work.
    """
    if is_hpo:
        return BurstDecision(GPUHUB_1, "HPO workload -> gpuhub-1 (fleet rule)", est_candidates)
    if est_candidates > local_capacity:
        return BurstDecision(
            GPUHUB_2, f"non-HPO mine of {est_candidates} > local_capacity {local_capacity} "
            f"-> burst to gpuhub-2", est_candidates)
    return BurstDecision(LOCAL, f"non-HPO mine of {est_candidates} <= local_capacity "
                         f"{local_capacity} -> local RTX 5090", est_candidates)
