"""Stage-2/2.5 report writers + v2.2 eval_distribution helpers."""

from finrl_pro_ds.reporting.challenge_target import (
    DEFAULT_PHASE_SPECS,
    compute_challenge_target_hit_rate,
    compute_challenge_target_hit_rates,
    parse_phase_spec_arg,
)
from finrl_pro_ds.reporting.eval_distribution import (
    compute_eval_distribution,
    summarize_scalar_actions,
)

__all__ = [
    "DEFAULT_PHASE_SPECS",
    "compute_challenge_target_hit_rate",
    "compute_challenge_target_hit_rates",
    "compute_eval_distribution",
    "parse_phase_spec_arg",
    "summarize_scalar_actions",
]
