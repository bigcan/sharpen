"""HPO (Hyperparameter Optimization) shared module.

Provides the canonical objective function, evaluation, env factory, and sampler
used by both serial (run_full_pipeline.py) and distributed HPO workers.
"""
from sharpen.hpo.env_factory import create_vector_env, make_env
from sharpen.hpo.evaluate import analyze_hpo_correlation, evaluate_for_hpo
from sharpen.hpo.objective import make_objective
from sharpen.hpo.sampler import create_sampler

__all__ = [
    "make_objective",
    "evaluate_for_hpo",
    "analyze_hpo_correlation",
    "create_vector_env",
    "make_env",
    "create_sampler",
]
