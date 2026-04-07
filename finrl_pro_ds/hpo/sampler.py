"""Optuna sampler creation — shared between serial and distributed HPO."""
import logging

from optuna.samplers import RandomSampler, TPESampler

logger = logging.getLogger("FinRL.HPO")


def create_sampler(hpo_config: dict, *, distributed: bool = False):
    """Create Optuna sampler from config.

    Supports 'tpe' (default) and 'random'.
    When ``distributed=True``, adds ``constant_liar=True`` and uses ``seed=None``
    so that each worker explores a different region of the search space.
    """
    sampler_type = hpo_config.get("sampler", "tpe").lower()
    seed = hpo_config.get("sampler_seed", 42)

    if distributed:
        seed = None  # Workers must NOT share the same seed

    if sampler_type == "random":
        logger.info("Using RandomSampler (seed=%s)", seed)
        return RandomSampler(seed=seed)

    # Default: TPE with multivariate correlation modeling
    kwargs = dict(seed=seed, n_startup_trials=10, multivariate=True)
    if distributed:
        kwargs["constant_liar"] = True
    logger.info(
        "Using TPESampler (seed=%s, n_startup_trials=10, multivariate=True, constant_liar=%s)",
        seed,
        kwargs.get("constant_liar", False),
    )
    return TPESampler(**kwargs)
