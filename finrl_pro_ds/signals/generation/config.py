"""Load the generation gates block into a FitnessConfig + evolve kwargs (no hardcoded gates).

Mirrors the ``signals/gates.py`` pattern: ``_GEN_DEFAULTS`` is the single code-side mirror of
the ``generation:`` block in ``configs/signal_eval.gates.yaml`` (the YAML is authoritative; the
defaults exist only so tests and an absent block behave sanely). Bounds are validated here so a
malformed block fails fast rather than silently mis-configuring the search.
"""
from __future__ import annotations

from pathlib import Path

from .fitness import FitnessConfig

# GP8-02: the wired substrates. Each panel maps to the EXACT base book its runner builds — the
# candidate is scored against that book, so a config naming a panel/sleeve set the runner does not
# build is a silent no-op. Fail fast instead. `cross_asset` = US ETF panel + {tsmom, rates_carry}
# (production_base_sleeves); `taiwan` = TAIEX ETF panel + TX/TE/TF TSMOM only (taiwan_base_sleeves,
# S553-cont step 2 — there is no Taiwan rates-carry sleeve).
_WIRED_SUBSTRATES: dict[str, frozenset[str]] = {
    "cross_asset": frozenset({"tsmom", "rates_carry"}),
    "taiwan": frozenset({"tsmom"}),
}

_GEN_DEFAULTS: dict = {
    "enabled": False, "panel": "cross_asset", "base_sleeves": ["tsmom", "rates_carry"],
    "hold_horizon": 21, "pop_size": 200, "n_generations": 40, "rng_seed": 7,
    "elite_frac": 0.30, "cost_bps": 0.0010, "ls_min_names": 6, "max_ast_nodes": 24,
    "turnover_soft_cap": 12.0, "lambda_turnover": 0.05, "lambda_complexity": 0.10,
    "min_combination_uplift": 0.10, "hlz_t_min": 3.0, "promising_dsr": 0.90,
    "max_base_corr": 0.70, "delta_median_min": 0.0, "frac_positive_min": 0.50,
    "holdout_frac": 0.25, "holdout_embargo_days": 21,
    "cpcv_n_groups": 6, "cpcv_k_test": 2, "cpcv_embargo_days": 21, "cpcv_purge_horizon": 1,
}


def _validate(g: dict) -> None:
    if not (1 <= int(g["cpcv_k_test"]) < int(g["cpcv_n_groups"])):
        raise ValueError("generation.cpcv_k_test must satisfy 1 <= k_test < cpcv_n_groups")
    if int(g["cpcv_embargo_days"]) < 0 or int(g["holdout_embargo_days"]) < 0:
        raise ValueError("generation embargo days must be >= 0")
    if not (0.0 < float(g["holdout_frac"]) < 1.0):
        raise ValueError("generation.holdout_frac must be in (0, 1)")
    if int(g["pop_size"]) < 2 or int(g["n_generations"]) < 1:
        raise ValueError("generation.pop_size >= 2 and n_generations >= 1 required")
    if int(g["max_ast_nodes"]) < 2:
        raise ValueError("generation.max_ast_nodes must be >= 2")
    if float(g["hlz_t_min"]) < 0 or not (0.0 <= float(g["promising_dsr"]) <= 1.0):
        raise ValueError("generation.hlz_t_min >= 0 and promising_dsr in [0,1] required")
    if not (0.0 <= float(g["max_base_corr"]) <= 1.0):
        raise ValueError("generation.max_base_corr must be in [0,1]")
    if not (0.0 <= float(g["frac_positive_min"]) <= 1.0):
        raise ValueError("generation.frac_positive_min must be in [0,1]")
    # GP8-02: the substrate keys are load-bearing — the runner builds the EXACT panel + base book a
    # substrate names, so a config naming an unwired panel/sleeve set is a silent no-op; fail fast.
    panel = str(g["panel"])
    if panel not in _WIRED_SUBSTRATES:
        raise ValueError(f"generation.panel must be one of {sorted(_WIRED_SUBSTRATES)} (the wired "
                         f"substrates), got {panel!r}")
    expected = _WIRED_SUBSTRATES[panel]
    if set(map(str, g["base_sleeves"])) != set(expected):
        raise ValueError(f"generation.base_sleeves for panel={panel!r} must be exactly "
                         f"{set(expected)} (the wired book), got {g['base_sleeves']!r}")


def load_generation_config(gates_path: str | Path) -> tuple[FitnessConfig, dict]:
    """Return (FitnessConfig, evolve_kwargs) from a gates YAML's ``generation:`` block.

    ``evolve_kwargs`` holds the loop/runner knobs (rng_seed, pop_size, …); the FitnessConfig
    holds the per-candidate scoring thresholds + combiner params. Raises on out-of-bounds."""
    import yaml

    with open(gates_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    g = {**_GEN_DEFAULTS, **dict(cfg.get("generation", {}))}
    _validate(g)

    fit = FitnessConfig(
        n_groups=int(g["cpcv_n_groups"]), k_test=int(g["cpcv_k_test"]),
        embargo=int(g["cpcv_embargo_days"]), purge_horizon=int(g["cpcv_purge_horizon"]),
        periods_per_year=252.0,        # the book is DAILY-marked (held between rebalances);
                                       # hold_horizon affects turnover only, not Sharpe annualization
        hlz_t_min=float(g["hlz_t_min"]), promising_dsr=float(g["promising_dsr"]),
        min_combination_uplift=float(g["min_combination_uplift"]),
        max_base_corr=float(g["max_base_corr"]),
        delta_median_min=float(g["delta_median_min"]),
        frac_positive_min=float(g["frac_positive_min"]),
        lambda_turnover=float(g["lambda_turnover"]),
        turnover_soft_cap=float(g["turnover_soft_cap"]),
        lambda_complexity=float(g["lambda_complexity"]),
        max_ast_nodes=int(g["max_ast_nodes"]))
    evolve_kwargs = dict(
        rng_seed=int(g["rng_seed"]), pop_size=int(g["pop_size"]),
        n_generations=int(g["n_generations"]), hold_horizon=int(g["hold_horizon"]),
        cost_bps=float(g["cost_bps"]), ls_min_names=int(g["ls_min_names"]),
        holdout_frac=float(g["holdout_frac"]), holdout_embargo=int(g["holdout_embargo_days"]),
        elite_frac=float(g["elite_frac"]))
    return fit, evolve_kwargs


# Cohort-evaluator + MC-null gates (Doc 1/2). An ABSENT ``cohort:`` block ⇒ these defaults ⇒ the
# funnel gates_hash is undisturbed (opt-in, exactly like ``generation.enabled``). The reused floors
# (promising_dsr / hlz_t_min / min_combination_uplift) are read from the ``generation:`` block so a
# single edit keeps the cohort and per-candidate gates in lock-step (Doc 1: no threshold loosened).
_COHORT_DEFAULTS: dict = {
    "enabled": False,               # opt-in; absent/false ⇒ the cohort path never runs
    "min_cohort_size": 3,
    "max_cohort_size": 12,
    "max_pairwise_corr": 0.35,      # admission de-dup on realized return streams (#2a)
    "combiner_redundancy_strength": 0.5,   # λ_r for the COHORT book only (live book stays 0.0)
    "alpha_cohort": 0.05,           # MC-null gate (new pre-registered gate, Doc 2 §3.3)
    "mc_n_replicates": 1000,        # B — binding gate wants ≥1000 (Doc 2 §3.3); 200 for dev
    "mc_block_length": 21,          # ℓ pinned (Politis–White fallback; ℓ-sweep test guards it)
}


def _validate_cohort(c: dict, g: dict) -> None:
    if not (2 <= int(c["min_cohort_size"]) <= int(c["max_cohort_size"])):
        raise ValueError("cohort: require 2 <= min_cohort_size <= max_cohort_size")
    if not (0.0 < float(c["max_pairwise_corr"]) <= 1.0):
        raise ValueError("cohort.max_pairwise_corr must be in (0, 1]")
    if float(c["combiner_redundancy_strength"]) < 0.0:
        raise ValueError("cohort.combiner_redundancy_strength must be >= 0")
    if not (0.0 < float(c["alpha_cohort"]) < 1.0):
        raise ValueError("cohort.alpha_cohort must be in (0, 1)")
    if int(c["mc_n_replicates"]) < 1 or int(c["mc_block_length"]) < 1:
        raise ValueError("cohort.mc_n_replicates and mc_block_length must be >= 1")


def load_cohort_config(gates_path: str | Path) -> tuple[object, dict]:
    """Return (CohortConfig, mc_kwargs) from a gates YAML's ``cohort:`` block (absent ⇒ defaults).

    ``mc_kwargs`` holds the MC-null knobs (``n_reps``, ``alpha_cohort``, ``block_length``, plus
    ``enabled``). The reused floors are inherited from the ``generation:`` block so the cohort book
    is judged at the SAME DSR/HLZ-t/uplift bars as each candidate (Doc 1 sequencing point 4)."""
    import yaml

    from .cohort import CohortConfig

    with open(gates_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    g = {**_GEN_DEFAULTS, **dict(cfg.get("generation", {}))}
    _validate(g)
    c = {**_COHORT_DEFAULTS, **dict(cfg.get("cohort", {}))}
    _validate_cohort(c, g)

    cohort_cfg = CohortConfig(
        max_cohort_size=int(c["max_cohort_size"]), min_cohort_size=int(c["min_cohort_size"]),
        max_pairwise_corr=float(c["max_pairwise_corr"]),
        promising_dsr=float(g["promising_dsr"]),          # REUSE the funnel floors
        cohort_hlz_t_min=float(g["hlz_t_min"]),
        min_book_uplift=float(g["min_combination_uplift"]),
        combiner_redundancy_strength=float(c["combiner_redundancy_strength"]))
    mc_kwargs = dict(
        enabled=bool(c["enabled"]), n_reps=int(c["mc_n_replicates"]),
        alpha_cohort=float(c["alpha_cohort"]), block_length=int(c["mc_block_length"]))
    return cohort_cfg, mc_kwargs


def load_generation_meta(gates_path: str | Path) -> dict:
    """Return the generation substrate ``{enabled, panel, base_sleeves}`` from the gates YAML,
    validated. ``enabled`` is the opt-in gate the standalone runner enforces (GP8-01); ``panel`` /
    ``base_sleeves`` are bounds-checked against the wired book (GP8-02) so editing them to an
    unsupported value fails fast rather than silently no-op-ing."""
    import yaml

    with open(gates_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    g = {**_GEN_DEFAULTS, **dict(cfg.get("generation", {}))}
    _validate(g)
    return {"enabled": bool(g["enabled"]), "panel": str(g["panel"]),
            "base_sleeves": [str(s) for s in g["base_sleeves"]]}
