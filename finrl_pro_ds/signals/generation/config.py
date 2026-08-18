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
    # The cap-rank 51-250 TWSE/TPEx small/mid-cap daily panel
    # (crucible/data/taiwan_smallcap_panel.py) — the substrate the 2026-07/08 probe campaigns were
    # scored on, which until now had an evaluation path and NO mining path.
    #
    # It SHARES the `taiwan` book (TX/TE/TF futures TSMOM) rather than importing the US ETF core the
    # way `us_equity` does, and that is a measurement, not a convention. Both candidates were priced
    # on this panel's own clock at hold 21 / 10bps:
    #   * TX/TE/TF futures TSMOM     SR +0.273, finite on 5291 of 5292 bars
    #   * US ETF {tsmom,rates_carry} SR +0.511, but only 4620 of 5292 bars (87.3%) match an ETF
    #     session — on the other 672 the base book marks nothing while the candidate does.
    # The binding constraint is THE COMPARATOR MUST NOT BLEED, and the Taiwan book clears it (+0.273
    # > 0), so the higher Sharpe does not buy enough to accept a comparator that is flat on 12.7% of
    # the panel — that gap is the same free-pass channel a bleeding book opens. `us_equity` had no
    # such choice: its native comparator (large-cap cross-sectional momentum) is a recorded NO-GO.
    "taiwan_smallcap": frozenset({"tsmom"}),
    # S553-cont-152: top-300 PIT S&P 500 names, daily (crucible/data/us_equity_panel.py). The book is
    # the SAME validated {tsmom, rates_carry} ETF core as `cross_asset`, computed on the ETF panel and
    # joined onto the equity clock by `us_equity_base_sleeves` — deliberately NOT an equity-native
    # book, because the obvious one (large-cap cross-sectional momentum) is a recorded NO-GO and a
    # losing comparator is what lets zero-alpha candidates clear the uplift gate by dilution.
    "us_equity": frozenset({"tsmom", "rates_carry"}),
    # S553-cont-151: the 12-instrument Dukascopy hourly panel + its own TSMOM book
    # (intraday_base_sleeves). There is no intraday rates-carry sleeve, so the book is TSMOM-only —
    # same shape as the Taiwan substrate.
    "intraday": frozenset({"tsmom"}),
    # S553-cont-151: the DERIVED-optimal substrate — 9 FX majors, 2008+, union grid + active mask.
    # Chosen by solving the detection/profitability inequality rather than by search; see
    # crucible/data/intraday_panel.py::build_fx_majors_panel. Same TSMOM-only book.
    "intraday_fx": frozenset({"tsmom"}),
}

_GEN_DEFAULTS: dict = {
    "enabled": False, "panel": "cross_asset", "base_sleeves": ["tsmom", "rates_carry"],
    # BARS PER YEAR on this substrate's clock — the Sharpe annualization factor (S553-cont-151).
    # 252 (daily) is the historical hardcode and stays the default, so every existing config and
    # every fixture is byte-identical. It is a CONFIG key because it is a property of the SUBSTRATE's
    # bar clock, not a threshold: an hourly panel puts 5,694 bars in a calendar year, and reporting
    # its Sharpes at 252/yr shrinks every annualized number by √(252/5694) = 0.21. That mis-scales
    # BOTH the uplift gate (min_combination_uplift is an economic ΔSR floor, so at 252 on an hourly
    # panel it silently demands 4.75x more than it says) and the power stamp's comparison against
    # `plausible_delta_sr_max`, whose 0.3-0.5 justification is calendar-annualized. Measured, not
    # assumed: scripts/research/crucible_frequency_invariance_probe.py.
    "periods_per_year": 252.0,
    "hold_horizon": 21,
    # Rebalance cadence for the BASE book only; None => same as hold_horizon (every existing config,
    # byte-identical). Decoupling exists because a base book is a COMPARATOR, and a comparator that
    # bleeds friction hands the marginal-uplift gate a free pass: blending any lower-turnover stream
    # into a losing book raises the combined Sharpe, so a candidate gets rewarded for trading LESS
    # rather than for predicting. Measured on the intraday substrate
    # (scripts/research/crucible_intraday_null_calibration.py): with the base at hold 21 bars
    # (calendar SR -0.78, 30.6%/yr cost drag), ZERO-ALPHA nulls post a median marginal DSR of +0.367
    # and clear the FULL corrected-contract gate 15.3% of the time against a ~1% nominal — a
    # false-positive factory. The one-basis discipline is preserved: base and candidate are still
    # marked on the same bars and charged the same cost_bps on their own turnover. Only the cadence
    # differs, which is a property of each strategy, not of the accounting.
    "base_hold_horizon": None,
    "pop_size": 200, "n_generations": 40, "rng_seed": 7,
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
    if float(g["periods_per_year"]) <= 0.0:
        raise ValueError("generation.periods_per_year must be > 0 (bars per year on the "
                         "substrate's own bar clock; 252 for daily)")
    if g["base_hold_horizon"] is not None and int(g["base_hold_horizon"]) < 1:
        raise ValueError("generation.base_hold_horizon must be >= 1 bar (or null to track "
                         "hold_horizon)")
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
        # Bars per year on the substrate's clock (default 252 = the historical daily hardcode). The
        # book is marked EVERY BAR (held between rebalances), so hold_horizon affects turnover only,
        # never the annualization — the bar spacing does.
        periods_per_year=float(g["periods_per_year"]),
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
    "analytic_floor_advisory": True,  # analytic SR*_cohort floor records but does not gate (2026-08-09)
    # v12.1 — admit CROSS-SECTIONAL pre-registrations into the cohort pool alongside overlays. FALSE
    # here (so an absent key reproduces the pre-v12.1 overlay-only pool byte-for-byte); the shipped
    # configs/crucible_cohort.gates.yaml sets it TRUE. See that file for the rationale + blast radius.
    "include_cross_sectional": False,
    # v13.0 — apply the funnel's hard-infeasibility (turnover) rule when assembling the cohort pool.
    "enforce_funnel_feasibility": False,
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


def load_cohort_config(
    gates_path: str | Path, cohort_gates_path: str | Path | None = None
) -> tuple[object, dict]:
    """Return (CohortConfig, mc_kwargs) for the weak-signal cohort evaluator (Doc 1/2).

    The ``cohort:`` block is read from ``cohort_gates_path`` (default
    ``configs/crucible_cohort.gates.yaml``) — a SEPARATE file from the funnel gates so adding/enabling
    the cohort gate leaves ``signal_eval.gates.yaml``'s raw bytes (hence the frozen ``crucible-v2.0``
    funnel ``gates_hash``, the moat) BYTE-IDENTICAL (ADR-1, lockbox precedent). The REUSED floors
    (``promising_dsr`` / ``hlz_t_min`` / ``min_combination_uplift``) are always read from
    ``gates_path``'s ``generation:`` block, so the cohort book is judged at the SAME
    DSR/HLZ-t/uplift bars as each candidate (Doc 1 sequencing point 4) and a single edit keeps them in
    lock-step. Back-compat: passing ``cohort_gates_path=gates_path`` (or leaving it None when the
    ``cohort:`` block lives in the funnel file, e.g. legacy tests) reads the block from ``gates_path``.

    ``mc_kwargs`` holds the MC-null knobs (``enabled``, ``n_reps``, ``alpha_cohort``, ``block_length``).
    The provenance ``cohort_gates_hash`` is computed CALLER-side by the orchestrator (symmetric with the
    funnel ``gates_hash``, which ``load_generation_config`` also does not compute) — this keeps the
    ``signals`` layer free of any ``crucible`` import."""
    import yaml

    from .cohort import CohortConfig

    with open(gates_path, encoding="utf-8") as fh:
        gcfg = yaml.safe_load(fh) or {}
    g = {**_GEN_DEFAULTS, **dict(gcfg.get("generation", {}))}
    _validate(g)

    # cohort block: from the dedicated file if given, else from the funnel file (back-compat).
    if cohort_gates_path is None:
        ccfg_raw = gcfg
    else:
        with open(cohort_gates_path, encoding="utf-8") as fh:
            ccfg_raw = yaml.safe_load(fh) or {}
    c = {**_COHORT_DEFAULTS, **dict(ccfg_raw.get("cohort", {}))}
    _validate_cohort(c, g)

    cohort_cfg = CohortConfig(
        max_cohort_size=int(c["max_cohort_size"]), min_cohort_size=int(c["min_cohort_size"]),
        max_pairwise_corr=float(c["max_pairwise_corr"]),
        promising_dsr=float(g["promising_dsr"]),          # REUSE the funnel floors
        cohort_hlz_t_min=float(g["hlz_t_min"]),
        min_book_uplift=float(g["min_combination_uplift"]),
        combiner_redundancy_strength=float(c["combiner_redundancy_strength"]),
        analytic_floor_advisory=bool(c["analytic_floor_advisory"]),
        include_cross_sectional=bool(c["include_cross_sectional"]),
        enforce_funnel_feasibility=bool(c["enforce_funnel_feasibility"]))
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
            "base_sleeves": [str(s) for s in g["base_sleeves"]],
            # None => the runner uses hold_horizon (unchanged for every pre-cont-151 config).
            "base_hold_horizon": (None if g["base_hold_horizon"] is None
                                  else int(g["base_hold_horizon"]))}
