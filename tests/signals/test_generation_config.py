"""C3.5 tripwires — the generation gates loader (no hardcoded thresholds) + bounds validation."""
from __future__ import annotations

from pathlib import Path

import pytest

from finrl_pro_ds.signals.generation.config import load_generation_config

ROOT = Path(__file__).resolve().parents[2]
GATES = ROOT / "configs" / "signal_eval.gates.yaml"


def test_loads_fitness_config_and_evolve_kwargs_from_real_gates() -> None:
    fit, ek = load_generation_config(GATES)
    # values come from configs/signal_eval.gates.yaml::generation, not code defaults
    assert fit.n_groups == 6 and fit.k_test == 2
    assert fit.periods_per_year == 252.0          # daily-marked book (not 252/hold)
    assert fit.promising_dsr == 0.90 and fit.hlz_t_min == 3.0
    assert ek["pop_size"] == 200 and ek["hold_horizon"] == 21
    assert 0.0 < ek["holdout_frac"] < 1.0


def test_absent_block_falls_back_to_safe_defaults(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("deflation: {promising_dsr: 0.9}\n", encoding="utf-8")
    fit, ek = load_generation_config(p)            # no generation block → defaults, no crash
    assert fit.max_ast_nodes == 24 and ek["rng_seed"] == 7


def test_bounds_validation_rejects_bad_block(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("generation: {cpcv_n_groups: 2, cpcv_k_test: 5}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="k_test"):
        load_generation_config(p)
