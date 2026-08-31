"""C3.5 tripwires — the generation gates loader (no hardcoded thresholds) + bounds validation."""
from __future__ import annotations

from pathlib import Path

import pytest

from sharpen.signals.generation.config import (
    load_generation_config,
    load_generation_meta,
)

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


# --------------------------------------------------------------------------- #
# Wired-substrate allow-list (GP8-02) — S553-cont step 3: Taiwan substrate accepted
# --------------------------------------------------------------------------- #
def test_taiwan_substrate_is_accepted(tmp_path: Path) -> None:
    """panel=taiwan + TSMOM-only base book (the wired TAIEX substrate) loads without error."""
    p = tmp_path / "tw.yaml"
    p.write_text("generation: {enabled: true, panel: taiwan, base_sleeves: [tsmom]}\n",
                 encoding="utf-8")
    fit, ek = load_generation_config(p)              # bounds + substrate validation must pass
    assert fit.n_groups == 6                         # other defaults still apply
    meta = load_generation_meta(p)
    assert meta["panel"] == "taiwan" and meta["base_sleeves"] == ["tsmom"]


def test_taiwan_substrate_rejects_wrong_base_sleeves(tmp_path: Path) -> None:
    """The Taiwan book is TSMOM-only — naming a sleeve set the runner does not build fails fast."""
    p = tmp_path / "tw_bad.yaml"
    p.write_text("generation: {panel: taiwan, base_sleeves: [tsmom, rates_carry]}\n",
                 encoding="utf-8")
    with pytest.raises(ValueError, match="base_sleeves for panel='taiwan'"):
        load_generation_config(p)


def test_cross_asset_rejects_taiwan_style_sleeves(tmp_path: Path) -> None:
    """Enforcement is PER-PANEL: cross_asset requires {tsmom, rates_carry}, not the Taiwan book."""
    p = tmp_path / "xa_bad.yaml"
    p.write_text("generation: {panel: cross_asset, base_sleeves: [tsmom]}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="base_sleeves for panel='cross_asset'"):
        load_generation_config(p)


def test_unknown_panel_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "unwired.yaml"
    p.write_text("generation: {panel: crypto_perp, base_sleeves: [tsmom]}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="generation.panel must be one of"):
        load_generation_config(p)


def test_taiwan_smallcap_substrate_is_accepted(tmp_path: Path) -> None:
    """panel=taiwan_smallcap + the TSMOM-only TX/TE/TF book (the wired small/mid-cap substrate)."""
    p = tmp_path / "tw_sc.yaml"
    p.write_text("generation: {enabled: true, panel: taiwan_smallcap, base_sleeves: [tsmom]}\n",
                 encoding="utf-8")
    fit, ek = load_generation_config(p)
    assert fit.n_groups == 6
    meta = load_generation_meta(p)
    assert meta["panel"] == "taiwan_smallcap" and meta["base_sleeves"] == ["tsmom"]


def test_taiwan_smallcap_rejects_the_us_book(tmp_path: Path) -> None:
    """The measurement chose the TX/TE/TF book over the US ETF core; naming the latter fails fast
    rather than silently scoring candidates against a book the runner never builds."""
    p = tmp_path / "tw_sc_bad.yaml"
    p.write_text("generation: {panel: taiwan_smallcap, base_sleeves: [tsmom, rates_carry]}\n",
                 encoding="utf-8")
    with pytest.raises(ValueError, match="base_sleeves for panel='taiwan_smallcap'"):
        load_generation_config(p)


def test_shipped_taiwan_smallcap_gates_generation_block_is_valid() -> None:
    """configs/taiwan_smallcap_signal_eval.gates.yaml is the miner's entry point for this substrate.

    `cost_bps` is pinned because it is the substrate's defining constraint: 0.0021 is the Taiwan
    one-way reality (commission + the 0.30% sell-side transaction tax), roughly 2x the US 0.0010 the
    other daily cells use. Pricing a small-cap L/S book at US friction would manufacture edges that
    cannot be traded here."""
    gates = ROOT / "configs" / "taiwan_smallcap_signal_eval.gates.yaml"
    fit, ek = load_generation_config(gates)
    assert fit.periods_per_year == 252.0 and ek["hold_horizon"] == 21
    assert ek["cost_bps"] == 0.0021
    meta = load_generation_meta(gates)
    assert meta["panel"] == "taiwan_smallcap"
    assert meta["base_sleeves"] == ["tsmom"]
    assert meta["enabled"] is False                    # opt-in stays off; a real run passes --force


def test_shipped_taiwan_gates_generation_block_is_valid() -> None:
    """The shipped configs/taiwan_signal_eval.gates.yaml carries a valid `taiwan` generation block
    (step 4) — the runner's opt-in gate is off (real run uses --force) and the book is TSMOM-only."""
    tw_gates = ROOT / "configs" / "taiwan_signal_eval.gates.yaml"
    fit, ek = load_generation_config(tw_gates)         # bounds + substrate validation must pass
    assert fit.periods_per_year == 252.0 and ek["hold_horizon"] == 21
    meta = load_generation_meta(tw_gates)
    assert meta["panel"] == "taiwan"
    assert meta["base_sleeves"] == ["tsmom"]
    assert meta["enabled"] is False                    # opt-in stays off; step-5 run passes --force
