"""Phase 4 Step 2 — cohort gates in a SEPARATE file keeps the frozen funnel gates_hash byte-identical.

The load-bearing regression: adding/enabling the weak-signal cohort gate must NOT change
``configs/signal_eval.gates.yaml``'s raw bytes, so the frozen ``crucible-v2.0`` moat hash
(519158fa1450) stays IDENTICAL (ADR-1, mirroring the lockbox precedent). If someone later folds a
``cohort:`` block into the funnel file "for convenience", these tests fail loudly.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sharpen.crucible.version import gates_hash
from sharpen.signals.generation.config import (
    load_cohort_config,
    load_generation_config,
)

ROOT = Path(__file__).resolve().parents[1]
FUNNEL = ROOT / "configs" / "signal_eval.gates.yaml"
COHORT = ROOT / "configs" / "crucible_cohort.gates.yaml"

# The frozen crucible-v2.0 funnel moat hash (named in configs/crucible_lockbox.gates.yaml). The cohort
# integration must leave this UNTOUCHED — that is the whole point of the separate-file design (ADR-1).
FROZEN_FUNNEL_HASH = "519158fa1450"


def test_funnel_gates_hash_is_frozen() -> None:
    """The moat regression: the funnel gates file's hash is unchanged by the cohort integration."""
    assert gates_hash(FUNNEL) == FROZEN_FUNNEL_HASH


def test_funnel_file_has_no_cohort_block() -> None:
    """ADR-1 guard: the cohort gates must live in their OWN file, never in signal_eval.gates.yaml."""
    cfg = yaml.safe_load(FUNNEL.read_text(encoding="utf-8")) or {}
    assert "cohort" not in cfg, "cohort gates must NOT be in the funnel file (ADR-1) — use crucible_cohort.gates.yaml"


def test_cohort_config_loads_from_separate_file() -> None:
    ccfg, mc = load_cohort_config(FUNNEL, COHORT)
    # ENABLED 2026-08-10 by operator authorisation (see the header of crucible_cohort.gates.yaml).
    # Previously asserted False as "opt-in; shipped disabled".
    assert mc["enabled"] is True
    # The analytic SR*_cohort floor is ADVISORY: recorded on the card, but it does not gate, so the
    # binding selection-aware MC null is reachable (2026-08-09 root-cause report §5b Finding 5).
    assert ccfg.analytic_floor_advisory is True
    assert ccfg.min_cohort_size == 3 and ccfg.max_cohort_size == 12
    assert ccfg.max_pairwise_corr == 0.35
    assert ccfg.combiner_redundancy_strength == 0.5
    assert mc["n_reps"] == 1000 and mc["alpha_cohort"] == 0.05 and mc["block_length"] == 21


def test_cohort_enabled_and_advisory_remain_opt_out_able() -> None:
    """The MECHANISM, not the shipped value: both switches must still be honoured when set off, so
    the pre-2026-08-10 behaviour stays reachable and reproducible."""
    import tempfile
    from pathlib import Path

    raw = yaml.safe_load(COHORT.read_text(encoding="utf-8")) or {}
    raw["cohort"]["enabled"] = False
    raw["cohort"]["analytic_floor_advisory"] = False
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cohort_off.yaml"
        p.write_text(yaml.safe_dump(raw), encoding="utf-8")
        ccfg, mc = load_cohort_config(FUNNEL, p)
    assert mc["enabled"] is False
    assert ccfg.analytic_floor_advisory is False


def test_reused_floors_come_from_generation_block() -> None:
    """The reused funnel floors are read from signal_eval.gates.yaml::generation (lock-step), NOT
    duplicated in the cohort file — a single edit keeps cohort and per-candidate gates aligned."""
    ccfg, _ = load_cohort_config(FUNNEL, COHORT)
    fit, _ = load_generation_config(FUNNEL)
    assert ccfg.promising_dsr == fit.promising_dsr
    assert ccfg.cohort_hlz_t_min == fit.hlz_t_min
    assert ccfg.min_book_uplift == fit.min_combination_uplift


def test_cohort_and_funnel_hashes_are_distinct_and_deterministic() -> None:
    fh, ch = gates_hash(FUNNEL), gates_hash(COHORT)
    assert fh != ch                                     # each gate set has its own provenance hash
    assert gates_hash(FUNNEL) == fh and gates_hash(COHORT) == ch   # deterministic (raw-bytes SHA-256)


def test_backcompat_reads_cohort_from_funnel_when_no_separate_path() -> None:
    """Legacy path: no cohort_gates_path ⇒ read the cohort block from the funnel file. The funnel has
    no cohort block, so this yields the disabled defaults (byte-identical pre-cohort behavior)."""
    ccfg, mc = load_cohort_config(FUNNEL)
    assert mc["enabled"] is False and ccfg.min_cohort_size == 3


def test_enabled_cohort_file_is_honored(tmp_path: Path) -> None:
    p = tmp_path / "cohort_on.gates.yaml"
    p.write_text("cohort:\n  enabled: true\n  min_cohort_size: 4\n", encoding="utf-8")
    ccfg, mc = load_cohort_config(FUNNEL, p)
    assert mc["enabled"] is True and ccfg.min_cohort_size == 4   # others fall back to defaults


def test_malformed_cohort_bounds_raise(tmp_path: Path) -> None:
    p = tmp_path / "bad.gates.yaml"
    p.write_text("cohort:\n  max_pairwise_corr: 1.5\n", encoding="utf-8")   # must be in (0, 1]
    with pytest.raises(ValueError):
        load_cohort_config(FUNNEL, p)
