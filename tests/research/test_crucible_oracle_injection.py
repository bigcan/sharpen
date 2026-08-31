"""Tests for the Crucible F2 oracle-injection demo (scripts/research/crucible_oracle_injection.py, cont-139).

The figure's claim: the shipped gate promotes an oracle only above an implausible realized-Sharpe WALL,
so a perfect-foresight timing oracle is rejected at every realistic frequency. These tests pin the
construction (perfect skill beats no-skill; realized Sharpe falls with hold), the wall (daily perfect
foresight passes, slow perfect foresight fails), and the deterministic wall/skill logic.
"""
from __future__ import annotations

import numpy as np

import scripts.research.crucible_oracle_injection as oi
from sharpen.signals.generation.fitness import FitnessConfig

_CFG = FitnessConfig(embargo=10)


def test_oracle_construction_perfect_beats_noskill_and_daily_beats_weekly() -> None:
    """Perfect foresight realizes a large positive Sharpe; no skill (p=0.5) ~0; and a shorter hold
    realizes a higher Sharpe than a longer one (the axis the promotion wall is mapped over)."""
    market = oi._market(3000, seed=1)
    perfect_daily, _ = oi._oracle_return(market, hold=1, skill_p=1.0, seed=0, cost_bps=0.0)
    noskill_daily, _ = oi._oracle_return(market, hold=1, skill_p=0.5, seed=0, cost_bps=0.0)
    perfect_weekly, _ = oi._oracle_return(market, hold=5, skill_p=1.0, seed=0, cost_bps=0.0)
    assert oi._ann_sharpe(perfect_daily) > 5.0                 # perfect daily foresight = huge Sharpe
    assert abs(oi._ann_sharpe(noskill_daily)) < 1.0            # p=0.5 (accuracy) -> ~0 Sharpe (no skill)
    assert oi._ann_sharpe(perfect_daily) > oi._ann_sharpe(perfect_weekly)   # SR falls as hold grows


def test_zero_sharpe_base_is_near_zero() -> None:
    base = oi._zero_sharpe_base(3000, seed=2)
    for v in base.values():
        # 0-drift noise -> ~0 Sharpe (sampling gives |SR| well under 1 on this length)
        assert abs(float(np.mean(v) / np.std(v) * np.sqrt(252))) < 1.0


def test_promotion_wall_logic() -> None:
    """Deterministic: the wall sits between the lowest-Sharpe PROMOTED oracle and the highest-Sharpe
    REJECTED one (no dependence on the stochastic gate scores)."""
    rows = [{"mean_own_sharpe": 18.0, "detection_rate": 1.0},
            {"mean_own_sharpe": 5.7, "detection_rate": 0.85},
            {"mean_own_sharpe": 2.7, "detection_rate": 0.20},
            {"mean_own_sharpe": 1.5, "detection_rate": 0.0}]
    w = oi._promotion_wall(rows)
    assert w["lowest_promoted_sharpe"] == 5.7
    assert w["highest_rejected_sharpe"] == 2.7
    assert 2.7 < w["wall_sharpe_approx"] < 5.7                 # an implausible promotion Sharpe


def test_hold_sweep_daily_passes_slow_perfect_foresight_fails() -> None:
    """The core: perfect DAILY foresight passes, but perfect SLOW foresight (lower realized Sharpe) is
    rejected — the gate's bar is a realized Sharpe far above any realistic frequency's reach."""
    hs = oi.run_hold_sweep(_CFG, t=1011, n_seeds=8, cost_bps=oi._COST_BPS, gen_n_eff=oi._GEN_N_EFF)
    rows = {r["label"]: r for r in hs["rows"]}
    assert rows["daily"]["detection_rate"] > 0.5              # perfect daily foresight IS promoted
    assert rows["semiannual"]["detection_rate"] < 0.5        # perfect slow foresight is NOT
    # realized Sharpe falls monotonically as the hold lengthens (the wall axis)
    srs = [rows[lbl]["mean_own_sharpe"] for _, lbl in oi._HOLDS]
    assert all(srs[i] > srs[i + 1] for i in range(len(srs) - 1)), srs
    # a perfect oracle with a genuinely high realized Sharpe (>1.5) is still rejected -> implausible wall
    assert hs["promotion_wall"]["highest_rejected_sharpe"] > 1.5


def test_skill_sweep_perfect_foresight_at_realistic_sharpe_is_rejected() -> None:
    """At a realistic-edge frequency (quarterly; perfect-foresight realized SR ~1.5), NO skill level —
    including perfect foresight — is promoted. The silver-platter rejection."""
    ss = oi.run_skill_sweep(_CFG, t=1011, hold=oi._SKILL_SWEEP_HOLD, n_seeds=8,
                            cost_bps=oi._COST_BPS, gen_n_eff=oi._GEN_N_EFF)
    assert ss["max_detection_rate"] < 0.5                     # never promoted, even at perfect skill
    perfect = next(r for r in ss["rows"] if r["skill_p"] == 1.0)
    assert perfect["detection_rate"] == 0.0
    assert perfect["mean_own_sharpe"] > ss["rows"][0]["mean_own_sharpe"]   # skill still raises realized SR
