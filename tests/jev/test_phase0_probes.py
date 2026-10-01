"""Tripwires for the Phase-0 statistics that DECIDE things (ATL x Jev plan §4): the C1 item truth and
changepoint, the adopted-cutoff rule, and the K2 planted event signal. Each test fails if the property
the decision rests on is broken — not merely if the script crashes."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sharpen.signals.features import make_synthetic_panel

_R = Path(__file__).resolve().parents[2] / "scripts" / "research"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _R / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cut = _load("jev_cutoff_probe")
summ = _load("atl_jev_phase0_summary")
power = _load("atl_jev_power_check")


def _trending_items(n_tickers=40, p_up=0.9, seed=0):
    """A market that rises in `p_up` of months — the case where a drift prior would look like skill."""
    rng = np.random.default_rng(seed)
    months = cut._months("2018-01", "2022-12")
    closes = {}
    for k in range(n_tickers):
        px = 100.0
        closes[(f"T{k}", cut._prev(months[0]))] = px
        for m in months:
            px *= 1.02 if rng.random() < p_up else 0.98
            closes[(f"T{k}", m)] = px
    univ = [(f"T{k}", f"Co {k}") for k in range(n_tickers)]
    return cut.build_items(univ, months, closes, seed=7), months


def test_item_truth_follows_polarity_and_return_sign():
    items, _ = _trending_items(n_tickers=3)
    for it in items:
        assert it["truth"] == ((it["ret"] > 0) if it["polarity"] == "higher" else (it["ret"] < 0))
        assert it["polarity"] in it["question"]["instructions"]


def test_yes_bias_scores_chance_but_a_drift_prior_scores_the_base_rate():
    """Why the adopted T_c rests on the month-level correlation, not on item accuracy."""
    items, _ = _trending_items()
    truth = np.array([it["truth"] for it in items])
    assert abs(truth.mean() - 0.5) < 0.04                       # always-"yes" is at chance
    drift_prior = np.array([it["polarity"] == "higher" for it in items])      # "stocks rise" prior
    assert (drift_prior == truth).mean() > 0.8                   # ...scores the 90% up-rate, not 0.5


def test_regime_statistic_is_blind_to_a_drift_prior_and_sees_memory():
    """The probe's month-level implied P(up) (polarity undone) is CONSTANT for a drift-prior model, so it
    cannot correlate with realized breadth; a model that remembers each month's move does correlate."""
    items, months = _trending_items(p_up=0.55, seed=3)
    midx = np.array([months.index(it["month"]) for it in items])
    pol_hi = np.array([it["polarity"] == "higher" for it in items])
    up = np.array([it["ret"] > 0 for it in items], dtype=float)
    real = np.bincount(midx, weights=up) / np.bincount(midx)

    def implied_by_month(p_yes):                                # the probe's own formula
        p_up = np.where(pol_hi, p_yes, 1 - p_yes)
        return np.bincount(midx, weights=p_up) / np.bincount(midx)

    drift_prior = implied_by_month(np.where(pol_hi, 0.6, 0.4))            # "stocks rise" prior
    assert np.ptp(drift_prior) == pytest.approx(0.0, abs=1e-12)
    memory = implied_by_month(np.where(pol_hi, up, 1 - up) * 0.8 + 0.1)   # recalls each item's move
    assert np.corrcoef(memory, real)[0, 1] > 0.9


def test_changepoint_recovers_a_planted_split_and_respects_min_segment():
    rng = np.random.default_rng(0)
    n_months, per = 60, 200
    midx = np.repeat(np.arange(n_months), per)
    acc = np.where(midx < 36, 0.75, 0.5)
    correct = (rng.random(midx.size) < acc).astype(float)
    cp = cut.changepoint(midx, correct, n_months, min_seg=6)
    assert abs(cp["split"] - 36) <= 1
    assert cp["acc_early"] == pytest.approx(0.75, abs=0.02) and cp["acc_late"] == pytest.approx(0.5, abs=0.02)
    flat = (rng.random(midx.size) < 0.5).astype(float)
    assert 6 <= cut.changepoint(midx, flat, n_months, min_seg=6)["split"] <= n_months - 6


def test_wilson_interval_known_value():
    lo, hi = cut.wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)


@pytest.mark.parametrize("corrs, embargo, expect", [
    ({"2021": 0.3, "2022": 0.5, "2023": -0.1, "2024": -0.4}, 1, ("2024-01", "2024-02")),
    ({"2021": -0.2, "2022": 0.4, "2023": -0.1, "2024": -0.3}, 1, ("2024-01", "2024-02")),  # later + resets
    ({"2021": 0.3, "2022": 0.2, "2023": -0.1, "2024": -0.3}, 12, ("2024-01", "2025-01")),
    ({"2021": 0.3, "2022": 0.2}, 1, (None, None)),
])
def test_adopted_cutoff_rule(corrs, embargo, expect):
    regime = {y: {"corr_implied_vs_realized": c} for y, c in corrs.items()}
    got = summ.adopted_cutoff(regime, embargo)
    assert (got["T_c"], got["clean_window_from"]) == expect


def test_adopted_cutoff_on_the_measured_2026_09_23_table():
    measured = {"2018": 0.176, "2019": 0.208, "2020": 0.692, "2021": 0.279, "2022": 0.539, "2023": 0.581,
                "2024": -0.105, "2025": -0.397, "2026": -0.719}
    got = summ.adopted_cutoff({y: {"corr_implied_vs_realized": c} for y, c in measured.items()}, 1)
    assert got == {"T_c": "2025-01", "clean_window_from": "2025-02", "first_nonpositive_year": "2024"}


def test_planted_event_signal_coverage_prefix_and_strength():
    # T x N sized so a block-constant noise score has ~(T/21)*N ~ 4.5k effective pairs (corr SE ~0.015)
    p = make_synthetic_panel(T=800, N=120, seed=1)
    block = int(round(0.33 * power.PERIOD))
    strong = power.PlantedEventSignal(p, 0.9, 11, "s", block)
    null = power.PlantedEventSignal(p, 0.0, 12, "n", block)
    s = strong.compute(p)
    cov = np.isfinite(s[power.PERIOD:-power.PERIOD]).sum() / p.active[power.PERIOD:-power.PERIOD].sum()
    assert cov == pytest.approx(block / power.PERIOD, abs=0.05)
    # causal-looking for the Tier-0 truncation tripwire: a truncated panel gets the exact row-prefix
    t = 500
    np.testing.assert_array_equal(strong.compute(p.truncated(t)), s[: t + 1])
    # strength: the score tracks the 21-day post-event drift for rho=0.9 and not for rho=0
    logc = np.log(p.close)
    fwd = np.full_like(logc, np.nan)
    fwd[:-block] = logc[block:] - logc[:-block]

    def corr(x):
        m = np.isfinite(x) & np.isfinite(fwd)
        return np.corrcoef(x[m], fwd[m])[0, 1]
    assert corr(s) > 0.3 and abs(corr(null.compute(p))) < 0.06


def test_planted_signal_rejects_a_non_prefix_panel():
    p = make_synthetic_panel(T=200, N=20, seed=2)
    sig = power.PlantedEventSignal(p, 0.5, 1, "s", 21)
    other = make_synthetic_panel(T=200, N=20, seed=3)
    other_dates = other.dates + np.timedelta64(1, "D")
    import dataclasses
    with pytest.raises(ValueError, match="row-prefix"):
        sig.compute(dataclasses.replace(other, dates=other_dates))
