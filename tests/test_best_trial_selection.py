"""Tests for `_select_best_trial` — the buy-and-hold hurdle applied at SELECTION.

Why selection and not the objective: returning a sentinel for every trial that misses the
hurdle makes every trial score identically, TPE sees a flat landscape, and the search
silently becomes random. That was measured live on gmgp1-spx500 (9 completed trials per
arm, all -999). The objective therefore stays informative and the gate moves here.

The behaviour that matters, and is easy to get wrong:
  * a qualifying trial must win even when a NON-qualifying trial has a higher raw value
  * when nothing qualifies, the run must still yield hyperparameters, but loudly
  * studies that never ran the hurdle at all must be untouched
"""
import optuna
import pytest

from scripts.run_full_pipeline import _select_best_trial


def _study(specs):
    """specs: list of (value, beat_buy_hold or None)."""
    study = optuna.create_study(direction="maximize")
    for value, beat in specs:
        attrs = {} if beat is None else {"beat_buy_hold": beat}
        study.add_trial(optuna.trial.create_trial(
            params={}, distributions={}, value=value, user_attrs=attrs,
            state=optuna.trial.TrialState.COMPLETE))
    return study


def test_prefers_qualifying_trial_over_higher_scoring_failure():
    """THE case that matters: a sub-benchmark trial with a better PF must NOT win."""
    study = _study([(2.5, False), (1.2, True), (0.9, True)])
    best = _select_best_trial(study)
    assert best.value == pytest.approx(1.2)
    assert best.user_attrs["beat_buy_hold"] is True


def test_picks_highest_among_qualifying():
    study = _study([(1.1, True), (1.9, True), (1.5, True)])
    assert _select_best_trial(study).value == pytest.approx(1.9)


def test_falls_back_when_nothing_qualifies():
    """Must still return something so downstream stages get hyperparameters."""
    study = _study([(1.4, False), (0.8, False)])
    best = _select_best_trial(study)
    assert best.value == pytest.approx(1.4)


def test_study_without_hurdle_is_unaffected():
    """Workstreams that never enabled the hurdle must behave exactly as before."""
    study = _study([(1.1, None), (2.2, None), (0.5, None)])
    assert _select_best_trial(study).value == pytest.approx(2.2)


def test_ignores_failed_trials():
    study = optuna.create_study(direction="maximize")
    study.add_trial(optuna.trial.create_trial(
        params={}, distributions={}, value=1.3,
        user_attrs={"beat_buy_hold": True}, state=optuna.trial.TrialState.COMPLETE))
    study.add_trial(optuna.trial.create_trial(
        params={}, distributions={}, state=optuna.trial.TrialState.FAIL))
    assert _select_best_trial(study).value == pytest.approx(1.3)


def test_single_qualifying_trial_wins_against_many_failures():
    study = _study([(3.0, False), (2.9, False), (1.01, True), (2.8, False)])
    assert _select_best_trial(study).value == pytest.approx(1.01)
