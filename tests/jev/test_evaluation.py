"""The screening legs (architecture §6, ADR-11). P2 and P3 are validated in both directions: a planted, informative
signal must pass and pure noise must fail. The placebo shuffles only within a release week, the P3 t follows the
Newey-West definition used by the funnel, and K3 keeps the funnel's rank order."""
from __future__ import annotations

import copy
import dataclasses
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sharpen.jev.evaluation import (
    fama_macbeth_betas,
    hac_t,
    k3_decision,
    permute_within_weeks,
    registered,
    release_weeks,
    run_p2,
    run_p3,
)
from sharpen.signals.features import make_synthetic_panel
from sharpen.signals.gates import Gates

ROOT = Path(__file__).resolve().parents[2]
PHASE1 = yaml.safe_load((ROOT / "configs" / "atl_jev.gates.yaml").read_text(encoding="utf-8"))["phase1"]
GATES = dataclasses.replace(Gates.from_yaml(ROOT / "configs" / "signal_eval.gates.yaml"), min_names_per_day=10)


def _p1(perms=19):
    p = copy.deepcopy(PHASE1)
    p["p2_placebo"]["permutations"] = perms
    return p


P = make_synthetic_panel(T=300, N=40, seed=3)
CAL = P.dates.astype("datetime64[D]")
FWD5 = P.forward_returns(5)


def _scores(value_fn, every=20, seed=0):
    """One filing per name every ``every`` rows, released that day (14:00Z is before 15:30 ET)."""
    rng = np.random.default_rng(seed)
    rows = []
    for j, t in enumerate(P.tickers):
        for r in range(j % every, P.T - 6, every):
            v = value_fn(r, j, rng)
            rows.append({"accession": f"{j}-{r}", "cik": j, "tickers": t,
                         "accepted_utc": np.datetime64(str(CAL[r]) + "T14:00", "ns"),
                         "surprise": v, "tone": math.nan, "quality": math.nan})
    return pd.DataFrame(rows)


PLANTED = _scores(lambda r, j, rng: FWD5[r, j])                  # knows the next 5 days
NOISE = _scores(lambda r, j, rng: rng.normal())


# --------------------------------------------------------------------------- P2
def test_permutation_stays_within_a_week_and_moves_block_triples_together():
    s = pd.DataFrame({"surprise": [1.0, 2.0, 3.0, 4.0, 5.0], "tone": [10.0, 20.0, 30.0, 40.0, 50.0],
                      "quality": [-1.0, -2.0, -3.0, -4.0, -5.0]})
    weeks = np.array([7, 7, 7, 9, -1])
    for seed in range(20):
        out = permute_within_weeks(s, weeks, np.random.default_rng(seed))
        assert sorted(out.surprise[:3]) == [1.0, 2.0, 3.0], "scores left their release week"
        assert out.surprise[3] == 4.0 and out.surprise[4] == 5.0, "a singleton or unreleased filing moved"
        assert (out.tone == out.surprise * 10).all() and (out.quality == -out.surprise).all()


def test_release_weeks_follow_the_release_row_not_the_acceptance_day():
    sat_after_close = np.datetime64("2024-01-05T21:30", "ns")     # Friday after the close -> Monday's row
    s = pd.DataFrame({"accepted_utc": [np.datetime64("2024-01-05T14:00", "ns"), sat_after_close,
                                       np.datetime64("2030-01-02T14:00", "ns")]})
    cal = np.array([d for d in np.arange("2024-01-01", "2024-01-20", dtype="datetime64[D]") if np.is_busday(d)])
    w = release_weeks(s, cal, PHASE1)
    assert w[0] == 202401 and w[1] == 202402 and w[2] == -1


def test_p2_passes_a_planted_signal_and_is_deterministic():
    p1 = _p1()
    a = run_p2("jev-surprise-63", p1, PLANTED, CAL, P, GATES)
    assert a["real_ic"] > 0.1 and a["placebo_ge_real"] == 0 and a["p"] == pytest.approx(1 / 20) and a["pass"]
    assert run_p2("jev-surprise-63", p1, PLANTED, CAL, P, GATES) == a


def test_p2_fails_noise():
    r = run_p2("jev-surprise-63", _p1(), NOISE, CAL, P, GATES)
    assert abs(r["real_ic"]) < 0.05 and not r["pass"]


def test_p2_fails_closed_on_a_nan_ic():
    empty = PLANTED.iloc[:0]
    r = run_p2("jev-surprise-63", _p1(perms=3), empty, CAL, P, GATES)
    assert math.isnan(r["real_ic"]) and r["p"] == 1.0 and not r["pass"]


# --------------------------------------------------------------------------- P3
def test_hac_t_reduces_to_the_iid_t_without_lags():
    b = np.random.default_rng(1).normal(0.2, 1.0, 400)
    t, n_eff = hac_t(b, 0)
    assert n_eff == 400 and t == pytest.approx(b.mean() / math.sqrt(np.var(b) / 400))
    assert math.isnan(hac_t(np.array([1.0, 2.0]), 5)[0])


def test_hac_t_discounts_autocorrelated_coefficients():
    rng = np.random.default_rng(2)
    b = np.empty(500)
    b[0] = 0.0
    for i in range(1, 500):                                  # overlapping 5-day returns make daily betas persist
        b[i] = 0.9 * b[i - 1] + rng.normal()
    b += 0.1
    t_iid, n_iid = hac_t(b, 0)
    t_nw, n_eff = hac_t(b, 5)
    assert n_eff < 0.5 * n_iid and abs(t_nw) < abs(t_iid)


def test_fama_macbeth_credits_only_marginal_information():
    rng = np.random.default_rng(7)
    T, N = 250, 60
    x_base, x_new, x_irrelevant, noise = (rng.normal(size=(T, N)) for _ in range(4))
    act = np.ones((T, N), bool)
    # the first regressor adds information beyond the baseline: its coefficient is clearly positive
    t_new, _ = hac_t(fama_macbeth_betas(0.3 * x_base + 0.3 * x_new + noise, [x_new, x_base], act, 10), 5)
    # the returns load only on the baseline: an irrelevant first regressor earns no credit
    t_none, _ = hac_t(fama_macbeth_betas(0.5 * x_base + noise, [x_irrelevant, x_base], act, 10), 5)
    assert t_new > 5 and abs(t_none) < 2.5


def _baselines(lm_fn, sim_fn):
    b = PLANTED[["accession", "cik", "tickers", "accepted_utc"]].copy()
    rng = np.random.default_rng(11)
    b["lm_tone"] = [lm_fn(i, rng) for i in range(len(b))]
    b["similarity"] = [sim_fn(i, rng) for i in range(len(b))]
    return b


def test_p3_passes_when_jev_carries_information_the_baselines_lack():
    base = _baselines(lambda i, rng: rng.normal(), lambda i, rng: rng.normal())
    r = run_p3("jev-surprise-63", PHASE1, PLANTED, base, CAL, P, GATES)
    assert r["t"] > PHASE1["p3_baseline"]["min_marginal_t"] and r["pass"] and r["n_days"] > 100


def test_p3_fails_when_the_information_is_the_baselines_and_jev_is_noise():
    base = _baselines(lambda i, rng: PLANTED.surprise.iloc[i], lambda i, rng: rng.normal())
    r = run_p3("jev-surprise-63", PHASE1, NOISE.assign(accession=PLANTED.accession), base, CAL, P, GATES)
    assert r["t"] < PHASE1["p3_baseline"]["min_marginal_t"] and not r["pass"]


def test_p2_and_p3_bars_are_read_from_phase1():
    strict = _p1()
    strict["p2_placebo"]["max_p"] = 0.01                        # below the 1/20 floor of 19 permutations
    assert not run_p2("jev-surprise-63", strict, PLANTED, CAL, P, GATES)["pass"]
    base = _baselines(lambda i, rng: rng.normal(), lambda i, rng: rng.normal())
    hard = copy.deepcopy(PHASE1)
    hard["p3_baseline"]["min_marginal_t"] = 1e6
    hard["p3_baseline"]["nw_lags"] = 1
    r = run_p3("jev-surprise-63", hard, PLANTED, base, CAL, P, GATES)
    assert not r["pass"] and r["nw_lags"] == 1 and r["min_t"] == 1e6


# --------------------------------------------------------------------------- K3
def test_k3_keeps_rank_order_and_fires_without_a_passer():
    ok, bad = {"pass": True}, {"pass": False}
    d = k3_decision(["b", "a", "c", "d"], {"a": ok, "b": ok, "c": ok, "d": bad},
                    {"a": ok, "b": bad, "c": ok, "d": ok})
    assert d["passers"] == ["a", "c"] and d["p4_candidate"] == "a" and not d["k3_fired"], \
        "b fails P3 and d fails P2: neither may pass screening"
    none = k3_decision(["a"], {"a": ok}, {})
    assert none["k3_fired"] and none["p4_candidate"] is None and none["verdict"].startswith("NO-GO")


def test_registered_rejects_an_unregistered_name():
    with pytest.raises(ValueError, match="not registered"):
        registered(PHASE1, PLANTED, CAL, ["jev-made-up"])
