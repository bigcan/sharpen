"""Tripwires for the one-line FinRL-X rerun (scripts/research/finrl_x_rerun.py).

A fresh clone has no results/finrl_x/hindsight/pool.json, so the growth lists are regenerated from the published pool
and seed. They must be the lists the study ran, FinRL-X's config must change only where pre-registered, and its
interpreter must be found on both virtual-environment layouts. Nothing here downloads data or runs FinRL-X.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "research")):
    if p not in sys.path:
        sys.path.insert(0, p)

import finrl_x_hindsight_runs as hr  # noqa: E402
import finrl_x_rerun as rerun  # noqa: E402


# --------------------------------------------------------------------------- growth lists
def test_published_draws_regenerate_exactly():
    arms = hr.published_arms()
    assert hr.draw_spec_sha256(hr.POOL) == "ebe2c118130ad1372b0a122dfe9828eb77877dc776180794552bf77215859d8f"
    assert arms["rand_02"] == ["ACN", "AMZN", "AVGO", "GOOGL", "MCD", "NVDA", "ORCL"]
    assert arms["rand_24"] == ["AAPL", "CSCO", "DIS", "GOOGL", "META", "NKE", "TXN"]
    rand = [tuple(g) for a, g in arms.items() if a.startswith("rand_")]
    assert len(rand) == 35 and len(set(rand)) == 35
    assert all(len(g) == 7 and set(g) <= set(hr.POOL) for g in rand)
    assert arms["mag7"] == ["AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL", "TSLA"]
    assert arms["pit_top7"] == ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "V", "HD"]
    assert len(arms) == 37


def test_a_changed_pool_breaks_the_fingerprint(monkeypatch):
    monkeypatch.setattr(hr, "POOL", hr.POOL[:-1] + ["TSLA"])
    with pytest.raises(RuntimeError):
        hr.published_arms()


def test_quick_arms_are_a_subset_of_the_full_set(monkeypatch, tmp_path):
    monkeypatch.setattr(hr, "OUT", tmp_path)  # no recorded pool.json: the fresh-clone path
    quick, every = rerun.select_arms(full=False)
    assert list(quick) == ["mag7", "pit_top7", "rand_02"] and len(every) == 37
    assert all(every[a] == g for a, g in quick.items())
    assert rerun.select_arms(full=True)[0] == every


def test_recorded_pool_must_agree(tmp_path):
    arms = hr.published_arms()
    draws = [arms[f"rand_{i:02d}"] for i in range(35)]
    path = tmp_path / "pool.json"
    path.write_text(json.dumps({"deterministic_top7": arms["pit_top7"], "random_draws": draws}))
    rerun.check_recorded_pool(arms, path)
    assert hr.arms(path) == arms and hr.arms(tmp_path / "absent.json") == arms
    draws[2] = draws[3]
    path.write_text(json.dumps({"deterministic_top7": arms["pit_top7"], "random_draws": draws}))
    with pytest.raises(RuntimeError):
        rerun.check_recorded_pool(arms, path)


# --------------------------------------------------------------------------- FinRL-X checkout
@pytest.mark.parametrize("rel", [".venv/Scripts/python.exe", ".venv/bin/python"])
def test_venv_python_is_found_on_both_layouts(tmp_path, rel):
    py = tmp_path / rel
    py.parent.mkdir(parents=True)
    py.write_text("")
    assert hr.venv_python(tmp_path) == py


def test_venv_python_before_the_venv_exists_follows_the_platform(tmp_path, monkeypatch):
    monkeypatch.setattr(hr.os, "name", "posix")
    assert hr.venv_python(tmp_path) == tmp_path / ".venv" / "bin" / "python"
    monkeypatch.setattr(hr.os, "name", "nt")
    assert hr.venv_python(tmp_path) == tmp_path / ".venv" / "Scripts" / "python.exe"


def test_arm_config_touches_only_the_growth_symbols_and_output_paths(tmp_path):
    base = {"asset_groups": {"group_a_growth_tech": {"max_assets": 2, "symbols": ["AAPL", "NVDA"]},
                             "group_b_real_assets": {"symbols": ["GLD"]}},
            "dates": {"start_date": "2017-01-01"},
            "paths": {"data_root": "./d", "output_root": "./o", "state_dir": "./s", "audit_dir": "./a",
                      "weights_dir": "./w"}}
    cfg = hr.arm_config(base, ["INTC", "ORCL"], tmp_path / "rand_00")
    changed = {k for k, v in hr.flatten(cfg).items() if hr.flatten(base)[k] != v}
    assert changed == {hr.GROUP_KEY} | {("paths", k) for k in hr.PATH_KEYS}
    assert base["asset_groups"]["group_a_growth_tech"]["symbols"] == ["AAPL", "NVDA"]  # base not mutated


def test_layouts_keep_the_original_campaign_apart_from_the_rerun(tmp_path):
    lay = rerun.rerun_layout(tmp_path)
    assert lay.runs == lay.out == rerun.OUT and lay.data_dir == rerun.OUT / "prices"
    assert tmp_path not in lay.data_dir.parents  # prices are written outside the FinRL-X checkout
    orig = hr.hindsight_layout()
    assert orig.runs == hr.OUT / "runs" and rerun.OUT not in orig.runs.parents


# --------------------------------------------------------------------------- printed table
def _report(cagr0: float, parity: list[str]) -> dict:
    grid = {b: {"cumulative_x": 4.0, "ann_return": cagr0 - 0.01 * i, "sharpe_arith": 1.0}
            for i, b in enumerate(("0", "2", "3", "5", "10"))}
    return {"cost_grid": grid, "parity_problems": parity,
            "benchmarks": {"QQQ": {"cumulative_x": 3.95, "ann_return": 0.193, "sharpe_arith": 0.91}},
            "strategy_0bps": {"annual_turnover": 62.5},
            "breakeven_vs_QQQ_bps": {"ann_return": 4.48, "sharpe_arith": 5.34}}


def test_table_reports_the_excess_at_two_bps_and_parity():
    arms = {a: [] for a in rerun.QUICK_ARMS}
    lines = rerun.render({"mag7": _report(0.22, []), "pit_top7": _report(0.13, ["x"]), "rand_02": _report(0.19, [])}, arms)
    mag7 = next(ln for ln in lines if ln.startswith("Magnificent 7 (as published)"))
    assert "22.00% / 21.00% / 18.00%" in mag7 and "+1.70 pts" in mag7 and mag7.endswith("PASS")
    assert next(ln for ln in lines if ln.startswith("7 largest")).endswith("FAIL")
    assert any(ln.startswith("Best random list (rand_02)") for ln in lines)
    assert any("4.48 bps per side on return, 5.34 on Sharpe" in ln for ln in lines)


def test_full_summary_counts_the_random_lists_that_beat_qqq():
    rows = {"mag7": {"d_cagr_2": 0.02}, "pit_top7": {"d_cagr_2": -0.07},
            "rand_00": {"d_cagr_2": -0.09}, "rand_01": {"d_cagr_2": 0.01}, "rand_02": {"d_cagr_2": -0.12}}
    call = {"call": "INCONCLUSIVE"}
    text = "\n".join(rerun.render_verdict({"rows": rows, "verdict": "INCONCLUSIVE", "primary": call,
                                           "sharpe_reading": call, "paper_cost_reading": call}))
    assert "beating QQQ at 2 bps: 1 of 3" in text
    assert "median -9.00, best rand_01 +1.00, worst rand_02 -12.00" in text
    assert "Magnificent 7 beats 3 of 3" in text
