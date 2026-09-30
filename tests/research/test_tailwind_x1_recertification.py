"""Tripwires for the TAILWIND X1 re-certification machinery (scripts/research/tailwind_x1_recertification.py).

Data-free: every test runs on synthetic returns or on the prereg file itself. None computes a
TAILWIND number; the certifying pass runs once, after operator sign-off.

Pinned both ways where a silent default could flip a verdict:
  * Wilson reproduces the audit's 14/21 interval, and n == 0 is undefined, not 0.
  * The walk is DISJOINT, advances on the run it scores, and right-censors both TIMEOUT kinds.
  * intraday_mae_mult reaches the simulator: a -3.5% day breaches a 4% halt at 1.4, not at 1.0.
  * A None statistic fails closed; a PASS with an integrity warning is REVIEW; exit map is N3.
  * Pins and the offline guard refuse, and the prereg ledger's arithmetic holds.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sharpen.prop.challenge_simulator import DAILY_BREACH, PASS, TIMEOUT, FirmRules, SizingPolicy

ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = ROOT / "scripts" / "research" / "tailwind_x1_recertification.py"


@pytest.fixture(scope="module")
def x1():
    spec = importlib.util.spec_from_file_location("tailwind_x1_recertification", _SCRIPT)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _rules(dd=0.10, daily=0.05, target=0.10, min_days=5) -> FirmRules:
    return FirmRules(name="t", profit_target=target, max_total_dd=dd, daily_loss_limit=daily,
                     max_days=None, min_trading_days=min_days, dd_mode="static")


POL = SizingPolicy(vol_multiplier=1.0)


# ---------------------------------------------------------------------------------- Wilson
def test_wilson_reproduces_audit_interval(x1):
    lo, hi = x1.wilson(14, 21)
    assert (round(lo, 3), round(hi, 3)) == (0.454, 0.828)


def test_wilson_edges(x1):
    assert x1.wilson(0, 0) is None
    lo, hi = x1.wilson(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12) and 0.0 < hi < 0.35
    lo, hi = x1.wilson(10, 10)
    assert hi == pytest.approx(1.0, abs=1e-12) and 0.65 < lo < 1.0
    with pytest.raises(ValueError):
        x1.wilson(11, 10)


# ---------------------------------------------------------------------------- disjoint walk
def test_walk_is_disjoint_and_starts_after_resolution(x1):
    r = np.full(30, 0.01)                    # 1.01**10 = 1.1046 >= 10% target on day 10
    rows = x1.disjoint_walk(r, _rules(), POL, horizon=756)
    assert [x["i"] for x in rows] == [0, 10, 20]
    assert all(x["outcome"] == PASS and x["days"] == 10 for x in rows)
    s = x1.summarize_walk(rows)
    assert (s["k_pass"], s["n_resolved"], s["p_pass"]) == (3, 3, 1.0)


def test_data_end_timeout_is_right_censored(x1):
    r = np.full(35, 0.01)                    # 3 passes, then 5 sessions that cannot reach target
    s = x1.summarize_walk(x1.disjoint_walk(r, _rules(), POL, horizon=756))
    assert (s["n_windows"], s["n_resolved"], s["n_censored_data_end"]) == (4, 3, 1)
    assert s["p_pass"] == 1.0                # the censored window is not a failure


def test_horizon_cap_timeout_is_right_censored_and_jumps_the_horizon(x1):
    r = np.zeros(12)
    rows = x1.disjoint_walk(r, _rules(), POL, horizon=5)
    assert [x["i"] for x in rows] == [0, 5, 10]
    assert [x["censored"] for x in rows] == ["horizon_cap", "horizon_cap", "data_end"]
    s = x1.summarize_walk(rows)
    assert s["n_resolved"] == 0 and s["p_pass"] is None and s["wilson_lower"] is None
    # M2 diagnostic: horizon-cap TIMEOUTs counted as fails (data-end stays censored)
    assert s["p_pass_horizon_timeouts_as_fail"] == 0.0


def test_m2_diagnostic_is_never_above_the_censored_estimate(x1):
    # 1.025**5 = 1.131: pass on day 5, a flat 5-day horizon cap, then another pass
    r = np.concatenate([np.full(5, 0.025), np.zeros(5), np.full(5, 0.025)])
    s = x1.summarize_walk(x1.disjoint_walk(r, _rules(), POL, horizon=5))
    assert (s["k_pass"], s["n_censored_horizon_cap"]) == (2, 1)
    assert s["p_pass_horizon_timeouts_as_fail"] == pytest.approx(2 / 3)
    assert s["p_pass_horizon_timeouts_as_fail"] < s["p_pass"]
    assert s["wilson_lower_horizon_timeouts_as_fail"] < s["wilson_lower"]


def test_cursor_advances_on_the_scored_run(x1):
    # day 0 loses 4.5%: the kills (4% halt) end the window on day 1, the firm (5%) does not.
    r = np.array([-0.045] + [0.02] * 12)
    kills, firm = _rules(dd=0.08, daily=0.04), _rules()
    k_rows = x1.disjoint_walk(r, kills, POL, horizon=756)
    f_rows = x1.disjoint_walk(r, firm, POL, horizon=756)
    assert k_rows[0]["outcome"] == DAILY_BREACH and k_rows[1]["i"] == 1
    assert f_rows[0]["outcome"] == PASS and f_rows[0]["i"] == 0 and len(f_rows) >= 1


def test_paired_needless_counts_firm_passes_the_kills_end(x1):
    r = np.array([-0.045] + [0.02] * 12)
    rows = x1.disjoint_walk(r, _rules(), POL, horizon=756, paired=_rules(dd=0.08, daily=0.04))
    s = x1.summarize_walk(rows)
    assert rows[0]["outcome"] == PASS and rows[0]["paired"] == DAILY_BREACH
    assert s["n_needless"] == 1 and s["needless_share"] == pytest.approx(1 / s["n_lead_passes"])


def test_intraday_mae_mult_reaches_the_simulator(x1):
    r = np.array([-0.035] + [0.02] * 12)     # -3.5%: under a 4% halt; x1.4 = -4.9% breaches it
    kills = _rules(dd=0.08, daily=0.04)
    at_1 = x1.disjoint_walk(r, kills, SizingPolicy(intraday_mae_mult=1.0), horizon=756)
    at_14 = x1.disjoint_walk(r, kills, SizingPolicy(intraday_mae_mult=1.4), horizon=756)
    assert at_1[0]["outcome"] == PASS
    assert at_14[0]["outcome"] == DAILY_BREACH


def test_p_pass_by_mae_applies_each_multiplier_to_the_gate_read(x1):
    # 0.965 * 1.015**9 = 1.103: one window reaches target on day 10 and one session is left.
    r = np.array([-0.035] + [0.015] * 10)
    out = x1.p_pass_by_mae(r, _rules(), _rules(dd=0.08, daily=0.04), [1.0, 1.4], horizon=756)
    assert set(out) == {"mae_1.0", "mae_1.4"}
    assert out["mae_1.0"]["declared_kills"]["k_pass"] == 1
    assert out["mae_1.0"]["declared_kills"]["k_daily_breach"] == 0
    assert out["mae_1.4"]["declared_kills"]["k_daily_breach"] == 1
    assert out["mae_1.4"]["firm_only"]["k_pass"] == 1           # -4.9% stays under the firm's 5%
    assert out["mae_1.4"]["firm_with_paired_kills"]["n_needless"] == 1


def test_walk_rejects_non_finite(x1):
    with pytest.raises(ValueError):
        x1.disjoint_walk(np.array([0.01, np.nan]), _rules(), POL, horizon=10)


def test_timeout_constant_is_the_simulators(x1):
    assert x1.TIMEOUT == TIMEOUT


# --------------------------------------------------------------------------------- verdict
def test_verdict_rule(x1):
    ok_d, ok_p = {"64": 0.96, "77": 0.955}, {"mae_1.0": 0.70, "mae_1.4": 0.66}
    assert x1.verdict_from(ok_d, 0.95, ok_p, 0.65, []) == "PASS"
    assert x1.verdict_from(ok_d, 0.95, ok_p, 0.65, ["curve WARN"]) == "REVIEW"
    assert x1.verdict_from({**ok_d, "96": 0.949}, 0.95, ok_p, 0.65, []) == "BLOCK"   # EVERY N
    assert x1.verdict_from(ok_d, 0.95, {**ok_p, "mae_1.4": 0.649}, 0.65, []) == "BLOCK"  # EVERY MAE
    assert x1.verdict_from({**ok_d, "96": None}, 0.95, ok_p, 0.65, []) == "BLOCK"   # fails closed
    assert x1.verdict_from(ok_d, 0.95, {**ok_p, "mae_1.4": None}, 0.65, []) == "BLOCK"
    with pytest.raises(ValueError):
        x1.verdict_from({}, 0.95, ok_p, 0.65, [])


def test_exit_code_map_is_n3(x1):
    assert x1.EXIT == {"PASS": 0, "REVIEW": 3, "BLOCK": 1, "FAIL": 1}


# ------------------------------------------------------------------------------ month ends
def test_month_end_sessions_drops_an_incomplete_month(x1):
    from sharpen.data import trading_calendar as tc

    idx = tc.sessions("2026-06-01", "2026-07-31")
    assert list(x1.month_end_sessions(idx)) == [pd.Timestamp("2026-06-30"), pd.Timestamp("2026-07-31")]
    assert list(x1.month_end_sessions(idx[idx <= "2026-07-20"])) == [pd.Timestamp("2026-06-30")]
    with pytest.raises(ValueError):          # a non-session (Juneteenth) is refused, not guessed
        x1.month_end_sessions(pd.DatetimeIndex(["2026-06-19"]))


# --------------------------------------------------------------------- pins and the guard
def test_pin_mismatch_is_refused(x1, tmp_path, monkeypatch):
    (tmp_path / "d.parquet").write_bytes(b"abc")
    monkeypatch.setattr(x1, "ROOT", tmp_path)
    with pytest.raises(x1.IntegrityError, match="pin mismatch"):
        x1.check_pins({"pins": {"d.parquet": "0" * 64}, "manifest_content_sha256_16": "x"})
    with pytest.raises(x1.IntegrityError, match="missing"):
        x1.check_pins({"pins": {"absent.parquet": "0" * 64}, "manifest_content_sha256_16": "x"})


def test_offline_guard_blocks_every_fetch_path_and_restores(x1, monkeypatch):
    import yfinance

    from sharpen.data import cross_asset_loader as cal
    from sharpen.data import treasury_curve_loader as tcl

    monkeypatch.setattr(x1, "check_pins", lambda prereg: None)
    before = (yfinance.download, cal.fetch_ohlcv_wide, tcl._fetch_yahoo_curve)
    with x1.offline_guard({}):
        for fn in (yfinance.download, cal.fetch_ohlcv_wide, tcl._fetch_yahoo_curve):
            with pytest.raises(x1.IntegrityError, match="offline guard"):
                fn(["SPY"])
    assert (yfinance.download, cal.fetch_ohlcv_wide, tcl._fetch_yahoo_curve) == before


# ------------------------------------------------------------------------------ the prereg
@pytest.fixture(scope="module")
def prereg():
    return yaml.safe_load((ROOT / "configs" / "tailwind_v1_x1.prereg.yaml").read_text(encoding="utf-8"))


def test_ledger_arithmetic(prereg):
    fam = {f["id"]: int(f["count"]) for f in prereg["ledger"]["families"]}
    tot = prereg["ledger"]["totals"]
    assert tot["strategy_only"] == sum(fam[k] for k in ("F1", "F2", "F3", "F4", "F5", "F6"))
    assert tot["full"] == sum(fam.values())
    assert tot["full"] in prereg["dsr"]["n_bracket"]
    assert max(prereg["dsr"]["n_bracket"]) >= tot["full"]


def test_every_threshold_resolves_to_a_gates_file(x1, prereg):
    refs = [prereg["dsr"]["min_dsr_source"], prereg["p_pass"]["min_p_pass_source"],
            prereg["p_pass"]["horizon_source"], prereg["render"]["needless_source"],
            prereg["p_pass"]["daily_breach_budget_source"], *prereg["render"]["gross_sources"]]
    vals = [x1._gate_value(r) for r in refs]
    assert all(isinstance(v, (int, float)) for v in vals)
    kills = prereg["p_pass"]["kills_source"]
    for key in kills["keys"]:
        assert isinstance(x1._gate_value({"file": kills["file"], "key": key}), (int, float))
    firm = x1._gate_value(prereg["p_pass"]["firm_source"])
    assert {"max_total_loss_pct", "daily_loss_limit_pct", "profit_target_step1_pct",
            "min_trading_days", "dd_mode"} <= set(firm)


# ------------------------------------------------------------------------ certify refusals
def _certify(x1, tmp_path, monkeypatch, prereg_edits: dict, commit, existing: bool = False):
    src = yaml.safe_load((ROOT / "configs" / "tailwind_v1_x1.prereg.yaml").read_text(encoding="utf-8"))
    src.update(prereg_edits)
    p = tmp_path / "prereg.yaml"
    p.write_text(yaml.safe_dump(src), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    if existing:
        (out / "x1_recertification_2026-01-01.json").write_text("{}")
    monkeypatch.setattr(x1, "PREREG", p)
    monkeypatch.setattr(x1, "OUT_DIR", out)
    monkeypatch.setattr(x1, "check_pins", lambda prereg: {})
    monkeypatch.setattr(x1, "lineage", lambda prereg, c: {})
    monkeypatch.setattr(x1, "_prereg_commit_if_clean", lambda: commit)

    def _no_pass(prereg):
        raise AssertionError("run_pass must not be reached")
    monkeypatch.setattr(x1, "run_pass", _no_pass)
    return x1.main(["--mode", "certify"]), sorted(f.name for f in out.iterdir())


def test_certify_refuses_a_draft(x1, tmp_path, monkeypatch):
    code, files = _certify(x1, tmp_path, monkeypatch, {"status": "draft"}, commit="abc")
    assert code == 1 and any(f.startswith("x1_attempt_FAIL_") for f in files)
    assert not any(f.startswith("x1_recertification_") for f in files)


def test_certify_refuses_an_uncommitted_prereg(x1, tmp_path, monkeypatch):
    signed = {"status": "registered", "operator_signoff": {"by": "t", "on": "2026-09-29"}}
    code, files = _certify(x1, tmp_path, monkeypatch, signed, commit=None)
    assert code == 1 and not any(f.startswith("x1_recertification_") for f in files)


def test_certify_refuses_a_second_pass(x1, tmp_path, monkeypatch):
    signed = {"status": "registered", "operator_signoff": {"by": "t", "on": "2026-09-29"}}
    code, files = _certify(x1, tmp_path, monkeypatch, signed, commit="abc", existing=True)
    assert code == 1 and files.count("x1_recertification_2026-01-01.json") == 1


def test_prereg_registers_the_audit_spec(prereg):
    assert prereg["window"] == {"start": "2007-04-30", "end": "2026-05-29"}
    assert prereg["basis"]["decision_lead_bars"] == 1
    assert prereg["p_pass"]["intraday_mae_mult"] == [1.0, 1.4]
    assert prereg["p_pass"]["binding_statistic"] == "wilson_lower"
    assert prereg["p_pass"]["gate_policy"] == "declared_kills"
    assert set(prereg["pins"]) == {
        "results/tailwind_v1/ohlcv_daily.parquet", "results/xsec_momentum/prices_daily.parquet",
        "results/tailwind_v1/prices_wide_daily.parquet", "results/carry_falsification/yahoo_curve.parquet"}
