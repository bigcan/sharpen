"""Forward runner (TAILWIND Tier-2 roadmap X2): the batch book, run forward from truncated data.

Pinned:
- **Forward reads equal the batch** at every cutoff, under both leads. Each read recomputes the
  signals from prices truncated at its as-of session and appends the next two calendar
  sessions. It is checked against the one-shot batch over the whole panel.
- **The month-end conviction switch fills on the right session**: the month-end itself under the
  lead, the next session without it.
- **The old documented live target misses month-end fills.** That is ``w[-1]`` of the
  un-extended window (T2-01), so the extension is load-bearing.
- **N5(b):** the prefix-consistency test, parametrized over the lead-0 and both TAILWIND configs
  through the same forward read.
- **The daily cycle end to end:** hash-chained log, settle, incremental parity (zero by
  construction), kills, stale data, in-progress bars, vendor re-adjustment (re-based, never
  booked as P&L), missed sessions and exit codes.
"""
from __future__ import annotations

import copy
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sharpen.data import financing as fin
from sharpen.data import trading_calendar as tc
from sharpen.envs.allocator_factory import linear_core_weights
from sharpen.paper import forward_runner as fr
from sharpen.paper.soak_metrics import UNKNOWN

from .conftest import allocator_arrays

ROOT = Path(__file__).resolve().parents[2]
_ASSETS = ["SPY", "QQQ", "IWM", "TLT", "IEF", "LQD"]
_CLASS = {"SPY": "equity", "QQQ": "equity", "IWM": "equity",
          "TLT": "rates", "IEF": "rates", "LQD": "rates"}

pytestmark = pytest.mark.filterwarnings("ignore:.*fill_method.*:FutureWarning")


def _panel(n: int = 330, seed: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = tc.sessions("2021-01-04", "2023-12-29")[:n]
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(100.0 * np.cumprod(1.0 + rng.normal(4e-4, 0.011, (n, 6)), axis=0),
                         index=idx, columns=_ASSETS)
    volume = pd.DataFrame(rng.uniform(1e6, 5e6, (n, 6)), index=idx, columns=_ASSETS)
    return close, volume


def _curve(close: pd.DataFrame) -> tuple[dict, dict]:
    days = tc.sessions(close.index[0] - pd.Timedelta(days=10), close.index[-1] + pd.Timedelta(days=10))
    y = pd.Series(np.linspace(1.5, 4.5, len(days)), index=days)
    return {"3m": y}, {"status": "PASS", "date_max": str(days[-1].date())}


def _cfg(lead: int = 1, *, financed: bool = False) -> dict:
    cfg = {
        "universe": {"assets": _ASSETS, "asset_class": _CLASS},
        "sleeves": {"momentum": {"assets": _ASSETS, "signal": "tsmom"},
                    "defensive": {"assets": _ASSETS, "beta_window": 63, "min_periods": 40}},
        "features": {"lookbacks": [21, 63], "skip": 5, "vol_window": 21},
        "env": {"type": "multi_asset_allocator", "initial_capital": 100000.0, "taker_fee": 0.0002,
                "slippage_base_bps": 1.0, "slippage_impact_bps": 5.0, "target_vol_asset": 0.10,
                "lev_cap": 2.0, "max_gross_exposure": 3.0, "min_trade_pct": 0.005},
        "risk_parity": {"trailing_window": 63, "min_periods": 21, "monthly_meta": True},
        "execution": {"shadow": "linear_core_tailwind", "decision_lead_bars": lead},
    }
    if financed:
        cfg["financing"] = {"model": "tbill", "tenor": "3m", "short_borrow_bps": 25}
    return cfg


@pytest.fixture(scope="module")
def gates() -> dict:
    return yaml.safe_load((ROOT / "configs" / "tailwind_v1.gates.yaml").read_text(encoding="utf-8"))


def _decide(cfg, close, volume, as_of, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return fr.compute_forward_decision(cfg, close, volume, as_of=as_of, **kw)


# --------------------------------------------------------------------------- #
# The forward read against the batch
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def panel():
    return _panel()


def _cutoffs(close: pd.DataFrame) -> list[pd.Timestamp]:
    """Every month-end eve in the live half of the panel plus mid-month cutoffs."""
    s = close.index
    eves = [s[i] for i in range(150, len(s) - 2) if tc.is_month_end(s[i + 1])]
    mids = [s[i] for i in (160, 185, 211, 240, 266, 297)]
    return sorted(set(eves + mids))


@pytest.mark.parametrize("lead", [0, 1])
def test_forward_reads_equal_the_batch_at_every_cutoff(panel, lead):
    close, volume = panel
    cfg = _cfg(lead)
    batch = _decide(cfg, close, volume, close.index[-1])
    dates = fr._session_dates(batch.union["timestamps"])
    for as_of in _cutoffs(close):
        d = _decide(cfg, close, volume, as_of)
        i = int(dates.get_loc(d.fill_session))
        np.testing.assert_array_equal(d.weights, batch.batch_weights[i - 1],
                                      err_msg=f"lead {lead} as-of {as_of.date()}")
        assert d.fill_session == tc.next_sessions(as_of, 1)[0]
        assert d.month_end_fill == tc.is_month_end(d.fill_session)


@pytest.mark.parametrize("lead, offset", [(1, 0), (0, 1)])
def test_the_month_end_switch_fills_on_the_right_session(panel, lead, offset):
    """Under the lead the new month's conviction fills AT the month-end close; without it,
    one session later. Read from the forward decisions themselves, per sleeve."""
    close, volume = panel
    cfg = _cfg(lead)
    s = close.index
    me = [i for i in range(170, len(s) - 3) if tc.is_month_end(s[i])][:3]
    assert me
    for i in me:
        convs = {f: _decide(cfg, close, volume, s[f - 1]).sleeve_conviction for f in range(i - 1, i + 3)}
        for sleeve in cfg["sleeves"]:
            switch = next((f for f in range(i, i + 3)
                           if convs[f][sleeve] != convs[f - 1][sleeve]), None)
            assert switch == i + offset, (lead, sleeve, s[i].date())


def test_the_old_window_reader_misses_month_end_fills(panel):
    """T2-01's teeth: ``linear_core_weights(window)[-1]`` without the appended sessions (the
    previously documented live target) disagrees with the batch at a month-end fill under the
    lead, while the forward read agrees."""
    close, volume = panel
    cfg = _cfg(1)
    batch = _decide(cfg, close, volume, close.index[-1])
    dates = fr._session_dates(batch.union["timestamps"])
    from sharpen.data.cross_asset_loader import build_two_sleeve_arrays, prepare_two_sleeve_payload
    from sharpen.paper.two_sleeve import TwoSleeveExecutor

    misses = 0
    for as_of in [a for a in _cutoffs(close) if tc.is_month_end(tc.next_sessions(a, 1)[0])]:
        c = close.loc[:as_of]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            payload = prepare_two_sleeve_payload(cfg, c, volume.loc[:as_of], manifest={},
                                                 assert_causal=False)
            bundle = build_two_sleeve_arrays(payload, c.index[0], c.index[-1])
            old = TwoSleeveExecutor(cfg).sim_oracle(bundle)[1]["combined_w"][-1]
        fill = tc.next_sessions(as_of, 1)[0]
        misses += int(np.abs(old - batch.batch_weights[int(dates.get_loc(fill)) - 1]).sum() > 0.05)
    assert misses >= 1


# --------------------------------------------------------------------------- #
# N5(b): prefix consistency over the lead-0 and TAILWIND configs, via the forward read
# --------------------------------------------------------------------------- #
def _forward_arrays(arrays: dict, m: int) -> dict:
    """The arrays a forward read at bar ``m`` sees: rows ``<= m`` as they are, plus two appended
    rows. Row ``m+1`` of conviction and vol is legitimately known at ``m`` (cutoff lag 1, as
    declared). Every other appended field is a placeholder copy of the last known row, so no
    price at or after ``m+1`` is read."""
    out = {}
    for k, v in arrays.items():
        if k == "assets" or np.ndim(v) == 0:
            out[k] = v
            continue
        v = np.asarray(v)
        if k in ("conviction_ary", "vol_ary"):
            tail = [v[m + 1], v[m + 1]]
        elif k == "timestamps":
            tail = [v[m + 1], v[m + 2]]          # the calendar, known ex ante
        else:
            tail = [v[m], v[m]]
        out[k] = np.concatenate([v[:m + 1], np.stack(tail)])
    return out


@pytest.mark.parametrize("name", ["cross_asset_momentum.yaml", "tailwind_v1.yaml",
                                  "tailwind_v1_challenge.yaml"])
def test_forward_prefix_consistency_all_configs(name):
    cfg = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
    arrays = {**allocator_arrays(T=520, n=18, seed=11), "conviction_cutoff_lag": 1, "vol_cutoff_lag": 1}
    W_full = linear_core_weights(arrays, cfg)
    ts = pd.to_datetime(arrays["timestamps"], unit="s")
    months = ts.to_period("M")
    ends = [m for m in range(300, 516) if months[m + 1] != months[m]]
    for m in sorted(set(ends + [m - 1 for m in ends] + [333, 404, 458])):
        W_ext = linear_core_weights(_forward_arrays(arrays, m), cfg)
        np.testing.assert_array_equal(W_ext[:m + 1], W_full[:m + 1], err_msg=f"{name} m={m}")


# --------------------------------------------------------------------------- #
# The hash-chained target log
# --------------------------------------------------------------------------- #
def _rec(fill: str, w: list[float], kind: str = "target") -> dict:
    return {"kind": kind, "as_of": "2024-01-02", "fill_session": fill, "assets": ["A", "B"],
            "weights": w}


def test_target_log_is_append_only_and_hash_chained(tmp_path):
    log = fr.TargetLog(tmp_path / "t.jsonl")
    r0, gap = log.append(_rec("2024-01-03", [0.5, -0.2]))
    assert gap is None and r0["seq"] == 0 and r0["prev_hash"] == "0" * 64
    r1, _ = log.append(_rec("2024-01-04", [0.4, -0.1]))
    assert r1["prev_hash"] == r0["hash"]
    same, gap = log.append(_rec("2024-01-04", [0.4, -0.1]))          # idempotent re-run
    assert same["seq"] == 1 and gap == 0.0
    kept, gap = log.append(_rec("2024-01-04", [0.9, 0.0]))            # different: first stands
    assert kept["weights"] == [0.4, -0.1] and gap > 0.4
    with pytest.raises(fr.TargetLogError, match="older"):
        log.append(_rec("2024-01-02", [0.0, 0.0]))
    flat, gap = log.append(_rec("2024-01-04", [0.0, 0.0], kind="flatten"))   # a kill supersedes
    assert gap is None and flat["seq"] == 2
    assert fr.TargetLog(tmp_path / "t.jsonl").for_session("2024-01-04")["kind"] == "flatten"
    assert log.verify()
    lines = (tmp_path / "t.jsonl").read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace("0.4", "0.41")
    (tmp_path / "t.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert not fr.TargetLog(tmp_path / "t.jsonl").verify()
    with pytest.raises(fr.TargetLogError, match="hash chain broken at line 2"):
        fr.TargetLog(tmp_path / "t.jsonl").records()


# --------------------------------------------------------------------------- #
# The daily cycle
# --------------------------------------------------------------------------- #
def _run(runner, close, volume, as_of, cfg, **kw):
    curve, cm = _curve(close) if fin.financing_spec(cfg).enabled else (None, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return runner.run_session(close, volume, as_of=as_of, curve=curve, curve_manifest=cm,
                                  assert_causal=False, **kw)


@pytest.fixture(scope="module")
def ran(tmp_path_factory, panel, gates):
    """45 consecutive daily runs of a financed lead-1 book."""
    close, volume = panel
    cfg = _cfg(1, financed=True)
    state = tmp_path_factory.mktemp("fwd")
    runner = fr.ForwardRunner(cfg, gates, state)
    reports = [_run(runner, close, volume, s, cfg) for s in close.index[240:285]]
    return {"cfg": cfg, "state": state, "runner": runner, "reports": reports,
            "close": close, "volume": volume}


def test_daily_cycle_books_every_logged_target(ran):
    fills = ran["runner"].fills()
    assert len(fills) == 44 and all(f["kind"] == "target" for f in fills)
    log = fr.TargetLog(ran["state"] / fr.TARGET_LOG)
    for f in fills:
        np.testing.assert_allclose(f["positions"], log.for_session(f["session"])["weights"],
                                   rtol=0, atol=1e-15)      # held + (target - held)
    assert ran["reports"][0]["settled"] == [] and ran["reports"][1]["settled"][0]["session"] == \
        ran["reports"][0]["logged"]["fill_session"]


def test_incremental_parity_is_zero_and_the_run_is_healthy(ran):
    last = ran["reports"][-1]
    v = last["verdict"]
    par = v["forward"]["incremental_parity"]
    assert par["weight_l1_drift_max"] <= 1e-12
    assert par["daily_return_te_bps_max"] < 1e-6
    assert par["missed_rebalances"] == 0 and abs(par["cost_drift_ratio"] - 1.0) < 1e-9
    assert v["groups"]["parity"]["status"] == "PASS"
    assert v["forward"]["target_log_chain_ok"]
    assert v["overall_status"] == UNKNOWN            # 44 sessions < the 90-day soak horizon
    assert last["exit_code"] == fr.EXIT_PASS


def test_the_scored_book_carries_per_sleeve_attribution(ran):
    """Audit T4-12b on the forward path: the runner scores with executor_sleeves, so the book
    must carry per-sleeve P&L. It comes from the batch the book is parity-checked against, so
    while parity is exact the sleeves sum to the per-class total."""
    v = ran["reports"][-1]["verdict"]
    sleeve_pnl, class_pnl = v["summary"]["sleeve_pnl"], v["summary"]["class_pnl"]
    assert sleeve_pnl is not None and set(sleeve_pnl) == set(ran["runner"].sleeve_names)
    assert any(abs(x) > 1e-6 for x in sleeve_pnl.values())
    assert sum(sleeve_pnl.values()) == pytest.approx(sum(class_pnl.values()), abs=1e-9)
    chk = v["groups"]["drift"]["checks"]["sleeve_attribution"]
    assert chk["status"] == "PASS" and chk["basis"] == "sleeve"


def test_the_scored_gross_kill_reads_actual_notional(ran):
    """Audit T4-12a on the forward path: fills record the book's actual notional gross, and
    the soak's gross kill is read on it."""
    v = ran["reports"][-1]["verdict"]
    assert v["groups"]["risk"]["checks"]["max_gross_exposure"]["basis"] == "notional"
    assert all("notional_gross" in f for f in ran["runner"].fills())


def test_rerunning_the_same_session_is_idempotent(ran, tmp_path):
    state = tmp_path / "copy"
    shutil.copytree(ran["state"], state)
    runner = fr.ForwardRunner(ran["cfg"], ran["runner"].gates_cfg, state)
    as_of = ran["close"].index[284]
    rep = _run(runner, ran["close"], ran["volume"], as_of, ran["cfg"])
    assert rep["settled"] == [] and rep["repeat_l1"] == 0.0
    assert len(runner.fills()) == 44


def test_stale_data_fails_closed_without_logging(ran, tmp_path):
    state = tmp_path / "copy"
    shutil.copytree(ran["state"], state)
    runner = fr.ForwardRunner(ran["cfg"], ran["runner"].gates_cfg, state)
    n_before = len(fr.TargetLog(state / fr.TARGET_LOG).records())
    as_of = ran["close"].index[285]
    with pytest.raises(fr.StaleDataError, match="data ends"):
        _run(runner, ran["close"].loc[:ran["close"].index[284]],
             ran["volume"].loc[:ran["close"].index[284]], as_of, ran["cfg"])
    assert len(fr.TargetLog(state / fr.TARGET_LOG).records()) == n_before


def test_an_in_progress_bar_is_dropped(panel):
    close, volume = panel
    cfg = _cfg(1)
    as_of = close.index[250]
    clean = _decide(cfg, close.loc[:as_of], volume.loc[:as_of], as_of)
    noisy = close.loc[:close.index[251]].copy()
    noisy.iloc[-1] *= 1.3                          # a partial bar printed before the close
    with_bar = _decide(cfg, noisy, volume.loc[:close.index[251]], as_of)
    np.testing.assert_array_equal(clean.weights, with_bar.weights)
    assert clean.data_sha16 == with_bar.data_sha16


def test_a_calendar_mismatch_fails_closed(panel):
    close, volume = panel
    holey = close.drop(close.index[200])
    with pytest.raises(fr.ForwardRunError, match="disagree with the NYSE calendar"):
        _decide(_cfg(1), holey, volume.drop(volume.index[200]), close.index[250])


def test_a_kill_file_flattens_the_next_session_and_stops_the_book(ran, tmp_path):
    state = tmp_path / "copy"
    shutil.copytree(ran["state"], state)
    runner = fr.ForwardRunner(ran["cfg"], ran["runner"].gates_cfg, state)
    (state / "KILL").write_text("operator", encoding="utf-8")
    rep = _run(runner, ran["close"], ran["volume"], ran["close"].index[285], ran["cfg"])
    assert rep["kill"]["reason"].startswith("kill_file") and rep["exit_code"] == fr.EXIT_FAIL
    assert rep["logged"]["kind"] == "flatten"
    rep2 = _run(runner, ran["close"], ran["volume"], ran["close"].index[286], ran["cfg"])
    assert rep2["settled"][-1]["kind"] == "flatten" and not np.any(rep2["settled"][-1]["positions"])
    assert rep2["exit_code"] == fr.EXIT_FAIL


def test_a_vendor_readjustment_is_rebased_not_booked(ran, tmp_path):
    """A refetch re-scales one asset's whole adjusted history (a dividend re-adjustment). The
    next session's equity must match the un-rescaled run: the entry prices are re-based, so the
    rescale is never booked as a loss."""
    as_of = ran["close"].index[285]
    out = {}
    for label, factor in (("same", 1.0), ("readjusted", 0.98)):
        state = tmp_path / label
        shutil.copytree(ran["state"], state)
        close = ran["close"].copy()
        close["TLT"] *= factor
        runner = fr.ForwardRunner(ran["cfg"], ran["runner"].gates_cfg, state)
        out[label] = _run(runner, close, ran["volume"], as_of, ran["cfg"])
    assert out["readjusted"]["equity"] == pytest.approx(out["same"]["equity"], rel=1e-9)
    st = json.loads((tmp_path / "readjusted" / fr.RUNNER_STATE).read_text(encoding="utf-8"))
    assert st["max_rebase_abs"] == pytest.approx(0.02, rel=1e-9)


def test_a_missed_run_holds_and_is_recorded(ran, tmp_path):
    state = tmp_path / "copy"
    shutil.copytree(ran["state"], state)
    runner = fr.ForwardRunner(ran["cfg"], ran["runner"].gates_cfg, state)
    idx = ran["close"].index
    rep = _run(runner, ran["close"], ran["volume"], idx[286], ran["cfg"])   # the run at idx[285] never happened
    kinds = [s["kind"] for s in rep["settled"]]
    assert kinds == ["target", "missed"]
    assert rep["verdict"]["forward"]["missed_sessions"] == [idx[286].date().isoformat()]


def test_exit_code_map():
    ok_groups = {g: {"status": "PASS"} for g in ("parity", "risk", "drift", "performance")}
    fwd = {"target_log_chain_ok": True, "action_drift": None}
    base = {"kill": None, "verdict": {"overall_status": "PASS", "groups": ok_groups, "forward": fwd}}
    assert fr.exit_code(base) == fr.EXIT_PASS
    assert fr.exit_code({**base, "verdict": {**base["verdict"], "overall_status": "REVIEW"}}) == fr.EXIT_REVIEW
    assert fr.exit_code({**base, "verdict": {**base["verdict"], "overall_status": "FAIL"}}) == fr.EXIT_FAIL
    horizon_only = {**base["verdict"], "overall_status": UNKNOWN,
                    "groups": {**ok_groups, "horizon": {"status": UNKNOWN}}}
    assert fr.exit_code({**base, "verdict": horizon_only}) == fr.EXIT_PASS
    parity_unknown = {**horizon_only, "groups": {**ok_groups, "parity": {"status": UNKNOWN}}}
    assert fr.exit_code({**base, "verdict": parity_unknown}) == fr.EXIT_FAIL
    assert fr.exit_code({**base, "kill": {"reason": "x"}}) == fr.EXIT_FAIL
    broken = {**base["verdict"], "forward": {**fwd, "target_log_chain_ok": False}}
    assert fr.exit_code({**base, "verdict": broken}) == fr.EXIT_FAIL
    drift = {**base["verdict"], "forward": {**fwd, "action_drift": {"momentum": {"status": "WARN"}}}}
    assert fr.exit_code({**base, "verdict": drift}) == fr.EXIT_REVIEW


def test_drift_trackers_follow_the_config(ran, tmp_path):
    """drift.enabled with a missing baseline is REVIEW (not wired), never silent; with no drift
    block there is nothing to report."""
    assert ran["runner"].drift_reports() is None
    cfg = copy.deepcopy(ran["cfg"])
    cfg["drift"] = {"enabled": True, "baseline_path": str(tmp_path / "absent.json")}
    runner = fr.ForwardRunner(cfg, ran["runner"].gates_cfg, tmp_path / "s")
    assert runner.drift_reports()["_status"] == UNKNOWN
