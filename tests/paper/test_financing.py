"""Financing leg (TAILWIND Tier-2 N2, 2026-09-29): carry = -rf plus a borrow fee on shorts.

The load-bearing property is the N2 acceptance test. A NET-LONG book with a short leg, whose
assets earn exactly the cash rate, has ZERO excess return, bar by bar. This runs through the
real union builder, the env and the PaperState replay. It fails if:
- the builders re-zero ``carry_ary`` (the book then earns net*rf);
- the env or PaperState stops accruing carry;
- either one accrues carry on the end-of-bar notional instead of the notional carried into the
  bar (a residual of about -net*g^2, 2e-8 per bar here);
- the short leg's carry sign flips.

The surrounding tests cover:
- the rate construction (act/360 at the prior-close yield, with a LEAK-2 negative test);
- the fail-closed spec;
- borrow charged on shorts only;
- env<->PaperState lockstep parity with non-zero carry and borrow (never exercised before,
  because every carry array was zero);
- the loader wiring;
- the research-side ``held_weights`` and ``excess_returns`` helpers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sharpen.data import cross_asset_loader as cal
from sharpen.data import financing as fin
from sharpen.envs.allocator_factory import make_allocator_env, monthly_rebal_conviction
from sharpen.envs.multi_asset_allocator_env import MultiAssetAllocatorEnv
from sharpen.paper.fill_engine import SimFillEngine
from sharpen.paper.paper_state import PaperState
from sharpen.paper.parity_harness import ParityHarness

from .conftest import allocator_arrays

ROOT = Path(__file__).resolve().parents[2]
_LEVERS_OFF = {"no_trade_band": 0.0, "rebalance_interval": 1, "cost_penalty_scale": 0.0}
_W_NET_LONG = np.array([1.0, 0.8, -0.5])      # net +1.3, gross 2.3, one short leg
_CAP = 100_000.0
# Zero-excess tolerance. The env's `entry_price + 1e-10` guard leaves ~1e-12 on the fill bar;
# every later bar is ~1e-16. The end-of-bar-notional mutation leaves -net*g^2 ~ 2.5e-8 per bar,
# so this separates them by 250x.
_ZERO_TOL = 1e-10


# --------------------------------------------------------------------------- #
# Synthetic market: every asset earns EXACTLY the per-bar cash rate (zero excess)
# --------------------------------------------------------------------------- #
def _curve(dates: pd.DatetimeIndex, level_pct: float = 5.0, seed: int = 0) -> dict:
    """A business-day 3m yield (PERCENT) around ``level_pct``, starting a week before ``dates``
    so the first interval has a prior print."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(dates[0] - pd.Timedelta(days=7), dates[-1])
    return {"3m": pd.Series(level_pct + np.cumsum(rng.normal(0.0, 0.03, len(days))), index=days)}


def _market(n_bars: int = 260, n_assets: int = 3, borrow_bps: float = 0.0):
    dates = pd.bdate_range("2023-01-02", periods=n_bars)
    spec = fin.FinancingSpec(model="tbill", short_borrow_bps=borrow_bps)
    rates = fin.financing_rates(_curve(dates), dates, spec)
    growth = np.cumprod(1.0 + rates["cash_rate"].to_numpy())     # bar return == cash rate
    price = 100.0 * np.repeat(growth[:, None], n_assets, axis=1)
    return dates, rates, price


def _env_arrays(price: np.ndarray, dates: pd.DatetimeIndex, financing: dict | None) -> dict:
    T, n = price.shape
    return {
        "price_ary": price,
        "tech_ary": np.zeros((T, n), np.float32),
        "vol_ary": np.full((T, n), 0.10),        # target_vol/vol = 1 => weight == conviction
        **fin.financing_arrays(financing, dates, n),
        "volume_ary": np.full((T, n), 1e15),
        "timestamps": (dates.asi8 // 10**9).astype(np.int64),
    }


def _drive_env(arrays: dict, action: np.ndarray) -> tuple[np.ndarray, float]:
    env = MultiAssetAllocatorEnv(
        **arrays, target_vol_asset=0.10, lev_cap=2.0, max_gross_exposure=10.0,
        taker_fee_pct=0.0, slippage_base_bps=0.0, slippage_impact_bps=0.0,
        min_trade_pct=0.0, turnover_penalty=0.0, reward_type="simple",
        circuit_breaker_threshold=0.0, initial_capital=_CAP)
    env.reset()
    rets, done, info = [], False, {}
    while not done:
        _, _, term, trunc, info = env.step(action)
        rets.append(info["step_return"])
        done = term or trunc
    return np.asarray(rets), float(info["cumulative_carry"])


# --------------------------------------------------------------------------- #
# N2 acceptance: zero asset excess return => zero book excess return
# --------------------------------------------------------------------------- #
def test_zero_excess_net_long_book_env():
    dates, rates, price = _market()
    rets, carry = _drive_env(_env_arrays(price, dates, rates), _W_NET_LONG)
    assert np.max(np.abs(rets)) < _ZERO_TOL, "financed excess return of a zero-excess book must be 0"
    assert carry < 0.0, "the leg must be live: a net-long book pays financing"

    # The same book UNFINANCED earns the cash rate on its net exposure, in closed form:
    # PV_k = cap * (1 + net * (p_k / p_1 - 1)), with positions opened at close 1.
    rets0, carry0 = _drive_env(_env_arrays(price, dates, None), _W_NET_LONG)
    assert carry0 == 0.0
    net, p = float(_W_NET_LONG.sum()), price[:, 0]
    pv = _CAP * (1.0 + net * (p / p[1] - 1.0))
    expect = np.zeros(len(p) - 1)
    expect[1:] = pv[2:] / pv[1:-1] - 1.0
    np.testing.assert_allclose(rets0, expect, rtol=1e-9, atol=_ZERO_TOL / 10)
    assert rets0[1:].min() > 0.0


def test_zero_excess_through_union_builder_and_replay():
    """The same property through the REAL union builder and the PaperState replay. This is the
    path that books the executor's scored P&L. Re-zeroing the builder's carry fails here."""
    dates, rates, price = _market()
    assets = ["A", "B", "C"]
    close = pd.DataFrame(price, index=dates, columns=assets)
    volume = pd.DataFrame(1e9, index=dates, columns=assets)
    union = cal.build_union_arrays(close, volume, assets, dates[0], dates[-1], financing=rates)
    W = np.tile(_W_NET_LONG, (len(dates) - 1, 1))
    harness = ParityHarness({"env": {"initial_capital": _CAP, "taker_fee": 0.0,
                                     "slippage_base_bps": 0.0, "slippage_impact_bps": 0.0}})
    live = harness._replay(union, W, fill_engine=None)
    assert np.max(np.abs(live.step_returns)) < _ZERO_TOL
    unfinanced = harness._replay({**union, "carry_ary": np.zeros_like(union["carry_ary"])}, W,
                                 fill_engine=None)
    assert unfinanced.step_returns[1:].min() > 0.0         # tripwire: the zero above is carry's


def test_borrow_is_charged_on_short_notional_only():
    dates, rates, price = _market(borrow_bps=50.0)
    # long-only: pays no borrow, so the zero-excess book stays at exactly 0
    long_only, _ = _drive_env(_env_arrays(price, dates, rates), np.array([1.0, 0.8, 0.0]))
    assert np.max(np.abs(long_only)) < _ZERO_TOL

    rets, _ = _drive_env(_env_arrays(price, dates, rates), _W_NET_LONG)
    b, p, short_w = rates["borrow_rate"].to_numpy(), price[:, 0], 0.5
    expect, pv = np.zeros(len(p) - 1), _CAP
    for k in range(1, len(p) - 1):              # the short is held over (k, k+1] from step 1
        pnl = -short_w * _CAP * (p[k] / p[1]) * b[k + 1]
        expect[k], pv = pnl / pv, pv + pnl
    np.testing.assert_allclose(rets, expect, rtol=1e-9, atol=_ZERO_TOL / 10)
    assert rets[1:].max() < 0.0


def test_replay_charges_borrow_like_the_env():
    """The union replay (the executor's scored book) books the borrow leg exactly as the env."""
    dates, rates, price = _market(borrow_bps=50.0)
    assets = ["A", "B", "C"]
    close = pd.DataFrame(price, index=dates, columns=assets)
    union = cal.build_union_arrays(close, pd.DataFrame(1e9, index=dates, columns=assets),
                                   assets, dates[0], dates[-1], financing=rates)
    assert "borrow_ary" in union
    live = ParityHarness({"env": {"initial_capital": _CAP, "taker_fee": 0.0,
                                  "slippage_base_bps": 0.0, "slippage_impact_bps": 0.0}})._replay(
        union, np.tile(_W_NET_LONG, (len(dates) - 1, 1)), fill_engine=None)
    env_rets, _ = _drive_env(_env_arrays(price, dates, rates), _W_NET_LONG)
    np.testing.assert_allclose(live.step_returns, env_rets, rtol=1e-12, atol=_ZERO_TOL / 10)
    assert live.step_returns[1:].max() < 0.0


# --------------------------------------------------------------------------- #
# env <-> PaperState lockstep with NON-ZERO carry + borrow (both legs exercised)
# --------------------------------------------------------------------------- #
def test_book_matches_env_with_financing(cfg):
    arrays = allocator_arrays(T=90, n=3, seed=4, volume=1e7)
    rng = np.random.default_rng(1)
    shape = arrays["carry_ary"].shape
    arrays["carry_ary"] = -np.abs(rng.normal(2e-4, 1e-4, shape))    # asset-varying rates
    arrays["borrow_ary"] = np.abs(rng.normal(1e-4, 5e-5, shape))
    env = make_allocator_env(arrays, cfg, overrides=_LEVERS_OFF, eval_mode=True)
    conv = monthly_rebal_conviction(arrays["timestamps"], arrays["conviction_ary"])
    env.reset()
    book = PaperState(n_assets=3, initial_capital=env.initial_capital)
    eng = SimFillEngine(taker_fee_pct=env.taker_fee_pct, slippage_base_bps=env.slippage_base_bps,
                        slippage_impact_bps=env.slippage_impact_bps)
    price, volume = arrays["price_ary"], arrays["volume_ary"]
    saw_short = False
    done = False
    while not done:
        k = env.step_idx
        old_pos = env.positions.copy()
        saw_short |= bool((old_pos < 0).any())
        _, _, term, trunc, info = env.step(conv[k])
        delta = info["position"] - old_pos
        pv_before = book.pv_before(price[k])
        fill = eng.fill(delta_weights=delta, ref_prices=price[k + 1],
                        pv_before=pv_before, dollar_volume=volume[k])
        binfo = book.step_bar(delta_weights=delta, fill=fill, prev_price=price[k],
                              price_now=price[k + 1], carry_rates=arrays["carry_ary"][k + 1],
                              pv_before=pv_before, borrow_rates=arrays["borrow_ary"][k + 1])
        assert abs(book.cumulative_carry - env.cumulative_carry) < 1e-9, f"carry drift at {k}"
        assert abs(book.margin_balance - env.margin_balance) < 1e-6, f"margin drift at {k}"
        assert abs(binfo["step_return"] - info["step_return"]) < 1e-12, f"return drift at {k}"
        done = term or trunc
    assert saw_short and env.cumulative_carry != 0.0, "scenario must exercise carry on both legs"


# --------------------------------------------------------------------------- #
# Rate construction (sharpen.data.financing)
# --------------------------------------------------------------------------- #
def test_cash_rate_is_act360_on_the_prior_close_yield():
    dates = pd.DatetimeIndex(["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"])  # Th F M Tu
    y = pd.Series([5.0, 4.0, 3.0, 2.0], index=dates)
    r = fin.per_bar_cash_rate(y, dates, day_count=360)
    np.testing.assert_allclose(r.to_numpy(), [0.0, 0.05 / 360, 0.04 * 3 / 360, 0.03 / 360],
                               rtol=1e-15, atol=0.0)
    r365 = fin.per_bar_cash_rate(y, dates, day_count=365)
    np.testing.assert_allclose(r365.to_numpy()[1:], r.to_numpy()[1:] * 360 / 365, rtol=1e-15)


def test_cash_rate_never_reads_the_close_it_accrues_into():
    """LEAK-2 negative test: the interval (t-1, t] accrues at the yield known at close t-1.
    Bumping the yield printed AT bar t must leave rate[t] unchanged and move rate[t+1]."""
    dates = pd.bdate_range("2024-01-01", periods=30)
    y = pd.Series(np.linspace(4.0, 5.0, 30), index=dates)
    base = fin.per_bar_cash_rate(y, dates, day_count=360)
    for t in (5, 17, 29):
        bumped = y.copy()
        bumped.iloc[t] += 3.0
        r = fin.per_bar_cash_rate(bumped, dates, day_count=360)
        assert r.iloc[t] == base.iloc[t]
        np.testing.assert_array_equal(r.iloc[:t + 1].to_numpy(), base.iloc[:t + 1].to_numpy())
        if t + 1 < len(dates):
            assert r.iloc[t + 1] != base.iloc[t + 1]


def test_cash_rate_reads_the_last_print_on_or_before_the_prior_close():
    dates = pd.DatetimeIndex(["2024-07-02", "2024-07-03", "2024-07-05"])
    y = pd.Series([5.0, 4.0], index=pd.DatetimeIndex(["2024-07-02", "2024-07-05"]))  # no 07-03 print
    r = fin.per_bar_cash_rate(y, dates, day_count=360)
    np.testing.assert_allclose(r.to_numpy(), [0.0, 0.05 / 360, 0.05 * 2 / 360], rtol=1e-15)


def test_cash_rate_fails_closed_without_a_prior_print():
    dates = pd.bdate_range("2024-01-01", periods=5)
    with pytest.raises(ValueError, match="no cash yield"):
        fin.per_bar_cash_rate(pd.Series([5.0], index=[dates[2]]), dates, day_count=360)


def test_borrow_rate_accrues_on_calendar_days():
    dates = pd.DatetimeIndex(["2024-01-05", "2024-01-08", "2024-01-09"])      # F M Tu
    b = fin.per_bar_borrow_rate(dates, short_borrow_bps=36.0, day_count=360)
    np.testing.assert_allclose(b.to_numpy(), [0.0, 0.0036 * 3 / 360, 0.0036 / 360], rtol=1e-15)


def test_financing_arrays_unfinanced_is_zero_carry_and_no_borrow():
    dates = pd.bdate_range("2024-01-01", periods=4)
    out = fin.financing_arrays(None, dates, 3)
    assert set(out) == {"carry_ary"} and not out["carry_ary"].any()
    assert out["carry_ary"].shape == (4, 3)


def test_financing_arrays_refuse_bars_outside_the_calendar():
    dates, rates, _ = _market(n_bars=10)
    with pytest.raises(ValueError, match="missing from the financing calendar"):
        fin.financing_arrays(rates, dates.append(pd.DatetimeIndex(["2030-01-02"])), 2)


# --------------------------------------------------------------------------- #
# Spec: opt-in, fail-closed
# --------------------------------------------------------------------------- #
def test_spec_absent_or_none_is_unfinanced():
    assert not fin.financing_spec({}).enabled
    assert not fin.financing_spec({"financing": {"model": "none"}}).enabled


@pytest.mark.parametrize("block, match", [
    ({"model": "tbil"}, "financing.model"),
    ({"model": "tbill", "day_count": 252}, "day_count"),
    ({"model": "tbill", "short_borrow_bps": -1}, "short_borrow_bps"),
    ({"model": "tbill", "short_borrow_bps": float("nan")}, "short_borrow_bps"),
    ({"model": "none", "short_borrow_bps": 25}, "needs financing.model"),
    ({"model": "tbill", "borrow_bps": 25}, "unknown key"),
])
def test_spec_fails_closed(block, match):
    with pytest.raises(ValueError, match=match):
        fin.financing_spec({"financing": block})


def test_financing_rates_refuses_an_unknown_tenor():
    dates = pd.bdate_range("2024-01-01", periods=5)
    with pytest.raises(ValueError, match="tenor"):
        fin.financing_rates(_curve(dates), dates, fin.FinancingSpec(model="tbill", tenor="1m"))


@pytest.mark.parametrize("name", ["tailwind_v1.yaml", "tailwind_v1_challenge.yaml"])
def test_tailwind_configs_declare_the_financing_leg(name):
    cfg = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
    spec = fin.financing_spec(cfg)
    assert (spec.model, spec.tenor, spec.day_count, spec.short_borrow_bps) == ("tbill", "3m", 360, 25.0)


# --------------------------------------------------------------------------- #
# Loader wiring (network-free: fetch + curve stubbed)
# --------------------------------------------------------------------------- #
_ASSETS = ["SPY", "QQQ", "IWM", "TLT", "IEF", "GLD"]
_CLASS = {"SPY": "equity", "QQQ": "equity", "IWM": "equity",
          "TLT": "rates", "IEF": "rates", "GLD": "commodity"}


def _stub_data(monkeypatch, T: int = 400):
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2016-01-01", periods=T)
    close = pd.DataFrame(100.0 * np.cumprod(1.0 + rng.normal(3e-4, 0.012, (T, 6)), axis=0),
                         index=idx, columns=_ASSETS)
    volume = pd.DataFrame(rng.uniform(1e6, 5e6, (T, 6)), index=idx, columns=_ASSETS)
    curve_calls: list = []

    def fake_fetch(assets, **_kw):
        return {"close": close[list(assets)], "volume": volume[list(assets)]}, {"status": "PASS"}

    def fake_curve(**kw):
        curve_calls.append(kw)
        return _curve(idx), {"status": "PASS", "date_max": str(idx[-1].date())}

    monkeypatch.setattr(cal, "fetch_and_clean", fake_fetch)
    monkeypatch.setattr(cal.tcl, "load_treasury_curve_with_manifest", fake_curve)
    return close, curve_calls


def _tailwind_like(financing: dict | None) -> dict:
    cfg = {"universe": {"assets": _ASSETS, "asset_class": _CLASS},
           "sleeves": {"momentum": {"assets": _ASSETS, "signal": "tsmom"},
                       "defensive": {"assets": _ASSETS, "beta_window": 63, "min_periods": 40}},
           "features": {"lookbacks": [21, 63], "skip": 5, "vol_window": 21}}
    if financing is not None:
        cfg["financing"] = financing
    return cfg


def test_two_sleeve_loader_finances_every_sleeve_and_the_union(monkeypatch):
    close, curve_calls = _stub_data(monkeypatch)
    data = cal.load_two_sleeve_data(
        _tailwind_like({"model": "tbill", "tenor": "3m", "short_borrow_bps": 25}))
    assert curve_calls, "the curve must load for financing even with no rates_carry sleeve"
    wdates = close.index[150:]
    bundle = cal.build_two_sleeve_arrays(data, wdates[0], wdates[-1])
    cash = data["financing"]["cash_rate"].reindex(wdates).to_numpy()
    borrow = data["financing"]["borrow_rate"].reindex(wdates).to_numpy()
    for key in ("momentum", "defensive", "union"):
        n = bundle[key]["carry_ary"].shape[1]
        np.testing.assert_array_equal(bundle[key]["carry_ary"], np.repeat(-cash[:, None], n, axis=1))
        np.testing.assert_array_equal(bundle[key]["borrow_ary"], np.repeat(borrow[:, None], n, axis=1))
    # the window's first interval reads the bar BEFORE the window (full-calendar rates)
    assert bundle["union"]["carry_ary"][0, 0] < 0.0


def test_two_sleeve_loader_without_financing_is_unchanged(monkeypatch):
    close, curve_calls = _stub_data(monkeypatch)
    data = cal.load_two_sleeve_data(_tailwind_like(None))
    assert data["financing"] is None and not curve_calls and data["curve_manifest"] is None
    bundle = cal.build_two_sleeve_arrays(data, close.index[150], close.index[-1])
    for key in ("momentum", "defensive", "union"):
        assert not bundle[key]["carry_ary"].any() and "borrow_ary" not in bundle[key]


def test_single_sleeve_loader_refuses_a_financing_leg():
    with pytest.raises(NotImplementedError, match="two-sleeve"):
        cal.load_cross_asset_data({"financing": {"model": "tbill"}, "universe": {"assets": ["SPY"]}})


# --------------------------------------------------------------------------- #
# Research side: held weights + excess returns
# --------------------------------------------------------------------------- #
def _xsec():
    sys.path.insert(0, str(ROOT / "scripts" / "research"))
    import xsec_momentum_falsification as mom
    return mom


def test_held_weights_regenerate_the_research_backtest():
    mom = _xsec()
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2020-01-01", periods=120)
    rets = pd.DataFrame(rng.normal(0, 0.01, (120, 3)), index=idx, columns=list("abc"))
    rets.iloc[0] = np.nan
    rebal = idx[[10, 40, 70, 100]]
    w = pd.DataFrame(rng.normal(0, 0.5, (4, 3)), index=rebal, columns=list("abc"))
    held = mom.held_weights(w, rets)
    gross, _, _ = mom.backtest(w, rets)
    np.testing.assert_array_equal((held * rets).sum(axis=1).to_numpy(), gross.to_numpy())
    assert held.loc[idx[10]].abs().sum() == 0.0          # decided at close 10, held from 11
    np.testing.assert_array_equal(held.loc[idx[11]].to_numpy(), w.loc[rebal[0]].to_numpy())


def test_excess_returns_is_r_minus_net_rf_minus_short_borrow():
    idx = pd.bdate_range("2024-01-01", periods=3)
    r = pd.Series([0.010, -0.002, 0.004], index=idx)
    held = pd.DataFrame([[1.0, -0.5], [0.6, 0.2], [0.0, -1.0]], index=idx, columns=["a", "b"])
    cash = pd.Series([1e-4, 2e-4, 3e-4], index=idx)
    borrow = pd.Series([1e-5, 1e-5, 2e-5], index=idx)
    ex = fin.excess_returns(r, held, cash, borrow)
    expect = [0.010 - 0.5 * 1e-4 - 0.5 * 1e-5, -0.002 - 0.8 * 2e-4, 0.004 + 1.0 * 3e-4 - 1.0 * 2e-5]
    np.testing.assert_allclose(ex.to_numpy(), expect, rtol=1e-12)
    with pytest.raises(ValueError, match="cash rate"):
        fin.excess_returns(r, held, cash.iloc[:2])


# --------------------------------------------------------------------------- #
# validate_config parses the block like the loader; artifacts stamp the model
# --------------------------------------------------------------------------- #
def _validator():
    sys.path.insert(0, str(ROOT))
    from scripts.validate_config import ValidationResult, check_financing_block
    return ValidationResult, check_financing_block


def _check_fin(cfg: dict):
    vr, check = _validator()
    r = vr()
    check(cfg, r)
    return r


def test_validate_config_parses_the_financing_block_like_the_loader():
    cfg = yaml.safe_load((ROOT / "configs" / "tailwind_v1_challenge.yaml").read_text(encoding="utf-8"))
    r = _check_fin(cfg)
    assert r.status == "PASS" and r.passed == ["financing = tbill 3m act/360 + 25 bp short borrow"]
    typo = {**cfg, "financing": {**cfg["financing"], "borrow_bps": 25}}
    assert _check_fin(typo).status == "FAIL"                         # unknown key
    assert _check_fin({"financing": {"model": "none"}}).status == "PASS"
    silent = _check_fin({})
    assert (silent.failures, silent.warnings, silent.passed) == ([], [], [])


def test_validate_config_fails_a_financing_leg_the_path_cannot_carry():
    """The single-sleeve (RL allocator) loader refuses a financing block, so declaring one there
    FAILs validation rather than running unfinanced."""
    rl = {"env": {"type": "multi_asset_allocator"}, "financing": {"model": "tbill"}}
    r = _check_fin(rl)
    assert r.status == "FAIL" and "linear-core" in r.failures[0]


def test_validate_config_warns_that_return_streams_stay_unfinanced():
    cfg = {"env": {"type": "multi_asset_allocator"},
           "execution": {"shadow": "linear_core_multi_sleeve"},
           "sleeves": {"momentum": {"signal": "tsmom"}, "vrp": {"type": "return_stream"}},
           "financing": {"model": "tbill"}}
    r = _check_fin(cfg)
    assert r.status == "WARN" and "['vrp']" in r.warnings[0]


def test_execution_stamp_records_the_lead_and_the_financing_model():
    from sharpen.envs.allocator_factory import execution_stamp
    cfg = yaml.safe_load((ROOT / "configs" / "tailwind_v1_challenge.yaml").read_text(encoding="utf-8"))
    assert execution_stamp(cfg) == {
        "decision_lead_bars": 1,
        "financing": {"model": "tbill", "tenor": "3m", "day_count": 360, "short_borrow_bps": 25.0}}
    assert execution_stamp({}) == {"decision_lead_bars": 0, "financing": {"model": "none"}}
