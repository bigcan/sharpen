"""Tests for the Options-VRP instrument A/B bake-off
(``scripts/research/options_vrp_skew_falsification.py``;
spec ``.agent/artifacts/options_vrp_instrument_ab_spec.md``).

Coverage:
  * ``leg_vega`` / ``leg_gamma`` per-leg primitives (call+put == straddle; 0 at expiry).
  * signed-leg ``build_structure``: leg counts/sides, long buys@ask vs short sells@bid,
    net credit(iron_fly) < credit(straddle), the inverse-only DATA-CLEAN filter that
    drops USDC-linear duplicates.
  * TRIPWIRES — the two safety claims an iron fly must earn: a BOUNDED settlement loss
    (the cap is real) and a SMALLER |net vega| than the straddle (wings shrink short vega).
  * the per-instrument phase-1 + tail gate (no inheritance) and the Sortino/CVaR selection.
  * SLOW (cached-chain) refactor-parity: with the data fix OFF the signed engine
    reproduces the committed straddle 0.6115 / strangle 1.0046 anchor EXACTLY, and the
    clean inverse-only path differs (the USDC-contamination regression guard).

Unit tests build a synthetic Deribit chain (no network); only the parity/integration
tests touch the cached parquet and are skipped when it is absent.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from sharpen.crypto import options_pricing as op

_ROOT = Path(__file__).resolve().parents[2]


def _load(mod_name: str, rel: str):
    spec = importlib.util.spec_from_file_location(mod_name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# scripts/ is not a package — load the falsification by path.
S = _load("vrp_skew_ab_test", "scripts/research/options_vrp_skew_falsification.py")

CHAIN_CACHE = _ROOT / "data" / "processed" / "deribit_chain"
HAVE_CHAIN = CHAIN_CACHE.exists() and any(CHAIN_CACHE.glob("chain_*.parquet"))


# ---------------------------------------------------------------------------
# Synthetic real-chain snapshot (self-consistent BS prices + greeks)
# ---------------------------------------------------------------------------
def _synth_chain(underlying=40_000.0, entry="2024-06-01", tenor_days=30, iv=0.60,
                 add_usdc=False):
    """A one-expiry strike ladder with coin-priced bid/ask/mark + real BS deltas. With
    ``add_usdc`` it also injects USD-priced ``BTC_USDC`` duplicates (the contamination)."""
    entry_ts = pd.Timestamp(entry, tz="UTC")
    expiry = entry_ts + pd.Timedelta(days=tenor_days)
    tau = tenor_days / S.ANN
    mults = [0.70, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.30, 1.45]
    rows = []

    def add(symbol, typ, K, coin_price, delta):
        rows.append({
            "symbol": symbol, "type": typ, "strike_price": float(K),
            "bid_price": coin_price * 0.99, "ask_price": coin_price * 1.01,
            "mark_price": coin_price, "mark_iv": iv * 100.0,
            "bid_iv": (iv - 0.005) * 100.0, "ask_iv": (iv + 0.005) * 100.0,
            "underlying_price": underlying, "expiry_dt": expiry, "delta": delta,
        })

    for m in mults:
        K = round(underlying * m, -2)
        c_usd = op.leg_price("call", underlying, K, iv, tau)
        p_usd = op.leg_price("put", underlying, K, iv, tau)
        c_coin, p_coin = c_usd / underlying, p_usd / underlying
        cd = op.leg_delta("call", underlying, K, iv, tau)
        pd_ = op.leg_delta("put", underlying, K, iv, tau)
        tag = f"BTC-{expiry:%d%b%y}-{int(K)}".upper()
        add(f"{tag}-C", "call", K, c_coin, cd)
        add(f"{tag}-P", "put", K, p_coin, pd_)
        if add_usdc:  # USDC-linear duplicate: price in USD, NOT coin (the X-unit leak)
            add(f"BTC_USDC-{expiry:%d%b%y}-{int(K)}-C".upper(), "call", K, c_usd, cd)
            add(f"BTC_USDC-{expiry:%d%b%y}-{int(K)}-P".upper(), "put", K, p_usd, pd_)

    return pd.DataFrame(rows), entry_ts


def _cfg(wing_delta=0.10):
    return S.SkewConfig(wing_delta=wing_delta)


# ---------------------------------------------------------------------------
# leg_vega / leg_gamma primitives
# ---------------------------------------------------------------------------
def test_leg_vega_gamma_additivity_and_expiry():
    op.bs_self_test()  # includes the new leg_vega/leg_gamma assertions
    Sp, K, sig, tau = 100.0, 110.0, 0.6, 30 / op.ANN
    # call vega == put vega at r=0
    assert op.leg_vega("call", Sp, K, sig, tau) == pytest.approx(op.leg_vega("put", Sp, K, sig, tau))
    # the strike's two legs sum to its straddle vega/gamma
    assert op.leg_vega("call", Sp, Sp, sig, tau) + op.leg_vega("put", Sp, Sp, sig, tau) == \
        pytest.approx(op.straddle_vega(Sp, Sp, sig, tau))
    assert op.leg_gamma("call", Sp, Sp, sig, tau) + op.leg_gamma("put", Sp, Sp, sig, tau) == \
        pytest.approx(op.straddle_gamma(Sp, Sp, sig, tau))
    # zero once expired / degenerate
    assert op.leg_vega("call", Sp, K, sig, 0.0) == 0.0
    assert op.leg_gamma("put", Sp, K, sig, -1.0) == 0.0
    assert op.leg_vega("call", Sp, K, 0.0, tau) == 0.0


# ---------------------------------------------------------------------------
# build_structure — signed legs
# ---------------------------------------------------------------------------
def test_build_structure_leg_counts_and_sides():
    snap, entry = _synth_chain()
    cfg = _cfg()
    straddle = S.build_structure(snap, entry, "straddle", cfg)
    strangle = S.build_structure(snap, entry, "strangle", cfg)
    fly = S.build_structure(snap, entry, "iron_fly", cfg)
    condor = S.build_structure(snap, entry, "iron_condor", cfg)

    assert [lg["side"] for lg in straddle] == [-1, -1]
    assert [lg["side"] for lg in strangle] == [-1, -1]
    assert [lg["side"] for lg in fly] == [-1, -1, +1, +1]
    assert [lg["side"] for lg in condor] == [-1, -1, +1, +1]
    # every leg carries both quotes (longs buy @ask, shorts sell @bid)
    for lg in fly:
        assert "bid" in lg and "ask" in lg and lg["ask"] >= lg["bid"]
    # wings strictly OUTSIDE the shorts (a real spread)
    sc, sp, wc, wp = fly
    assert wc["K"] > sc["K"] and wp["K"] < sp["K"]


def test_build_structure_unknown_kind_raises():
    snap, entry = _synth_chain()
    with pytest.raises(ValueError):
        S.build_structure(snap, entry, "butterfly_supreme", _cfg())


def test_build_structure_bad_wing_delta_raises():
    snap, entry = _synth_chain()
    with pytest.raises(ValueError):
        S.build_structure(snap, entry, "iron_fly", _cfg(wing_delta=0.30))  # >= SHORT_DELTA


def test_straddle_strikes_atm_strangle_25delta():
    snap, entry = _synth_chain(underlying=40_000.0)
    straddle = S.build_structure(snap, entry, "straddle", _cfg())
    strangle = S.build_structure(snap, entry, "strangle", _cfg())
    # ATM call/put share the nearest-to-spot strike
    assert straddle[0]["K"] == straddle[1]["K"]
    # 25-delta shorts straddle the spot and are further OTM than the ATM
    assert strangle[0]["K"] > straddle[0]["K"] and strangle[1]["K"] < straddle[0]["K"]


# ---------------------------------------------------------------------------
# Signed entry accounting: net credit(iron_fly) < credit(straddle); fees scale w/ #legs
# ---------------------------------------------------------------------------
def _entry_accounting(legs, cfg):
    """Replicate simulate_real_chain's entry credit + fee for one set of legs."""
    U = legs[0]["underlying"]
    short_prem = sum(lg["bid"] * U for lg in legs if lg["side"] == -1)
    n = (cfg.premium_frac * cfg.initial_capital) / short_prem
    net_credit = n * sum((-lg["side"]) * (lg["bid"] if lg["side"] == -1 else lg["ask"]) * U
                         for lg in legs)
    fee = 0.0
    for lg in legs:
        px = lg["bid"] if lg["side"] == -1 else lg["ask"]
        fee += min(cfg.option_fee_pct_underlying * U, cfg.option_fee_cap_pct_premium * px * U) * n
    return n, net_credit, fee


def test_iron_fly_nets_smaller_credit_and_more_fees():
    snap, entry = _synth_chain()
    cfg = _cfg()
    straddle = S.build_structure(snap, entry, "straddle", cfg)
    fly = S.build_structure(snap, entry, "iron_fly", cfg)
    n_s, credit_s, fee_s = _entry_accounting(straddle, cfg)
    n_f, credit_f, fee_f = _entry_accounting(fly, cfg)
    # same shorts ⇒ same sizing; wings bought ⇒ smaller net credit, ~2x the leg fees
    assert n_f == pytest.approx(n_s)
    assert 0 < credit_f < credit_s
    assert fee_f > fee_s


# ---------------------------------------------------------------------------
# TRIPWIRE 1 — the iron-fly settlement loss is BOUNDED (the cap is real)
# ---------------------------------------------------------------------------
def _settlement_pnl(legs, n, net_credit_usd, S_exp):
    intr = sum(lg["side"] * (max(S_exp - lg["K"], 0.0) if lg["type"] == "call"
                             else max(lg["K"] - S_exp, 0.0)) for lg in legs)
    return net_credit_usd + n * intr


def test_tripwire_iron_fly_loss_is_bounded_straddle_is_not():
    snap, entry = _synth_chain(underlying=40_000.0)
    cfg = _cfg()
    fly = S.build_structure(snap, entry, "iron_fly", cfg)
    straddle = S.build_structure(snap, entry, "straddle", cfg)
    n_f, credit_f, _ = _entry_accounting(fly, cfg)
    n_s, credit_s, _ = _entry_accounting(straddle, cfg)

    sc, sp, wc, wp = fly
    max_width = max(wc["K"] - sc["K"], sp["K"] - wp["K"])
    cap = -(n_f * max_width - credit_f)  # theoretical worst settlement loss

    grid = [0.0, 1.0, sc["K"], wp["K"], wc["K"], 2 * sc["K"], 100 * sc["K"]]
    fly_losses = [_settlement_pnl(fly, n_f, credit_f, x) for x in grid]
    # bounded: never worse than the (wing_width - net_credit) cap (tiny float slack)
    assert min(fly_losses) >= cap - 1.0
    assert min(fly_losses) > -1e9  # finite
    # the naked straddle is UNBOUNDED: a 100x up-move loses ~ n * (100x - K)
    straddle_far = _settlement_pnl(straddle, n_s, credit_s, 100 * sc["K"])
    assert straddle_far < cap * 10  # far below the fly's bounded floor


# ---------------------------------------------------------------------------
# TRIPWIRE 2 — long wings SHRINK |net short vega| (the safety, quantified)
# ---------------------------------------------------------------------------
def _net_vega(legs, n, Sp, tau):
    return n * sum(lg["side"] * op.leg_vega(lg["type"], Sp, lg["K"], lg["mark_iv"], tau)
                   for lg in legs)


def test_tripwire_iron_fly_net_vega_below_straddle():
    snap, entry = _synth_chain(underlying=40_000.0)
    cfg = _cfg()
    straddle = S.build_structure(snap, entry, "straddle", cfg)
    fly = S.build_structure(snap, entry, "iron_fly", cfg)
    n_s, _, _ = _entry_accounting(straddle, cfg)
    n_f, _, _ = _entry_accounting(fly, cfg)
    tau = straddle[0]["tau"]
    v_s = abs(_net_vega(straddle, n_s, 40_000.0, tau))
    v_f = abs(_net_vega(fly, n_f, 40_000.0, tau))
    assert 0 < v_f < v_s  # wings (long +vega) reduce the short book's |net vega|


# ---------------------------------------------------------------------------
# DATA-CLEAN — the inverse-only filter drops USDC-linear duplicates
# ---------------------------------------------------------------------------
def test_inverse_only_filter_drops_usdc_duplicates():
    clean, entry = _synth_chain(add_usdc=False)
    dirty, _ = _synth_chain(add_usdc=True)
    assert dirty["symbol"].str.contains("_USDC").any()  # contamination present in input
    cfg = _cfg()
    # with the filter ON, the USDC duplicates are invisible: selection is IDENTICAL to the
    # clean chain and every selected leg is coin-priced (<1.0) — the de-contamination fix.
    fly_clean = S.build_structure(clean, entry, "iron_fly", cfg)
    fly_dirty = S.build_structure(dirty, entry, "iron_fly", cfg)
    assert [lg["K"] for lg in fly_clean] == [lg["K"] for lg in fly_dirty]
    for lg in fly_dirty:
        assert lg["bid"] < 1.0 and lg["ask"] < 1.0
    # with the filter OFF, the USD-priced duplicate CAN be picked (here forced first so the
    # delta-tie argmin grabs it) — the bug the filter closes: a USD price read as coin.
    S.INVERSE_ONLY = False
    try:
        usdc_first = pd.concat([dirty[dirty.symbol.str.contains("_")],
                                dirty[~dirty.symbol.str.contains("_")]], ignore_index=True)
        fly_off = S.build_structure(usdc_first, entry, "iron_fly", cfg)
        assert any(lg["bid"] > 1.0 for lg in fly_off)  # contamination leaks in
    finally:
        S.INVERSE_ONLY = True


# ---------------------------------------------------------------------------
# Per-instrument gate (no inheritance) + Sortino/CVaR selection
# ---------------------------------------------------------------------------
def _metric_stub(**over):
    base = dict(net_sharpe=0.8, net_pf=1.3, recent_12m_sharpe=0.5, net_max_dd=0.10,
                worst_window_dd_pct=10.0, max_net_vega_per_100k=20_000.0,
                cvar95_pct_daily=-1.0, pos_years=5, n_years=6, sortino=0.8,
                sample_n_months=60, bootstrap_sortino_ci=None)
    base.update(over)
    return base


def test_instrument_verdict_independent_no_inheritance():
    g = S.GATES
    go = S._instrument_verdict(_metric_stub(), g)
    assert go["verdict"] == "GO" and not go["kills"]
    # a different payoff is scored on its OWN — a weak Sharpe fails regardless of others
    nogo = S._instrument_verdict(_metric_stub(net_sharpe=0.2, net_pf=1.0, sortino=0.2), g)
    assert nogo["verdict"] == "NO-GO"
    assert any("Sharpe" in k for k in nogo["kills"]) and any("PF" in k for k in nogo["kills"])


def test_instrument_verdict_tail_kills_fire():
    g = S.GATES
    v = S._instrument_verdict(_metric_stub(worst_window_dd_pct=30.0), g)  # > 25
    assert v["verdict"] == "NO-GO" and any("worst-window DD" in k for k in v["kills"])
    v2 = S._instrument_verdict(_metric_stub(cvar95_pct_daily=-9.0), g)  # < -8 floor
    assert any("CVaR95" in k for k in v2["kills"])
    v3 = S._instrument_verdict(_metric_stub(max_net_vega_per_100k=60_000.0), g)  # > 50k
    assert any("vega" in k for k in v3["kills"])


def test_selection_ranks_by_sortino_with_cvar_guard():
    g = S.GATES
    cells = [
        {"asset": "BTC", "instrument": "iron_fly",
         "metrics": _metric_stub(sortino=0.9, net_sharpe=0.6, cvar95_pct_daily=-2.0),
         "verdict": {"verdict": "GO", "kills": []}},
        {"asset": "BTC", "instrument": "straddle",
         "metrics": _metric_stub(sortino=1.4, net_sharpe=1.1, cvar95_pct_daily=-3.0),
         "verdict": {"verdict": "GO", "kills": []}},
        {"asset": "ETH", "instrument": "strangle",  # high Sortino but a NO-GO cell
         "metrics": _metric_stub(sortino=2.0), "verdict": {"verdict": "NO-GO", "kills": ["x"]}},
    ]
    sel = S.select_instrument(cells, g)
    # winner is the highest-Sortino PASSER (straddle 1.4), not the NO-GO ETH 2.0
    assert sel["winner"]["instrument"] == "straddle" and sel["winner"]["asset"] == "BTC"
    assert sel["ranking"][0]["sortino"] >= sel["ranking"][1]["sortino"]
    assert sel["n_go_cells"] == 2


def test_selection_cvar_floor_and_sortino_floor_block_winner():
    g = S.GATES
    # a GO cell whose Sortino is below the floor ⇒ not eligible ⇒ no winner
    cells = [{"asset": "BTC", "instrument": "strangle",
              "metrics": _metric_stub(sortino=0.10, cvar95_pct_daily=-1.0),
              "verdict": {"verdict": "GO", "kills": []}}]
    assert S.select_instrument(cells, g)["winner"] is None
    # a GO cell with a great Sortino but a CVaR below the floor ⇒ also blocked
    cells2 = [{"asset": "BTC", "instrument": "strangle",
               "metrics": _metric_stub(sortino=1.5, cvar95_pct_daily=-9.0),
               "verdict": {"verdict": "GO", "kills": []}}]
    assert S.select_instrument(cells2, g)["winner"] is None


def test_no_passer_keeps_incumbent():
    g = S.GATES
    cells = [{"asset": "BTC", "instrument": "iron_fly",
              "metrics": _metric_stub(), "verdict": {"verdict": "NO-GO", "kills": ["x"]}}]
    sel = S.select_instrument(cells, g)
    assert sel["winner"] is None and sel["ranking"] == []


def test_require_pass_phase1_flag_is_wired():
    """The flag must actually change behavior (not declared-but-not-wired): a NO-GO cell
    is excluded when True, admitted when False."""
    g = dict(S.GATES)
    cells = [{"asset": "BTC", "instrument": "iron_fly",
              "metrics": _metric_stub(sortino=0.9, cvar95_pct_daily=-1.0),
              "verdict": {"verdict": "NO-GO", "kills": ["x"]}}]
    g["require_pass_phase1"] = True
    assert S.select_instrument(cells, g)["winner"] is None        # phase1 NO-GO excluded
    g["require_pass_phase1"] = False
    assert S.select_instrument(cells, g)["winner"] is not None    # admitted when not required


def test_select_by_is_wired_not_hardcoded():
    g = dict(S.GATES)
    cells = [
        {"asset": "BTC", "instrument": "high_sortino",
         "metrics": _metric_stub(sortino=2.0, net_sharpe=0.6, cvar95_pct_daily=-1.0),
         "verdict": {"verdict": "GO", "kills": []}},
        {"asset": "BTC", "instrument": "high_sharpe",
         "metrics": _metric_stub(sortino=0.6, net_sharpe=1.5, cvar95_pct_daily=-1.0),
         "verdict": {"verdict": "GO", "kills": []}},
    ]
    g["select_by"] = "sortino"
    assert S.select_instrument(cells, g)["winner"]["instrument"] == "high_sortino"
    g["select_by"] = "sharpe"
    assert S.select_instrument(cells, g)["winner"]["instrument"] == "high_sharpe"
    g["select_by"] = "omega"  # unsupported ⇒ fail loud, never silently mis-rank+mis-report
    with pytest.raises(ValueError):
        S.select_instrument(cells, g)


# ---------------------------------------------------------------------------
# TRIPWIRE — LEAK-2 causality: a future month's chain/spot cannot move a past cycle
# ---------------------------------------------------------------------------
def test_tripwire_future_does_not_affect_past_cycle():
    m1, _ = _synth_chain(underlying=40_000.0, entry="2024-06-01", tenor_days=30)
    m2, _ = _synth_chain(underlying=44_000.0, entry="2024-07-01", tenor_days=30)
    snaps = {(2024, 6): m1, (2024, 7): m2}
    dates = pd.date_range("2024-06-01", "2024-08-05", freq="D", tz="UTC")
    spot = {pd.Timestamp(d): 40_000.0 * (1.0 + 0.001 * (i % 10 - 5)) for i, d in enumerate(dates)}
    fnd = {pd.Timestamp(d): 0.0 for d in dates}
    cfg = _cfg()
    cutoff = pd.Timestamp("2024-07-01", tz="UTC")

    pnl_a, _ = S.simulate_real_chain("BTC", snaps, spot, fnd, "iron_fly", cfg)
    # CORRUPT the future: blow up month-2's chain prices AND every July+ spot
    m2c = m2.copy()
    for col in ("bid_price", "ask_price", "mark_price"):
        m2c[col] = m2c[col] * 10.0
    snaps_c = {(2024, 6): m1, (2024, 7): m2c}
    spot_c = {k: (v * 5.0 if k >= cutoff else v) for k, v in spot.items()}
    pnl_b, _ = S.simulate_real_chain("BTC", snaps_c, spot_c, fnd, "iron_fly", cfg)

    past = pnl_a.index < cutoff
    # the first cycle (June) is BYTE-IDENTICAL — it never reads the future month/spots
    assert (pnl_a[past].to_numpy() == pnl_b[past].to_numpy()).all()
    # and the corruption is real (the tripwire CAN fail): the future P&L did move
    assert (pnl_a[~past] != pnl_b[~past]).any()


# ---------------------------------------------------------------------------
# SLOW — refactor parity + the USDC-contamination regression (needs cached chain)
# ---------------------------------------------------------------------------
def _btc_sharpe(kind, cfg):
    from sharpen.crypto.data import deribit_options_loader as dol
    from sharpen.crypto.data import tardis_options_chain_loader as tcl
    from sharpen.crypto.features import options_vol_features as ovf
    raw = dol.load({"universe": {"assets": ["BTC", "ETH"]}})
    chain = tcl.load_chain_snapshots("2021-04", "2026-06")
    chain["snapshot_date"] = pd.to_datetime(chain["snapshot_date"], utc=True)
    chain["expiry_dt"] = pd.to_datetime(chain["expiry_dt"], utc=True)
    snaps = {(d.year, d.month): g for d, g in chain.groupby("snapshot_date")}
    perp = ovf._to_date_index(raw.perp["BTC"])["close"]
    fund = ovf._daily_funding(raw.funding["BTC"])
    spot = {pd.Timestamp(d): float(v) for d, v in perp.items()}
    fnd = {pd.Timestamp(d): float(v) for d, v in fund.items()}
    pnl, _ = S.simulate_real_chain("BTC", snaps, spot, fnd, kind, cfg)
    return S._metrics(pnl, cfg)["net_sharpe"]


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_CHAIN, reason="needs cached Tardis chain parquet")
def test_refactor_parity_reproduces_contaminated_anchor():
    """With the data fix OFF, the signed engine MUST reproduce the committed real-chain
    anchor EXACTLY — proving the signed-leg refactor is parity-clean and the 0.61→-0.25
    swing is purely the USDC de-contamination, not the refactor."""
    cfg = _cfg()
    S.INVERSE_ONLY = False
    try:
        assert _btc_sharpe("straddle", cfg) == pytest.approx(0.611536442405016, abs=1e-6)
        assert _btc_sharpe("strangle", cfg) == pytest.approx(1.0046141555399581, abs=1e-6)
    finally:
        S.INVERSE_ONLY = True


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_CHAIN, reason="needs cached Tardis chain parquet")
def test_inverse_only_decontamination_changes_the_anchor():
    """The clean inverse-only straddle is materially WORSE than the contaminated 0.61 —
    the regression guard that the USDC fix stays in force."""
    cfg = _cfg()
    S.INVERSE_ONLY = True
    clean = _btc_sharpe("straddle", cfg)
    assert clean < 0.0  # cleaned BTC straddle is negative-Sharpe (was a spurious +0.61)


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_CHAIN, reason="needs cached Tardis chain parquet")
def test_bakeoff_runs_and_verdict_serializes_strict_json():
    import json
    r = S.run_instrument_ab()
    assert r["verdict"] in ("GO", "NO-GO")
    assert r["sample_n_months"] > 0
    assert set(r["per_instrument"]["BTC"]) == set(S.GATES["instruments"])
    json.dumps(r, allow_nan=False)  # strict JSON (no NaN/Inf) by construction
