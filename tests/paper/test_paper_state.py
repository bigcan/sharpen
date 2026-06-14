"""PaperState — book accounting pinned to MultiAssetAllocatorEnv + persistence + order-gen.

The direct env-pin (``test_book_matches_env_step_by_step``) is the diagnostic backbone:
it drives the env and the book in lockstep on the SAME realized weight deltas and
asserts positions / entry_prices / entry_notionals / margin / equity match at EVERY
step. If the book ever drifts from the validated env accounting, this localizes it to
a single step (the aggregate keystone parity test only sees the end-to-end sum).
"""
from __future__ import annotations

import numpy as np

from finrl_pro_ds.envs.allocator_factory import linear_core_weights, make_allocator_env, monthly_rebal_conviction
from finrl_pro_ds.envs.multi_asset_allocator_env import MultiAssetAllocatorEnv
from finrl_pro_ds.paper.fill_engine import SimFillEngine
from finrl_pro_ds.paper.paper_state import PaperState, generate_orders

from .conftest import allocator_arrays

_LEVERS_OFF = {"no_trade_band": 0.0, "rebalance_interval": 1, "cost_penalty_scale": 0.0}


def test_book_matches_env_step_by_step(cfg):
    """The book reproduces the env's per-step accounting exactly (every state var)."""
    arrays = allocator_arrays(T=90, n=3, seed=4, volume=1e7)
    env = make_allocator_env(arrays, cfg, overrides=_LEVERS_OFF, eval_mode=True)
    conv = monthly_rebal_conviction(arrays["timestamps"], arrays["conviction_ary"])
    env.reset()

    book = PaperState(n_assets=3, initial_capital=env.initial_capital)
    eng = SimFillEngine(taker_fee_pct=env.taker_fee_pct,
                        slippage_base_bps=env.slippage_base_bps,
                        slippage_impact_bps=env.slippage_impact_bps)
    price, volume, carry = arrays["price_ary"], arrays["volume_ary"], arrays["carry_ary"]

    done = False
    while not done:
        k = env.step_idx
        old_pos = env.positions.copy()
        _, _, term, trunc, info = env.step(conv[k])
        delta = info["position"] - old_pos                  # env's realized (post dust/cap) delta

        prev_price, price_now = price[k], price[k + 1]
        pv_before = book.pv_before(prev_price)
        fill = eng.fill(delta_weights=delta, ref_prices=price_now,
                        pv_before=pv_before, dollar_volume=volume[k])
        binfo = book.step_bar(delta_weights=delta, fill=fill, prev_price=prev_price,
                              price_now=price_now, carry_rates=carry[k + 1], pv_before=pv_before)

        np.testing.assert_allclose(book.positions, env.positions, atol=1e-9,
                                   err_msg=f"position drift at step {k}")
        np.testing.assert_allclose(book.entry_prices, env.entry_prices, atol=1e-7,
                                   err_msg=f"entry_price drift at step {k}")
        np.testing.assert_allclose(book.entry_notionals, env.entry_notionals, atol=1e-6,
                                   err_msg=f"entry_notional drift at step {k}")
        assert abs(book.margin_balance - env.margin_balance) < 1e-6, f"margin drift at step {k}"
        assert abs(binfo["portfolio_value"] - info["portfolio_value"]) < 1e-6, f"PV drift at step {k}"
        assert abs(binfo["step_return"] - info["step_return"]) < 1e-9, f"return drift at step {k}"
        done = term or trunc


def test_book_tracks_linear_core_weights(cfg):
    """Feeding the frozen-core weight trajectory, the book's positions equal it exactly
    (the executor's live invariant: held == the env-processed target)."""
    arrays = allocator_arrays(T=120, n=4, seed=9, volume=1e7)
    W = linear_core_weights(arrays, cfg)
    book = PaperState(n_assets=4, initial_capital=float(cfg["env"]["initial_capital"]))
    eng = SimFillEngine(taker_fee_pct=cfg["env"]["taker_fee"],
                        slippage_base_bps=cfg["env"]["slippage_base_bps"],
                        slippage_impact_bps=cfg["env"]["slippage_impact_bps"])
    price, volume, carry = arrays["price_ary"], arrays["volume_ary"], arrays["carry_ary"]
    for k in range(len(price) - 1):
        pv_before = book.pv_before(price[k])
        delta = generate_orders(W[k], book.positions, min_trade_pct=0.0)
        fill = eng.fill(delta_weights=delta, ref_prices=price[k + 1],
                        pv_before=pv_before, dollar_volume=volume[k])
        book.step_bar(delta_weights=delta, fill=fill, prev_price=price[k],
                      price_now=price[k + 1], carry_rates=carry[k + 1], pv_before=pv_before)
        np.testing.assert_allclose(book.positions, W[k], atol=1e-12,
                                   err_msg=f"book position != W[{k}]")


def test_generate_orders_deadband():
    target = np.array([0.5, 0.002, -0.3])
    held = np.array([0.1, 0.0, -0.3])
    # delta = [0.4, 0.002, 0.0]; 0.002 < 0.005 ⇒ zeroed.
    np.testing.assert_allclose(generate_orders(target, held, min_trade_pct=0.005), [0.4, 0.0, 0.0])
    # deadband 0 ⇒ raw delta passes through.
    np.testing.assert_allclose(generate_orders(target, held, min_trade_pct=0.0), [0.4, 0.002, 0.0])


def test_persistence_round_trip(tmp_path):
    book = PaperState(n_assets=3, initial_capital=100_000.0, assets=["SPY", "TLT", "GLD"])
    book.positions = np.array([0.3, -0.5, 0.0])
    book.entry_prices = np.array([100.0, 50.0, 0.0])
    book.entry_notionals = np.array([30_000.0, 50_000.0, 0.0])
    book.margin_balance = 98_000.0
    book.realized_pnl = 123.45
    book.cumulative_fees = 45.6
    book.cumulative_carry = 2.1
    book.peak_equity = 101_000.0
    book.as_of_ts = 1_700_000_000

    book.save(tmp_path)
    b2 = PaperState.load(tmp_path)

    assert b2.n_assets == 3
    assert b2.assets == ["SPY", "TLT", "GLD"]
    np.testing.assert_array_equal(b2.positions, book.positions)
    np.testing.assert_array_equal(b2.entry_prices, book.entry_prices)
    np.testing.assert_array_equal(b2.entry_notionals, book.entry_notionals)
    assert b2.margin_balance == book.margin_balance
    assert b2.realized_pnl == book.realized_pnl
    assert b2.cumulative_fees == book.cumulative_fees
    assert b2.cumulative_carry == book.cumulative_carry
    assert b2.peak_equity == book.peak_equity
    assert b2.as_of_ts == book.as_of_ts


def test_book_matches_env_through_liquidation():
    """The liquidation guard (margin < 0 ⇒ force-close + floor at 0) is never hit by the
    linear core, so the keystone replay can't exercise it. Force it: a max-leverage long
    into an 80% crash, closed the next bar, realizes a loss > margin. Drive env + book in
    lockstep and assert they reach the SAME liquidated state (positions 0, margin 0)."""
    price = np.array([[100.0], [100.0], [20.0]])          # 80% crash at the close bar
    T = 3
    common = dict(
        price_ary=price, tech_ary=np.zeros((T, 1), np.float32),
        vol_ary=np.full((T, 1), 0.05),                    # scale = target_vol/vol = 2.0 = lev_cap
        carry_ary=np.zeros((T, 1)), volume_ary=np.full((T, 1), 1e15),
        timestamps=(np.arange(T, dtype=np.int64) * 86400),
    )
    env = MultiAssetAllocatorEnv(
        **common, target_vol_asset=0.10, lev_cap=2.0, max_gross_exposure=10.0,
        taker_fee_pct=0.0, slippage_base_bps=0.0, slippage_impact_bps=0.0,
        min_trade_pct=0.0, turnover_penalty=0.0, reward_type="simple",
        circuit_breaker_threshold=0.0, initial_capital=100_000.0,
    )
    env.reset()
    book = PaperState(n_assets=1, initial_capital=100_000.0)
    eng = SimFillEngine(taker_fee_pct=0.0, slippage_base_bps=0.0, slippage_impact_bps=0.0)

    liquidation_seen = False
    for i, act in enumerate([np.array([1.0]), np.array([0.0])]):   # open max long, then close into the crash
        k = env.step_idx
        old = env.positions.copy()
        _, _, _, _, info = env.step(act)
        delta = info["position"] - old
        pv_before = book.pv_before(price[k])
        fill = eng.fill(delta_weights=delta, ref_prices=price[k + 1],
                        pv_before=pv_before, dollar_volume=common["volume_ary"][k])
        binfo = book.step_bar(delta_weights=delta, fill=fill, prev_price=price[k],
                              price_now=price[k + 1], carry_rates=common["carry_ary"][k + 1],
                              pv_before=pv_before)
        np.testing.assert_allclose(book.positions, env.positions, atol=1e-9)
        assert abs(book.margin_balance - env.margin_balance) < 1e-6
        assert abs(binfo["portfolio_value"] - info["portfolio_value"]) < 1e-6
        if i == 1:
            liquidation_seen = (env.margin_balance == 0.0 and not env.positions.any())
    assert liquidation_seen, "liquidation guard was not exercised — test scenario is stale"
    assert book.margin_balance == 0.0 and not book.positions.any()


def test_fresh_book_defaults():
    book = PaperState(n_assets=5, initial_capital=250_000.0)
    assert book.margin_balance == 250_000.0
    assert book.peak_equity == 250_000.0
    assert book.positions.shape == (5,) and not book.positions.any()
    assert len(book.assets) == 5


def test_load_preserves_liquidated_book(tmp_path):
    """A liquidated/flattened book persists margin_balance == 0.0; load() must NOT
    resurrect it to initial_capital via the __post_init__ unset-heuristic (P10-02)."""
    book = PaperState(n_assets=3, initial_capital=100_000.0, assets=["SPY", "TLT", "GLD"])
    book.margin_balance = 0.0                 # liquidation guard floored it to exactly 0.0
    book.peak_equity = 137_000.0              # the real historical peak (DD is measured from here)
    book.positions = np.zeros(3)
    book.realized_pnl = -100_000.0
    book.save(tmp_path)

    b2 = PaperState.load(tmp_path)
    assert b2.margin_balance == 0.0           # NOT resurrected to 100_000
    assert b2.peak_equity == 137_000.0        # preserved, not reset to initial
    assert not b2.positions.any()


def test_resume_continuity(tmp_path, cfg):
    """run → save → load → continue must equal an uninterrupted run (the justification for a
    separate forward book). A to_frame/load regression that round-trips fields but corrupts
    continuation would pass test_persistence_round_trip yet silently break a resumed soak."""
    arrays = allocator_arrays(T=120, n=4, seed=13, volume=1e7)
    W = linear_core_weights(arrays, cfg)
    price, volume, carry = arrays["price_ary"], arrays["volume_ary"], arrays["carry_ary"]
    eng = SimFillEngine(taker_fee_pct=cfg["env"]["taker_fee"],
                        slippage_base_bps=cfg["env"]["slippage_base_bps"],
                        slippage_impact_bps=cfg["env"]["slippage_impact_bps"])
    cap = float(cfg["env"]["initial_capital"])

    def step(book, k):
        pv_before = book.pv_before(price[k])
        delta = generate_orders(W[k], book.positions, min_trade_pct=0.0)
        fill = eng.fill(delta_weights=delta, ref_prices=price[k + 1], pv_before=pv_before,
                        dollar_volume=volume[k])
        return book.step_bar(delta_weights=delta, fill=fill, prev_price=price[k],
                             price_now=price[k + 1], carry_rates=carry[k + 1], pv_before=pv_before)

    cont = PaperState(n_assets=4, initial_capital=cap)
    info_c = None
    for k in range(len(price) - 1):
        info_c = step(cont, k)

    mid = (len(price) - 1) // 2
    book = PaperState(n_assets=4, initial_capital=cap)
    for k in range(mid):
        step(book, k)
    book.save(tmp_path)                       # interrupt: persist + reload mid-soak
    book = PaperState.load(tmp_path)
    info_r = None
    for k in range(mid, len(price) - 1):
        info_r = step(book, k)

    np.testing.assert_allclose(book.positions, cont.positions, atol=1e-9)
    np.testing.assert_allclose(book.entry_notionals, cont.entry_notionals, atol=1e-6)
    assert abs(book.margin_balance - cont.margin_balance) < 1e-6
    assert abs(book.cumulative_fees - cont.cumulative_fees) < 1e-6
    assert abs(info_r["portfolio_value"] - info_c["portfolio_value"]) < 1e-6
