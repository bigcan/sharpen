"""Tripwires for the timing of the paper executor's P&L attribution (TAILWIND Tier-2 N11).

``ParityHarness._replay`` fills weight row ``W[k]`` at close ``k+1``, so ``W[k]`` first earns the
``(k+1)->(k+2)`` move and the book's step-``k`` return is ``W[k-1]`` times the ``k->k+1`` move.
Both attribution sites must pair them that way: ``ParityHarness._attribution`` (``class_pnl``,
read by the ``max_single_class_pnl_share`` drift gate) and
``TwoSleeveExecutor._sleeve_attribution`` (``sleeve_pnl``, reporting). Pairing ``W[k]`` with the
``k->k+1`` move credits each row with the move BEFORE it was entered (audit findings T2-10 /
T4-09 / T6-13).

Pinned in both directions on zero-cost books: the summed attribution reconciles to the book's
gross P&L (the sum of its step returns), and the one-bar-early pairing on the same book does not.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sharpen.envs import allocator_factory
from sharpen.paper import TwoSleeveExecutor
from sharpen.paper.parity_harness import ParityHarness

ROOT = Path(__file__).resolve().parents[2]
_ZERO_COST = {"taker_fee": 0.0, "slippage_base_bps": 0.0, "slippage_impact_bps": 0.0}


# ------------------------------------------------------------------------- hand case
def test_hand_case_credits_only_the_moves_after_the_fill():
    """EQ gains 10% over 0->1, before the first rows fill at close 1; the book then earns 2% on
    EQ and 2.5% on RT over 1->2, and nothing after. Only the 1->2 moves count. The one-bar-early
    pairing also credits the 10%: equity 0.06, 83% of the class P&L instead of 44%."""
    price = np.array([[100.0, 100.0], [110.0, 100.0], [112.2, 102.5], [112.2, 102.5]])
    W = np.array([[0.5, 0.5], [0.5, 0.5], [0.0, 0.0]])        # row k fills at close k+1
    ts = (pd.bdate_range("2021-03-01", periods=4).asi8 // 10**9).astype(np.int64)
    arrays = {"price_ary": price, "volume_ary": np.full(price.shape, 1e12),
              "carry_ary": np.zeros(price.shape), "timestamps": ts, "assets": ["EQ", "RT"]}
    cfg = {"env": dict(_ZERO_COST),
           "universe": {"assets": ["EQ", "RT"], "asset_class": {"EQ": "equity", "RT": "rates"}}}
    live = ParityHarness(cfg)._replay(arrays, W, fill_engine=None)
    assert live.step_returns == pytest.approx([0.0, 0.0225, 0.0], abs=1e-9)
    assert live.class_pnl == pytest.approx({"equity": 0.01, "rates": 0.0125}, abs=1e-12)
    assert live.max_single_class_pnl_share() == pytest.approx(0.0125 / 0.0225)

    # The same book as two single-asset sleeves at alpha 0.5 each.
    sleeve_pnl = TwoSleeveExecutor._sleeve_attribution(
        {"momentum": 2 * W[:, :1], "defensive": 2 * W[:, 1:]},
        {"momentum": ["EQ"], "defensive": ["RT"]},
        {"momentum": np.full(3, 0.5), "defensive": np.full(3, 0.5)}, ["EQ", "RT"], price)
    assert sleeve_pnl == pytest.approx({"momentum": 0.01, "defensive": 0.0125}, abs=1e-12)


# ------------------------------------------------------------------------- reconciliation
def _month_end_idx(ts: np.ndarray) -> np.ndarray:
    months = pd.to_datetime(ts, unit="s").to_period("M")
    return pd.Series(np.arange(len(ts))).groupby(months.values).max().to_numpy()


def _zero_cost_bundle(*, T: int = 300, n: int = 4, jump: float = 0.005, seed: int = 11) -> dict:
    """Two month-held sleeves over four assets at constant vol (the target, so each weight is
    the held conviction), made adversarial for the attribution. Both sleeves flip the whole
    cross-section at every month-end, and the moves into the two closes that can fill that
    switch (d under the decision lead, d+1 under the legacy drive) go toward the NEW book. The
    move just before the fill is therefore earned on the OLD position under either drive. Small
    noise elsewhere keeps the risk-parity vols live."""
    rng = np.random.default_rng(seed)
    ts = (pd.bdate_range("2019-01-01", periods=T).asi8 // 10**9).astype(np.int64)
    months = pd.factorize(pd.to_datetime(ts, unit="s").to_period("M"))[0]
    sign = np.where(months % 2 == 0, 1.0, -1.0)
    rets = rng.normal(0.0, 0.0005, (T - 1, n))                 # rets[k]: the k->k+1 move
    for d in _month_end_idx(ts)[:-1]:                         # interior month-ends
        rets[d - 1:d + 1] += jump * sign[d] * np.linspace(0.5, 1.5, n)
    price = 100.0 * np.vstack([np.ones(n), np.cumprod(1.0 + rets, axis=0)])
    names = [f"A{i}" for i in range(n)]

    def sleeve(conv: np.ndarray) -> dict:
        return {"price_ary": price.copy(), "tech_ary": conv.astype(np.float32),
                "vol_ary": np.full((T, n), 0.10), "carry_ary": np.zeros((T, n)),
                "volume_ary": np.full((T, n), 1e12), "timestamps": ts, "conviction_ary": conv,
                "assets": names, "conviction_cutoff_lag": 1, "vol_cutoff_lag": 1}

    union = {"price_ary": price.copy(), "volume_ary": np.full((T, n), 1e12),
             "carry_ary": np.zeros((T, n)), "timestamps": ts, "assets": names}
    return {"momentum": sleeve(np.repeat(sign[:, None], n, axis=1)),
            "defensive": sleeve(sign[:, None] * np.linspace(0.3, 0.9, n)), "union": union}


def _zero_cost_cfg(lead: int, names: list[str]) -> dict:
    """The TAILWIND book config with every cost at zero, so the book's P&L is gross, and two
    asset classes over the synthetic names."""
    cfg = yaml.safe_load((ROOT / "configs" / "tailwind_v1.yaml").read_text(encoding="utf-8"))
    cfg["env"].update(_ZERO_COST, min_trade_pct=0.0)
    cfg.setdefault("execution", {})["decision_lead_bars"] = lead
    half = len(names) // 2
    cfg["universe"] = {"assets": names, "asset_class": {
        a: "equity" if i < half else "rates" for i, a in enumerate(names)}}
    return cfg


@pytest.mark.parametrize("lead", [0, 1])
def test_attribution_reconciles_to_the_book_gross_pnl(lead):
    if lead and not hasattr(allocator_factory, "decision_lead_bars"):
        pytest.skip("execution.decision_lead_bars (full-codebase audit R-3) is not in this tree")
    bundle = _zero_cost_bundle()
    live, _ = TwoSleeveExecutor(_zero_cost_cfg(lead, bundle["union"]["assets"])).run(bundle)
    r = live.step_returns
    assert live.cumulative_fees[-1] == 0.0                     # zero cost: net P&L == gross
    book = float(r.sum())
    # The attribution weights are target labels; the book holds fixed entry notionals, sized
    # on the pre-fill equity and then drifting with price. The two agree to second order in
    # the step moves, not exactly (the residual is <= ~1.5% of the gross activity here).
    tol = 0.05 * float(np.abs(r).sum())
    by_class = sum(live.class_pnl.values())
    by_sleeve = sum(live.sleeve_pnl.values())
    assert by_class == pytest.approx(by_sleeve, abs=1e-12)     # both sites pair the same way
    assert abs(by_class - book) <= tol, (by_class, book, tol)
    assert abs(by_sleeve - book) <= tol, (by_sleeve, book, tol)

    # Teeth: on the same book the one-bar-early pairing misses by far more than the tolerance.
    rets = ParityHarness._asset_returns(bundle["union"]["price_ary"])
    early = float((live.weights * rets).sum())
    assert abs(early - book) > 5 * tol, (early, book, tol)
