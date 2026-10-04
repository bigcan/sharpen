"""FinRL-X bridge: our engine must reproduce FinRL-X's own BacktestEngine NAV on the same long-only weights.

Runs FinRL-X in its own venv (C:/FinRL/FinRL-Trading/.venv); skipped when the clone is absent.
Run: python -m pytest -q tests/research/test_finrlx_strategy_bridge.py -p no:cacheprovider
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]
for p in (str(_ROOT / "scripts"), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from research.finrlx_strategy import finrlx_bridge as fb  # noqa: E402
from research.finrlx_strategy.paths import FINRLX_VENV_PY  # noqa: E402

needs_clone = pytest.mark.skipif(not FINRLX_VENV_PY.exists(), reason="FinRL-X clone/venv not present")


def _inputs(n: int = 300, seed: int = 11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-02", periods=n)
    px = pd.DataFrame(np.exp(np.cumsum(rng.normal(0.0003, 0.01, (n, 3)), axis=0)), index=idx, columns=["A", "B", "C"])
    dec = idx[::21]
    tg = pd.DataFrame(rng.dirichlet([1, 1, 1], len(dec)) * 0.8, index=dec, columns=px.columns)
    rf = pd.Series(0.0001, index=idx)
    return px, tg, rf


def test_contract_refuses_shorts():
    px, tg, _ = _inputs()
    tg.iloc[3, 0] = -0.1
    with pytest.raises(ValueError):
        fb.contract_weights(tg, px.index)


def test_contract_rows_sum_to_one_with_cash():
    px, tg, _ = _inputs()
    w = fb.contract_weights(tg, px.index)
    assert np.allclose(w.sum(axis=1).loc[tg.index[0]:], 1.0)


@needs_clone
@pytest.mark.parametrize("tc_bps", [0.0, 5.0])
def test_parity_with_finrlx_backtest_engine(tc_bps, tmp_path, monkeypatch):
    monkeypatch.setattr(fb, "RESULTS", tmp_path)
    px, tg, rf = _inputs()
    out = fb.parity(px, tg, rf, tc_bps=tc_bps, spread_bps_yr=15.0, start=px.index[0], name=f"t{tc_bps:g}")
    assert out["max_abs_rel_nav_diff"] < (1e-12 if tc_bps == 0 else 5e-5)


@needs_clone
def test_planted_lag_mismatch_breaks_parity(tmp_path, monkeypatch):
    """Planted bug: hand FinRL-X unlagged weights (trade at the decision close). Parity must break."""
    monkeypatch.setattr(fb, "RESULTS", tmp_path)
    px, tg, rf = _inputs()
    cal = px.index
    pxc = px.copy()
    pxc[fb.CASH] = fb.cash_price(rf, 15.0)
    w_dec = fb.contract_weights(tg, cal)
    ours = fb.replica(pxc, w_dec, 0.0)
    theirs = fb.run_finrlx(pxc, w_dec, 0.0, "planted", tmp_path / "planted")   # NOT shifted
    j = ours.index.intersection(theirs.index)
    assert float((ours.loc[j] / ours.loc[j].iloc[0] / (theirs.loc[j] / theirs.loc[j].iloc[0]) - 1).abs().max()) > 1e-4
