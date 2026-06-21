"""Tests for the cross-asset OHLCV loader → env-array builder (Phase 4, S553-cont-34).

Two layers:
  - **Synthetic, no network**: array contracts + the two correctness-critical
    normalization invariants (``vol_ary`` is the RAW vol-scaling denominator, never
    z-scored; the obs ``baseline_weight`` column is passthrough, never z-scored —
    z-scoring it would break ADR-3) + an array-level look-ahead tripwire (LEAK-2).
  - **Real data (cache/network gated)**: the loader's ``baseline_weight`` reproduces
    the validated linear-core net Sharpe ~0.60 (analytic monthly backtest) — proves
    the loader builds the SAME book the falsification validated.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finrl_pro_ds.data import cross_asset_loader as loader
from finrl_pro_ds.features import cross_asset_signals as cas

ROOT = Path(__file__).resolve().parents[2]
ANN = 252


# --------------------------------------------------------------------------- #
# Synthetic fixture (deterministic; long enough to clear the 252+skip+vol warmup)
# --------------------------------------------------------------------------- #
def _synthetic():
    rng = np.random.default_rng(7)
    T = 700
    idx = pd.bdate_range("2017-01-02", periods=T)
    tickers = ["A", "B", "C", "D", "E", "F"]
    drift = np.array([0.0005, 0.0002, 0.0, -0.0001, 0.0004, 0.0001])
    steps = 1.0 + drift + rng.normal(0, 0.011, (T, len(tickers)))
    close = pd.DataFrame(100.0 * np.cumprod(steps, axis=0), index=idx, columns=tickers)
    volume = pd.DataFrame(rng.uniform(1e6, 5e6, (T, len(tickers))), index=idx, columns=tickers)
    asset_class = {"A": "eq", "B": "eq", "C": "eq", "D": "fx", "E": "fx", "F": "fx"}
    signals = cas.compute(close, asset_class=asset_class)
    return close, volume, tickers, asset_class, signals


def _arrays(norm_window=252):
    close, volume, tickers, _, signals = _synthetic()
    return loader.build_allocator_arrays(
        signals, close, volume, tickers, close.index[0], close.index[-1],
        norm_window=norm_window,
    ), close, volume, tickers, signals


# --------------------------------------------------------------------------- #
# Array contracts
# --------------------------------------------------------------------------- #
def test_array_shapes_and_tech_cols():
    arrays, close, _, tickers, _ = _arrays()
    T, N = len(close), len(tickers)
    assert arrays["price_ary"].shape == (T, N)
    assert arrays["vol_ary"].shape == (T, N)
    assert arrays["carry_ary"].shape == (T, N)
    assert arrays["volume_ary"].shape == (T, N)
    assert arrays["conviction_ary"].shape == (T, N)
    assert arrays["timestamps"].shape == (T,)
    expected = ["sig_tsmom_63", "sig_tsmom_126", "sig_tsmom_252",
                "trend_conviction", "vol", "xs_rank", "baseline_weight"]
    assert arrays["tech_cols"] == expected
    assert arrays["tech_ary"].shape == (T, N * len(expected))


def test_carry_is_zero_v1():
    arrays, *_ = _arrays()
    assert np.all(arrays["carry_ary"] == 0.0), "v1 is TSMOM-only (ADR-6): carry must be 0"


def test_all_arrays_finite():
    arrays, *_ = _arrays()
    for k in ("price_ary", "tech_ary", "vol_ary", "carry_ary", "volume_ary", "conviction_ary"):
        assert np.isfinite(arrays[k]).all(), f"{k} has non-finite entries"


def test_timestamps_are_epoch_seconds_ascending():
    arrays, close, *_ = _arrays()
    ts = arrays["timestamps"]
    assert (np.diff(ts) > 0).all(), "timestamps must be strictly ascending"
    assert pd.to_datetime(ts[0], unit="s").date() == close.index[0].date()


def test_volume_ary_is_dollar_volume():
    """F1 (Fable 2026-06-11): ``volume_ary`` must be DOLLAR volume (shares × close),
    NOT raw share count — the env divides order notional by it to get a dimensionless
    participation ratio, so raw share volume overstated slippage impact by ~price."""
    arrays, close, volume, tickers, _ = _arrays()
    price = arrays["price_ary"]
    share_vol = (volume[tickers].reindex(close.index)
                 .ffill().fillna(0.0).to_numpy(np.float64))
    np.testing.assert_allclose(arrays["volume_ary"], share_vol * price, rtol=1e-12, atol=1e-6)
    # And it must NOT be the raw share volume (the F1 bug); dollar volume is ~price× larger.
    assert not np.allclose(arrays["volume_ary"], share_vol), \
        "volume_ary is raw share volume (F1 regressed) — must be shares × price"


# --------------------------------------------------------------------------- #
# Normalization invariants (the correctness-critical part)
# --------------------------------------------------------------------------- #
def test_vol_ary_is_raw_not_zscored():
    """vol_ary is the vol-scaling denominator — must be RAW realized vol, never
    z-scored (z-scores would be ~[-5,5]; raw annualized vol is positive ~[0,2])."""
    arrays, close, _, tickers, signals = _arrays()
    raw_vol = (signals.pivot(index="date", columns="ticker", values="vol")
               .reindex(index=close.index, columns=tickers).to_numpy(np.float64))
    raw_vol = np.nan_to_num(raw_vol, nan=0.0, posinf=0.0, neginf=0.0)
    np.testing.assert_allclose(arrays["vol_ary"], raw_vol, atol=1e-12)
    # Sanity: post-warmup vols are positive and NOT clipped to a z-score range.
    post = arrays["vol_ary"][400:]
    assert (post > 0).any() and post.max() < 5.0 and post.min() >= 0.0


def test_baseline_weight_in_tech_is_passthrough():
    """The obs baseline_weight column must equal the RAW linear-core weight (ADR-3:
    the RL must be able to copy it to match the core) — NOT a z-score of it."""
    arrays, close, _, tickers, signals = _arrays()
    tcols = arrays["tech_cols"]
    bw_local = tcols.index("baseline_weight")
    n_tech = len(tcols)
    raw_bw = (signals.pivot(index="date", columns="ticker", values="baseline_weight")
              .reindex(index=close.index, columns=tickers).ffill().fillna(0.0))
    for j, tk in enumerate(tickers):
        col = arrays["tech_ary"][:, j * n_tech + bw_local]
        np.testing.assert_allclose(col, raw_bw[tk].to_numpy(np.float32), atol=1e-5,
                                   err_msg=f"baseline_weight for {tk} was normalized")


def test_vol_column_in_tech_is_normalized():
    """The unbounded `vol` obs column SHOULD be window-local z-scored (≠ raw vol,
    clipped to ±5)."""
    arrays, close, _, tickers, signals = _arrays()
    tcols = arrays["tech_cols"]
    vol_local = tcols.index("vol")
    n_tech = len(tcols)
    raw_vol = (signals.pivot(index="date", columns="ticker", values="vol")
               .reindex(index=close.index, columns=tickers).ffill().fillna(0.0))
    tech_vol = arrays["tech_ary"][:, 0 * n_tech + vol_local]
    assert not np.allclose(tech_vol, raw_vol[tickers[0]].to_numpy(np.float32)), \
        "vol obs column should be z-scored, not raw"
    assert tech_vol.max() <= 5.0 + 1e-4 and tech_vol.min() >= -5.0 - 1e-4


def test_sign_and_conviction_columns_are_passthrough():
    """Bounded signal columns (signs ∈ {-1,0,1}, conviction/rank ∈ [-1,1]) pass
    through unnormalized."""
    arrays, close, _, tickers, signals = _arrays()
    tcols = arrays["tech_cols"]
    n_tech = len(tcols)
    for name in ("sig_tsmom_63", "trend_conviction", "xs_rank"):
        local = tcols.index(name)
        raw = (signals.pivot(index="date", columns="ticker", values=name)
               .reindex(index=close.index, columns=tickers).ffill().fillna(0.0))
        col = arrays["tech_ary"][:, 0 * n_tech + local]
        np.testing.assert_allclose(col, raw[tickers[0]].to_numpy(np.float32), atol=1e-5,
                                   err_msg=f"{name} should be passthrough")


# --------------------------------------------------------------------------- #
# Look-ahead tripwire (LEAK-2) at the array-builder level
# --------------------------------------------------------------------------- #
def test_array_builder_is_causal():
    """Perturbing FUTURE bars must not change any array row at <= t (price excepted,
    since price IS the perturbed series at future rows)."""
    close, volume, tickers, asset_class, signals = _synthetic()
    base = loader.build_allocator_arrays(
        signals, close, volume, tickers, close.index[0], close.index[-1], norm_window=252)

    tp = int(len(close) * 0.6)
    close2 = close.copy()
    close2.iloc[tp + 1:] = close2.iloc[tp + 1:] * 1.3
    signals2 = cas.compute(close2, asset_class=asset_class)
    after = loader.build_allocator_arrays(
        signals2, close2, volume, tickers, close2.index[0], close2.index[-1], norm_window=252)

    for key in ("tech_ary", "vol_ary", "conviction_ary"):
        np.testing.assert_allclose(
            base[key][:tp + 1], after[key][:tp + 1], atol=1e-9,
            err_msg=f"LEAK-2: future perturbation changed {key} at rows <= {tp}")


# --------------------------------------------------------------------------- #
# Real-data parity (cache/network gated): loader signals reproduce the ~0.60 core
# --------------------------------------------------------------------------- #
def _analytic_monthly_net_sharpe(close: pd.DataFrame, baseline_weight: pd.DataFrame,
                                 cost: float = 0.0002) -> float:
    """Falsification-style monthly TSMOM net Sharpe from the loader's baseline_weight
    (w_eff = monthly weight shift(1); cost on rebalance turnover). Mirrors the
    keystone reference, sourced from the loader."""
    rets = close.pct_change()
    key = close.index.to_period("M")
    last = pd.DatetimeIndex(pd.Series(close.index, index=close.index).groupby(key).max().values)
    warmup = max(cas.DEFAULT_LOOKBACKS) + cas.DEFAULT_SKIP + cas.DEFAULT_VOL_WINDOW
    rebal = last[last >= close.index[warmup]]
    w_rebal = baseline_weight.loc[rebal].fillna(0.0)
    w_daily = w_rebal.reindex(close.index).ffill().fillna(0.0)
    gross = (w_daily.shift(1).fillna(0.0) * rets).sum(axis=1)
    dw = w_rebal.diff().abs().sum(axis=1)
    dw.iloc[0] = w_rebal.iloc[0].abs().sum()
    cost_daily = dw.reindex(close.index).fillna(0.0) * cost
    net = (gross - cost_daily).dropna().to_numpy()
    return float(net.mean() / net.std() * np.sqrt(ANN)) if net.std() > 0 else 0.0


# --------------------------------------------------------------------------- #
# Step-4 forward-path hardening: stale-scan + earned manifest status (P1-03/P1-05)
# and the loader freshness-gate (P1-02). No network — synthetic wides / monkeypatch.
# --------------------------------------------------------------------------- #
def _wide_frames(close: pd.DataFrame) -> dict:
    """Build {open,high,low,close,volume} wide frames from a close frame (OHLC == close,
    positive volume) — the shape ``_clean_wide`` / ``fetch_ohlcv_wide`` produce."""
    vol = pd.DataFrame(1e6, index=close.index, columns=close.columns)
    return {"open": close.copy(), "high": close.copy(), "low": close.copy(),
            "close": close.copy(), "volume": vol}


def _clean_close(T=200, seed=4) -> pd.DataFrame:
    idx = pd.bdate_range("2018-01-02", periods=T)
    rng = np.random.default_rng(seed)
    px = 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.011, (T, 2)), axis=0)
    return pd.DataFrame(px, index=idx, columns=["AAA", "BBB"])


def test_clean_wide_clean_data_not_flagged():
    """A normal random-walk ETF is NOT stale-flagged (no false positives at daily scale)."""
    wide = _wide_frames(_clean_close())
    _, report = loader._clean_wide(wide)
    assert all(not r["stale_flagged"] for r in report.values())
    assert all(r["stale_suspect_frac"] == 0.0 for r in report.values())


def test_clean_wide_flags_stale_run():
    """A >= min_run_bars flat-close run (the gmgp1-gold stale-print class) is flagged and
    its stale-print metrics recorded — the scan the manifest status is earned from."""
    close = _clean_close()
    close.iloc[100:114, close.columns.get_loc("AAA")] = float(close.iloc[100]["AAA"])  # 14 flat bars
    wide = _wide_frames(close)
    _, report = loader._clean_wide(wide)
    assert report["AAA"]["stale_flagged"] is True
    assert report["AAA"]["stale_suspect_frac"] > 0.0
    assert report["BBB"]["stale_flagged"] is False


def test_cache_covers_end_freshness_matrix():
    """_cache_covers_end: a finite end beyond date_max refetches; within tol reuses; an
    end=None cache is trusted by default but freshness-gated under require_fresh."""
    man = {"date_max": "2026-06-05"}
    # finite end within / beyond the cache
    assert loader._cache_covers_end(man, "2026-06-05", require_fresh=False, tol_days=5)
    assert loader._cache_covers_end(man, "2026-06-08", require_fresh=False, tol_days=5)   # within tol
    assert not loader._cache_covers_end(man, "2026-07-01", require_fresh=False, tol_days=5)
    # end=None: trusted unless require_fresh, then must be near today (2026-06-05 is far past)
    assert loader._cache_covers_end(man, None, require_fresh=False, tol_days=5)
    assert not loader._cache_covers_end(man, None, require_fresh=True, tol_days=5)


def test_fetch_and_clean_refetches_when_cache_too_old(tmp_path, monkeypatch):
    """A scheduled run asking for data beyond the cache's date_max must REFETCH, not freeze
    on the stale cache (P1-02). Counts the (monkeypatched) network fetch."""
    calls = {"n": 0}

    def fake_fetch(assets, start, end, *, auto_adjust=True):
        calls["n"] += 1
        return _wide_frames(_clean_close())               # index ends 2018 — deliberately old

    monkeypatch.setattr(loader, "fetch_ohlcv_wide", fake_fetch)
    # 1st call seeds the cache (date_max ~2018-10).
    loader.fetch_and_clean(["AAA", "BBB"], "2018-01-01", None, cache_dir=tmp_path)
    assert calls["n"] == 1
    # 2nd call (end=None, rolling-to-latest, not require_fresh) reuses the cache (no refetch)...
    loader.fetch_and_clean(["AAA", "BBB"], "2018-01-01", None, cache_dir=tmp_path)
    assert calls["n"] == 1
    # ...but asking for a finite end far beyond the cache's date_max forces a refetch
    # (the silent-freeze the audit named, P1-02).
    loader.fetch_and_clean(["AAA", "BBB"], "2018-01-01", "2026-06-30", cache_dir=tmp_path)
    assert calls["n"] == 2


def test_manifest_status_earned_from_stale_scan(tmp_path, monkeypatch):
    """The manifest status is DERIVED from the stale scan (P1-05), not a hardcoded 'PASS':
    clean data → PASS; a flat-print ticker → not PASS, with the flagged ticker recorded."""
    def clean_fetch(assets, start, end, *, auto_adjust=True):
        return _wide_frames(_clean_close())

    monkeypatch.setattr(loader, "fetch_ohlcv_wide", clean_fetch)
    _, man = loader.fetch_and_clean(["AAA", "BBB"], "2018-01-01", "2018-10-31",
                                    cache_dir=tmp_path, force_refetch=True)
    assert man["status"] == "PASS" and man["stale_scan"]["flagged_tickers"] == []

    def stale_fetch(assets, start, end, *, auto_adjust=True):
        close = _clean_close()
        close.iloc[100:114, close.columns.get_loc("AAA")] = float(close.iloc[100]["AAA"])
        return _wide_frames(close)

    monkeypatch.setattr(loader, "fetch_ohlcv_wide", stale_fetch)
    _, man2 = loader.fetch_and_clean(["AAA", "BBB"], "2018-01-01", "2018-10-31",
                                     cache_dir=tmp_path, force_refetch=True)
    assert man2["status"] in {"WARN", "FAIL"}
    assert "AAA" in man2["stale_scan"]["flagged_tickers"]


_CACHE = loader.DEFAULT_CACHE_DIR / "ohlcv_daily.parquet"


@pytest.mark.skipif(not _CACHE.exists(),
                    reason="real OHLCV cache absent (run cross_asset_pipeline / loader once)")
def test_loader_baseline_reproduces_linear_core():
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "cross_asset_momentum.yaml").read_text(encoding="utf-8"))
    data = loader.load_cross_asset_data(cfg)
    close = data["close"]
    bw = data["signals"].pivot(index="date", columns="ticker", values="baseline_weight") \
        .reindex(columns=close.columns)
    sharpe = _analytic_monthly_net_sharpe(close, bw)
    # Validated core ~0.60 (falsification 0.601, keystone 0.615). Band absorbs
    # yfinance auto-adjust vintage drift; a leak would inflate >>1, a broken signal ~0.
    assert 0.45 <= sharpe <= 0.85, f"loader baseline net Sharpe {sharpe:.3f} off the ~0.60 core"
