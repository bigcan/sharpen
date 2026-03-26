"""
G1 Tests: Gold (GC) Level-1 Feature Pipeline
=============================================

Tests that:
1. Level-1 (n_levels=1) produces exactly 18 micro features
2. Level-5 (n_levels=5, default) still produces 40 micro features (backward compat)
3. CME macro features have dow_sin/cos instead of funding_sin/cos
4. Crypto macro features still have funding_sin/cos (backward compat)
5. No NaN in output after warmup period
6. Feature column lists are correct
"""

import numpy as np
import pandas as pd

from finrl_pro_ds.data.feature_engineering import (
    MACRO_FEATURE_COLS,
    MICRO_FEATURE_COLS,
    DeepScalperFeatureEngineer,
    get_macro_feature_cols,
    get_micro_feature_cols,
)

# ─── Fixtures ──────────────────────────────────────────────────────────

def _make_lob_df(n_rows: int = 500, n_levels: int = 5, seed: int = 42) -> pd.DataFrame:
    """Create synthetic LOB data with the specified number of levels."""
    rng = np.random.RandomState(seed)

    base_price = 2700.0  # Gold ~$2700/oz
    timestamps = pd.date_range("2025-06-01", periods=n_rows, freq="5min")

    data = {"timestamp": timestamps}

    # Generate LOB levels
    for i in range(1, n_levels + 1):
        offset = (i - 1) * 0.10
        data[f"bid_price_{i}"] = base_price - offset + rng.randn(n_rows) * 0.5
        data[f"ask_price_{i}"] = base_price + 0.20 + offset + rng.randn(n_rows) * 0.5
        data[f"bid_vol_{i}"] = np.abs(rng.randn(n_rows) * 10 + 50)
        data[f"ask_vol_{i}"] = np.abs(rng.randn(n_rows) * 10 + 50)

    # Ensure no crossed quotes
    for i in range(1, n_levels + 1):
        data[f"ask_price_{i}"] = np.maximum(
            data[f"ask_price_{i}"], data[f"bid_price_{i}"] + 0.10,
        )

    # OHLCV (needed for macro features)
    data["open"] = base_price + rng.randn(n_rows) * 2
    data["high"] = data["open"] + np.abs(rng.randn(n_rows) * 3)
    data["low"] = data["open"] - np.abs(rng.randn(n_rows) * 3)
    data["close"] = data["open"] + rng.randn(n_rows) * 2
    data["volume"] = np.abs(rng.randn(n_rows) * 1000 + 5000)

    return pd.DataFrame(data)


# ─── Micro Feature Column Tests ──────────────────────────────────────

class TestGetMicroFeatureCols:
    """Test get_micro_feature_cols() returns correct lists."""

    def test_level5_returns_40_dims(self):
        cols = get_micro_feature_cols(n_levels=5)
        assert len(cols) == 40
        assert 'slope_asym' in cols
        assert 'depth_drain' in cols

    def test_level1_returns_18_dims(self):
        cols = get_micro_feature_cols(n_levels=1)
        assert len(cols) == 18
        assert 'slope_asym' not in cols
        assert 'depth_drain' not in cols

    def test_level1_expected_features(self):
        cols = get_micro_feature_cols(n_levels=1)
        expected = [
            'microprice_basis', 'dofi_1', 'dofi_int_1', 'total_obi',
            'dist_bid_1', 'dist_ask_1', 'obi_1',
            'spread_bps', 'dofi_velocity',
            'obi_burst', 'obi_trend', 'dofi_burst', 'microprice_range', 'spread_max',
            'obi_accel', 'dofi_accel', 'microprice_accel', 'spread_velocity',
        ]
        assert cols == expected

    def test_level3_no_slope_asym_has_depth_drain(self):
        cols = get_micro_feature_cols(n_levels=3)
        assert 'slope_asym' not in cols
        assert 'depth_drain' in cols
        # 1 microprice + 3 dofi + 3 dofi_int + 1 total_obi + 3 dist_bid + 3 dist_ask
        # + 3 obi + 1 spread + 1 dofi_vel + 5 cross-TF + 4 accel + 1 depth_drain = 29
        assert len(cols) == 29

    def test_level5_matches_legacy_constant(self):
        """Level-5 dynamic list must match the original MICRO_FEATURE_COLS constant."""
        cols = get_micro_feature_cols(n_levels=5)
        assert cols == list(MICRO_FEATURE_COLS)

    def test_no_duplicate_columns(self):
        for n in [1, 2, 3, 4, 5]:
            cols = get_micro_feature_cols(n_levels=n)
            assert len(cols) == len(set(cols)), f"Duplicates for n_levels={n}"


# ─── Macro Feature Column Tests ────────────────────────────────────

class TestGetMacroFeatureCols:
    """Test get_macro_feature_cols() returns correct lists."""

    def test_crypto_has_funding(self):
        cols = get_macro_feature_cols(asset_class='crypto')
        assert 'funding_sin' in cols
        assert 'funding_cos' in cols
        assert 'dow_sin' not in cols
        assert len(cols) == 15

    def test_cme_has_dow(self):
        cols = get_macro_feature_cols(asset_class='cme_futures')
        assert 'dow_sin' in cols
        assert 'dow_cos' in cols
        assert 'funding_sin' not in cols
        assert len(cols) == 15

    def test_crypto_matches_legacy_constant(self):
        cols = get_macro_feature_cols(asset_class='crypto')
        assert cols == list(MACRO_FEATURE_COLS)

    def test_both_have_15_dims(self):
        assert len(get_macro_feature_cols('crypto')) == 15
        assert len(get_macro_feature_cols('cme_futures')) == 15
        assert len(get_macro_feature_cols('cme')) == 15


# ─── Process Micro Tests ─────────────────────────────────────────────

class TestProcessMicroLevel1:
    """Test process_micro with n_levels=1 (Gold)."""

    def test_produces_18_features(self):
        fe = DeepScalperFeatureEngineer(config={'n_levels': 1})
        df = _make_lob_df(n_rows=300, n_levels=1)
        result = fe.process_micro(df)

        expected_cols = get_micro_feature_cols(n_levels=1)
        for col in expected_cols:
            assert col in result.columns, f"Missing micro feature: {col}"

    def test_no_nan_after_warmup(self):
        fe = DeepScalperFeatureEngineer(config={'n_levels': 1})
        df = _make_lob_df(n_rows=300, n_levels=1)
        result = fe.process_micro(df)

        expected_cols = get_micro_feature_cols(n_levels=1)
        # Check after warmup (row 200+)
        for col in expected_cols:
            nan_count = result[col].iloc[200:].isna().sum()
            assert nan_count == 0, f"NaN in {col} after warmup: {nan_count}"

    def test_features_bounded(self):
        """All micro features after normalization should be in [-1, 1]."""
        fe = DeepScalperFeatureEngineer(config={'n_levels': 1})
        df = _make_lob_df(n_rows=300, n_levels=1)
        fe.process_micro(df)

        expected_cols = get_micro_feature_cols(n_levels=1)
        for col in expected_cols:
            vals = df[col].iloc[200:].values
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                assert vals.min() >= -1.01, f"{col} min={vals.min():.4f} < -1"
                assert vals.max() <= 1.01, f"{col} max={vals.max():.4f} > 1"

    def test_does_not_access_level2_plus(self):
        """Level-1 processing must not try to read bid_price_2, etc."""
        fe = DeepScalperFeatureEngineer(config={'n_levels': 1})
        df = _make_lob_df(n_rows=300, n_levels=1)
        # df only has *_1 columns. If process_micro tries to access *_2, it crashes.
        fe.process_micro(df)  # Should not raise


class TestProcessMicroLevel5Compat:
    """Backward compatibility: Level-5 still works identically."""

    def test_produces_40_features(self):
        fe = DeepScalperFeatureEngineer(config={'n_levels': 5})
        df = _make_lob_df(n_rows=300, n_levels=5)
        fe.process_micro(df)

        for col in MICRO_FEATURE_COLS:
            assert col in df.columns, f"Missing: {col}"

    def test_default_is_level5(self):
        fe = DeepScalperFeatureEngineer()
        assert fe.n_levels == 5

    def test_slope_asym_present(self):
        fe = DeepScalperFeatureEngineer(config={'n_levels': 5})
        df = _make_lob_df(n_rows=300, n_levels=5)
        fe.process_micro(df)
        assert 'slope_asym' in df.columns

    def test_depth_drain_present(self):
        fe = DeepScalperFeatureEngineer(config={'n_levels': 5})
        df = _make_lob_df(n_rows=300, n_levels=5)
        fe.process_micro(df)
        assert 'depth_drain' in df.columns


# ─── Process Macro Tests ─────────────────────────────────────────────

class TestProcessMacroCME:
    """Test process_macro for CME assets."""

    def test_cme_has_dow_encoding(self):
        fe = DeepScalperFeatureEngineer(config={'asset_class': 'cme_futures'})
        df = _make_lob_df(n_rows=300, n_levels=1)
        result = fe.process_macro(df)
        assert 'dow_sin' in result.columns
        assert 'dow_cos' in result.columns
        assert 'funding_sin' not in result.columns

    def test_crypto_has_funding_encoding(self):
        fe = DeepScalperFeatureEngineer(config={'asset_class': 'crypto'})
        df = _make_lob_df(n_rows=300, n_levels=5)
        result = fe.process_macro(df)
        assert 'funding_sin' in result.columns
        assert 'funding_cos' in result.columns
        assert 'dow_sin' not in result.columns

    def test_macro_15_dims_both(self):
        fe_crypto = DeepScalperFeatureEngineer(config={'asset_class': 'crypto'})
        fe_cme = DeepScalperFeatureEngineer(config={'asset_class': 'cme_futures'})

        df_btc = _make_lob_df(n_rows=300, n_levels=5)
        df_gc = _make_lob_df(n_rows=300, n_levels=1)

        result_btc = fe_crypto.process_macro(df_btc)
        result_gc = fe_cme.process_macro(df_gc)

        # Both should have exactly 15 feature columns (+ timestamp if present)
        feat_cols_btc = [c for c in result_btc.columns if c != 'timestamp']
        feat_cols_gc = [c for c in result_gc.columns if c != 'timestamp']
        assert len(feat_cols_btc) == 15
        assert len(feat_cols_gc) == 15


# ─── Feature Engineer Instance Tests ─────────────────────────────────

class TestFeatureEngineerConfig:
    """Test FE instance configuration."""

    def test_default_config(self):
        fe = DeepScalperFeatureEngineer()
        assert fe.n_levels == 5
        assert fe.asset_class == 'crypto'
        assert len(fe.micro_feature_cols) == 40
        assert len(fe.macro_feature_cols) == 15

    def test_gold_config(self):
        fe = DeepScalperFeatureEngineer(config={
            'n_levels': 1,
            'asset_class': 'cme_futures',
        })
        assert fe.n_levels == 1
        assert fe.asset_class == 'cme_futures'
        assert len(fe.micro_feature_cols) == 18
        assert len(fe.macro_feature_cols) == 15
        assert 'dow_sin' in fe.macro_feature_cols
        assert 'funding_sin' not in fe.macro_feature_cols
