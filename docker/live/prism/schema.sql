-- PRISM Database Schema
-- Version: 3.0

-- 1. Market Data (Bronze Layer)
-- Stores raw OHLCV data.
CREATE TABLE IF NOT EXISTS market_data (
    timestamp TIMESTAMPTZ NOT NULL,
    asset_ticker TEXT NOT NULL,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume FLOAT,
    PRIMARY KEY (timestamp, asset_ticker)
);

-- 2. Features (Silver Layer)
-- Stores engineered features (technicals, stationary series).
CREATE TABLE IF NOT EXISTS features (
    timestamp TIMESTAMPTZ NOT NULL,
    asset_ticker TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    feature_value FLOAT,
    PRIMARY KEY (timestamp, asset_ticker, feature_name)
);

-- 3. Model Performance (Evaluation Layer)
-- Stores CRPS and other metrics for every model at every step.
CREATE TABLE IF NOT EXISTS model_performance (
    timestamp TIMESTAMPTZ NOT NULL,
    experiment_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value FLOAT NOT NULL,
    PRIMARY KEY (timestamp, experiment_id, model_name, metric_name)
);
CREATE INDEX IF NOT EXISTS idx_perf_exp_model ON model_performance (experiment_id, model_name);

-- 4. Arbitrator Weights (Audit Layer)
-- Stores the dynamic weights assigned to experts over time.
CREATE TABLE IF NOT EXISTS arbitrator_weights (
    timestamp TIMESTAMPTZ NOT NULL, 
    experiment_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    weight FLOAT NOT NULL,
    PRIMARY KEY (timestamp, experiment_id, model_name)
);
CREATE INDEX IF NOT EXISTS idx_weight_exp_time ON arbitrator_weights (experiment_id, timestamp);

-- 5. Dual HMM Model Versions (must be created before predictions for FK)
CREATE TABLE IF NOT EXISTS hmm_dual_model_versions (
    version_id          TEXT PRIMARY KEY,
    asset_ticker        TEXT NOT NULL,
    timeframe           TEXT NOT NULL DEFAULT 'daily',
    model_type          TEXT NOT NULL,           -- 'price' or 'vol'
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    training_start      TIMESTAMPTZ,
    training_end        TIMESTAMPTZ,
    n_training_samples  INT,
    model_blob          BYTEA NOT NULL,
    hyperparameters     JSONB,
    is_active           BOOLEAN DEFAULT TRUE,
    UNIQUE (asset_ticker, timeframe, model_type, created_at)
);
CREATE INDEX IF NOT EXISTS idx_dual_model_active
    ON hmm_dual_model_versions (asset_ticker, timeframe, model_type, is_active)
    WHERE is_active = TRUE;

-- 6. Dual HMM Regime Predictions (3x3 composite: price x vol)
CREATE TABLE IF NOT EXISTS hmm_dual_regime_predictions (
    timestamp           TIMESTAMPTZ NOT NULL,
    asset_ticker        TEXT NOT NULL,
    timeframe           TEXT NOT NULL DEFAULT 'daily',
    price_regime        INT NOT NULL,           -- PriceRegime: 0=BEARISH, 1=NEUTRAL, 2=BULLISH
    vol_regime          INT NOT NULL,           -- VolRegime: 0=LOW_VOL, 1=NORMAL_VOL, 2=HIGH_VOL
    composite_code      INT NOT NULL,           -- price_regime * 3 + vol_regime (0..8)
    prob_bearish        FLOAT NOT NULL,
    prob_neutral        FLOAT NOT NULL,
    prob_bullish        FLOAT NOT NULL,
    prob_low_vol        FLOAT NOT NULL,
    prob_normal_vol     FLOAT NOT NULL,
    prob_high_vol       FLOAT NOT NULL,
    confidence          FLOAT,
    price_model_version TEXT NOT NULL REFERENCES hmm_dual_model_versions(version_id),
    vol_model_version   TEXT NOT NULL REFERENCES hmm_dual_model_versions(version_id),
    PRIMARY KEY (timestamp, asset_ticker, timeframe)
);
CREATE INDEX IF NOT EXISTS idx_dual_regime_ticker_time
    ON hmm_dual_regime_predictions (asset_ticker, timestamp DESC);
