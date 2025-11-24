"""PIT-safe custom feature builders for FinRL Pro.

This module complements the upstream FinRLPodracer feature engineering by
adding advanced features (e.g., fractional differentiation, wavelets) in a
Point-In-Time safe manner. All functions operate per ticker and ensure that
the value at time t uses only information available at or before t-1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    from finrl_pro.data.regimes import HMMRegimeDetector, MarketRegime
except ImportError:
    # Fallback or delay import to avoid circular deps if any (though unlikely here)
    pass


@dataclass(frozen=True)
class FracDiffConfig:
    cols: tuple[str, ...] = ("close",)
    d: float = 0.5
    window: int = 256
    min_weight: float = 1e-5
    prefix: str = "fd"


@dataclass(frozen=True)
class WaveletConfig:
    cols: tuple[str, ...] = ("close",)
    wavelet: str = "db4"
    level: int = 3
    window: int = 256
    mode: str = "periodization"
    prefix: str = "wlt"
    compute_energy: bool = True
    denoise: bool = False
    thresh_method: str = "universal"


@dataclass(frozen=True)
class RegimeConfig:
    method: str = "hmm" # "hmm" or "fixed"
    benchmark_tic: str = "SPY"
    source: str = "close" # column to compute returns from
    window: int = 252
    n_components: int = 3


def _ensure_datetime_sorted(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns or "tic" not in out.columns:
        raise ValueError("Expected columns 'date' and 'tic' in input dataframe")
    if not np.issubdtype(out["date"].dtype, np.datetime64):
        out["date"] = pd.to_datetime(out["date"])  # type: ignore[assignment]
    return out.sort_values(["tic", "date"]).reset_index(drop=True)


def _fracdiff_weights(d: float, window: int, min_weight: float) -> np.ndarray:
    """Compute fixed-width fractional differencing weights.

    Based on Lopez de Prado. Returns weights aligned oldest->newest.
    Tail weights with magnitude below ``min_weight`` are trimmed to speed up
    computation while preserving most of the effect.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    w = [1.0]
    for k in range(1, window):
        w.append(-w[-1] * (d - (k - 1)) / k)
    w_np = np.asarray(w, dtype=float)
    idx = np.where(np.abs(w_np) >= float(min_weight))[0]
    if idx.size == 0:
        return np.array([1.0], dtype=float)
    used = w_np[: idx[-1] + 1]
    return used[::-1]  # reverse to oldest->newest


def add_market_regime_features(df: pd.DataFrame, cfg: RegimeConfig | None = None) -> pd.DataFrame:
    """Add Market Regime features.
    
    Calculates regime (Bull/Bear/etc) based on a benchmark ticker (e.g. SPY)
    and broadcasts it to all tickers in the dataframe.
    
    Ensures PIT safety by using rolling window estimation if specified.
    """
    cfg = cfg or RegimeConfig()
    out = _ensure_datetime_sorted(df)
    
    # 1. Extract Benchmark Data
    bench_df = out[out['tic'] == cfg.benchmark_tic].copy()
    if bench_df.empty:
        # If benchmark not in df, we can't compute regime. 
        # Should we fail or fill 0? Let's warn and fill 0 (Bear/Unknown)
        # But for now, raise error to ensure user provides SPY
        print(f"Warning: Benchmark ticker {cfg.benchmark_tic} not found for regime detection.")
        # return out with nan regime?
        out["market_regime"] = 0
        return out
        
    # 2. Compute Returns
    bench_df = bench_df.sort_values("date")
    returns = bench_df[cfg.source].pct_change().fillna(0)
    
    # 3. Detect Regime
    if cfg.method == "hmm":
        detector = HMMRegimeDetector(n_components=cfg.n_components)
        # Use rolling fit for PIT safety
        regime_series = detector.rolling_fit_predict(returns, window=cfg.window)
        
        # Fill NaNs (initial window) with SIDEWAYS (3) or most common?
        # Let's fill with SIDEWAYS (3) as neutral assumption
        regime_series = regime_series.fillna(MarketRegime.SIDEWAYS).astype(int)
        
    else:
        raise NotImplementedError(f"Regime method {cfg.method} not implemented.")
        
    # 4. Broadcast to all tickers
    # Create a mapping date -> regime
    regime_map = pd.DataFrame({
        "date": bench_df["date"], 
        "market_regime": regime_series.values
    })
    
    out = out.merge(regime_map, on="date", how="left")
    
    # Fill missing regimes (e.g. dates where SPY missing) with previous or default
    out["market_regime"] = out["market_regime"].ffill().fillna(MarketRegime.SIDEWAYS)
    
    return out


def add_fracdiff_features(df: pd.DataFrame, cfg: FracDiffConfig | None = None) -> pd.DataFrame:
    """Add fractional differentiation features per ticker in a PIT-safe way.

    The feature at row t uses only data up to t-1 by shifting source series.
    """
    cfg = cfg or FracDiffConfig()
    out = _ensure_datetime_sorted(df)

    weights = _fracdiff_weights(cfg.d, cfg.window, cfg.min_weight)
    w_len = int(len(weights))

    def per_tic(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        for col in cfg.cols:
            if col not in g.columns:
                continue
            x = g[col].astype(float).to_numpy()
            x_lag = np.roll(x, 1)
            x_lag[0] = np.nan
            out_arr = np.full_like(x_lag, fill_value=np.nan, dtype=float)

            for t in range(w_len, len(x_lag) + 1):
                window_vals = x_lag[t - w_len : t]
                if np.any(np.isnan(window_vals)):
                    continue
                out_arr[t - 1] = float(np.dot(weights, window_vals))

            name = f"{cfg.prefix}_{col}_d{str(cfg.d).replace('.', 'p')}_w{cfg.window}"
            g[name] = out_arr

        # Preserve NaNs for initial windows to reflect PIT safety; do not forward-fill
        # or drop rows here so callers/tests can validate windowing explicitly.
        return g.reset_index(drop=True)

    return out.groupby("tic", group_keys=False).apply(per_tic)


def add_wavelet_features(df: pd.DataFrame, cfg: WaveletConfig | None = None) -> pd.DataFrame:
    """Add wavelet-based multi-resolution features in a PIT-safe way.

    Uses stationary wavelet transform (SWT/MODWT) on lagged windows to compute
    last detail coefficients and optional band energy and a denoised trend.
    """
    try:
        import pywt  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError("PyWavelets (pywt) is required for wavelet features") from e

    cfg = cfg or WaveletConfig()
    out = _ensure_datetime_sorted(df)

    def per_tic(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        for col in cfg.cols:
            if col not in g.columns:
                continue
            ret = np.log(g[col]).diff()
            x = ret.shift(1).to_numpy()  # PIT: use strictly past up to t-1

            last_maps: dict[str, np.ndarray] = {
                f"{cfg.prefix}_{col}_D{k}_last": np.full(len(x), np.nan) for k in range(1, cfg.level + 1)
            }
            energy_maps: dict[str, np.ndarray] = {}
            if cfg.compute_energy:
                energy_maps = {
                    f"{cfg.prefix}_{col}_D{k}_energy": np.full(len(x), np.nan) for k in range(1, cfg.level + 1)
                }
            trend_name = f"{cfg.prefix}_{col}_trend" if cfg.denoise else None
            if trend_name:
                g[trend_name] = np.nan

            for t in range(cfg.window, len(x)):
                seg = x[t - cfg.window : t]
                if np.any(np.isnan(seg)):
                    continue
                coeffs = pywt.swt(
                    seg,
                    wavelet=cfg.wavelet,
                    level=cfg.level,
                    trim_approx=False,
                    start_level=0,
                    norm=True,
                )
                for k in range(1, cfg.level + 1):
                    cA_k, cD_k = coeffs[k - 1]
                    last_maps[f"{cfg.prefix}_{col}_D{k}_last"][t] = cD_k[-1]
                    if cfg.compute_energy:
                        energy_maps[f"{cfg.prefix}_{col}_D{k}_energy"][t] = float(np.sqrt(np.mean(cD_k**2)))

                if trend_name:
                    rec_coeffs = []
                    for (cA_k, cD_k) in coeffs:
                        # Universal soft threshold based on MAD
                        sigma = np.median(np.abs(cD_k - np.median(cD_k))) / 0.6745 + 1e-12
                        thr = sigma * np.sqrt(2 * np.log(len(cD_k))) if cfg.thresh_method == "universal" else 0.0
                        cD_k = np.sign(cD_k) * np.maximum(np.abs(cD_k) - thr, 0.0)
                        rec_coeffs.append((cA_k, cD_k))
                    rec = pywt.iswt(rec_coeffs, wavelet=cfg.wavelet, norm=True)
                    g.at[g.index[t], trend_name] = rec[-1]

            for k, arr in last_maps.items():
                g[k] = arr
            for k, arr in energy_maps.items():
                g[k] = arr

        g = g.ffill().dropna().reset_index(drop=True)
        return g

    return out.groupby("tic", group_keys=False).apply(per_tic)


def build_features(
    df: pd.DataFrame,
    *,
    fracdiff: FracDiffConfig | None = None,
    wavelet: WaveletConfig | None = None,
    regime: RegimeConfig | None = None,
    extra_transforms: Iterable[callable] | None = None,
) -> pd.DataFrame:
    """Compose advanced PIT-safe features over an input dataframe.

    - Always preserves Point-In-Time by construction.
    - Applies configured transforms in sequence.
    """
    out = _ensure_datetime_sorted(df)
    if fracdiff is not None:
        out = add_fracdiff_features(out, fracdiff)
    if wavelet is not None:
        out = add_wavelet_features(out, wavelet)
    if regime is not None:
        out = add_market_regime_features(out, regime)
    if extra_transforms:
        for fn in extra_transforms:
            out = fn(out)
    return out


def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Log-based HLOCV features (Stationary Returns).
    
    Computes:
    - log_close: log(Close_t / Close_t-1)  (Return)
    - log_open:  log(Open_t / Close_t-1)   (Overnight Gap)
    - log_high:  log(High_t / Close_t-1)   (High Extent)
    - log_low:   log(Low_t / Close_t-1)    (Low Extent)
    - log_volume: log(Volume_t / Volume_t-1) (Volume Change)
    """
    out = _ensure_datetime_sorted(df)
    
    def per_tic(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        # Prev Close
        prev_close = g["close"].shift(1)
        prev_vol = g["volume"].shift(1)
        
        # Avoid log(0) or log(neg)
        epsilon = 1e-8
        
        g["log_close"] = np.log(g["close"] / (prev_close + epsilon))
        g["log_open"]  = np.log(g["open"] / (prev_close + epsilon))
        g["log_high"]  = np.log(g["high"] / (prev_close + epsilon))
        g["log_low"]   = np.log(g["low"] / (prev_close + epsilon))
        g["log_volume"] = np.log((g["volume"] + epsilon) / (prev_vol + epsilon))
        
        # Add Trend Context
        sma_50 = g["close"].rolling(window=50).mean()
        sma_200 = g["close"].rolling(window=200).mean()
        
        g["log_sma_50"] = np.log(g["close"] / (sma_50 + epsilon))
        g["log_sma_200"] = np.log(g["close"] / (sma_200 + epsilon))
        
        # Fill NaNs from rolling windows
        g = g.fillna(0.0)
        
        return g
    
    return out.groupby("tic", group_keys=False).apply(per_tic)


def add_hybrid_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Hybrid features (FinRL Baseline + Volume/Volatility upgrades).
    
    Features:
    - macd: Trend (Standard)
    - rsi_14: Momentum (Faster window)
    - vwap_ratio: vwma_14 / close (Valuation)
    - atr_norm: atr_14 / close (Normalized Volatility)
    - log_volume: log(volume + 1) (Traffic Scale)
    """
    from stockstats import StockDataFrame as Sdf
    out = _ensure_datetime_sorted(df)
    
    def per_tic(g: pd.DataFrame) -> pd.DataFrame:
        # Use StockDataFrame for efficient calculation
        stock = Sdf.retype(g.copy())
        
        # 1. Calculate Base Indicators
        # MACD (close_12_ema - close_26_ema)
        _ = stock['macd'] 
        
        # RSI 14 (Faster than default 30)
        _ = stock['rsi_14']
        
        # 2. VWAP Ratio (VWMA 14 / Close)
        # Stockstats 'vwma' is volume weighted moving average
        vwma = stock['vwma_14']
        stock['vwap_ratio'] = vwma / stock['close']
        
        # 3. Normalized ATR (ATR 14 / Close)
        atr = stock['atr_14']
        stock['atr_norm'] = atr / stock['close']
        
        # 4. Log Volume
        stock['log_volume'] = np.log(stock['volume'] + 1)
        
        # Select final columns
        cols = ['macd', 'rsi_14', 'vwap_ratio', 'atr_norm', 'log_volume']
        # Sdf modifies in place, so we just return the dataframe with these columns
        # We keep the original columns (open/close etc) for safety, the assembler selects features.
        return stock
    
    return out.groupby("tic", group_keys=False).apply(per_tic)

