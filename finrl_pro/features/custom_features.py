"""PIT-safe custom feature builders for FinRL Pro.

This module complements the upstream FinRLPodracer feature engineering by
adding advanced features (e.g., fractional differentiation, wavelets) in a
Point-In-Time safe manner. All functions operate per ticker and ensure that
the value at time t uses only information available at or before t-1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


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
    if extra_transforms:
        for fn in extra_transforms:
            out = fn(out)
    return out

