"""Classic TradingView-style technical indicators as cross-sectional Signals.

~12 widely-used price/volume indicators (RSI, Bollinger %b, Stochastic, Williams %R, CCI,
ROC, MFI, OBV-momentum, MACD histogram, ATR%, PPO). Each is a causal per-name time series
that the harness ranks cross-sectionally. ``expected_sign`` encodes the conventional
short-horizon read (overbought/overextended → reversal = -1; momentum/flow → +1); the IC
reveals whether that read actually predicts (a negative IC = anti-predictive as labelled).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import Panel
from ..spec import SignalSpec
from . import operators as op


def _ema(x: np.ndarray, span: int, *, alpha: float | None = None) -> np.ndarray:
    df = pd.DataFrame(np.asarray(x, dtype=np.float64))
    ewm = df.ewm(alpha=alpha, min_periods=span) if alpha else df.ewm(span=span, min_periods=span)
    return ewm.mean().to_numpy()


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(a, float) / np.where(np.asarray(b, float) != 0, b, np.nan)


def _rsi(p: Panel, n: int = 14) -> np.ndarray:
    d = op.delta(p.close, 1)
    up, dn = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    rs = _safe_div(_ema(up, n, alpha=1.0 / n), _ema(dn, n, alpha=1.0 / n))   # Wilder smoothing
    return 100.0 - 100.0 / (1.0 + rs)


def _bollinger_pctb(p: Panel, n: int = 20, k: float = 2.0) -> np.ndarray:
    m, s = op.ts_mean(p.close, n), op.stddev(p.close, n)
    return _safe_div(p.close - (m - k * s), 2.0 * k * s)


def _stochastic_k(p: Panel, n: int = 14) -> np.ndarray:
    ll, hh = op.ts_min(p.low, n), op.ts_max(p.high, n)
    return _safe_div(p.close - ll, hh - ll) * 100.0


def _williams_r(p: Panel, n: int = 14) -> np.ndarray:
    ll, hh = op.ts_min(p.low, n), op.ts_max(p.high, n)
    return _safe_div(hh - p.close, hh - ll) * -100.0


def _cci(p: Panel, n: int = 20) -> np.ndarray:
    tp = (p.high + p.low + p.close) / 3.0
    sma = op.ts_mean(tp, n)
    md = op.ts_mean(np.abs(tp - sma), n)
    return _safe_div(tp - sma, 0.015 * md)


def _roc(p: Panel, n: int = 10) -> np.ndarray:
    return _safe_div(p.close, op.delay(p.close, n)) - 1.0


def _mfi(p: Panel, n: int = 14) -> np.ndarray:
    tp = (p.high + p.low + p.close) / 3.0
    rmf = tp * p.volume
    pos = np.where(op.delta(tp, 1) > 0, rmf, 0.0)
    neg = np.where(op.delta(tp, 1) < 0, rmf, 0.0)
    ratio = _safe_div(op.ts_sum(pos, n), op.ts_sum(neg, n))
    return 100.0 - 100.0 / (1.0 + ratio)


def _obv_momentum(p: Panel, n: int = 20) -> np.ndarray:
    obv = np.nancumsum(op.sign(op.delta(p.close, 1)) * p.volume, axis=0)
    return obv - op.delay(obv, n)


def _macd_hist(p: Panel) -> np.ndarray:
    macd = _ema(p.close, 12) - _ema(p.close, 26)
    return macd - _ema(macd, 9)


def _atr_pct(p: Panel, n: int = 14) -> np.ndarray:
    pc = op.delay(p.close, 1)
    tr = np.maximum.reduce([p.high - p.low, np.abs(p.high - pc), np.abs(p.low - pc)])
    return _safe_div(_ema(tr, n, alpha=1.0 / n), p.close)


def _ppo(p: Panel) -> np.ndarray:
    e12, e26 = _ema(p.close, 12), _ema(p.close, 26)
    return _safe_div(e12 - e26, e26) * 100.0


class TVSignal:
    def __init__(self, name: str, fn, expected_sign: int, hypothesis: str) -> None:
        self._fn = fn
        self.spec = SignalSpec(name=name, hypothesis=hypothesis, family="technical",
                               expected_sign=expected_sign)

    def compute(self, panel: Panel) -> np.ndarray:
        out = np.asarray(self._fn(panel), dtype=np.float64)
        return np.where(np.isfinite(out), out, np.nan)


SIGNALS = [
    TVSignal("tv_rsi14", _rsi, -1, "high RSI = overbought → short-horizon reversal"),
    TVSignal("tv_bbpctb20", _bollinger_pctb, -1, "high Bollinger %b → reversal"),
    TVSignal("tv_stoch14", _stochastic_k, -1, "high stochastic %K → reversal"),
    TVSignal("tv_williamsr14", _williams_r, -1, "Williams %R near 0 (overbought) → reversal"),
    TVSignal("tv_cci20", _cci, -1, "high CCI = overextended → reversal"),
    TVSignal("tv_roc10", _roc, 1, "10d rate-of-change → momentum continuation"),
    TVSignal("tv_roc20", lambda p: _roc(p, 20), 1, "20d rate-of-change → momentum"),
    TVSignal("tv_mfi14", _mfi, -1, "high money-flow index = overbought → reversal"),
    TVSignal("tv_obvmom20", _obv_momentum, 1, "on-balance-volume momentum → continuation"),
    TVSignal("tv_macdhist", _macd_hist, 1, "positive MACD histogram → momentum"),
    TVSignal("tv_atrpct14", _atr_pct, -1, "high ATR% (volatility) → low-vol anomaly"),
    TVSignal("tv_ppo", _ppo, 1, "percentage price oscillator → momentum"),
]
