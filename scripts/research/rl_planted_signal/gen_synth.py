"""Synthetic 1-min OHLCV whose 15-min log returns follow an AR(1) with coefficient rho.

The planted signal is observable through the real ``MultiScaleOHLCVHandler`` with no custom
feature column: feature 0 of the base scale is ``log_return``, so if the 15-min returns follow
``u_k = rho * u_{k-1} + eta_k`` the per-bar predictive IC of the last observed return for the
return the action earns is exactly ``rho``. The optimal policy is ``position ~ clip(u_{k-1}/sigma)``.

Each 15-min move is split into 15 equal 1-min increments plus zero-sum jitter, so the 15-min close
reproduces ``u`` exactly while open/high/low look like a real bar. Timestamps run 24/7.

Usage: python gen_synth.py <rho> <out.parquet> [n_base_bars] [seed]
"""
from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

log = logging.getLogger("rl_planted_signal.gen_synth")


def make(rho: float, n_base: int, seed: int, sigma: float = 0.0015) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    eta = rng.normal(0.0, sigma * np.sqrt(max(1e-12, 1.0 - rho * rho)), size=n_base)
    u = np.empty(n_base)
    u[0] = rng.normal(0.0, sigma)
    for k in range(1, n_base):
        u[k] = rho * u[k - 1] + eta[k]

    m = 15
    inc = np.repeat(u / m, m)
    jit = rng.normal(0.0, sigma / (3.0 * np.sqrt(m)), size=n_base * m)
    jit = jit - np.repeat(jit.reshape(n_base, m).mean(axis=1), m)    # zero-sum per 15-min bar
    close = np.exp(np.cumsum(inc + jit) + np.log(100.0))
    open_ = np.concatenate([[100.0], close[:-1]])
    wig = np.abs(rng.normal(0.0, sigma / (2 * np.sqrt(m)), size=n_base * m))
    high = np.maximum(open_, close) * (1.0 + wig)
    low = np.minimum(open_, close) * (1.0 - wig)
    vol = rng.lognormal(mean=6.0, sigma=0.3, size=n_base * m)
    ts = pd.date_range("2018-01-01 00:00:00", periods=n_base * m, freq="1min")
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low,
                         "close": close, "volume": vol})


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    rho, out = float(sys.argv[1]), sys.argv[2]
    n_base = int(sys.argv[3]) if len(sys.argv) > 3 else 200_000
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 11
    df = make(rho, n_base, seed)
    df.to_parquet(out, index=False)
    c = df.set_index("timestamp")["close"].resample("15min", label="left", closed="left").last()
    r = np.diff(np.log(c.to_numpy()))
    log.info("rho_target=%s rho_realised=%.4f n_base=%d rows=%d sigma_15m=%.5f -> %s",
             rho, np.corrcoef(r[:-1], r[1:])[0, 1], n_base, len(df), r.std(), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
