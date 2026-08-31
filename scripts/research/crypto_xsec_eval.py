"""Crypto cross-sectional probes (K1 reversal, K2 momentum) + negative control (K3).

Pre-registration: `docs/research/crypto_xsec_preregistration_2026-07-31.md` (committed with an empty
Results section BEFORE this ran). Same locked `sharpen/signals/` scorecard as every other probe
today, so results are directly comparable.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sharpen.signals.features import Panel  # noqa: E402
from sharpen.signals.multiplicity import Multiplicity  # noqa: E402
from sharpen.signals.scorecard import (  # noqa: E402
    Gates,
    evaluate_batch,
    to_markdown,
    write_scorecard,
)
from sharpen.signals.spec import SignalSpec  # noqa: E402

log = logging.getLogger("crypto_xsec")

_HZ = (1, 5, 10, 21, 63)
_NEU = ("winsor", "zscore")        # no size control: N=10 makes a size regression degenerate
_UNI = "crypto_perp_10"
_CP = "crypto_standard"
REV_LB = 7                          # pre-registered
MOM_LB = 30                         # pre-registered


class _PastReturn:
    """Causal trailing simple return over `lookback` days. Row t uses close[t] and close[t-lb]."""

    def __init__(self, name: str, hypothesis: str, sign: int, lookback: int) -> None:
        self.lookback = lookback
        self.spec = SignalSpec(name=name, hypothesis=hypothesis, family="technical",
                               expected_sign=sign, horizons=_HZ, neutralization=_NEU,
                               universe=_UNI, cost_profile=_CP)

    def compute(self, panel: Panel) -> np.ndarray:
        c = np.asarray(panel.close, dtype=np.float64)
        out = np.full(c.shape, np.nan, dtype=np.float64)
        lb = self.lookback
        if lb < c.shape[0]:
            with np.errstate(invalid="ignore", divide="ignore"):
                out[lb:] = np.where(c[:-lb] > 0, c[lb:] / c[:-lb] - 1.0, np.nan)
        return out


class _NullControl:
    """K3: deterministic pseudo-random score, no price information (validated instrument)."""

    def __init__(self, name: str, hypothesis: str, sign: int) -> None:
        self.spec = SignalSpec(name=name, hypothesis=hypothesis, family="technical",
                               expected_sign=sign, horizons=_HZ, neutralization=_NEU,
                               universe=_UNI, cost_profile=_CP)

    def compute(self, panel: Panel) -> np.ndarray:
        t, n = panel.close.shape
        v = np.random.default_rng(20260731).standard_normal((t, n))
        return np.where(np.isfinite(panel.close), v, np.nan)


def build_panel(path: Path, adv_window: int = 20) -> Panel:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["day"] = df["timestamp"].dt.tz_convert("UTC").dt.normalize()
    # hourly -> daily OHLCV (last close, summed volume) per ticker
    agg = df.groupby(["day", "ticker"]).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum")).reset_index()
    piv = {k: agg.pivot(index="day", columns="ticker", values=k).sort_index()
           for k in ("open", "high", "low", "close", "volume")}
    close = piv["close"]
    tickers = tuple(close.columns)
    dates = close.index.to_numpy(dtype="datetime64[ns]")
    C = close.to_numpy(dtype=np.float64)
    V = piv["volume"].to_numpy(dtype=np.float64)
    adv = pd.DataFrame(C * V).rolling(adv_window, min_periods=5).mean().shift(1).to_numpy()
    active = np.isfinite(C) & np.isfinite(adv) & (adv > 0)
    wide = active.sum(axis=1) >= 6          # need a usable cross-section
    active &= wide[:, None]
    return Panel(dates=dates, tickers=tickers,
                 open=piv["open"].to_numpy(dtype=np.float64),
                 high=piv["high"].to_numpy(dtype=np.float64),
                 low=piv["low"].to_numpy(dtype=np.float64),
                 close=C, volume=V, active=active, adv_usd=adv,
                 sector_id=np.zeros(len(tickers), dtype=int),
                 meta={"survivorship_free": False, "source": "crypto_cache/silver_ohlcv",
                       "universe_def": _UNI, "return_basis": "perp_close",
                       "usable_days": int(wide.sum())})


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Crypto cross-sectional probes K1/K2 + control K3")
    ap.add_argument("--data", default="data/crypto_cache/silver_ohlcv.parquet")
    ap.add_argument("--gates", default="configs/crypto_xsec.gates.yaml")
    ap.add_argument("--out", default="results/crypto_xsec")
    args = ap.parse_args()

    data = (ROOT / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    gates_path = (ROOT / args.gates) if not Path(args.gates).is_absolute() else Path(args.gates)
    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)

    panel = build_panel(data)
    log.info("Panel N=%d T=%d (%s..%s) usable days=%d tickers=%s",
             panel.N, panel.T, str(panel.dates[0])[:10], str(panel.dates[-1])[:10],
             panel.meta["usable_days"], ",".join(panel.tickers))

    k1 = _PastReturn("crypto_xs_reversal",
                     "7-day cross-sectional REVERSAL across crypto perps: liquidity provision to "
                     "uninformed retail flow on a venue where ~5bp costs cannot kill it",
                     -1, REV_LB)
    k2 = _PastReturn("crypto_xs_momentum",
                     "30-day cross-sectional MOMENTUM across crypto perps (documented crypto "
                     "cross-sectional continuation)",
                     1, MOM_LB)
    k3 = _NullControl("crypto_null_control",
                      "NEGATIVE CONTROL — no price information; must score IC ~ 0. A significant "
                      "result invalidates the run rather than finding an edge", 1)

    for s in (k1, k2, k3):
        v = s.compute(panel)
        cov = 100.0 * np.isfinite(v)[panel.active].mean() if panel.active.any() else 0.0
        log.info("  signal %-22s coverage %.1f%%", s.spec.name, cov)

    gates = Gates.from_yaml(gates_path)
    mult = Multiplicity.preregistered(
        3, substrate="crypto_perp",
        provenance="docs/research/crypto_xsec_preregistration_2026-07-31.md", gates=None)
    rs = evaluate_batch({s.spec.name: s for s in (k1, k2, k3)}, panel, gates, "crypto_xsec",
                        multiplicity=mult)
    jp, mp = write_scorecard(rs, out_dir)
    try:
        print("\n" + to_markdown(rs) + "\n")
    except UnicodeEncodeError:
        log.warning("console cannot encode markdown — read %s", mp)
    log.info("Scorecard: %s | %s", mp, jp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
