"""Cross-sectional COUNTRY-EQUITY momentum (C1) + negative control (C2) through the locked funnel.

Pre-registration: `docs/research/country_momentum_preregistration_2026-07-31.md` (committed with an
empty Results section BEFORE this ran).

Builds a Panel from free yfinance country-ETF closes and scores both signals with the SAME
`finrl_pro_ds/signals/` scorecard used for every Taiwan probe — no statistic re-implemented, so the
results are directly comparable to P1/R1/R2/S1.
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

from finrl_pro_ds.signals.features import Panel  # noqa: E402
from finrl_pro_ds.signals.multiplicity import Multiplicity  # noqa: E402
from finrl_pro_ds.signals.scorecard import (  # noqa: E402
    Gates,
    evaluate_batch,
    to_markdown,
    write_scorecard,
)
from finrl_pro_ds.signals.spec import SignalSpec  # noqa: E402

log = logging.getLogger("country_momentum")

# US-listed single-country equity ETFs. Fixed list, chosen for coverage breadth BEFORE any result
# was seen (pre-registration §1); no ticker is added or dropped after the fact.
TICKERS = ["EWA", "EWC", "EWD", "EWG", "EWH", "EWI", "EWJ", "EWK", "EWL", "EWM", "EWN", "EWO",
           "EWP", "EWQ", "EWS", "EWU", "EWW", "EWY", "EWZ", "EZA", "EWT", "EPP", "THD", "TUR",
           "EPOL", "ECH", "EIDO", "EPHE", "INDA", "FXI"]

_HZ = (1, 5, 10, 21, 63)
_NEU = ("winsor", "zscore", "size")     # sector omitted on purpose — all names are country ETFs
_UNI = "intl_country_equity_etf"
_CP = "etf_standard"

MOM_LOOKBACK = 252      # pre-registered 12 months
MOM_SKIP = 21           # pre-registered 1-month skip
MIN_NAMES = 8           # a cross-section this size is the floor for a rank IC


class _XSMomentum:
    """C1: 12-1 momentum. Row t uses close[t-skip] and close[t-lookback], BOTH <= t-skip."""

    def __init__(self, name: str, hypothesis: str, sign: int, lookback: int, skip: int) -> None:
        self.lookback, self.skip = lookback, skip
        self.spec = SignalSpec(name=name, hypothesis=hypothesis, family="technical",
                               expected_sign=sign, horizons=_HZ, neutralization=_NEU,
                               universe=_UNI, cost_profile=_CP)

    def compute(self, panel: Panel) -> np.ndarray:
        c = np.asarray(panel.close, dtype=np.float64)
        t = c.shape[0]
        out = np.full(c.shape, np.nan, dtype=np.float64)
        lb, sk = self.lookback, self.skip
        if lb >= t:
            return out
        # out[i] = c[i-sk] / c[i-lb] - 1  for i >= lb. Both operands are <= i-sk, so the most
        # recent month is skipped entirely (standard 12-1) and nothing at or after i is read.
        num = c[lb - sk: t - sk]          # rows i-sk for i = lb .. t-1
        den = c[0: t - lb]                # rows i-lb for i = lb .. t-1
        with np.errstate(invalid="ignore", divide="ignore"):
            out[lb:] = np.where(den > 0, num / den - 1.0, np.nan)
        return out


class _NullControl:
    """C2: deterministic pseudo-random score with NO price information.

    Seeded from (row index, column index) only, so it is identical on every run and cannot
    accidentally encode anything about returns. If this scores 'significant', the harness is
    manufacturing significance on this substrate and C1 must be discarded (pre-reg §1).
    """

    def __init__(self, name: str, hypothesis: str, sign: int) -> None:
        self.spec = SignalSpec(name=name, hypothesis=hypothesis, family="technical",
                               expected_sign=sign, horizons=_HZ, neutralization=_NEU,
                               universe=_UNI, cost_profile=_CP)

    def compute(self, panel: Panel) -> np.ndarray:
        t, n = panel.close.shape
        rng = np.random.default_rng(20260731)
        v = rng.standard_normal((t, n))
        # only defined where the price panel is defined, so coverage matches C1's
        return np.where(np.isfinite(panel.close), v, np.nan)


def build_panel(start: str, end: str, adv_window: int = 20) -> Panel:
    import yfinance as yf
    raw = yf.download(TICKERS, start=start, end=end, progress=False, auto_adjust=True)
    close = raw["Close"].copy().sort_index()
    vol = raw["Volume"].reindex_like(close)
    high = raw["High"].reindex_like(close)
    low = raw["Low"].reindex_like(close)
    opn = raw["Open"].reindex_like(close)
    keep = [t for t in TICKERS if t in close.columns and close[t].notna().sum() >= 750]
    close, vol, high, low, opn = (d[keep] for d in (close, vol, high, low, opn))
    close = close.dropna(how="all")
    vol, high, low, opn = (d.reindex(close.index) for d in (vol, high, low, opn))

    dates = close.index.to_numpy(dtype="datetime64[ns]")
    tickers = tuple(close.columns)
    C = close.to_numpy(dtype=np.float64)
    V = vol.to_numpy(dtype=np.float64)
    dollar = C * V
    adv = pd.DataFrame(dollar).rolling(adv_window, min_periods=5).mean().shift(1).to_numpy()
    active = np.isfinite(C) & np.isfinite(adv) & (adv > 0)
    # a day is usable only if the cross-section is wide enough for a rank IC
    wide = active.sum(axis=1) >= MIN_NAMES
    active &= wide[:, None]
    return Panel(dates=dates, tickers=tickers,
                 open=opn.to_numpy(dtype=np.float64), high=high.to_numpy(dtype=np.float64),
                 low=low.to_numpy(dtype=np.float64), close=C, volume=V,
                 active=active, adv_usd=adv,
                 sector_id=np.zeros(len(tickers), dtype=int),
                 meta={"survivorship_free": False, "source": "yfinance",
                       "universe_def": _UNI, "return_basis": "adjusted_close",
                       "usable_days": int(wide.sum())})


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Cross-sectional country momentum C1 + control C2")
    ap.add_argument("--start", default="1996-01-01")
    ap.add_argument("--end", default="2026-07-31")
    ap.add_argument("--gates", default="configs/country_momentum.gates.yaml")
    ap.add_argument("--out", default="results/country_momentum")
    args = ap.parse_args()

    gates_path = (ROOT / args.gates) if not Path(args.gates).is_absolute() else Path(args.gates)
    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)

    panel = build_panel(args.start, args.end)
    log.info("Panel N=%d T=%d (%s..%s) | usable days (>=%d names)=%d",
             panel.N, panel.T, str(panel.dates[0])[:10], str(panel.dates[-1])[:10],
             MIN_NAMES, panel.meta["usable_days"])

    c1 = _XSMomentum(
        "ctry_xs_momentum",
        "12-1 cross-sectional momentum across US-listed single-country equity ETFs: country-level "
        "information diffuses slowly across borders, producing cross-sectional continuation",
        1, MOM_LOOKBACK, MOM_SKIP)
    c2 = _NullControl(
        "ctry_null_control",
        "NEGATIVE CONTROL — deterministic pseudo-random score with no price information; must "
        "score IC ~ 0. A 'significant' result here invalidates the run rather than finding an edge",
        1)

    for s in (c1, c2):
        v = s.compute(panel)
        cov = 100.0 * np.isfinite(v)[panel.active].mean() if panel.active.any() else 0.0
        log.info("  signal %-20s coverage %.1f%% of active cells", s.spec.name, cov)

    gates = Gates.from_yaml(gates_path)
    mult = Multiplicity.preregistered(
        2, substrate="intl_country_etf",
        provenance="docs/research/country_momentum_preregistration_2026-07-31.md",
        gates=None)
    rs = evaluate_batch({s.spec.name: s for s in (c1, c2)}, panel, gates, "country_momentum",
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
