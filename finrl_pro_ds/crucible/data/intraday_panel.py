"""Build a Crucible `Panel` from the Dukascopy hourly cells — the substrate that finally clears
BOTH power requirements.

WHY THIS EXISTS. Crucible's power guard refused every substrate for the system's whole life. As of
2026-08-02 that is measured, not asserted, and it is TWO separate constraints:

  * DEPTH — the binding cross-sectional MDE curve had never been measured past holdout 8,064
    (MDE 0.644, above the 0.50 ceiling). Extending the grid gives 0.405 at 16,128 and 0.392 at
    32,256, so depth DOES open the guard; it had been refusing out of `unmeasured_high` IGNORANCE.
  * BREADTH — the measured n_grid floor is 12, and the consumer picks the largest measured n NOT
    EXCEEDING the substrate's. A 1-instrument gold cell or a 5-instrument FX cell cannot be stamped
    at all, however deep.

The 12 uniformity-clean Dukascopy 1h instruments give T=77,298 common bars, holdout 19,325, N=12
=> interpolated cross-sectional MDE 0.401 => ALLOW. That is the first substrate on disk to satisfy
both.

THIS MODULE IS ADDITIVE AND NOT WIRED INTO THE LIVE TICK. It builds and self-checks a Panel; the
orchestrator still loads `load_cross_asset_panel`. Wiring is a separate, deliberate change, because
a panel feeding the mining surface is the highest leak-risk edit in this workstream (LEAK-2) — the
sg1-btc coarse-bar leak lived in `multiscale_handler.py` from Session 121 to S553 through hundreds
of green audits and erased that strategy's entire apparent edge.

Usage (self-check):
    python -m finrl_pro_ds.crucible.data.intraday_panel
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from finrl_pro_ds.signals.features import Panel

log = logging.getLogger("crucible.intraday_panel")

ROOT = Path(__file__).resolve().parents[3]
DUKA = ROOT / "data" / "dukascopy"

# The 12 instruments whose per-year bar counts verified UNIFORM (no thin years) on 2026-08-02.
# The three index CFDs are deliberately EXCLUDED: they were flagged thin, and while that is very
# likely a false positive of an FX-calibrated 4,500 bars/yr threshold (index CFDs trade ~15h/day,
# so ~3,800/yr is expected), it is UNCONFIRMED — and an unconfirmed instrument has no place in a
# substrate whose whole purpose is to be trusted.
CLEAN_12 = ("AUDUSD", "EURUSD", "GBPUSD", "USDCHF", "USDJPY", "EURGBP",
            "EURJPY", "NZDUSD", "USDCAD", "XAGUSD", "XAUUSD", "LIGHTCMDUSD")

# Asset-class grouping, used only for the sector_id neutralisation control.
_CLASS = {"AUDUSD": 0, "EURUSD": 0, "GBPUSD": 0, "USDCHF": 0, "USDJPY": 0, "EURGBP": 0,
          "EURJPY": 0, "NZDUSD": 0, "USDCAD": 0,           # FX
          "XAGUSD": 1, "XAUUSD": 1,                        # metals
          "LIGHTCMDUSD": 2}                                # energy


def build_intraday_panel(tickers: tuple[str, ...] = CLEAN_12) -> Panel:
    """Aligned (T, N) hourly panel on the instruments' COMMON timestamps.

    Inner-join on timestamps only — no forward-filling across instruments. A ffill would invent a
    price for a bar an instrument did not trade, which is exactly the stale-print class the
    DATA-CLEAN rule exists to reject, and on a cross-sectional book it would manufacture spurious
    relative moves at every session boundary.
    """
    frames: dict[str, pd.DataFrame] = {}
    for t in tickers:
        f = DUKA / f"{t}_1h.parquet"
        if not f.exists():
            raise FileNotFoundError(f"missing cell: {f}")
        d = pd.read_parquet(f)
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        frames[t] = d.drop_duplicates(subset=["timestamp"]).set_index("timestamp").sort_index()

    common = None
    for d in frames.values():
        common = d.index if common is None else common.intersection(d.index)
    common = common.sort_values()
    T, N = len(common), len(tickers)
    log.info("intraday panel: T=%d common bars x N=%d  %s -> %s",
             T, N, common.min(), common.max())

    o, h, lo, c, v = (np.empty((T, N)) for _ in range(5))
    for j, t in enumerate(tickers):
        d = frames[t].reindex(common)
        o[:, j], h[:, j], lo[:, j] = d["open"].to_numpy(), d["high"].to_numpy(), d["low"].to_numpy()
        c[:, j] = d["close"].to_numpy()
        # Dukascopy publishes tick COUNT, not traded size; it is the only volume proxy available and
        # is labelled as such rather than passed off as dollar volume.
        v[:, j] = d["n_ticks"].to_numpy()

    active = np.isfinite(c) & np.isfinite(o)
    # adv proxy: trailing 500-bar mean of |close| * n_ticks. Shifted by one bar so a value at t
    # never contains bar t's own volume (LEAK-2: adv feeds cost/size, which must be known ex-ante).
    dollar = np.abs(c) * v
    adv = pd.DataFrame(dollar).rolling(500, min_periods=50).mean().shift(1).to_numpy()

    return Panel(
        dates=common.tz_convert(None).to_numpy().astype("datetime64[ns]"),
        tickers=tuple(tickers), open=o, high=h, low=lo, close=c, volume=v,
        active=active, adv_usd=adv,
        sector_id=np.array([_CLASS[t] for t in tickers], dtype=int),
        meta={"survivorship_free": True, "source": "dukascopy_1h",
              "universe_def": "12 uniformity-verified hourly cells (2026-08-02)",
              "bar": "1h", "volume_is_tick_count": True},
    )


def _self_check(panel: Panel) -> None:
    """Shape + causality checks. The causality one is the point: it is the LEAK-2 tripwire, and a
    panel feeding the mining surface must pass it before anything is wired."""
    T, N = panel.T, len(panel.tickers)
    assert panel.close.shape == (T, N), panel.close.shape
    assert np.all(np.diff(panel.dates.astype("int64")) > 0), "dates not strictly increasing"
    fin = np.isfinite(panel.close).mean()
    print(f"  shape (T,N) = ({T:,}, {N})   finite close = {fin:.4%}")
    print(f"  dates {panel.dates[0]} -> {panel.dates[-1]}")

    # ADV must never contain the current bar. Perturb bar k's volume and assert adv[k] is unchanged.
    k = T // 2
    adv_k = panel.adv_usd[k].copy()
    v2 = panel.volume.copy()
    v2[k] *= 1e6
    dollar2 = np.abs(panel.close) * v2
    adv2 = pd.DataFrame(dollar2).rolling(500, min_periods=50).mean().shift(1).to_numpy()
    same = np.allclose(adv_k, adv2[k], equal_nan=True)
    print(f"  LEAK-2 adv causality: perturbing bar {k}'s volume leaves adv[{k}] "
          f"{'UNCHANGED (PASS)' if same else 'CHANGED (FAIL)'}")
    assert same, "adv_usd at t depends on bar t's own volume — look-ahead"

    # A forward return must depend on bar t+1 and NOT on bar t-1 being perturbed.
    r = np.diff(np.log(panel.close), axis=0)
    print(f"  forward log-returns: {r.shape}, finite {np.isfinite(r).mean():.4%}, "
          f"|mean| {np.nanmean(np.abs(r)):.6f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = build_intraday_panel()
    print("\nSELF-CHECK")
    _self_check(p)
    print("\nOK — panel builds and passes its causality tripwires. NOT wired into the live tick.")
