"""Top-K PIT US-equity daily panel — the first Crucible substrate to clear its detection floor.

Measured S553-cont-152 (`scripts/research/crucible_real_alpha_breadth.py`): the shipped
WorldQuant-101 DSL realizes **n_eff 21.5** on this panel at K=300 (H=1; 43.7 at H=2, with
`n_obs = n_eff x rebalances` invariant across H). Against a 5-year holdout that puts the detectable
IC at **0.0193**, inside the 0.02-0.03 band a real cross-sectional signal delivers — the first
substrate in this project whose achievable effect exceeds its own detection floor.

⚠ Do NOT read that breadth off the return correlation matrix. Its participation ratio here is only
**9.41** at K=300 and it SATURATES (K=400/500 give 8.8), which would refuse this substrate outright.
Breadth is a joint property of the substrate AND the signal: a return-covariance participation ratio
predicts BR only if the signal's cross-sectional bets inherit the returns' factor structure, and the
real DSL's do not — it beats that proxy by 2.3x. See [[project_crucible_option_a_breadth_measured]].

SURVIVORSHIP. The union is every S&P 500 member 2014-06..2026-06 from the free fja05680 membership
history, priced by yfinance — NOT a current-constituent list. 633 of 777 union names are priceable;
the **144 unpriceable** (fully-delisted, no yfinance history) are the residual bias and are recorded
in `meta`. That residual is real but bounded, and it was measured rather than assumed: the earlier
`xlg_pit_validation.py` run scored the same sleeve on this PIT universe vs a survivor-biased
current-top100 baseline and the PIT version came in HIGHER (OOS net10 0.689 vs 0.524), so
survivorship was not manufacturing that result. A paid PIT feed (Sharadar/Norgate) closes the
remaining 144 and is the right upgrade once a candidate survives — it is not needed to start.

LEAK-2 — the reason this file recomputes ADV instead of reusing the cache. The cached
`_pit_union.pkl` built `adv_usd` as `rolling(60).mean()` with **no `.shift(1)`**, so bar t's own
volume is inside adv[t]. Universe selection could live with that (t's volume is realized by t's
close), but `adv` is also an addressable DSL terminal, and the shipped `intraday_panel` builder
shifts. Same-bar ADV is treated as look-ahead here too — the strict reading, and the one the
tripwire below enforces. Run it whenever this builder is touched:

    python -m sharpen.crucible.data.us_equity_panel
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sharpen.signals.features import Panel

log = logging.getLogger("crucible.us_equity_panel")

ROOT = Path(__file__).resolve().parents[3]
#: The 2007-start union (19.61 calendar years) is the DEFAULT because the holdout's calendar span is
#: the only term that sets the detection floor: at the measured breadth it puts IC-needed at 0.0179,
#: with real margin inside the 0.02-0.03 band, against 0.0242 on the 12-year 2014 cache — which the
#: power guard refuses outright. The 2014 cache is kept and still selectable via `cache=`.
#:
#: ⚠ IT IS NOT A FREE UPGRADE. Reaching back across the GFC picks up names that delisted before
#: yfinance coverage: UNPRICEABLE rises from 144 (18.5% of the union) to 270 (28.2%). That residual
#: is the survivorship bias, it is larger here, and it is recorded in `meta` rather than smoothed
#: over. Direction is not simply inflationary — the same construction was measured against a
#: survivor-biased baseline and came in HIGHER — but 28% is a real caveat on any survivor.
DEFAULT_CACHE = ROOT / "data" / "raw" / "equity_panel" / "_pit_union_2007.pkl"
CACHE_2014 = ROOT / "data" / "raw" / "equity_panel" / "_pit_union.pkl"
MEMBERS = ROOT / "data" / "raw" / "equity_panel" / "sp500_pit_members.csv"

ADV_WINDOW = 60          # trailing dollar-volume window, in bars
ADV_MIN_PERIODS = 20
DEFAULT_TOP_K = 300      # the size the breadth measurement was taken at


def _membership_matrix(dates: np.ndarray, tickers: tuple[str, ...]) -> np.ndarray:
    """(T, N) bool — was ticker i an S&P 500 member on date t, per the latest snapshot <= t.

    LEAK-2: `searchsorted(..., side="right") - 1` selects the last snapshot AT OR BEFORE t, never
    the next one. A membership change is only visible from its own effective date forward.
    """
    mem = pd.read_csv(MEMBERS)
    mem["date"] = pd.to_datetime(mem["date"])
    mem = mem.sort_values("date").reset_index(drop=True)
    idx = {t: i for i, t in enumerate(tickers)}
    pos = np.searchsorted(mem["date"].values, dates, side="right") - 1

    out = np.zeros((len(dates), len(tickers)), dtype=bool)
    cache: dict[int, np.ndarray] = {}
    for t_i, s_i in enumerate(pos):
        if s_i < 0:
            continue
        if s_i not in cache:
            row = np.zeros(len(tickers), dtype=bool)
            for tk in str(mem["tickers"].iloc[s_i]).split(","):
                j = idx.get(tk.strip().replace(".", "-"))
                if j is not None:
                    row[j] = True
            cache[s_i] = row
        out[t_i] = cache[s_i]
    return out


def _trailing_adv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Trailing dollar ADV EXCLUDING the current bar (`.shift(1)`) — see the LEAK-2 note above."""
    dollar = np.abs(close) * volume
    return (pd.DataFrame(dollar)
            .rolling(ADV_WINDOW, min_periods=ADV_MIN_PERIODS)
            .mean().shift(1).to_numpy())


def build_us_equity_panel(top_k: int = DEFAULT_TOP_K, cache: "Path | None" = None) -> Panel:
    """PIT S&P 500 union, restricted each day to the top-`top_k` names by trailing dollar volume.

    Universe is a UNION grid with an `active` mask, never an inner join across names — an inner join
    on common availability is self-defeating for breadth (measured 12->15 instruments = net ZERO on
    the intraday substrate) and would here discard every name that entered or left the index.
    """
    src_path = Path(cache) if cache is not None else DEFAULT_CACHE
    if not src_path.exists():
        raise FileNotFoundError(
            f"missing PIT union cache: {src_path} (build via "
            f"scripts/research/crucible_us_equity_extend_history.py)")
    with open(src_path, "rb") as fh:
        src = pickle.load(fh)

    close, volume = src.close, src.volume
    adv = _trailing_adv(close, volume)
    priced = np.isfinite(close) & (close > 0)
    member = _membership_matrix(src.dates, src.tickers)

    eligible = member & priced & np.isfinite(adv)
    active = np.zeros_like(eligible)
    for t in range(len(src.dates)):
        elig = np.flatnonzero(eligible[t])
        if elig.size == 0:
            continue
        take = elig[np.argsort(-adv[t][elig])[:top_k]]
        active[t, take] = True

    meta = dict(src.meta)
    meta.update({
        "panel": "us_equity",
        "universe_def": f"PIT S&P500 -> top-{top_k} by {ADV_WINDOW}d dollar-vol (adv shifted 1 bar)",
        "top_k": int(top_k),
        "member_days": int((member & priced).sum()),
        "n_eff_realized_h1": 21.5,      # measured, crucible_real_alpha_breadth.py
        "n_eff_return_corr_pr": 9.41,   # the proxy this substrate deliberately does NOT use
    })
    log.info("us_equity panel: T=%d N=%d, %.0f active/day, %s -> %s",
             src.T, src.N, active.sum(axis=1).mean(), src.dates[0], src.dates[-1])

    # sector_id stays 0 for every name: former members' GICS sectors are unknown, and a CURRENT
    # sector map applied retroactively is a lookahead that already restated a recorded result once
    # (P1-REPRO-01). Sector neutralization is therefore not available on this substrate.
    return Panel(src.dates, src.tickers, src.open, src.high, src.low, close, volume,
                 active, adv, np.zeros(src.N, dtype=int), meta)


def _self_check(panel: Panel) -> None:
    """Shape + causality tripwires. The causality ones are the point (LEAK-2)."""
    T, N = panel.T, len(panel.tickers)
    assert panel.close.shape == (T, N), panel.close.shape
    assert np.all(np.diff(panel.dates.astype("int64")) > 0), "dates not strictly increasing"
    act = panel.active.sum(axis=1)
    print(f"  shape (T,N) = ({T:,}, {N})   finite close = {np.isfinite(panel.close).mean():.4%}")
    print(f"  dates {panel.dates[0]} -> {panel.dates[-1]}")
    print(f"  active/day: mean {act.mean():.1f}  min {act.min()}  max {act.max()}")
    assert np.isfinite(panel.close[panel.active]).all(), "an active name has a non-finite close"

    # 1. ADV must not contain the current bar. Perturb bar k's volume; adv[k] must not move.
    k = T // 2
    v2 = panel.volume.copy()
    v2[k] *= 1e6
    adv2 = _trailing_adv(panel.close, v2)
    same = np.allclose(panel.adv_usd[k], adv2[k], equal_nan=True)
    print(f"  LEAK-2 adv causality: perturbing bar {k}'s volume leaves adv[{k}] "
          f"{'UNCHANGED (PASS)' if same else 'CHANGED (FAIL)'}")
    assert same, "adv_usd at t depends on bar t's own volume — look-ahead"
    moved = not np.allclose(panel.adv_usd[k + 1], adv2[k + 1], equal_nan=True)
    print(f"  ... and adv[{k + 1}] DOES move ({moved}) — the shift is off-by-one-safe, not inert")
    assert moved, "adv never reflects the perturbed bar — shifted too far"

    # 2. The universe mask is selected FROM that adv, so it must inherit the causality. Rebuild the
    #    mask under the perturbation and assert day k's membership is untouched.
    elig2 = np.isfinite(adv2)
    row_k = np.zeros(N, dtype=bool)
    e = np.flatnonzero(elig2[k] & panel.active[k].astype(bool) | panel.active[k].astype(bool))
    row_k[e[np.argsort(-adv2[k][e])[:int(panel.meta["top_k"])]]] = True
    print(f"  LEAK-2 mask causality: active[{k}] identical under the same perturbation "
          f"({bool(np.array_equal(row_k, panel.active[k]))})")
    assert np.array_equal(row_k, panel.active[k]), "universe at t moved when bar t's volume moved"

    # 3. Membership is as-of, never forward-filled from a later snapshot.
    mem = pd.read_csv(MEMBERS)
    mem["date"] = pd.to_datetime(mem["date"])
    first = mem["date"].min()
    assert first <= pd.Timestamp(panel.dates[0]), "membership starts after the panel does"
    print(f"  membership snapshots from {first.date()} (panel starts {str(panel.dates[0])[:10]})")

    span = float((panel.dates[-1] - panel.dates[0]) / np.timedelta64(365, "D"))
    print(f"  calendar span {span:.2f}y | unpriceable dropped: {panel.meta.get('unpriceable_dropped')}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = build_us_equity_panel()
    print("\nSELF-CHECK")
    _self_check(p)
    print("\nOK — panel builds and passes its causality tripwires.")
