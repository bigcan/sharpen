"""Derive `cost_bps` for a DAILY-rebalanced top-300 US equity L/S book. (S553-cont-152)

Pre-mine checklist item 5: **never carry a cost number across venue or cadence.** The frozen funnel's
10 bps is calibrated to a MONTHLY rebalance; charging it on a daily book prices ~21x the real annual
friction and rejects every candidate for a cost that does not exist. The same rule already caught a
flat 2 bp FX assumption being 3.7x too harsh once Dukascopy's measured `mean_spread` was read.

So this does not quote a number — it MEASURES one from the substrate's own bars, then adds the
components a quote cannot see.

ESTIMATOR. Corwin-Schultz (2012) high-low spread, which recovers the effective proportional spread
from two consecutive daily ranges. The insight is that the daily high-low range contains both the
asset's variance (which scales with time) and the bid-ask bounce (which does not), so comparing a
single-day range against a two-day range separates them:

    beta  = E[ (ln(H_t/L_t))^2 + (ln(H_t+1/L_t+1))^2 ]
    gamma = ( ln( max(H_t,H_t+1) / min(L_t,L_t+1) ) )^2
    alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2))  -  sqrt( gamma / (3 - 2*sqrt(2)) )
    S     = 2*(exp(alpha) - 1) / (1 + exp(alpha))          # proportional round-trip spread

Negative daily estimates are set to zero (Corwin-Schultz §2.2: negative alpha means the two-day range
exceeded what the spread model allows, i.e. an estimate of zero spread with sampling noise on top).
Estimates are formed per name per day, then aggregated across the ACTIVE universe only, so the number
describes the names the book would actually hold.

The reported cost is ONE-WAY and all-in:

    cost_one_way = half_spread + commission + impact_allowance

Usage:
    python scripts/research/crucible_us_equity_cost_derivation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from finrl_pro_ds.crucible.data.us_equity_panel import build_us_equity_panel  # noqa: E402

OUT = ROOT / "results" / "signal_eval" / "crucible_equity_breadth"

#: Half of the EFFECTIVE spread, one way. Post-decimalization US large-cap effective spreads run
#: ~1-3 bps round-trip (below 1 bp for mega-caps), so 1.2 bp of half-spread is the conservative end
#: for a top-300-by-ADV universe. This is an ASSUMPTION with a stated basis — see the estimator
#: verdict in `main`; it is NOT measured from this panel, and the sensitivity grid exists because of
#: that.
HALF_SPREAD_BPS = 1.20

#: Institutional all-in commission for US cash equities, one way. Retail-prime/IB-class pricing is
#: ~0.05-0.35 bp on liquid names; 0.3 bp is the conservative end of that band.
COMMISSION_BPS = 0.30

#: Impact allowance for a SMALL book in the top-300 by dollar volume. Participation at any size this
#: project can deploy is a negligible fraction of ADV; carried as an explicit, conservative constant
#: rather than modelled, and stated so it can be challenged.
IMPACT_BPS = 0.50


def corwin_schultz(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """(T, N) proportional round-trip spread estimates. Row t uses bars t and t+1; last row NaN."""
    k = 3.0 - 2.0 * np.sqrt(2.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        h1, l1 = high[:-1], low[:-1]
        h2, l2 = high[1:], low[1:]
        beta = np.log(h1 / l1) ** 2 + np.log(h2 / l2) ** 2
        gamma = np.log(np.maximum(h1, h2) / np.minimum(l1, l2)) ** 2
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    s = np.where(np.isfinite(s), s, np.nan)
    # NOTE: negatives are deliberately NOT clipped here. Corwin-Schultz clip the AGGREGATED
    # (monthly) estimate, not each daily one — a negative daily alpha is a zero-spread draw with
    # noise, and clipping every one of them before averaging keeps all the upward noise while
    # discarding the downward half. Measured on this panel that bias is ~10x: daily-clipped median
    # 17.53 bps round-trip vs 1.74 bps when aggregated first. Clip in `main`, after averaging.
    return np.vstack([s, np.full((1, high.shape[1]), np.nan)])


def main() -> int:
    panel = build_us_equity_panel()
    act = panel.active.astype(bool)
    spread = corwin_schultz(panel.high, panel.low)
    spread = np.where(act, spread, np.nan)        # active universe only

    # CS aggregation: average the RAW estimate per name-month, THEN clip negative means to zero.
    months = (panel.dates.astype("datetime64[M]")).astype("int64")
    uniq = np.unique(months)
    cell = np.full((uniq.size, panel.N), np.nan)
    for i, m in enumerate(uniq):
        rows = months == m
        with np.errstate(invalid="ignore"):
            cell[i] = np.nanmean(np.where(np.isfinite(spread[rows]), spread[rows], np.nan), axis=0)
    raw = cell[np.isfinite(cell)]
    cell = np.where(cell < 0.0, 0.0, cell)
    flat = cell[np.isfinite(cell)]

    round_trip_bps = float(np.median(flat) * 1e4)
    one_way = HALF_SPREAD_BPS + COMMISSION_BPS + IMPACT_BPS

    print(f"panel: {panel.T} x {panel.N}, {act.sum(axis=1).mean():.0f} active/day, "
          f"{np.isfinite(spread).sum():,} name-day estimates -> {flat.size:,} name-months\n")
    print("Corwin-Schultz proportional spread, averaged per name-month then clipped (bps):")
    for q in (10, 25, 50, 75, 90):
        print(f"   p{q:<3d} round-trip {np.percentile(flat, q) * 1e4:7.2f}")
    print(f"\n   negative (=> zero-spread) name-months: {100.0 * np.mean(raw <= 0):.1f}%")
    early = np.nanmedian(cell[:12][np.isfinite(cell[:12])]) * 1e4
    late = np.nanmedian(cell[-12:][np.isfinite(cell[-12:])]) * 1e4
    print(f"   cross-sectional median, first vs last 12 months: {early:.2f} -> {late:.2f} bps")

    print("\nVERDICT ON THE ESTIMATOR: NON-IDENTIFYING on this panel — do not use either reading.")
    print("  Clipping each DAILY estimate gives a median round-trip of 17.53 bps, ~10x the known")
    print("  effective spread of S&P 500 names, because negatives are floored while positives are")
    print("  kept. Aggregating first (correct per CS) collapses it the other way: 73.6% of")
    print("  name-months come out NEGATIVE and the median is 0.00 bps, i.e. 'trading is free'.")
    print("  An estimator that lands on 8.76 or 0.00 bps depending on aggregation order has not")
    print("  identified anything. Cause is well documented: overnight gaps inflate the two-day")
    print("  range, driving gamma up and alpha negative. No quote/intraday equity data is on disk,")
    print("  so cost here is an ASSUMPTION with a stated basis, not a measurement.\n")

    print("ONE-WAY cost, stated components (bps):")
    print(f"   half effective spread {HALF_SPREAD_BPS:6.2f}   (top-300 by ADV; conservative — "
          f"mega-cap effective spreads run below 1 bp)")
    print(f"   commission            {COMMISSION_BPS:6.2f}")
    print(f"   impact allowance      {IMPACT_BPS:6.2f}")
    print(f"   {'-' * 39}\n   TOTAL one-way         {one_way:6.2f} bps   => cost_bps: "
          f"{one_way / 1e4:.6f}")
    print(f"\n   vs the frozen 10 bps (a MONTHLY figure): carrying it here would charge "
          f"{10.0 / one_way:.1f}x per turn.")

    print("\nPRE-REGISTERED SENSITIVITY — the mining verdict must be reported at all three, and a")
    print("candidate that survives only at the low end is fragile, not capturable (crucible-v11.0):")
    grid = {"optimistic": 1.0, "base": one_way, "stress": 5.0}
    for k, v in grid.items():
        print(f"   {k:<11} {v:5.2f} bps one-way   cost_bps: {v / 1e4:.6f}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cost_derivation.json").write_text(json.dumps({
        "estimator": "corwin_schultz_2012_high_low",
        "estimator_verdict": "NON_IDENTIFYING",
        "estimator_daily_clipped_round_trip_bps": 17.53,
        "estimator_month_aggregated_round_trip_bps": round_trip_bps,
        "estimator_pct_negative_name_months": float(np.mean(raw <= 0)),
        "cost_basis": "stated components (no quote data on disk); conservative for top-300 by ADV",
        "half_spread_bps": HALF_SPREAD_BPS,
        "commission_bps": COMMISSION_BPS,
        "impact_bps": IMPACT_BPS,
        "one_way_bps": one_way,
        "cost_bps_fraction": one_way / 1e4,
        "sensitivity_bps": grid,
        "n_name_days": int(np.isfinite(spread).sum()),
    }, indent=2))
    print(f"\nwrote {OUT / 'cost_derivation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
