"""Taiwan small/mid-cap alt-data probes through the cross-sectional IC funnel (probe step 3).

Pre-registration: ``docs/research/taiwan_smallcap_altdata_probes_preregistration_2026-07-15.md``.
Reads the step-1 tidy parquets + the step-2 ``membership.parquet``, builds a
:class:`sharpen.signals.Panel` over the cap-rank 51-250 small/mid band with the THREE
alt-data channels injected as causally as-of-aligned ``feature_slots`` (CR-9), and runs the three
pre-registered signals through the existing deflated 6-tier funnel (``evaluate_batch``) under the
parallel-pathway gates (``configs/taiwan_smallcap_altdata.gates.yaml``).

  P1 ``tw_smallcap_mom_rev``      sign +1  month-revenue YoY growth (post-announcement drift)
  P2 ``tw_smallcap_margin_crowd`` sign -1  Δ21d retail margin utilization (leverage crowding → reversal)
  P3 ``tw_smallcap_holder_conc``  sign +1  Δ4w big-holder (>400-lot) 集保 concentration (accumulation)

CAUSALITY (LEAK-2): every channel enters the panel through :func:`_asof_grid`, which stamps
``value[t,n]`` = the last event whose ``avail_date <= panel.dates[t]`` — the public-availability lag
computed once in ``fetch_taiwan_fundamentals_finmind``. ``Panel.truncated`` slices the slots, so the
Tier-0 truncation tripwire (``compute(truncated)[t] == compute(full)[t]``) holds by construction.
Size-neutralization is MANDATORY and lives in each ``SignalSpec.neutralization`` (read by
``evaluate_signal``) — it cannot be silently dropped.

Usage:
    python scripts/research/taiwan_smallcap_altdata_eval.py \
        --data data/taiwan_smallcap --gates configs/taiwan_smallcap_altdata.gates.yaml \
        --out results/taiwan_smallcap_altdata
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sharpen.crucible.data import taiwan_smallcap_panel as tsp  # noqa: E402
from sharpen.signals import (  # noqa: E402
    Gates,
    Multiplicity,
    Panel,
    evaluate_batch,
    load_multiplicity_gates,
    to_markdown,
    write_scorecard,
)
from sharpen.signals.spec import SignalSpec  # noqa: E402

# The panel builder and its causal-alignment helpers moved into the library
# (crucible/data/taiwan_smallcap_panel.py) when `taiwan_smallcap` was wired as a Crucible substrate,
# so the miner and these probes score the SAME panel. Re-exported under their original private names
# because the Q1/Q2 and S1 probe scripts import them from this module as `base._asof_grid` etc.
_asof_grid = tsp.asof_grid
_month_revenue_yoy = tsp.month_revenue_yoy
_causal_total_return_factor = tsp.causal_total_return_factor
_daily_membership = tsp.daily_membership
_sector_map = tsp.sector_map
_FROZEN_POOL = tsp._FROZEN_POOL


def _margin_util(margin, shareholding):
    """``[stock_id, avail_date, margin_util]`` — see :func:`taiwan_smallcap_panel.balance_util`."""
    return tsp.balance_util(margin, shareholding,
                            balance_col="margin_balance", out_col="margin_util")


log = logging.getLogger("taiwan_smallcap_altdata")

# Pre-registered spec constants — MUST reproduce the frozen content-hashes (asserted below and in
# tests). Any edit here that changes a hash is a pre-registration violation.
_NEU = ("winsor", "zscore", "sector", "size")   # "size" MANDATORY (killed the June mirage)
_HZ = (1, 5, 10, 21, 63)
_UNI = "twse_smallcap_caprank_51_250"
_CP = "taiwan_standard"
PREREG_HASHES = {
    "tw_smallcap_mom_rev": "60680e61ff85",
    "tw_smallcap_margin_crowd": "e0a4c719bfe0",
    "tw_smallcap_holder_conc": "1be26f02ee6a",
}
MARGIN_LOOKBACK = 21   # Δ trading days for margin utilization (≈ 1 month)
HOLDER_LOOKBACK = 20   # Δ trading days for 集保 concentration (≈ 4 weeks)


# --------------------------------------------------------------------------- #
# Pre-registered signals — read their channel from panel.feature_slots (CR-9)
# --------------------------------------------------------------------------- #
class _SlotLevel:
    """P1: return the causally-aligned channel level directly (the 'momentum' IS the YoY level)."""

    def __init__(self, name: str, hypothesis: str, sign: int, slot: str) -> None:
        self.slot = slot
        self.spec = SignalSpec(name=name, hypothesis=hypothesis, family="altdata",
                               expected_sign=sign, horizons=_HZ, neutralization=_NEU,
                               universe=_UNI, cost_profile=_CP)

    def compute(self, panel: Panel) -> np.ndarray:
        v = np.asarray(panel.feature_slots[self.slot], dtype=np.float64)
        return v if v.ndim == 2 else np.tile(v[:, None], (1, panel.N))


class _SlotDelta:
    """P2/P3: causal Δ over ``lookback`` trading days of an aligned channel level (row t uses <= t)."""

    def __init__(self, name: str, hypothesis: str, sign: int, slot: str, lookback: int) -> None:
        self.slot, self.lookback = slot, lookback
        self.spec = SignalSpec(name=name, hypothesis=hypothesis, family="altdata",
                               expected_sign=sign, horizons=_HZ, neutralization=_NEU,
                               universe=_UNI, cost_profile=_CP)

    def compute(self, panel: Panel) -> np.ndarray:
        v = np.asarray(panel.feature_slots[self.slot], dtype=np.float64)
        out = np.full(v.shape, np.nan, dtype=np.float64)
        lb = self.lookback
        if lb < v.shape[0]:
            out[lb:] = v[lb:] - v[:-lb]          # causal: t vs t-lb, both <= t
        return out


def build_signals() -> dict[str, object]:
    p1 = _SlotLevel("tw_smallcap_mom_rev",
                    "Monthly-revenue YoY growth predicts cross-sectional continuation in TWSE/TPEx "
                    "small-mid caps (post-announcement drift under thin analyst coverage)",
                    1, "mrev_yoy")
    p2 = _SlotDelta("tw_smallcap_margin_crowd",
                    "Rising retail margin-financing balance (leverage crowding, normalized by shares) "
                    "predicts cross-sectional reversal in small-mid caps",
                    -1, "margin_util", MARGIN_LOOKBACK)
    p3 = _SlotDelta("tw_smallcap_holder_conc",
                    "Rising big-holder shareholding concentration (>400-lot tier share, 集保) predicts "
                    "cross-sectional continuation in small-mid caps (informed accumulation)",
                    1, "holder_conc", HOLDER_LOOKBACK)
    sigs = {s.spec.name: s for s in (p1, p2, p3)}
    for name, s in sigs.items():                 # anti-p-hacking seal
        got = s.spec.content_hash()
        if got != PREREG_HASHES[name]:
            raise SystemExit(f"SPEC DRIFT: {name} hash {got} != pre-registered {PREREG_HASHES[name]} "
                             "— a pre-registration violation; revert the spec edit.")
    return sigs


# --------------------------------------------------------------------------- #
# Panel construction — delegated to the library builder (one implementation)
# --------------------------------------------------------------------------- #
def build_panel(data: Path, *, adv_window: int = 20) -> Panel:
    """The locked cap-rank 51-250 panel with the three 2026-07-15 alt-data channels.

    Thin delegation to :func:`taiwan_smallcap_panel.build_taiwan_smallcap_panel`, which owns the
    universe, the causal total-return basis, the frozen sector map and the as-of channel alignment.
    ``channels`` is left at the default (:data:`~taiwan_smallcap_panel.CORE_CHANNELS`) so this
    campaign's scorecards are byte-identical to the sealed ones; the miner asks for every channel.
    """
    return tsp.build_taiwan_smallcap_panel(data, adv_window=adv_window,
                                           channels=tsp.CORE_CHANNELS)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Taiwan small/mid-cap alt-data probes through the funnel")
    ap.add_argument("--data", default="data/taiwan_smallcap")
    ap.add_argument("--gates", default="configs/taiwan_smallcap_altdata.gates.yaml")
    ap.add_argument("--out", default="results/taiwan_smallcap_altdata")
    ap.add_argument("--adv-window", type=int, default=20)
    args = ap.parse_args()

    data = (ROOT / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    gates_path = (ROOT / args.gates) if not Path(args.gates).is_absolute() else Path(args.gates)
    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)

    panel = build_panel(data, adv_window=args.adv_window)
    log.info("Panel: pool N=%d, T=%d (%s..%s) | liquid days(>=25)=%d | %s | return=%s",
             panel.N, panel.T, str(panel.dates[0])[:10], str(panel.dates[-1])[:10],
             panel.meta["liquid_days_ge25"], panel.meta["universe_def"], panel.meta["return_basis"])
    for k, s in panel.feature_slots.items():
        log.info("  slot %-12s coverage: %.1f%% of active cells",
                 k, 100.0 * np.isfinite(s)[panel.active].mean() if panel.active.any() else 0.0)

    gates = Gates.from_yaml(gates_path)
    signals = build_signals()
    # U5 multiplicity: this probe's hypothesis set was FROZEN at three (P1/P2/P3) before any
    # result was seen, so the batch pool and the honest hypothesis count coincide here — the
    # declaration records WHY they coincide instead of leaving it to look like the pre-U5
    # accident of submitting three signals in one call.
    mult = Multiplicity.preregistered(
        3, substrate="taiwan_smallcap_altdata",
        provenance="docs/research/taiwan_smallcap_altdata_probes_preregistration_2026-07-15.md",
        gates=load_multiplicity_gates())
    rs = evaluate_batch(signals, panel, gates, "taiwan_smallcap_altdata", multiplicity=mult)
    jp, mp = write_scorecard(rs, out_dir)
    print("\n" + to_markdown(rs) + "\n")
    log.info("Scorecard: %s | %s", mp, jp)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    raise SystemExit(main())
