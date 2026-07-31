"""Q1/Q2 — Taiwan small/mid-cap INSTITUTIONAL-FLOW probes through the locked signal funnel.

Pre-registration: `docs/research/taiwan_smallcap_institutional_flow_preregistration_2026-07-31.md`
(committed in 8ea97511 with an empty Results section, BEFORE the data was even fetched).

Q1 `tw_smallcap_foreign_flow`  sign +1  — 21d foreign net share flow / 21d traded volume
Q2 `tw_smallcap_trust_flow`    sign +1  — identical construction on investment-trust flow,
                                          a MECHANISM TEST (see pre-reg §2), not a second shot.

Reuses `taiwan_smallcap_altdata_eval`'s locked panel builder and `_asof_grid` verbatim, so the
universe, PIT membership, causal total-return basis and size/sector controls are byte-identical to
the campaign that produced P1.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from finrl_pro_ds.signals.features import Panel  # noqa: E402
from finrl_pro_ds.signals.multiplicity import Multiplicity  # noqa: E402
from finrl_pro_ds.signals.scorecard import (  # noqa: E402
    Gates,
    evaluate_batch,
    to_markdown,
    write_scorecard,
)
from finrl_pro_ds.signals.spec import SignalSpec  # noqa: E402
import taiwan_smallcap_altdata_eval as base  # noqa: E402

log = logging.getLogger("taiwan_smallcap_inst")

_HZ = (1, 5, 10, 21, 63)
_NEU = ("winsor", "zscore", "sector", "size")
_UNI = "twse_smallcap_caprank_51_250"
_CP = "tw_smallcap_standard"
FLOW_WINDOW = 21          # pre-registered

# Honest CUMULATIVE count on this substrate. The pre-reg said 5 (3 alt-data + these 2), but R1/R2
# and S1 have since been run on the SAME panel, so the true count is 8. Using the larger, stricter
# number rather than the stale one the document happens to name.
DECLARED_HYPOTHESES = 8


def _flow_events(inst: pd.DataFrame, col: str, out_col: str) -> pd.DataFrame:
    """Rolling `FLOW_WINDOW`-day SUM of a net-flow column, per stock, carrying its avail_date.

    The sum ends at the row's own trading date and the row is already stamped
    ``avail_date = date + 1 business day`` by the fetcher (T86 publishes after the close), so the
    value can only enter the panel on a session strictly after every bar it uses (LEAK-2).
    """
    d = inst[["stock_id", "date", "avail_date", col]].copy()
    d["stock_id"] = d["stock_id"].astype(str)
    d["date"] = pd.to_datetime(d["date"])
    d["avail_date"] = pd.to_datetime(d["avail_date"])
    d = d.sort_values(["stock_id", "date"])
    d[out_col] = (d.groupby("stock_id")[col]
                  .rolling(FLOW_WINDOW, min_periods=FLOW_WINDOW).sum().reset_index(level=0, drop=True))
    return d.dropna(subset=[out_col])[["stock_id", "avail_date", out_col]]


class _FlowIntensity:
    """Q1/Q2: net institutional share flow over `FLOW_WINDOW` days, as a fraction of the shares
    actually traded over the same window. Dimensionless and scale-free, so it is not a size proxy.

    Denominator comes from the PANEL's own volume (bars <= t), numerator from the causally
    as-of-aligned flow slot — both strictly past-looking.
    """

    def __init__(self, name: str, hypothesis: str, sign: int, slot: str) -> None:
        self.slot = slot
        self.spec = SignalSpec(name=name, hypothesis=hypothesis, family="altdata",
                               expected_sign=sign, horizons=_HZ, neutralization=_NEU,
                               universe=_UNI, cost_profile=_CP)

    def compute(self, panel: Panel) -> np.ndarray:
        flow = np.asarray(panel.feature_slots[self.slot], dtype=np.float64)
        vol = np.asarray(panel.volume, dtype=np.float64)
        v = pd.DataFrame(vol).rolling(FLOW_WINDOW, min_periods=FLOW_WINDOW).sum().to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(np.isfinite(v) & (v > 0), flow / v, np.nan)


def build_panel_with_flows(data: Path, adv_window: int = 20) -> Panel:
    panel = base.build_panel(data, adv_window=adv_window)
    inst = pd.read_parquet(data / "institutional.parquet")
    log.info("institutional.parquet: %d rows, %d ids, %s..%s",
             len(inst), inst["stock_id"].nunique(),
             pd.to_datetime(inst["date"]).min().date(), pd.to_datetime(inst["date"]).max().date())
    slots = dict(panel.feature_slots)
    for col, slot in (("foreign_net", "foreign_flow"), ("trust_net", "trust_flow")):
        ev = _flow_events(inst, col, slot)
        slots[slot] = base._asof_grid(ev, panel.dates, panel.tickers, slot)
    return replace(panel, feature_slots=slots)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Taiwan small/mid-cap institutional-flow probes Q1/Q2")
    ap.add_argument("--data", default="data/taiwan_smallcap")
    ap.add_argument("--gates", default="configs/taiwan_smallcap_altdata.gates.yaml")
    ap.add_argument("--out", default="results/taiwan_smallcap_institutional")
    args = ap.parse_args()

    data = (ROOT / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    gates_path = (ROOT / args.gates) if not Path(args.gates).is_absolute() else Path(args.gates)
    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)

    panel = build_panel_with_flows(data)
    for k in ("foreign_flow", "trust_flow"):
        cov = np.isfinite(panel.feature_slots[k])[panel.active].mean() if panel.active.any() else 0
        log.info("  slot %-14s coverage %.1f%% of active cells", k, 100.0 * cov)

    q1 = _FlowIntensity(
        "tw_smallcap_foreign_flow",
        "Net FOREIGN institutional share flow (21d, as a fraction of traded volume) predicts "
        "cross-sectional CONTINUATION in TWSE/TPEx small-mid caps — informed foreign buying "
        "under-reacted to in a retail-dominated, thin-coverage band",
        1, "foreign_flow")
    q2 = _FlowIntensity(
        "tw_smallcap_trust_flow",
        "Identical construction on domestic INVESTMENT-TRUST flow. MECHANISM TEST: Q1>>Q2 supports "
        "informed foreign flow; Q1~=Q2 means the signal is generic price pressure, which falsifies "
        "the stated mechanism even if the IC is positive",
        1, "trust_flow")

    for s in (q1, q2):
        v = s.compute(panel)
        cov = 100.0 * np.isfinite(v)[panel.active].mean() if panel.active.any() else 0.0
        log.info("  signal %-26s coverage %.1f%%", s.spec.name, cov)

    gates = Gates.from_yaml(gates_path)
    mult = Multiplicity.preregistered(
        DECLARED_HYPOTHESES, substrate="taiwan_smallcap",
        provenance=("docs/research/taiwan_smallcap_institutional_flow_preregistration_"
                    "2026-07-31.md"),
        gates=base.load_multiplicity_gates())
    rs = evaluate_batch({s.spec.name: s for s in (q1, q2)}, panel, gates,
                        "taiwan_smallcap_institutional", multiplicity=mult)
    jp, mp = write_scorecard(rs, out_dir)
    try:
        print("\n" + to_markdown(rs) + "\n")
    except UnicodeEncodeError:
        log.warning("console cannot encode markdown — read %s", mp)
    log.info("Scorecard: %s | %s", mp, jp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
