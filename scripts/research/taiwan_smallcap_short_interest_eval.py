"""Taiwan small/mid-cap SHORT-INTEREST probe (S1) through the locked signal funnel.

Pre-registration: `docs/research/taiwan_smallcap_short_interest_preregistration_2026-07-31.md`
(committed with an empty Results section BEFORE this ran).

Reuses `taiwan_smallcap_altdata_eval`'s panel builder and helpers verbatim, then swaps ONE thing:
the short leg of `margin_short.parquet` replaces the long leg P2 used. Causality is the same
construction as P2 — shares merged as-of backward, value stamped T+1 — so nothing about the
leakage surface changes.

Capturability is enforced here as a pass condition (pre-registered §3), because the funnel's own
`PROMISING` verdict does not consult it.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sharpen.signals.features import Panel  # noqa: E402
from sharpen.signals.multiplicity import Multiplicity  # noqa: E402
from sharpen.signals.scorecard import (  # noqa: E402
    Gates,
    evaluate_batch,
    to_markdown,
    write_scorecard,
)
from sharpen.crucible.data import taiwan_smallcap_panel as tsp  # noqa: E402
from sharpen.signals.spec import SignalSpec  # noqa: E402
import taiwan_smallcap_altdata_eval as base  # noqa: E402

log = logging.getLogger("taiwan_smallcap_short")

_HZ = (1, 5, 10, 21, 63)
_NEU = ("winsor", "zscore", "sector", "size")
_UNI = "twse_smallcap_caprank_51_250"
_CP = "tw_smallcap_standard"
DECLARED_HYPOTHESES = 8       # cumulative across all campaigns on this substrate (pre-reg §4)


def _short_util(margin: pd.DataFrame, shareholding: pd.DataFrame) -> pd.DataFrame:
    """``[stock_id, avail_date, short_util]`` = short balance / causal total shares.

    The SAME causal construction as the margin leg, with `short_balance` substituted for
    `margin_balance` — which is why both now route through one library function
    (:func:`taiwan_smallcap_panel.balance_util`) rather than two hand-copied ones: shares are merged
    as-of BACKWARD on the balance row's own trading date (so only an already-public share count is
    used) and the result is stamped T+1 because TWSE publishes balances after the close.
    """
    return tsp.balance_util(margin, shareholding,
                            balance_col="short_balance", out_col="short_util")


class _SlotLevel:
    """S1 reads the causally-aligned short-interest LEVEL (the canonical form of the anomaly)."""

    def __init__(self, name: str, hypothesis: str, sign: int, slot: str) -> None:
        self.slot = slot
        self.spec = SignalSpec(name=name, hypothesis=hypothesis, family="altdata",
                               expected_sign=sign, horizons=_HZ, neutralization=_NEU,
                               universe=_UNI, cost_profile=_CP)

    def compute(self, panel: Panel) -> np.ndarray:
        v = np.asarray(panel.feature_slots[self.slot], dtype=np.float64)
        return v if v.ndim == 2 else np.tile(v[:, None], (1, panel.N))


def build_panel_with_short(data: Path, adv_window: int = 20) -> Panel:
    """The locked panel, plus one extra feature slot: `short_util`."""
    panel = base.build_panel(data, adv_window=adv_window)
    margin = pd.read_parquet(data / "margin_short.parquet")
    shareholding = pd.read_parquet(data / "shareholding.parquet")
    slot = base._asof_grid(_short_util(margin, shareholding), panel.dates, panel.tickers,
                           "short_util")
    slots = dict(panel.feature_slots)
    slots["short_util"] = slot
    from dataclasses import replace
    return replace(panel, feature_slots=slots)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Taiwan small/mid-cap SHORT-INTEREST probe (S1)")
    ap.add_argument("--data", default="data/taiwan_smallcap")
    ap.add_argument("--gates", default="configs/taiwan_smallcap_price.gates.yaml")
    ap.add_argument("--out", default="results/taiwan_smallcap_short")
    ap.add_argument("--adv-window", type=int, default=20)
    args = ap.parse_args()

    data = (ROOT / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    gates_path = (ROOT / args.gates) if not Path(args.gates).is_absolute() else Path(args.gates)
    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)

    panel = build_panel_with_short(data, adv_window=args.adv_window)
    cov = np.isfinite(panel.feature_slots["short_util"])[panel.active].mean()
    log.info("Panel N=%d T=%d | short_util coverage %.1f%% of active cells",
             panel.N, panel.T, 100.0 * cov)

    s1 = _SlotLevel(
        "tw_smallcap_short_interest",
        "High relative short interest (融券 balance / shares) predicts cross-sectional "
        "UNDERPERFORMANCE in TWSE/TPEx small-mid caps — informed short selling, the most "
        "replicated cross-sectional predictor in the literature",
        -1, "short_util")

    gates = Gates.from_yaml(gates_path)
    mult = Multiplicity.preregistered(
        DECLARED_HYPOTHESES, substrate="taiwan_smallcap",
        provenance="docs/research/taiwan_smallcap_short_interest_preregistration_2026-07-31.md",
        gates=base.load_multiplicity_gates())
    rs = evaluate_batch({s1.spec.name: s1}, panel, gates, "taiwan_smallcap_short",
                        multiplicity=mult)
    jp, mp = write_scorecard(rs, out_dir)
    try:
        print("\n" + to_markdown(rs) + "\n")
    except UnicodeEncodeError:                     # cp950 console; the artifact is already on disk
        log.warning("console cannot encode the markdown — read %s", mp)
    log.info("Scorecard: %s | %s", mp, jp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
