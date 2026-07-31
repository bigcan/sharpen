"""Taiwan small/mid-cap PRICE-ONLY probes (R1/R2) through the locked signal funnel.

Pre-registration: `docs/research/taiwan_smallcap_price_probes_preregistration_2026-07-31.md`
(committed with an empty Results section BEFORE this script was run — the sign commitment is
verifiable from git history).

Reuses `taiwan_smallcap_altdata_eval.build_panel` verbatim, so the universe, the PIT membership,
the causal total-return basis and the size/sector controls are byte-identical to the campaign that
produced P1. The only new thing here is two price-derived signals; nothing about the panel, the
gates or the scorecard is re-implemented.

Both signals are strictly causal — row t uses bars <= t only — and the harness's Tier-0 hygiene
check re-verifies that independently.
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
from taiwan_smallcap_altdata_eval import build_panel, load_multiplicity_gates  # noqa: E402

log = logging.getLogger("taiwan_smallcap_price")

_HZ = (1, 5, 10, 21, 63)
_NEU = ("winsor", "zscore", "sector", "size")
_UNI = "twse_smallcap_caprank_51_250"
_CP = "tw_smallcap_standard"

REVERSAL_LOOKBACK = 21     # pre-registered
IVOL_WINDOW = 63           # pre-registered

# Cumulative declared hypothesis count across BOTH campaigns + the quota-blocked institutional
# pair. Pre-registered-but-unrun hypotheses still count — that is the file-drawer control.
DECLARED_HYPOTHESES = 7


class _PastReturn:
    """R1: causal trailing simple return over ``lookback`` sessions. Row t uses close[t] and
    close[t-lookback], both <= t."""

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
            prev = c[:-lb]
            with np.errstate(invalid="ignore", divide="ignore"):
                out[lb:] = np.where(prev > 0, c[lb:] / prev - 1.0, np.nan)
        return out


class _TrailingVol:
    """R2: causal trailing standard deviation of daily returns over ``window`` sessions.

    Uses a cumulative-sum formulation over a NaN-filled return matrix so a name's inactive gaps do
    not silently borrow another period's data: cells whose window is not fully populated stay NaN.
    """

    def __init__(self, name: str, hypothesis: str, sign: int, window: int) -> None:
        self.window = window
        self.spec = SignalSpec(name=name, hypothesis=hypothesis, family="technical",
                               expected_sign=sign, horizons=_HZ, neutralization=_NEU,
                               universe=_UNI, cost_profile=_CP)

    def compute(self, panel: Panel) -> np.ndarray:
        c = np.asarray(panel.close, dtype=np.float64)
        t, n = c.shape
        r = np.full((t, n), np.nan, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            r[1:] = np.where(c[:-1] > 0, c[1:] / c[:-1] - 1.0, np.nan)
        w = self.window
        out = np.full((t, n), np.nan, dtype=np.float64)
        if w >= t:
            return out
        ok = np.isfinite(r)
        rz = np.where(ok, r, 0.0)
        cs = np.cumsum(rz, axis=0)
        cs2 = np.cumsum(rz * rz, axis=0)
        cn = np.cumsum(ok.astype(np.float64), axis=0)
        # window (t-w, t] — strictly past-and-present, no future bar enters
        s1 = cs[w:] - cs[:-w]
        s2 = cs2[w:] - cs2[:-w]
        cnt = cn[w:] - cn[:-w]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = s1 / cnt
            var = s2 / cnt - mean * mean
            sd = np.sqrt(np.maximum(var, 0.0))
        # require the window essentially complete, else the vol is a small-sample artifact
        out[w:] = np.where(cnt >= 0.8 * w, sd, np.nan)
        return out


def build_signals() -> dict[str, object]:
    r1 = _PastReturn(
        "tw_smallcap_st_reversal",
        "Short-horizon cross-sectional REVERSAL in TWSE/TPEx small-mid caps: liquidity providers "
        "earn a premium for absorbing uninformed retail order flow in a retail-dominated market",
        -1, REVERSAL_LOOKBACK)
    r2 = _TrailingVol(
        "tw_smallcap_ivol",
        "High-volatility 'lottery' small-mid caps are over-bought by retail and subsequently "
        "underperform (lottery preference + limits to arbitrage); total vol as an IVOL proxy "
        "after sector+size neutralization",
        -1, IVOL_WINDOW)
    return {s.spec.name: s for s in (r1, r2)}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Taiwan small/mid-cap PRICE-ONLY probes (R1/R2)")
    ap.add_argument("--data", default="data/taiwan_smallcap")
    ap.add_argument("--gates", default="configs/taiwan_smallcap_price.gates.yaml")
    ap.add_argument("--out", default="results/taiwan_smallcap_price")
    ap.add_argument("--adv-window", type=int, default=20)
    args = ap.parse_args()

    data = (ROOT / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    gates_path = (ROOT / args.gates) if not Path(args.gates).is_absolute() else Path(args.gates)
    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)

    panel = build_panel(data, adv_window=args.adv_window)
    log.info("Panel: N=%d T=%d (%s..%s) | liquid days=%d | %s",
             panel.N, panel.T, str(panel.dates[0])[:10], str(panel.dates[-1])[:10],
             panel.meta["liquid_days_ge25"], panel.meta["universe_def"])

    gates = Gates.from_yaml(gates_path)
    signals = build_signals()
    for name, s in signals.items():
        v = s.compute(panel)
        cov = 100.0 * np.isfinite(v)[panel.active].mean() if panel.active.any() else 0.0
        log.info("  signal %-24s coverage: %.1f%% of active cells", name, cov)

    mult = Multiplicity.preregistered(
        DECLARED_HYPOTHESES, substrate="taiwan_smallcap",
        provenance="docs/research/taiwan_smallcap_price_probes_preregistration_2026-07-31.md",
        gates=load_multiplicity_gates())
    rs = evaluate_batch(signals, panel, gates, "taiwan_smallcap_price", multiplicity=mult)
    jp, mp = write_scorecard(rs, out_dir)
    print("\n" + to_markdown(rs) + "\n")
    log.info("Scorecard: %s | %s", mp, jp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
