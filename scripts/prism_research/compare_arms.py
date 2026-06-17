"""Cross-arm comparison for the PRISM daily re-eval (base vs v2 vs fresh fine-tune).

Reads the per-arm predictive-eval summaries produced by `prism_predictive_eval.py`
(one `arm_<name>/` dir per arm, each with a full-history CSV and an `_oos` CSV) and
emits a single side-by-side table so the operator can read the headline question:
**does fine-tuning Chronos unlock an edge the zero-shot base lacks, on the clean OOS
window?**

Edge flag (mirrors the eval headline + the pre-registered kill criterion):
    Chronos edge  <=>  chronos_live AND |chronos_ic| > chronos_ic_null95 AND
                       chronos_net_pf > 1.0   (net of the eval's one-way cost)
The GAHMM regime block is Chronos-independent, so it is reported once as a reference and
cross-checked for drift across arms (a free HMM-stability check).

Usage:
    python scripts/prism_research/compare_arms.py
    python scripts/prism_research/compare_arms.py --arms base v2 ft3 --assets btc gold eurusd
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prism_research.compare_arms")

RESULTS_ROOT = PROJECT_ROOT / "results" / "prism_research"


def _load_arm(results_root: Path, arm: str) -> pd.DataFrame:
    """Load an arm's full + _oos summaries, tagging window and arm (dir is authoritative)."""
    arm_dir = results_root / f"arm_{arm}"
    frames = []
    for suffix, win in (("", "full"), ("_oos", "oos")):
        path = arm_dir / f"prism_predictive_eval_summary{suffix}.csv"
        if not path.exists():
            logger.warning("missing %s (skipping)", path)
            continue
        df = pd.read_csv(path)
        df["arm"] = arm                      # dir name is authoritative over the CSV's tag
        df["window_kind"] = win
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _chronos_edge(r: dict) -> bool:
    ic = r.get("chronos_ic", np.nan)
    null95 = r.get("chronos_ic_null95", np.inf)
    pf = r.get("chronos_net_pf", 0.0)
    return bool(r.get("chronos_live", False)
                and np.isfinite(ic) and np.isfinite(null95)
                and abs(ic) > null95 and pf > 1.0)


def _fmt(r: dict) -> str:
    if not r.get("chronos_live", False):
        return "inert"
    ic = r.get("chronos_ic", float("nan"))
    null95 = r.get("chronos_ic_null95", float("nan"))
    pf = r.get("chronos_net_pf", float("nan"))
    mdd = r.get("chronos_mdd", float("nan"))
    flag = "EDGE" if _chronos_edge(r) else "----"
    return f"{flag} IC={ic:+.3f}(n95={null95:.3f}) PF={pf:.2f} MDD={100 * mdd:+.0f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PRISM eval arms (base/v2/ft3)")
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--arms", nargs="+", default=["base", "v2", "ft3"])
    parser.add_argument("--assets", nargs="+", default=["btc", "gold", "eurusd"])
    args = parser.parse_args()

    results_root = Path(args.results_root)
    all_rows = pd.concat([_load_arm(results_root, a) for a in args.arms], ignore_index=True)
    if all_rows.empty:
        logger.error("No arm summaries found under %s.", results_root)
        sys.exit(1)

    out_path = results_root / "arm_comparison.csv"
    all_rows.to_csv(out_path, index=False)
    logger.info("Saved raw cross-arm table -> %s\n", out_path)

    # ----- headline: Chronos directional edge, OOS window, asset x arm -----
    for win in ("oos", "full"):
        block = all_rows[all_rows["window_kind"] == win]
        if block.empty:
            continue
        logger.info("=" * 78)
        logger.info("CHRONOS directional readout — %s window", win.upper())
        logger.info("=" * 78)
        header = "  ".join(f"{a:>34}" for a in args.arms)
        logger.info("%-7s  %s", "asset", header)
        for asset in args.assets:
            cells = []
            for arm in args.arms:
                sel = block[(block["asset"] == asset) & (block["arm"] == arm)]
                cells.append(f"{_fmt(sel.iloc[0].to_dict()) if len(sel) else 'n/a':>34}")
            logger.info("%-7s  %s", asset, "  ".join(cells))

    # ----- GAHMM regime reference (Chronos-independent; check cross-arm drift) -----
    if "regime_ic" in all_rows.columns:
        logger.info("\n%s", "=" * 78)
        logger.info("GAHMM regime IC (bull-bear) — should be ~arm-invariant (drift = HMM init)")
        logger.info("=" * 78)
        ref = all_rows[all_rows["window_kind"] == "oos"]
        if ref.empty:
            ref = all_rows
        for asset in args.assets:
            vals = {arm: ref[(ref["asset"] == asset) & (ref["arm"] == arm)]["regime_ic"].tolist()
                    for arm in args.arms}
            flat = [v[0] for v in vals.values() if v and np.isfinite(v[0])]
            drift = (max(flat) - min(flat)) if len(flat) >= 2 else float("nan")
            logger.info("  %-7s regime_ic per arm=%s  (max-min drift=%.4f)",
                        asset, {a: (round(v[0], 4) if v else None) for a, v in vals.items()}, drift)

    # ----- verdict line -----
    oos = all_rows[all_rows["window_kind"] == "oos"]
    edges = [(r["arm"], r["asset"]) for r in oos.to_dict("records") if _chronos_edge(r)]
    logger.info("\n%s", "=" * 78)
    if edges:
        logger.info("OOS Chronos EDGE detected (arm, asset): %s", edges)
        logger.info("-> Does NOT auto-promote. Triggers PRISM-revival Tier-2 deep lifecycle "
                    "audit + half-cost robustness + market-beta check before any capital.")
    else:
        logger.info("No OOS Chronos edge in any arm. Fine-tuning does not rescue the daily "
                    "PRISM signal; the cont-49 falsification stands (now base+v2+fine-tune).")


if __name__ == "__main__":
    main()
