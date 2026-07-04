"""Measure the alt-data OVERLAY candidate pool's return-stream diversity (Doc 3 Part C metric).

This is the falsification the weak-signal-ensemble design set says to run BEFORE building the
generative proposer (Doc 3 = ``crucible_diverse_proposer_spec.md``, "Success metric" + "Sequencing"
step 1): **is the supply of return-stream diversity even the problem?** It wires ONLY pieces that
already exist — no new statistics, no new algorithms:

    load_cross_asset_panel  ─┐
    bridge_altdata_feature_slots (widened, cont-110)  ─┤→ Panel with alt-data feature slots
    production_base_sleeves → _combined_book  ────────────→ base book (the overlay multiplier)
    LibrarySeedProposer(overlay only) → _overlay_returns ─→ one return stream per overlay candidate
    measure_pool_diversity(returns, sources=…)  ─────────→ ρ̄_pool + de-correlated admit count

and prints the two headline numbers Doc 3 pre-registers:

  * **ρ̄_pool** (``mean_abs_pairwise_corr``) — the pool's mean |return correlation|. cont-107's pool
    was ~1 within its two idea-clusters. Breadth should push the CROSS-source figure down.
  * **# de-correlated admits** at ``cohort.max_pairwise_corr`` — cont-107 would admit ≈1–2; the
    target is ``>= cohort.min_cohort_size``.

Decision rule (pre-registered, both directions cheap to falsify):
  * admits ``>=`` ``min_cohort_size``  → data breadth WAS the bottleneck; Phase 4 (wiring the cohort
    evaluator into the live evolve loop) is now justified and the generative proposer (Parts A/B)
    stays correctly deferred.
  * admits ``<``  ``min_cohort_size``  → the proposer (near-duplicate templates), not the data, is the
    binding constraint; redirect to Parts A/B and do NOT build Phase 4 speculatively.

CAVEAT (surfaced in the output, not hidden): every overlay candidate is ``tanh(z)·base_book`` — a
tilt on the SAME base book — so the shared multiplier induces correlation even when the tilts come
from orthogonal DATA. A pool that stays correlated here can therefore mean either "the data is not
orthogonal" OR "the overlay *mechanism* re-correlates orthogonal data through the common book". The
within- vs cross-source split is what separates those two readings.

Offline to BUILD; running it needs live FRED/COT/EDGAR (``FRED_API_KEY`` / ``SEC_EDGAR_UA`` in
``.env``) — the one operator-gated step. It is a CPU/pandas run (minutes), NOT a GPU job. Advisory
only: it reports diversity; it never issues a verdict and touches no gate.

Usage:
  python scripts/research/measure_altdata_pool_diversity.py --start 2008-01-01
  python scripts/research/measure_altdata_pool_diversity.py --start 2008-01-01 --end 2024-12-31 \
      --max-slots-per-source 4
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env (override=False so a real shell export always wins) — mirrors crucible_orchestrator so
# the alt-data bridge picks up FRED_API_KEY / SEC_EDGAR_UA without a per-session export.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except ModuleNotFoundError:
    pass

from finrl_pro_ds.crucible import DataCatalog  # noqa: E402
from finrl_pro_ds.crucible.agentic.proposer import (  # noqa: E402
    LibrarySeedProposer,
    ProposalContext,
)
from finrl_pro_ds.crucible.data.altdata_bridge import (  # noqa: E402
    COT_TERMINAL_ASSET_CLASS,
    bridge_altdata_feature_slots,
)
from finrl_pro_ds.data.cross_asset_panel_loader import load_cross_asset_panel  # noqa: E402
from finrl_pro_ds.signals.features import Panel  # noqa: E402
from finrl_pro_ds.signals.generation.base_sleeves import production_base_sleeves  # noqa: E402
from finrl_pro_ds.signals.generation.cohort import measure_pool_diversity  # noqa: E402
from finrl_pro_ds.signals.generation.cohort_eval import assemble_overlay_pool  # noqa: E402
from finrl_pro_ds.signals.generation.config import (  # noqa: E402
    load_cohort_config,
    load_generation_config,
)
from finrl_pro_ds.signals.generation.fitness import _combined_book  # noqa: E402
from finrl_pro_ds.signals.generation.grammar import available_terminals  # noqa: E402

log = logging.getLogger("measure_altdata_pool_diversity")

DEFAULT_GATES = ROOT / "configs" / "signal_eval.gates.yaml"
DEFAULT_PANEL_CFG = ROOT / "configs" / "cross_asset_momentum.yaml"


def _panel_ts(panel: Panel) -> np.ndarray:
    """Panel dates → float seconds since epoch (the timestamp convention the combiner expects,
    identical to crucible_orchestrator._panel_ts)."""
    return panel.dates.astype("datetime64[s]").astype(np.int64).astype(np.float64)


def _source_label(term: str) -> str:
    """Data source / asset-class of a feature-slot terminal, for the within- vs cross-source split.

    COT carries the underlying's asset class (``cot:metal`` vs ``cot:ag`` vs ``cot:rate``) so
    positioning on orthogonal markets reads as cross-source, not one lumped ``cot``. FRED = macro,
    EDGAR = fundamental. Mirrors Doc 3 Part C's "expose source/asset-class per terminal"."""
    if term.startswith("cot:"):
        return f"cot:{COT_TERMINAL_ASSET_CLASS.get(term, 'unknown')}"
    for prefix in ("fred", "edgar", "gdelt", "stooq"):
        if term.startswith(prefix + ":"):
            return prefix
    return term.split(":", 1)[0] if ":" in term else "ohlcv"


def _term_of(formula: str, feature_slots: tuple[str, ...]) -> str | None:
    """The feature-slot terminal an overlay ``formula`` references. Each overlay is
    ``template.format(t=term)`` so the terminal string appears verbatim; pick the LONGEST matching
    slot so e.g. ``cot:gold_comm_net`` is not shadowed by a shorter substring."""
    hits = [s for s in feature_slots if s in formula]
    return max(hits, key=len) if hits else None


def build_overlay_pool(
    panel: Panel,
    base_book: np.ndarray,
    *,
    cost_bps: float,
    max_proposals: int,
) -> tuple[dict[str, np.ndarray], dict[str, str], list[str]]:
    """Return ``(overlay_returns, sources, culled)`` for the alt-data overlay candidates.

    Uses the SHIPPED offline proposer (overlay-only, so the CS seed bank never truncates the batch)
    and the LIBRARY ``assemble_overlay_pool`` (the funnel's own ``_overlay_returns`` path, DRY with the
    cohort evaluator) — so the measured streams are EXACTLY what the cohort evaluator would admit from.
    A degenerate (all-NaN / globally constant) overlay is culled and recorded in ``culled``. Source
    labelling stays here (report-only) since it needs the crucible bridge's asset-class map."""
    slots = tuple(k for k in panel.feature_slots.keys())
    ctx = ProposalContext(
        available_terminals=available_terminals(panel),
        max_proposals=max_proposals,
    )
    proposals = [p for p in LibrarySeedProposer(include_cross_sectional=False).propose(ctx)
                 if p.candidate_type == "overlay"]
    overlay_formulas = {p.name: p.formula for p in proposals}
    returns, culled = assemble_overlay_pool(panel, base_book, overlay_formulas, cost_bps=cost_bps)

    sources: dict[str, str] = {}
    for p in proposals:
        if p.name not in returns:
            continue
        term = _term_of(p.formula, slots)
        sources[p.name] = _source_label(term) if term is not None else "unknown"
    return returns, sources, culled


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Measure the alt-data overlay candidate pool's return-stream diversity "
                    "(Doc 3 Part C — 'measure data breadth first').")
    ap.add_argument("--start", default="2008-01-01", help="panel start date (ISO)")
    ap.add_argument("--end", default=None, help="panel end date (ISO); default = panel max")
    ap.add_argument("--config", default=str(DEFAULT_GATES),
                    help="gates YAML (generation: + cohort: blocks) — supplies cost_bps, "
                         "hold_horizon, max_pairwise_corr, min_cohort_size")
    ap.add_argument("--panel-config", default=str(DEFAULT_PANEL_CFG),
                    help="cross-asset panel config (universe + PIT loader)")
    ap.add_argument("--max-slots-per-source", type=int, default=None,
                    help="cap feature slots per connector (Doc 3 Part C point 3); default = no cap")
    ap.add_argument("--max-proposals", type=int, default=999,
                    help="proposer batch cap (kept high so no overlay is truncated)")
    ap.add_argument("--out", default=str(ROOT / "results" / "altdata_pool_diversity"),
                    help="directory for the JSON report")
    args = ap.parse_args()

    _fit, ek = load_generation_config(args.config)
    ccfg, _mc = load_cohort_config(args.config)
    cost_bps = float(ek["cost_bps"])
    hold_horizon = int(ek["hold_horizon"])

    # 1) real cross-asset panel + the widened alt-data bridge (the one operator-gated / networked step)
    log.info("loading cross-asset panel %s..%s", args.start, args.end or "(max)")
    panel = load_cross_asset_panel(args.start, args.end, config_path=Path(args.panel_config))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog = DataCatalog(out_dir / "catalog.db")
    bar_end = args.end or np.datetime_as_string(panel.dates.max(), unit="D")
    slots = bridge_altdata_feature_slots(
        bar_dates=panel.dates, start=args.start, end=bar_end, catalog=catalog,
        max_slots_per_source=args.max_slots_per_source)
    catalog.close()

    if not slots:
        log.error("NO alt-data feature slots bridged — set FRED_API_KEY / SEC_EDGAR_UA in %s (the "
                  "bridge degrades a missing credential to a per-source skip). Nothing to measure.",
                  ROOT / ".env")
        return 3
    import dataclasses
    panel = dataclasses.replace(panel, feature_slots={**panel.feature_slots, **slots})
    log.info("bridged %d alt-data feature slots: %s", len(slots), sorted(slots))

    # 2) base book (the overlay multiplier), built the SAME way the real generation tick builds it
    base = production_base_sleeves(panel, hold_horizon=hold_horizon, cost_bps=cost_bps,
                                   start=args.start, end=args.end)
    base_book = _combined_book(base, _panel_ts(panel), _fit)

    # 3) overlay pool → return streams → diversity
    returns, sources, culled = build_overlay_pool(
        panel, base_book, cost_bps=cost_bps, max_proposals=args.max_proposals)
    log.info("overlay pool: %d live candidates (%d culled as degenerate)", len(returns), len(culled))
    if len(returns) < 2:
        log.error("only %d live overlay candidate(s) — a pool of <2 has no pairwise diversity to "
                  "measure. Bridge more sources (widen the alt-data connectors).", len(returns))
        return 4

    rep = measure_pool_diversity(returns, max_pairwise_corr=ccfg.max_pairwise_corr, sources=sources)

    # 4) verdict against the pre-registered admission gate (advisory — never touches a gate)
    admits = rep.n_decorrelated_admits
    supply_ok = admits >= ccfg.min_cohort_size
    n_sources = len(set(sources.values()))

    def _f(x: float) -> str:
        return "nan" if not np.isfinite(x) else f"{x:.4f}"

    print("\n" + "=" * 78)
    print("ALT-DATA OVERLAY POOL DIVERSITY  (Doc 3 Part C — 'measure data breadth first')")
    print("=" * 78)
    print(f"  panel window            : {str(panel.dates[0])[:10]} .. {str(panel.dates[-1])[:10]} "
          f"(T={panel.T}, N={panel.N})")
    print(f"  alt-data feature slots  : {len(slots)}  across {n_sources} source/asset-class label(s)")
    print(f"  overlay candidates       : {rep.n_candidates} live  ({len(culled)} culled)")
    print(f"  cohort gates            : max_pairwise_corr={ccfg.max_pairwise_corr}  "
          f"min_cohort_size={ccfg.min_cohort_size}")
    print("-" * 78)
    print(f"  ρ̄_pool  (mean |corr|)   : {_f(rep.mean_abs_pairwise_corr)}   [HEADLINE breadth metric]")
    print(f"  median |corr|           : {_f(rep.median_abs_pairwise_corr)}")
    print(f"  within-source mean|corr|: {_f(rep.mean_within_source_abs_corr)}")
    print(f"  cross-source  mean|corr|: {_f(rep.mean_cross_source_abs_corr)}   "
          f"[breadth should push THIS down]")
    print(f"  frac pairs |corr|<= cap : {_f(rep.frac_pairs_below_cap)}")
    print(f"  # de-correlated admits  : {admits}   (target >= {ccfg.min_cohort_size})")
    print("-" * 78)
    if supply_ok:
        print(f"  VERDICT: SUPPLY OK — {admits} de-correlated admits >= min_cohort_size "
              f"{ccfg.min_cohort_size}.")
        print("           Data breadth was the binding constraint. Phase 4 (wire the cohort")
        print("           evaluator into the live evolve loop) is now justified; the generative")
        print("           proposer (Doc 3 Parts A/B) stays correctly deferred.")
    else:
        print(f"  VERDICT: SUPPLY STILL THIN — only {admits} admits < min_cohort_size "
              f"{ccfg.min_cohort_size}.")
        print("           The proposer/overlay mechanism, not the data, is the binding constraint.")
        print("           Do NOT build Phase 4 speculatively; redirect to Doc 3 Parts A/B")
        print("           (generative QD proposer + novelty pressure).")
    print("  NOTE: every overlay is tanh(z)·base_book, so the shared book multiplier can keep the")
    print("        pool correlated even on orthogonal DATA. If cross-source mean|corr| stays high")
    print("        while sources ARE economically orthogonal, the OVERLAY MECHANISM (not the data)")
    print("        is re-correlating them — a Parts-A/B finding, not a 'bridge more data' one.")
    print("=" * 78 + "\n")

    report = {
        "advisory": "diversity instrument only — issues no verdict and touches no gate",
        "panel": {"start": str(panel.dates[0])[:10], "end": str(panel.dates[-1])[:10],
                  "T": int(panel.T), "N": int(panel.N)},
        "n_alt_slots": len(slots), "alt_slots": sorted(slots),
        "n_sources": n_sources, "sources": sources,
        "n_overlay_live": rep.n_candidates, "n_overlay_culled": len(culled), "culled": culled,
        "gates": {"max_pairwise_corr": ccfg.max_pairwise_corr,
                  "min_cohort_size": ccfg.min_cohort_size},
        "mean_abs_pairwise_corr": rep.mean_abs_pairwise_corr,
        "median_abs_pairwise_corr": rep.median_abs_pairwise_corr,
        "mean_within_source_abs_corr": rep.mean_within_source_abs_corr,
        "mean_cross_source_abs_corr": rep.mean_cross_source_abs_corr,
        "frac_pairs_below_cap": rep.frac_pairs_below_cap,
        "n_decorrelated_admits": admits,
        "supply_ok": bool(supply_ok),
    }
    out_path = out_dir / "pool_diversity_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
