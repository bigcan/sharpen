"""Crucible P2 runner — the manual agentic hypothesis loop (spec §8, P2 row).

Wires the Hypothesis Author (CR-1: sees only ``ledger_agent_view``) → pre-registered overlay/CS
specs → C3 ``evolve`` mine + T0–T5 deflate → DiscoveryCards → run manifest. The default proposer is
the deterministic offline ``LibrarySeedProposer`` (no LLM/network), so the loop reproduces and the
synthetic mode is a pure ``0 PROMISING`` null-safety run (the P2 exit gate). The harness caps at
PROMISING; every card is ``incubation_status=PENDING_P4`` (the CR-8 lockbox is P4) and a deploy read
of any survivor still requires a human-triggered Tier-2 deep audit.

Modes:
  --mode synthetic   end-to-end on a synthetic panel carrying a ``macro:regime`` feature slot (so
                     BOTH the cross_sectional and overlay paths run). Expect 0 PROMISING on noise.
  --mode real        cross-asset ETF panel + production base sleeves (as ``generate_alphas``); the
                     Author additionally proposes overlays on any feature slots the panel carries.

Usage:
  python scripts/research/crucible_hypothesis_loop.py --mode synthetic
  python scripts/research/crucible_hypothesis_loop.py --mode real --start 2008-01-01
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finrl_pro_ds.crucible import CRUCIBLE_VERSION, DataCatalog, TrialLedger, gates_hash  # noqa: E402
from finrl_pro_ds.crucible.agentic import (  # noqa: E402
    HypothesisAuthor,
    LibrarySeedProposer,
    LlmProposer,
    run_hypothesis_loop,
)
from finrl_pro_ds.signals.features import Panel  # noqa: E402
from finrl_pro_ds.signals.generation.config import (  # noqa: E402
    load_generation_config,
    load_generation_meta,
)
from finrl_pro_ds.signals.generation.grammar import available_terminals  # noqa: E402

log = logging.getLogger("crucible_hypothesis_loop")
DEFAULT_GATES = ROOT / "configs" / "signal_eval.gates.yaml"


def _panel_ts(panel: Panel) -> np.ndarray:
    return panel.dates.astype("datetime64[s]").astype(np.int64).astype(np.float64)


def _synthetic_panel(t: int, n: int, *, seed: int = 0) -> Panel:
    """A NOISE OHLCV panel carrying a synthetic ``macro:regime`` feature slot (drives the overlay
    path). No planted edge — the point is 0 PROMISING (the filter is what P2 validates)."""
    rng = np.random.default_rng(seed)
    base = np.cumsum(0.01 * rng.standard_normal((t, n)), axis=0)
    close = np.exp(base + rng.uniform(3.0, 5.0, size=n))
    open_ = close * (1 + 0.001 * rng.standard_normal((t, n)))
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    vol = rng.uniform(1e6, 1e8, (t, n))
    dates = (np.datetime64("2010-01-04") + np.arange(t) * np.timedelta64(1, "D")
             ).astype("datetime64[ns]")
    regime = (np.sin(2.0 * np.pi * np.arange(t) / 80.0) + 0.2 * rng.standard_normal(t)
              ).astype(np.float64)
    return Panel(dates, tuple(f"S{i:02d}" for i in range(n)), open_, high, low, close, vol,
                 np.ones((t, n), bool), close * vol, rng.integers(0, 4, size=n),
                 {"survivorship_free": True, "source": "synthetic_noise"},
                 feature_slots={"macro:regime": regime})


def _proxy_base_sleeves(panel: Panel, hold: int = 21) -> dict[str, np.ndarray]:
    """Inline TSMOM + reversal proxy books (stand-in so the loop runs end-to-end on synthetic)."""
    from finrl_pro_ds.signals.eval_harness import _ls_weights

    c = panel.close
    fwd1 = panel.forward_returns(1)

    def book(score: np.ndarray) -> np.ndarray:
        out = np.full(panel.T, np.nan)
        w = np.zeros(panel.N)
        for t in range(panel.T - 1):
            if t % hold == 0:
                w = _ls_weights(score[t], panel.active[t], min_names=6)
            out[t] = float(np.nansum(w * fwd1[t]))
        return np.nan_to_num(out)

    mom = np.full_like(c, np.nan)
    mom[252:] = c[252:] / c[:-252] - 1.0
    rev = np.full_like(c, np.nan)
    rev[21:] = -(c[21:] / c[:-21] - 1.0)
    return {"tsmom": book(mom), "rates_carry": book(rev)}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="Crucible P2 — manual agentic hypothesis loop")
    ap.add_argument("--config", default=str(DEFAULT_GATES))
    ap.add_argument("--mode", choices=("synthetic", "real"), default="synthetic")
    ap.add_argument("--start", default="2008-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--t", type=int, default=900)
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--max-proposals", type=int, default=32)
    ap.add_argument("--proposer", choices=("library", "llm"), default="library",
                    help="library = deterministic offline seed bank (default, no cost/network). "
                         "llm = live LLM-backed proposer (spec Part A2); needs ANTHROPIC_API_KEY, "
                         "opt-in only — never the default.")
    ap.add_argument("--proposal-ts", default=None,
                    help="ISO proposal timestamp (CR-2/CR-8); defaults to now (UTC).")
    ap.add_argument("--out", default=str(ROOT / "results" / "crucible"))
    ap.add_argument("--force", action="store_true",
                    help="run even when generation.enabled is false in the gates")
    args = ap.parse_args()
    if args.proposer == "llm" and not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("--proposer llm requires ANTHROPIC_API_KEY in the environment (fails closed) — "
                  "not set, aborting before the (possibly slow) panel load.")
        return 1

    cfg, ek = load_generation_config(args.config)
    meta = load_generation_meta(args.config)
    if not meta["enabled"] and not args.force:
        log.warning("generation.enabled is false in %s — no-op. Pass --force to run anyway.",
                    args.config)
        return 0

    if args.mode == "synthetic":
        panel = _synthetic_panel(args.t, args.n)
        base = _proxy_base_sleeves(panel, hold=ek["hold_horizon"])
        panel_key = "synthetic"
    elif meta["panel"] == "taiwan":
        from finrl_pro_ds.data.taiwan_panel_loader import load_taiwan_panel
        from finrl_pro_ds.signals.generation.base_sleeves import taiwan_base_sleeves
        panel = load_taiwan_panel(args.start, args.end)
        base = taiwan_base_sleeves(panel, hold_horizon=ek["hold_horizon"], cost_bps=ek["cost_bps"],
                                   start=args.start, end=args.end)
        panel_key = "taiwan"
    else:
        from finrl_pro_ds.data.cross_asset_panel_loader import load_cross_asset_panel
        from finrl_pro_ds.signals.generation.base_sleeves import production_base_sleeves
        panel = load_cross_asset_panel(args.start, args.end,
                                       config_path=ROOT / "configs" / "cross_asset_momentum.yaml")
        base = production_base_sleeves(panel, hold_horizon=ek["hold_horizon"],
                                       cost_bps=ek["cost_bps"], start=args.start, end=args.end)
        panel_key = meta["panel"]
    ts = _panel_ts(panel)

    out_dir = Path(args.out) / panel_key
    out_dir.mkdir(parents=True, exist_ok=True)
    ghash = gates_hash(args.config)
    proposal_ts = args.proposal_ts or datetime.now(timezone.utc).isoformat()

    ledger = TrialLedger(out_dir / "trial_ledger.db")
    catalog = DataCatalog(out_dir / "catalog.db")
    asset_classes = tuple(sorted({r["asset_class"] for r in catalog.list_series()}))
    proposer = LlmProposer() if args.proposer == "llm" else LibrarySeedProposer()
    author = HypothesisAuthor(proposer, ledger, max_proposals=args.max_proposals)

    # Propose OUTSIDE run_hypothesis_loop (mirrors the P3 orchestrator's own pattern) so (a) the
    # proposer's real post-call token usage is available BEFORE we need to pass token_cost= — reading
    # it any other way would race the call that populates it — and (b) pre_proposed= below stops the
    # loop from calling propose() a second time, which would double an LLM proposer's real API cost
    # (CR-7). Byte-identical to the previous behavior for the default `library` proposer: these are
    # the exact same terminals/context/propose calls run_hypothesis_loop made internally.
    terminals = available_terminals(panel)
    context = author.build_context(terminals, asset_classes=asset_classes)
    specs = author.propose(context, proposal_ts=proposal_ts)

    run_id = f"hyp-{panel_key}-{ghash}"
    result = run_hypothesis_loop(
        panel=panel, base_returns=base, timestamps=ts, cfg=cfg, evolve_kwargs=ek, author=author,
        run_id=run_id, crucible_version=CRUCIBLE_VERSION, gates_hash=ghash,
        proposal_ts=proposal_ts, catalog_asset_classes=asset_classes,
        data_snapshot_hash=catalog.snapshot_hash(), pre_proposed=specs,
        token_cost=getattr(proposer, "last_usage", {}).get("total_tokens", 0))

    cards_dir = out_dir / "cards"
    for card in result.cards:
        card.write(cards_dir)
    result.manifest.write(out_dir)
    summary = {
        "advisory": "harness caps at PROMISING; deploy read requires a Tier-2 deep audit",
        "mode": args.mode, "panel": panel_key, "run_id": run_id,
        "n_preregistered": len(result.specs), "dropped_pre_compute": result.dropped,
        "n_promising": result.n_promising,
        "agent_model_id": result.manifest.agent_model_id, "token_cost": result.manifest.token_cost,
        "file_drawer_N_before": result.manifest.file_drawer_N_before,
        "file_drawer_N_after": result.manifest.file_drawer_N_after,
        "cards": [c.candidate_hash for c in result.cards],
    }
    (out_dir / "loop_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ledger.close()
    catalog.close()

    log.info("mode=%s  preregistered=%d  dropped=%d  PROMISING=%d  -> %s  (manifest %s, gates %s)",
             args.mode, len(result.specs), result.dropped, result.n_promising, out_dir,
             result.manifest.content_hash(), ghash)
    if not result.cards:
        log.info("0 PROMISING (expected on noise / efficient cells — the filter is the point).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
