"""Batch-evaluate a registry of candidate signals -> ranked Signal Scorecard.

Usage
-----
    python scripts/research/eval_signals.py --batch demo01 \
        [--registry finrl_pro_ds.signals.library.demo] \
        [--panel synthetic[:T,N] | parquet:PATH | sharadar] \
        [--gates configs/signal_eval.gates.yaml] [--out results/signal_eval/<batch>]

The registry module must expose a module-level ``SIGNALS: list[Signal]``. ``n_trials`` for
the multiple-testing deflation is the number of candidates with a finite IC-IR — the honest
multiple-comparison count. Outputs ``scorecard.{json,md}``.

Design: docs/research/signal_eval_system_design.md.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from finrl_pro_ds.signals import Gates, evaluate_batch, make_synthetic_panel, to_markdown, write_scorecard
from finrl_pro_ds.signals.features import Panel

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATES = ROOT / "configs" / "signal_eval.gates.yaml"


def build_panel(spec: str) -> Panel:
    """Resolve a ``--panel`` spec into a Panel. ``synthetic[:T,N]`` builds a demo panel;
    ``sharadar`` / ``parquet:`` are the real-data paths (P8, require the loader + key)."""
    if spec.startswith("synthetic"):
        t, n = 1500, 60
        if ":" in spec:
            ts, ns = spec.split(":", 1)[1].split(",")
            t, n = int(ts), int(ns)
        return make_synthetic_panel(T=t, N=n, seed=0)
    if spec == "sharadar":
        raise SystemExit("--panel sharadar needs the Sharadar PanelLoader (P8) and "
                         "NASDAQ_DATA_LINK_API_KEY in .env — not yet wired.")
    if spec.startswith("parquet:"):
        raise SystemExit("--panel parquet:<path> needs the PanelLoader parquet path (P8).")
    raise SystemExit(f"unknown --panel spec: {spec!r}")


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description="Evaluate + rank candidate signals.")
    ap.add_argument("--batch", required=True, help="batch name (output subdir)")
    ap.add_argument("--registry", default="finrl_pro_ds.signals.library.demo",
                    help="module exposing SIGNALS: list[Signal]")
    ap.add_argument("--panel", default="synthetic", help="synthetic[:T,N] | parquet:PATH | sharadar")
    ap.add_argument("--gates", default=None, help="gates YAML (default: configs/signal_eval.gates.yaml)")
    ap.add_argument("--out", default=None, help="output dir (default: results/signal_eval/<batch>)")
    args = ap.parse_args(argv)

    gates = Gates.from_yaml(args.gates) if args.gates else Gates.from_yaml(DEFAULT_GATES)
    mod = importlib.import_module(args.registry)
    if not hasattr(mod, "SIGNALS"):
        raise SystemExit(f"registry {args.registry!r} has no module-level SIGNALS list")
    signals = {s.spec.name: s for s in mod.SIGNALS}
    if len(signals) != len(mod.SIGNALS):
        raise SystemExit("duplicate signal names in registry SIGNALS")

    panel = build_panel(args.panel)
    rs = evaluate_batch(signals, panel, gates, batch_name=args.batch)

    out = Path(args.out) if args.out else ROOT / "results" / "signal_eval" / args.batch
    jp, mp = write_scorecard(rs, out)
    print(to_markdown(rs))
    print(f"\nwrote: {jp}\n       {mp}")
    return rs


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # Windows console: render the Unicode table
        except Exception:  # noqa: BLE001
            pass
    main()
