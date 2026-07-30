"""Batch-evaluate a registry of candidate signals -> ranked Signal Scorecard.

Usage
-----
    python scripts/research/eval_signals.py --batch demo01 \
        [--registry finrl_pro_ds.signals.library.demo] \
        [--panel synthetic[:T,N] | parquet:PATH | sharadar] \
        [--gates configs/signal_eval.gates.yaml] [--out results/signal_eval/<batch>]

The registry module must expose a module-level ``SIGNALS: list[Signal]``. Outputs
``scorecard.{json,md}``.

**Multiplicity (U5).** The deflation count defaults to the batch pool — the number of
candidates with a finite IC-IR — which is honest only when the batch IS the hypothesis set.
Sweeping a library in slices, or re-running across panels, tests more hypotheses than any one
batch shows, so declare the real count with ``--n-hypotheses N`` (a pre-registered set) or
``--multiplicity-ledger PATH --substrate NAME`` (cumulative across runs, deduplicated by spec
hash). Undeclared runs are not blocked; their cards carry a "multiplicity is batch-shaped"
caveat.

Design: docs/research/signal_eval_system_design.md.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # runnable uninstalled, from any CWD

from finrl_pro_ds.signals import (  # noqa: E402  (after sys.path bootstrap)
    Gates,
    HypothesisLedger,
    Multiplicity,
    evaluate_batch,
    load_multiplicity_gates,
    make_synthetic_panel,
    to_markdown,
    write_scorecard,
)
from finrl_pro_ds.signals.features import Panel  # noqa: E402
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
    if spec.startswith("sp500"):
        # sp500 | sp500:2015-01-01 | sp500:2015-01-01:2026-06-01 | sp500:2015-01-01::120
        from finrl_pro_ds.data.equity_panel_loader import load_sp500_panel
        parts = spec.split(":")[1:]
        start = parts[0] if len(parts) > 0 and parts[0] else "2010-01-01"
        end = parts[1] if len(parts) > 1 and parts[1] else None
        max_names = int(parts[2]) if len(parts) > 2 and parts[2] else None
        return load_sp500_panel(start=start, end=end, max_names=max_names)
    if spec == "sharadar":
        raise SystemExit("--panel sharadar needs the Sharadar PanelLoader and "
                         "NASDAQ_DATA_LINK_API_KEY in .env — not yet wired (use --panel sp500).")
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
    ap.add_argument("--n-hypotheses", type=int, default=None,
                    help="U5: pre-registered hypothesis count this batch is a slice of "
                         "(deflation uses max(batch_pool, this))")
    ap.add_argument("--multiplicity-ledger", default=None,
                    help="U5: JSON ledger accumulating distinct hypotheses per substrate "
                         "(default: multiplicity.ledger_path in "
                         "configs/crucible_multiplicity.gates.yaml)")
    ap.add_argument("--substrate", default=None,
                    help="U5: ledger key (default: --batch)")
    ap.add_argument("--prereg", default="",
                    help="U5: pre-registration doc path, recorded as declaration provenance")
    args = ap.parse_args(argv)

    gates = Gates.from_yaml(args.gates) if args.gates else Gates.from_yaml(DEFAULT_GATES)
    mod = importlib.import_module(args.registry)
    if not hasattr(mod, "SIGNALS"):
        raise SystemExit(f"registry {args.registry!r} has no module-level SIGNALS list")
    signals = {s.spec.name: s for s in mod.SIGNALS}
    if len(signals) != len(mod.SIGNALS):
        raise SystemExit("duplicate signal names in registry SIGNALS")

    panel = build_panel(args.panel)

    # U5 multiplicity. Ledger and pre-registered count are composable: the ledger supplies the
    # cumulative count, --n-hypotheses raises it further if the pre-registration was larger
    # than what has actually been recorded so far. Neither can LOWER the deflation — the
    # harness takes max(batch_pool, declared).
    mgates = load_multiplicity_gates()
    ledger_path = args.multiplicity_ledger or mgates.get("ledger_path")
    substrate = args.substrate or args.batch
    mult: Multiplicity | None = None
    if ledger_path:
        lp = Path(ledger_path)
        mult = HypothesisLedger(lp if lp.is_absolute() else ROOT / lp).declare(
            substrate, signals, gates=mgates)
    if args.n_hypotheses is not None:
        n = max(int(args.n_hypotheses), mult.n_hypotheses if mult else 0)
        mult = Multiplicity(n, "preregistered" if mult is None else mult.source, substrate,
                            args.prereg or (mult.provenance if mult else ""),
                            bool(mgates.get("require_declared", False)),
                            mult.ledger_hash if mult else "")
    if mult is not None:
        print(f"multiplicity: n={mult.n_hypotheses} source={mult.source} "
              f"substrate={substrate!r} provenance={mult.provenance or '—'}")

    rs = evaluate_batch(signals, panel, gates, batch_name=args.batch, multiplicity=mult)

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
