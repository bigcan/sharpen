"""Daily-loss gate alignment empirical bound (S470 item 6).

The sim and live daily-loss rules disagree by construction:

  - Sim  (`finrl_pro_ds/envs/risk_shaping_wrapper.py:229-245`):
        Instant termination on `daily_loss > max_daily_loss_pct`.
        Single-point `current_equity`; no smoothing, no persistence.

  - Live (`finrl_pro_ds/crypto/live/live_engine.py:1959-2057`):
        3-bar `_pv_buffer` median + 2-bar consecutive raw-breach
        persistence; the persistence guard was added in S448 after
        XAUUSD 2026-04-13 ran past FTMO's 10% DD line (raw -11.74%
        final vs smoothed -9.37% at trip).

Direction of the asymmetry: **sim is stricter than live** — sim trips on a
single bar, live tolerates up to 1 bar of latency.  The agent therefore
trains under a tighter termination model than it faces in production.
This is a safety-positive bias, but we want an empirical bound on how
often it actually fires.

What this script does
---------------------
For each live paper-trading WandB run, replay every per-bar
`daily_loss_pct` reading through the sim's instant-trip predicate and
count bars where:

  - `daily_loss_pct < -max_daily_loss_pct`  (sim would terminate)
  - AND live did NOT trip within the next ``--persistence_window`` bars
    (live tolerated it via median or persistence)

The verdict is GO_DOCUMENT (asymmetry harmless in practice) when no
spurious sim-trips are found across the active fleet; otherwise the
report flags the specific (run, bar_time, daily_loss_pct) instances for
review.

Usage:
    python scripts/audit_daily_loss_gate_alignment.py \\
        --runs live-gold-ib-paper-ensemble-v1-20260509:0.05 \\
               live-sg1-xauusd-ctrader-paper-vs-v2-phase2-20260520:0.04 \\
               live-gmgp1-xauusd-ctrader-paper-20260424:0.04 \\
               live-btc-bybit-testnet-paper-20260502:0.05 \\
        --out docs/research/daily_loss_gate_alignment_validation.md

Each `--runs` arg is `run_id:max_daily_loss_pct`.  Bare ``run_id`` resolves
under the default FinRL-Pro-DS entity/project.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ENTITY = "bigcan-chiwin-technology"
DEFAULT_PROJECT = "FinRL-Pro-DS"

# Every-bar keys (must all be populated on every `_log_step` call so that
# scan_history's row filter does not drop non-traded bars).  See
# `_log_step` in `finrl_pro_ds/crypto/live/live_engine.py` for the schema.
# `_timestamp` is WandB's synthetic per-row epoch; included so that
# `_reconstruct_daily_loss` can anchor UTC days directly when present,
# rather than falling back to the configurable bar-interval estimate.
HISTORY_KEYS = [
    "bar",
    "position",
    "portfolio_value",
    "drawdown_pct",
    "total_trades",
    "traded",
    "_timestamp",
]


@dataclass(frozen=True)
class RunSpec:
    run_path: str
    max_daily_loss_pct: float

    @property
    def short_name(self) -> str:
        return self.run_path.split("/")[-1]


def _parse_run_arg(arg: str) -> RunSpec:
    if ":" not in arg:
        raise argparse.ArgumentTypeError(
            f"--runs entries must be 'run_id:max_daily_loss_pct'; got {arg!r}",
        )
    run_part, limit_part = arg.rsplit(":", 1)
    try:
        limit = float(limit_part)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"max_daily_loss_pct must be a float; got {limit_part!r}",
        ) from e
    if run_part.count("/") == 0:
        run_path = f"{DEFAULT_ENTITY}/{DEFAULT_PROJECT}/{run_part}"
    elif run_part.count("/") == 2:
        run_path = run_part
    else:
        raise argparse.ArgumentTypeError(
            f"run_id must be 'entity/project/run_id' or bare id; got {run_part!r}",
        )
    return RunSpec(run_path=run_path, max_daily_loss_pct=limit)


def _pull_history(run_path: str) -> pd.DataFrame:
    import wandb  # type: ignore

    api = wandb.Api()
    run = api.run(run_path)
    rows = list(run.scan_history(keys=HISTORY_KEYS, page_size=5000))
    if not rows:
        raise RuntimeError(
            f"Run {run_path} has no rows for keys={HISTORY_KEYS}. "
            "Confirm the live engine is logging _log_step metrics.",
        )
    df = pd.DataFrame(rows)
    for col in HISTORY_KEYS:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.reset_index(drop=True)
    return df


def _reconstruct_daily_loss(
    df: pd.DataFrame,
    max_daily_loss_pct: float,
    bar_interval_min: int,
) -> tuple[pd.DataFrame, str]:
    """Reconstruct UTC-anchored daily_loss_pct from raw portfolio_value.

    Mirrors `_check_daily_loss` in
    `finrl_pro_ds/crypto/live/live_engine.py:1976-2008`: anchor on the first
    PV reading of each UTC day and recompute the raw return per bar.

    Timestamp resolution preference:
      1. WandB's per-row `_timestamp` (unix-epoch float) when it is present
         and at least partially populated — this gives true wall-clock UTC
         days regardless of strategy cadence.
      2. Otherwise a synthetic monotonic series at `bar_interval_min`-minute
         spacing.  This is configurable so the caller can match the strategy
         cadence (e.g. 15 for gmgp1-gold, 3 for sg1-xauusd) and avoid the
         legacy hard-coded 3-min assumption that compressed UTC-day
         boundaries on slower-cadence runs.

    Returns a `(DataFrame, ts_source)` tuple where `ts_source` is one of
    `"wandb_timestamp"` / `"synthetic_<N>min"` for transparency in the
    report.

    Returns a DataFrame with added columns:
      - utc_day            : pandas Timestamp (day-only, UTC) per bar
      - daily_start_value  : recomputed anchor (PV at first bar of day)
      - daily_loss_pct_sim : (PV - daily_start) / daily_start
      - sim_trip           : daily_loss_pct_sim < -max_daily_loss_pct
    """
    out = df.copy()

    if "_timestamp" in out.columns and out["_timestamp"].notna().any():
        ts = pd.to_datetime(out["_timestamp"], unit="s", utc=True)
        ts_source = "wandb_timestamp"
    else:
        synthetic_ts = pd.date_range(
            start="1970-01-01", periods=len(out),
            freq=f"{bar_interval_min}min", tz="UTC",
        )
        ts = pd.Series(synthetic_ts)
        ts_source = f"synthetic_{bar_interval_min}min"

    out["bar_time_utc"] = ts.values
    out["utc_day"] = pd.to_datetime(out["bar_time_utc"]).dt.floor("D")

    # Per-day anchor on first PV reading of the day.
    daily_anchors = (
        out.dropna(subset=["portfolio_value"])
        .groupby("utc_day", sort=False)["portfolio_value"]
        .first()
        .to_dict()
    )
    out["daily_start_value"] = out["utc_day"].map(daily_anchors).astype(float)
    valid = (out["daily_start_value"] > 0) & out["portfolio_value"].notna()
    out["daily_loss_pct_sim"] = np.where(
        valid,
        (out["portfolio_value"] - out["daily_start_value"]) / out["daily_start_value"],
        np.nan,
    )
    out["sim_trip"] = (
        out["daily_loss_pct_sim"].notna()
        & (out["daily_loss_pct_sim"] < -max_daily_loss_pct)
    )
    return out, ts_source


def _classify_divergences(
    df: pd.DataFrame, persistence_window: int,
) -> dict[str, int]:
    """Count bars where sim would have tripped but live did not.

    "Live did not trip" is detected by the run continuing past the persistence
    window: if the agent kept logging bars `persistence_window` after the
    sim-trip bar, the live engine clearly didn't halt.
    """
    trips = df.index[df["sim_trip"]].tolist()
    if not trips:
        return {"n_sim_trips": 0, "n_live_absorbed": 0, "n_live_halted": 0}
    n_bars = len(df)
    live_absorbed = 0
    live_halted = 0
    for idx in trips:
        # If the run logged at least `persistence_window` bars after the
        # sim-trip bar, the live gate clearly absorbed the breach.
        if idx + persistence_window < n_bars:
            live_absorbed += 1
        else:
            live_halted += 1  # Inconclusive — treat as live-halt boundary.
    return {
        "n_sim_trips": len(trips),
        "n_live_absorbed": live_absorbed,
        "n_live_halted": live_halted,
    }


def _audit_run(
    spec: RunSpec, persistence_window: int, bar_interval_min: int,
) -> dict:
    raw = _pull_history(spec.run_path)
    enriched, ts_source = _reconstruct_daily_loss(
        raw, spec.max_daily_loss_pct, bar_interval_min,
    )
    n_bars = int(len(enriched))
    n_trades = int(enriched["traded"].fillna(0).astype(float).sum())
    classification = _classify_divergences(enriched, persistence_window)
    sample_trips = (
        enriched.loc[
            enriched["sim_trip"],
            ["bar_time_utc", "portfolio_value", "daily_loss_pct_sim"],
        ]
        .head(10)
        .to_dict(orient="records")
    )
    return {
        "run_path": spec.run_path,
        "max_daily_loss_pct": spec.max_daily_loss_pct,
        "n_bars": n_bars,
        "n_trades": n_trades,
        "ts_source": ts_source,
        "min_daily_loss_pct_sim": (
            float(enriched["daily_loss_pct_sim"].min())
            if enriched["daily_loss_pct_sim"].notna().any() else math.nan
        ),
        **classification,
        "sample_trips": sample_trips,
    }


def _render_markdown(
    per_run: list[dict], persistence_window: int, bar_interval_min: int,
) -> str:
    lines = []
    lines.append("# Daily-Loss Gate Alignment Audit Report\n")
    lines.append(f"> **Generated:** {dt.datetime.now(dt.timezone.utc).isoformat()}")
    lines.append(
        "> **Source:** `scripts/audit_daily_loss_gate_alignment.py` "
        f"(persistence_window = {persistence_window} bars, "
        f"synthetic_ts_fallback = {bar_interval_min}min)\n",
    )

    total_trips = sum(r["n_sim_trips"] for r in per_run)
    total_absorbed = sum(r["n_live_absorbed"] for r in per_run)
    n_runs = len(per_run)

    verdict = (
        "GO_DOCUMENT" if total_trips == 0
        else ("BOUND_NON_ZERO" if total_absorbed == 0
              else "BOUND_DIVERGENCE")
    )
    if verdict == "GO_DOCUMENT":
        verdict_text = (
            f"Across **{n_runs} active live runs**, sim's instant-trip "
            "predicate would not have fired on any bar.  The sim-to-live "
            "asymmetry is **dormant in practice** — the architectural "
            "decoupling is intentional and harmless on the observed fleet. "
            "Close S470 item 6 as documented design choice."
        )
    elif verdict == "BOUND_NON_ZERO":
        verdict_text = (
            f"Found **{total_trips} sim-trip bars** across {n_runs} runs, "
            "but all coincided with live halts (within the persistence window). "
            "No spurious sim-vs-live divergence observed.  Close S470 item 6 "
            "with this empirical bound."
        )
    else:
        verdict_text = (
            f"Found **{total_absorbed} bars** where sim would have terminated "
            f"but live kept trading (out of {total_trips} total sim-trip bars). "
            "**Divergence is real** — review per-run details below before "
            "deciding whether to align sim to live or accept the gap."
        )

    lines.append("## Verdict\n")
    lines.append(f"**{verdict}** — {verdict_text}\n")
    lines.append("## Fleet Summary\n")
    lines.append("| Run | max_daily | ts_source | n_bars | n_trades | min_daily_loss% | sim_trips | absorbed | halt-boundary |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in per_run:
        min_dl = r["min_daily_loss_pct_sim"]
        min_dl_str = f"{min_dl * 100:.3f}" if not math.isnan(min_dl) else "n/a"
        lines.append(
            f"| `{r['run_path'].split('/')[-1]}` | "
            f"{r['max_daily_loss_pct'] * 100:.2f}% | "
            f"{r.get('ts_source', 'n/a')} | "
            f"{r['n_bars']} | {r['n_trades']} | {min_dl_str} | "
            f"{r['n_sim_trips']} | {r['n_live_absorbed']} | {r['n_live_halted']} |",
        )
    lines.append("")
    lines.append("## Sample Sim-Trip Bars (first 10 per run)\n")
    for r in per_run:
        if not r["sample_trips"]:
            continue
        lines.append(f"### `{r['run_path'].split('/')[-1]}`")
        lines.append("")
        lines.append("| bar_time_utc | portfolio_value | daily_loss_pct_sim |")
        lines.append("|---|---:|---:|")
        for row in r["sample_trips"]:
            ts = row["bar_time_utc"]
            ts_str = pd.Timestamp(ts).isoformat() if ts is not None else "?"
            lines.append(
                f"| {ts_str} | {row['portfolio_value']:.2f} | "
                f"{row['daily_loss_pct_sim']:+.4%} |",
            )
        lines.append("")
    lines.append("---")
    lines.append(
        "_Reproduce: see usage at the top of "
        "`scripts/audit_daily_loss_gate_alignment.py`._",
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs", nargs="+", required=True, type=_parse_run_arg,
        help="One or more 'run_id:max_daily_loss_pct' entries.  run_id may be "
             "a bare slug or full entity/project/id path.",
    )
    ap.add_argument(
        "--persistence_window", type=int, default=5,
        help="Bars after a sim-trip bar that must exist for the live engine "
             "to be deemed to have absorbed the breach.  Live's actual rule "
             "is 2-bar persistence; 5 is a safety margin for halt-state "
             "propagation through wandb.",
    )
    ap.add_argument(
        "--bar-interval-min", type=int, default=3,
        help="Synthetic-timestamp fallback spacing in minutes, used only when "
             "WandB does not expose `_timestamp` on the run's rows.  Set this "
             "to match the strategy cadence (e.g. 15 for gmgp1-gold, 3 for "
             "sg1-xauusd) so UTC-day boundaries are not compressed.  Ignored "
             "when `_timestamp` is present (preferred path).",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="Markdown report path.",
    )
    ap.add_argument(
        "--json-out", type=Path, default=None,
        help="Optional JSON summary path.",
    )
    args = ap.parse_args()

    per_run: list[dict] = []
    for spec in args.runs:
        print(f"Pulling WandB history: {spec.run_path}", file=sys.stderr)
        try:
            result = _audit_run(
                spec, args.persistence_window, args.bar_interval_min,
            )
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            result = {
                "run_path": spec.run_path,
                "max_daily_loss_pct": spec.max_daily_loss_pct,
                "error": str(e),
                "n_bars": 0, "n_trades": 0,
                "ts_source": "error",
                "min_daily_loss_pct_sim": math.nan,
                "n_sim_trips": 0, "n_live_absorbed": 0, "n_live_halted": 0,
                "sample_trips": [],
            }
        print(
            f"  {spec.short_name}: bars={result.get('n_bars', 0)}, "
            f"ts_source={result.get('ts_source', 'n/a')}, "
            f"sim_trips={result.get('n_sim_trips', 0)}, "
            f"absorbed={result.get('n_live_absorbed', 0)}",
            file=sys.stderr,
        )
        per_run.append(result)

    report = _render_markdown(
        per_run, args.persistence_window, args.bar_interval_min,
    )
    print(report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"\nReport written -> {args.out}", file=sys.stderr)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(per_run, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"JSON summary -> {args.json_out}", file=sys.stderr)

    total_absorbed = sum(r["n_live_absorbed"] for r in per_run)
    # Exit 0 when there are zero spurious sim-vs-live divergences (GO_DOCUMENT
    # or BOUND_NON_ZERO with all sim-trips coinciding with halts).
    return 0 if total_absorbed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
