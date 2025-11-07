"""Point-in-time (PIT) validator for feature CSVs.

Checks that feature columns are strictly trailing using shift(1) by recomputing
basic rolling stats and verifying column equality at t to estimations from t-1.

Usage expects a CSV with a time index column and raw columns:
  - close (required) and optional: open, high, low, volume
  - feature columns names provided via --features

Outputs a JSON summary with per-feature PIT pass/fail and a failure count.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        return [row for row in rdr]


def _rolling_mean(vals: List[float], window: int) -> List[float]:
    out: List[float] = []
    s = 0.0
    for i, v in enumerate(vals):
        s += v
        if i >= window:
            s -= vals[i - window]
        if i + 1 >= window:
            out.append(s / window)
        else:
            out.append(0.0)
    return out


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="finrl_pro.eval.pit_validator")
    ap.add_argument("--csv", required=True, help="Path to features CSV")
    ap.add_argument("--features", nargs="+", required=True, help="Feature columns to validate as trailing")
    ap.add_argument("--price-col", default="close", help="Base price column (default: close)")
    ap.add_argument("--window", type=int, default=20, help="Window size for basic rolling estimator")
    ap.add_argument("--out-json", default=None, help="Optional output JSON path")
    args = ap.parse_args(argv or None)

    rows = _read_csv(Path(args.csv))
    if not rows:
        raise RuntimeError("Empty CSV")
    # Build series
    try:
        price = [float(r[args.price_col]) for r in rows]
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Price column '{args.price_col}' missing or invalid") from e
    estimate = _rolling_mean(price, int(args.window))
    # shift(1): align estimator to t-1 by dropping last and prepending 0
    shifted = [0.0] + estimate[:-1]

    summary: Dict[str, Dict[str, float | int | bool]] = {}
    for feat in args.features:
        diffs = 0
        for i, r in enumerate(rows):
            try:
                v = float(r[feat])
            except Exception:
                continue
            if abs(v - shifted[i]) > 1e-9:
                diffs += 1
        summary[feat] = {"pit_pass": diffs == 0, "violations": diffs}

    payload = {"window": int(args.window), "price_col": args.price_col, "features": summary}
    blob = json.dumps(payload, indent=2, sort_keys=True)
    if args.out_json:
        Path(args.out_json).write_text(blob, encoding="utf-8")
    else:
        print(blob)


if __name__ == "__main__":  # pragma: no cover
    main()

