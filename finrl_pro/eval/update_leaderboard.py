"""CLI: Upsert leaderboard rows from matrix evals.

Reads `reports/matrix/eval_report.json` and `reports/matrix/runs.json`,
computes PSR from each run's `reports/<fingerprint>/returns.csv`, and
inserts or updates rows in `docs/leaderboard.md` for single-asset and
multi-asset tables.

Unknown fields are preserved on update or set to '-' on insert.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from finrl_pro.eval.statistics import probabilistic_sharpe_ratio
from finrl_pro.eval.risk_summary_gate import enforce_turnover_cost_limits


@dataclass(slots=True)
class EvalRow:
    fingerprint: str
    config: str
    is_multi: bool
    sharpe: Optional[float]
    maxdd: Optional[float]
    volatility: Optional[float]
    psr: Optional[float]
    agent: str
    reward: str
    notes: str


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_returns(path: Path) -> list[float]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        out: list[float] = []
        for row in rdr:
            try:
                out.append(float(row.get("return") or row.get("daily_return") or row.get("ret") or ""))
            except Exception:
                continue
    return out


def _derive_agent_and_reward(cfg_path: Path) -> tuple[str, str]:
    agent = "-"
    reward = "-"
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        mv = ((cfg.get("training") or {}).get("module_versions") or {})
        a = str(mv.get("agent", "")).strip()
        if a:
            agent = a
        # Reward hint (optional)
        r = str(((cfg.get("training") or {}).get("reward", ""))).strip()
        if r:
            reward = r
    except Exception:
        pass
    # Fallback to filename hints
    stem = cfg_path.stem.lower()
    if agent == "-":
        for cand in ("ppo", "sac", "a2c", "td3"):
            if cand in stem:
                agent = cand.upper()
                break
    if reward == "-":
        if "reward_logr" in stem or "logr" in stem:
            reward = "logR"
    return agent, reward


def _to_eval_rows(matrix_dir: Path) -> list[EvalRow]:
    runs = _load_json(matrix_dir / "runs.json")
    evals = _load_json(matrix_dir / "eval_report.json")
    eval_by_fp: Dict[str, Dict[str, Any]] = {str(e["fingerprint_id"]): e for e in evals}
    out: list[EvalRow] = []
    for r in runs:
        fp = str(r.get("fingerprint_id"))
        cfg = Path(str(r.get("config")))
        ev = eval_by_fp.get(fp)
        if not ev:
            continue
        metrics = dict(ev.get("evaluated_metrics") or {})
        sharpe = float(metrics.get("sharpe_ratio", 0.0)) if metrics.get("sharpe_ratio") is not None else None
        maxdd = float(metrics.get("max_drawdown", 0.0)) if metrics.get("max_drawdown") is not None else None
        vol = float(metrics.get("volatility", 0.0)) if metrics.get("volatility") is not None else None
        # PSR from artifacts
        rets_csv = Path("reports") / fp / "returns.csv"
        psr: Optional[float]
        try:
            rets = _read_returns(rets_csv)
            psr = float(probabilistic_sharpe_ratio(rets)) if rets else None
        except Exception:
            psr = None
        agent, reward = _derive_agent_and_reward(Path(cfg))
        row = EvalRow(
            fingerprint=fp,
            config=str(cfg.as_posix()),
            is_multi=("multi" in cfg.stem.lower()),
            sharpe=sharpe,
            maxdd=maxdd,
            volatility=vol,
            psr=psr,
            agent=agent,
            reward=reward,
            notes=cfg.stem,
        )
        out.append(row)
    return out


def _find_table_bounds(lines: list[str], heading: str) -> tuple[int, int]:
    """Return start,end indices (inclusive start, exclusive end) for the table under a heading."""
    n = len(lines)
    # Find heading
    hidx = -1
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith(heading.lower()):
            hidx = i
            break
    if hidx == -1:
        return -1, -1
    # Find first header row starting with '|'
    s = -1
    for i in range(hidx + 1, n):
        if lines[i].lstrip().startswith("|"):
            s = i
            break
        if lines[i].startswith("## ") and i > hidx:
            return -1, -1
    if s == -1:
        return -1, -1
    # Find end: first non '|' line or next heading
    e = n
    for i in range(s + 1, n):
        if not lines[i].lstrip().startswith("|"):
            e = i
            break
    return s, e


def _parse_header_cols(header_line: str) -> list[str]:
    return [c.strip() for c in header_line.strip().strip("|").split("|")]


def _row_to_cells(row_line: str) -> list[str]:
    return [c.strip() for c in row_line.strip().strip("|").split("|")]


def _cells_to_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _upsert_rows(table_lines: list[str], rows: list[EvalRow]) -> list[str]:
    if len(table_lines) < 2:
        return table_lines
    header = table_lines[0]
    divider = table_lines[1]
    body = table_lines[2:]
    cols = _parse_header_cols(header)
    col_idx = {name: idx for idx, name in enumerate(cols)}
    fp_idx = col_idx.get("Fingerprint")
    # Prepare a dict from existing rows by fingerprint
    existing: Dict[str, list[str]] = {}
    for ln in body:
        if not ln.strip().startswith("|"):
            continue
        cells = _row_to_cells(ln)
        if fp_idx is not None and fp_idx < len(cells):
            existing[cells[fp_idx]] = cells
    today = date.today().isoformat()

    def fmt_pct(x: Optional[float]) -> str:
        return f"{x * 100:.1f}%" if x is not None else "-"

    def fmt_or_dash(x: Optional[float], places: int = 2) -> str:
        return (f"{x:.{places}f}" if x is not None else "-")

    updated: Dict[str, list[str]] = dict(existing)
    for r in rows:
        # Initialize new row with '-' defaults at correct length
        new_cells = ["-"] * len(cols)
        # Fill known columns if present
        if "Date" in col_idx:
            new_cells[col_idx["Date"]] = today
        if "Agent" in col_idx:
            new_cells[col_idx["Agent"]] = r.agent or "-"
        if "Reward" in col_idx:
            new_cells[col_idx["Reward"]] = r.reward or "-"
        if "Sharpe_test" in col_idx:
            new_cells[col_idx["Sharpe_test"]] = fmt_or_dash(r.sharpe, 2)
        if "PSR_test" in col_idx:
            new_cells[col_idx["PSR_test"]] = fmt_or_dash(r.psr, 2)
        if "MaxDD_test" in col_idx:
            new_cells[col_idx["MaxDD_test"]] = fmt_pct(r.maxdd)
        if "Fingerprint" in col_idx:
            new_cells[col_idx["Fingerprint"]] = r.fingerprint
        if "Notes" in col_idx:
            new_cells[col_idx["Notes"]] = r.notes

        # Merge with existing row if present: preserve non-dash existing values
        if r.fingerprint in existing:
            cur = existing[r.fingerprint]
            merged = cur[:]
            for i in range(min(len(merged), len(new_cells))):
                if new_cells[i] != "-" and new_cells[i] != "":
                    merged[i] = new_cells[i]
            updated[r.fingerprint] = merged
        else:
            updated[r.fingerprint] = new_cells

    # Rebuild table: header + divider + sorted by date desc then fingerprint
    rebuilt = [header, divider]
    # Keep order: existing first preserving original order, then new inserts
    seen_fp: set[str] = set()
    for ln in body:
        if not ln.strip().startswith("|"):
            continue
        cells = _row_to_cells(ln)
        key = cells[fp_idx] if fp_idx is not None and fp_idx < len(cells) else ""
        if key in updated:
            rebuilt.append(_cells_to_row(updated[key]))
            seen_fp.add(key)
    for fp, cells in updated.items():
        if fp not in seen_fp:
            rebuilt.append(_cells_to_row(cells))
    return rebuilt


def _apply_upserts(leaderboard: Path, rows: list[EvalRow]) -> None:
    lines = leaderboard.read_text(encoding="utf-8").splitlines()
    # Single-asset table
    s_start, s_end = _find_table_bounds(lines, "## Table")
    # Multi-asset table
    m_start, m_end = _find_table_bounds(lines, "## Multi-Asset")

    if s_start != -1 and s_end != -1:
        s_table = lines[s_start:s_end]
        s_rows = [r for r in rows if not r.is_multi]
        new_s = _upsert_rows(s_table, s_rows)
        lines = lines[:s_start] + new_s + lines[s_end:]

    if m_start != -1 and m_end != -1:
        # Adjust indices if single table changed sizes
        if s_start != -1 and s_end != -1:
            delta = len(lines[:s_start] + new_s + lines[s_end:]) - len(lines)
            if delta != 0:
                # recompute after previous modification
                lines2 = lines
                m_start, m_end = _find_table_bounds(lines2, "## Multi-Asset")
        m_table = lines[m_start:m_end]
        m_rows = [r for r in rows if r.is_multi]
        new_m = _upsert_rows(m_table, m_rows)
        lines = lines[:m_start] + new_m + lines[m_end:]

    leaderboard.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="finrl_pro.eval.update_leaderboard", description="Update leaderboard from matrix evals")
    ap.add_argument("--matrix-dir", default="reports/matrix", help="Directory with runs.json and eval_report.json")
    ap.add_argument("--leaderboard", default="docs/leaderboard.md", help="Leaderboard Markdown path")
    args = ap.parse_args(argv or None)

    matrix_dir = Path(args.matrix_dir)
    enforce_turnover_cost_limits(matrix_dir)
    rows = _to_eval_rows(matrix_dir)
    _apply_upserts(Path(args.leaderboard), rows)


if __name__ == "__main__":  # pragma: no cover
    main()
