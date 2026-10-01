"""Collect and compare the planted-signal ladder arms.

Fetches every ``results/rl_planted_signal/<slots>/<arm>/result.json`` from the fleet (or reads a local
directory with ``--local``), then prints one table and writes ``collated.json``. ``--slots`` picks the
ladder (``slot_*`` raw features, ``std_slot_*`` standardized); each ladder carries its own control:

* each arm's SAC frictionless Sharpe +/- SE, the linear oracle's Sharpe on the SAME test window,
  the SAC/oracle fraction, corr(action, planted signal), exposure, turnover, final alpha;
* the learning signal per rung = corr(action, signal) and Sharpe minus the zero-edge control's,
  in units of the Sharpe SE (the control fixes the false-positive level of the whole measurement);
* alpha / entropy / Q trajectories at 0, 25, 50, 75, 100% of training.

Usage:
    python scripts/research/rl_planted_signal/collate.py --out results/rl_planted_signal/collated
    python scripts/research/rl_planted_signal/collate.py --slots 'std_slot_*' --out results/rl_planted_signal/collated_std
    python scripts/research/rl_planted_signal/collate.py --local <dir-with-slot_*> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

log = logging.getLogger("rl_planted_signal.collate")

REMOTE_ROOT = "/workspace/DeepScalper/results/rl_planted_signal"
INSTANCES = ("gpuhub-1", "gpuhub-2")
RHO = {"ic015": 0.15, "ic005": 0.05, "ic002": 0.02, "ic000": 0.0}


def _fetch_remote(slots: str) -> list[dict]:
    from remote_cmd import remote_cmd
    out = []
    for inst in INSTANCES:
        listing = remote_cmd(f"ls {REMOTE_ROOT}/{slots}/*/result.json 2>/dev/null", instance_name=inst)
        for path in [ln.strip() for ln in listing.splitlines() if ln.strip().endswith("result.json")]:
            raw = remote_cmd(f"cat {path}", timeout=120, instance_name=inst)
            raw = raw[raw.index("{"):raw.rindex("}") + 1]
            d = json.loads(raw)
            d["result"]["instance"], d["result"]["path"] = inst, path
            out.append(d)
    return out


def _read_local(root: Path, slots: str) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob(f"{slots}/*/result.json"))]


def _traj(trace: list[dict]) -> dict:
    if not trace:
        return {}
    idx = [0, len(trace) // 4, len(trace) // 2, 3 * len(trace) // 4, len(trace) - 1]
    return {k: [round(trace[i][k], 4) for i in idx] for k in ("alpha", "entropy", "q1")}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default=None)
    ap.add_argument("--slots", default="slot_*")
    ap.add_argument("--out", default=str(REPO / "results" / "rl_planted_signal" / "collated"))
    args = ap.parse_args()
    docs = _read_local(Path(args.local), args.slots) if args.local else _fetch_remote(args.slots)
    if not docs:
        log.error("no result.json found")
        return 1

    rows = []
    for d in docs:
        r = d["result"]
        rho = RHO.get(Path(r["parquet"]).stem)
        s, o = r["sac"], r["oracle"]
        rows.append({
            "arm": r["arm"], "rho": rho, "overrides": r["overrides"], "vec_sync": r.get("vec_sync"),
            "raw_channel_norm": r.get("raw_channel_norm", "raw"),
            "obs_feat0_std": round(s["obs_feat0_std"], 5) if "obs_feat0_std" in s else None,
            "sac_sharpe": round(s["sharpe_frictionless"], 3), "sharpe_se": round(s["sharpe_se"], 3),
            "oracle_sharpe": round(o["sharpe_frictionless"], 3),
            "frac_of_oracle": (round(r["sac_fraction_of_oracle"], 3)
                               if r.get("sac_fraction_of_oracle") is not None else None),
            "corr_action_signal": round(s["corr_action_signal"], 4),
            "corr_position_signal": round(s["corr_position_signal"], 4),
            "mean_abs_position": round(s["mean_abs_position"], 3),
            "mean_position": round(s["mean_position"], 3), "turnover_per_bar": round(s["turnover_per_bar"], 4),
            "n_test_bars": s["n_bars"], "final_alpha": round(r["final_alpha"], 5),
            "gradient_steps": r["gradient_steps"], "train_hours": round(r["train_seconds"] / 3600, 2),
            "grad_skips": r["grad_skips"], "trajectory": _traj(d.get("trace", [])),
        })
    rows.sort(key=lambda x: (-(x["rho"] or 0), x["arm"]))
    ctrl = next((x for x in rows if x["rho"] == 0.0), None)
    for x in rows:
        if ctrl is not None and x is not ctrl:
            x["sharpe_minus_control_in_se"] = round((x["sac_sharpe"] - ctrl["sac_sharpe"])
                                                    / max(x["sharpe_se"], 1e-9), 2)

    hdr = (f"{'arm':<12}{'rho':>6}{'SAC SR':>9}{'+/-SE':>7}{'oracle':>8}{'frac':>7}{'corr(a,x)':>10}"
           f"{'|pos|':>7}{'mean pos':>9}{'turn':>7}{'alpha':>8}{'vs ctrl':>8}  overrides")
    lines = [hdr, "-" * len(hdr)]
    for x in rows:
        lines.append(f"{x['arm']:<12}{x['rho']!s:>6}{x['sac_sharpe']:>9.2f}{x['sharpe_se']:>7.2f}"
                     f"{x['oracle_sharpe']:>8.2f}{x['frac_of_oracle'] if x['frac_of_oracle'] is not None else '-':>7}"
                     f"{x['corr_action_signal']:>10.3f}{x['mean_abs_position']:>7.2f}{x['mean_position']:>9.2f}"
                     f"{x['turnover_per_bar']:>7.3f}{x['final_alpha']:>8.4f}"
                     f"{x.get('sharpe_minus_control_in_se', '-')!s:>8}  {x['overrides']}")
    lines.append("")
    for x in rows:
        lines.append(f"{x['arm']:<12} alpha {x['trajectory'].get('alpha')}  entropy "
                     f"{x['trajectory'].get('entropy')}  q1 {x['trajectory'].get('q1')}")
    table = "\n".join(lines)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "collated.json").write_text(json.dumps(rows, indent=2))
    (out / "collated.txt").write_text(table + "\n", encoding="utf-8")
    log.info("\n%s", table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
