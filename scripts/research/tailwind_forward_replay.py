"""TAILWIND forward-runner acceptance replay (Tier-2 roadmap X2, 2026-09-29).

X2's acceptance test is a growing-window check. For every session ``s`` of the 2006-2026 panel,
the forward runner decides the next session's weights from prices truncated at ``s``: every
signal is recomputed from those prices, and the next two calendar sessions are appended. Each
decision is then compared with the one-shot batch over the whole panel. Pass criteria:
- the forward and batch weights agree to L1 < ``max_weight_l1_drift`` (0.05) at EVERY step;
- at every month-end, the forward book switches to the new month's conviction on the same
  session as the batch (the fill date), which must be the month-end session itself under
  ``execution.decision_lead_bars: 1``.

The same logged decisions are then booked session by session through ``ForwardRunner`` (the
persisted book, target log, fills and kills), and the book is scored with the runner's own
verdict. Decisions are computed in parallel: each is independent, since it reads only
prices ``<= s``. Booking is sequential.

Reads only local caches, through ``read_cached_ohlcv``, which never fetches. Writes the artifact
``results/tailwind_v1/forward_replay_<config>.json`` and the replay book under
``results/forward/replay_<config>/``. Run:

    python scripts/research/tailwind_forward_replay.py --config configs/tailwind_v1.yaml --workers 14
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sharpen.data import trading_calendar as tc  # noqa: E402
from sharpen.data import treasury_curve_loader as tcl  # noqa: E402
from sharpen.data.cross_asset_loader import (  # noqa: E402
    _needs_curve,
    build_two_sleeve_arrays,
    prepare_two_sleeve_payload,
    read_cached_ohlcv,
)
from sharpen.envs.allocator_factory import decision_lead_bars, monthly_rebal_conviction  # noqa: E402
from sharpen.paper.forward_runner import (  # noqa: E402
    SESSIONS_AHEAD,
    ForwardRunner,
    _session_dates,
    compute_forward_decision,
    extend_with_next_sessions,
    exit_code,
    target_record,
    truncate_to_session,
)
from sharpen.paper.two_sleeve import TwoSleeveExecutor  # noqa: E402

log = logging.getLogger("tailwind_forward_replay")
_G: tuple = ()


def _quiet() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*fill_method.*")
    logging.getLogger("cross_asset_loader").setLevel(logging.ERROR)
    logging.getLogger("sharpen.paper.paper_state").setLevel(logging.WARNING)


def _init(cfg, close, volume, curve, curve_manifest) -> None:
    global _G
    _quiet()
    _G = (cfg, close, volume, curve, curve_manifest)


def _decide(as_of: str) -> dict:
    cfg, close, volume, curve, cm = _G
    d = compute_forward_decision(cfg, close, volume, as_of=as_of, curve=curve,
                                 curve_manifest=cm, assert_causal=False)
    return target_record(d, cfg)


def _batch_held_conviction(cfg, close, volume, curve, cm) -> tuple[pd.DatetimeIndex, dict]:
    """The batch's held conviction per step (the conviction each step's fill trades), per
    sleeve, on the full panel extended by the next calendar sessions."""
    c, v = truncate_to_session(close, volume, close.index[-1])
    c_ext, v_ext, _ = extend_with_next_sessions(c, v)
    payload = prepare_two_sleeve_payload(cfg, c_ext, v_ext, manifest={}, curve=curve,
                                         curve_manifest=dict(cm) if cm else None,
                                         assert_causal=False)
    bundle = build_two_sleeve_arrays(payload, c_ext.index[0], c_ext.index[-1])
    lead = decision_lead_bars(cfg)
    held = {}
    for s in TwoSleeveExecutor(cfg).sleeve_names:
        cm_s = monthly_rebal_conviction(bundle[s]["timestamps"], bundle[s]["conviction_ary"])
        held[s] = cm_s[lead: lead + len(cm_s) - 1]          # row k -> the step filling at k+1
    return _session_dates(bundle["union"]["timestamps"]), held


def _switch_session(convs: list[np.ndarray], sessions: list[pd.Timestamp]) -> pd.Timestamp | None:
    for j in range(1, len(convs)):
        if not np.array_equal(convs[j], convs[j - 1]):
            return sessions[j]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/tailwind_v1.yaml")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--start", default=None, help="first as-of session (default: panel start)")
    ap.add_argument("--end", default=None, help="last as-of session (default: panel end - 1)")
    ap.add_argument("--reuse-decisions", action="store_true",
                    help="reuse the saved decisions of a previous run with the same window")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _quiet()

    cfg_path = ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    gates_path = ROOT / cfg["ensemble"]["gates_file"] if "ensemble" in cfg and cfg["ensemble"].get(
        "gates_file") else cfg_path.with_suffix(".gates.yaml")
    gates = yaml.safe_load(gates_path.read_text(encoding="utf-8"))
    cache_dir = ROOT / cfg["data"]["cache_dir"]
    wide, manifest = read_cached_ohlcv(cache_dir, cfg["universe"]["assets"])
    close, volume = wide["close"], wide["volume"]
    curve = cm = None
    if _needs_curve(cfg):
        if not tcl.RESEARCH_CURVE_CACHE.exists():
            raise FileNotFoundError(f"research curve cache missing: {tcl.RESEARCH_CURVE_CACHE}")
        curve, cm = tcl.load_treasury_curve_with_manifest()

    sessions = close.index
    start = pd.Timestamp(args.start) if args.start else sessions[1]
    end = pd.Timestamp(args.end) if args.end else sessions[-2]
    as_ofs = [s for s in sessions if start <= s <= end]
    t0 = time.time()
    final = compute_forward_decision(cfg, close, volume, as_of=sessions[-1], curve=curve,
                                     curve_manifest=cm, assert_causal=True)
    log.info("batch reference built in %.1fs (tripwires on); %d forward decisions to compute",
             time.time() - t0, len(as_ofs))

    t0 = time.time()
    saved = ROOT / "results" / "forward" / f"replay_{cfg_path.stem}_decisions.jsonl"
    records: dict[str, dict] = {}
    if args.reuse_decisions and saved.exists():
        for line in saved.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            records[rec["as_of"]] = rec
        missing = [s for s in as_ofs if s.date().isoformat() not in records]
        if missing:
            raise SystemExit(f"--reuse-decisions: {len(missing)} as-of sessions missing from {saved}")
        records = {s.date().isoformat(): records[s.date().isoformat()] for s in as_ofs}
        log.info("reused %d saved decisions from %s", len(records), saved.name)
    else:
        with cf.ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                                    initargs=(cfg, close, volume, curve, cm)) as pool:
            for n, rec in enumerate(pool.map(_decide, [s.date().isoformat() for s in as_ofs],
                                             chunksize=8), 1):
                records[rec["as_of"]] = rec
                if n % 500 == 0:
                    log.info("  %d/%d decisions (%.0fs)", n, len(as_ofs), time.time() - t0)
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_text("".join(json.dumps(r) + "\n" for r in records.values()), encoding="utf-8")
    t_decide = time.time() - t0

    # ---- 1. forward vs batch weights, every step --------------------------------------------
    dates = _session_dates(final.union["timestamps"])
    l1, rows = [], []
    for rec in records.values():
        i = int(dates.get_loc(pd.Timestamp(rec["fill_session"])))
        gap = float(np.abs(np.asarray(rec["weights"]) - final.batch_weights[i - 1]).sum())
        l1.append(gap)
        rows.append((rec["fill_session"], rec["month_end_fill"], gap))
    l1 = np.asarray(l1)
    gate = float(gates["paper_soak"]["parity"]["max_weight_l1_drift"])

    # ---- 2. the fill date of every month-end conviction switch ------------------------------
    b_dates, b_held = _batch_held_conviction(cfg, close, volume, curve, cm)
    by_fill = {r["fill_session"]: r for r in records.values()}
    month_ends = [f for f, me, _ in rows if me]
    agree, on_month_end, compared = 0, 0, 0
    disagreements = []
    for f in month_ends:
        f = pd.Timestamp(f)
        win = [s for s in tc.sessions(f - pd.Timedelta(days=10), f + pd.Timedelta(days=10))]
        win = [s for s in win if s.date().isoformat() in by_fill and s in b_dates]
        k = win.index(f)
        win = win[max(0, k - 2): k + 3]
        for s in TwoSleeveExecutor(cfg).sleeve_names:
            live = [np.asarray(by_fill[x.date().isoformat()]["sleeve_conviction"][s]) for x in win]
            bat = [np.asarray(b_held[s][int(b_dates.get_loc(x)) - 1]) for x in win]
            sw_live, sw_bat = _switch_session(live, win), _switch_session(bat, win)
            if sw_bat is None:
                continue                         # no conviction change this month-end
            compared += 1
            agree += int(sw_live == sw_bat)
            on_month_end += int(sw_bat == f)
            if sw_live != sw_bat:
                disagreements.append({"month_end": f.date().isoformat(), "sleeve": s,
                                      "live": None if sw_live is None else sw_live.date().isoformat(),
                                      "batch": sw_bat.date().isoformat()})

    # ---- 3. book the same decisions through the persisted forward runner ---------------------
    state_dir = ROOT / "results" / "forward" / f"replay_{cfg_path.stem}"
    shutil.rmtree(state_dir, ignore_errors=True)
    runner = ForwardRunner(cfg, gates, state_dir)
    t0 = time.time()
    for s in as_ofs:
        runner.step(as_of=s, union=final.union, record=records[s.date().isoformat()])
    verdict = runner.evaluate(union=final.union, batch_weights=final.batch_weights,
                              sleeve_batch=final.sleeve_batch)
    t_book = time.time() - t0
    fills = runner.fills()
    kill = json.loads((state_dir / "runner_state.json").read_text(encoding="utf-8")).get("killed")

    me_l1 = np.asarray([g for _, me, g in rows if me])
    out = {
        "question": "Does the forward runner reproduce the batch executor from truncated data?",
        "config": args.config, "gates": str(gates_path.relative_to(ROOT)).replace("\\", "/"),
        "cache": {"dir": str(cache_dir.relative_to(ROOT)).replace("\\", "/"),
                  "content_sha256_16": manifest.get("content_sha256_16"),
                  "reconstructed": bool(manifest.get("reconstructed"))},
        "sessions_ahead": SESSIONS_AHEAD, "decision_lead_bars": decision_lead_bars(cfg),
        "n_decisions": len(records), "first_fill": rows[0][0] if rows else None,
        "last_fill": rows[-1][0] if rows else None,
        "weights_vs_batch": {
            "gate_max_weight_l1_drift": gate,
            "max_l1_all_steps": float(l1.max()), "mean_l1_all_steps": float(l1.mean()),
            "steps_over_gate": int((l1 > gate).sum()), "steps_nonzero": int((l1 > 0).sum()),
            "n_month_end_fills": int(len(me_l1)),
            "max_l1_month_end_fills": float(me_l1.max()) if len(me_l1) else None,
        },
        "month_end_fill_dates": {
            "sleeve_switches_compared": compared,
            "live_fill_date_equals_batch": agree,
            "batch_switch_on_the_month_end_session": on_month_end,
            "disagreements": disagreements[:20],
        },
        "book": {
            "booked_sessions": len(fills),
            "overall_status": verdict.get("overall_status"),
            "kill": kill,
            "exit_code": exit_code({"kill": kill, "verdict": verdict}),
            "groups": {g: b["status"] for g, b in (verdict.get("groups") or {}).items()},
            "incremental_parity": (verdict.get("forward") or {}).get("incremental_parity"),
            "summary": {k: verdict.get("summary", {}).get(k) for k in (
                "total_return_pct", "max_drawdown_pct", "max_gross_exposure")},
            "state_dir": str(state_dir.relative_to(ROOT)).replace("\\", "/"),
        },
        "timing_s": {"decide_parallel": round(t_decide, 1), "book_sequential": round(t_book, 1),
                     "workers": args.workers},
    }
    out["acceptance"] = {
        "l1_below_gate_every_step": bool(out["weights_vs_batch"]["steps_over_gate"] == 0),
        "month_end_fill_date_match_100pct": bool(compared > 0 and agree == compared),
        "month_end_switch_on_month_end": bool(compared > 0 and on_month_end == compared),
    }
    path = ROOT / "results" / "tailwind_v1" / f"forward_replay_{cfg_path.stem}.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n_decisions", "weights_vs_batch",
                                          "month_end_fill_dates", "book", "acceptance",
                                          "timing_s")}, indent=2, default=str))
    return 0 if all(out["acceptance"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
