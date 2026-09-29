"""Daily forward runner for a linear-core paper book (TAILWIND Tier-2 roadmap X2).

Schedule it once per trading day after the NYSE close, for example at 17:30 America/New_York.
Each run does four things:
1. It resolves the as-of session: the latest session whose close plus ``--settle-minutes`` has
   passed, or ``--as-of``.
2. It loads prices through that session.
   - Live mode (default) fetches the union OHLCV and the Treasury curve fresh into the
     runner's OWN data directory, ``<state-dir>/data``. It never writes a certifying cache
     (audit T1-03 / T6-08).
   - ``--offline`` reads the config's cache read-only; use it for a historical as-of.
   Any bar after the as-of session is dropped as in-progress, and data that does not reach the
   as-of session fails the run: no target is logged, so re-run once the vendor has the bar.
3. It runs ``ForwardRunner.run_session`` (``sharpen/paper/forward_runner.py``):
   - decide the next session's weights from data ``<= as-of``;
   - settle the logged target(s) at the as-of close;
   - apply the kills;
   - append the new target to the hash-chained log and persist the book.
4. It scores the book and writes ``<state-dir>/verdict.json``. Scoring is incremental parity
   against today's batch recomputation plus the pre-registered ``paper_soak`` gates. The exit
   code is PASS 0, REVIEW 3 or FAIL 1.

``--loop`` keeps the process alive, runs after every session close and serves ``PaperMetrics``
on ``--metrics-port`` (Prometheus). A one-shot run writes the same values into
``verdict.json``.

    python scripts/run_forward_paper.py --config configs/tailwind_v1.yaml
    python scripts/run_forward_paper.py --config configs/tailwind_v1.yaml --offline --as-of 2026-07-30
    python scripts/run_forward_paper.py --config configs/tailwind_v1_challenge.yaml --loop --metrics-port 9108
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import logging
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sharpen.data import trading_calendar as tc  # noqa: E402
from sharpen.data import treasury_curve_loader as tcl  # noqa: E402
from sharpen.data.cross_asset_loader import (  # noqa: E402
    _needs_curve,
    fetch_and_clean,
    read_cached_ohlcv,
)
from sharpen.paper.forward_runner import (  # noqa: E402
    EXIT_FAIL,
    ForwardRunError,
    ForwardRunner,
    StaleDataError,
    flatten_record,
)
from sharpen.paper.soak_metrics import PaperMetrics, serialize_verdict  # noqa: E402

log = logging.getLogger("run_forward_paper")


def _gates_path(config: dict, config_path: Path, override: str | None) -> Path:
    if override:
        return ROOT / override
    gates_file = (config.get("ensemble") or {}).get("gates_file")
    return ROOT / gates_file if gates_file else config_path.with_suffix(".gates.yaml")


def load_prices(config: dict, *, state_dir: Path, offline: bool):
    """``(close, volume, curve, curve_manifest, provenance)``. Live mode refetches into the
    runner's own data directory; offline mode reads the config's cache without writing."""
    assets = list(config["universe"]["assets"])
    if offline:
        wide, manifest = read_cached_ohlcv(ROOT / config["data"]["cache_dir"], assets)
        curve = cm = None
        if _needs_curve(config):
            if not tcl.RESEARCH_CURVE_CACHE.exists():
                raise FileNotFoundError(f"offline: research curve missing {tcl.RESEARCH_CURVE_CACHE}")
            curve, cm = tcl.load_treasury_curve_with_manifest()
        source = "offline"
    else:
        data_dir = state_dir / "data"
        wide, manifest = fetch_and_clean(
            assets, start=config.get("data", {}).get("start_date", "2006-01-01"), end=None,
            cache_dir=data_dir, force_refetch=True)
        curve = cm = None
        if _needs_curve(config):
            curve, cm = tcl.load_treasury_curve_with_manifest(
                cache_dir=data_dir, research_cache=None, require_fresh=True)
        source = "live"
    if manifest.get("status") == "FAIL":
        raise ForwardRunError(f"OHLCV manifest status FAIL: {manifest.get('stale_scan')}")
    if cm is not None and cm.get("status") == "FAIL":
        raise ForwardRunError("Treasury curve manifest status FAIL")
    prov = {"source": source, "ohlcv_sha256_16": manifest.get("content_sha256_16"),
            "ohlcv_date_max": manifest.get("date_max"),
            "curve_status": None if cm is None else cm.get("status"),
            "curve_date_max": None if cm is None else cm.get("date_max")}
    return wide["close"], wide["volume"], curve, cm, prov


def run_once(args, config: dict, config_path: Path, gates: dict, state_dir: Path,
             metrics: PaperMetrics | None) -> int:
    if args.as_of:
        if not tc.is_session(args.as_of):
            log.error("--as-of %s is not an NYSE session", args.as_of)
            return EXIT_FAIL
        as_of = pd.Timestamp(args.as_of).normalize()
    else:
        as_of = tc.last_complete_session(dt.datetime.now(dt.timezone.utc),
                                         settle_minutes=args.settle_minutes)
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()[:16]
    runner = ForwardRunner(config, gates, state_dir, config_sha16=config_sha)
    try:
        close, volume, curve, cm, prov = load_prices(config, state_dir=state_dir,
                                                     offline=args.offline)
    except Exception as e:                                 # noqa: BLE001 - data unavailable
        # Like stale data: no target can be computed, the next session holds, the job retries.
        # Not an engine error, so no emergency flatten.
        log.error("data unavailable for %s: %s", as_of.date(), e)
        return EXIT_FAIL
    try:
        report = runner.run_session(close, volume, as_of=as_of, curve=curve, curve_manifest=cm)
    except StaleDataError as e:
        log.error("STALE: %s. No target logged; re-run once the vendor has the bar.", e)
        return EXIT_FAIL
    except Exception as e:                                 # noqa: BLE001 - fail closed on any error
        log.error("forward run failed at %s: %s\n%s", as_of.date(), e, traceback.format_exc())
        if dict(config.get("safety") or {}).get("emergency_flatten_on_error"):
            try:
                runner.targets.append(flatten_record(
                    as_of, list(config["universe"]["assets"]),
                    f"emergency_flatten_on_error: {type(e).__name__}", config,
                    config_sha16=config_sha))
                log.error("safety.emergency_flatten_on_error: flatten logged for the next session")
            except Exception as e2:                        # noqa: BLE001
                log.error("emergency flatten could not be logged: %s", e2)
        return EXIT_FAIL
    report["provenance"] = prov
    verdict = report["verdict"]
    serialize_verdict({**verdict, "session_report": {k: v for k, v in report.items()
                                                     if k != "verdict"}},
                      state_dir / "verdict.json")
    log.info("as-of %s: settled %d session(s), logged %s, equity %.2f, status %s, exit %d",
             report["as_of"], len(report["settled"]), report["logged"], report["equity"],
             verdict.get("overall_status"), report["exit_code"])
    if metrics is not None and runner.last_live is not None:
        metrics.update(runner.last_live, runner.last_parity, verdict)
    return int(report["exit_code"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--gates", default=None, help="gates yaml (default: ensemble.gates_file, "
                                                  "else <config>.gates.yaml)")
    ap.add_argument("--state-dir", default=None,
                    help="book + logs (default: results/forward/<config stem>)")
    ap.add_argument("--as-of", default=None, help="session to act on (default: last complete)")
    ap.add_argument("--offline", action="store_true",
                    help="read the config's cache read-only instead of fetching")
    ap.add_argument("--settle-minutes", type=int, default=30,
                    help="minutes after the 16:00 ET close before a session counts as complete")
    ap.add_argument("--loop", action="store_true",
                    help="stay alive: run after every session close and serve PaperMetrics")
    ap.add_argument("--metrics-port", type=int, default=0, help="Prometheus port (0 = off)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config_path = (ROOT / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    gates = yaml.safe_load(_gates_path(config, config_path, args.gates).read_text(encoding="utf-8"))
    state_dir = (ROOT / args.state_dir) if args.state_dir else ROOT / "results" / "forward" / config_path.stem
    state_dir.mkdir(parents=True, exist_ok=True)
    corr_window = int(dict(dict(gates.get("paper_soak") or {}).get("drift") or {}).get(
        "corr_window_days", 252))
    metrics = None
    if args.metrics_port:
        metrics = PaperMetrics(port=args.metrics_port, strategy_name=config_path.stem,
                               corr_window=corr_window)
        metrics.start()

    if not args.loop:
        return run_once(args, config, config_path, gates, state_dir, metrics)
    if args.as_of:
        log.error("--loop and --as-of are exclusive")
        return EXIT_FAIL
    while True:                                            # pragma: no cover - daemon loop
        code = run_once(args, config, config_path, gates, state_dir, metrics)
        now = dt.datetime.now(dt.timezone.utc)
        nxt = tc.next_sessions(tc.last_complete_session(now, settle_minutes=args.settle_minutes), 1)[0]
        wake = dt.datetime.combine(nxt.date(), tc.REGULAR_CLOSE, tc.NYSE_TZ) + dt.timedelta(
            minutes=args.settle_minutes)
        wait = max(60.0, (wake - now).total_seconds())
        if code != 0:
            wait = min(wait, 1800.0)                       # retry a failed run within 30 min
        log.info("next run at %s (exit %d; sleeping %.0fs)", wake.isoformat(), code, wait)
        time.sleep(wait)


if __name__ == "__main__":
    raise SystemExit(main())
