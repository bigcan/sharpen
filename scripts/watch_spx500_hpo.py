"""Emit one line per newly COMPLETED (or failed) gmgp1-spx500 HPO trial.

Written for the Monitor tool: stdout is the event stream, so this prints only on state
CHANGE rather than dumping the study every poll. Covers failure states as well as
completions — a monitor that greps only for successes is silent through a crashloop, which
is indistinguishable from "still running".

Exits once both variants reach --target trials, so the watch ends on its own.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import optuna
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
STUDIES = {"A/long-short": "gmgp1_spx500_ls_hpo_20260815",
           "B/long-only": "gmgp1_spx500_lo_hpo_20260815"}


def main() -> int:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=30)
    ap.add_argument("--poll", type=int, default=180)
    args = ap.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = optuna.storages.RDBStorage(os.environ["DISTRIBUTED_HPO_DB_URL"])
    seen: dict[str, set[int]] = {k: set() for k in STUDIES}

    while True:
        done_all = True
        for label, name in STUDIES.items():
            # RETRY BEFORE REPORTING. The remote Postgres returns the occasional
            # StorageInternalError on a freshly-opened connection; worker logs show zero
            # storage errors during the same window, so it is this poller's connection and
            # not the training runs. Reporting every blip over a 17h run trains the reader
            # to ignore the monitor, which is exactly when a real failure gets missed.
            st = None
            for attempt in range(3):
                try:
                    st = optuna.load_study(study_name=name, storage=storage)
                    break
                except Exception as e:
                    last = e
                    time.sleep(5 * (attempt + 1))
            if st is None:
                print(f"[{label}] study unavailable after 3 tries: {type(last).__name__}",
                      flush=True)
                done_all = False
                continue
            finished = [t for t in st.trials
                        if t.state in (optuna.trial.TrialState.COMPLETE,
                                       optuna.trial.TrialState.FAIL,
                                       optuna.trial.TrialState.PRUNED)]
            for t in finished:
                if t.number in seen[label]:
                    continue
                seen[label].add(t.number)
                if t.state == optuna.trial.TrialState.COMPLETE:
                    best = max((x.value for x in st.trials
                                if x.value is not None), default=float("nan"))
                    print(f"[{label}] trial {t.number} COMPLETE PF={t.value:.4f} "
                          f"| best={best:.4f} | {len(seen[label])}/{args.target}", flush=True)
                else:
                    print(f"[{label}] trial {t.number} {t.state.name} "
                          f"| {len(seen[label])}/{args.target}", flush=True)
            if len(seen[label]) < args.target:
                done_all = False
        if done_all:
            print("ALL HPO TRIALS FINISHED for both variants", flush=True)
            return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
