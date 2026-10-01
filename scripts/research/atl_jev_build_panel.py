"""Build the ATL x Jev screening price panel (Phase 2 step 6; architecture ADR-3, ADR-8).

    python scripts/research/atl_jev_build_panel.py

Reads the cached yfinance fetch of the S&P 500 point-in-time union, cuts it at ``screening_window[1]``, runs the
project cleaner and the recorded OHLC bracketing (``sharpen.jev.panel``), and saves the panel with its full trading
calendar to ``data/atl_jev/panels/screening.pkl`` plus a manifest. Refuses to save a panel the frozen Tier-0 would
halt on (any OHLC violation).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sharpen.jev.panel import screening_panel  # noqa: E402
from sharpen.jev.stamps import gates_sha, git_commit  # noqa: E402

GATES = ROOT / "configs" / "atl_jev.gates.yaml"
EQ = ROOT / "data" / "raw" / "equity_panel"
RAW = EQ / "_pit_union_2007.pkl"
OUT = ROOT / "data" / "atl_jev" / "panels" / "screening.pkl"

log = logging.getLogger("atl_jev_build_panel")


class _Remap(pickle.Unpickler):
    """The cached panel was pickled before the finrl_pro_ds -> sharpen rename."""

    def find_class(self, module, name):
        if module.startswith("finrl_pro_ds"):
            module = "sharpen" + module[len("finrl_pro_ds"):]
        return super().find_class(module, name)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    phase1 = yaml.safe_load(GATES.read_bytes())["phase1"]
    raw_bytes = RAW.read_bytes()
    with open(RAW, "rb") as fh:
        raw = _Remap(fh).load()
    panel, calendar, report = screening_panel(raw, phase1, EQ / "sp500_constituents.csv")
    if report["ohlc_violations_after"]["total"] != 0:
        raise SystemExit(f"panel still has OHLC violations {report['ohlc_violations_after']}: the frozen Tier-0 "
                         "would halt, so it is not saved")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    blob = pickle.dumps({"panel": panel, "calendar": calendar}, protocol=pickle.HIGHEST_PROTOCOL)
    tmp = OUT.with_name(OUT.name + ".tmp")
    tmp.write_bytes(blob)
    os.replace(tmp, OUT)
    manifest = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source": str(RAW.relative_to(ROOT)), "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "panel_sha256": hashlib.sha256(blob).hexdigest(), "gates_sha": gates_sha(GATES),
                "screening_window": phase1["screening_window"], **git_commit(ROOT), "report": report}
    OUT.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    log.info("screening panel %s: %s", OUT, json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
