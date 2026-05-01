"""One-shot wrapper that re-uses sg1_xauusd_ensemble_eval.py against the
DEPLOYED WF fold-07 ensemble (seeds 42 / 2025 / 3141, mounted 2026-04-21)
rather than the script's hardcoded L1 multiseed batch-2 set (seeds 42 /
789 / 456).

Q1 2026 OOS regime-drift verification only — patches the module constants
in-process, then calls main(). No permanent script edit.

Usage:
    python scripts/_q1_2026_sg1_xauusd_wf_ensemble_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import sg1_xauusd_ensemble_eval as ee  # noqa: E402

# Override the module-level checkpoint registry to match the live container's
# BIND mounts. Equal weights for ens_pf_weighted (no per-seed PFs available
# for these specific WF ckpts on a uniform Q1-2026 window).
ee.SEED_CHECKPOINTS = {
    42:   "checkpoints/WF_seed42_fold_07_20260421_220234/checkpoint_final.pth",
    2025: "checkpoints/WF_seed2025_fold_07_20260421_214535/checkpoint_final.pth",
    3141: "checkpoints/WF_seed3141_fold_07_20260421_215709/checkpoint_final.pth",
}
ee.SEED_PFS = {42: 1.0, 2025: 1.0, 3141: 1.0}

# Replace RULES — the script hardcodes solo_42/solo_789/solo_456 (L1 deployment).
# WF deployment uses seeds 42/2025/3141 instead.
ee.RULES = [
    ("solo_42",         lambda: ee._agg_solo(42)),
    ("solo_2025",       lambda: ee._agg_solo(2025)),
    ("solo_3141",       lambda: ee._agg_solo(3141)),
    ("ens_mean",        lambda: ee._agg_mean),
    ("ens_median",      lambda: ee._agg_median),
    ("ens_agreement",   lambda: ee._agg_agreement),
    ("ens_pf_weighted", lambda: ee._agg_pf_weighted),
]

# Forward CLI args to the wrapped main() — pass --config, --device, etc.
sys.argv = ["sg1_xauusd_ensemble_eval.py"] + sys.argv[1:]
ee.main()
