"""Golden-file integration test for `scripts/recompute_stage_2_5_verdicts.py`.

Reproduces the pipeline end-to-end on a synthetic but representative trajectory
pair (ensemble vs. best-solo) and asserts the resulting v2.5 verdict file has
the right schema and decision. Pinning the schema means downstream verdict
consumers (audit dashboards, paper-deploy memos) won't silently break when
fields are added.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.recompute_stage_2_5_verdicts import recompute  # noqa: E402


def _make_trajectory(n_bars: int, mean_ret: float, vol: float, seed: int) -> pd.DataFrame:
    """Synthesize a deterministic trajectory with bar-aligned per-bar returns.

    Returns a DataFrame with the columns the recompute tool consumes:
    portfolio_value (for `_per_bar_returns`) + action_agg (for diversity).
    """
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=mean_ret, scale=vol, size=n_bars)
    pv = np.cumprod(1.0 + rets) * 1e5
    return pd.DataFrame({
        "portfolio_value": pv,
        "action_agg": rng.uniform(-1.0, 1.0, size=n_bars),
    })


def _seed_trajectory_pool(test_dir: Path, seeds: list[int]) -> None:
    """Write per-solo + chosen-rule trajectories so the diversity audit fires."""
    for s in seeds:
        traj = _make_trajectory(n_bars=2048, mean_ret=2e-5, vol=4e-4, seed=s)
        traj.to_parquet(test_dir / f"solo_{s}_trajectory.parquet")


def test_recompute_emits_v25_schema_and_promotes_strong_ensemble(tmp_path):
    workstream = "synth_ws"
    out_dir = tmp_path / f"results/{workstream}_ensemble"
    test_dir = out_dir / "test"
    test_dir.mkdir(parents=True)

    seeds = [42, 123, 999]
    chosen_rule = "ens_pf_weighted"

    _seed_trajectory_pool(test_dir, seeds)
    # Ensemble: stronger drift, lower vol → both PF and MDD bootstrap channels
    # should clear 0.90 with high confidence.
    ens_traj = _make_trajectory(n_bars=2048, mean_ret=8e-5, vol=2e-4, seed=7)
    ens_traj.to_parquet(test_dir / f"{chosen_rule}_trajectory.parquet")

    # Stand-in for prior verdict.json (only fields the recompute tool consumes).
    prior_verdict = {
        "workstream": workstream,
        "protocol": "v2.3_stage_2_5_val_selection",
        "val_window": "2025-01-01 -> 2025-06-01",
        "test_window": "2025-06-01 -> 2025-09-01",
        "seeds": seeds,
        "val_pf_by_rule": {chosen_rule: 1.85},
        "chosen_rule": chosen_rule,
        "chosen_rule_val_pf": 1.85,
        "chosen_rule_test_pf": 2.30,
        "test_solo_pfs": {str(s): 1.6 + 0.05 * i for i, s in enumerate(seeds)},
        "best_solo_seed_on_test": 999,
        "best_solo_test_pf": 1.70,
        "uplift": 1.35,
        "legacy_decision": "PROMOTE_ENSEMBLE",
    }
    (out_dir / "verdict.json").write_text(json.dumps(prior_verdict))

    # Standalone gates yaml (canonical v2.5 layout).
    gates_file = tmp_path / "gates.yaml"
    gates_file.write_text(yaml.safe_dump({
        "gates": {
            "ensemble_bootstrap_p_pf_promote": 0.90,
            "ensemble_bootstrap_p_mdd_promote": 0.90,
            "ensemble_bootstrap_p_pf_ambiguous": 0.75,
            "ensemble_bootstrap_resamples": 2000,  # smaller B for fast test
            "ensemble_bootstrap_block_len": None,
            "ensemble_uplift_min": 1.10,
            "ensemble_ambiguous_min": 1.05,
            "ensemble_top_k": 3,
            "ensemble_diversity_lambda": 1.0,
        }
    }))

    # Run on tmp_path so results/<workstream>_ensemble resolves under it.
    cwd_save = Path.cwd()
    try:
        import os
        os.chdir(tmp_path)
        verdict = recompute(
            workstream=workstream,
            gates_file=gates_file,
            legacy_promote=1.10,
            legacy_ambiguous=1.05,
        )
    finally:
        import os
        os.chdir(cwd_save)

    # --- v2.5 schema contract ---
    assert verdict["schema_version"] == "2.5"
    assert verdict["protocol"].startswith("v2.5_stage_2_5_bootstrap_primary")
    assert verdict["primary_basis"] == "bootstrap"
    assert verdict["decision_source"] == "bootstrap_v2.5_recomputed"

    # --- bootstrap block ---
    bs = verdict["bootstrap"]
    assert set(bs) >= {
        "p_pf_ens_better", "p_mdd_ens_better",
        "block_len_used", "resamples", "thresholds_used", "reason",
    }
    assert bs["thresholds_used"]["p_pf_promote"] == 0.90
    assert bs["thresholds_used"]["p_pf_ambiguous"] == 0.75
    # Synthetic data: ens has ~3.7σ stronger drift on lower vol → bootstrap should
    # promote (PF dominance at minimum).
    assert verdict["decision"] in ("PROMOTE", "PROMOTE_DD_ONLY")

    # --- legacy_uplift block (audit-only) ---
    lu = verdict["legacy_uplift"]
    assert set(lu) >= {
        "uplift_ratio", "promote_threshold", "ambiguous_band", "would_have_decided",
    }
    assert lu["uplift_ratio"] == 1.35
    assert lu["ambiguous_band"] == [1.05, 1.10]
    assert lu["would_have_decided"] == "PROMOTE_ENSEMBLE"

    # --- back-compat keys preserved ---
    assert verdict["bootstrap_decision"] == verdict["decision"]
    assert verdict["uplift"] == 1.35

    # --- written file matches in-memory ---
    out_path = tmp_path / f"results/{workstream}_ensemble/verdict_v2_5.json"
    assert out_path.exists(), "v2.5 verdict file not written"
    on_disk = json.loads(out_path.read_text())
    assert on_disk["schema_version"] == "2.5"
    assert on_disk["decision"] == verdict["decision"]
