"""Tests for the TAILWIND challenge config's Protocol-v2 paper-deploy wiring.

Covers the six `--stage paper-deploy` FAILs closed on 2026-08-17 (risk.static_peak,
safety.kill_file, safety.flatten_on_kill_file, drift.enabled, the 8 v2.2 gates.drift keys and
the 3 gates.safe_mode keys) plus the drift-baseline generator that keeps
`drift.baseline_path` from being a phantom.

The binding test is `test_paper_deploy_stage_has_no_failures` — it runs the real validator, so
it fails if any of the six regress OR if a new paper-deploy rule lands unmet. The per-key tests
below it exist to localize WHICH one broke.

No network and no price cache: the generator's dependencies are monkeypatched with synthetic
series. Includes a negative causality tripwire (removing `.shift(1)` from the trailing-vol
sampler must fail).
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml as _yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "research"
CFG_PATH = ROOT / "configs" / "tailwind_v1_challenge.yaml"

# Mirrors scripts/validate_config.py:_V22_DRIFT_GATE_KEYS / _V22_SAFE_MODE_KEYS. Duplicated
# deliberately: if the validator's contract changes, this test should fail loudly rather than
# silently track it.
V22_DRIFT_KEYS = (
    "window_bars", "min_bars_before_check",
    "deadband_frac_warn", "deadband_frac_crit",
    "saturation_frac_warn", "saturation_frac_crit",
    "action_kl_warn", "action_kl_crit",
)
V22_SAFE_MODE_KEYS = (
    "crit_triggers_flatten", "crit_repeat_window_hours", "crit_repeat_count_before_lockout",
)


@pytest.fixture(scope="module")
def cfg():
    return _yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def baseline_mod():
    """Import the generator by path (scripts/ is not a package)."""
    sys.path.insert(0, str(SCRIPTS))
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "tailwind_drift_baseline", SCRIPTS / "tailwind_drift_baseline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------------------
# The binding end-to-end check
# ---------------------------------------------------------------------------------------
def test_paper_deploy_stage_has_no_failures():
    """`validate_config --stage paper-deploy` must report zero FAIL lines.

    This is the gap-closure assertion. WARNs are tolerated (gates.retrain and retrain_policy
    are still open and are WARN-by-design for this release cycle).
    """
    proc = subprocess.run(
        [sys.executable, "scripts/validate_config.py",
         "--config", "configs/tailwind_v1_challenge.yaml", "--stage", "paper-deploy"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    out = proc.stdout + proc.stderr
    failures = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("FAIL")]
    assert not failures, "paper-deploy FAILs regressed:\n" + "\n".join(failures)
    assert "status=FAIL" not in out


# ---------------------------------------------------------------------------------------
# Per-key localization for the six
# ---------------------------------------------------------------------------------------
def test_static_peak_true_and_no_funded_phase(cfg):
    """Challenge phase grades max-loss statically; trailing peak is funded-only."""
    assert cfg["risk"]["static_peak"] is True
    # A `challenge.phase: funded` block would invert the requirement — this file is step-1/2.
    assert (cfg.get("challenge") or {}).get("phase") != "funded"


def test_kill_path_declared_and_flattens(cfg):
    safety = cfg["safety"]
    assert safety["kill_file"], "engine reads safety.kill_file"
    assert safety["flatten_on_kill_file"] is True, (
        "§8.3 CRIT must close positions; 'halt without close' keeps accruing toward the "
        "daily-loss limit after the operator believes trading stopped (S491)"
    )


def test_drift_enabled_with_resolvable_baseline_path(cfg):
    drift = cfg["drift"]
    assert drift["enabled"] is True
    assert drift["baseline_path"], "prop-firm paper-deploy rejects drift without a baseline"
    # The generator must actually write where the config points.
    src = (SCRIPTS / "tailwind_drift_baseline.py").read_text(encoding="utf-8")
    assert 'BASELINE_PATH = OUT / "drift_baseline.json"' in src
    assert drift["baseline_path"].endswith("results/tailwind_v1/drift_baseline.json")


def test_v22_drift_and_safe_mode_keys_present_inline(cfg):
    """Must be INLINE: check_drift_safemode_gates reads cfg['gates'] and does NOT resolve
    ensemble.gates_file (unlike check_gates_block). Putting these in the .gates file only
    would validate green in review and FAIL in the validator."""
    gates = cfg["gates"]
    assert not set(V22_DRIFT_KEYS) - set(gates["drift"])
    assert not set(V22_SAFE_MODE_KEYS) - set(gates["safe_mode"])


def test_drift_threshold_ordering(cfg):
    d = cfg["gates"]["drift"]
    assert d["deadband_frac_warn"] < d["deadband_frac_crit"]
    assert d["saturation_frac_warn"] < d["saturation_frac_crit"]
    assert d["action_kl_warn"] < d["action_kl_crit"]


def test_drift_windows_sized_for_a_daily_book(cfg):
    """Not the 15-min RL defaults (1000/500), which are ~4y/2y of DAILY bars and could never
    arm inside the 90-day soak the gates file requires."""
    d = cfg["drift"]
    assert cfg["data"]["frequency"] == "1d"
    assert d["window_bars"] == cfg["risk_parity"]["trailing_window"]
    assert d["min_bars_before_check"] == cfg["risk_parity"]["min_periods"]
    assert d["min_bars_before_check"] < 90, "must arm inside min_soak_calendar_days"
    # The inline gates block must not drift from the runtime block.
    assert cfg["gates"]["drift"]["window_bars"] == d["window_bars"]
    assert cfg["gates"]["drift"]["min_bars_before_check"] == d["min_bars_before_check"]


def test_no_numeric_risk_thresholds_duplicated_into_the_config(cfg):
    """Sizing/risk numbers live in the .gates file. Duplicating them here would create a THIRD
    sizing mechanism beside the research basis and the executor path — the defect the
    2026-07-31 sizing reconciliation exists to prevent."""
    leaked = {"max_drawdown_kill_pct", "daily_loss_halt_pct", "max_drawdown_pct",
              "daily_loss_limit_pct", "max_gross_exposure"} & set(cfg["risk"])
    assert not leaked, f"risk thresholds belong in the .gates file, found {leaked}"


# ---------------------------------------------------------------------------------------
# Drift-baseline generator
# ---------------------------------------------------------------------------------------
def _synthetic_frame(n=400, n_assets=3):
    idx = pd.bdate_range("2020-01-01", periods=n)
    cols = [f"A{i}" for i in range(n_assets)]
    rng = np.random.default_rng(0)
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(n, n_assets)), axis=0)),
        index=idx, columns=cols,
    )
    return close, cols


def test_conviction_out_of_range_is_rejected(baseline_mod, monkeypatch):
    """Negative tripwire: passing vol-scaled WEIGHTS (unbounded to lev_cap) instead of
    conviction must raise, not silently produce meaningless deadband/saturation fractions."""
    close, cols = _synthetic_frame()
    rebal = close.index[::21]
    bad = pd.DataFrame(3.0, index=rebal, columns=cols)  # lev_cap-scale weights, not conviction

    monkeypatch.setattr(baseline_mod, "sleeve_convictions",
                        lambda: (rebal, cols, {"momentum": bad}))
    monkeypatch.setattr(baseline_mod, "book_trailing_vol",
                        lambda r: pd.Series(0.1, index=r))

    with pytest.raises(ValueError, match=r"outside \[-1, 1\]"):
        baseline_mod.build_baseline()


def test_baseline_shape_is_per_sleeve(baseline_mod, monkeypatch):
    close, cols = _synthetic_frame()
    rebal = close.index[::21]
    rng = np.random.default_rng(1)
    conv = {
        name: pd.DataFrame(rng.uniform(-1, 1, size=(len(rebal), len(cols))),
                           index=rebal, columns=cols)
        for name in ("momentum", "defensive")
    }
    monkeypatch.setattr(baseline_mod, "sleeve_convictions", lambda: (rebal, cols, conv))
    monkeypatch.setattr(
        baseline_mod, "book_trailing_vol",
        lambda r: pd.Series(np.linspace(0.05, 0.25, len(r)), index=r))

    out = baseline_mod.build_baseline()

    assert set(out["eval_distribution_by_sleeve"]) == {"momentum", "defensive"}
    for block in out["eval_distribution_by_sleeve"].values():
        assert set(block["by_asset"]) == set(cols)
        assert block["n"] == len(rebal)
        # bar_vol + quartile shares were supplied, so regime bucketing must have engaged.
        assert "by_vol_quartile" in block
        assert "regime_bucketing" not in block
    assert out["provenance"]["n_rebalances"] == len(rebal)


def test_trailing_vol_is_causal(baseline_mod, monkeypatch):
    """A return spike at date T must NOT appear in the trailing vol sampled AT T (only from
    T+1). Deleting the `.shift(1)` in book_trailing_vol makes this fail."""
    idx = pd.bdate_range("2020-01-01", periods=300)
    base = pd.Series(0.001, index=idx)
    spiked = base.copy()
    spike_at = idx[250]
    spiked.loc[spike_at] = 0.5

    def _patch(series):
        monkeypatch.setattr(baseline_mod.pf, "build_momentum_net", lambda: series)
        monkeypatch.setattr(baseline_mod.atb, "build_defensive_net", lambda: series)
        monkeypatch.setattr(baseline_mod.pf, "risk_parity",
                            lambda nets: (nets[0], nets[0].index, tuple(nets)))

    rebal = pd.DatetimeIndex([spike_at, idx[251]])

    _patch(base)
    clean = baseline_mod.book_trailing_vol(rebal)
    _patch(spiked)
    dirty = baseline_mod.book_trailing_vol(rebal)

    assert dirty.loc[spike_at] == pytest.approx(clean.loc[spike_at]), (
        "trailing vol AT the spike bar saw the spike — look-ahead (LEAK-2)"
    )
    assert dirty.loc[idx[251]] > clean.loc[idx[251]] * 2, (
        "spike never propagated at all — the test is vacuous, check the fixture"
    )
