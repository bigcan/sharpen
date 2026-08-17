"""Tests for the TAILWIND Stage-4 recent-OOS artifact (audit item X4, finding P8-10).

The artifact's whole value is that it is a HOLDOUT, so its two leak surfaces get executing
negative tripwires:

  * `test_risk_parity_scalars_are_frozen` — the 10%-vol scalars must be fitted on pre-cutoff
    data only. `portfolio_frontier.risk_parity` uses a FULL-SAMPLE constant; if this script
    ever reverts to that, post-cutoff returns would scale themselves. Mutating the fit mask
    to the full sample must fail.
  * `test_baseline_excludes_the_oos_window` — the subperiod baseline must not see post-cutoff
    data, or the holdout helps set the bar it is judged against.

Plus the verdict-bucket boundaries and the two "this number is not what it looks like" guards
(underpowered verdict, compliance window artifact).

No network: every test drives the pure functions with synthetic series.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml as _yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "research"
GATES = ROOT / "configs" / "tailwind_v1_stage4.gates.yaml"


@pytest.fixture(scope="module")
def x4():
    sys.path.insert(0, str(SCRIPTS))
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "tailwind_stage4_recent_oos", SCRIPTS / "tailwind_stage4_recent_oos.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gates():
    return _yaml.safe_load(GATES.read_text(encoding="utf-8"))["stage4_recent_oos"]


def _two_regime_sleeves(cutoff="2026-06-01", n=1200):
    """Pre-cutoff vol 1x, post-cutoff vol 10x. Any scalar fitted on the full sample is
    dragged by the post-cutoff regime; one fitted pre-cutoff only is not."""
    idx = pd.bdate_range("2022-01-03", periods=n)
    cut = pd.Timestamp(cutoff)
    rng = np.random.default_rng(3)
    scale = np.where(idx < cut, 0.005, 0.05)
    a = pd.Series(rng.normal(0.0002, 1.0, n) * scale, index=idx)
    b = pd.Series(rng.normal(0.0002, 1.0, n) * scale, index=idx)
    return a, b, cut


# ---------------------------------------------------------------------------------------
# Leak tripwires
# ---------------------------------------------------------------------------------------
def test_risk_parity_scalars_are_frozen(x4):
    """NEGATIVE TRIPWIRE. Scalars fitted pre-cutoff must be insensitive to a post-cutoff
    regime change. Widening the fit mask to the full sample makes this fail."""
    a, b, cut = _two_regime_sleeves()

    _, scal_full = x4.combined_book_frozen_scalars(a, b, cut)

    # Same pre-cutoff data, post-cutoff returns replaced with a different regime entirely.
    a2, b2 = a.copy(), b.copy()
    post = a2.index >= cut
    a2[post] *= 7.0
    b2[post] *= 7.0
    _, scal_perturbed = x4.combined_book_frozen_scalars(a2, b2, cut)

    assert scal_full["k_momentum"] == pytest.approx(scal_perturbed["k_momentum"]), (
        "risk-parity scalar moved when only POST-cutoff returns changed — the holdout is "
        "scaling itself (LEAK-2)"
    )
    assert scal_full["k_bab"] == pytest.approx(scal_perturbed["k_bab"])
    assert scal_full["fitted_through"] < str(cut.date())


def test_scalars_do_respond_to_pre_cutoff_data(x4):
    """Guards the test above from being vacuous: the scalars must still be real fits."""
    a, b, cut = _two_regime_sleeves()
    _, base = x4.combined_book_frozen_scalars(a, b, cut)

    a2 = a.copy()
    a2[a2.index < cut] *= 4.0
    _, moved = x4.combined_book_frozen_scalars(a2, b, cut)

    assert moved["k_momentum"] < base["k_momentum"] / 2, (
        "scalar ignored a 4x change in PRE-cutoff vol — it is not fitting anything"
    )


def test_baseline_excludes_the_oos_window(x4, gates):
    """NEGATIVE TRIPWIRE. Post-cutoff returns must not move the baseline."""
    idx = pd.bdate_range("2006-01-03", periods=5200)
    cut = pd.Timestamp("2026-06-01")
    rng = np.random.default_rng(5)
    book = pd.Series(rng.normal(0.0003, 0.006, len(idx)), index=idx)

    base_a = x4.subperiod_baseline(book, gates["subperiods"], cut)

    poisoned = book.copy()
    poisoned[poisoned.index >= cut] += 0.05      # enormous post-cutoff drift
    base_b = x4.subperiod_baseline(poisoned, gates["subperiods"], cut)

    assert base_a["median_pf"] == pytest.approx(base_b["median_pf"]), (
        "baseline moved with post-cutoff data — the holdout is setting its own bar"
    )
    assert base_a["computed_on"] == "pre-cutoff only"


def test_incomplete_trailing_month_is_dropped(x4):
    """P2-01 anti-pattern: a mid-month run must not emit a partial-month rebalance."""
    idx = pd.bdate_range("2006-01-03", end="2026-08-14")       # ends mid-August
    close = pd.DataFrame(100.0, index=idx, columns=list(dict.fromkeys(x4.mom.ALL_TICKERS)))
    rebal = x4.confirmed_month_end_rebalances(close)
    assert rebal[-1].to_period("M") < idx[-1].to_period("M"), (
        "last rebalance sits in the still-open month — partial-month freeze (P2-01)"
    )


# ---------------------------------------------------------------------------------------
# Verdict + honesty guards
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("oos_pf,expected", [
    (1.00, "HOLD"),        # 1.00 >= 0.8 * 1.20
    (0.96, "HOLD"),        # exactly at the hold boundary
    (0.90, "WATCH"),
    (0.72, "WATCH"),       # exactly at the watch boundary
    (0.60, "RETRAIN"),
])
def test_verdict_buckets(x4, gates, oos_pf, expected):
    assert x4.verdict_for(oos_pf, 1.20, gates) == expected


def test_verdict_is_indeterminate_without_inputs(x4, gates):
    assert x4.verdict_for(None, 1.2, gates) == "INDETERMINATE"
    assert x4.verdict_for(1.2, None, gates) == "INDETERMINATE"
    assert x4.verdict_for(1.2, 0.0, gates) == "INDETERMINATE"


def test_compliance_window_artifact_is_flagged_not_reported_as_failure(x4, gates):
    """A monthly book cannot produce 4 active days in 2 rebalances. That must surface as
    INDETERMINATE (`passes is None`), not as a book FAIL."""
    idx = pd.bdate_range("2026-06-01", periods=53)
    daily = pd.Series(0.001, index=idx)
    turnover = pd.Series(0.0, index=idx)
    turnover.iloc[[5, 26]] = 1.0                              # 2 rebalance days

    block = x4.compliance_block(daily, turnover, gates["compliance"], n_rebalances=2)

    assert block["window_sufficient"] is False
    assert block["passes"] is None, "a window artifact must not be reported as a FAIL"
    assert block["day_share_leg_passes"] is True, "the day-share leg IS meaningful here"


def test_compliance_is_decisive_when_the_window_is_long_enough(x4, gates):
    idx = pd.bdate_range("2025-01-01", periods=250)
    daily = pd.Series(0.001, index=idx)
    turnover = pd.Series(0.0, index=idx)
    turnover.iloc[[5, 26, 47, 68, 89, 110]] = 1.0             # 6 rebalances
    block = x4.compliance_block(daily, turnover, gates["compliance"], n_rebalances=6)
    assert block["window_sufficient"] is True
    assert block["passes"] is True


def test_gates_declare_the_artifact_non_deploy_gating(gates):
    """CLAUDE.md forbids reading a deploy-gating OOS verdict without a Tier-2. X4 creates the
    evidence; it must not claim authority."""
    assert gates["deploy_gating"] is False


def test_no_numeric_gate_is_hardcoded_in_the_script():
    """Every threshold must come from the gates file (CLAUDE.md anti-pattern)."""
    src = (SCRIPTS / "tailwind_stage4_recent_oos.py").read_text(encoding="utf-8")
    for literal in ("0.8", "0.6", "0.50", "756", "10000"):
        assert f"= {literal}" not in src, f"threshold {literal} hardcoded in the script"
