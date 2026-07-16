"""The `robustness.min_subperiod_ic_ir` gate (crucible-v4.0, S553-cont-133 F2b).

Two coupled behaviours, tested in the order they must hold:

1. the REPAIR (`tier3_robustness`) — a subperiod that is a coverage HOLE is excluded, not scored,
   and a short-but-populated panel is untouched (the exact-no-op property the repair rests on);
2. the WIRING (`scorecard._finalize`) — the gate every gates YAML has declared since v2.0
   ("no negative subperiod") is finally read, and an UNMEASURABLE robustness is not a pass.

The CRU-1 regression at the bottom pins the recorded small-cap probe verdict: wiring is only
legitimate because it is monotone-stricter and preserves it.
"""
from __future__ import annotations

import numpy as np

from finrl_pro_ds.signals import Gates, SignalSpec
from finrl_pro_ds.signals.eval_harness import (
    Deflation,
    GrossPower,
    HorizonIC,
    HygieneResult,
    Robustness,
    tier3_robustness,
)
from finrl_pro_ds.signals.features import Panel
from finrl_pro_ds.signals.scorecard import SignalScorecard, _finalize


class Trail:
    spec = SignalSpec(name="trail1", hypothesis="trailing return predicts", family="technical",
                      expected_sign=1, horizons=(1, 5), neutralization=())

    def compute(self, panel: Panel) -> np.ndarray:
        c = np.log(panel.close)
        out = np.full(c.shape, np.nan)
        out[1:] = c[1:] - c[:-1]
        return out


def _panel(t: int = 600, n: int = 50, rho: float = 0.45, seed: int = 0,
           active_from: int = 0) -> Panel:
    """Momentum panel; ``active_from`` blanks membership before that row (the Taiwan hole)."""
    rng = np.random.default_rng(seed)
    eps = 0.01 * rng.standard_normal((t, n))
    ret = np.zeros((t, n))
    for i in range(1, t):
        ret[i] = rho * ret[i - 1] + eps[i]
    close = np.exp(np.cumsum(ret, axis=0) + rng.uniform(3.0, 5.0, size=n))
    open_ = close * np.exp(0.0005 * rng.standard_normal((t, n)))
    high = np.maximum(open_, close) * np.exp(np.abs(0.001 * rng.standard_normal((t, n))))
    low = np.minimum(open_, close) * np.exp(-np.abs(0.001 * rng.standard_normal((t, n))))
    vol = rng.uniform(1e5, 1e7, size=(t, n))
    active = np.ones((t, n), bool)
    active[:active_from] = False
    dates = (np.datetime64("2012-01-02")
             + np.arange(t) * np.timedelta64(1, "D")).astype("datetime64[ns]")
    return Panel(dates, tuple(f"M{i:03d}" for i in range(n)), open_, high, low, close, vol,
                 active, close * vol, rng.integers(0, 5, size=n),
                 {"survivorship_free": False, "source": "synthetic"})


def _gates(**over) -> Gates:
    base = {"universe": {"min_names_per_day": 4}, "coverage": {"min_days": 150},
            "gross_power": {"horizons": [1, 5], "primary_horizon": 1}}
    for k, v in over.items():
        base[k] = {**base.get(k, {}), **v}
    return Gates.from_dict(base)


# ---- 1. the repair -----------------------------------------------------------

def test_coverage_hole_subperiod_is_excluded_not_scored() -> None:
    """The Taiwan pathology: membership starts long after the first bar, so subperiod 1 is a
    near-empty window whose IC-IR is noise. It must be dropped, not averaged or gated on."""
    # 600 rows, membership from row 580 => subperiod 1 (rows 0..150) has ZERO valid days and
    # subperiod 4 is only ~20/150 populated: both are holes, not estimates.
    p = _panel(t=600, active_from=580)
    rob = tier3_robustness(Trail(), p, _gates(), neutralization=(), expected_sign=1, horizon=1)

    assert rob.subperiod_valid_days[0] == 0
    assert not np.isfinite(rob.subperiod_ic_ir[0])          # excluded, not scored
    assert not np.isfinite(rob.subperiod_ic_ir[1])
    # every window is a hole => robustness is UNMEASURABLE, and says so rather than inventing a value
    assert not np.isfinite(rob.min_subperiod_ic_ir)
    assert not np.isfinite(rob.mean_subperiod_ic_ir)


def test_hole_cannot_inflate_mean_or_trip_the_gate() -> None:
    """A hole in ONE window must not contaminate the min/mean drawn from the populated ones."""
    p = _panel(t=800, active_from=210)     # subperiod 1 (rows 0..200) empty; 2-4 fully populated
    rob = tier3_robustness(Trail(), p, _gates(), neutralization=(), expected_sign=1, horizon=1)

    assert rob.subperiod_valid_days[0] == 0
    assert not np.isfinite(rob.subperiod_ic_ir[0])
    assert all(np.isfinite(x) for x in rob.subperiod_ic_ir[1:])
    # min/mean are drawn from the populated windows ONLY
    populated = [x for x in rob.subperiod_ic_ir[1:]]
    assert rob.min_subperiod_ic_ir == min(populated)
    assert np.isclose(rob.mean_subperiod_ic_ir, float(np.mean(populated)))


def test_short_but_fully_populated_panel_is_untouched() -> None:
    """The exact-no-op property: the floor is span-aware, so a legitimately SHORT panel whose
    active window matches its date range keeps every subperiod. A flat absolute floor would NaN
    them all and silently block PROMISING on every short substrate."""
    p = _panel(t=500)                      # 125-row subperiods, 100% populated
    rob = tier3_robustness(Trail(), p, _gates(), neutralization=(), expected_sign=1, horizon=1)

    assert all(np.isfinite(x) for x in rob.subperiod_ic_ir)
    assert all(v > 100 for v in rob.subperiod_valid_days)
    assert np.isfinite(rob.min_subperiod_ic_ir)


def test_recent_oos_years_is_read_from_gates() -> None:
    """`recent_oos_years` was the second dead key in the `robustness:` block — declared in every
    gates YAML but never read (the hardcoded default just happened to match)."""
    p = _panel(t=800)
    g2 = _gates(robustness={"recent_oos_years": 1})
    g5 = _gates(robustness={"recent_oos_years": 3})
    assert g2.recent_oos_years == 1 and g5.recent_oos_years == 3

    r1 = tier3_robustness(Trail(), p, g2, neutralization=(), expected_sign=1, horizon=1,
                          recent_years=g2.recent_oos_years)
    r3 = tier3_robustness(Trail(), p, g5, neutralization=(), expected_sign=1, horizon=1,
                          recent_years=g5.recent_oos_years)
    assert r1.recent_n_days < r3.recent_n_days


# ---- 2. the wiring -----------------------------------------------------------

def _card(min_sub: float, *, subperiods: tuple[float, ...] = ()) -> SignalScorecard:
    """A card that clears every OTHER PROMISING leg, so robustness alone decides the verdict."""
    hp = HorizonIC(horizon=1, ic_mean=0.05, ic_std=0.2, ic_ir=0.30, ic_tstat=9.0, n_days=1000,
                   ci_low=0.02, ci_high=0.08, p_le_0=0.0, decile_spread=0.01,
                   decile_monotonic=True, sign_used=1)
    gross = GrossPower(by_horizon={1: hp}, primary_horizon=1, breadth=0.5, decay_halflife=5.0,
                       primary_ic_series=np.zeros(3), primary_ic_days=np.zeros(3, dtype=np.int64))
    rob = Robustness(4, subperiods or (min_sub,) * 4, min_sub, min_sub, 0.2, 504, (1000,) * 4)
    return SignalScorecard("s", "technical", "hash", HygieneResult(True, True, 0, 1000, True, ()),
                           gross, None, float("nan"), "PENDING", (), robustness=rob)


def _defl() -> Deflation:
    return Deflation(dsr=1.0, psr=1.0, mintrl_days=1.0, mintrl_years=1.0, fdr_q=0.0, n_trials=3,
                     sr_star=0.0, n_eff=3.0, fdr_q_bhy=0.0, hlz_pass=True)


def test_gate_blocks_a_signal_that_inverts_in_a_subperiod() -> None:
    """The whole point: before v4.0 this card scored PROMISING on dsr/IC/FDR alone."""
    out = _finalize(_card(-0.05), _defl(), _gates(), _panel(t=200))
    assert out.verdict == "LOGGED"
    assert any("regime-fragile" in c for c in out.caveats)


def test_gate_passes_a_robust_signal() -> None:
    out = _finalize(_card(0.10), _defl(), _gates(), _panel(t=200))
    assert out.verdict == "PROMISING"
    assert not any("regime-fragile" in c for c in out.caveats)


def test_unmeasurable_robustness_is_not_a_pass() -> None:
    """Absence of evidence must not read as evidence of robustness — the DSR leg's precedent."""
    out = _finalize(_card(float("nan")), _defl(), _gates(), _panel(t=200))
    assert out.verdict == "LOGGED"
    assert any("robustness undefined" in c for c in out.caveats)


def test_gate_threshold_is_read_from_the_gates_file_not_hardcoded() -> None:
    card = _card(-0.05)
    assert _finalize(card, _defl(), _gates(), _panel(t=200)).verdict == "LOGGED"
    # a substrate that pre-registers a tolerant floor gets a tolerant gate — no hardcoded 0.0
    lax = _gates(robustness={"min_subperiod_ic_ir": -0.10})
    assert lax.min_subperiod_ic_ir == -0.10
    assert _finalize(card, _defl(), lax, _panel(t=200)).verdict == "PROMISING"


def test_shipped_gates_yamls_expose_the_robustness_block() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for name in ("signal_eval.gates.yaml", "taiwan_signal_eval.gates.yaml",
                 "taiwan_smallcap_altdata.gates.yaml"):
        g = Gates.from_yaml(root / "configs" / name)
        assert g.min_subperiod_ic_ir == 0.0, name       # "no negative subperiod", now enforced
        assert g.recent_oos_years == 2, name


# ---- 3. CRU-1 regression -----------------------------------------------------

def test_cru1_recorded_smallcap_verdicts_are_preserved() -> None:
    """Wiring a gate after results are known is only legitimate if it is monotone-STRICTER and
    changes no recorded verdict. These are the values recorded in
    ``results/taiwan_smallcap_altdata/scorecard.json`` (S553-cont-133).

    P1's binding min (+0.109) comes from subperiod 4 — a ~1300-day window — NOT from the 48-day
    hole whose 1.536 the repair drops, so the repair and the gate agree with the recorded verdict.
    """
    p1 = (1.5363463056376288, 0.3471458928797888, 0.28328263086656974, 0.10912022684301258)
    gates = _gates()

    # P1 mom_rev: PROMISING, and it stays PROMISING under the newly-live gate.
    assert _finalize(_card(0.10912022684301258, subperiods=p1), _defl(), gates,
                     _panel(t=200)).verdict == "PROMISING"

    # P2/P3 have negative minima but were already LOGGED on inverted sign + DSR 0.000; the gate
    # only adds a second, independent reason to reject them.
    for min_sub in (-0.23083577300600336, -0.18548157321284778):
        assert _finalize(_card(min_sub), _defl(), gates, _panel(t=200)).verdict == "LOGGED"
