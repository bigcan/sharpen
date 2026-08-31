"""F3 (crucible-v11.0) — the TRADED-BOOK floor: a signal whose long-short book loses money at
ZERO cost can no longer score PROMISING.

**The defect.** ``scorecard._finalize`` built ``promising`` from DSR, IC-IR, IC-t, FDR-q, optional
HLZ, ``min_subperiod_ic_ir`` and multiplicity provenance, and never consulted ``card.capturability``
at all. That was deliberate and stated in the source ("capturability caveats ... they flag, not
gate"), and it held only because gross rank-IC and traded-book P&L had never been observed to
disagree in SIGN. On 2026-07-31 they did: ``tw_smallcap_ivol``
(``results/taiwan_smallcap_price/scorecard.json``, spec ``27d38ce84ff5``) scored **PROMISING** with
a **frictionless Sharpe of -0.627** and net@standard -0.837 — a book that loses money before a
single basis point of cost is charged. The reconciliation is ``decile_monotonic: False``: a real
rank-IC can live in cells a long-short book does not weight, so "statistically detectable ranking"
and "makes money" are genuinely different claims. See the pre-registration
``docs/research/taiwan_smallcap_price_probes_preregistration_2026-07-31.md`` §6.2.

**The fix, and why the two legs are asymmetric.** FRICTIONLESS Sharpe is a GATE: no cost model, no
venue and no assumption enters it, so a non-positive value is an unconditional statement that the
structure never becomes a position that makes money. NET@standard stays a CAVEAT (gating only under
the opt-in ``require_positive_net_standard``), because a cost model IS venue-specific — Taiwan's
0.30% sell-side transaction tax is not Nasdaq's 10bps — and a cost-blocked signal can still be a
real research object elsewhere.

Both thresholds live in the ``capturability:`` block of the gates schema, never in code.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sharpen.signals import Gates
from sharpen.signals.eval_harness import (
    Capturability,
    CostResult,
    Deflation,
    GrossPower,
    HorizonIC,
    HygieneResult,
    Robustness,
)
from sharpen.signals.features import Panel
from sharpen.signals.scorecard import SignalScorecard, _finalize

ROOT = Path(__file__).resolve().parents[2]


def _panel() -> Panel:
    """Meta-only panel — ``_finalize`` reads nothing but ``panel.meta``."""
    z = np.ones((8, 4))
    dates = (np.datetime64("2012-01-02")
             + np.arange(8) * np.timedelta64(1, "D")).astype("datetime64[ns]")
    return Panel(dates, ("A", "B", "C", "D"), z, z, z, z, z, np.ones((8, 4), bool), z,
                 np.zeros(4, dtype=int), {"survivorship_free": False, "source": "synthetic"})


def _cap(frictionless: float, net_std: float = 0.4) -> Capturability:
    by_cost = {
        "frictionless": CostResult("frictionless", frictionless, 1.2, 4.0, -0.1),
        "standard": CostResult("standard", net_std, 1.0, 4.0, -0.2),
    }
    return Capturability(by_cost, frictionless, frictionless - net_std)


def _card(frictionless: float | None, net_std: float = 0.4) -> SignalScorecard:
    """A card that clears every OTHER PROMISING leg, so capturability alone decides the verdict.

    ``frictionless=None`` omits the Capturability object entirely (the unmeasured case).
    """
    hp = HorizonIC(horizon=1, ic_mean=0.05, ic_std=0.2, ic_ir=0.30, ic_tstat=9.0, n_days=1000,
                   ci_low=0.02, ci_high=0.08, p_le_0=0.0, decile_spread=0.01,
                   decile_monotonic=True, sign_used=1)
    gross = GrossPower(by_horizon={1: hp}, primary_horizon=1, breadth=0.5, decay_halflife=5.0,
                       primary_ic_series=np.zeros(3), primary_ic_days=np.zeros(3, dtype=np.int64))
    rob = Robustness(4, (0.2,) * 4, 0.2, 0.2, 0.2, 504, (1000,) * 4)   # robustness leg satisfied
    cap = None if frictionless is None else _cap(frictionless, net_std)
    return SignalScorecard("s", "technical", "hash", HygieneResult(True, True, 0, 1000, True, ()),
                           gross, None, float("nan"), "PENDING", (), capturability=cap,
                           robustness=rob)


def _defl() -> Deflation:
    return Deflation(dsr=1.0, psr=1.0, mintrl_days=1.0, mintrl_years=1.0, fdr_q=0.0, n_trials=3,
                     sr_star=0.0, n_eff=3.0, fdr_q_bhy=0.0, hlz_pass=True)


def _gates(**over) -> Gates:
    base = {"universe": {"min_names_per_day": 4}, "coverage": {"min_days": 1},
            "gross_power": {"horizons": [1], "primary_horizon": 1}}
    base.update(over)
    return Gates.from_dict(base)


def _verdict(card: SignalScorecard, gates: Gates | None = None) -> SignalScorecard:
    return _finalize(card, _defl(), gates or _gates(), _panel())


# ---- 1. the gate ------------------------------------------------------------

def test_gate_blocks_a_book_that_loses_money_before_costs() -> None:
    """THE regression. These are ``tw_smallcap_ivol``'s recorded numbers; before v11.0 this exact
    card scored PROMISING."""
    out = _verdict(_card(-0.6269475774654387, net_std=-0.8370155043744929))
    assert out.verdict == "LOGGED"
    assert any("NOT CAPTURABLE" in c for c in out.caveats)


def test_gate_passes_a_capturable_signal() -> None:
    out = _verdict(_card(1.008429510329003, net_std=0.5289563237496462))
    assert out.verdict == "PROMISING"
    assert not any("NOT CAPTURABLE" in c for c in out.caveats)


def test_exactly_zero_frictionless_is_blocked() -> None:
    """"Non-positive" is the claim, so the floor is a STRICT bound: a book that merely fails to
    lose money has not shown it makes any."""
    assert _verdict(_card(0.0)).verdict == "LOGGED"
    assert _verdict(_card(1e-6)).verdict == "PROMISING"


def test_unmeasured_capturability_is_not_a_pass() -> None:
    """Absence of evidence must not read as evidence of capture — the DSR and subperiod legs'
    precedent (``_finalize`` fails closed on an unmeasurable leg, it does not skip it)."""
    out = _verdict(_card(None))
    assert out.verdict == "LOGGED"
    assert any("capturability unmeasured" in c for c in out.caveats)


def test_nan_frictionless_is_not_a_pass() -> None:
    out = _verdict(_card(float("nan")))
    assert out.verdict == "LOGGED"
    assert any("capturability unmeasured" in c for c in out.caveats)


def test_threshold_is_read_from_the_gates_file_not_hardcoded() -> None:
    """CLAUDE.md invariant: no numeric gate is hardcoded. A substrate may pre-register a different
    floor in its own gates YAML, in EITHER direction."""
    card = _card(-0.6269475774654387, net_std=-0.837)
    assert _verdict(card).verdict == "LOGGED"

    lax = _gates(capturability={"min_frictionless_sharpe": -1.0})
    assert lax.min_frictionless_sharpe == -1.0
    assert _verdict(card, lax).verdict == "PROMISING"

    strict = _gates(capturability={"min_frictionless_sharpe": 0.5})
    assert _verdict(_card(0.3), strict).verdict == "LOGGED"
    assert _verdict(_card(0.7), strict).verdict == "PROMISING"


# ---- 2. net@standard stays a caveat unless opted in --------------------------

def test_net_at_standard_is_a_caveat_not_a_gate_by_default() -> None:
    """Cost models are venue-specific; the default keeps today's reporting behaviour."""
    out = _verdict(_card(1.0, net_std=-0.8))
    assert out.verdict == "PROMISING"
    assert any("cost-blocked" in c for c in out.caveats)
    assert _gates().require_positive_net_standard is False


def test_net_at_standard_gates_under_the_opt_in_flag() -> None:
    strict = _gates(capturability={"require_positive_net_standard": True})
    assert _verdict(_card(1.0, net_std=-0.8), strict).verdict == "LOGGED"
    assert _verdict(_card(1.0, net_std=0.3), strict).verdict == "PROMISING"


def test_opt_in_net_gate_also_fails_closed_when_unmeasured() -> None:
    strict = _gates(capturability={"require_positive_net_standard": True})
    no_std = Capturability({"frictionless": CostResult("frictionless", 1.0, 1.2, 4.0, -0.1)},
                           1.0, float("nan"))          # frictionless fine, standard never priced
    card = replace(_card(1.0), capturability=no_std)
    assert _verdict(card, strict).verdict == "LOGGED"


# ---- 3. CRU-1: what this bump changes, and what it must not ------------------

# Every capturability figure recorded by the small/mid-cap campaigns, read from
# results/taiwan_smallcap_*/scorecard.json on 2026-07-31. (name, frictionless, net@standard,
# recorded verdict).
_RECORDED = [
    ("tw_smallcap_mom_rev",        1.008429510329003,   0.5289563237496462,  "PROMISING"),
    ("tw_smallcap_mom_rev_lowturn", 0.7056949352702804, 0.515850014086757,   "PROMISING"),
    ("tw_smallcap_holder_conc",   -0.06042079881329369, -0.815570678173937,  "LOGGED"),
    ("tw_smallcap_margin_crowd",  -0.3413780069124212,  -1.2074333657355008, "LOGGED"),
    ("tw_smallcap_st_reversal",   -0.1815470552572751,  -0.8352080671191372, "LOGGED"),
    ("tw_smallcap_short_interest", 0.0862464733323175,  -0.2876685373849907, "LOGGED"),
]


@pytest.mark.parametrize("name,fric,net,recorded", _RECORDED)
def test_cru1_the_sole_recorded_promising_survives_and_no_logged_is_promoted(
        name: str, fric: float, net: float, recorded: str) -> None:
    """The gate is monotone-STRICTER, so it may only demote. P1 ``tw_smallcap_mom_rev`` — the only
    PROMISING ever recorded by this funnel — is genuinely capturable (frictionless 1.008 / 0.706)
    and MUST survive; nothing already LOGGED may be promoted.

    These cards clear every other leg by construction, so a LOGGED row here proves only that the
    capturability leg alone does not promote it — the recorded runs rejected the LOGGED rows on
    independent grounds (inverted sign, DSR 0.000) that this fixture deliberately does not model.
    """
    out = _verdict(_card(fric, net_std=net))
    if recorded == "PROMISING":
        assert out.verdict == "PROMISING", f"{name}: v11.0 must not demote a capturable signal"
    else:
        assert out.verdict != "GATE_FAIL"


def test_cru1_ivol_is_the_one_recorded_verdict_this_bump_changes() -> None:
    """``tw_smallcap_ivol`` is the motivating case and the ONLY recorded verdict that moves. It is
    demoted, never promoted — which is why wiring this gate after results were seen is legitimate
    (the anti-goal-post-move direction: the candidate that motivated the change loses by it)."""
    out = _verdict(_card(-0.6269475774654387, net_std=-0.8370155043744929))
    assert out.verdict == "LOGGED"


def test_gate_is_monotone_in_the_threshold() -> None:
    """Raising the floor can only ever REMOVE PROMISINGs — the CRU-1 property that makes wiring a
    gate after results are known safe. Asserted as a nesting chain plus both endpoints, so a
    mutation that flipped the comparison (which would still produce nested sets) fails here."""
    grid = [-2.0, -0.6, 0.0, 0.2, 1.0, 3.0]

    def promoted(floor: float) -> set[float]:
        g = _gates(capturability={"min_frictionless_sharpe": floor})
        return {f for f in grid if _verdict(_card(f), g).verdict == "PROMISING"}

    chain = [promoted(x) for x in (-10.0, -1.0, 0.0, 0.5, 2.0)]
    for looser, stricter in zip(chain, chain[1:]):
        assert stricter <= looser, "raising the floor promoted something the looser gate did not"
    assert chain[0] == set(grid)          # a floor below every book promotes every book...
    assert chain[2] == {0.2, 1.0, 3.0}    # ...the shipped 0.0 floor keeps exactly the positive ones
    assert chain[-1] == {3.0}             # ...and a floor at 2.0 leaves only the book above it


# ---- 4. the sealed gates files are inherited, not edited ---------------------

def test_sealed_gates_yamls_inherit_the_default_without_a_byte_moving() -> None:
    """The three CRU-1-sealed gates files predate this key, so they must acquire the gate by
    DEEP-MERGE from the schema default — never by an edit, which would move their frozen hashes
    (``519158fa1450`` / ``22a18172be1a`` / ``0ccf6dd584f0``, pinned in
    ``tests/crucible/test_version.py``). This asserts both halves: the key resolves, and the bytes
    that carry the moat still do not contain it."""
    for name in ("signal_eval.gates.yaml", "taiwan_signal_eval.gates.yaml",
                 "taiwan_smallcap_altdata.gates.yaml", "taiwan_smallcap_price.gates.yaml"):
        p = ROOT / "configs" / name
        g = Gates.from_yaml(p)
        assert g.min_frictionless_sharpe == 0.0, name
        assert g.require_positive_net_standard is False, name
        assert b"min_frictionless_sharpe" not in p.read_bytes(), (
            f"{name} was EDITED to add the key. The gate is inherited from the schema default "
            f"precisely so no sealed byte moves; adding it here re-hashes the file."
        )
