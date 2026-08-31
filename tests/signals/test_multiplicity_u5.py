"""U5 tripwires — multiplicity accounting is no longer a function of submission shape.

The defect (audit 2026-07-29 RC-7): ``tier4_deflation`` set ``n_trials = len(batch)``, so the
multiple-testing correction measured how the caller chose to slice its search. Splitting 100
candidates into ten batches of ten deflated each one against 10.

These lock the four properties the fix must have, in the order they matter:
  1. the LOOPHOLE is closed — a declared count makes a sliced sweep deflate like a whole one;
  2. the direction is MONOTONE-STRICTER — declaring can only lower DSR, never raise it, so no
     recorded verdict can be flipped LOGGED -> PROMISING by this change (CRU-1);
  3. ``use_effective_n`` (the C2.1 correlation haircut) COMPOSES without undoing (1) — it was
     the obvious wrong fix here, being monotone-looser;
  4. the ledger is REPRODUCIBLE — dedup by spec hash, so a replay lands on the same count.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from sharpen.signals import (
    Gates,
    HypothesisLedger,
    Multiplicity,
    evaluate_batch,
    load_multiplicity_gates,
    resolve_multiplicity,
    to_json,
)

from test_evaluator_holes_c2 import Trail, _momentum_panel  # same dir (rootdir-relative)


def _gates(**over) -> Gates:
    base = {"universe": {"min_names_per_day": 4},
            "coverage": {"min_days": 150},
            "gross_power": {"horizons": [1, 5], "primary_horizon": 1}}
    for k, v in over.items():
        base[k] = {**base.get(k, {}), **v} if isinstance(v, dict) else v
    return Gates.from_dict(base)


def _trails(ks) -> dict:
    return {s.spec.name: s for s in (Trail(k) for k in ks)}


# =========================================================== 1. the loophole ====

def test_batch_size_alone_no_longer_sets_the_deflation_count() -> None:
    """THE defect, measured on the NULL panel (rho=0) — the regime a multiple-testing
    correction exists for, and the only one where DSR is unsaturated enough to read.

    Two Trails submitted as one batch deflate against 2; declaring them a slice of a
    40-hypothesis search deflates against 40, and both the hurdle and the verdict move.
    """
    panel = _momentum_panel(rho=0.0, seed=3)
    signals = _trails([1, 2])

    sliced = evaluate_batch(signals, panel, _gates(), batch_name="sliced")
    declared = evaluate_batch(signals, panel, _gates(), batch_name="declared",
                              multiplicity=Multiplicity.preregistered(40, substrate="s"))

    assert sliced.n_multiplicity == 2 and sliced.multiplicity_source == "batch"
    assert declared.n_multiplicity == 40 and declared.multiplicity_source == "preregistered"

    by_s = {c.name: c for c in sliced.cards}
    by_d = {c.name: c for c in declared.cards}
    for name in signals:
        ds, dd = by_s[name].deflation, by_d[name].deflation
        assert ds.n_trials == dd.n_trials == 2                  # the batch pool is untouched
        assert dd.sr_star > ds.sr_star                          # a strictly higher hurdle
        assert dd.dsr < ds.dsr                                  # and a strictly lower verdict


def test_declared_count_raises_the_hurdle_even_where_dsr_saturates() -> None:
    """On a strongly-planted panel every DSR pins at 1.0, so DSR alone cannot show the fix
    working. ``sr_star`` — the deflated-Sharpe hurdle the order statistic produces — is the
    unsaturated observable, and it must rise monotonically with the declared count."""
    panel = _momentum_panel()
    signals = _trails([1, 2])
    stars = []
    for n in (None, 40, 1000):
        m = Multiplicity(n, "preregistered", "s") if n else None
        rs = evaluate_batch(signals, panel, _gates(), batch_name="t", multiplicity=m)
        stars.append(rs.cards[0].deflation.sr_star)
    assert stars[0] < stars[1] < stars[2]


def test_five_batches_of_two_match_one_batch_of_ten_under_a_ledger(tmp_path) -> None:
    """The loophole in its operational form: sweeping a library in slices must not buy a
    weaker correction than sweeping it whole. The final slice — the one that would report a
    survivor — deflates against all ten hypotheses, exactly as the whole-batch run does."""
    panel = _momentum_panel()
    ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    whole = evaluate_batch(_trails(ks), panel, _gates(), batch_name="whole")
    assert whole.n_multiplicity == 10

    ledger = HypothesisLedger(tmp_path / "hyp.json")
    counts = []
    for pair in zip(ks[::2], ks[1::2]):
        sigs = _trails(pair)
        mult = ledger.declare("sweep", sigs)
        rs = evaluate_batch(sigs, panel, _gates(), batch_name="slice", multiplicity=mult)
        assert rs.n_trials == 2                       # each batch really is a 2-signal batch
        counts.append(rs.n_multiplicity)

    assert counts == [2, 4, 6, 8, 10]                 # the correction grows with the search
    assert counts[-1] == whole.n_multiplicity         # and ends where the whole sweep sits


# ================================================ 2. monotone-stricter (CRU-1) ====

@pytest.mark.parametrize("declared", [0, 1, 2, 3, 10, 100])
def test_declaring_never_loosens_the_deflation(declared: int) -> None:
    """CRU-1 property test: for ANY declared count the resulting DSR is <= the batch-shaped
    DSR. This is what lets the bump claim recorded verdicts cannot flip LOGGED -> PROMISING."""
    panel = _momentum_panel()
    signals = _trails([1, 2, 3])
    base = {c.name: c for c in evaluate_batch(signals, panel, _gates(), batch_name="b").cards}
    m = Multiplicity(declared, "preregistered", "s") if declared else None
    got = {c.name: c for c in evaluate_batch(signals, panel, _gates(), batch_name="d",
                                             multiplicity=m).cards}
    for name in signals:
        db, dg = base[name].deflation.dsr, got[name].deflation.dsr
        if np.isfinite(db) and np.isfinite(dg):
            assert dg <= db + 1e-12


def test_under_declaration_falls_back_to_the_batch_and_says_so() -> None:
    """Declaring FEWER hypotheses than were submitted must not weaken the correction — the
    batch is the honest floor — and the mismatch must be visible, not smoothed over."""
    n, src = resolve_multiplicity(10, Multiplicity(3, "preregistered", "s"))
    assert n == 10 and src == "preregistered+batch"
    assert resolve_multiplicity(10, None) == (10, "batch")
    assert resolve_multiplicity(10, Multiplicity(0, "preregistered", "s")) == (10, "batch")


def test_omitting_multiplicity_is_byte_identical_to_pre_u5() -> None:
    """Back-compat: every existing call site passes no multiplicity, so its DSR must be the
    number it was before U5 — the batch pool size."""
    panel = _momentum_panel()
    signals = _trails([1, 2, 3])
    rs = evaluate_batch(signals, panel, _gates(), batch_name="compat")
    for c in rs.cards:
        assert c.deflation.n_multiplicity == c.deflation.n_trials == 3
        assert c.deflation.multiplicity_source == "batch"


def test_single_signal_batch_stays_undefined_even_when_declared() -> None:
    """A declared count raises the deflation MAGNITUDE; it must not make DSR computable where
    there is no trial dispersion to estimate. Manufacturing a DSR there would be LOOSER (NaN
    reads as LOGGED today), which would break the monotone-stricter claim."""
    panel = _momentum_panel()
    rs = evaluate_batch(_trails([2]), panel, _gates(), batch_name="one",
                        multiplicity=Multiplicity(50, "preregistered", "s"))
    assert not np.isfinite(rs.cards[0].deflation.dsr)


# ============================================ 3. composition with use_effective_n ====

def test_effective_n_composes_as_a_haircut_and_cannot_undo_the_floor() -> None:
    """``use_effective_n`` is monotone-LOOSER (n_eff <= n_trials) and was the trap fix. With a
    declared count it must act as a RATIO on that count, never as a replacement that drags the
    deflation back below the pre-U5 flagged value."""
    panel = _momentum_panel()
    signals = _trails([1, 2, 3])
    eff = _gates(deflation={"use_effective_n": True})

    flagged = evaluate_batch(signals, panel, eff, batch_name="eff")
    both = evaluate_batch(signals, panel, eff, batch_name="both",
                          multiplicity=Multiplicity(30, "preregistered", "s"))
    # the haircut still applies (correlated Trails => n_eff < 3 => scaled count < 30)
    assert flagged.n_multiplicity < 3 or flagged.n_multiplicity == 2   # clamped at 2
    assert both.n_multiplicity >= flagged.n_multiplicity
    assert both.n_multiplicity < 30                                    # haircut was applied
    by_f = {c.name: c for c in flagged.cards}
    by_b = {c.name: c for c in both.cards}
    for name in signals:
        df, db = by_f[name].deflation.dsr, by_b[name].deflation.dsr
        if np.isfinite(df) and np.isfinite(db):
            assert db <= df + 1e-12        # composing never loosens vs the flag alone


# ================================================== 4. ledger reproducibility ====

def test_ledger_is_idempotent_and_accumulates_across_runs(tmp_path) -> None:
    p = tmp_path / "hyp.json"
    led = HypothesisLedger(p)
    a, b = _trails([1, 2]), _trails([3, 4])

    assert led.record("sub", a) == 2
    assert led.record("sub", a) == 2                 # replay: same hashes => no inflation
    assert led.record("sub", b) == 4                 # genuinely new candidates accumulate
    assert led.count("other") == 0                   # substrates are isolated

    reloaded = HypothesisLedger(p)                   # survives a process boundary
    assert reloaded.count("sub") == 4
    assert reloaded.content_hash() == led.content_hash()


def test_ledger_hash_ignores_timestamps_but_tracks_the_hypothesis_set(tmp_path) -> None:
    l1, l2 = HypothesisLedger(tmp_path / "a.json"), HypothesisLedger(tmp_path / "b.json")
    l1.record("s", _trails([1, 2]))
    l2.record("s", _trails([2, 1]))                  # same set, different order/time
    assert l1.content_hash() == l2.content_hash()
    l2.record("s", _trails([3]))
    assert l1.content_hash() != l2.content_hash()


def test_corrupt_ledger_refuses_rather_than_restarting_at_zero(tmp_path) -> None:
    """A silently-reset ledger is a silently-restored loophole, so an unreadable file must be
    a loud failure, not a fresh count."""
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        HypothesisLedger(p)


def test_declare_returns_a_multiplicity_carrying_provenance(tmp_path) -> None:
    p = tmp_path / "hyp.json"
    m = HypothesisLedger(p).declare("sub", _trails([1, 2, 3]))
    assert m.n_hypotheses == 3 and m.source == "ledger"
    assert m.provenance == str(p) and len(m.ledger_hash) == 12


# ============================================================ policy + emit ====

def test_require_declared_blocks_promising_only_when_opted_in(monkeypatch) -> None:
    """The policy flag is monotone-stricter (PROMISING -> LOGGED) and OFF by default, so
    turning it on cannot resurrect a verdict, and leaving it off cannot rewrite the record."""
    panel = _momentum_panel()
    signals = _trails([1, 2, 3])
    # force a PROMISING-shaped pass by flooring every gate, then check only the policy leg
    g = _gates(gross_power={"horizons": [1, 5], "primary_horizon": 1,
                            "promising_ic_ir": -9.0, "promising_ic_tstat": -9.0},
               deflation={"promising_dsr": 0.0, "fdr_q_max": 1.0},
               robustness={"min_subperiod_ic_ir": -9.0})
    lax = evaluate_batch(signals, panel, g, batch_name="lax")
    assert any(c.verdict == "PROMISING" for c in lax.cards)
    assert all(any("batch-shaped" in cv for cv in c.caveats) for c in lax.cards
               if c.deflation is not None)

    strict = evaluate_batch(signals, panel, g, batch_name="strict",
                            multiplicity=Multiplicity(0, "batch", "s", require_declared=True))
    assert all(c.verdict != "PROMISING" for c in strict.cards)

    declared = evaluate_batch(signals, panel, g, batch_name="ok",
                              multiplicity=Multiplicity(3, "preregistered", "s",
                                                        require_declared=True))
    assert any(c.verdict == "PROMISING" for c in declared.cards)
    assert all(not any("batch-shaped" in cv for cv in c.caveats) for c in declared.cards)


def test_multiplicity_is_serialized_for_provenance() -> None:
    panel = _momentum_panel()
    rs = evaluate_batch(_trails([1, 2]), panel, _gates(), batch_name="emit",
                        multiplicity=Multiplicity(25, "preregistered", "s", provenance="doc.md"))
    d = json.loads(json.dumps(to_json(rs)))
    assert d["n_multiplicity"] == 25 and d["multiplicity_source"] == "preregistered"
    assert d["multiplicity_provenance"] == "doc.md" and d["n_trials"] == 2
    card = next(c for c in d["cards"] if c["deflation"])
    assert card["deflation"]["n_multiplicity"] == 25


def test_shipped_policy_defaults_are_permissive() -> None:
    """The shipped YAML must not silently demote existing pathways."""
    g = load_multiplicity_gates()
    assert g["require_declared"] is False and g["ledger_path"] is None
