"""Crucible versioning tripwires — the gates hash is the provenance anchor (spec §5)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from finrl_pro_ds.crucible.version import (
    CRUCIBLE_BASELINE_VERSION,
    CRUCIBLE_VERSION,
    gates_hash,
)

ROOT = Path(__file__).resolve().parents[2]
GATES = ROOT / "configs" / "signal_eval.gates.yaml"
TAIWAN_GATES = ROOT / "configs" / "taiwan_signal_eval.gates.yaml"
SMALLCAP_ALTDATA_GATES = ROOT / "configs" / "taiwan_smallcap_altdata.gates.yaml"


def test_versions_are_distinct_and_tagged_form() -> None:
    # v6.0 = the audit §5 CORRECTED CONTRACT becomes a selectable production decision layer (opt-in,
    # per-substrate, default "shipped"): one Jobson-Korkie-Memmel Sharpe-difference z + a BINDING LORD++
    # p-gate + the three cheap guards, DROPPING the F1-sealed marginal_t and F2-sealed dsr_aug legs
    # (measured 0/170 lifetime pass each, vs 170/170 for the uplift leg — the 2026-07-29 audit).
    #
    # MAJOR, and the FIRST bump where CRU-1 verdict-preservation is deliberately NOT claimed: it is not
    # monotone-stricter, which is the entire point — a candidate the shipped contract rejected can pass
    # the corrected one. The 0-PROMISING record must be RE-SCORED under the new contract, never pooled
    # with it. It still touches NO gate byte (all three frozen hashes below are unchanged; the corrected
    # thresholds live in configs/crucible_corrected_contract.gates.yaml), and the manifest now pins
    # `contract` + `corrected_gates_hash` so no verdict can be read without knowing which gate made it.
    # v7.0 = the substrate-power stamp is CANDIDATE-TYPE aware and fails CLOSED on an unmeasured type.
    # Every MDE curve before 2026-07-29 was measured on the OVERLAY path yet applied to cross_sectional
    # substrates too; the newly-measured cross-sectional surface is EQUAL-OR-WORSE at matched depth
    # (holdout 1011: 1.833 vs overlay 1.281), so the guard was UNDER-stating MDE by ~35-40% on those
    # mines — the fail-OPEN direction v5.0 exists to close. MAJOR (a live gate's decision FUNCTION),
    # monotone-STRICTER at every depth, NO gate byte moved.
    # v7.1 = per-name (T,N) alt-data reachable by the CROSS-SECTIONAL search (audit U3). v2.9's C2-06
    # fix excluded ALL slots to stop BROADCAST terminals rank()ing to dead constants; that also closed
    # the only route for PER-NAME data, confining every non-price dataset to a per-day timing overlay
    # (T obs instead of T×N). Now filtered by SHAPE. MINOR (v2.1 precedent): adds terminals + eval
    # reach, changes no verdict function and no gate byte; byte-identical without a (T,N) slot.
    # v8.0 = the CORRECTED contract is the DEFAULT, and the power guard FAILS CLOSED when it cannot
    # measure (a missing calibration curve used to SKIP the guard entirely and mine unguarded — acute
    # once the default flipped, since `results/` is gitignored so curves are absent on every fresh
    # clone). MAJOR on both counts. Note it does not start any mining: every real substrate is still
    # refused (worst-across-types MDE 1.833 at holdout 1011 vs the 0.50 ceiling), and that ceiling was
    # not moved to manufacture an ALLOW.
    # v8.1 = TWSE T86 wired into PER-NAME (T,N) slots (audit U3a), so the Taiwan substrate carries data
    # the cross-sectional search can use. v7.1 opened that channel but nothing produced a (T,N) slot.
    # The data was already per-stock ("<ticker>:<field>"); the bridge was FLATTENING it into 40
    # broadcast terminals. ADDITIVE (broadcast untouched, overlay unchanged); MINOR per v2.8.
    # v9.0 = multiplicity accounting stops depending on SUBMISSION SHAPE (audit RC-7 / U5). DSR's
    # trial count was the BATCH size, so ten batches of ten deflated against 10 rather than 100 —
    # the multiple-comparison correction was measuring the caller's for-loop, on the LIVE scorecard
    # path (xlg_megacap_ic_gate sweeps one alpha list over four universes: a silent divide-by-four).
    # Now max(batch_pool, declared), declared by pre-registration or a per-substrate HypothesisLedger.
    # Deliberately NOT fixed via use_effective_n, which is monotone-LOOSER (n_eff <= n_trials) and is
    # a correlation adjustment, not a multiplicity control; it now composes as a RATIO on the declared
    # count. MAJOR (verdict function) but CRU-1 HOLDS — unlike v6.0 — because max() can only add
    # trials and DSR is monotone non-increasing in them. NO gate byte moved (the require_declared
    # policy lives in configs/crucible_multiplicity.gates.yaml, shipped false).
    # v10.0 = the search gets a MEMORY (audit U4 / RC-5) + Tier-0 causality is ENFORCED on generated
    # genomes (U6 / RC-8) — the bump that CLOSES the 2026-07-29 audit roadmap. killed_families() was
    # structurally empty (no funnel path ever wrote a KILLED_VERDICTS value), so the proposer prompt read
    # "(none)" across 403 trials. Rejections are now classified DECISIVE (implied_mde <= multiple x the
    # active contract's economic floor ⇒ the "no" is informative ⇒ family dies) or UNDERPOWERED (parked
    # with its MDE, re-admitted when the substrate gains real power); dedup canonicalizes commutative AST
    # order. evolve() truncation-probes every distinct genome before fitness and culls a leaky one.
    # MAJOR because dedup + re-admission change WHICH hypotheses a tick tests — but CRU-1
    # verdict-preservation is claimed and VERIFIED (semantic dedup drops nothing from the offline seed
    # bank; the U6 probe leaves a causal search byte-identical; new ledger columns nullable and the
    # verdict vocabulary untouched, so no historical row changes meaning). Ships with the U7 calibration,
    # which CONFIRMED uplift_min at 0.10 (comment-only hash move on the corrected-contract file,
    # 2f4639a48415 -> e60079a1d94b) and surfaced RC-11: the uplift null is substrate-dependent by ~200x
    # and the Taiwan overlay path is un-calibrated. Still nothing mines — audit U8 binds.
    assert CRUCIBLE_VERSION == "crucible-v10.0"
    assert CRUCIBLE_BASELINE_VERSION == "crucible-v1.0"
    assert CRUCIBLE_VERSION != CRUCIBLE_BASELINE_VERSION


def test_current_version_has_a_matching_git_tag() -> None:
    """``version.py`` states its own rule — "keep a matching git tag (``crucible-vMAJOR.MINOR``) so
    ``run_manifest.crucible_version`` is anchored to an immutable commit" — but nothing enforced it,
    and it rotted: **v2.7, v2.8, v2.9 and v3.0 all shipped untagged**, so every manifest stamped with
    those versions pointed at nothing. v3.0 was the costly one — a MAJOR verdict-function change
    (the F14 audit fixes) whose runs could not be tied to the code that produced them. All four were
    registered retroactively in S553-cont-134, on their bump commits per the v1.0–v2.6 convention.
    This test makes the convention EXECUTABLE so the anchor cannot silently rot again — a declared
    rule with no consumer is exactly the class of gap this session was opened to close.

    NOTE (expected red): a bump commit is failing until its tag exists. That is the forcing function,
    not a bug — bump ``CRUCIBLE_VERSION``, then ``git tag -a <version>`` on that commit.

    SKIPs rather than fails when git or the tag list is unavailable (no git, shallow clone, exported
    tree, Docker COPY). A tripwire that reds for environmental reasons is one someone silences —
    which is precisely how the frozen gates hashes nearly got re-pinned (see the CRLF test below)."""
    if shutil.which("git") is None:
        pytest.skip("git unavailable — cannot verify the tag anchor")
    try:
        r = subprocess.run(["git", "tag", "-l", "crucible-*"], cwd=ROOT,
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:      # not a repo / git broken
        pytest.skip(f"git tag failed ({e}) — cannot verify the tag anchor")
    if r.returncode != 0:
        pytest.skip(f"git tag returned {r.returncode} — not a usable repo here")
    tags = set(r.stdout.split())
    if not tags:
        pytest.skip("no crucible-* tags in this checkout (shallow clone?) — nothing to verify")
    assert CRUCIBLE_VERSION in tags, (
        f"CRUCIBLE_VERSION is {CRUCIBLE_VERSION!r} but no matching git tag exists. A run_manifest "
        f"stamped {CRUCIBLE_VERSION!r} would not resolve to an immutable commit — the exact gap that "
        f"left v2.7-v3.0 unanchored. Create it on the bump commit: "
        f"`git tag -a {CRUCIBLE_VERSION} -m '...'`. Do NOT delete this test to go green."
    )


def test_funnel_gates_hash_still_frozen_at_v2_0() -> None:
    """P5 must not perturb the funnel moat: the frozen crucible-v2.0 gates_hash is unchanged."""
    assert gates_hash(GATES) == "519158fa1450"


def test_taiwan_gates_hash_frozen() -> None:
    """CRU-1 for the Taiwan substrate. The Taiwan funnel runs on its OWN gates file
    (``configs/taiwan_signal_eval.gates.yaml``, generation thresholds identical to the frozen
    cross-asset file — see ADR-A3) and every real Taiwan run pins hash ``22a18172be1a``. Without a
    frozen reference here, an edit to that file would silently pass CRU-1 for ~95% of the mining
    record (the S553-cont-131 independent audit flagged the gap). This test registers the reference so
    any byte change to the Taiwan gate — including a threshold move — trips a red, exactly as the
    cross-asset moat test does. A DELIBERATE Taiwan gate change must bump BOTH this hash and the
    version, per the CRU-1 MAJOR/MINOR protocol."""
    assert gates_hash(TAIWAN_GATES) == "22a18172be1a"


def test_smallcap_altdata_probe_gates_hash_registered() -> None:
    """CRU-1 registration of the small/mid-cap alt-data PROBE gates file (audit §5/§6.7).

    The pre-registered probes run on their OWN gates file
    (``configs/taiwan_smallcap_altdata.gates.yaml``) — a PARALLEL pathway (lockbox/cohort ADR
    precedent), so it does NOT touch the frozen funnel moats (``519158fa1450`` / ``22a18172be1a``,
    asserted above and unchanged). Pinning its hash here means any byte change to the probe gate —
    including a threshold move after results are seen — trips a red, exactly as the funnel moats do:
    the anti-goal-post-move seal for the probe pathway. A DELIBERATE change must bump this reference."""
    assert gates_hash(SMALLCAP_ALTDATA_GATES) == "0ccf6dd584f0"


def test_gates_files_are_lf_so_the_moat_is_portable() -> None:
    """The three hashes above are over RAW BYTES, so they are line-ending-sensitive: a stock Windows
    checkout (``core.autocrlf=true``) rewrites every gates YAML to CRLF and moves all three
    (``519158fa1450`` -> ``d04d7e747b48``, etc.), reddening the CRU-1 tripwires for a reason that has
    nothing to do with the gates. That failure mode is dangerous, not merely annoying: the obvious
    way to "fix" three red hash assertions is to re-pin the constants, which would silently destroy
    the anti-goal-post-move seal. It also means a ``gates_hash`` stamped into a run_manifest differs
    by platform, so ``crucible reproduce`` cannot verify a run across OSes.

    ``.gitattributes`` pins ``configs/*.gates.yaml text eol=lf`` to prevent it. This asserts the rule
    actually took effect in THIS working tree (an editor can still save CRLF), and fails loudly with
    the diagnosis instead of leaving three unexplained hash mismatches."""
    for p in (GATES, TAIWAN_GATES, SMALLCAP_ALTDATA_GATES):
        assert b"\r\n" not in p.read_bytes(), (
            f"{p.name} has CRLF line endings. gates_hash() is a SHA-256 over raw bytes, so every "
            f"frozen CRU-1 hash in this file will mismatch. This is a CHECKOUT ARTIFACT, not a gate "
            f"change — do NOT 're-pin' the hashes to make them pass. Restore LF instead: "
            f"`rm configs/*.gates.yaml && git checkout -- configs/` with the "
            f"`configs/*.gates.yaml text eol=lf` rule present in .gitattributes."
        )


def test_gates_hash_is_stable_and_12_hex() -> None:
    h1 = gates_hash(GATES)
    h2 = gates_hash(GATES)
    assert h1 == h2                                  # deterministic over identical bytes
    assert len(h1) == 12 and all(c in "0123456789abcdef" for c in h1)


def test_gates_hash_changes_on_any_edit(tmp_path: Path) -> None:
    """ANY byte change (even a comment) must move the hash — that is what makes goal-post moving
    visible in a run's provenance (spec §5)."""
    p = tmp_path / "g.yaml"
    p.write_text("generation:\n  delta_median_min: 0.0\n", encoding="utf-8")
    before = gates_hash(p)
    p.write_text("generation:\n  delta_median_min: 0.0\n  # a new comment\n", encoding="utf-8")
    assert gates_hash(p) != before
