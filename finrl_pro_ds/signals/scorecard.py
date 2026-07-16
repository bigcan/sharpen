"""Signal Scorecard + the batch orchestrator + ranking + JSON/Markdown emit.

``evaluate_batch`` runs every candidate through T0 (hygiene) + T1 (gross power) + T2
(capturability) + T3 (robustness) [+ T5 (orthogonality) if a FactorBook is supplied], then
T4 (batch deflation, ``n_trials`` = #candidates with a finite IC-IR), assigns a verdict from
deflation (DSR/FDR) + gross power (IC-IR/t) + subperiod robustness (crucible-v4.0),
and ranks lexicographically: primary = deflated IC-IR (DSR), then raw IC-IR, then
orthogonality (less correlated wins), then subperiod robustness. The scorecard ALWAYS
reports gross IC even for cost-blocked signals (LOGGED, not dropped), and the verdict tops
out at PROMISING — never GO (promotion needs survivorship-free re-validation + a Tier-2
audit; ADR-3 / CLAUDE.md).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .eval_harness import (
    Capturability,
    compute_scores,
    CPCVResult,
    Deflation,
    FactorBook,
    GrossPower,
    HorizonIC,
    HygieneResult,
    Orthogonality,
    Robustness,
    tier0_hygiene,
    tier1_gross_power,
    tier2_capturability,
    tier3_robustness,
    tier3_5_cpcv,
    tier4_deflation,
    tier5_orthogonality,
)
from .features import Panel
from .gates import Gates

if TYPE_CHECKING:
    from .protocol import Signal


@dataclass(frozen=True, slots=True)
class SignalScorecard:
    name: str
    family: str
    spec_hash: str
    hygiene: HygieneResult
    gross: GrossPower | None
    deflation: Deflation | None
    rank_key: float
    verdict: str               # GATE_FAIL | LOGGED | PROMISING
    caveats: tuple[str, ...]
    capturability: Capturability | None = None
    robustness: Robustness | None = None
    orthogonality: Orthogonality | None = None
    cpcv: CPCVResult | None = None


@dataclass(frozen=True, slots=True)
class RankedScorecard:
    batch_name: str
    primary_horizon: int
    n_trials: int
    cards: list[SignalScorecard]
    panel_meta: dict
    gates: dict


def evaluate_signal(sig: "Signal", panel: Panel, gates: Gates,
                    factor_book: FactorBook | None = None) -> SignalScorecard:
    """T0 + T1 + T2 + T3 (+ T5 if a FactorBook is given). Deflation is filled at batch time."""
    # F1: a two-sided signal (expected_sign=0) needs its sign chosen from returns; choosing
    # it from the full sample is a look-ahead and a causal variant is not yet implemented, so
    # reject it cleanly rather than emit a biased card.
    if sig.spec.expected_sign == 0:
        hy0 = HygieneResult(False, False, 0, 0, False,
                            ("expected_sign=0 (two-sided) unsupported: causal sign-selection "
                             "unimplemented (audit F1)",))
        return SignalScorecard(sig.spec.name, sig.spec.family, sig.spec.content_hash(), hy0,
                               None, None, float("nan"), "GATE_FAIL", ())
    hy = tier0_hygiene(sig, panel, min_days=gates.min_days,
                       min_names_per_day=gates.min_names_per_day,
                       max_ohlc_violations=gates.max_ohlc_violations)
    gross = cap = rob = orth = cpcv = None
    if hy.passed:
        ns, es, ph = sig.spec.neutralization, sig.spec.expected_sign, gates.primary_horizon
        mn = gates.min_names_per_day
        scores = compute_scores(sig, panel, ns)  # neutralize ONCE; reuse across tiers
        gross = tier1_gross_power(sig, panel, gates.horizons, primary_horizon=ph,
                                  neutralization=ns, expected_sign=es, scores=scores,
                                  min_names=mn)
        cap = tier2_capturability(sig, panel, gates, neutralization=ns, expected_sign=es,
                                  hold_horizon=ph, scores=scores)
        rob = tier3_robustness(sig, panel, gates, neutralization=ns, expected_sign=es,
                               horizon=ph, scores=scores, min_names=mn,
                               recent_years=gates.recent_oos_years)
        if gates.cpcv_enabled:
            cpcv = tier3_5_cpcv(sig, panel, gates, neutralization=ns, expected_sign=es,
                                horizon=ph, n_groups=gates.cpcv_n_groups,
                                k_test=gates.cpcv_k_test, embargo_days=gates.cpcv_embargo_days,
                                scores=scores)
        if factor_book is not None:
            orth = tier5_orthogonality(sig, panel, factor_book, neutralization=ns,
                                       expected_sign=es, horizon=ph, scores=scores)
    verdict = "GATE_FAIL" if not hy.passed else "PENDING"
    return SignalScorecard(sig.spec.name, sig.spec.family, sig.spec.content_hash(), hy,
                           gross, None, float("nan"), verdict, (), capturability=cap,
                           robustness=rob, orthogonality=orth, cpcv=cpcv)


def _finalize(card: SignalScorecard, defl: Deflation | None, gates: Gates,
              panel: Panel) -> SignalScorecard:
    caveats: list[str] = []
    if not panel.meta.get("survivorship_free", False):
        caveats.append("survivorship-biased data — results are UPPER BOUNDS")

    if card.gross is None:  # failed Tier-0
        return replace(card, deflation=defl, rank_key=float("-inf"), verdict="GATE_FAIL",
                       caveats=tuple(card.hygiene.reasons) + tuple(caveats))

    hp = card.gross.by_horizon[card.gross.primary_horizon]
    dsr = defl.dsr if defl is not None else float("nan")
    if defl is None or not np.isfinite(dsr):
        caveats.append("deflation undefined (batch too small for DSR)")

    # capturability caveats (do not change the gross-IC verdict — they flag, not gate)
    if card.capturability is not None:
        std = card.capturability.by_cost.get("standard")
        if std is not None and np.isfinite(std.net_sharpe) and std.net_sharpe <= 0:
            caveats.append("cost-blocked: net Sharpe <= 0 at standard cost "
                           "(structure without capture)")
        elif (np.isfinite(card.capturability.cost_wall)
              and card.capturability.cost_wall > gates.cost_wall_caution):
            caveats.append(f"high cost-wall ({card.capturability.cost_wall:.2f})")
    if (card.orthogonality is not None and np.isfinite(card.orthogonality.max_abs_corr)
            and card.orthogonality.max_abs_corr > 0.7):
        fac = max(card.orthogonality.corr_by_factor.items(),
                  key=lambda kv: abs(kv[1]) if np.isfinite(kv[1]) else -1.0)[0]
        caveats.append(f"highly correlated to {fac} "
                       f"(|corr|={card.orthogonality.max_abs_corr:.2f})")

    # F2b (v4.0) — the `robustness.min_subperiod_ic_ir` gate. Declared in every gates YAML
    # ("no negative subperiod") and pre-registered as a threshold since v2.0, but never read: the
    # verdict came from dsr/IC/FDR alone, so a signal that inverted in a subperiod still scored
    # PROMISING. Starved windows are excluded upstream (tier3_robustness's min-valid-days floor),
    # so this reads a measured estimate or none at all. As with the DSR leg, an UNMEASURABLE
    # robustness is not a pass — absence of evidence must not read as evidence of robustness.
    msi = card.robustness.min_subperiod_ic_ir if card.robustness is not None else float("nan")
    if not np.isfinite(msi):
        caveats.append("subperiod robustness undefined (no subperiod has enough valid days)")
    elif msi < gates.min_subperiod_ic_ir:
        caveats.append(f"regime-fragile: min subperiod IC-IR {msi:.3f} < "
                       f"{gates.min_subperiod_ic_ir:g} (inverts in a sampled subperiod)")

    # HLZ / BHY hurdle (C2.3): reported + caveated; folded into PROMISING only under require_hlz.
    hlz_pass = bool(defl is not None and defl.hlz_pass)
    promising = (
        defl is not None and np.isfinite(dsr) and dsr >= gates.promising_dsr
        and np.isfinite(hp.ic_ir) and hp.ic_ir >= gates.promising_ic_ir
        and np.isfinite(hp.ic_tstat) and hp.ic_tstat >= gates.promising_ic_tstat
        and np.isfinite(defl.fdr_q) and defl.fdr_q <= gates.fdr_q_max
        and (hlz_pass or not gates.require_hlz)
        and np.isfinite(msi) and msi >= gates.min_subperiod_ic_ir
    )
    if promising and not hlz_pass:
        caveats.append(f"fails HLZ hurdle (t>{gates.hlz_t_min:g} & BHY-FDR<={gates.fdr_q_max:g}) "
                       "— promotion-blocking")
    # CPCV fragility flag (C2.2): a negative 5th-percentile OOS path is a robustness caveat.
    if (card.cpcv is not None and np.isfinite(card.cpcv.oos_sharpe_p05)
            and card.cpcv.oos_sharpe_p05 < 0.0):
        caveats.append(f"CPCV fragile: p05 OOS Sharpe {card.cpcv.oos_sharpe_p05:.2f}<0 "
                       f"({card.cpcv.frac_paths_positive:.0%} of {card.cpcv.n_paths} paths +)")
    verdict = "PROMISING" if promising else "LOGGED"
    rank_key = float(dsr) if np.isfinite(dsr) else float("-inf")
    return replace(card, deflation=defl, rank_key=rank_key, verdict=verdict,
                   caveats=tuple(caveats))


def _rank_tuple(c: SignalScorecard) -> tuple[float, float, float, float]:
    dsr = c.deflation.dsr if (c.deflation is not None and np.isfinite(c.deflation.dsr)) else -1.0
    ir = float("-inf")
    if c.gross is not None:
        v = c.gross.by_horizon[c.gross.primary_horizon].ic_ir
        ir = float(v) if np.isfinite(v) else float("-inf")
    orth = 0.0  # less correlated to the factor book = higher (better)
    if c.orthogonality is not None and np.isfinite(c.orthogonality.max_abs_corr):
        orth = -float(c.orthogonality.max_abs_corr)
    rob = 0.0
    if c.robustness is not None and np.isfinite(c.robustness.min_subperiod_ic_ir):
        rob = float(c.robustness.min_subperiod_ic_ir)
    return dsr, ir, orth, rob


def rank(cards: list[SignalScorecard]) -> list[SignalScorecard]:
    """DSR desc -> IC-IR desc -> orthogonality desc -> robustness desc; GATE_FAIL last."""
    evaluated = [c for c in cards if c.verdict != "GATE_FAIL"]
    failed = [c for c in cards if c.verdict == "GATE_FAIL"]
    evaluated.sort(key=_rank_tuple, reverse=True)
    return evaluated + failed


def evaluate_batch(signals: dict[str, "Signal"], panel: Panel, gates: Gates,
                   batch_name: str, factor_book: FactorBook | None = None) -> RankedScorecard:
    partials = {name: evaluate_signal(sig, panel, gates, factor_book)
                for name, sig in signals.items()}
    gp = {name: c.gross for name, c in partials.items() if c.gross is not None}
    defl = tier4_deflation(gp, gates) if gp else {}
    cards = [_finalize(c, defl.get(name), gates, panel) for name, c in partials.items()]
    return RankedScorecard(batch_name, gates.primary_horizon, len(gp), rank(cards),
                           dict(panel.meta), dict(gates.raw))


# ---- serialization ----------------------------------------------------------

def _f(x) -> float | None:
    xf = float(x)
    return xf if np.isfinite(xf) else None


def _horizon_json(h: HorizonIC) -> dict:
    return {"horizon": h.horizon, "ic_mean": _f(h.ic_mean), "ic_ir": _f(h.ic_ir),
            "ic_tstat": _f(h.ic_tstat), "n_days": int(h.n_days), "ci_low": _f(h.ci_low),
            "ci_high": _f(h.ci_high), "p_le_0": _f(h.p_le_0), "decile_spread": _f(h.decile_spread),
            "decile_monotonic": bool(h.decile_monotonic), "sign_used": int(h.sign_used)}


def _card_json(c: SignalScorecard) -> dict:
    g, d, cap, rob, orth, cpcv = (c.gross, c.deflation, c.capturability, c.robustness,
                                  c.orthogonality, c.cpcv)
    return {
        "name": c.name, "family": c.family, "spec_hash": c.spec_hash, "verdict": c.verdict,
        "rank_key": _f(c.rank_key),
        "hygiene": {"passed": bool(c.hygiene.passed), "causal": bool(c.hygiene.causal),
                    "ohlc_violations": int(c.hygiene.ohlc_violations),
                    "valid_days": int(c.hygiene.valid_days), "reasons": list(c.hygiene.reasons)},
        "gross": None if g is None else {
            "primary_horizon": g.primary_horizon, "breadth": _f(g.breadth),
            "decay_halflife": _f(g.decay_halflife),
            "by_horizon": {str(h): _horizon_json(hi) for h, hi in g.by_horizon.items()}},
        "deflation": None if d is None else {
            "dsr": _f(d.dsr), "psr": _f(d.psr), "mintrl_years": _f(d.mintrl_years),
            "fdr_q": _f(d.fdr_q), "n_trials": int(d.n_trials), "sr_star": _f(d.sr_star),
            "n_eff": _f(d.n_eff), "fdr_q_bhy": _f(d.fdr_q_bhy), "hlz_pass": bool(d.hlz_pass)},
        "cpcv": None if cpcv is None else {
            "n_groups": cpcv.n_groups, "k_test": cpcv.k_test, "n_paths": cpcv.n_paths,
            "oos_sharpe_mean": _f(cpcv.oos_sharpe_mean), "oos_sharpe_std": _f(cpcv.oos_sharpe_std),
            "oos_sharpe_p05": _f(cpcv.oos_sharpe_p05),
            "frac_paths_positive": _f(cpcv.frac_paths_positive),
            "embargo_days": cpcv.embargo_days, "purge_horizon": cpcv.purge_horizon},
        "capturability": None if cap is None else {
            "frictionless_sharpe": _f(cap.frictionless_sharpe), "cost_wall": _f(cap.cost_wall),
            "by_cost": {k: {"net_sharpe": _f(v.net_sharpe), "net_pf": _f(v.net_pf),
                            "turnover_ann": _f(v.turnover_ann), "max_dd": _f(v.max_dd)}
                        for k, v in cap.by_cost.items()}},
        "robustness": None if rob is None else {
            "min_subperiod_ic_ir": _f(rob.min_subperiod_ic_ir),
            "mean_subperiod_ic_ir": _f(rob.mean_subperiod_ic_ir),
            "recent_ic_ir": _f(rob.recent_ic_ir),
            "subperiod_ic_ir": [_f(x) for x in rob.subperiod_ic_ir],
            "subperiod_valid_days": [int(x) for x in rob.subperiod_valid_days]},
        "orthogonality": None if orth is None else {
            "max_abs_corr": _f(orth.max_abs_corr), "r2_explained": _f(orth.r2_explained),
            "residual_sharpe": _f(orth.residual_sharpe), "n_days": orth.n_days,
            "corr_by_factor": {k: _f(v) for k, v in orth.corr_by_factor.items()}},
        "caveats": list(c.caveats),
    }


def to_json(rs: RankedScorecard) -> dict:
    meta = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in rs.panel_meta.items()}
    return {"batch_name": rs.batch_name, "primary_horizon": rs.primary_horizon,
            "n_trials": rs.n_trials, "panel_meta": meta,
            "cards": [_card_json(c) for c in rs.cards]}


def to_markdown(rs: RankedScorecard) -> str:
    sf = rs.panel_meta.get("survivorship_free")
    lines = [
        f"# Signal Scorecard — {rs.batch_name}", "",
        f"- primary horizon: **{rs.primary_horizon}d**  ·  deflation n_trials: **{rs.n_trials}**",
        f"- survivorship-free data: **{sf}**  (False ⇒ all results are UPPER BOUNDS)", "",
        "| # | signal | family | verdict | IC-IR | DSR | Neff | FDR-q | BHY-q | HLZ "
        "| cpcvOOS | netSh@std | costWall | breadth |",
        "|---|--------|--------|---------|-------|-----|------|-------|-------|-----"
        "|---------|-----------|----------|---------|",
    ]
    rnk = 0
    for c in rs.cards:
        if c.gross is not None:
            hp = c.gross.by_horizon[c.gross.primary_horizon]
            ir = f"{hp.ic_ir:.3f}"
            br = f"{c.gross.breadth:.2f}" if np.isfinite(c.gross.breadth) else "—"
        else:
            ir = br = "—"
        dsr = (f"{c.deflation.dsr:.3f}" if c.deflation is not None
               and np.isfinite(c.deflation.dsr) else "—")
        q = (f"{c.deflation.fdr_q:.3f}" if c.deflation is not None
             and np.isfinite(c.deflation.fdr_q) else "—")
        neff = (f"{c.deflation.n_eff:.1f}" if c.deflation is not None
                and np.isfinite(c.deflation.n_eff) else "—")
        qb = (f"{c.deflation.fdr_q_bhy:.3f}" if c.deflation is not None
              and np.isfinite(c.deflation.fdr_q_bhy) else "—")
        hlz = ("✓" if (c.deflation is not None and c.deflation.hlz_pass) else "✗") \
            if c.deflation is not None else "—"
        cpcv = (f"{c.cpcv.oos_sharpe_mean:.2f}" if c.cpcv is not None
                and np.isfinite(c.cpcv.oos_sharpe_mean) else "—")
        nsh = cw = "—"
        if c.capturability is not None:
            std = c.capturability.by_cost.get("standard")
            if std is not None and np.isfinite(std.net_sharpe):
                nsh = f"{std.net_sharpe:.2f}"
            if np.isfinite(c.capturability.cost_wall):
                cw = f"{c.capturability.cost_wall:.2f}"
        if c.verdict != "GATE_FAIL":
            rnk += 1
            num = str(rnk)
        else:
            num = "—"
        lines.append(f"| {num} | {c.name} | {c.family} | {c.verdict} | {ir} | {dsr} | {neff} "
                     f"| {q} | {qb} | {hlz} | {cpcv} | {nsh} | {cw} | {br} |")
    if rs.cards and rs.cards[0].caveats:
        lines += ["", "**Caveats (top signal):** " + "; ".join(rs.cards[0].caveats)]
    return "\n".join(lines)


def write_scorecard(rs: RankedScorecard, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jp, mp = out / "scorecard.json", out / "scorecard.md"
    jp.write_text(json.dumps(to_json(rs), indent=2), encoding="utf-8")
    mp.write_text(to_markdown(rs), encoding="utf-8")
    return jp, mp
