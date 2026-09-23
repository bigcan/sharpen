"""Phase-0 summary — fold the C1 / C3 / K2 artifacts and the measured throughput into one decision
record (ATL x Jev plan §4, Phase 0). Reads ``results/atl_jev/phase0/*.json``; calls nothing.

The one judgment in here is the ADOPTED knowledge cutoff, and it is written down as a rule:
the pre-registered C1 rule (two-segment changepoint on item accuracy) returns "no knowledge" because
item accuracy is diluted by months where the cross-section split ~50/50. The exploratory month-level
statistic (does Jev's mean implied P(up) move WITH the realized share of risers?) shows memory of
market REGIMES. Adopted rule, conservative by construction: T_c = January of the year AFTER the first
calendar year whose regime correlation is <= 0 and stays <= 0 in every later year — a one-year buffer,
because memory fades rather than stopping at a month boundary. The pre-registered embargo is then added.

    python scripts/research/atl_jev_phase0_summary.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
GATES = ROOT / "configs" / "atl_jev.gates.yaml"
P0 = ROOT / "results" / "atl_jev" / "phase0"


def _load(name: str) -> dict | None:
    p = P0 / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def adopted_cutoff(regime: dict[str, dict], embargo_months: int) -> dict:
    years = sorted(regime)
    first_nonpos = next((y for i, y in enumerate(years)
                         if all(regime[z]["corr_implied_vs_realized"] <= 0 for z in years[i:])), None)
    if first_nonpos is None:
        return {"T_c": None, "clean_window_from": None, "first_nonpositive_year": None}
    t_c = f"{int(first_nonpos) + 1:04d}-01"
    m = 1 + embargo_months
    return {"T_c": t_c, "clean_window_from": f"{int(first_nonpos) + 1 + (m - 1) // 12:04d}-{(m - 1) % 12 + 1:02d}",
            "first_nonpositive_year": first_nonpos}


def main() -> int:
    graw = GATES.read_bytes()
    gates = yaml.safe_load(graw)
    jcfg, feas = gates["jev"], gates["phase0"]["feasibility"]
    c1, c3, k2 = _load("cutoff_probe.json"), _load("identification_probe.json"), _load("power_check.json")
    # Measured Jev rates come from the LIVE-run sidecars, never from an artifact a cached re-run may have
    # regenerated with zero requests (which would silently price the corpus at $0 and pass K0).
    r1, r3 = _load("jev_rates_cutoff.json"), _load("jev_rates_identification.json")
    missing = [n for n, v in (("cutoff_probe", c1), ("identification_probe", c3), ("power_check", k2),
                              ("jev_rates_cutoff", r1), ("jev_rates_identification", r3)) if v is None]
    if missing:
        print(f"missing artifacts: {missing}", file=sys.stderr)
        return 1
    if not (r1["requests"] > 0 and r1["questions_sent"] > 0 and r3["requests"] > 0):
        print("measured Jev rates are empty — K0 cannot be estimated", file=sys.stderr)
        return 1

    cut = adopted_cutoff(c1["exploratory_regime_memory_by_year"], int(c1["gates"]["embargo_months"]))

    # K0: cost + wall-clock for the full corpus, from MEASURED usage (identification probe = full filings).
    # The three estimation inputs below are assumptions of the estimate, not gates (those are in the YAML).
    u = r3
    tok_per_request = u["input_tokens"] / u["requests"]                # ~one filing + 2 questions
    q_tok = r1["input_tokens"] / r1["questions_sent"]                  # tokens per short question
    per_filing = tok_per_request + 8 * q_tok                          # a ~10-question questionnaire
    n = int(feas["corpus_filings_estimate"])
    cost = n * per_filing * float(jcfg["price_usd_per_m_input"]) / 1e6
    jev_hours = n * (u["latency_s"]["p50"] or 0) / int(jcfg["concurrency"]) / 3600
    edgar_hours = n * 2 / 4.0 / 3600                                  # 2 requests/filing at 4 req/s
    k0 = bool(cost > float(feas["max_corpus_cost_usd"])
              or (jev_hours + edgar_hours) > float(feas["max_corpus_wallclock_hours"]))

    verdict_universes = {k: v for k, v in k2["universes"].items() if k != "djia_whatif"}
    record = {
        "phase": 0, "gates_sha256": hashlib.sha256(graw).hexdigest(),
        "C1_knowledge_cutoff": {
            "preregistered_rule": c1["decision"],
            "exploratory_regime_memory_by_year": c1["exploratory_regime_memory_by_year"],
            "adopted": {**cut, "rule": "one-year buffer after the first year from which regime correlation "
                                       "stays <= 0; deviation from the pre-registered rule, conservative direction"}},
        "C3_identification": {"decision": c3["decision"], "summary": c3["summary"], "chance": c3["chance"],
                              "n_filings": c3["n_filings"]},
        "K2_power": {"decision": k2["decision"], "by_universe": {
            k: {x: v.get(x) for x in ("mde80_target_ic", "mde80_measured_ic", "reachable_at_plausible_ic",
                                      "active_median", "min_names_per_day")} for k, v in k2["universes"].items()},
            "verdict_universes": sorted(verdict_universes)},
        "K0_feasibility": {"tokens_per_filing_est": round(per_filing), "corpus_filings": n,
                           "cost_usd_est": round(cost, 2), "jev_hours_est": round(jev_hours, 2),
                           "edgar_hours_est": round(edgar_hours, 1), "fires": k0},
        "jev_served_models": sorted(set(r1["served_models"]) | set(r3["served_models"])
                                    | set(c1["jev_usage"]["served_models"]) | set(c3["jev_usage"]["served_models"])),
        "live_run_spend_usd": round(r1.get("cost_usd", 0) + r3.get("cost_usd", 0), 4),   # pilots excluded
    }
    # Plan C6: answers from different served versions are different models — flag, never average silently.
    record["mixed_served_versions"] = len(record["jev_served_models"]) > 1
    stop = k0 or record["K2_power"]["decision"]["K2_fires"] or record["mixed_served_versions"]
    record["phase0_verdict"] = "STOP" if stop else "PROCEED to Phase 1"
    (P0 / "phase0_summary.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("C1_knowledge_cutoff", "K0_feasibility", "phase0_verdict")},
                     indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
