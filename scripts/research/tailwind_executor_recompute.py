"""Recompute DSR and P(pass) on the TAILWIND **executor path** (Tier-2 audit
2026-08-18, item #1 of "Minimum before any capital",
``docs/research/tailwind_v1_deep_lifecycle_audit_2026-08-18.md`` P3-01):

    "Recompute DSR / PBO / P(pass) / render on the executor's mechanics -- or state
    explicitly, in every artifact, that the certifying numbers describe a different book."

Every deploy-facing number to date (DSR 0.896, PBO 0.0009, P(pass) 0.746, RENDER_CLEAR)
was computed on the RESEARCH basis (``xsec_momentum_falsification.backtest()``: monthly
ffilled weights, 1-bar lag, 2bps one-way turnover cost, NO gross cap / slippage / borrow).
The wired executor (``TwoSleeveExecutor`` -> ``MultiAssetAllocatorEnv`` -> ``PaperState``)
applies four mechanisms the research basis does not: a 3.0x combined-book gross cap that
binds on 99.13% of rebalances (P3-01a), the env's actual multi-bar execution lag, base+
impact slippage, and fixed-notional PaperState accounting. The audit's P3-01 finder
ESTIMATED the combined effect via a pandas reconstruction (0.601 -> 0.422, ~-30% SR) but
explicitly flagged it as indicative, not measured, because it omitted fixed-entry-notional
accounting and the 2-sleeve alpha combine.

This script does not estimate -- it RUNS the real production code path
(``TwoSleeveExecutor.sim_oracle`` on the real 18-ETF union, the exact combine + replay the
live book would use) and feeds the resulting daily net-return series into the SAME
deflated-Sharpe and P(pass) machinery the research-basis gates use
(``audit_tailwind_book.py`` / ``tailwind_forward_path_render.py``), so this is the executor
number, not an estimate of it.

DSR: same momentum-grid selection surface (18 books, the multiplicity that is unaffected
by execution mechanics) and n_trials from ``configs/tailwind_v1.gates.yaml`` (the
own-capital 0.95 bar), but the OBSERVED Sharpe/skew/kurtosis/n_obs come from the executor's
own combined daily return series instead of the research basis.

P(pass): the executor's combined series is already sized by the real config levers
(target_vol_asset/lev_cap/max_gross -> ~9.83% realised vol per
``docs/research/tailwind_sizing_reconciliation_2026-08-01.md``), so it is fed DIRECTLY into
``tailwind_forward_path_render.needless_termination`` (no additional vol-rescaling -- that
rescaling is what the research-basis render needed to reach a target vol from its 6.92%
native series; the executor series is already at its real production sizing).

Emits ``results/tailwind_v1/executor_recompute.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml as _yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tailwind_v1"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tailwind_forward_path_render as tfr  # noqa: E402
import xsec_momentum_falsification as mom  # noqa: E402

from sharpen.crypto.eval.statistics import (  # noqa: E402
    block_bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    excess_kurtosis,
    skewness,
)
from sharpen.data.cross_asset_loader import (  # noqa: E402
    build_two_sleeve_arrays,
    load_two_sleeve_data,
)
from sharpen.paper import TwoSleeveExecutor  # noqa: E402
from sharpen.prop.challenge_simulator import FirmRules, SizingPolicy  # noqa: E402

ANN = mom.ANN
CHALLENGE_CFG = ROOT / "configs" / "tailwind_v1_challenge.yaml"
OWN_CAPITAL_GATES = ROOT / "configs" / "tailwind_v1.gates.yaml"


def sh(s: pd.Series) -> float:
    return round(mom.sharpe(s.dropna()), 3)


def build_executor_series(cfg: dict) -> tuple[pd.Series, dict, dict]:
    """Drive the REAL TwoSleeveExecutor on the real 18-ETF union and return the combined
    book's daily net-return series (index = calendar dates), the oracle dict, and the
    per-sleeve/alpha detail. This is production code (``sharpen.paper.two_sleeve``),
    not a reconstruction -- gross cap, execution lag and slippage are whatever the real
    ``MultiAssetAllocatorEnv`` + ``PaperState`` apply, not an approximation of them."""
    data = load_two_sleeve_data(cfg, force_refetch=False, require_fresh=False)
    close = data["close"]
    bundle = build_two_sleeve_arrays(data, close.index[0], close.index[-1])

    ex = TwoSleeveExecutor(cfg)
    oracle, detail = ex.sim_oracle(bundle)

    ts = np.asarray(bundle["union"]["timestamps"], dtype=np.int64)
    n = len(oracle["step_returns"])
    idx = pd.to_datetime(ts[1:1 + n], unit="s")
    series = pd.Series(np.asarray(oracle["step_returns"], dtype=np.float64), index=idx)
    return series, oracle, detail


def recompute_dsr(exec_net: pd.Series) -> dict:
    """DSR on the executor path: same selection surface (18-book momentum grid, n_trials
    from the own-capital gates file) as ``audit_tailwind_book.py``'s F_deflated_sharpe, but
    observed Sharpe/skew/kurtosis/n_obs come from the EXECUTOR's combined daily returns."""
    of = _yaml.safe_load(OWN_CAPITAL_GATES.read_text(encoding="utf-8"))["overfitting"]
    n_trials_pre = int(of["dsr_n_trials"])
    min_dsr = float(of["min_dsr"])
    bracket_N = [int(n) for n in of.get("dsr_n_trials_bracket", [n_trials_pre])]
    block_days = int(of.get("block_bootstrap_block_days", 21))

    grid_ann = mom.graded_book_sharpes()             # the 18-book selection surface (unaffected
    trial_sharpes_daily = [s / (ANN ** 0.5) for s in grid_ann.values()]  # by execution mechanics)

    cd = exec_net.dropna().to_numpy(dtype=np.float64)
    obs_sr_ann = mom.sharpe(exec_net)
    obs_sr_daily = float(obs_sr_ann / (ANN ** 0.5))
    g1, g2 = skewness(cd.tolist()), excess_kurtosis(cd.tolist())

    def _dsr(N):
        return deflated_sharpe_ratio(
            obs_sr_daily, trial_sharpes_daily, n_obs=len(cd),
            skew=g1, excess_kurt=g2, n_trials=N, periods_per_year=ANN)

    dsr_pre = _dsr(n_trials_pre) or {}
    dsr_val = dsr_pre.get("dsr")
    boot = block_bootstrap_sharpe_ci(cd, block=block_days, periods_per_year=ANN)
    dsr_pass = dsr_val is not None and dsr_val >= min_dsr

    return {
        "gate_source": str(OWN_CAPITAL_GATES.relative_to(ROOT)).replace("\\", "/"),
        "executor_observed_sharpe_ann": round(obs_sr_ann, 4),
        "executor_observed_sr_daily": round(obs_sr_daily, 6),
        "n_obs": len(cd),
        "skew": round(g1, 4),
        "excess_kurtosis": round(g2, 4),
        "n_trials_pre_registered": n_trials_pre,
        "dsr_at_pre_registered_N": (None if dsr_val is None else round(dsr_val, 4)),
        "sr_star_ann_at_pre_registered_N": (
            round(dsr_pre["sr_star_ann"], 4) if dsr_pre else None),
        "dsr_bracket_executor": {
            str(N): (None if (d := _dsr(N)) is None else round(d["dsr"], 4))
            for N in bracket_N},
        "block_bootstrap_sharpe_ci95": (None if boot is None else
                                        {k: (round(v, 4) if isinstance(v, float) else v)
                                         for k, v in boot.items()}),
        "min_dsr": min_dsr,
        "pass": dsr_pass,
        "research_basis_for_comparison": {
            "honest_dsr_n24": 0.896,
            "source": "results/tailwind_v1/audit_tailwind.json (F_deflated_sharpe)",
        },
        "note": ("Same 18-book momentum selection surface + n_trials as the research-basis "
                 "gate (the multiplicity is a property of WHICH book was chosen, not of how "
                 "it executes). Observed Sharpe/skew/kurtosis/n_obs are the EXECUTOR's own "
                 "combined daily net-return series (real gross cap, execution lag, slippage, "
                 "fixed-notional PaperState accounting) -- not a reconstruction of it."),
    }


def recompute_p_pass(exec_net: pd.Series) -> dict:
    """P(pass) on the executor path: feed the executor's own (already-sized) combined
    daily return series directly into the same challenge-simulator machinery
    ``tailwind_forward_path_render.py`` uses, with NO additional vol-rescaling -- the
    executor series is already at its real production sizing (env.target_vol_asset /
    lev_cap / max_gross_exposure), unlike the research basis which needs rescaling from
    its native 6.92% vol to reach a target."""
    gates = _yaml.safe_load(tfr.CHALLENGE_GATES.read_text(encoding="utf-8"))
    cg = gates["challenge_pass_gate"]
    risk = gates["paper_soak"]["risk"]
    hard = cg["firm_hard_limits"]

    dd_kill = float(risk["max_drawdown_kill_pct"]) / 100.0
    daily_halt = float(risk["daily_loss_halt_pct"]) / 100.0
    firm_dd = float(hard["max_total_loss_pct"]) / 100.0
    firm_daily = float(hard["daily_loss_limit_pct"]) / 100.0
    target_step1 = float(hard["profit_target_step1_pct"]) / 100.0
    min_days = int(hard["min_trading_days"])
    daily_budget = float(cg["daily_breach_budget"])
    p_pass_expected = float(cg["p_pass_expected"]["ftmo_step1"])
    min_p_pass = float(cg["min_p_pass_step1"])

    fpr = gates["forward_path_render"]
    max_needless = float(fpr["max_needless_share"])
    max_days = int(fpr["render_horizon_days"])

    policy = SizingPolicy(vol_multiplier=1.0)  # already sized -- no rescale
    firm_rules = FirmRules(name="FTMO_step1_firm", profit_target=target_step1,
                           max_total_dd=firm_dd, daily_loss_limit=firm_daily,
                           max_days=None, min_trading_days=min_days, dd_mode=hard["dd_mode"])
    internal_rules = FirmRules(name="FTMO_step1_internal_kills", profit_target=target_step1,
                               max_total_dd=dd_kill, daily_loss_limit=daily_halt,
                               max_days=None, min_trading_days=min_days, dd_mode=hard["dd_mode"])

    nt = tfr.needless_termination(exec_net, firm_rules, internal_rules, policy, max_days)
    seq = nt["disjoint_windows"]
    p_pass_overlapping = nt["firm_limits"]["p_pass"]
    p_pass_disjoint = seq["p_firm_pass"]                # the BINDING read (project convention:
    needless_share = seq["needless_share_of_firm_passes"]  # see tailwind_forward_path_render.py's
    # own "binding_read" / "the disjoint one is the honest DD profile" precedent -- overlapping
    # starts share ~99% of their data with their neighbour and inflate apparent precision.

    return {
        "gates_source": str(tfr.CHALLENGE_GATES.relative_to(ROOT)).replace("\\", "/"),
        "p_pass_overlapping": p_pass_overlapping,
        "p_pass_disjoint_windows": p_pass_disjoint,
        "binding_read": "p_pass_disjoint_windows",
        "n_disjoint_challenges": seq["n_disjoint_challenges"],
        "needless_share_of_firm_passes": needless_share,
        "needless_share_wilson_ci95": seq["needless_share_wilson_ci95"],
        "max_acceptable_needless_share": max_needless,
        "daily_breach_rate": nt["firm_limits"]["p_daily_breach"],
        "daily_breach_budget": daily_budget,
        "min_p_pass_gate": min_p_pass,
        "pass_p_pass_gate": bool(p_pass_disjoint >= min_p_pass),
        "pass_p_pass_gate_overlapping_only": bool(p_pass_overlapping >= min_p_pass),
        "pass_needless_gate": bool(needless_share is not None and needless_share <= max_needless),
        "pass_daily_breach_gate": bool(nt["firm_limits"]["p_daily_breach"] <= daily_budget),
        "research_basis_for_comparison": {
            "p_pass_expected_simulator": p_pass_expected,
            "source": "configs/tailwind_v1_challenge_v2.gates.yaml challenge_pass_gate.p_pass_expected",
        },
        "full_detail": nt,
        "note": ("Fed the EXECUTOR's own combined daily net-return series directly (already "
                 "sized by env.target_vol_asset/lev_cap/max_gross_exposure -- no vol-rescale "
                 "applied, unlike the research-basis render which rescales FROM 6.92% native "
                 "vol). p_pass_disjoint_windows (n=26 non-overlapping challenge windows) is the "
                 "BINDING read; p_pass_overlapping (0.99 correlated draws between neighbouring "
                 "starts) overstates precision and is reported for comparison only."),
    }


def main() -> dict:
    cfg = _yaml.safe_load(CHALLENGE_CFG.read_text(encoding="utf-8"))

    print("[1/3] driving the REAL TwoSleeveExecutor on the 18-ETF union "
          "(gross cap + execution lag + slippage + PaperState accounting) ...")
    exec_net, oracle, detail = build_executor_series(cfg)

    ann_vol = float(exec_net.std(ddof=1) * np.sqrt(ANN))
    ann_ret = float(exec_net.mean() * ANN)
    eq = (1.0 + exec_net.fillna(0.0)).cumprod()
    max_dd = float((eq / eq.cummax() - 1.0).min())

    out: dict = {
        "book": "tailwind-v1 executor path (momentum TSMOM + BAB defensive, TwoSleeveExecutor)",
        "config": str(CHALLENGE_CFG.relative_to(ROOT)).replace("\\", "/"),
        "data_range": {"start": str(exec_net.index[0].date()), "end": str(exec_net.index[-1].date()),
                       "n_days": len(exec_net)},
        "executor_book": {
            "ann_vol_pct": round(ann_vol * 100, 3),
            "ann_ret_pct": round(ann_ret * 100, 3),
            "sharpe": round(mom.sharpe(exec_net), 4),
            "max_dd_pct": round(max_dd * 100, 2),
            "predicted_vol_pct_from_sizing_reconciliation": round(
                0.0334 * float(cfg["env"]["max_gross_exposure"]) * 100, 2),
            "env_levers": {k: cfg["env"][k] for k in
                          ("target_vol_asset", "lev_cap", "max_gross_exposure",
                           "taker_fee", "slippage_base_bps", "slippage_impact_bps")},
        },
    }

    print("[2/3] recomputing DSR on the executor's own daily return series ...")
    out["dsr_executor_path"] = recompute_dsr(exec_net)

    print("[3/3] recomputing P(pass) on the executor's own daily return series ...")
    out["p_pass_executor_path"] = recompute_p_pass(exec_net)

    dsr_pass = out["dsr_executor_path"]["pass"]
    pp_pass = out["p_pass_executor_path"]["pass_p_pass_gate"]
    out["verdict"] = {
        "question": "Do DSR and P(pass), recomputed on the EXECUTOR's actual mechanics "
                    "(gross cap, execution lag, slippage, fixed-notional accounting), still "
                    "clear their gates -- or were the certified numbers describing a "
                    "different (research-basis) book?",
        "dsr_own_capital_gate_pass": dsr_pass,
        "p_pass_challenge_gate_pass": pp_pass,
        "dsr_executor_vs_research": {
            "executor": out["dsr_executor_path"]["dsr_at_pre_registered_N"],
            "research_honest": 0.896,
        },
        "p_pass_executor_vs_research": {
            "executor_disjoint": out["p_pass_executor_path"]["p_pass_disjoint_windows"],
            "research_simulator": out["p_pass_executor_path"][
                "research_basis_for_comparison"]["p_pass_expected_simulator"],
        },
        "capital_gate": ("This closes Tier-2 audit item #1 (2026-08-18, 'Minimum before any "
                         "capital'). It does NOT by itself authorise a live attempt -- the "
                         "other open items (borrow/financing modeling P3-03, real-data "
                         "end-to-end executor test P3-02, drift/kill wiring P10-03/04, "
                         "operator go-ahead) still stand."),
    }
    (OUT / "executor_recompute.json").write_text(json.dumps(out, indent=2, default=str))

    print("=" * 78)
    print("TAILWIND-v1 — DSR + P(pass) RECOMPUTED ON THE EXECUTOR PATH")
    print("=" * 78)
    eb = out["executor_book"]
    print(f"executor book: {out['data_range']['n_days']} days "
          f"{out['data_range']['start']} -> {out['data_range']['end']}")
    print(f"  ann_vol={eb['ann_vol_pct']}%  ann_ret={eb['ann_ret_pct']}%  "
          f"sharpe={eb['sharpe']}  max_dd={eb['max_dd_pct']}%")
    d = out["dsr_executor_path"]
    print(f"DSR(N={d['n_trials_pre_registered']}) executor = {d['dsr_at_pre_registered_N']}  "
          f"(min {d['min_dsr']})  -> {'PASS' if d['pass'] else 'FAIL'}   "
          f"[research-basis honest DSR = 0.896]")
    p = out["p_pass_executor_path"]
    print(f"P(pass) executor disjoint = {p['p_pass_disjoint_windows']}  "
          f"(n={p['n_disjoint_challenges']}, gate {p['min_p_pass_gate']})  "
          f"-> {'PASS' if p['pass_p_pass_gate'] else 'FAIL'}   "
          f"[research-basis simulator = {p['research_basis_for_comparison']['p_pass_expected_simulator']}]")
    print(f"needless-termination share = {p['needless_share_of_firm_passes']}  "
          f"(max {p['max_acceptable_needless_share']})  "
          f"-> {'PASS' if p['pass_needless_gate'] else 'FAIL'}")
    print("-" * 78)
    print(f"VERDICT: DSR {'PASS' if dsr_pass else 'FAIL'} / P(pass) "
          f"{'PASS' if pp_pass else 'FAIL'} on the executor path")
    print("=" * 78)
    return out


if __name__ == "__main__":
    main()
