"""Aggregate Q1 2026 OOS blindspot backtest results into a single summary.

Pulls per-strategy backtest_test/* metrics from each strategy's wandb-summary.json
(located via run_id from the corresponding pipeline log), bucketizes against the
strategy's val-phase baseline, and writes:
  - results/q1_2026_oos_blindspot/<strategy>/verdict.json
  - results/q1_2026_oos_blindspot/summary.md

Bucket logic (per project_paper_checkpoints_q1_2026_blindspot.md):
  PF degradation < 20%       -> HOLD
  20% <= degradation < 40%   -> WATCH
  degradation >= 40%, PF<1, or MaxDD breach -> RETRAIN
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs" / "q1_2026_oos_blindspot"
RESULTS_DIR = PROJECT_ROOT / "results" / "q1_2026_oos_blindspot"
WANDB_DIR = PROJECT_ROOT / "wandb"

# Strategies + their deployed-ckpt setup
STRATEGIES = {
    "gmgp1-btc-deployed": {
        "log": "gmgp1_btc_deployed.log",
        "label": "GMGP1-BTC (deployed L1 multiseed, 2026-04-07 mount)",
        "asset_class": "crypto",
        "max_dd_breach_pct": 8.0,  # Velotrade trailing
        "ckpt": "gmgp1-btc-l1-multiseed_20260407_070227/checkpoint_final.pth",
        "uses_pipeline": True,
    },
    "gmgp1-btc-seed456-fold3-alt": {
        "verdict_existing": "results/gmgp1_btc_recent_oos/verdict.json",
        "label": "GMGP1-BTC (alt: seed-456 fold-3 WF graduate, S495-cont)",
        "asset_class": "crypto",
        "max_dd_breach_pct": 8.0,
        "ckpt": "WF_seed456_fold_03_20260423_212007/checkpoint_final.pth",
        "uses_pipeline": False,
    },
    "gmgp1-gold-deployed": {
        "log": "gmgp1_gold_deployed.log",
        "label": "GMGP1-GOLD (deployed v5 seed-456 on MGC contract)",
        "asset_class": "cme_futures",
        "max_dd_breach_pct": 10.0,
        "ckpt": "gmgp1_v5_l1_seed456/checkpoint_final.pth",
        "uses_pipeline": True,
    },
    "gmgp1-xauusd-deployed": {
        "log": "gmgp1_xauusd_deployed.log",
        "label": "GMGP1-XAUUSD (deployed WF fold-07 seed-42 SOLO_BEST)",
        "asset_class": "cfd_gold",
        "max_dd_breach_pct": 8.0,  # FTMO trailing
        "ckpt": "WF_seed42_fold_07_20260424_061803/checkpoint_final.pth",
        "uses_pipeline": True,
        # Reference baseline = WF fold-07 strict-OOS solo_42 test PF (2026-03-06 -> 2026-04-06)
        # from results/gmgp1_xauusd_oanda_wf_ensemble/fold_07/verdict.json.
        "baseline_pf": 2.260,
        "in_sample_overlap_warn": "fold-7 train ended 2026-02-06 -> Jan-1 to Feb-6 is in-sample for this ckpt",
    },
    "sg1-btc-deployed": {
        "log": "sg1_btc_deployed.log",
        "label": "SG-1-BTC (deployed DECAY-01 WF fold-7 seed-456 SOLO, cost-corrected)",
        "asset_class": "crypto",
        "max_dd_breach_pct": 8.0,  # Velotrade trailing
        # Re-pointed S553-cont (audit P8-01): was the RETIRED seed-42 l1-multiseed
        # ckpt; the live policy is the DECAY-01 SOLO seed-456 (bundle solo_v1.tar.gz,
        # live_sg1_btc_bybit.yaml). Re-run configs/sg1_btc_velotrade_q1_2026_oos_backtest.yaml
        # (cost-corrected, window 2026-02-03 -> 2026-04-30) against this ckpt to
        # refresh sg1_btc_deployed.log before aggregating.
        "ckpt": "sg1-btc-decay01-wf-fold7-seed456_20260521_230637/checkpoint_final.pth",
        "uses_pipeline": True,
        "in_sample_overlap_warn": (
            "deployed seed-456 train cutoff 2026-02-01 — OOS window must start "
            "AFTER it (config set to 2026-02-03)"
        ),
    },
    "sg1-xauusd-deployed": {
        "log": "sg1_xauusd_deployed.log",
        "label": "SG-1-XAUUSD (deployed WF fold-07 ensemble seeds 42/2025/3141, ens_agreement)",
        "asset_class": "cfd_gold",
        "max_dd_breach_pct": 8.0,
        "ckpt": "WF_seed{42,2025,3141}_fold_07_20260421_*/checkpoint_final.pth",
        "uses_pipeline": False,
        # SG-1 XAUUSD's OWN fold-7 ens_agreement strict-OOS test PF
        # (results/sg1_xauusd_ensemble_wf/fold_07/ens_agreement_metrics.json -> pf_bar = 1.955).
        # n_bars=1522 (~25 active days, fold-7 strict 1-month test window).
        # FIXED 2026-04-29: previously used 2.227 from GMGP1's fold-7 verdict (wrong strategy).
        "baseline_pf": 1.955,
        "in_sample_overlap_warn": "fold-7 train ended 2026-02-06 -> Jan-1 to Feb-6 is in-sample",
    },
}

WANDB_RUN_RE = re.compile(r"runs/([a-z0-9]{8})\b")


def find_wandb_summary(log_path: Path) -> Optional[Path]:
    """Parse wandb run id from log, then find that run's wandb-summary.json."""
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    m = WANDB_RUN_RE.search(text)
    if not m:
        return None
    run_id = m.group(1)
    # Match either online or offline run dir
    for prefix in ("run-", "offline-run-"):
        candidates = list(WANDB_DIR.glob(f"{prefix}*-{run_id}"))
        if candidates:
            summary = candidates[0] / "files" / "wandb-summary.json"
            if summary.exists():
                return summary
    return None


def bucket(test_pf: float, val_pf: float, max_dd_pct: float, breach_pct: float) -> tuple[str, dict]:
    """Apply HOLD/WATCH/RETRAIN bucket logic. Note: when test_pf > val_pf
    (improvement), degradation is negative — that's a HOLD with no concern."""
    degradation = (val_pf - test_pf) / val_pf if val_pf > 0 else 0.0
    info = {
        "degradation_pct": round(degradation * 100, 2),
        "test_pf": round(test_pf, 4),
        "val_pf": round(val_pf, 4),
        "max_dd_pct": round(max_dd_pct, 4),
        "max_dd_breach_threshold_pct": breach_pct,
    }
    if test_pf < 1.0:
        return "RETRAIN", {**info, "reason": "PF<1 absolute"}
    if abs(max_dd_pct) >= breach_pct:
        return "RETRAIN", {**info, "reason": f"MDD {max_dd_pct:.2f}% >= breach {breach_pct}%"}
    if degradation >= 0.40:
        return "RETRAIN", {**info, "reason": f"degradation {degradation*100:.1f}% >= 40%"}
    if degradation >= 0.20:
        return "WATCH", {**info, "reason": f"degradation {degradation*100:.1f}% in [20,40)"}
    return "HOLD", {**info, "reason": f"degradation {degradation*100:.1f}% < 20%"}


# Recent-OOS returns above this (over a ~2-3 month window) are not realizable net
# of cost — they signal a frictionless/compounding artifact, not real edge.
IMPLAUSIBLE_RETURN_PCT = 1000.0


def _apply_return_sanity(v: dict) -> None:
    """Guard against frictionless/compounding artifacts (audit P8-07).

    The retired seed-42 verdict bucketed a +1,351,291% total return as HOLD purely
    on PF degradation, because ``bucket()`` never inspects total return. A recent-OOS
    backtest whose return is physically implausible cannot be trusted: WARN and force
    RETRAIN so the gate never silently green-lights an artifact. The usual cause is a
    missing cost model (slippage=0) and/or uncapped equity compounding.
    """
    ret = v.get("metrics", {}).get("test_total_return_pct")
    try:
        ret = float(ret)
    except (TypeError, ValueError):
        return
    if abs(ret) <= IMPLAUSIBLE_RETURN_PCT:
        return
    v.setdefault("warnings", []).append(
        f"implausible test_total_return={ret:,.0f}% (> {IMPLAUSIBLE_RETURN_PCT:.0f}%) — "
        "frictionless/compounding artifact; verify cost model (slippage>0) and equity "
        "compounding before trusting this verdict (audit P8-07)"
    )
    if v.get("bucket") in ("HOLD", "WATCH", "PASS-on-file"):
        v["bucket_info"] = {
            **v.get("bucket_info", {}),
            "overridden_from": v.get("bucket"),
            "reason": "implausible return — gate cannot trust a frictionless artifact",
        }
        v["bucket"] = "RETRAIN"


def _parse_test_metrics_from_log(log_path: Path) -> dict:
    """Fallback for offline-mode runs: parse `wandb: backtest_test/<key> <val>`
    lines from the pipeline log. Returns {key: float} dict."""
    if not log_path.exists():
        return {}
    out = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^wandb:\s+backtest_(test|val)/(\S+)\s+([-\d.eE+]+)$", line.strip())
        if m:
            phase, key, val = m.group(1), m.group(2), m.group(3)
            try:
                out[f"backtest_{phase}/{key}"] = float(val)
            except ValueError:
                pass
    return out


def _parse_validation_summary_lines(log_path: Path) -> list[dict]:
    """Pipeline emits `Validation Complete. Return=...% Sharpe=... Sortino=...
    MaxDD=...% WinRate=...%` once per backtest phase (val, then test). Returns
    a list of dicts in the order they appear. Headline-only fallback when no
    PF is in the log."""
    if not log_path.exists():
        return []
    rows = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.search(
            r"Validation Complete\. Return=([-\d.]+)%, Sharpe=([-\d.]+), "
            r"Sortino=([-\d.]+), MaxDD=([-\d.]+)%, WinRate=([-\d.]+)%",
            line,
        )
        if m:
            rows.append({
                "return_pct": float(m.group(1)),
                "sharpe": float(m.group(2)),
                "sortino": float(m.group(3)),
                "max_dd_pct": float(m.group(4)),
                "win_rate_pct": float(m.group(5)),
            })
    return rows


def aggregate_pipeline(strategy_id: str, cfg: dict) -> Optional[dict]:
    """Pull metrics from the run_full_pipeline.py wandb-summary.json,
    falling back to log-line parsing for offline runs."""
    log_path = LOGS_DIR / cfg["log"]
    summary_path = find_wandb_summary(log_path)
    summary = {}
    if summary_path:
        summary = json.loads(summary_path.read_text())
    if not summary:
        # Offline-mode fallback
        summary = _parse_test_metrics_from_log(log_path)
    test_pf = summary.get("backtest_test/profit_factor")
    val_pf = summary.get("backtest_val/profit_factor")
    test_dd = summary.get("backtest_test/max_drawdown", 0.0) * 100
    if test_pf is None:
        return None
    # If no val PF available, fall back to per-strategy `baseline_pf` in cfg
    # (e.g. WF fold-7 reference for gmgp1-xauusd from results/gmgp1_xauusd_oanda_wf_ensemble/fold_07/verdict.json).
    if val_pf is None:
        val_pf = cfg.get("baseline_pf")
    if val_pf is None:
        return None
    b, info = bucket(test_pf, val_pf, test_dd, cfg["max_dd_breach_pct"])
    return {
        "strategy": strategy_id,
        "label": cfg["label"],
        "checkpoint": cfg["ckpt"],
        "asset_class": cfg["asset_class"],
        "wandb_summary": str(summary_path.relative_to(PROJECT_ROOT)) if summary_path else f"(parsed from log: {log_path.relative_to(PROJECT_ROOT)})",
        "metrics": {
            "test_profit_factor": test_pf,
            "test_sharpe_daily": summary.get("backtest_test/sharpe_daily"),
            "test_sortino": summary.get("backtest_test/sortino"),
            "test_max_drawdown_pct": test_dd,
            "test_total_return_pct": summary.get("backtest_test/total_return", 0.0) * 100,
            "test_trade_count": summary.get("backtest_test/trade_count"),
            "test_win_rate_pct": summary.get("backtest_test/win_rate"),
            "test_omega": summary.get("backtest_test/omega"),
            "val_profit_factor": val_pf,
            "val_max_drawdown_pct": summary.get("backtest_val/max_drawdown", 0.0) * 100,
        },
        "bucket": b,
        "bucket_info": info,
        "warnings": [cfg["in_sample_overlap_warn"]] if "in_sample_overlap_warn" in cfg else [],
    }


def aggregate_existing_verdict(strategy_id: str, cfg: dict) -> Optional[dict]:
    """Read an existing verdict.json (the seed-456 alt for gmgp1-btc)."""
    p = PROJECT_ROOT / cfg["verdict_existing"]
    if not p.exists():
        return None
    v = json.loads(p.read_text())
    m = v.get("metrics", {})
    return {
        "strategy": strategy_id,
        "label": cfg["label"],
        "checkpoint": cfg["ckpt"],
        "asset_class": cfg["asset_class"],
        "source": str(p.relative_to(PROJECT_ROOT)),
        "metrics": {
            "test_profit_factor": m.get("pf_bar"),
            "test_max_drawdown_pct": m.get("trailing_max_drawdown_pct"),
            "test_total_return_pct": m.get("total_return_pct"),
            "test_trade_count": m.get("trade_count"),
        },
        "bucket": "PASS-on-file",
        "bucket_info": {"reason": "Pre-existing Stage-4 recent-OOS verdict", "overall": v.get("overall")},
    }


def aggregate_ensemble_eval(strategy_id: str, cfg: dict) -> Optional[dict]:
    """Pull metrics from sg1_xauusd_ensemble_eval.py output dir.

    The eval script writes to a fixed `results/sg1_xauusd_ensemble/` path that
    is shared with prior unrelated runs. To avoid picking up STALE results
    (e.g. the Apr-20 L1 multiseed eval), require both the summary.csv and
    every per-rule metrics.json to have been written AFTER the corresponding
    log was started.
    """
    out_dir = PROJECT_ROOT / "results" / "sg1_xauusd_ensemble"
    summary_csv = out_dir / "summary.csv"
    if not summary_csv.exists():
        return None
    log_path = LOGS_DIR / "sg1_xauusd_deployed.log"
    if log_path.exists():
        log_start = log_path.stat().st_mtime - 60  # 60s buffer
        if summary_csv.stat().st_mtime < log_start:
            return None  # stale relative to current run
    import csv
    rows = list(csv.DictReader(summary_csv.open()))
    # Find ens_agreement (deployed rule)
    deployed_rule_row = next((r for r in rows if r.get("label") == "ens_agreement"), None)
    if not deployed_rule_row:
        return None
    test_pf = float(deployed_rule_row.get("pf_bar", 0))
    test_dd = float(deployed_rule_row.get("trailing_max_drawdown_pct", 0))
    val_pf = cfg.get("baseline_pf")
    if val_pf is None:
        b, info = "needs_baseline", {"reason": "no baseline_pf provided"}
    else:
        b, info = bucket(test_pf, val_pf, test_dd, cfg["max_dd_breach_pct"])
    return {
        "strategy": strategy_id,
        "label": cfg["label"],
        "checkpoint": cfg["ckpt"],
        "asset_class": cfg["asset_class"],
        "source": str(summary_csv.relative_to(PROJECT_ROOT)),
        "metrics": {
            "test_profit_factor": test_pf,
            "test_max_drawdown_pct": test_dd,
            "test_total_return_pct": float(deployed_rule_row.get("total_return_pct", 0)),
            "test_trade_count": int(float(deployed_rule_row.get("trade_count", 0))),
            "test_active_days": int(float(deployed_rule_row.get("active_days", 0))),
            "test_max_single_day_share": float(deployed_rule_row.get("max_single_day_share", 0)),
            "test_ftmo_compliance_pass": deployed_rule_row.get("ftmo_compliance_pass") == "True",
            "test_trailing_dd_buffer_pp": float(deployed_rule_row.get("trailing_dd_buffer_pp", 0)),
            "test_daily_dd_buffer_pp": float(deployed_rule_row.get("daily_dd_buffer_pp", 0)),
            "val_profit_factor": val_pf,
        },
        "bucket": b,
        "bucket_info": info,
        "warnings": [cfg["in_sample_overlap_warn"]] if "in_sample_overlap_warn" in cfg else [],
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid, cfg in STRATEGIES.items():
        if not cfg.get("uses_pipeline"):
            if "verdict_existing" in cfg:
                v = aggregate_existing_verdict(sid, cfg)
            else:
                v = aggregate_ensemble_eval(sid, cfg)
        else:
            v = aggregate_pipeline(sid, cfg)
        if v is None:
            print(f"[skip] {sid}: results not ready")
            continue
        _apply_return_sanity(v)
        out_path = RESULTS_DIR / sid / "verdict.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(v, indent=2))
        print(f"[ok] {sid} -> {out_path.relative_to(PROJECT_ROOT)} (bucket={v['bucket']})")
        rows.append(v)

    # Write summary.md
    md = ["# Q1 2026 OOS Blindspot — Verdict Summary",
          "",
          f"Window: 2026-01-01 → 2026-04-12 (uniform). Strategies: {len(rows)}/6.",
          "",
          "| Strategy | Bucket | Q1 PF | Val PF | Δ% | MDD | Trades | Notes |",
          "|---|---|---|---|---|---|---|---|"]
    for v in rows:
        m = v["metrics"]
        bi = v.get("bucket_info", {})
        notes = "; ".join(v.get("warnings", [])) or bi.get("reason", "")
        md.append(
            f"| {v['label']} | **{v['bucket']}** | "
            f"{m.get('test_profit_factor', '—'):.3f} | "
            f"{m.get('val_profit_factor', '—') if isinstance(m.get('val_profit_factor'), float) else '—'} | "
            f"{bi.get('degradation_pct', '—')} | "
            f"{m.get('test_max_drawdown_pct', '—'):.2f}% | "
            f"{m.get('test_trade_count', '—')} | "
            f"{notes} |"
        )
    # Auto-generated summary at table_summary.md; the hand-curated narrative
    # (with method, per-strategy interpretation, decision implications) lives
    # at summary.md and is never overwritten here.
    auto_path = RESULTS_DIR / "table_summary.md"
    auto_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nAuto-table -> {auto_path.relative_to(PROJECT_ROOT)}")
    print(f"(Narrative report at {(RESULTS_DIR / 'summary.md').relative_to(PROJECT_ROOT)} preserved.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
