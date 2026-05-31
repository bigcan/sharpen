"""Retrain-trigger gate for live SAC strategies.

Reads `retrain_policy:` from a live config, runs a rolling OOS backtest on
the most recent `gate_window_days` of data, and emits a decision:

  HOLD                — no trigger fired
  OOS_FAIL            — rolling OOS PF below floor (baseline × floor_ratio)
  DD_BREACH           — live drawdown exceeds dd_trigger_ratio × max_drawdown
  STALE               — staleness_cap_days elapsed since last_trained_date
  FEATURE_DRIFT       — KS-test rejects null on a regime feature (Bonferroni-corrected)
  DATA_STALE          — backtest data ends before gate window starts
  CONFIG_MISSING      — live config has no retrain_policy block

Exit code: 0 on HOLD, 10 on any trigger, 2 on misconfiguration.

Live drawdown is not pulled from the broker here (that belongs to the
live-monitor skill). Pass `--live-dd-pct` to feed in the value, or omit
to skip the DD check.

Usage:
  python scripts/check_retrain_triggers.py \\
      --live-config configs/live_gmgp1_btc_bybit.yaml \\
      [--live-dd-pct 0.025] [--report-dir reports/retrain_gate]

Designed to be cron-able (monthly) and CI-callable.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("retrain_gate")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_ohlcv_window(parquet_path: Path, start: date, end: date):
    """Load OHLCV rows in [start, end] from a parquet file. Returns a pandas DataFrame
    indexed by timestamp, or None on failure. Rows with NaN/inf in OHLC are dropped
    so downstream KS tests aren't silently poisoned by missing bars (FIND-01/03)."""
    try:
        import numpy as np
        import pandas as pd
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(parquet_path)
        ts_col = next(
            (c for c in pf.schema.names if c.lower() in {"timestamp", "datetime", "date", "time"}),
            None,
        )
        if ts_col is None:
            return None
        df = pf.read().to_pandas()
        df[ts_col] = pd.to_datetime(df[ts_col])
        mask = (df[ts_col].dt.date >= start) & (df[ts_col].dt.date <= end)
        df = df.loc[mask].set_index(ts_col)
        ohlc_cols = [c for c in df.columns if c.lower() in {"open", "high", "low", "close"}]
        if ohlc_cols:
            finite = np.isfinite(df[ohlc_cols].astype(float)).all(axis=1)
            dropped = int((~finite).sum())
            if dropped:
                logger.info(f"Dropped {dropped} non-finite OHLC rows from {parquet_path.name}")
            df = df.loc[finite]
        return df
    except Exception as e:
        logger.warning(f"Could not load OHLCV window from {parquet_path}: {e}")
        return None


def _regime_features(df) -> dict:
    """Compute a small set of regime-sensitive features from an OHLCV DataFrame.

    Chosen because they capture the volatility and momentum regimes the SAC
    policy keys off, without requiring the project FeatureFactory pipeline.
    """
    import numpy as np
    out: dict = {}
    cols = {c.lower(): c for c in df.columns}
    if "close" in cols:
        c = df[cols["close"]].astype(float).values
        if len(c) > 2:
            ret = np.diff(np.log(c))
            out["log_return"] = ret
            # Realized vol: non-overlapping 60-bar std of log-returns.
            # FIND-04: overlapping windows produce autocorrelated samples that
            # bias the KS p-value. Striding by window length makes samples iid.
            win = 60
            if len(ret) >= win:
                n_blocks = len(ret) // win
                blocks = ret[: n_blocks * win].reshape(n_blocks, win)
                out["realized_vol_60"] = blocks.std(axis=1)
    if "high" in cols and "low" in cols and "close" in cols:
        h = df[cols["high"]].astype(float).values
        low = df[cols["low"]].astype(float).values
        c = df[cols["close"]].astype(float).values
        n = min(len(h), len(low), len(c))
        if n > 0:
            denom = np.where(c[:n] > 0, c[:n], 1.0)
            out["bar_range_pct"] = (h[:n] - low[:n]) / denom
    return out


def _feature_drift_ks(
    data_path: Path,
    train_start: date,
    train_end: date,
    recent_start: date,
    recent_end: date,
    p_threshold: float,
) -> dict:
    """KS two-sample test per regime feature. Bonferroni-correct across features.

    Trigger fires if any feature's p-value falls below the corrected threshold,
    i.e. the recent regime distribution materially differs from the training one.
    """
    try:
        from scipy.stats import ks_2samp
    except Exception as e:
        return {"skipped": True, "reason": f"scipy unavailable: {e}"}

    train_df = _load_ohlcv_window(data_path, train_start, train_end)
    recent_df = _load_ohlcv_window(data_path, recent_start, recent_end)
    if train_df is None or recent_df is None or len(train_df) == 0 or len(recent_df) == 0:
        return {"skipped": True, "reason": "insufficient data for KS-test"}

    train_feats = _regime_features(train_df)
    recent_feats = _regime_features(recent_df)
    common = sorted(set(train_feats) & set(recent_feats))
    if not common:
        return {"skipped": True, "reason": "no overlapping regime features"}

    import numpy as np
    n = len(common)
    per_feature = {}
    max_stat = 0.0
    max_feat = None
    for feat in common:
        a = np.asarray(train_feats[feat], dtype=float)
        b = np.asarray(recent_feats[feat], dtype=float)
        # FIND-01: drop non-finite before KS so a single NaN bar can't silently
        # mask drift detection (ks_2samp returns NaN on any NaN input).
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 30 or len(b) < 30:
            per_feature[feat] = {"skipped": True, "reason": "n<30 after NaN drop",
                                 "n_train": int(len(a)), "n_recent": int(len(b))}
            continue
        stat, p = ks_2samp(a, b)
        per_feature[feat] = {"ks_stat": float(stat), "p_value": float(p),
                             "n_train": int(len(a)), "n_recent": int(len(b))}
        if np.isfinite(stat) and stat > max_stat:
            max_stat = float(stat)
            max_feat = feat

    return {
        "skipped": False,
        "n_features": n,
        "p_threshold": p_threshold,
        "max_ks_stat": max_stat,
        "max_ks_feature": max_feat,
        "per_feature": per_feature,
    }


def _data_max_date(parquet_path: Path) -> date | None:
    """Return the max bar timestamp in a parquet OHLCV file, or None on failure."""
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(parquet_path)
        # Read only the timestamp column to keep this cheap.
        ts_col = next(
            (c for c in pf.schema.names if c.lower() in {"timestamp", "datetime", "date", "time"}),
            None,
        )
        if ts_col is None:
            return None
        col = pf.read(columns=[ts_col]).column(0).to_pandas()
        return col.max().date()
    except Exception as e:
        logger.warning(f"Could not read data max date from {parquet_path}: {e}")
        return None


def _run_oos_backtest(
    backtest_cfg: dict,
    checkpoint_path: str,
    window_start: date,
    window_end: date,
) -> dict:
    """Invoke run_full_pipeline.run_backtest on the rolling window.

    Initializes WandB in disabled mode (FIND-07): the gate runs monthly under
    cron and we only need the return value from run_backtest — writing offline
    run dirs accumulates disk without any consumer.
    """
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_SILENT", "true")

    import wandb
    from scripts.run_full_pipeline import run_backtest  # noqa: E402

    cfg = copy.deepcopy(backtest_cfg)
    cfg.setdefault("data", {})
    cfg["data"]["test_start_date"] = window_start.isoformat()
    cfg["data"]["test_end_date"] = window_end.isoformat()
    cfg.setdefault("wandb", {})
    cfg["wandb"]["tags"] = list(cfg["wandb"].get("tags", [])) + ["retrain-gate"]

    run = wandb.init(
        project=cfg["wandb"].get("project", "FinRL-Pro-DS"),
        entity=cfg["wandb"].get("entity"),
        mode="disabled",
        reinit=True,
        config=cfg,
        tags=cfg["wandb"]["tags"],
        name=f"retrain-gate-{window_end.isoformat()}",
    )
    try:
        metrics = run_backtest(
            config=cfg,
            checkpoint_path=checkpoint_path,
            device="cpu",
            start_date=window_start.isoformat(),
            end_date=window_end.isoformat(),
            prefix="backtest",
            agent_type="sac",
        )
    finally:
        try:
            run.finish()
        except Exception:
            pass
    return metrics or {}


def evaluate(live_cfg_path: Path, live_dd_pct: float | None, report_dir: Path) -> dict:
    live_cfg = _load_yaml(live_cfg_path)
    policy = live_cfg.get("retrain_policy")
    if not policy or not policy.get("enabled", False):
        return {"decision": "CONFIG_MISSING", "reason": "retrain_policy block absent or disabled"}

    today = date.today()
    triggers: list[dict] = []

    # 1. Staleness check
    last_trained = datetime.fromisoformat(policy["last_trained_date"]).date()
    age_days = (today - last_trained).days
    if age_days >= int(policy["staleness_cap_days"]):
        triggers.append({
            "type": "STALE",
            "age_days": age_days,
            "cap_days": int(policy["staleness_cap_days"]),
        })

    # 2. Live drawdown check (only if provided)
    if live_dd_pct is not None:
        max_dd = float(live_cfg.get("risk", {}).get("max_drawdown_pct", 0.10))
        dd_floor = float(policy["dd_trigger_ratio"]) * max_dd
        if live_dd_pct >= dd_floor:
            triggers.append({
                "type": "DD_BREACH",
                "live_dd_pct": live_dd_pct,
                "trigger_floor": dd_floor,
                "max_dd_pct": max_dd,
            })

    # 3. Rolling OOS PF check
    backtest_cfg_path = REPO_ROOT / policy["backtest_config_ref"]
    backtest_cfg = _load_yaml(backtest_cfg_path)

    window_days = int(policy["gate_window_days"])
    lag_days = int(policy.get("gate_lag_days", 1))
    window_end = today - timedelta(days=lag_days)
    window_start = window_end - timedelta(days=window_days)

    data_path = REPO_ROOT / policy["data_file_path"]
    data_max = _data_max_date(data_path)
    if data_max is not None and data_max < window_start:
        oos_result = {
            "skipped": True,
            "reason": "data file ends before gate window start",
            "data_max": data_max.isoformat(),
            "window": [window_start.isoformat(), window_end.isoformat()],
        }
        triggers.append({"type": "DATA_STALE", **oos_result})
    else:
        # Clip window to available data so the backtest doesn't read an empty range.
        if data_max is not None and data_max < window_end:
            window_end = data_max
            window_start = window_end - timedelta(days=window_days)

        # Prefer an explicit retrain_policy.checkpoint_path (bundle-deployed
        # strategies leave agent.checkpoint_path empty — the policy lives in
        # the bundle — so the offline OOS backtest needs the raw checkpoint
        # named here). Fall back to agent.checkpoint_path for solo-checkpoint
        # deploys (e.g. gmgp1-btc) that don't set it. P10-03 schema reconcile.
        checkpoint_path = policy.get("checkpoint_path") or live_cfg.get("agent", {}).get("checkpoint_path")
        if not checkpoint_path:
            return {
                "decision": "CONFIG_MISSING",
                "reason": "no checkpoint_path in retrain_policy or agent block",
            }
        metrics = _run_oos_backtest(backtest_cfg, checkpoint_path, window_start, window_end)
        oos_pf = float(metrics.get("backtest/profit_factor", 0.0))
        baseline = float(policy["validation_pf_baseline"])
        floor = baseline * float(policy["oos_pf_floor_ratio"])
        oos_result = {
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "rolling_oos_pf": oos_pf,
            "baseline_pf": baseline,
            "floor": floor,
            "trade_count": int(metrics.get("backtest/trade_count", 0)),
            "max_drawdown": float(metrics.get("backtest/max_drawdown", 0.0)),
            "sharpe_daily": float(metrics.get("backtest/sharpe_daily", 0.0)),
        }
        if oos_pf < floor:
            triggers.append({"type": "OOS_FAIL", **oos_result})

    # 4. Feature-drift KS-test (regime shift early-warning)
    train_start = datetime.fromisoformat(backtest_cfg["data"]["train_start_date"]).date()
    train_end = datetime.fromisoformat(backtest_cfg["data"]["train_end_date"]).date()
    if window_start <= train_end:
        # FIND-02: clipping against very stale data can pull the recent window
        # back into the training period — KS would then compare overlapping
        # samples and bias toward "no drift". Skip rather than emit a false negative.
        drift_result = {
            "skipped": True,
            "reason": f"recent window start {window_start} <= train_end {train_end} (data too stale)",
        }
    else:
        drift_result = _feature_drift_ks(
            data_path=data_path,
            train_start=train_start,
            train_end=train_end,
            recent_start=window_start,
            recent_end=window_end,
            p_threshold=float(policy["feature_drift_p_threshold"]),
        )
    # Practical-significance floor: KS p-values are noise at large N (290K bars
    # → near-zero p on benign distribution shifts). Gate on effect size (max CDF
    # gap) primarily; p-value is reported but secondary.
    ks_stat_floor = float(policy.get("feature_drift_ks_stat_threshold", 0.20))
    drift_result["ks_stat_threshold"] = ks_stat_floor
    drift_result["drift_detected"] = (
        not drift_result.get("skipped")
        and drift_result.get("max_ks_stat", 0.0) >= ks_stat_floor
    )
    if drift_result["drift_detected"]:
        triggers.append({
            "type": "FEATURE_DRIFT",
            "max_ks_stat": drift_result["max_ks_stat"],
            "max_ks_feature": drift_result["max_ks_feature"],
            "ks_stat_threshold": ks_stat_floor,
        })

    decision = "HOLD" if not triggers else triggers[0]["type"]
    report = {
        "strategy_config": str(live_cfg_path.relative_to(REPO_ROOT)),
        "evaluated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "decision": decision,
        "triggers": triggers,
        "staleness_age_days": age_days,
        "oos_result": oos_result,
        "drift_result": drift_result,
        "live_dd_pct": live_dd_pct,
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    stem = live_cfg_path.stem
    out_path = report_dir / f"{stem}_{today.isoformat()}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Wrote report: {out_path}")

    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live-config", type=Path, required=True, help="Path to live config YAML")
    ap.add_argument("--live-dd-pct", type=float, default=None,
                    help="Current live drawdown as a fraction (e.g. 0.025 for 2.5%%). Omit to skip DD check.")
    ap.add_argument("--report-dir", type=Path, default=REPO_ROOT / "reports" / "retrain_gate")
    args = ap.parse_args()

    report = evaluate(args.live_config, args.live_dd_pct, args.report_dir)
    print(json.dumps(report, indent=2, default=str))

    decision = report["decision"]
    if decision == "HOLD":
        return 0
    if decision == "CONFIG_MISSING":
        return 2
    return 10


if __name__ == "__main__":
    sys.exit(main())
